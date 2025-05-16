import argparse

import json
import re
import copy
import time
import multiprocessing
from pathlib import Path
import concurrent.futures
from enum import Enum
import asyncio
from typing import cast
import traceback
import math
import random

from tqdm import tqdm
import openai


from lean_trees import utils
from lean_trees.repl_adapter.interaction import LeanServer, LeanInteractionException
from lean_trees.core.lean import LeanProofState
from eval.interlm_adapter import InterLMMiniF2FAdapter
from lean_trees.repl_adapter.data import ReplGoalInfo

# TODO!!! Blackbox show see proof termination (but not errors) - maybe?

MINIF2F_HEADER = (
    "import Mathlib\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology\n"
)

# Whitebox: the model gets access to open goals list
# Graybox: the model does not see the goals list, but knows when the proof is complete and when a tactic failed
# Blackbox: the model must generate the "done" tactic to finish the proof


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--openai_key_file", type=Path, default="openai_api_key.txt")

    parser.add_argument("--parallel_rollouts", type=int, default=1)

    parser.add_argument("--method", type=str, choices=["blackbox", "graybox", "whitebox"])
    parser.add_argument("--ban_complex_tactics", action="store_true")
    parser.add_argument("--max_steps", type=int, default=20, help="Maximum number of step attempts in one rollout.")
    parser.add_argument("--max_rollouts", type=int, default=5)

    parser.add_argument("--max_completion_tokens", type=int, default=128)

    parser.add_argument("--benchmark_cache_dir", type=Path, default="benchmarks")
    parser.add_argument("--output_dir", type=Path, default="rollouts_openai")

    parser.add_argument("--repl_exe", type=Path, default="/home/kripner/repos/lean-repl-fork/.lake/build/bin/repl")
    parser.add_argument("--project_path", type=Path, default="/home/kripner/troja/arcoss-lean-repo-v4-19-rc2")

    parser.add_argument("--force", action="store_true")

    return parser


ARGS_WHITELIST = [
    "model", "max_steps", "max_rollouts", "method", "ban_complex_tactics"
]

class Logger:
    def __init__(self, log_dir: Path):
        self._model_outputs_path = log_dir / "model_outputs.txt"
        self._rollout_steps_path = log_dir / "rollout_steps.txt"
        self._incomplete_rollouts_path = log_dir / "incomplete_rollouts.txt"
        self._exceptions_path = log_dir / "exceptions.txt"
        self._stats_path = log_dir / "stats.json"
        self._final_proofs_path = log_dir / "proofs.lean"

        self._start_time = time.time()

    def log_model_output(self, messages: list, output: str):
        with open(self._model_outputs_path, "a") as f:
            quotes = '"""'
            for message in messages:
                f.write(f"{message['role']}:\n{quotes}\n{message['content']}{quotes}\n")
            f.write(f"-->\n{quotes}{output}{quotes}\n\n")

    def log_rollout_step(self, proof_so_far: str, tactic: str | None, output: str, status: "ProofStatus | None"):
        with open(self._rollout_steps_path, "a") as f:
            if tactic is None:
                f.write(f"{proof_so_far}\n->INVALID SYNTAX:\n{output}")
            else:
                f.write(f"{proof_so_far}\n->{tactic}\nStatus: {status}")
            f.write("\n\n")

    def log_incomplete_rollout(self, theorem_with_proof: str, cause: str):
        with open(self._incomplete_rollouts_path, "a") as f:
            f.write(f"{theorem_with_proof}\n")
            f.write(f"cause: {cause}\n\n")

    def log_exception(self, theorem: str, exception: Exception, note: str):
        error_traceback = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        print(f"Unhandled exception ({note}): {exception}\n{error_traceback}")
        with open(self._exceptions_path, "a") as f:
            f.write(f"{theorem}\n")
            f.write(f"Exception: {exception}\n")
            f.write(f"Note: {note}\n")
            f.write(f"Traceback:\n{error_traceback}\n\n")

    def log_stats(self, completed_rollouts: "list[RolloutProofSearch]"):
        stats = self._calculate_stats(completed_rollouts)
        for k, v in stats.items():
            print(f"{k}: {v}")
        self._stats_path.write_text(json.dumps(stats, indent=4))

    def log_final_proof(self, rollout: "RolloutProofSearch", note: str):
        if not self._final_proofs_path.exists():
            with open(self._final_proofs_path, "w") as f:
                f.write(MINIF2F_HEADER + "\n")
                f.write(
                    "-- MiniF2F proofs found by linear rollouts\n\n"
                )

        proof = rollout.get_completed_proof()
        with open(self._final_proofs_path, "a") as f:
            if proof:
                f.write(proof + "\n")
            else:
                f.write(f"{rollout.theorem}\n")
            f.write(f"-- Note: {note}\n\n")

    def _calculate_stats(self, completed_rollouts: "list[RolloutProofSearch]") -> dict:
        total_steps = sum([r.total_steps for r in completed_rollouts])
        runtime = time.time() - self._start_time
        return {
            "completed": len(completed_rollouts),

            "proven": len([r for r in completed_rollouts if r.proven]),
            "proven_rate": len([r for r in completed_rollouts if r.proven]) / len(
                completed_rollouts) if completed_rollouts else 0,

            "total_steps": total_steps,
            "invalid_syntax_count": sum([r.invalid_syntax_count for r in completed_rollouts]),
            "invalid_syntax_rate": sum([r.invalid_syntax_count for r in completed_rollouts]) / total_steps,
            "exceptions_count": sum([r.total_exceptions for r in completed_rollouts]),
            "exceptions_rate": sum([r.total_exceptions for r in completed_rollouts]) / total_steps,
            "lean_errors_count": sum([r.total_lean_errors for r in completed_rollouts]),
            "lean_errors_rate": sum([r.total_lean_errors for r in completed_rollouts]) / total_steps,

            "total_expansions": sum([r.total_expansions for r in completed_rollouts]),
            "total_rollouts": sum([r.total_rollouts for r in completed_rollouts]),

            "runtime": runtime,
            "steps_per_second": total_steps / runtime,
        }

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised.
        while True:
            try:
                return func(*args, **kwargs)

            except errors:
                # Retry on specified errors.
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)

    return wrapper


class ModelProvider:
    # Model costs per 1K tokens in dollars
    MODEL_COSTS = {
        "gpt-4.1": {"input": 0.001, "output": 0.004},
    }

    def __init__(self, args: argparse.Namespace, logger: Logger):
        self.args = args
        self.logger = logger
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if not args.openai_key_file.exists():
            raise ValueError(f"OpenAI API key file {args.openai_key_file} does not exist.")
        with open(args.openai_key_file, "r") as f:
            openai_api_key = f.read().strip()
        self.client = openai.AsyncOpenAI(
            api_key=openai_api_key,
        )

    def get_cost(self) -> float:
        """Calculate total cost in dollars based on token usage."""
        if self.args.model not in self.MODEL_COSTS:
            return 0.0
        costs = self.MODEL_COSTS[self.args.model]
        input_cost = (self.total_input_tokens / 1000) * costs["input"]
        output_cost = (self.total_output_tokens / 1000) * costs["output"]
        return input_cost + output_cost

    def get_token_stats(self) -> dict:
        """Get statistics about token usage."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost": self.get_cost()
        }

    def log_token_stats(self):
        stats = self.get_token_stats()
        for k, v in stats.items():
            print(f"{k}: {v}")

    async def generate_single(self, inp: str) -> str:
        messages = [
            {
                "role": "developer",
                "content": get_system_prompt(self.args),
            },
            {
                "role": "user",
                "content": inp,
            }
        ]
        completion = await self.client.chat.completions.create(
            model=self.args.model,
            messages=messages,
            max_completion_tokens=self.args.max_completion_tokens,
            tool_choice=None,
        )
        output = completion.choices[0].message.content
        
        self.total_input_tokens += completion.usage.prompt_tokens
        self.total_output_tokens += completion.usage.completion_tokens
        
        print(inp)
        print("-->")
        print(output)
        print("--------")
        self.logger.log_model_output(messages, output)
        return output

    async def generate(self, input: list[str]) -> list[str]:
        return await asyncio.gather(*[self.generate_single(prompt) for prompt in input])


class ProofStatus(Enum):
    Incomplete = "incomplete"
    Completed = "completed"
    Error = "error"

BANNED_TACTICS = ["simp", "linarith", "ring", "norm_num", "cancel_denoms", "aesop", "omega", "decide"]
ALLOWED_TACTICS = [ "assumption", "exists", "intro", "rfl", "symm", "subst", "exact", "apply", "refine", "exfalso", "contradiction", "suffices", "change", "generalize", "specialize", "obtain", "show", "rw", "unfold", "replace", "constructor", "injection", "left", "right", "cases", "induction", "split", "by_cases", "trivial"]

def get_system_prompt(args: argparse.Namespace) -> str:
    prompt = """
Suggest the next tactic to use in a Lean 4 proof.
Only respond with the tactic enclosed in <tactic> </tactic> tags, nothing else.
Only output one Lean tactic (no informal text, no nested tactics, no comments, no sorry keyword).
    """.strip()
    if args.ban_complex_tactics:
        prompt += "\n" + f"""
You can only use one of the following tactics: {", ".join(ALLOWED_TACTICS)}
        """.strip()

    if args.method == "blackbox":
        prompt += "\n" + f"""
If the proof is already complete, you must generate the "done" tactic to finish the proof search.
        """.strip()

    return prompt
        

class RolloutProofSearch:
    def __init__(self, args: argparse.Namespace, theorem: str, logger: Logger):
        self.args = args
        self.theorem = theorem
        self.repl: LeanServer | None = None
        self.logger = logger

        # The current rollout - partial or completed proof.
        self.proof = []
        # List of tried tactics that did not work since the last successful step.
        self.failed_tactics = []
        # Whether a valid proof was found (in that case, it is inside self.proof).
        self.proven = False
        # Total number of rollout step attempts in current rollout.
        self.steps_in_rollout = 0

        # Total number of rollout step attempts in all rollouts.
        self.total_steps = 0
        # Number of step attempts that failed due to invalid syntax of model output.
        self.invalid_syntax_count = 0
        # Number of step attempts that failed due to an exception.
        self.total_exceptions = 0
        # Number of step attempts that failed due to a Lean error.
        self.total_lean_errors = 0
        # Number of completed rollouts.
        self.total_rollouts = 0
        # Number of outputs produced by the model.
        self.total_expansions = 0

        self._lean_state = None
        self._progress_bar = tqdm(
            total=args.max_steps * args.max_rollouts,
            desc=f"{theorem[:30]}{'...' if len(theorem) > 30 else ''}",
            leave=True,
        )

    async def next_prompt(self, timeout: float = 30) -> str:
        self.total_steps += 1
        self.steps_in_rollout += 1
        self._progress_bar.update(1)
        return await asyncio.wait_for(self._next_prompt(), timeout)

    async def _next_prompt(self) -> str:
        assert not self.proven
        user_message = "Suggest next tactic:\n" + self._get_proof_str() + "\n\n"
        if self.failed_tactics and self.args.method in ["graybox", "whitebox"]:
            failed_tactics_str = "\n".join(
                f"\"{tactic[:100]}\""
                for tactic in list(reversed(self.failed_tactics))[:5]
            )
            user_message += f"The following tactics were already tried and did not work, so do not try them again:\n{failed_tactics_str}\n\n"
        if self.args.method == "whitebox":
            if self._lean_state is None:
                status, lean_exc = await self._check_proof()
                if lean_exc:
                    raise lean_exc
                assert status == ProofStatus.Incomplete
            state_str = f"Currently open goals are listed bellow - it might be useful when selecting the tactic:\n{self._lean_state}"
            state_str = state_str[:512]  # Addressing degenerate cases where a type of a hypothesis is too long.
            state_str += "\n"
            user_message = f"{user_message}\n{state_str}"
        return user_message

    async def execute_step(self, response: str, timeout: float = 30):
        await asyncio.wait_for(self._execute_step(response), timeout)

    async def _execute_step(self, response: str):
        assert not self.proven
        proof_before = self._get_proof_str()
        self.total_expansions += 1

        tactic = self._parse_response(response)
        # before_postprocess = tactic
        if tactic:
            tactic = self._postprocess_tactic(tactic)
        # print(f"Before postprocess: '{before_postprocess}', after: '{tactic}'")


        status = None
        if not tactic:
            self.invalid_syntax_count += 1
        elif (
            "sorry" in tactic or
            "apply?" in tactic or
            self.args.ban_complex_tactics and (
                any(banned in tactic for banned in BANNED_TACTICS) or
                not any(tactic.startswith(allowed) for allowed in ALLOWED_TACTICS)
            )
        ):
            self.total_lean_errors += 1
            status = ProofStatus.Error
            if self.args.method in ["graybox", "whitebox"]:
                self.failed_tactics.append(tactic)
        else:
            if self.args.method in ["graybox", "whitebox"]:
                self.proof.append(tactic)
                status, lean_exc = await self._check_proof()
                if status == ProofStatus.Completed:
                    self.proven = True
                elif status == ProofStatus.Error:
                    self.total_lean_errors += 1
                    self.proof.pop()
                    self.failed_tactics.append(tactic)
                else:
                    self.failed_tactics = []
            else:
                assert self.args.method == "blackbox"
                if tactic.strip() == "done":
                    status, lean_exc = await self._check_proof()
                    if status == ProofStatus.Completed:
                        self.proven = True
                    else:
                        self.new_rollout()
                else:
                    self.proof.append(tactic)

        self.logger.log_rollout_step(proof_before, tactic, response, status)

    def new_rollout(self):
        self.total_rollouts += 1
        self.proof = []
        self.failed_tactics = []
        self.steps_in_rollout = 0
        self._lean_state = None

    def get_completed_proof(self) -> str | None:
        if not self.proven:
            return None
        return self._get_proof_str()

    async def stop(self):
        try:
            await self.repl.stop_async_safe()
        except Exception as e:
            print(f"Could not stop REPL: {e}")
        self._progress_bar.close()

    def _parse_response(self, response: str) -> str | None:
        if "</tactic>" not in response:
            # Address potential cut-off.
            response += "</tactic>"
        match = re.search(r"<tactic>(.*?)</tactic>", response, re.DOTALL)
        return match.group(1) if match else None

    def _postprocess_tactic(self, tactic: str) -> str:
        if ";" in tactic:
            tactic = tactic[:tactic.index(";")]

        return tactic.strip()

    async def _check_proof(self) -> tuple[ProofStatus, LeanInteractionException | None]:
        try:
            proof_str = self._get_proof_str(include_iterate_sorry=True)
            response = await self.repl.send_command_async(proof_str)
        except LeanInteractionException as e:
            return ProofStatus.Error, e

        if any(m["data"] == "Goals accomplished!" for m in response["messages"]):
            return ProofStatus.Completed, None

        assert "sorries" in response, f"No sorries in response:\n{proof_str}\n-->\n{response}"
        goals = [ReplGoalInfo.goal_from_repl_data(sorry_data["goalInfo"]) for sorry_data in response["sorries"]]
        self._lean_state = LeanProofState(goals)
        return ProofStatus.Incomplete, None

    async def _check_complete_proof(self) -> ProofStatus:
        completed_proof_str = self._get_proof_str()
        try:
            response = await self.repl.send_command_async(completed_proof_str)
            if not any(m["data"] == "Goals accomplished!" for m in response["messages"]):
                print(f"WARNING: No goals to be solved but the proof is not complete:\n {completed_proof_str}\n-->\n{response}")
                return ProofStatus.Error
            return ProofStatus.Completed
        except LeanInteractionException as e:
            print(f"WARNING: No goals to be solved but the proof threw an exception:\n {completed_proof_str}\n-->\n{e}")
            return ProofStatus.Error

    def _get_proof_str(self, include_sorry: bool=False, include_iterate_sorry: bool=False) -> str:
        assert not (include_iterate_sorry and include_sorry)
        steps = self.proof
        if include_iterate_sorry:
            steps = steps + ["iterate sorry"]
        if include_sorry:
            steps = steps + ["sorry"]

        return (
                self.theorem[:-len("sorry")] +
                ("\n" if len(steps) > 0 else "") +
                "\n".join([f"  {tactic}" for tactic in steps])
        )


async def start_repls(args: argparse.Namespace, rollouts: list[RolloutProofSearch]):
    async def _start_repl(rollout: RolloutProofSearch):
        print(f"Starting REPL for {rollout.theorem[:30]}{'...' if len(rollout.theorem) > 30 else ''}...")
        assert rollout.repl is None
        repl = LeanServer(args.repl_exe, args.project_path)
        await repl.start_async()
        await repl.send_command_async(MINIF2F_HEADER)
        rollout.repl = repl

    print(f"Starting {len(rollouts)} REPL(s)...")
    startup_tasks = [_start_repl(rollout) for rollout in rollouts]
    await asyncio.gather(*startup_tasks)


async def run_step(
        rollouts: list[RolloutProofSearch], model_provider: ModelProvider
) -> list[tuple[Exception, str] | None]:
    errors = [None] * len(rollouts)

    prompts_and_errors = await asyncio.gather(*[rollout.next_prompt() for rollout in rollouts], return_exceptions=True)
    prompts = cast(list[str] | None, [p for p in prompts_and_errors if not isinstance(p, Exception)])
    for i in range(len(rollouts)):
        if isinstance(prompts_and_errors[i], Exception):
            errors[i] = (prompts_and_errors[i], "in rollout.next_prompt()")

    outputs = await model_provider.generate([prompts.pop(0) for e in errors if e is None])
    assert len(prompts) == 0
    outputs_with_holes = [outputs.pop(0) if e is None else None for e in errors]
    assert len(outputs) == 0

    step_results = await asyncio.gather(
        *[
            rollout.execute_step(output) if output is not None else asyncio.sleep(0)
            for rollout, output in zip(rollouts, outputs_with_holes)
        ],
        return_exceptions=True,
    )

    for i in range(len(rollouts)):
        if errors[i]:
            continue  # Already failed on next_prompt().
        elif isinstance(step_results[i], Exception):
            errors[i] = (step_results[i], "in rollout.execute_step()")

    return errors


async def run_rollouts(
        args: argparse.Namespace,
        theorems: list[str],
        model_provider: ModelProvider,
        logger: Logger,
) -> list[RolloutProofSearch]:
    pending_theorems = copy.copy(theorems)
    rollouts: list[RolloutProofSearch] = []
    done_rollouts = []
    parallel_rollouts = args.parallel_rollouts

    while len(rollouts) + len(pending_theorems) > 0:
        while len(rollouts) < parallel_rollouts and pending_theorems:
            theorem = pending_theorems.pop(0)
            rollout = RolloutProofSearch(args, theorem, logger)
            rollouts.append(rollout)

        to_be_started = [r for r in rollouts if r.repl is None]
        if len(to_be_started) > 0:
            await start_repls(args, to_be_started)

        errors = await run_step(rollouts, model_provider)

        continuing_rollouts: list[RolloutProofSearch] = []
        for rollout, error_and_note in zip(rollouts, errors):
            done = False

            if rollout.proven:
                logger.log_final_proof(rollout, "proof found")
                done = True
            elif error_and_note:
                error, note = error_and_note
                rollout.total_exceptions += 1
                logger.log_exception(rollout.theorem, error, note=note)
                logger.log_incomplete_rollout(
                    rollout._get_proof_str(include_sorry=False), f"unhandled exception {note}: {error}"
                )

                if "next_prompt" in note:
                    # If next_prompt() fails, it is probably because the theorem statement is invalid.
                    logger.log_final_proof(rollout, "next_prompt() failed")
                    done = True
                else:
                    assert "execute_step" in note
                    rollout.new_rollout()
                    assert rollout.repl is not None
                    await rollout.repl.stop_async_safe()
                    rollout.repl = None  # Will be restarted.
            elif rollout.steps_in_rollout >= args.max_steps:
                logger.log_incomplete_rollout(rollout._get_proof_str(include_sorry=False), "max_steps reached")
                rollout.new_rollout()

            if not done and rollout.total_rollouts >= args.max_rollouts:
                logger.log_final_proof(rollout, "max_rollouts reached")
                done = True

            if done:
                done_rollouts.append(rollout)
                await rollout.stop()
            else:
                continuing_rollouts.append(rollout)

        if len(continuing_rollouts) < len(rollouts):
            logger.log_stats(done_rollouts)
            model_provider.log_token_stats()
        rollouts = continuing_rollouts

    return done_rollouts


def replace_theorem_with_example(theorems: list[str]) -> list[str]:
    modified_theorems = []
    for theorem in theorems:
        words = theorem.split()
        assert len(words) > 2
        modified_theorems.append(
            "example" + theorem[len(words[0] + " " + words[1]):]
        )
    return modified_theorems


def main():
    args = get_parser().parse_args()
    utils.resolve_paths(args)

    descriptor = utils.get_args_descriptor(args, param_whitelist=set(ARGS_WHITELIST))
    log_dir = args.output_dir / descriptor
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_dir}")
    logger = Logger(log_dir)

    utils.dump_args(args, log_dir)

    adapter = InterLMMiniF2FAdapter(args.benchmark_cache_dir)
    benchmark = adapter.fetch_minif2f()

    model_provider = ModelProvider(args, logger)

    theorems = benchmark.test_theorems
    print(f"Will try to prove {len(theorems)} theorems.")

    # Avoid repeated definition + self-references.
    theorems = replace_theorem_with_example(theorems)

    start_time = time.time()
    completed_rollouts = asyncio.run(run_rollouts(args, theorems, model_provider, logger))
    end_time = time.time()
    assert len(completed_rollouts) == len(theorems)

    print(f"Completed {len(completed_rollouts)} theorems.")
    print(f"Runtime: {end_time - start_time} seconds, {(end_time - start_time) / len(theorems)} seconds per theorem")
    logger.log_stats(completed_rollouts)
    model_provider.log_token_stats()

if __name__ == "__main__":
    main()

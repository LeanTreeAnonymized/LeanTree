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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import DynamicCache
from tqdm import tqdm

from lean_trees import utils
from lean_trees.repl_adapter.interaction import LeanServer, LeanInteractionException
from lean_trees.core.lean import LeanProofState
from eval.interlm_adapter import InterLMMiniF2FAdapter
from lean_trees.repl_adapter.data import ReplGoalInfo
from rollouts_dataset import ALLOWED_TACTICS
from model.lean.lean_model import LeanSerializer

MINIF2F_HEADER = (
    "import Mathlib\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology\n"
)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)

    parser.add_argument("--whitebox", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--max_gpus", type=int, default=0)

    parser.add_argument("--max_steps", type=int, default=25, help="Maximum number of step attempts in one rollout.")
    parser.add_argument("--max_rollouts", type=int, default=10)

    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_tactic_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)

    parser.add_argument("--benchmark_cache_dir", type=Path, default="benchmarks")
    parser.add_argument("--output_dir", type=Path, default="rollouts_finetuned")

    parser.add_argument("--repl_exe", type=Path, default="/home/kripner/repos/lean-repl-fork/.lake/build/bin/repl")
    parser.add_argument("--project_path", type=Path, default="/home/kripner/troja/arcoss-lean-repo-v4-19-rc2")

    parser.add_argument("--force", action="store_true")

    return parser


ARGS_WHITELIST = [
    "seed", "checkpoint", "max_steps", "max_rollouts", "whitebox", "max_tactic_length", "temperature", "repetition_penalty",
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

    def log_model_outputs(self, outputs: list[str]):
        with open(self._model_outputs_path, "a") as f:
            for output in outputs:
                quotes = '"""\n'
                f.write(f"{quotes}{output}\n{quotes}\n\n")

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


class SingleModel:
    def __init__(
            self,
            args,
            model,
            tokenizer,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompts: list[str]) -> tuple[list[str], list[str]]:
        encoded = self.tokenizer(
            prompts,
            max_length=self.args.max_input_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        # Create the logits processor to force starting with an allowed tactic
        first_token_processor = ForceFirstTokenFromAllowedList(self.tokenizer, ALLOWED_TACTICS)
        logits_processor = LogitsProcessorList([first_token_processor])

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.args.temperature,
                max_new_tokens=self.args.max_tactic_length,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=self.args.repetition_penalty,
                logits_processor=logits_processor,

                # Note: The '<|im_end|>' is a hack because some Qwen finetunes were trained to output im_end instead of endoftext...
                stop_strings=["<|im_end|>", "<|endoftext|>", "\n\n"],
                tokenizer=self.tokenizer,
            )
        # Extract only the new tokens (exclude prompt tokens).
        output_tokens = outputs[:, encoded["input_ids"].shape[1]:].tolist()
        output_decoded = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        # Log the whole output including prompt and thinking.
        all_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        del outputs
        del output_tokens
        torch.cuda.empty_cache()

        return output_decoded, all_decoded


_model: SingleModel | None = None


def _process_init(args: argparse.Namespace, device: str):
    global _model
    if _model is not None:
        print(f"Model already loaded for device {device}!")
        return

    dev_idx = int(device.split(":")[-1])
    print(f"Setting device {dev_idx}...")
    torch.cuda.set_device(dev_idx)

    print(f"Loading model for device {device}")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype="auto")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    assert tokenizer.pad_token is not None

    _model = SingleModel(args, model, tokenizer)
    print(f"Model created for device {device}")


class ForceFirstTokenFromAllowedList(LogitsProcessor):
    """
    Forces the model to start with one of the allowed tactics by checking prefixes
    and forcing the next token to continue a valid tactic path.
    """
    def __init__(self, tokenizer, allowed_words: list[str]):
        self.tokenizer = tokenizer
        self.allowed_tactics = allowed_words
        self.done = None
        self.prompt_length = None
        
        self.tactic_tokens = []
        for tactic in allowed_words:
            self.tactic_tokens.append(tokenizer(tactic, add_special_tokens=False).input_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        assert input_ids.ndim == 2
        assert scores.ndim == 2
        batch_size, vocab_size = scores.shape
            
        if self.done is None:
            self.done = [False] * batch_size
            self.prompt_length = input_ids.shape[1]
        if all(self.done):
            return scores
            
        all_prefix_tokens = input_ids[:, self.prompt_length:]
        modified_scores = scores.clone()
        for batch_idx in range(batch_size):
            if self.done[batch_idx]:
                continue
                
            prefix_tokens = all_prefix_tokens[batch_idx].tolist()
            
            # Find tactics that match this prefix.
            matching_tactics = []
            for tactic_idx, tactic_tokens in enumerate(self.tactic_tokens):
                if len(prefix_tokens) <= len(tactic_tokens) and prefix_tokens == tactic_tokens[:len(prefix_tokens)]:
                    matching_tactics.append(tactic_idx)
            
            if not matching_tactics:
                print(f"WARNING: Generated prefix doesn't match any tactic: {self.tokenizer.decode(prefix_tokens)}")
                self.done[batch_idx] = True
                continue
                
            # If any tactic is complete, we're done.
            if any(len(prefix_tokens) == len(self.tactic_tokens[tactic_idx]) for tactic_idx in matching_tactics):
                self.done[batch_idx] = True
                continue
                
            valid_next_tokens = set()
            for tactic_idx in matching_tactics:
                valid_next_tokens.add(self.tactic_tokens[tactic_idx][len(prefix_tokens)])
            
            # Mask out unallowed tokens.
            mask = torch.full((vocab_size,), -float("inf"), device=scores.device)
            for token_id in valid_next_tokens:
                mask[token_id] = 0
                
            modified_scores[batch_idx] = modified_scores[batch_idx] + mask
            
        return modified_scores


def _generate_on_device(sub_prompts):
    return _model.generate(sub_prompts)


class ModelProvider:
    def __init__(
            self,
            args: argparse.Namespace,
            logger: Logger,
    ):
        self.args = args
        self.checkpoint = args.checkpoint
        self.per_device_batch_size = args.per_device_batch_size
        self.logger = logger

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available.")
        if args.max_gpus > 0:
            n_gpus = min(n_gpus, args.max_gpus)
        print(f"Will use {n_gpus} GPU(s) for inference.")
        self.n_gpus = n_gpus

        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_gpus,
            # CUDA context must not be shared between processes, otherwise we get:
            # "Cannot re-initialize CUDA in forked subprocess"
            mp_context=multiprocessing.get_context("spawn"),
        )

    def __enter__(self):
        list(self.executor.map(
            _process_init,
            [self.args] * self.n_gpus,
            [f"cuda:{i}" for i in range(self.n_gpus)]
        ))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

    def generate(self, prompts: list[str]) -> list[str]:
        """
        Splits `prompts` into batches of size `per_device_batch_size`, 
        uses only as many GPUs as needed, runs generation in parallel,
        and returns the concatenated outputs.
        """
        batch_size = self.per_device_batch_size
        if len(prompts) > batch_size * self.n_gpus:
            raise ValueError(
                f"Number of prompts ({len(prompts)}) can be at most "
                f"per_device_batch_size ({batch_size}) × #GPUs ({self.n_gpus})."
            )

        # determine how many GPUs (i.e. batches) we need
        n_batches = (len(prompts) + batch_size - 1) // batch_size
        assert n_batches <= self.n_gpus

        chunks = [
            prompts[i * batch_size: (i + 1) * batch_size]
            for i in range(n_batches)
        ]

        print(f"Generating {len(chunks)} chunk(s) on {n_batches} GPU(s)...")
        per_gpu_outputs = list(self.executor.map(_generate_on_device, chunks))

        self.logger.log_model_outputs([sample for _, debug_log in per_gpu_outputs for sample in debug_log])
        return [out for batch, _ in per_gpu_outputs for out in batch]


class ProofStatus(Enum):
    Incomplete = "incomplete"
    Completed = "completed"
    Error = "error"


class RolloutProofSearch:
    def __init__(self, args: argparse.Namespace, theorem: str, tokenizer, logger: Logger):
        self.args = args
        self.theorem = theorem
        self.repl: LeanServer | None = None
        self.tokenizer = tokenizer
        self.logger = logger

        # The current rollout - partial or completed proof.
        self.proof = []
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
        if self.args.whitebox and self._lean_state is None:
            status, lean_exc = await self._check_proof()
            if lean_exc:
                raise lean_exc
            assert status == ProofStatus.Incomplete
            
        lean_state = None
        if self.args.whitebox:
            # Note: taking just the first goal is a hack, because the model only saw factorized states during training.
            lean_state = LeanProofState([self._lean_state.goals[0]])
        theorem_statement = self.theorem[:-len("sorry")]
        proof_prefix = self.proof
        return LeanSerializer.serialize_rollout_input_instruct(
            self.tokenizer,
            theorem_statement,
            proof_prefix,
            lean_state,
            self.args.whitebox,
        )

    async def execute_step(self, response: str, timeout: float = 30):
        await asyncio.wait_for(self._execute_step(response), timeout)

    # TODO!!!: blackbox should not be able to know that the state is solved
    async def _execute_step(self, response: str):
        assert not self.proven
        proof_before = self._get_proof_str()
        self.total_expansions += 1

        tactic = response.strip()

        if not tactic:
            self.invalid_syntax_count += 1
            status = None
        elif (
            "apply?" in tactic or
            not any(tactic.startswith(allowed) for allowed in ALLOWED_TACTICS)
        ):
            self.total_lean_errors += 1
            status = ProofStatus.Error
        else:
            self.proof.append(tactic)
            status, lean_exc = await self._check_proof()
            if status == ProofStatus.Completed:
                self.proven = True
            elif status == ProofStatus.Error:
                self.total_lean_errors += 1
                self.proof.pop()

        self.logger.log_rollout_step(proof_before, tactic, response, status)

    def new_rollout(self):
        self.total_rollouts += 1
        self.proof = []
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

    outputs = model_provider.generate([prompts.pop(0) for e in errors if e is None])
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
    parallel_rollouts = args.per_device_batch_size * model_provider.n_gpus
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    while len(rollouts) + len(pending_theorems) > 0:
        while len(rollouts) < parallel_rollouts and pending_theorems:
            theorem = pending_theorems.pop(0)
            rollout = RolloutProofSearch(args, theorem, tokenizer, logger)
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
    utils.setup_seeds(args.seed)

    descriptor = utils.get_args_descriptor(args, param_whitelist=set(ARGS_WHITELIST))
    log_dir = args.output_dir / descriptor
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_dir}")
    logger = Logger(log_dir)

    utils.dump_args(args, log_dir)

    adapter = InterLMMiniF2FAdapter(args.benchmark_cache_dir)
    benchmark = adapter.fetch_minif2f()

    print(f"Loading models from {args.checkpoint}...")
    model_provider = ModelProvider(args, logger)

    theorems = benchmark.test_theorems
    print(f"Will try to prove {len(theorems)} theorems.")

    # Avoid repeated definition + self-references.
    theorems = replace_theorem_with_example(theorems)

    start_time = time.time()
    with model_provider:
        completed_rollouts = asyncio.run(run_rollouts(args, theorems, model_provider, logger))
    end_time = time.time()
    assert len(completed_rollouts) == len(theorems)

    print(f"Completed {len(completed_rollouts)} theorems.")
    print(f"Runtime: {end_time - start_time} seconds, {(end_time - start_time) / len(theorems)} seconds per theorem")
    logger.log_stats(completed_rollouts)


if __name__ == "__main__":
    main()

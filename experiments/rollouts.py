import argparse
from collections import defaultdict
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
from lean_trees.core.lean import LeanProofState, LeanGoal
from eval.interlm_adapter import InterLMMiniF2FAdapter
from lean_trees.repl_adapter.data import ReplGoalInfo

MINIF2F_HEADER = (
    "import Mathlib\n"
    "set_option maxHeartbeats 0\n"
    "open BigOperators Real Nat Topology\n"
)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whitebox", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    # deepseek-ai/DeepSeek-Prover-V2-7B
    # Qwen/Qwen3-32B
    # Qwen/Qwen3-30B-A3B
    # meta-llama/Llama-3.1-8B-Instruct
    # deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    parser.add_argument("--checkpoint", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--max_gpus", type=int, default=0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--ban_complex_tactics", type=bool, default=True)

    parser.add_argument("--max_steps", type=int, default=25, help="Maximum number of step attempts in one rollout.")
    parser.add_argument("--max_rollouts", type=int, default=10)

    parser.add_argument("--max_thinking_length", type=int, default=512)  # TODO
    parser.add_argument("--max_tactic_length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition_penalty", type=float, default=1.5)

    parser.add_argument("--benchmark_cache_dir", type=Path, default="benchmarks")
    parser.add_argument("--output_dir", type=Path, default="rollouts")

    parser.add_argument("--repl_exe", type=Path, default="/home/kripner/repos/lean-repl-fork/.lake/build/bin/repl")
    parser.add_argument("--project_path", type=Path, default="/home/kripner/troja/arcoss-lean-repo-v4-19-rc2")

    parser.add_argument("--force", action="store_true")

    return parser


ARGS_WHITELIST = [
    "seed", "checkpoint", "max_steps", "max_rollouts", "whitebox", "enable_thinking",
    "max_thinking_length", "max_tactic_length", "temperature", "repetition_penalty", "ban_complex_tactics",
]

def byte_size_to_human(size_bytes):
    if size_bytes == 0:
        return "0 B"
    power_names = ["B", "KB", "MB", "GB", "TB"]
    power_idx = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, power_idx)
    size = round(size_bytes / power, 2)
    return f"{size} {power_names[power_idx]}"


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


class InjectAtStepIfMissing(LogitsProcessor):
    """
    After exactly `trigger_len` new tokens beyond the prompt have been generated,
    inject the given `injection_ids` one by one, then stop forcing and let the model continue normally.
    If the sentinel token is already present before the trigger, do nothing.
    """

    def __init__(self, trigger_len: int, injection_ids: list[int], sentinel_token_id: int | None):
        assert len(injection_ids) > 0
        self.trigger_len = trigger_len
        self.injection_ids = injection_ids
        self.sentinel_token_id = sentinel_token_id

        self.prompt_len = None
        self.injected_pos = 0
        self.done = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        assert input_ids.ndim == 2
        assert scores.ndim == 2
        assert input_ids.shape[0] == scores.shape[0]
        batch_size, seq_len = input_ids.shape
        vocab_size = scores.shape[1]

        # On the very first invocation, record how many tokens were in the prompt
        if self.prompt_len is None:
            self.prompt_len = seq_len

        # If already injected (or it's not the right step), do nothing
        generated = seq_len - self.prompt_len
        if self.done or generated != self.trigger_len + self.injected_pos:
            return scores

        # Inject the given `injection_ids` for each batch item that is still missing the sentinel token
        forced_scores = scores.clone()
        for i in range(batch_size):
            if (
                self.sentinel_token_id is not None and
                self.sentinel_token_id in input_ids[i][:seq_len - self.injected_pos].tolist()
            ):
                continue
            # zero-out all logits except the one for token_id
            mask = torch.full((vocab_size,), -float("inf"), device=scores.device)
            mask[self.injection_ids[self.injected_pos]] = 0.0
            forced_scores[i] = mask

        self.injected_pos += 1
        if self.injected_pos == len(self.injection_ids):
            self.done = True
        return forced_scores


class SingleModel:
    def __init__(
            self,
            args,
            model,
            tokenizer,
            precomputed_kv_cache: DynamicCache | None,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

        self.max_thinking_length = args.max_thinking_length
        self.max_tactic_length = args.max_tactic_length
        self.temperature = args.temperature
        self.repetition_penalty = args.repetition_penalty

        self._kv_cache = precomputed_kv_cache

    def generate(self, prompts: list[str]) -> tuple[list[str], list[str]]:
        if self.args.checkpoint.startswith("deepseek-ai/DeepSeek-R1-Distill-Llama"):
            # See https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B#usage-recommendations
            messages = [[
                {
                    "role": "user",
                    "content": USER_PROMPT + "\n\n" + prompt,
                }
            ] for prompt in prompts]
        else:
            messages = [[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ] for prompt in prompts]

        if self.args.checkpoint.startswith("Qwen/Qwen3"):
            return self._generate_qwen(messages)
        elif self.args.checkpoint.startswith("deepseek-ai/DeepSeek-Prover") or self.args.checkpoint.startswith("meta-llama/Llama-3.1"):
            return self._generate_deepseek_or_llama(messages)
        elif self.args.checkpoint.startswith("deepseek-ai/DeepSeek-R1-Distill-Llama"):
            return self._generate_deepseek_llama(messages)
        else:
            raise ValueError(f"Unknown model family: {self.args.checkpoint}")

    def _generate_qwen(self, messages: list[list[dict]]) -> tuple[list[str], list[str]]:
        injection_text = "... OK, done.</think>\n<tactic>"
        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # https://huggingface.co/Qwen/Qwen3-32B/blob/main/README.md
            # enable_thinking=self.args.enable_thinking,
            # We add the <think>...</think> tokens manually. With empty <think></think> tags, the model sometimes
            # starts thinking instead of generating the tactic, so we put something inside even if thinking is disabled.
            enable_thinking=True,
        )
        if not self.args.enable_thinking:
            texts = [text + "<think>" + injection_text for text in texts]
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        kv_cache = None
        if self._kv_cache is not None:
            # The cache is extended during generation.
            kv_cache = copy.deepcopy(self._kv_cache)
            if len(messages) < self.args.per_device_batch_size:
                kv_cache.key_cache = [key_tensor[:len(messages)] for key_tensor in kv_cache.key_cache]
                kv_cache.value_cache = [value_tensor[:len(messages)] for value_tensor in kv_cache.value_cache]

        logits_processors = None
        if self.args.enable_thinking:
            # injection_text = "... Wrapping up:"
            # injection_ids = self.tokenizer(injection_text, add_special_tokens=False).input_ids
            #
            # # TODO: this should also only fire when </think> not present
            # wrapping_up_injector = InjectAtStep(
            #     trigger_len=TODO,
            #     injection_ids=injection_ids,  # then inject our delimiter
            # )

            injection_ids = self.tokenizer(
                injection_text,
                add_special_tokens=False,
            ).input_ids
            end_thinking_injector = InjectAtStepIfMissing(
                trigger_len=self.max_thinking_length,
                sentinel_token_id=151668,  # </think>
                injection_ids=injection_ids,
            )
            logits_processors = LogitsProcessorList([end_thinking_injector])

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                max_new_tokens=self.max_thinking_length + self.max_tactic_length if self.args.enable_thinking else self.max_tactic_length,
                past_key_values=kv_cache,
                logits_processor=logits_processors,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Extract only the new tokens (exclude prompt tokens).
        output_tokens = outputs[:, encoded["input_ids"].shape[1]:].tolist()
        tactic_decoded = []
        for tactic_tokens_single in output_tokens:
            if self.args.enable_thinking:
                try:
                    # rindex finding 151668 (</think>)
                    index = len(tactic_tokens_single) - tactic_tokens_single[::-1].index(151668)
                except ValueError:
                    index = 0
                tactic_decoded.append(
                    self.tokenizer.decode(tactic_tokens_single[index:], skip_special_tokens=True).strip("\n")
                )
            else:
                tactic_decoded.append(
                    "<tactic>" +
                    self.tokenizer.decode(tactic_tokens_single, skip_special_tokens=True).strip("\n")
                )

        # Log the whole output including prompt and thinking.
        all_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        del outputs
        del output_tokens
        torch.cuda.empty_cache()

        return tactic_decoded, all_decoded

    def _generate_deepseek_or_llama(self, messages: list[list[dict]]) -> tuple[list[str], list[str]]:
        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.args.enable_thinking:
            texts = [text + "Let's think step by step.\n" for text in texts]
        else:
            texts = [text + "\n<tactic>" for text in texts]

        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        kv_cache = None
        if self._kv_cache is not None:
            # The cache is extended during generation.
            kv_cache = copy.deepcopy(self._kv_cache)
            if len(messages) < self.args.per_device_batch_size:
                kv_cache.key_cache = [key_tensor[:len(messages)] for key_tensor in kv_cache.key_cache]
                kv_cache.value_cache = [value_tensor[:len(messages)] for value_tensor in kv_cache.value_cache]

        logits_processors = None
        if self.args.enable_thinking:
            injection_ids = self.tokenizer(
                "... OK, done.\n<tactic>",
                add_special_tokens=False,
            ).input_ids
            end_thinking_injector = InjectAtStepIfMissing(
                trigger_len=self.max_thinking_length,
                sentinel_token_id=None,  # </think>
                injection_ids=injection_ids,
            )
            logits_processors = LogitsProcessorList([end_thinking_injector])

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_thinking_length + self.max_tactic_length if self.args.enable_thinking else self.max_tactic_length,
                past_key_values=kv_cache,
                logits_processor=logits_processors,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Extract only the new tokens (exclude prompt tokens).
        output_tokens = outputs[:, encoded["input_ids"].shape[1]:].tolist()
        tactic_decoded = []
        for tactic_tokens_single in output_tokens:
            tactic = self.tokenizer.decode(tactic_tokens_single, skip_special_tokens=True).strip("\n")
            if not self.args.enable_thinking:
                tactic = "<tactic>" + tactic
            else:
                if "<tactic>" in tactic:
                    tactic = tactic[tactic.index("<tactic>"):]
            tactic_decoded.append(tactic)

        # Log the whole output including prompt and thinking.
        all_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        del outputs
        del output_tokens
        torch.cuda.empty_cache()

        return tactic_decoded, all_decoded

    def _generate_deepseek_llama(self, messages: list[list[dict]]) -> tuple[list[str], list[str]]: 
        injection_text = "\n... OK, I'm now done thinking and will output the tactic.\n</think>\n<tactic>"
        texts = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not self.args.enable_thinking:
            texts = [text + "<think>" + injection_text for text in texts]
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        logits_processors = None
        if self.args.enable_thinking:
            injection_ids = self.tokenizer(
                injection_text,
                add_special_tokens=False,
            ).input_ids
            end_thinking_injector = InjectAtStepIfMissing(
                trigger_len=self.max_thinking_length,
                sentinel_token_id=128014,  # </think>
                injection_ids=injection_ids,
            )
            logits_processors = LogitsProcessorList([end_thinking_injector])

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_thinking_length + self.max_tactic_length if self.args.enable_thinking else self.max_tactic_length,
                logits_processor=logits_processors,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Extract only the new tokens (exclude prompt tokens).
        output_tokens = outputs[:, encoded["input_ids"].shape[1]:].tolist()
        tactic_decoded = []
        for tactic_tokens_single in output_tokens:
            if self.args.enable_thinking:
                try:
                    # rindex finding 128014 (</think>)
                    index = len(tactic_tokens_single) - tactic_tokens_single[::-1].index(128014)
                except ValueError:
                    index = 0
                tactic_decoded.append(
                    self.tokenizer.decode(tactic_tokens_single[index:], skip_special_tokens=True).strip("\n")
                )
            else:
                tactic_decoded.append(
                    "<tactic>" +
                    self.tokenizer.decode(tactic_tokens_single, skip_special_tokens=True).strip("\n")
                )

        # Log the whole output including prompt and thinking.
        all_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        del outputs
        del output_tokens
        torch.cuda.empty_cache()

        return tactic_decoded, all_decoded


_model: SingleModel | None = None


def _process_init(args: argparse.Namespace, device: str):
    global _model
    if _model is not None:
        print(f"Model already loaded for device {device}!")
        return

    dev_idx = int(device.split(":")[-1])
    print(f"Setting device {dev_idx}...")
    torch.cuda.set_device(dev_idx)

    # See https://github.com/huggingface/safetensors/issues/562#issuecomment-2634544710
    # Doesn't seem to help.
    # checkpoint_dir = Path("/home/kripner/troja/project_1/.hf_home/hub/models--Qwen--Qwen3-32B/snapshots/30b8421510892303dc5ddd6cd0ac90ca2053478d")
    # if checkpoint_dir.exists():
    #     print(f"Pre-loading safetensors files from {checkpoint_dir} ...")
    #     for filename in checkpoint_dir.glob("*.safetensors"):
    #         with open(filename, "rb") as f:
    #             f.read()

    print(f"Loading model for device {device}")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype="auto")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tokenizer.pad_token is None:
        print("Setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    # TODO
    kv_cache = None
    # kv_cache = _precompute_kv_cache(args, model, tokenizer)
    # kv_cache.key_cache = [key_tensor.to(device) for key_tensor in kv_cache.key_cache]
    # kv_cache.value_cache = [value_tensor.to(device) for value_tensor in kv_cache.value_cache]

    _model = SingleModel(args, model, tokenizer, kv_cache)
    print(f"Model created for device {device}")


# TODO: does not work for DeepSeek-R1-Distill-Llama where the system prompt is prepended to the user message
def _precompute_kv_cache(args: argparse.Namespace, model, tokenizer):
    print("Precomputing KV-cache for the system prompt...")

    prefix = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "!!!!",
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix = prefix[:prefix.index("!!!!")]

    inputs = tokenizer(
        [prefix] * args.per_device_batch_size,
        return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        kv_cache = model(**inputs).past_key_values
    print(f"KV-cache size: {len(kv_cache)}")
    return kv_cache


def _generate_on_device(idx_and_prompts):
    idx, sub_prompts = idx_and_prompts
    # print(f"Generating on GPU {idx} with {len(sub_prompts)} prompt(s)...")
    return _model.generate(sub_prompts)


def _generate_on_device_with_profiling(idx_and_prompts, profile_file: Path = None):
    idx, sub_prompts = idx_and_prompts
    if profile_file is None:
        profile_file = Path(f"generate_snapshot_{idx}.pickle")

    if profile_file.exists():
        return _model.generate(sub_prompts)

    torch.cuda.memory._record_memory_history()
    outputs = _model.generate(sub_prompts)
    torch.cuda.memory._dump_snapshot(str(profile_file))
    return outputs


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
        per_gpu_outputs = list(self.executor.map(_generate_on_device, enumerate(chunks)))

        self.logger.log_model_outputs([sample for _, debug_log in per_gpu_outputs for sample in debug_log])
        return [out for batch, _ in per_gpu_outputs for out in batch]


class ProofStatus(Enum):
    Incomplete = "incomplete"
    Completed = "completed"
    Error = "error"

BANNED_TACTICS = ["simp", "linarith", "ring", "norm_num", "cancel_denoms", "aesop", "omega", "decide"]

ALLOWED_TACTICS = [
    "assumption",
    "exists",
    "intro",
    "rfl",
    "symm",
    "subst",
    "exact",
    "apply",
    "refine",
    "exfalso",
    "contradiction",
    "suffices",
    "change",
    "generalize",
    "specialize",
    "obtain",
    "show",
    "rw",
    "unfold",
    "replace",
    "constructor",
    "injection",
    "left",
    "right",
    "cases",
    "induction",
    "split",
    "by_cases",
    "trivial",
]

# TODO: for DeepSeek-Prover, we should use a ```lean ... ``` code block instead of <tactic> ... </tactic> tags
SYSTEM_PROMPT = f"""
Lean 4.
Suggest a next tactic to use in the proof that user provides (user also may provide the current internal Lean state).
Only respond with the tactic enclosed in <tactic> </tactic> tags, nothing else.
Only output one Lean tactic (no informal text, no nested tactics, no comments).
The output tactic must not contain the sorry keyword.
You can only use one of the following tactics: {", ".join(ALLOWED_TACTICS)}
""".lstrip()

# Used for DeepSeek-R1-Distill-Llama which does not support the system prompt.
USER_PROMPT = f"""
Suggest a next tactic to use in the Lean 4 proof listed bellow (bellow might also be the current internal Lean state).
Only respond with the tactic enclosed in <tactic> </tactic> tags, nothing else.
Only output one Lean tactic (no informal text, no nested tactics, no comments).
The output tactic must not contain the sorry keyword.
You can only use one of the following tactics: {", ".join(ALLOWED_TACTICS)}
""".lstrip()


# TODO: REMOVE
# SYSTEM_PROMPT = """
# Your job is to extract all the variables (including hypotheses) in the theorem and copy them on the output. Only output the variables, nothing else, followed by an exclamation mark.
# """.strip()

_rw_pattern = re.compile(r'(\brw)\s+(?!\[)(.+?)(?=(?:\s+at\b|$))')
_rewrite_pattern = re.compile(r'(\brewrite)\s+(?!\[)(.+?)(?=(?:\s+at\b|$))')

def goal_to_theorem(goal: LeanGoal) -> str:
    type_to_hyps = defaultdict(list)
    for h in goal.hypotheses:
        type_to_hyps[h.type].append(h.user_name)

    hyps_str = []
    for type, names in sorted(type_to_hyps.items(), key=lambda x: len(x[0])):
        hyps_str.append(f"({' '.join(names)} : {type})")

    return f"example {" ".join(hyps_str)} : {goal.type} := by"

class RolloutProofSearch:
    def __init__(self, args: argparse.Namespace, theorem: str, logger: Logger):
        self.args = args
        self.theorem = theorem
        self.repl: LeanServer | None = None
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
        user_message = "Suggest next tactic in a Lean 4 proof:\n" + self._get_proof_str() + "\n"
        if self.args.whitebox:
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


        if not tactic:
            self.invalid_syntax_count += 1
            status = None
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

    def _parse_response(self, response: str) -> str | None:
        if "</tactic>" not in response:
            # Address potential cut-off.
            response += "</tactic>"
        match = re.search(r"<tactic>(.*?)</tactic>", response, re.DOTALL)
        return match.group(1) if match else None

    def _postprocess_tactic(self, tactic: str) -> str:
        if not tactic:
            return tactic

        lines = [
            line for line in tactic.split("\n")
            if line.strip() and not line.strip().startswith("--")
        ]

        if not lines:
            return tactic

        first_indent = len(tactic) - len(tactic.lstrip())
        i = 1
        while i < len(lines) and len(lines[i]) - len(lines[i].lstrip()) > first_indent:
            i += 1
        tactic_lines = [line.strip() for line in lines[:i]]
        tactic = "\n".join(tactic_lines).strip()

        if "<;>" in tactic and not tactic.startswith("have "):
            tactic = tactic[:tactic.index("<;>")]
        if " with " in tactic:
            tactic = tactic[:tactic.index(" with ")]
        if ":= by" in tactic:
            tactic = tactic[:tactic.index(":= by")]
        if ";" in tactic:
            tactic = tactic[:tactic.index(";")]
        tactic = tactic.replace("`", "")
        tactic = tactic.rstrip(",")

        if tactic.startswith("by "):
            tactic = tactic[len("by "):]

        tactic = _rw_pattern.sub(r'\1 [\2]', tactic)
        tactic = _rewrite_pattern.sub(r'\1 [\2]', tactic)

        return tactic.strip()

    async def _check_proof(self) -> tuple[ProofStatus, LeanInteractionException | None]:
        try:
            proof_str = self._get_proof_str(include_iterate_sorry=True)
            response = await self.repl.send_command_async(proof_str)
        except LeanInteractionException as e:
            # if "no goals to be solved" in str(e):
            #     # Double-check.
            #     return await self._check_complete_proof(), None
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

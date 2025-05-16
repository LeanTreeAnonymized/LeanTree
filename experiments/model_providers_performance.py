import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Iterator, Iterable
import time

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.eq import EqEnv, EqEnvMetadata, EqTheorem
from model.eq.huggingface_model import EqHuggingfaceModel
from model.eq.eq_model import EqModel
from core.eq import EqStepSerializer
from model.eq.vllm_model import EqVllmModel
import core.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("dataset_file", type=Path)
parser.add_argument("checkpoint", type=str)
parser.add_argument("--generation_method", type=str, choices=["sampling", "beam_search"])
parser.add_argument("--max_theorems", type=int, default=1000)
# batch_size=3 seems to be the biggest that can fit on A100 40GB with num_samples=32.
parser.add_argument("--batch_size", type=int, default=3,
                    help="Will be ignored by vLLM which chooses the batch size automatically.")
parser.add_argument("--parallelism", type=int, default=64)
parser.add_argument("--num_samples", type=int, default=32)
parser.add_argument("--output_dir", type=Path, default="plots/model_providers_performance")
parser.add_argument("--load_existing", action="store_true", help="Load already computed results from the log_dir.")

PARAMS_DESCRIPTOR_BLACKLIST = {
    "dataset_file",
    "output_dir"
}

EXPERIMENT_NAMES = [
    "huggingface",
    "huggingface-tf32",
    "vllm",
]

def main(args):
    assert torch.cuda.is_available()
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    print(f"GPU name: {device_name}")

    args_descriptor = utils.get_args_descriptor(
        args, param_blacklist=PARAMS_DESCRIPTOR_BLACKLIST, include_slurm_id=False, include_time=False,
    )
    log_dir = args.output_dir / args_descriptor
    log_dir.mkdir(exist_ok=True, parents=True)
    print(f"Will save results to {log_dir}")

    theorems = list(islice(load_theorems(args.dataset_file), args.max_theorems))

    existing_results = {}
    if args.load_existing:
        with open(log_dir / "results.json") as f:
            existing_results = json.load(f)
            print(f"Loaded existing results: {existing_results}")
    new_results = run_measurements(args, theorems, set(EXPERIMENT_NAMES) - existing_results.keys())
    results = {
        **existing_results,
        **new_results,
    }

    utils.dump_args(args, log_dir)
    with open(log_dir / "args.json", "w") as f:
        data = {
            **{k: str(v) for k, v in args.__dict__.items()},
            "gpu_name": device_name,
            "theorems": len(theorems),
        }
        json.dump(data, f, indent=4, sort_keys=True)
    with open(log_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)

    save_plot(results, log_dir / "results.png")


def save_plot(results: dict[str, dict], plot_path: Path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))

    tok_per_sec = {model: results[model]["tokens_per_second"] for model in results}
    tok_per_sample = {model: results[model]["tokens_per_sample"] for model in results}

    # Throughput.
    ax[0].bar(*zip(*tok_per_sec.items()), width=0.5)
    ax[0].set_title("Throughput")
    ax[0].set_ylabel("tok/s")
    ax[0].set_ylim(0, max(tok_per_sec.values()) + 5)
    ax[0].grid(axis="y", linestyle="--", alpha=0.7)

    # Token count.
    ax[1].bar(*zip(*tok_per_sample.items()), width=0.5)
    ax[1].set_title("Sample Length")
    ax[1].set_ylabel("tok/sample")
    ax[1].set_ylim(0, max(tok_per_sample.values()) + 5)
    ax[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_path)


def run_measurements(args: argparse.Namespace, theorems: list[EqTheorem], experiments: set[str]) -> dict:
    env = EqEnv.create(EqEnvMetadata.from_dict({"simple_env": False, "n_vars": 5}), seed=0)
    theorems = list(theorems)
    results = {}

    if "huggingface" in experiments:
        torch.backends.cuda.matmul.allow_tf32 = False
        huggingface_model = EqHuggingfaceModel(
            checkpoint=args.checkpoint, env=env, batch_size=args.batch_size, generation_method=args.generation_method
        )
        results["huggingface"] = run_measurement(args, huggingface_model, theorems, "Huggingface")
        del huggingface_model

    if "huggingface-tf32" in experiments:
        torch.backends.cuda.matmul.allow_tf32 = True
        huggingface_model = EqHuggingfaceModel(
            checkpoint=args.checkpoint, env=env, batch_size=args.batch_size, generation_method=args.generation_method
        )
        results["huggingface-tf32"] = run_measurement(args, huggingface_model, theorems, "Huggingface (TF32)")
        del huggingface_model

    if "vllm" in experiments:
        vllm_model = EqVllmModel(checkpoint=args.checkpoint, env=env, generation_method=args.generation_method)
        results["vllm"] = run_measurement(args, vllm_model, theorems, "vLLM")
        del vllm_model

    return results


def run_measurement(args: argparse.Namespace, model_provider: EqModel, theorems: list[EqTheorem], name: str) -> dict:
    print(f"Running {name} ...")
    start_time = time.time()

    total_tokens = 0
    for i in tqdm(range(0, len(theorems), args.parallelism)):
        batch = theorems[i:i + args.parallelism]
        result, stats = model_provider.predict(batch, args.num_samples)
        assert len(result) == len(batch)
        assert stats.generated_tokens > 0
        total_tokens += stats.generated_tokens

    end_time = time.time()
    total_seconds = end_time - start_time

    tokens_per_second = total_tokens / total_seconds
    print(f"{name} tok/s: {tokens_per_second}")
    tokens_per_sample = total_tokens / len(theorems) / args.num_samples
    print(f"{name} tok/sample: {tokens_per_sample}")
    return {
        "tokens_per_second": tokens_per_second,
        "tokens_per_sample": tokens_per_sample,
    }


def load_theorems(dataset_file: Path) -> Iterator[EqTheorem]:
    with open(dataset_file) as f:
        for i, line in enumerate(f):
            if i == 0:
                # Metadata header.
                env = EqEnv.create(EqEnvMetadata.from_dict(json.loads(line)), seed=0)
                continue
            text = json.loads(line)["text"]

            try:
                tactic = EqStepSerializer.deserialize_tactic(text, env)
            except Exception as e:
                print(f"Deserialization error on line {i + 1}: {e}")
                raise
            yield EqTheorem(tactic.initial_eq)


if __name__ == "__main__":
    main(parser.parse_args())

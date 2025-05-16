import argparse
import asyncio
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Final, Callable, AsyncIterator, Iterator, TextIO
import subprocess

import numpy as np

from eval.data import ProofSearchResult, EvalStats
from eval.interlm_adapter import InterLMMiniF2FAdapter
from eval.verifier import export_lean_proofs
from lean_trees import utils
from lean_trees.repl_adapter.interaction import LeanServer
from lean_trees.repl_adapter.server_pool import LeanServerPool
from lean_trees.utils import Logger
from model.lean.lean_model import LeanModel
from search.base import ProofSearch
from search.methods.mcts_search import MCTSProofSearch
from search.methods.mock_proof_search import MockProofSearch
from search.methods.random_search import RandomProofSearch


# TODO: we should definitely export the goal states as well, e.g. to facilitate robust verification of the tree
#   This further hints towards unifying this with the ProofTreeVerifier module

# TODO: baselines: GPTo4.5, o1, o3


# TODO: instead of packaging imports and context in env_setup, we should be able to construct and pass a LeanFile with
#  that info
class LeanEvaluator:
    @classmethod
    async def evaluate_single_async(
            cls, method: ProofSearch, theorem_with_sorry: str, env: LeanServer
    ) -> ProofSearchResult:
        try:
            proof_state = await env.start_proof_from_sorry_async(theorem_with_sorry)
        except Exception as e:
            print("Could not start the proof!")
            traceback.print_exc()
            return ProofSearchResult.from_error(theorem_with_sorry, e)

        # noinspection PyBroadException
        try:
            start_time = time.time()
            maybe_proof, metadata = await method.search_async(proof_state)
            duration = time.time() - start_time
        except Exception as e:
            traceback.print_exc()
            return ProofSearchResult.from_error(theorem_with_sorry, e)
        if maybe_proof is None:
            return ProofSearchResult.from_proof_not_found(theorem_with_sorry, duration, metadata)
        return ProofSearchResult.from_proof_found(theorem_with_sorry, maybe_proof, duration, metadata)


class ParallelLeanEvaluator:
    PROOF_SEARCH_TIMEOUT: Final[int] = 600  # In seconds.

    def __init__(
            self,
            repl_exe: Path,
            project_path: Path,
            method_provider: Callable[[], ProofSearch],
            header: str,
            num_repls: int,
            max_concurrency: int,
            logger: Logger | None = None,
    ):
        self.repl_exe = repl_exe
        self.project_path = project_path
        self.method_provider = method_provider
        self.header = header
        self.max_concurrency = max_concurrency
        self.logger = logger

        self.server_pool = LeanServerPool(
            repl_exe,
            project_path,
            max_servers=num_repls,
            env_setup_async=self._env_setup_async,
            logger=self.logger,
        )

    async def _env_setup_async(self, env: LeanServer):
        await env.send_command_async(self.header)

    async def evaluate_single_async(self, theorem_with_sorry: str, env: LeanServer) -> ProofSearchResult:
        return await LeanEvaluator.evaluate_single_async(
            self.method_provider(),
            theorem_with_sorry,
            env,
        )

    async def evaluate_all_async(self, theorems_with_sorry: list[str]) -> AsyncIterator[ProofSearchResult]:
        async def evaluate_with_timeout(thm):
            try:
                async with semaphore:
                    async with await self.server_pool.get_server_async() as env:
                        result = await asyncio.wait_for(
                            self.evaluate_single_async(thm, env),
                            timeout=self.PROOF_SEARCH_TIMEOUT
                        )
                        return result
            except asyncio.TimeoutError:
                return ProofSearchResult.from_timed_out(thm, self.PROOF_SEARCH_TIMEOUT)

        # Ramp up the servers.
        await self.server_pool.max_out_servers_async()
        # Run tasks concurrently with a limit on concurrency.
        semaphore = asyncio.Semaphore(self.max_concurrency)

        tasks = [evaluate_with_timeout(thm) for thm in theorems_with_sorry]
        # Yield results as they complete.
        for task in asyncio.as_completed(tasks):
            yield await task

    async def evaluate_all_serial_async(self, theorems_with_sorry: list[str]) -> AsyncIterator[ProofSearchResult]:
        for thm in theorems_with_sorry:
            async with await self.server_pool.get_server_async() as env:
                yield await self.evaluate_single_async(thm, env)


def aggregate_debug_times(results: list[ProofSearchResult]) -> dict:
    debug_times = defaultdict(lambda: {"total": 0.0, "count": 0.0})
    for result in results:
        for name, data in result.metadata["debug_times"].items():
            debug_times[name]["total"] += data["total"]
            debug_times[name]["count"] += data["count"]
    for name, data in debug_times.items():
        data["avg"] = data["total"] / data["count"]
    return debug_times


def print_debug_times(results: list[ProofSearchResult]):
    results = [result for result in results if result.metadata and result.metadata.get("debug_times")]
    debug_times = aggregate_debug_times(results)
    total_runtime = sum(result.runtime for result in results if "debug_times" in result.metadata)
    total_measured = sum(data["avg"] for data in debug_times.values())

    print(f"Total runtime: {total_runtime:.4f} seconds")
    print(f"Total measured time: {total_measured:.4f} seconds")
    for name, data in debug_times.items():
        print(
            f"{name}: {data['avg']:.4f} seconds ({data['avg'] / total_measured * 100:.4f}% of measured) ({data['avg'] / total_runtime * 100:.4f}% of total)")
    print(
        f"Non-measured time: {total_runtime - total_measured:.4f} seconds ({100 - total_measured / total_runtime * 100:.4f}%)")


async def run_eval_async(args: argparse.Namespace, theorems: list[str], evaluator: ParallelLeanEvaluator,
                         out_f: TextIO):
    stats = EvalStats.empty()
    iterator = (
        evaluator.evaluate_all_async(theorems)
        if not args.serial
        else evaluator.evaluate_all_serial_async(theorems)
    )
    results = []
    async for result in iterator:
        stats.search_results.append(result)
        results.append(result)

        print(f"====== Theorem ======")
        print(f"{result.theorem}\n--")
        if result.proof:
            print(result.proof.pretty_print())
        if result.error:
            print(f"ERROR: {result.error}")
        if result.timed_out:
            print("TIMED OUT")
        if result.runtime:
            print(f"runtime: {result.runtime} seconds")
        if result.metadata:
            print(f"metadata: {result.metadata}")

        print("--")
        print(stats.pretty_print())
        print()
        print_debug_times(results)
        print()

        out_f.write(json.dumps(result.serialize()) + "\n")
        out_f.flush()
    out_f.write(json.dumps(stats.serialize()) + "\n")

def run_lean_file(args: argparse.Namespace, lean_file: Path):
    subprocess.run(["lake", "env", "lean", str(lean_file)], cwd=args.project_path, check=True)


def run_eval(args: argparse.Namespace):
    utils.setup_seeds(args.seed)

    if args.serial:
        args.num_repls = 1
        args.max_concurrency = 1

    out_path = args.benchmark_result_dir / f"{args.experiment}.jsonl"
    if out_path.exists() and not args.force:
        print(f"Result file already exists: {out_path}")
        return
    args.benchmark_result_dir.mkdir(parents=True, exist_ok=True)
    print(f"Will save results to: {out_path}")

    if args.benchmark == "minif2f":
        adapter = InterLMMiniF2FAdapter(args.benchmark_cache_dir)
        benchmark = adapter.fetch_minif2f()
    else:
        raise Exception(f"Unknown benchmark: {args.benchmark}")

    model = None
    if args.method != "mock":
        print(f"Loading model ...")
        assert args.checkpoint, "Model checkpoint has to be specified."
        outputs_save_path = None
        if args.save_outputs:
            outputs_save_path = args.benchmark_result_dir / f"{args.experiment}.llm_outputs.txt"
            print(f"Will save LLM outputs to: {outputs_save_path}")
            if outputs_save_path.exists():
                outputs_save_path.unlink()
        model = LeanModel.build_multi_gpu_hf_model(
        # model = LeanModel.build_hf_model(
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_generation_length=args.max_generation_length,
            outputs_save_path=outputs_save_path,
        )

    def method_provider():
        if args.method == "mcts":
            return MCTSProofSearch(
                model=model,
                num_iters=args.num_iters,
                num_samples=args.num_samples,
                # inspector=TextSearchInspector(),
                use_tqdm=True,
            )
        elif args.method == "random":
            return RandomProofSearch(
                model=model,
                num_iters=args.num_iters,
                batch_size=args.batch_size,
                rng=np.random.default_rng(args.seed),
            )
        elif args.method == "mock":
            return MockProofSearch(duration=5)
        else:
            raise Exception(f"Unknown method: {args.method}")

    header = "\n".join(benchmark.global_context.imports + benchmark.global_context.open_namespaces)
    evaluator = ParallelLeanEvaluator(
        args.repl_exe,
        args.project_path,
        method_provider,
        header,
        args.num_repls,
        args.max_concurrency,
        utils.Logger(utils.LogLevel.INFO),
        # utils.Logger(utils.LogLevel.DEBUG),
    )

    with open(out_path, "w") as out_f:
        max_index = None if not args.count else (args.count if not args.offset else args.offset + args.count)
        theorems = benchmark.val_theorems[args.offset:max_index]

        loop = asyncio.get_event_loop()
        start_time = time.time()
        loop.run_until_complete(run_eval_async(args, theorems, evaluator, out_f))
        total_time = time.time() - start_time

    print(f"{total_time:.2f} seconds, {len(theorems)} theorems, {total_time / len(theorems):.2f} seconds/theorem")

    export_path = out_path.with_suffix(".lean")
    print(f"Exporting lean proofs to: {export_path}")
    export_lean_proofs(out_path, export_path, args.benchmark_cache_dir, force=True)

    print(f"Verifying using Lean: {export_path}")
    run_lean_file(args, export_path)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    methods = parser.add_subparsers(dest="method")
    mcts = methods.add_parser("mcts")
    random = methods.add_parser("random")
    mock = methods.add_parser("mock")

    for p in [mcts, random, mock]:
        p.add_argument("--benchmark", type=str, choices=["minif2f"], default="minif2f")
        p.add_argument("--experiment", type=str, default="minif2f",
                       help="Name of the experiment. Will be used for the output file name.")
        p.add_argument("--offset", type=int, default=None)
        p.add_argument("--count", type=int, default=None)
        p.add_argument("--benchmark_cache_dir", type=Path, default="benchmarks")
        p.add_argument("--benchmark_result_dir", type=Path, default="benchmark_results")
        p.add_argument("--force", action="store_true")
        # TODO: remove the defaults!
        p.add_argument("--repl_exe", type=Path, default="/home/m/repos/lean-repl-fork/.lake/build/bin/repl")
        p.add_argument("--project_path", type=Path, default="/home/m/arcoss/project_1/arcoss-lean-repo-v4-19-rc2")

        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--num_repls", type=int, default=16)
        p.add_argument("--max_concurrency", type=int, default=256)
        p.add_argument("--serial", action="store_true", help="Run in serial mode, with 1 REPL and no concurrency.")

        p.add_argument("--save_outputs", action="store_true")

    for p in [mcts, random]:
        # Model params
        p.add_argument("--checkpoint", type=str)
        p.add_argument("--batch_size", type=int, default=32)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--max_generation_length", type=int, default=256)  # TODO: tune

    mcts.add_argument("--num_iters", type=int, default=100)
    mcts.add_argument("--num_samples", type=int, default=32)

    random.add_argument("--num_iters", type=int, default=100)
    random.add_argument("--num_samples", type=int, default=8)

    return parser


def main():
    args = get_parser().parse_args()
    utils.resolve_paths(args)

    run_eval(args)


if __name__ == "__main__":
    main()

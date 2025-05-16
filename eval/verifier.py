import argparse
import json
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from eval.data import ProofSearchResult
from eval.interlm_adapter import InterLMMiniF2FAdapter
from lean_trees import utils
from lean_trees.core.abstraction import ProofGoal, ProofBranch, ProofStep
from lean_trees.repl_adapter.interaction import LeanServer
from search.base import Proof


# TODO: merge with ProofTreeVerifier
class ProofVerifier:
    def __init__(
            self,
            env_provider: Callable[[], LeanServer],
            env_setup: Callable[[LeanServer], None],
    ):
        self.env_provider = env_provider
        self.env_setup = env_setup

    def verify_proof(self, theorem_with_sorry: str, proof: Proof):
        with self.env_provider() as env:
            self.env_setup(env)
            init_state = env.start_proof_from_sorry(theorem_with_sorry)
            proof_states: dict[ProofGoal, ProofBranch] = {proof.goal: init_state}

            def visitor(goal: ProofGoal, step: ProofStep):
                assert goal in proof_states
                state = proof_states[goal]

                expected_nodes = step.children
                actual_states = state.apply_tactic(step.tactic)

                assert len(expected_nodes) == len(actual_states), \
                    f"Lean gave different number of children ({len(actual_states)}) than listed in the tree ({len(expected_nodes)})."
                proof_states.update(
                    {sub_node: sub_state for sub_node, sub_state in zip(expected_nodes, actual_states)}
                )

            proof.traverse_preorder(visitor)


def run_verify(args: argparse.Namespace):
    adapter = InterLMMiniF2FAdapter(args.benchmark_cache_dir)
    benchmark = adapter.fetch_minif2f()

    def env_provider():
        return LeanServer(args.repl_exe, args.project_path, utils.Logger(utils.LogLevel.DEBUG))

    def env_setup(env: LeanServer):
        env.send_command("\n".join(benchmark.global_context.imports))
        env.send_command("\n".join(benchmark.global_context.open_namespaces))

    verifier = ProofVerifier(env_provider, env_setup)
    max_index = None if not args.count else (args.count if not args.offset else args.offset + args.count)
    print(f"offset={args.offset}, max_index={max_index}")
    lines = [
        json.loads(line)
        for line in args.result_file.read_text().splitlines()[args.offset:max_index]
    ]
    for sample in lines:
        if "thm" not in sample:
            continue
        result = ProofSearchResult.deserialize(sample)
        if result.proof is None:
            print(f"No proof for: {sample["thm"]}")
            continue
        print(f"Verifying proof of: {sample["thm"]}")
        verifier.verify_proof(sample["thm"], result.proof)


def run_export(args: argparse.Namespace):
    export_lean_proofs(args.result_file, args.out_file, args.benchmark_cache_dir, args.force)

def export_lean_proofs(result_file: Path, out_file: Path, benchmark_cache_dir: Path, force: bool = False):
    if out_file.exists() and not force:
        print(f"Already exists: {out_file}. Terminating. To overwrite, use --force.")
        return

    adapter = InterLMMiniF2FAdapter(benchmark_cache_dir)
    benchmark = adapter.fetch_minif2f()

    with open(out_file, "w") as f:
        f.write("\n".join(benchmark.global_context.imports))
        f.write("\n\n")
        f.write("\n".join(benchmark.global_context.open_namespaces))
        f.write("\n\n")
        for line in tqdm(result_file.read_text().splitlines()):
            data = json.loads(line)
            if "theorem" not in data:
                continue
            result = ProofSearchResult.deserialize(data)
            if result.proof is None:
                f.write(result.theorem)
            else:
                linear_proof = result.proof.to_linear_proof()
                f.write(result.theorem.replace("sorry", "") + "\n")
                f.write("".join("  " + line for line in linear_proof.splitlines(keepends=True)))
            f.write("\n\n")

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest="action")

    verify = actions.add_parser("verify")
    # TODO: remove the defaults!
    verify.add_argument("result_file", type=Path)
    verify.add_argument("--repl_exe", type=Path, default="/home/m/repos/lean-repl-v4.15.0/.lake/build/bin/repl")
    verify.add_argument("--project_path", type=Path, default="/home/m/arcoss/project_1/arcoss-lean-repo")
    verify.add_argument("--offset", type=int, default=None)
    verify.add_argument("--count", type=int, default=None)
    verify.add_argument("--benchmark_cache_dir", type=Path, default="benchmarks")

    export = actions.add_parser("export")
    export.add_argument("result_file", type=Path)
    export.add_argument("out_file", type=Path)
    export.add_argument("--benchmark_cache_dir", type=Path, default="benchmarks")
    export.add_argument("--force", action="store_true")

    return parser


def main():
    args = get_parser().parse_args()
    utils.resolve_paths(args)

    if args.action == "verify":
        run_verify(args)
    elif args.action == "export":
        run_export(args)
    else:
        raise Exception()

if __name__ == "__main__":
    main()
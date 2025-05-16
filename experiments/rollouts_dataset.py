import argparse
import json
import random
from pathlib import Path
from typing import TextIO, Iterator

from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer

from lean_trees import utils
from lean_trees.core.lean_file import LeanFile, StoredError, LeanTheorem, LeanTacticBlock
from lean_trees.core.proof_tree import ProofTreeNode, ProofTree
from lean_trees.file_span import FileSpan
from model.lean.lean_model import LeanSerializer


def create_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("tree_dataset", type=Path)
    convert_parser.add_argument("--theorem_per_line", action="store_true")
    convert_parser.add_argument("--output_dir", type=Path, default="datasets/lean/text")
    convert_parser.add_argument("--tmp_file_extension", type=str, default=".tmp")
    convert_parser.add_argument("--force", action="store_true", help="Override the output file if it already exists.")
    convert_parser.add_argument("--whitebox", action="store_true", help="Use whitebox rollouts.")
    convert_parser.add_argument("--random_val_ratio", type=float, default=0.02)
    convert_parser.add_argument("--random_val_max_size", type=int, default=500)
    convert_parser.add_argument("--tokenizer", type=str, required=True)

    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("text_dataset", type=Path)

    return parser

ALLOWED_TACTICS = [
    "assumption",
    "exists",
    "intro",
    "intros",
    "rintro",
    "rfl",
    "symm",
    "subst",
    "subst_eqs",
    "subst_vars",
    "eq_refl",
    "ac_rfl",
    "ac_nf",
    "ac_nf0",
    "exact",
    "apply",
    "refine",
    "exfalso",
    "contradiction",
    "false_or_by_contra",
    "suffices",
    "change",
    "generalize",
    "specialize",
    "obtain",
    "show",
    "norm_cast",
    "push_cast",
    "exact_mod_cast",
    "apply_mod_cast",
    "rw_mod_cast",
    "assumption_mod_cast",
    "rw",
    "rewrite",
    "erw",
    "rwa",
    "unfold",
    "replace",
    "constructor",
    "injection",
    "injections",
    "left",
    "right",
    "cases",
    "rcases",
    "fun_cases",
    "induction",
    "fun_induction",
    "nofun",
    "nomatch",
    "split",
    "by_cases",
    "trivial",
]

class DeepSeekProverV1TheoremFinder:
    def __init__(self):
        dataset = load_dataset("deepseek-ai/DeepSeek-Prover-V1")
        
        # Create a dictionary mapping theorem names to formal statements
        print("Loading DeepSeekProverV1 dataset ...")
        self.theorem_dict = {}
        for split in dataset.keys():
            for item in dataset[split]:
                if "name" in item and "formal_statement" in item:
                    self.theorem_dict[item["name"]] = item["formal_statement"]
        print(f"Loaded {len(self.theorem_dict)} theorems from DeepSeekProverV1 dataset.")
    
    def find_theorem(self, name: str) -> str:
        return self.theorem_dict.get(name, "")

def create_rollouts_dataset(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading tree dataset {args.tree_dataset}")
    base_filename = str(args.tree_dataset.stem) + "-rollout_samples"
    if args.whitebox:
        base_filename += "-whitebox"
    train_file = args.output_dir / f"{base_filename}.train.jsonl"
    random_val_file = args.output_dir / f"{base_filename}.random_val.jsonl"
    if not args.force:
        if train_file.exists():
            print(f"Exiting because train file already exists: {train_file}")
            return
        if random_val_file.exists():
            print(f"Exiting because random_val file already exists: {random_val_file}")
            return
    print(f"Train set: {train_file}")
    print(f"Random validation set: {random_val_file}")
    train_tmp = str(train_file) + args.tmp_file_extension
    random_val_tmp = str(random_val_file) + args.tmp_file_extension
    args.output_dir.mkdir(parents=True, exist_ok=True)

    name_finder = DeepSeekProverV1TheoremFinder() if args.theorem_per_line else None
    print("Counting samples to enable train/validation split ...")
    nodes_count = sum(1 for _ in iterate_valid_rollout_nodes(args, name_finder))
    train_indices, random_val_indices = get_train_val_split(nodes_count, args.random_val_ratio, args.random_val_max_size)
    print("Converting samples ...")
    with (open(train_tmp, "w") as train_f, open(random_val_tmp, "w") as random_val_f):
        train_buff, random_val_buff = ShuffleBuffer(train_f), ShuffleBuffer(random_val_f)
        for i, (thm_statement, node) in enumerate(iterate_valid_rollout_nodes(args, name_finder)):
            assert (i in train_indices) != (i in random_val_indices)
            buffer = (train_buff if i in train_indices else random_val_buff)

            if thm_statement.startswith("theorem "):
                thm_statement = replace_theorem_with_example(thm_statement)
            elif not thm_statement.startswith("example "):
                continue

            proof_prefix = LeanSerializer.serialize_proof_prefix(node)
            inp = LeanSerializer.serialize_rollout_input_instruct(tokenizer, thm_statement, proof_prefix, node.state, args.whitebox)
            out = LeanSerializer.serialize_rollout_output_instruct(node)
            if len(inp) + len(out) > 1024:
                continue
            if "⋯" in inp:
                continue

            inp = hack_remove_dates(inp)
            buffer.add_sample(json.dumps({
                "input": inp,
                "output": out,
                "type": "rollout",
            }, ensure_ascii=False))
            if len(node.tactic.children) == 0 and not args.whitebox:
                proof_prefix_leaf = proof_prefix + [LeanSerializer.serialize_tactic_output(node)]
                inp_leaf = LeanSerializer.serialize_rollout_input_instruct(tokenizer, thm_statement, proof_prefix_leaf, None, args.whitebox)
                inp_leaf = hack_remove_dates(inp_leaf)
                out_leaf = "<tactic>done</tactic>"
                buffer.add_sample(json.dumps({
                    "input": inp_leaf,
                    "output": out_leaf,
                    "type": "rollout_leaf",
                }, ensure_ascii=False))
        train_buff.force_flush()
        random_val_buff.force_flush()

    print(f"Moving generated train file from {train_tmp} to {train_file}")
    Path(train_tmp).replace(train_file)
    print(f"Moving generated random validation file from {random_val_tmp} to {random_val_file}")
    Path(random_val_tmp).replace(random_val_file)

# For Llama-Instruct models.
def hack_remove_dates(inp: str) -> str:
    lines = inp.splitlines(keepends=True)
    lines = [
        l for l in lines
        if not (
            l.startswith("Cutting Knowledge Date:") or
            l.startswith("Today Date:")
        )
    ]
    return "".join(lines)

def replace_theorem_with_example(theorem: str) -> str:
    words = theorem.split()
    assert len(words) > 2
    return "example" + theorem[len(words[0] + " " + words[1]):]


class ShuffleBuffer:
    def __init__(self, output: TextIO, max_size: int = 4096, seed: int = 0):
        self.output = output
        self.max_size = max_size
        self._buffer = []
        self._rng = random.Random(seed)

    def add_sample(self, sample: str):
        self._buffer.append(sample)
        if len(self._buffer) > self.max_size:
            self.force_flush()
            self._buffer = []

    def force_flush(self):
        self._rng.shuffle(self._buffer)
        for sample in self._buffer:
            self.output.write(sample + "\n")


def get_train_val_split(samples_count: int, random_val_ratio: float, random_val_max_size: int) -> tuple[set[int], set[int]]:
    random_val_count = min(random_val_max_size, int(samples_count * random_val_ratio))
    val_indices = set(np.random.choice(samples_count, random_val_count, replace=False))
    train_indices = set(range(samples_count)) - val_indices
    return train_indices, val_indices

def get_thm_statement(args: argparse.Namespace, thm: LeanTheorem, name_finder: DeepSeekProverV1TheoremFinder | None) -> str:
    if args.theorem_per_line:
        return name_finder.find_theorem(thm.name).strip()
    else:
        proof_span = thm.by_blocks[0].span
        thm_statement_span = FileSpan(
            thm.span.start,
            proof_span.start,
        )
        thm_statement = thm_statement_span.read_from_file(thm.file.path).strip()
        thm_statement = utils.remove_empty_lines(utils.remove_comments(thm_statement))
        return thm_statement

def iterate_valid_rollout_nodes(args: argparse.Namespace, name_finder: DeepSeekProverV1TheoremFinder | None) -> Iterator[tuple[str, ProofTreeNode]]:
    if args.theorem_per_line:
        it = iterate_valid_nodes_theorems(args.tree_dataset)
    else:
        it = iterate_valid_nodes_files(args.tree_dataset)

    for thm, by_block in it:
        if len(thm.by_blocks) > 1:
            # We only want singleton by-blocks.
            continue
        proof_tree = thm.by_blocks[0].tree
        if not isinstance(proof_tree, ProofTree):
            continue
        thm_statement = get_thm_statement(args, thm, name_finder)
        if ":= by" not in thm_statement:
            # We only want pure tactic proofs.
            continue
        for node in by_block.tree.get_nodes():
            if node.tactic is None:
                continue
            tactic = node.tactic.tactic.tactic.strip()
            if len(tactic) > 100:
                # We don't want too long tactics.
                continue
            if not any(tactic.startswith(allowed) for allowed in ALLOWED_TACTICS):
                continue
            yield thm_statement, node


def iterate_valid_nodes_files(dataset_path: Path) -> Iterator[tuple[LeanTheorem, LeanTacticBlock]]:
    with open(dataset_path) as inp_f:
        for line in tqdm(inp_f):
            file = LeanFile.deserialize(json.loads(line))
            for thm in file.theorems:
                if isinstance(thm, StoredError):
                    continue
                for by_block in thm.by_blocks:
                    if isinstance(by_block.tree, StoredError):
                        continue
                    yield thm, by_block

def iterate_valid_nodes_theorems(dataset_path: Path):
    with open(dataset_path) as inp_f:
        for line in tqdm(inp_f):
            thm = LeanTheorem.deserialize(json.loads(line))
            if isinstance(thm, StoredError):
                continue
            for by_block in thm.by_blocks:
                if isinstance(by_block.tree, StoredError):
                    continue
                yield thm, by_block

def run_stats(args):
    input_lengths = []
    output_lengths = []
    with open(args.text_dataset) as inp_f:
        for line in tqdm(inp_f):
            data = json.loads(line)
            input_lengths.append(len(data["input"]))
            output_lengths.append(len(data["output"]))
    print(f"Average input length: {sum(input_lengths) / len(input_lengths)}")
    print(f"Average output length: {sum(output_lengths) / len(output_lengths)}")
    print(f"Max input length: {max(input_lengths)}")
    print(f"Max output length: {max(output_lengths)}")

def main(args):
    if args.action == "convert":
        create_rollouts_dataset(args)
    elif args.action == "stats":
        run_stats(args)
    else:
        raise ValueError(f"Unknown action: {args.action}")

if __name__ == "__main__":
    main(create_parser().parse_args())
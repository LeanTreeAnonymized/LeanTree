from typing import Self
import shutil
import subprocess
from pathlib import Path

from lean_trees import utils
from lean_trees.core.lean_file import LeanFile, LeanTheorem, LeanTacticBlock, StoredError
from lean_trees.data_extraction.tree_builder import ProofTreeBuilder
from lean_trees.data_extraction.tree_postprocessor import ProofTreePostprocessor
from lean_trees.repl_adapter.data import SingletonProofTree
from lean_trees.repl_adapter.interaction import LeanServer, LeanInteractionException
from lean_trees.repl_adapter.data_extraction import LeanFileParser
from lean_trees.repl_adapter.singleton_trees import SingletonTreeBuilder


# TODO: use (probably) logging.Logger
# TODO: repl_exe_path should be a settable attribute

class LeanProject:
    def __init__(
            self,
            project_path: Path | str,  # TODO: should be optional - create in cwd/arcoss-lean-project if not given
            create: bool = False,
    ):
        self.project_path = Path(project_path)
        if create and (not self.project_path.exists() or not any(self.project_path.iterdir())):
            self.create(self.project_path)

    def environment(
            self,
            repl_exe_path: Path | str | None = None,
            logger: utils.Logger | None = None,
    ) -> LeanServer:
        return LeanServer(self._get_repl_path(repl_exe_path), self.project_path, logger)

    def load_theorem(
            self,
            theorem: str,
            env: LeanServer,
    ) -> LeanTheorem:
        checkpoint = env.checkpoint()
        loaded_unit = env.send_theorem(theorem)
        env.rollback_to(checkpoint)

        loaded_unit.trees = SingletonTreeBuilder.build_singleton_trees(loaded_unit)
        for tree in loaded_unit.trees:
            ProofTreePostprocessor.transform_proof_tree(tree)
        result = ProofTreeBuilder.run_proof_trees(theorem, loaded_unit, env)
        env.rollback_to(checkpoint)
        return result

    def load_file(
            self,
            path: Path | str,
            repl_exe_path: Path | str | None = None,
            use_cache: bool = True,
            logger: utils.Logger | None = None,
            store_assertion_errors: bool = True,
    ) -> LeanFile:
        path = Path(path)
        assert path.is_file()
        assert str(path).endswith(".lean")

        loaded_file = LeanFileParser.load_lean_file(
            self._get_repl_path(repl_exe_path), self.project_path, path, use_cache,
        )
        for unit in loaded_file.units:
            if len(unit.proof_steps) == 0:
                continue

            try:
                unit.trees = SingletonTreeBuilder.build_singleton_trees(unit)
                for tree in unit.trees:
                    ProofTreePostprocessor.transform_proof_tree(tree)
            except (AssertionError, LeanInteractionException) as e:
                if store_assertion_errors:
                    unit.trees = e
                    continue
                else:
                    raise

        file = LeanFile(
            path=Path(loaded_file.path),
            imports=loaded_file.imports,
            theorems=[],
        )
        block_to_tree: dict[LeanTacticBlock, SingletonProofTree] = {}
        for unit in loaded_file.units:
            if unit.trees is None:
                continue
            if isinstance(unit.trees, Exception):
                file.theorems.append(StoredError.from_exception(unit.trees))
                continue
            by_blocks = []
            theorem = LeanTheorem(
                span=unit.span,
                file=file,
                by_blocks=by_blocks,
                context=unit.global_context,
            )
            file.theorems.append(theorem)
            for singleton_tree in unit.trees:
                by_block = LeanTacticBlock(
                    theorem=theorem,
                    tree=None,
                    span=singleton_tree.span,
                )
                by_blocks.append(by_block)
                block_to_tree[by_block] = singleton_tree

        failed_theorems = []
        with self.environment(repl_exe_path, logger) as env:
            for theorem, init_proof_states in env.start_file_proofs(file):
                if isinstance(init_proof_states, Exception):
                    failed_theorems.append((theorem, init_proof_states))
                    continue
                init_proof_states = list(init_proof_states)
                assert len(theorem.by_blocks) == len(init_proof_states)
                for by_block, init_proof_state in zip(theorem.by_blocks, init_proof_states):
                    singleton_tree = block_to_tree[by_block]
                    try:
                        tree = ProofTreeBuilder.run_proof_tree(singleton_tree, init_proof_state)
                    except (AssertionError, LeanInteractionException) as e:
                        if store_assertion_errors:
                            by_block.tree = StoredError.from_exception(e)
                            continue
                        else:
                            raise
                    by_block.tree = tree

        
        for i in range(len(file.theorems)):
            error = [err for t, err in failed_theorems if t == file.theorems[i]]
            if error:
                file.theorems[i] = StoredError.from_exception(error[0])
        return file

    # TODO: user should be able to choose which libraries get included
    @classmethod
    def create(
            cls,
            project_path: Path | str,
            lean_toolchain: str | None = None,
            suppress_output: bool = True,
    ) -> Self:
        project_path = Path(project_path)
        # Check that `lake` is in PATH.
        if shutil.which("lake") is None:
            raise RuntimeError(
                "Unable to find 'lake' in PATH. Please install Lean 4 and lake before creating a Lean project. See: "
                "https://leanprover-community.github.io/install/linux.html"
            )

        if project_path.exists():
            if any(project_path.iterdir()):
                raise FileExistsError(
                    f"Cannot create Lean project: directory exists and is not empty: {project_path}"
                )
        else:
            project_path.mkdir(parents=True)

        def run_command(args):
            """Helper to run a command with or without suppressed output."""
            if suppress_output:
                result = subprocess.run(args, cwd=project_path, text=True, capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Command {args} failed with code {result.returncode}\n"
                        f"stderr:\n{result.stderr}"
                    )
            else:
                result = subprocess.run(args, cwd=project_path)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Command {args} failed with code {result.returncode}"
                    )

        run_command(["lake", "init", ".", "math"])

        if lean_toolchain:
            toolchain_file = project_path / "lean-toolchain"
            toolchain_file.write_text(f"{lean_toolchain}\n")

        run_command(["lake", "build"])

        return cls(project_path)

    @classmethod
    def _get_repl_path(cls, repl_exe_path: Path | str | None = None) -> Path:
        if repl_exe_path is None:
            repl_exe_path = cls._get_default_repl_path()
        repl_exe_path = Path(repl_exe_path)
        if not repl_exe_path.exists():
            raise Exception(f"REPL executable does not exist: {repl_exe_path}")
        return repl_exe_path

    @classmethod
    def _get_default_repl_path(cls) -> Path:
        return Path(__file__).parent / "lean-repl"

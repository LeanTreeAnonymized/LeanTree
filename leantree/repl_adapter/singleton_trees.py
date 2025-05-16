from leantree.file_span import FileSpan
from leantree.repl_adapter.data import ReplCompilationUnit, SingletonProofTreeNode, SingletonProofTree, \
    SingletonProofTreeEdge, ReplProofStepInfo

class SingletonTreeBuilder:
    @classmethod
    def build_singleton_trees(cls, unit: ReplCompilationUnit) -> list[SingletonProofTree]:
        """Piece together loaded proof steps based on metavariable IDs."""

        def create_proof_tree(start_idx: int) -> tuple[SingletonProofTree, int]:
            root = SingletonProofTreeNode.from_goal(unit.proof_steps[start_idx].goal_before)
            all_goals: dict[str, SingletonProofTreeNode] = {root.id: root}
            i = start_idx
            while i < len(unit.proof_steps):
                step = unit.proof_steps[i]
                if step.goal_before.mvar_id not in all_goals:
                    # Root's ID always is in `all_goals`, so `i` will get incremented at least once. We assert it to be sure.
                    assert i > start_idx
                    break
                i += 1

                goal_before = all_goals[step.goal_before.mvar_id]
                assert goal_before.tactic is None, (
                    "Reusing closed goal!\n"
                    f"ID: {goal_before.id}\n"
                    f"Already assigned tactic: {goal_before.tactic.tactic_string if goal_before.tactic is not None else "None"}\n"
                    f"New tactic: {step.tactic_string}\n"
                )

                for goal in step.all_children():
                    if goal.mvar_id not in all_goals:
                        all_goals[goal.mvar_id] = SingletonProofTreeNode.from_goal(goal)
                tactic = SingletonProofTreeEdge.from_step_info(step, all_goals)
                assert not any(child.parent is not None for child in tactic.all_children()), (
                    "Reusing a child!\n"
                    f"ID: {goal_before.id}\n"
                    f"Tactic: {tactic.tactic_string}\n"
                    f"... with children:\n"
                    f"{"\n".join(f"{child.id} ---> {child.parent.id if child.parent else "None"}\n" for child in tactic.all_children())}"
                )
                goal_before.set_edge(tactic)
            tree = SingletonProofTree(
                root,
                span=FileSpan.get_containing_span([
                    n.tactic.span for n in root.get_subtree_nodes()
                    if n.tactic is not None and n.tactic.span is not None
                ])
            )
            return tree, i

        cls._check_if_unit_supported(unit)
        trees = []
        step_idx = 0
        while step_idx < len(unit.proof_steps):
            new_tree, step_idx = create_proof_tree(step_idx)
            trees.append(new_tree)
        return trees

    @classmethod
    def _check_if_unit_supported(cls, unit: ReplCompilationUnit):
        # `simp_rw` cannot be detected like the others because it is split by Lean and not visible (similarly to `rw`).
        if unit.pretty_print is not None and "simp_rw [" in unit.pretty_print:
            raise AssertionError(f"`simp_rw` tactic is not yet supported.")

        unsupported_tactics = [
            "calc",
            "conv",
        ]
        for tactic in unsupported_tactics:
            if any(edge.tactic_string.strip().startswith(tactic) for edge in unit.proof_steps):
                raise AssertionError(f"`{tactic}` tactic is not yet supported")

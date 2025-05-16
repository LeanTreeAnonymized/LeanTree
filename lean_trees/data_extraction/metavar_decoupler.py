from lean_trees.core.lean import LeanProofState, LeanGoal, LeanTactic
from lean_trees.data_extraction.proof_tree import ProofTree, ProofTreeNode, ProofTreeEdge
from lean_trees.repl_adapter.singleton_trees import SingletonProofTree, SingletonProofTreeNode


class TreeMetavarDecoupler:
    def merge_dependent_branches(self, src_tree: SingletonProofTree) -> ProofTree:
        def node_fn(src_node: SingletonProofTreeNode) -> list[SingletonProofTreeNode]:
            node = goal_to_node[src_node.goal]
            assert node.tactic is None

            branches = src_node.tactic.data.mctx_after.partition_independent_goals(
                [child.goal for child in src_node.tactic.children]
            )

            src_children = []
            children = []
            for branch_goals in branches:
                # Children in this SingletonProofTree branch, sorted by their appearance in the source file. We will
                # explore them in this order.
                branch_children = sorted(
                    [src_child for src_child in src_node.tactic.children if src_child.goal in branch_goals],
                    key=lambda child: child.tactic.data.span.start,
                )
                src_children.extend(branch_children)

                child = ProofTreeNode(
                    state=LeanProofState(branch_goals)
                )
                for goal in branch_goals:
                    goal_to_node[goal] = child
                children.append(child)

            tactic_info = src_node.tactic.data
            node.tactic = ProofTreeEdge(
                tactic=LeanTactic(tactic_info.tactic_string),
                span=tactic_info.span,
                parent=node,
                children=children,
                tactic_depends_on=tactic_info.tactic_depends_on,
            )
            # TODO: modify goal_to_node of all siblings of `node`

        goal_to_node: dict[LeanGoal, ProofTreeNode] = {
            src_tree.root.goal: ProofTreeNode(
                state=LeanProofState([src_tree.root.goal])
            )
        }
        src_tree.traverse(node_fn)
        return ProofTree(goal_to_node[src_tree.root.goal])

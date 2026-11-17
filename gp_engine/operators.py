"""
Genetic Operators for Genetic Programming:
- Subtree Crossover (Recombination)
- Subtree Mutation
- Point / Node Replacement Mutation
- Hoist Mutation (Bloat Reduction)
"""

from typing import List, Tuple, Optional
import numpy as np
from .node import Node
from .tree import Individual
from .population import TreeGenerator
from .functions import FUNCTION_REGISTRY


def get_all_nodes_with_parents(
    root: Node
) -> List[Tuple[Optional[Node], Optional[int], Node]]:
    """
    Returns list of tuples (parent, child_index, node) for every node in tree.
    For the root node itself, parent is None and child_index is None.
    """
    result = [(None, None, root)]
    queue = [(root, c_idx, child) for c_idx, child in enumerate(root.children)]

    while queue:
        parent, c_idx, node = queue.pop(0)
        result.append((parent, c_idx, node))
        for next_idx, next_child in enumerate(node.children):
            queue.append((node, next_idx, next_child))

    return result


class GeneticOperators:
    """
    Implements standard GP variation operators.
    """

    def __init__(self, rng: Optional[np.random.RandomState] = None):
        self.rng = rng if rng is not None else np.random.RandomState()

    def subtree_crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        max_depth: int = 8
    ) -> Tuple[Individual, Individual]:
        """
        Subtree Crossover:
        Selects a random crossover point in each parent, and swaps the subtrees.
        If depth limit is violated, the child falls back to cloning the parent.
        """
        child1 = parent1.clone()
        child2 = parent2.clone()

        nodes1 = get_all_nodes_with_parents(child1.root)
        nodes2 = get_all_nodes_with_parents(child2.root)

        p1_parent, p1_idx, p1_node = nodes1[self.rng.randint(len(nodes1))]
        p2_parent, p2_idx, p2_node = nodes2[self.rng.randint(len(nodes2))]

        # Deep clone fragments to swap
        sub1 = p1_node.clone()
        sub2 = p2_node.clone()

        # Place sub2 into child1
        if p1_parent is None:
            child1.root = sub2
        else:
            p1_parent.children[p1_idx] = sub2

        # Place sub1 into child2
        if p2_parent is None:
            child2.root = sub1
        else:
            p2_parent.children[p2_idx] = sub1

        child1.invalidate_cache()
        child2.invalidate_cache()

        # Depth control: fallback if exceeds maximum allowed depth
        if child1.depth > max_depth:
            child1 = parent1.clone()
        if child2.depth > max_depth:
            child2 = parent2.clone()

        return child1, child2

    def subtree_mutation(
        self,
        parent: Individual,
        tree_generator: TreeGenerator,
        max_mutation_depth: int = 3,
        max_depth: int = 8
    ) -> Individual:
        """
        Subtree Mutation:
        Selects a random node in parent and replaces it with a newly grown subtree.
        """
        child = parent.clone()
        nodes = get_all_nodes_with_parents(child.root)
        p_parent, p_idx, p_node = nodes[self.rng.randint(len(nodes))]

        # Grow random new subtree
        new_depth = int(self.rng.randint(1, max_mutation_depth + 1))
        new_subtree = tree_generator.build_tree(method='grow', max_depth=new_depth)

        if p_parent is None:
            child.root = new_subtree
        else:
            p_parent.children[p_idx] = new_subtree

        child.invalidate_cache()

        if child.depth > max_depth:
            return parent.clone()

        return child

    def point_mutation(
        self,
        parent: Individual,
        tree_generator: TreeGenerator,
        p_mutate_node: float = 0.15
    ) -> Individual:
        """
        Point Mutation:
        Visits each node and with probability p_mutate_node, replaces it with
        an equivalent node (function with same arity, or terminal).
        """
        child = parent.clone()
        nodes = get_all_nodes_with_parents(child.root)

        for _, _, node in nodes:
            if self.rng.rand() < p_mutate_node:
                if node.node_type == 'function':
                    target_arity = node.arity
                    # Find functions with matching arity
                    candidates = [
                        name for name, info in tree_generator.function_set.items()
                        if info.arity == target_arity
                    ]
                    if candidates:
                        node.value = str(self.rng.choice(candidates))
                elif node.node_type == 'variable':
                    # Mutate to another variable or constant
                    if tree_generator.n_features > 1 and self.rng.rand() > 0.3:
                        node.value = int(self.rng.randint(0, tree_generator.n_features))
                    else:
                        node.node_type = 'constant'
                        node.value = round(float(self.rng.uniform(
                            tree_generator.constant_range[0],
                            tree_generator.constant_range[1]
                        )), 3)
                elif node.node_type == 'constant':
                    # Gaussian perturbation of constant
                    jitter = float(self.rng.normal(0, 0.5))
                    node.value = round(float(node.value + jitter), 3)

        child.invalidate_cache()
        return child

    def hoist_mutation(
        self,
        parent: Individual
    ) -> Individual:
        """
        Hoist Mutation (Bloat Control):
        Selects a random non-root subtree and replaces the root with it,
        effectively shearing off outer levels and pruning complexity.
        """
        child = parent.clone()
        nodes = get_all_nodes_with_parents(child.root)
        if len(nodes) <= 1:
            return child

        # Pick non-root node
        non_roots = [n for p, idx, n in nodes if p is not None]
        if not non_roots:
            return child

        chosen_node = non_roots[self.rng.randint(len(non_roots))]
        child.root = chosen_node.clone()
        child.invalidate_cache()
        return child

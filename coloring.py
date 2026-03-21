from typing import Dict, List, Set, Tuple, Any
from csp import Constraint, CSP


class GraphColoringConstraint(Constraint[str, int]):
    """Constraint for graph coloring."""

    def __init__(self, node1: str, node2: str):
        super().__init__([node1, node2])
        self.node1 = node1
        self.node2 = node2

    def is_satisfied(self, assignment: Dict[str, int]) -> bool:
        """Check if adjacent nodes have different colors."""
        if self.node1 not in assignment or self.node2 not in assignment:
            return True
        return assignment[self.node1] != assignment[self.node2]

    def get_conflicted_variables(self, assignment: Dict[str, int]) -> Set[str]:
        """Return nodes that are in conflict."""
        if self.node1 in assignment and self.node2 in assignment:
            if assignment[self.node1] == assignment[self.node2]:
                return {self.node1, self.node2}
        return set()


class GraphColoringCSP(CSP[str, int]):
    """Graph coloring problem as a CSP."""

    def __init__(self, nodes: List[str], edges: List[tuple], num_colors: int):
        super().__init__()

        # Add variables (nodes) with domains (colors)
        for node in nodes:
            self.add_variable(node, list(range(num_colors)))

        # Add constraints for edges
        for node1, node2 in edges:
            self.add_constraint(GraphColoringConstraint(node1, node2))


if __name__ == "__main__":

    # Create and solve a hard-to-color Erdos-Renyi graph using CSP.

    import time, argparse
    from graph import UndirectedGraph_, erdos_renyi_, HAS_NX_MPL

    parser = argparse.ArgumentParser(
        description="Demo coloring an Erdos-Renyi graph G(n,p).",
        epilog="""
    Erdos-Renyi graphs, G(n,p), transition from having small, disconnected components to
    having a giant component of size O(n) when the average degree c = np is about 1, where n
    is the number of nodes and p is the uniform probability that nodes are connected.

    The 3-coloring threshold, where G(n,p) becomes 3-colorable is believe to be about
    c = 4.69, above which the graph become non-3-colorable with high probability as the
    space of valid 3-colorings splits into many separated clusters, making it hard for
    local search or other simple algorithms to find a valid solution - so that, at lower
    edge densities, below 4.96/n, finding a coloring can be difficult due to this clustering
    transition or 'freezing transistion'.
    """,
    )
    parser.add_argument("-n", help="number of vertices", type=int, default=80)
    parser.add_argument("-c", help="colorability threshold", type=float, default=4.25)
    parser.add_argument("-show", help="Toggle showing graphs", action="store_false")
    parser.add_argument("-fc", help="Use foward checking", action="store_false")
    parser.add_argument("-seed", help="set random seed", type=int, default=1)
    args = parser.parse_args()

    n = args.n
    p = args.c / n
    num_colors = 3

    print(f"Vertices: {n}, Edge probability: {p:.4f} (c={args.c})")

    graph = erdos_renyi_(UndirectedGraph_(), n=n, p=p, seed=args.seed)

    # Extract nodes and edges from the graph
    nodes = [v for v in graph.get_vertices()]
    edges = [(e[0], e[1]) for e in graph.get_edges()]

    print(f"Graph has {len(nodes)} nodes and {len(edges)} edges")
    if len(nodes) > 0:
        max_possible_edges = len(nodes) * (len(nodes) - 1) // 2
        print(f"Edge density: {len(edges) / max_possible_edges:.3f}")

    # Create and solve CSP
    print(f"Finding {num_colors}-coloring using CSP solver", end=" ")
    if args.fc:
        print("with forward checking.")
    else:
        print("without forward checking.")

    # Create CSP instance
    csp = GraphColoringCSP(nodes, edges, num_colors)

    # Solve with forward checking for better performance
    start_solve_time = time.time()
    coloring = csp.solve(use_forward_checking=args.fc)
    end_solve_time = time.time()

    if coloring:
        print(f"Solution found in {end_solve_time - start_solve_time:.3f} seconds")
        print(f"Backtrack count: {csp.get_backtrack_count()}")

        # Verify the coloring is valid using the graph structure
        conflicts = 0
        for node1_str, node2_str in edges:
            if coloring.get(node1_str) == coloring.get(node2_str):
                conflicts += 1

        if conflicts == 0:
            print("Valid coloring confirmed (no adjacent nodes share colors)")

            # Count color distribution
            color_counts: Dict[int, int] = {}
            for color in coloring.values():
                color_counts[color] = color_counts.get(color, 0) + 1

            print("Color distribution:")
            for color in sorted(color_counts.keys()):
                print(
                    f"  Color {color}: {color_counts[color]} nodes ({color_counts[color]/len(nodes)*100:.1f}%)"
                )

        else:
            print(f"Invalid coloring: {conflicts} conflicts found")

        if args.show and HAS_NX_MPL:
            import matplotlib.pyplot as plt
            import networkx as nx
            from graph import nx2ax, graph2nx

            fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

            nx2ax(graph2nx(graph), ax1, show_weights=False, coloring=coloring)
            ax1.set_title("Erdos-Renyi graph")
            ax1.axis("off")

            plt.tight_layout()
            plt.show()

    else:
        print(f"No solution found in {end_solve_time - start_solve_time:.3f} seconds")
        print(f"Backtrack count: {csp.get_backtrack_count()}")
        print("The graph may not be 3-colorable with these constraints.")

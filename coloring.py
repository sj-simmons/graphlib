from csp import CSP
from typing import Any, Dict, List, Set, Optional, Tuple
from graph import UndirectedGraph_

class GraphColoringCSP(CSP):
    """
    Specialized CSP for graph coloring problems.
    Extends the base CSP class with graph-specific functionality.
    """

    def __init__(self, graph: UndirectedGraph_, num_colors: int = 3):
        """
        Initialize for graph coloring with specified number of colors.

        Args:
            graph: An undirected graph
            num_colors: Number of colors to use (default: 3)
        """
        domain = list(range(num_colors))
        super().__init__(graph, domain)
        self.num_colors = num_colors

    def is_valid_coloring(self, coloring: Dict[Any, Any]) -> bool:
        """
        Check if a coloring is valid (no adjacent vertices have same color).

        Args:
            coloring: Dictionary mapping vertices to colors

        Returns:
            True if coloring is valid, False otherwise
        """
        for vertex in self.variables:
            if vertex not in coloring:
                return False
            for neighbor in self.constraints[vertex]:
                if coloring[vertex] == coloring[neighbor]:
                    return False
        return True

    def visualize_solution(self, solution: Dict[Any, Any], title: str = "Graph Coloring"):
        """
        Visualize the graph coloring solution if matplotlib is available.

        Args:
            solution: Coloring solution dictionary
            title: Title for the plot
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from graph import graph2nx, nx2ax, largenx2ax

            if not solution:
                print("No solution to visualize")
                return

            # Convert to networkx
            nx_graph = graph2nx(self.graph)

            # Create color map
            color_map = []
            for node in nx_graph.nodes():
                color_idx = solution.get(node, 0)
                # Use a colormap
                cmap = plt.cm.Set2 if len(self.graph) <= 50 else plt.cm.tab10
                color = cmap(color_idx / max(1, self.num_colors - 1))
                color_map.append(color)

            pos = nx.spring_layout(nx_graph)
            #pos = nx.planar_layout(nx_graph)

            # Draw graph
            fig, ax = plt.subplots(figsize=(10, 8))
            if len(self.graph) <= 50:
                node_size = nx2ax(nx_graph, ax, seed=42, show_weights=False, pos=pos)
            else:
                node_size = largenx2ax(nx_graph, ax, seed=42, tiny=False, pos=pos)

            # Override with colored nodes
            nx.draw_networkx_nodes(
                nx_graph, pos, ax=ax,
                node_color=color_map,
                node_size=node_size,
                edgecolors='black'
            )

            ax.set_title(f"{title}\n{self.num_colors}-coloring")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib or NetworkX not available for visualization")
            print("Solution:", solution)

if __name__ == "__main__":

    # Demonstrate the CSP solver on graph coloring problems.

    from graph import planar_

    print("Heuristic Comparison on Planar Graphs")

    graph = planar_(UndirectedGraph_(), n=200, remove_probability=0.0, seed=1)
    #graph = planar_(UndirectedGraph_(), n=250, remove_probability=0.01, seed=1)
    print(f"Coloring a graph of size {len(graph)}")

    heuristic_configs = [
        ("No heuristics", False, False, False, False),
        ("MRV only", True, False, False, True),
        ("MRV + Degree", True, True, False, True),
        ("MRV + LCV", True, False, True, True),
        ("All heuristics", True, True, True, True),
    ]

    last_solution = None
    for name, mrv, degree, lcv, fc in heuristic_configs:
        csp = GraphColoringCSP(graph, num_colors=3)
        solution, stats = csp.solve(use_mrv=mrv, use_degree=degree, use_lcv=lcv,
                                   use_forward_checking=fc)
        if solution:
            # Double check that this a valid coloring
            assert csp.is_valid_coloring(solution)
            found = "Yes"
            last_solution = solution
        else:
            found = "No"
        print(f"   {name:20} | Solution: {found:3} | "
              f"Assignments: {stats['assignments']:3} | "
              f"Backtracks: {stats['backtracks']:2}")

    if last_solution:
        csp.visualize_solution(last_solution, "Planar Graph Coloring")

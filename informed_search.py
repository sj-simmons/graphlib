from collections import deque
import heapq
import graph
from typing import Any, List, Optional, Tuple, Union


class UndirectedGraph(graph.UndirectedGraph_):
    def __init__(self) -> None:
        super().__init__()

    def greedy(self, start_vertex, goal_vertex, heuristic):
        """
        Find a path from start_vertex to goal_vertex using greedy best-first search.
        Always expands the node that appears closest to the goal according to the heuristic.

        Args:
            start_vertex: Starting vertex for the path
            goal_vertex: Goal vertex to reach
            heuristic: A dictionary mapping vertex names to estimated distances to the goal

        Returns:
            tuple: (path, total_weight) where path is a list of vertices from start to goal,
                   and total_weight is the sum of edge weights along the found path.
                   Returns (None, 0) if no path exists.
        """
        if not self.has_vertex(start_vertex) or not self.has_vertex(goal_vertex):
            return None, 0

        # Priority queue: (heuristic_cost, vertex, path, total_weight)
        # We use heuristic estimate as the primary key for heapq
        frontier = [
            (heuristic.get(start_vertex, float("inf")), start_vertex, [start_vertex], 0)
        ]
        visited = set()

        while frontier:
            _, current_vertex, path, total_weight = heapq.heappop(frontier)

            # Skip if we've already visited this vertex
            if current_vertex in visited:
                continue

            # Mark as visited
            visited.add(current_vertex)

            # Check if we found the goal
            if current_vertex == goal_vertex:
                return path, total_weight

            # Explore neighbors
            for neighbor, edge_weight in self.graph[current_vertex].items():
                if neighbor not in visited:
                    # Use heuristic value for the neighbor to prioritize
                    # If heuristic doesn't have the neighbor, use infinity
                    h = heuristic.get(neighbor, float("inf"))
                    heapq.heappush(
                        frontier,
                        (h, neighbor, path + [neighbor], total_weight + edge_weight),
                    )

        return None, 0


if __name__ == "__main__":

    from graph import twenty_, graph2nx, nx2ax, HAS_NX_MPL

    graph = twenty_(UndirectedGraph())

    print(graph)

    # Define start and goal vertices
    start_vertex = "N0"
    goal_vertex = "N19"

    # Create a simple heuristic (straight-line distance approximation)
    # For the 20-node graph, we'll use a simple heuristic based on node indices
    heuristic = {}
    for i in range(20):
        node = f"N{i}"
        # Simple heuristic: distance based on node number difference
        # This is just for demonstration purposes
        heuristic[node] = abs(i - 19) * 5

    # Run greedy search
    greedy_path, greedy_weight = graph.greedy(start_vertex, goal_vertex, heuristic)

    # Run uninformed searches for comparison
    from uninformed_search import UndirectedGraph as UninformedGraph
    uninformed_graph = twenty_(UninformedGraph())
    dfs_path, dfs_weight = uninformed_graph.dfs(start_vertex, goal_vertex)
    bfs_path, bfs_weight = uninformed_graph.bfs(start_vertex, goal_vertex)
    ucs_path, ucs_weight = uninformed_graph.ucs(start_vertex, goal_vertex)

    # Print a summary
    print("\n" + "=" * 50)
    print("Search Algorithm Comparison:")
    print("=" * 50)
    print(f"Start: {start_vertex}, Goal: {goal_vertex}")
    print()

    algorithms = [
        ("DFS", dfs_path, dfs_weight),
        ("BFS", bfs_path, bfs_weight),
        ("UCS", ucs_path, ucs_weight),
        ("Greedy", greedy_path, greedy_weight),
    ]

    for algo_name, path, total_weight in algorithms:
        if path:
            print(f"{algo_name}: {len(path)-1} steps, total weight {total_weight:.1f}")
        else:
            print(f"{algo_name}: No path found")

    if HAS_NX_MPL:

        import matplotlib.pyplot as plt
        import networkx as nx

        # Create a figure to display all search paths
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Convert the graph to networkx for visualization using graph.py's function
        nx_graph = graph2nx(graph)

        # Use a consistent layout for all subplots
        pos = nx.spring_layout(nx_graph, seed=42)

        # Flatten axes for easier iteration
        flat_axes = axes.flatten()

        all_algorithms = [
            ("DFS", dfs_path, dfs_weight),
            ("BFS", bfs_path, bfs_weight),
            ("UCS", ucs_path, ucs_weight),
            ("Greedy", greedy_path, greedy_weight),
        ]

        for i, (algo_name, path, total_weight) in enumerate(all_algorithms):
            ax = flat_axes[i]

            # Draw the base graph using nx2ax from graph.py
            nx2ax(nx_graph, ax, seed=42, show_weights=True)

            # Highlight the path if found
            if path:
                # Highlight path edges
                path_edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(
                    nx_graph,
                    pos,
                    ax=ax,
                    edgelist=path_edges,
                    edge_color="steelblue",
                    width=3,
                )
                # Highlight path nodes
                nx.draw_networkx_nodes(
                    nx_graph,
                    pos,
                    ax=ax,
                    nodelist=path,
                    node_color="lightsteelblue",
                    node_size=600,
                    edgecolors="black",
                )
                # Highlight start and goal nodes
                nx.draw_networkx_nodes(
                    nx_graph,
                    pos,
                    ax=ax,
                    nodelist=[start_vertex, goal_vertex],
                    node_color="lightskyblue",
                    node_size=700,
                    edgecolors="black",
                )

                ax.set_title(f"{algo_name} Path\nTotal Weight: {total_weight:.1f}")

                # Print detailed information to console
                print(f"\n{algo_name} Details:")
                print(f"  Path: {path}")
                print(f"  Total weight: {total_weight:.1f}")
                print(f"  Number of steps: {len(path)-1}")
            else:
                ax.set_title(f"{algo_name}: No Path Found")

            ax.set_axis_on()
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        plt.suptitle(
            f"Search Algorithms from {start_vertex} to {goal_vertex}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

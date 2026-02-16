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

    def astar(self, start_vertex, goal_vertex, heuristic):
        """
        Find a path from start_vertex to goal_vertex using A* search algorithm.
        Expands nodes based on f(n) = g(n) + h(n), where:
        - g(n) is the actual cost from start to current node
        - h(n) is the heuristic estimate from current node to goal

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

        # Priority queue: (f_score, g_score, vertex, path)
        # f_score = g_score + heuristic
        g_score = {start_vertex: 0}
        f_score = {start_vertex: heuristic.get(start_vertex, float("inf"))}

        frontier = [(f_score[start_vertex], g_score[start_vertex], start_vertex, [start_vertex])]
        visited = set()

        while frontier:
            _, current_g, current_vertex, path = heapq.heappop(frontier)

            # Skip if we've already visited this vertex with a better g_score
            if current_vertex in visited:
                continue

            # Mark as visited
            visited.add(current_vertex)

            # Check if we found the goal
            if current_vertex == goal_vertex:
                return path, current_g

            # Explore neighbors
            for neighbor, edge_weight in self.graph[current_vertex].items():
                if neighbor not in visited:
                    # Calculate tentative g_score for neighbor
                    tentative_g = current_g + edge_weight

                    # If this path to neighbor is better than any previous one
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic.get(neighbor, float("inf"))
                        heapq.heappush(
                            frontier,
                            (f_score[neighbor], tentative_g, neighbor, path + [neighbor])
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
    # Run A* search
    astar_path, astar_weight = graph.astar(start_vertex, goal_vertex, heuristic)

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
        ("A*", astar_path, astar_weight),
    ]

    for algo_name, path, total_weight in algorithms:
        if path:
            print(f"{algo_name}: {len(path)-1} steps, total weight {total_weight:.1f}")
        else:
            print(f"{algo_name}: No path found")

    if HAS_NX_MPL:

        import matplotlib.pyplot as plt
        import networkx as nx

        # Convert the graph to networkx for visualization using graph.py's function
        nx_graph = graph2nx(graph)

        # Use a consistent layout for all subplots
        pos = nx.spring_layout(nx_graph, seed=42)

        # Create a figure to display all search paths (2x3 grid for 5 algorithms)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Flatten axes for easier iteration
        flat_axes = axes.flatten()

        # Hide the last subplot (6th) since we only have 5 algorithms
        flat_axes[-1].axis('off')

        all_algorithms = [
            ("DFS", dfs_path, dfs_weight),
            ("BFS", bfs_path, bfs_weight),
            ("UCS", ucs_path, ucs_weight),
            ("Greedy", greedy_path, greedy_weight),
            ("A*", astar_path, astar_weight),
        ]

        for i, (algo_name, path, total_weight) in enumerate(all_algorithms):
            ax = flat_axes[i]

            # Draw the base graph using nx2ax from graph.py with the precomputed layout
            nx2ax(nx_graph, ax, seed=42, show_weights=True, pos=pos)

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

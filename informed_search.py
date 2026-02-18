from collections import deque
import heapq
import uninformed_search
from typing import Any, List, Optional, Tuple, Union


class UndirectedGraph(uninformed_search.UndirectedGraph):
    def __init__(self) -> None:
        super().__init__()

    def greedy(self, start_vertex, goal_vertex, heuristic):
        """
        Find a path using greedy best-first search.

        The algorithm always expands the node that appears closest to the goal
        according to the provided heuristic, without considering the cost
        incurred so far.

        Args:
            start_vertex: Starting vertex for the path.
            goal_vertex: Goal vertex to reach.
            heuristic: Dictionary mapping each vertex to an estimated distance
                       to the goal.

        Returns:
            A tuple (path, total_weight) where:
            - path: List of vertices from start to goal, or None if no path exists.
            - total_weight: Sum of edge weights along the found path, or 0 if no path.
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
        Find a path using the A* search algorithm.

        Nodes are expanded based on f(n) = g(n) + h(n), where:
        - g(n) is the actual cost from the start vertex to node n.
        - h(n) is the heuristic estimate from node n to the goal.

        Args:
            start_vertex: Starting vertex for the path.
            goal_vertex: Goal vertex to reach.
            heuristic: Dictionary mapping each vertex to an estimated distance
                       to the goal.

        Returns:
            A tuple (path, total_weight) where:
            - path: List of vertices from start to goal, or None if no path exists.
            - total_weight: Sum of edge weights along the found path, or 0 if no path.
        """
        if not self.has_vertex(start_vertex) or not self.has_vertex(goal_vertex):
            return None, 0

        # Priority queue: (f_score, g_score, vertex, path)
        # f_score = g_score + heuristic
        g_score = {start_vertex: 0}
        f_score = {start_vertex: heuristic.get(start_vertex, float("inf"))}

        frontier = [
            (f_score[start_vertex], g_score[start_vertex], start_vertex, [start_vertex])
        ]
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
                        f_score[neighbor] = tentative_g + heuristic.get(
                            neighbor, float("inf")
                        )
                        heapq.heappush(
                            frontier,
                            (
                                f_score[neighbor],
                                tentative_g,
                                neighbor,
                                path + [neighbor],
                            ),
                        )

        return None, 0

    def astar2(self, start_vertex, goal_vertex, heuristic):
        """
        Find a path using A* search while collecting efficiency metrics.

        This version is identical to `astar` but also tracks and returns
        statistics about the search process.

        Args:
            start_vertex: Starting vertex for the path.
            goal_vertex: Goal vertex to reach.
            heuristic: Dictionary mapping each vertex to an estimated distance
                       to the goal.

        Returns:
            A tuple (path, total_weight, efficiency_metrics) where:
            - path: List of vertices from start to goal, or None if no path exists.
            - total_weight: Sum of edge weights along the found path, or 0 if no path.
            - efficiency_metrics: Dictionary with keys:
                * 'nodes_expanded': number of nodes popped from the frontier.
                * 'nodes_visited': number of unique nodes visited.
                * 'frontier_max_size': maximum size of the frontier during search.
                * 'path_length': number of vertices in the found path (0 if none).
        """
        if not self.has_vertex(start_vertex) or not self.has_vertex(goal_vertex):
            # Initialize metrics even when no path exists
            efficiency_metrics = {
                "nodes_expanded": 0,
                "nodes_visited": 0,
                "frontier_max_size": 0,
                "path_length": 0,
            }
            return None, 0, efficiency_metrics

        # Initialize efficiency measurement variables
        nodes_expanded = 0
        nodes_visited = 0
        frontier_max_size = 0

        # Priority queue: (f_score, g_score, vertex, path)
        # f_score = g_score + heuristic
        g_score = {start_vertex: 0}
        f_score = {start_vertex: heuristic.get(start_vertex, float("inf"))}

        frontier = [
            (f_score[start_vertex], g_score[start_vertex], start_vertex, [start_vertex])
        ]
        visited = set()

        while frontier:
            # Update frontier max size
            frontier_max_size = max(frontier_max_size, len(frontier))

            _, current_g, current_vertex, path = heapq.heappop(frontier)
            nodes_expanded += 1

            # Skip if we've already visited this vertex with a better g_score
            if current_vertex in visited:
                continue

            # Mark as visited
            visited.add(current_vertex)
            nodes_visited += 1

            # Check if we found the goal
            if current_vertex == goal_vertex:
                # Calculate efficiency metrics
                efficiency_metrics = {
                    "nodes_expanded": nodes_expanded,
                    "nodes_visited": nodes_visited,
                    "frontier_max_size": frontier_max_size,
                    "path_length": len(path),
                }
                return path, current_g, efficiency_metrics

            # Explore neighbors
            for neighbor, edge_weight in self.graph[current_vertex].items():
                if neighbor not in visited:
                    # Calculate tentative g_score for neighbor
                    tentative_g = current_g + edge_weight

                    # If this path to neighbor is better than any previous one
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic.get(
                            neighbor, float("inf")
                        )
                        heapq.heappush(
                            frontier,
                            (
                                f_score[neighbor],
                                tentative_g,
                                neighbor,
                                path + [neighbor],
                            ),
                        )

        # No path found - return metrics
        efficiency_metrics = {
            "nodes_expanded": nodes_expanded,
            "nodes_visited": nodes_visited,
            "frontier_max_size": frontier_max_size,
            "path_length": 0,
        }
        return None, 0, efficiency_metrics


if __name__ == "__main__":

    from graph import graph2nx, nx2ax, HAS_NX_MPL

    from graph import watts_strogatz_

    n, k = 28, 6
    graph = watts_strogatz_(UndirectedGraph(), n=n, k=k, weight_range=(0, 20))

    # from graph import barabasi_albert_
    # n, m = 50,3
    # graph = barabasi_albert_(UndirectedGraph(), n=n, m=m)

    start_vertex = 0
    goal_vertex = (n - 1) // 2

    # print(graph)

    # Create a simple heuristic (straight-line distance approximation)
    # For the n-node graph, we'll use a simple heuristic based on node indices
    heuristic = {}
    for node in range(n):
        # Simple heuristic: distance based on node number difference
        # This is just for demonstration purposes
        heuristic[node] = abs(node - goal_vertex) * 5

    # Run greedy search
    greedy_path, greedy_weight = graph.greedy(start_vertex, goal_vertex, heuristic)
    # Run A* search with efficiency measurement (with heuristic)
    astar2_path, astar2_weight, astar2_metrics = graph.astar2(
        start_vertex, goal_vertex, heuristic
    )
    # Run A* search with zero heuristic (equivalent to UCS)
    zero_heuristic = {node: 0 for node in range(n)}
    astar2_zero_path, astar2_zero_weight, astar2_zero_metrics = graph.astar2(
        start_vertex, goal_vertex, zero_heuristic
    )

    # Run uninformed searches for comparison
    dfs_path, dfs_weight = graph.dfs(start_vertex, goal_vertex)
    bfs_path, bfs_weight = graph.bfs(start_vertex, goal_vertex)

    # Print a summary
    print("\n" + "=" * 50)
    print("Search Algorithm Comparison:")
    print("=" * 50)
    print(f"Start: {start_vertex}, Goal: {goal_vertex}")
    print()

    algorithms = [
        ("DFS", dfs_path, dfs_weight),
        ("BFS", bfs_path, bfs_weight),
        ("A* (zero heuristic)", astar2_zero_path, astar2_zero_weight),
        ("Greedy", greedy_path, greedy_weight),
        ("A* (non-zero heuristic)", astar2_path, astar2_weight),
    ]

    for algo_name, path, total_weight in algorithms:
        if path:
            print(f"{algo_name}: {len(path)-1} steps, total weight {total_weight:.1f}")
        else:
            print(f"{algo_name}: No path found")

    # Print efficiency metrics comparison for astar2 with different heuristics
    print("\n" + "=" * 50)
    print("A*2 Efficiency Metrics Comparison:")
    print("=" * 50)

    print("\n1. With Zero Heuristic (equivalent to UCS):")
    print("-" * 40)
    if astar2_zero_path:
        print(f"Path found: {astar2_zero_path}")
        print(f"Total weight: {astar2_zero_weight:.1f}")
    else:
        print("No path found")
    print(f"Nodes expanded: {astar2_zero_metrics['nodes_expanded']}")
    print(f"Nodes visited: {astar2_zero_metrics['nodes_visited']}")
    print(f"Maximum frontier size: {astar2_zero_metrics['frontier_max_size']}")
    print(f"Path length: {astar2_zero_metrics['path_length']}")

    print("\n2. With Non-zero Heuristic:")
    print("-" * 40)
    if astar2_path:
        print(f"Path found: {astar2_path}")
        print(f"Total weight: {astar2_weight:.1f}")
    else:
        print("No path found")
    print(f"Nodes expanded: {astar2_metrics['nodes_expanded']}")
    print(f"Nodes visited: {astar2_metrics['nodes_visited']}")
    print(f"Maximum frontier size: {astar2_metrics['frontier_max_size']}")
    print(f"Path length: {astar2_metrics['path_length']}")

    print("\n3. Comparison Summary:")
    print("-" * 40)
    print(
        f"Nodes expanded difference: {astar2_zero_metrics['nodes_expanded'] - astar2_metrics['nodes_expanded']} "
        + f"(zero: {astar2_zero_metrics['nodes_expanded']}, heuristic: {astar2_metrics['nodes_expanded']})"
    )
    print(
        f"Nodes visited difference: {astar2_zero_metrics['nodes_visited'] - astar2_metrics['nodes_visited']} "
        + f"(zero: {astar2_zero_metrics['nodes_visited']}, heuristic: {astar2_metrics['nodes_visited']})"
    )
    print(
        f"Frontier max size difference: {astar2_zero_metrics['frontier_max_size'] - astar2_metrics['frontier_max_size']} "
        + f"(zero: {astar2_zero_metrics['frontier_max_size']}, heuristic: {astar2_metrics['frontier_max_size']})"
    )

    # Check if both found paths and compare weights
    if astar2_zero_path and astar2_path:
        print(
            f"Path weight difference: {astar2_zero_weight - astar2_weight:.1f} "
            + f"(zero: {astar2_zero_weight:.1f}, heuristic: {astar2_weight:.1f})"
        )
        print(
            f"Path length difference: {astar2_zero_metrics['path_length'] - astar2_metrics['path_length']} "
            + f"(zero: {astar2_zero_metrics['path_length']}, heuristic: {astar2_metrics['path_length']})"
        )
    else:
        print("Note: One or both algorithms did not find a path")

    if HAS_NX_MPL:

        import matplotlib.pyplot as plt
        import networkx as nx

        # Convert the graph to networkx for visualization using graph.py's function
        nx_graph = graph2nx(graph)

        # Use a consistent layout for all subplots
        pos = nx.spring_layout(nx_graph, seed=42)

        # Create a figure to display selected search paths (1x3 grid for 3 algorithms)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Select only the three algorithms we want to visualize
        selected_algorithms = [
            ("Greedy", greedy_path, greedy_weight),
            ("A* (zero heuristic)", astar2_zero_path, astar2_zero_weight),
            ("A* (non-zero heuristic)", astar2_path, astar2_weight),
        ]

        for i, (algo_name, path, total_weight) in enumerate(selected_algorithms):
            ax = axes[i]

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
                    width=4,
                    alpha=0.5,
                )
                # Highlight path nodes
                nx.draw_networkx_nodes(
                    nx_graph,
                    pos,
                    ax=ax,
                    nodelist=path,
                    node_color="lightskyblue",
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
            f"Selected Search Algorithms from {start_vertex} to {goal_vertex}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

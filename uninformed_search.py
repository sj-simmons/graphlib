from collections import deque
import heapq
import graph
from typing import Any, List, Optional, Tuple, Union


class UndirectedGraph(graph.UndirectedGraph_):
    def __init__(self) -> None:
        super().__init__()

    def dfs(
        self, start_vertex: Any, goal_vertex: Any
    ) -> Tuple[Optional[List[Any]], Union[int, float]]:
        """
        Find a path from start_vertex to goal_vertex using DFS (iterative)
        and calculate the total weight.

        Args:
            start_vertex: Starting vertex for the path
            goal_vertex: Goal vertex to reach

        Returns:
            tuple: (path, total_weight) where path is a list of vertices from start to goal,
                   and total_weight is the sum of edge weights along the path.
                   Returns (None, 0) if no path exists.
        """
        if not self.has_vertex(start_vertex) or not self.has_vertex(goal_vertex):
            return None, 0

        stack = [(start_vertex, [start_vertex], 0)]  # (vertex, path, total_weight)
        visited = set([start_vertex])

        while stack:
            current_vertex, path, total_weight = stack.pop()

            # Check if we found the goal
            if current_vertex == goal_vertex:
                return path, total_weight

            # Explore neighbors
            for neighbor, weight in self.graph[current_vertex].items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor], total_weight + weight))

        return None, 0

    def bfs(
        self, start_vertex: Any, goal_vertex: Any
    ) -> Tuple[Optional[List[Any]], Union[int, float]]:
        """
        Find the shortest path (by number of edges) from start_vertex to goal_vertex
        using BFS and calculate the total weight.

        Args:
            start_vertex: Starting vertex for the path
            goal_vertex: Goal vertex to reach

        Returns:
            tuple: (path, total_weight) where path is a list of vertices from start to goal,
                   and total_weight is the sum of edge weights along the shortest path.
                   Returns (None, 0) if no path exists.
        """
        if not self.has_vertex(start_vertex) or not self.has_vertex(goal_vertex):
            return None, 0

        # Queue will store (vertex, path, total_weight)
        queue = deque()
        queue.append((start_vertex, [start_vertex], 0))
        visited = set([start_vertex])

        while queue:
            current_vertex, path, total_weight = queue.popleft()

            # Check if we found the goal
            if current_vertex == goal_vertex:
                return path, total_weight

            # Explore neighbors
            for neighbor, weight in self.graph[current_vertex].items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], total_weight + weight))

        return None, 0

    def ucs_(
        self, start_vertex: Any, goal_vertex: Any
    ) -> Tuple[Optional[List[Any]], Union[int, float]]:
        """
        Find the path with minimum total weight from start_vertex to goal_vertex
        using Uniform Cost Search (Dijkstra's algorithm for non-negative weights).

        Args:
            start_vertex: Starting vertex for the path
            goal_vertex: Goal vertex to reach

        Returns:
            tuple: (path, total_weight) where path is a list of vertices from start to goal,
                   and total_weight is the minimum sum of edge weights along path.
                   Returns (None, 0) if no path exists.
        """
        if not self.has_vertex(start_vertex) or not self.has_vertex(goal_vertex):
            return None, 0

        # Priority queue: (total_cost, vertex, path)
        # We use total_cost as the primary key for heapq
        frontier = [(0, start_vertex, [start_vertex])]
        visited = {start_vertex: 0}

        while frontier:
            current_cost, current_vertex, path = heapq.heappop(frontier)

            # Check if we found the goal
            if current_vertex == goal_vertex:
                return path, current_cost

            # Explore neighbors
            for neighbor, edge_weight in self.graph[current_vertex].items():
                new_cost = current_cost + edge_weight

                # If we haven't seen this neighbor, or we found a cheaper path to it
                if neighbor not in visited.keys() or new_cost < visited[neighbor]:
                    visited[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))

        return None, 0

    def ucs(
        self, start_vertex: Any, goal_vertex: Any
    ) -> Tuple[Optional[List[Any]], Union[int, float]]:
        """
        Find the path with minimum total weight from start_vertex to goal_vertex
        using Uniform Cost Search (Dijkstra's algorithm for non-negative weights).

        This is an optimized version of the usc_ method above.

        Args:
            start_vertex: Starting vertex for the path
            goal_vertex: Goal vertex to reach

        Returns:
            tuple: (path, total_weight) where path is a list of vertices from start to goal,
                   and total_weight is the minimum sum of edge weights along path.
                   Returns (None, 0) if no path exists.
        """
        if not self.has_vertex(start_vertex) or not self.has_vertex(goal_vertex):
            return None, 0

        # Priority queue: (total_cost, vertex, path)
        # We use total_cost as the primary key for heapq
        frontier = [(0, start_vertex, [start_vertex])]

        # Track visited nodes and their best known cost
        visited = set()
        cost_so_far = {start_vertex: 0}

        while frontier:
            current_cost, current_vertex, path = heapq.heappop(frontier)

            # If we've already found a better path to this node, skip it
            if current_vertex in visited and current_cost > cost_so_far[current_vertex]:
                continue

            # Check if we found the goal
            if current_vertex == goal_vertex:
                return path, current_cost

            # Mark as visited with this cost
            visited.add(current_vertex)

            # Explore neighbors
            for neighbor, edge_weight in self.graph[current_vertex].items():
                new_cost = current_cost + edge_weight

                # If we haven't seen this neighbor, or we found a cheaper path to it
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))

        return None, 0


if __name__ == "__main__":

    from graph import twenty_

    graph = twenty_(UndirectedGraph())

    print(graph)

    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        # Create a figure to display BFS, DFS, and UCS paths
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Convert the graph to networkx for visualization
        nx_graph = graph.to_networkx()

        # Use a consistent layout for all subplots
        pos = nx.spring_layout(nx_graph, seed=42)

        # Define start and goal vertices
        start_vertex = "N0"
        goal_vertex = "N19"

        # Run each search algorithm
        algorithms = [
            ("DFS", graph.dfs(start_vertex, goal_vertex)),
            ("BFS", graph.bfs(start_vertex, goal_vertex)),
            ("UCS", graph.ucs(start_vertex, goal_vertex)),
        ]

        for i, (algo_name, (path, total_weight)) in enumerate(algorithms):
            ax = axes[i]

            # Draw the base graph
            nx.draw(
                nx_graph,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightgray",
                node_size=500,
                font_size=8,
                font_weight="bold",
                edge_color="gray",
                width=1,
            )

            # Highlight the path if found
            if path:
                # Highlight path edges
                path_edges = list(zip(path[:-1], path[1:]))
                nx.draw_networkx_edges(
                    nx_graph, pos, ax=ax, edgelist=path_edges, edge_color="red", width=3
                )
                # Highlight path nodes
                nx.draw_networkx_nodes(
                    nx_graph, pos, ax=ax, nodelist=path, node_color="red", node_size=600
                )
                # Highlight start and goal nodes
                nx.draw_networkx_nodes(
                    nx_graph,
                    pos,
                    ax=ax,
                    nodelist=[start_vertex, goal_vertex],
                    node_color="green",
                    node_size=700,
                )

                # Add edge weights
                edge_labels = nx.get_edge_attributes(nx_graph, "weight")
                nx.draw_networkx_edge_labels(
                    nx_graph, pos, ax=ax, edge_labels=edge_labels, font_size=6
                )

                ax.set_title(f"{algo_name} Path\nTotal Weight: {total_weight:.1f}")

                # Print information to console
                print(f"{algo_name}:")
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

        # Print a summary
        print("\n" + "=" * 50)
        print("Search Algorithm Comparison:")
        print("=" * 50)
        for algo_name, (path, total_weight) in algorithms:
            if path:
                print(
                    f"{algo_name}: {len(path)-1} steps, total weight {total_weight:.1f}"
                )
            else:
                print(f"{algo_name}: No path found")

    except ImportError as e:
        print(f"Required GUI visualization libraries not found: {e}")
        print("\nPlease consider installing these libraries:")
        print("pip install matplotlib networkx")

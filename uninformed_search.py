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

    def all_simple_paths(self, start_vertex: Any, goal_vertex: Any) -> List[List[Any]]:
        """
        Find all simple paths (paths without cycles) from start_vertex to goal_vertex
        using iterative depth-first search (DFS).

        A simple path is a path where no vertex appears more than once. This method
        explores all possible routes between the start and goal vertices, excluding
        any paths that would create cycles.

        Args:
            start_vertex: Starting vertex for the paths
            goal_vertex: Goal vertex to reach

        Returns:
            List[List[Any]]: A list of all simple paths, where each path is a list of
                            vertices from start_vertex to goal_vertex. Returns an empty
                            list if no paths exist or if either vertex is not in the graph.

        Examples:
            >>> graph = UndirectedGraph()
            >>> graph.add_edge("A", "B")
            >>> graph.add_edge("B", "C")
            >>> graph.add_edge("A", "C")
            >>> paths = graph.all_simple_paths("A", "C")
            >>> print(paths)
            [['A', 'B', 'C'], ['A', 'C']]

        Note:
            - The number of simple paths can grow exponentially with graph size
            - Edge weights are not considered in this search
            - The algorithm uses iterative DFS to avoid recursion depth limits
        """
        if not self.has_vertex(start_vertex) or not self.has_vertex(goal_vertex):
            return []

        stack = [(start_vertex, [start_vertex])]
        all_paths = []

        while stack:
            current_vertex, path = stack.pop()

            if current_vertex == goal_vertex:
                all_paths.append(path.copy())
                continue  # Continue searching for other paths

            for neighbor, _ in self.graph[current_vertex].items():
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))

        return all_paths


if __name__ == "__main__":

    from graph import twenty_, graph2nx, nx2ax, HAS_NX_MPL, watts_strogatz_

    # graph = twenty_(UndirectedGraph())
    n = 30
    graph = watts_strogatz_(UndirectedGraph(), n=n, k=4)

    print(graph)

    # Define start and goal vertices
    start_vertex = 0
    goal_vertex = n // 2

    # Run each search algorithm
    algorithms = [
        ("DFS", graph.dfs(start_vertex, goal_vertex)),
        ("BFS", graph.bfs(start_vertex, goal_vertex)),
        ("UCS", graph.ucs(start_vertex, goal_vertex)),
    ]

    # Print a summary
    print("\n" + "=" * 50)
    print("Search Algorithm Comparison:")
    print("=" * 50)
    print("Number of nodes:", len(graph.graph.keys()))
    for algo_name, (path, total_weight) in algorithms:
        if path:
            print(f"{algo_name}: {len(path)-1} steps, total weight {total_weight:.1f}")
        else:
            print(f"{algo_name}: No path found")

    if HAS_NX_MPL:

        import matplotlib.pyplot as plt
        import networkx as nx

        # Create a figure to display BFS, DFS, and UCS paths
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Convert the graph to networkx for visualization using graph.py's function
        nx_graph = graph2nx(graph)

        # Use a consistent layout for all subplots
        pos = nx.spring_layout(nx_graph, seed=42)

        for i, (algo_name, (path, total_weight)) in enumerate(algorithms):
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
                    width=3,
                    alpha=.5,
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

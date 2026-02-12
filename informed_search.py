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

    from graph import twenty_

    graph = twenty_(UndirectedGraph())

    print("Original graph:")
    print(graph)
    print()

    # Test the spt method
    start_vertex = "N0"

    # Try to visualize if matplotlib and networkx are available
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

    except ImportError as e:
        print(f"\nVisualization libraries not available: {e}")
        print("To visualize the graph and tree, install matplotlib and networkx:")
        print("pip install matplotlib networkx")

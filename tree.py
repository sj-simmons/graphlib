from typing import Any
from graph import UndirectedGraph_


class UndirectedGraph(UndirectedGraph_):
    def __init__(self) -> None:
        super().__init__()

    def dfs_tree(self, start_vertex: Any) -> "UndirectedGraph":
        """
        Perform Depth-First Search from the given start vertex and return the DFS tree.

        Args:
            start_vertex: The vertex to start DFS from

        Returns:
            UndirectedGraph: A new graph representing the DFS tree

        Raises:
            ValueError: If start_vertex is not in the graph
        """
        if not self.has_vertex(start_vertex):
            raise ValueError(f"Vertex {start_vertex} not found in graph")

        # Initialize the tree
        tree = UndirectedGraph()
        tree.add_vertex(start_vertex)

        # Stack for DFS: (current_vertex, parent_vertex)
        stack = [(start_vertex, None)]
        visited = {start_vertex}

        while stack:
            current, parent = stack.pop()

            # Add edge to parent if not the root
            if parent is not None:
                weight = self.get_weight(parent, current)
                tree.add_edge(parent, current, weight)

            # Explore neighbors in reverse order for consistent DFS
            neighbors = self.get_neighbors(current)
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, current))

        return tree

    def bfs_tree(self, start_vertex: Any) -> "UndirectedGraph":
        """
        Perform Breadth-First Search from the given start vertex and return the BFS tree.

        Args:
            start_vertex: The vertex to start BFS from

        Returns:
            UndirectedGraph: A new graph representing the BFS tree

        Raises:
            ValueError: If start_vertex is not in the graph
        """
        if not self.has_vertex(start_vertex):
            raise ValueError(f"Vertex {start_vertex} not found in graph")

        # Initialize the tree
        tree = UndirectedGraph()
        tree.add_vertex(start_vertex)

        # Queue for BFS: (current_vertex, parent_vertex)
        from collections import deque

        queue = deque([(start_vertex, None)])
        visited = {start_vertex}

        while queue:
            current, parent = queue.popleft()

            # Add edge to parent if not the root
            if parent is not None:
                weight = self.get_weight(parent, current)
                tree.add_edge(parent, current, weight)

            # Explore neighbors in order
            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current))

        return tree

    def prim_mst(self, start_vertex: Any = None) -> "UndirectedGraph":
        """
        Find the Minimum Spanning Tree (MST) using Prim's algorithm.

        Args:
            start_vertex: The vertex to start building the MST from.
                         If None, use the first vertex from get_vertices().

        Returns:
            UndirectedGraph: A new graph representing the MST.

        Raises:
            ValueError: If start_vertex is provided and not in the graph,
                       or if the graph is empty.
        """
        if self.is_empty():
            raise ValueError("Graph is empty, cannot find MST")

        # Determine start vertex
        if start_vertex is None:
            start_vertex = self.get_vertices()[0]
        elif not self.has_vertex(start_vertex):
            raise ValueError(f"Vertex {start_vertex} not found in graph")

        # Initialize MST
        mst = UndirectedGraph()
        mst.add_vertex(start_vertex)

        # Min-heap to store edges: (weight, current_vertex, parent_vertex)
        # We'll use a priority queue to always pick the minimum weight edge
        import heapq

        heap = []

        # Track visited vertices
        visited = set([start_vertex])

        # Add all edges from start_vertex to the heap
        for neighbor, weight in self.graph[start_vertex].items():
            heapq.heappush(heap, (weight, neighbor, start_vertex))

        # While there are edges in the heap and not all vertices are in MST
        while heap and len(visited) < len(self.get_vertices()):
            # Get the minimum weight edge
            weight, current, parent = heapq.heappop(heap)

            # Skip if current vertex is already in MST
            if current in visited:
                continue

            # Add current vertex to MST
            visited.add(current)
            mst.add_vertex(current)
            mst.add_edge(parent, current, weight)

            # Add edges from current vertex to neighbors not yet in MST
            for neighbor, edge_weight in self.graph[current].items():
                if neighbor not in visited:
                    heapq.heappush(heap, (edge_weight, neighbor, current))

        # Note: If the graph is disconnected, the MST will only include
        # vertices reachable from start_vertex
        return mst

    def spt(self, start_vertex: Any = None) -> "UndirectedGraph":
        """
        Find the Shortest Path Tree (SPT) from start_vertex to all reachable vertices
        using Dijkstra's algorithm. This is a slight modification of prim_mst.

        Args:
            start_vertex: The vertex to start building the SPT from.
                         If None, use the first vertex from get_vertices().

        Returns:
            UndirectedGraph: A new graph representing the Shortest Path Tree.

        Raises:
            ValueError: If start_vertex is provided and not in the graph,
                       or if the graph is empty.
        """
        if self.is_empty():
            raise ValueError("Graph is empty, cannot find SPT")

        # Determine start vertex
        if start_vertex is None:
            start_vertex = self.get_vertices()[0]
        elif not self.has_vertex(start_vertex):
            raise ValueError(f"Vertex {start_vertex} not found in graph")

        # Initialize SPT
        spt = UndirectedGraph()
        spt.add_vertex(start_vertex)

        # Min-heap to store: (total_distance, current_vertex, parent_vertex)
        # Unlike Prim's, we track total distance from start, not just edge weight
        import heapq

        heap = []

        # Track visited vertices and their distances from start
        visited = set([start_vertex])
        distances = {start_vertex: 0}

        # Add all edges from start_vertex to the heap with their weights as distances
        for neighbor, weight in self.graph[start_vertex].items():
            total_distance = weight  # Distance from start to neighbor
            heapq.heappush(heap, (total_distance, neighbor, start_vertex))
            distances[neighbor] = total_distance

        # While there are vertices to process
        while heap:
            # Get the vertex with smallest total distance from start
            total_distance, current, parent = heapq.heappop(heap)

            # Skip if current vertex is already in SPT (visited)
            if current in visited:
                continue

            # Add current vertex to SPT
            visited.add(current)
            spt.add_vertex(current)
            # Add edge with the original weight (not total distance)
            edge_weight = self.get_weight(parent, current)
            spt.add_edge(parent, current, edge_weight)

            # Update distances and add neighbors to heap
            for neighbor, weight in self.graph[current].items():
                if neighbor not in visited:
                    new_distance = total_distance + weight
                    # If we found a shorter path to neighbor, update
                    if neighbor not in distances or new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        heapq.heappush(heap, (new_distance, neighbor, current))

        # Note: If the graph is disconnected, the SPT will only include
        # vertices reachable from start_vertex
        return spt


if __name__ == "__main__":
    from graph import twenty_, graph2nx, nx2ax, HAS_NX_MPL

    graph = twenty_(UndirectedGraph())

    # Get DFS and BFS trees
    start_vertex = "N0"
    dfs_tree = graph.dfs_tree(start_vertex)
    bfs_tree = graph.bfs_tree(start_vertex)

    print("Original graph (20-node example):")
    print(graph)

    # Print statistics about the DFS tree
    print(f"\nDFS Tree starting from {start_vertex}:")
    print(f"DFS Tree vertices: {len(dfs_tree)}")
    print(f"DFS Tree edges: {len(dfs_tree.get_edges())}")

    # Print statistics about the BFS tree
    print(f"\nBFS Tree starting from {start_vertex}:")
    print(f"BFS Tree vertices: {len(bfs_tree)}")
    print(f"BFS Tree edges: {len(bfs_tree.get_edges())}")

    # Compare the trees
    print("\n" + "=" * 50)
    print("Comparison:")
    print(
        f"Both trees should have the same number of vertices reachable from {start_vertex}"
    )
    print(f"DFS Tree number of vertices: {len(dfs_tree)}")
    print(f"BFS Tree number of vertices: {len(bfs_tree)}")

    # Helper function to check reachability
    def get_reachable_vertices(graph, start_vertex):
        """Get all vertices reachable from start_vertex using DFS."""
        reachable = set()
        stack = [start_vertex]
        visited = set([start_vertex])
        while stack:
            current = stack.pop()
            reachable.add(current)
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return reachable

    # Get reachable vertices
    reachable_from_start = get_reachable_vertices(graph, start_vertex)
    all_vertices = graph.get_vertices()

    print(f"\nTotal vertices in original graph: {len(all_vertices)}")
    print(f"Vertices reachable from {start_vertex}: {len(reachable_from_start)}")

    # Test Prim's MST
    print("\n" + "=" * 50)
    print("Testing Prim's Minimum Spanning Tree:")
    try:
        mst = graph.prim_mst(start_vertex)
        print(f"MST vertices: {len(mst)}")
        print(f"MST edges: {len(mst.get_edges())}")

        # Calculate total weight of MST
        total_weight = sum(w for _, _, w in mst.get_edges())
        print(f"Total MST weight: {total_weight:.2f}")

        # If MST includes all reachable vertices, it's valid
        if len(mst) == len(reachable_from_start):
            print("✓ MST includes all reachable vertices")
        else:
            print(
                "⚠ MST may not include all reachable vertices (graph may be disconnected)"
            )

    except ValueError as e:
        print(f"Error computing MST: {e}")

    # Test Shortest Path Tree
    print("\n" + "=" * 50)
    print("Testing Shortest Path Tree (Dijkstra's algorithm):")
    try:
        spt_tree = graph.spt(start_vertex)
        print(f"SPT vertices: {len(spt_tree)}")
        print(f"SPT edges: {len(spt_tree.get_edges())}")

        # Calculate total weight of SPT
        total_spt_weight = sum(w for _, _, w in spt_tree.get_edges())
        print(f"Total SPT weight: {total_spt_weight:.2f}")

        print(f"\nSPT includes {len(spt_tree)} vertices")
        if len(spt_tree) == len(reachable_from_start):
            print("✓ SPT includes all reachable vertices")
        else:
            print(
                "⚠ SPT may not include all reachable vertices (graph may be disconnected)"
            )

    except ValueError as e:
        print(f"Error computing SPT: {e}")

    if HAS_NX_MPL:
        import matplotlib.pyplot as plt
        import networkx as nx

        # Create Figure 1: Original graph, DFS tree, and BFS tree in one row
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

        # Convert graphs to networkx for visualization
        nx_original = graph2nx(graph)
        nx_dfs = graph2nx(dfs_tree)
        nx_bfs = graph2nx(bfs_tree)

        # Use a consistent layout for all subplots based on the original graph
        pos = nx.spring_layout(nx_original, seed=42)

        # Plot original graph
        ax = axes1[0]
        nx2ax(nx_original, ax, seed=42, show_weights=True, pos=pos)
        # Highlight start vertex
        nx.draw_networkx_nodes(
            nx_original,
            pos,
            ax=ax,
            nodelist=[start_vertex],
            node_color="lightsteelblue",
            node_size=700,
            edgecolors="black",
        )
        # Calculate total weight of original graph
        edge_labels_original = nx.get_edge_attributes(nx_original, "weight")
        total_original_weight = sum(edge_labels_original.values())
        ax.set_title(f"Original Graph\nTotal Weight: {total_original_weight:.2f}")

        # Plot DFS tree - use the same layout for consistency
        ax = axes1[1]
        nx2ax(nx_dfs, ax, seed=42, show_weights=True, pos=pos)
        # Highlight start vertex
        nx.draw_networkx_nodes(
            nx_dfs,
            pos,
            ax=ax,
            nodelist=[start_vertex],
            node_color="lightsteelblue",
            node_size=700,
            edgecolors="black",
        )
        # Calculate total weight of DFS tree
        edge_labels_dfs = nx.get_edge_attributes(nx_dfs, "weight")
        total_dfs_weight = sum(edge_labels_dfs.values())
        ax.set_title(
            f"DFS Tree from {start_vertex}\nTotal Weight: {total_dfs_weight:.2f}"
        )

        # Plot BFS tree - use the same layout for consistency
        ax = axes1[2]
        nx2ax(nx_bfs, ax, seed=42, show_weights=True, pos=pos)
        # Highlight start vertex
        nx.draw_networkx_nodes(
            nx_bfs,
            pos,
            ax=ax,
            nodelist=[start_vertex],
            node_color="lightsteelblue",
            node_size=700,
            edgecolors="black",
        )
        # Calculate total weight of BFS tree
        edge_labels_bfs = nx.get_edge_attributes(nx_bfs, "weight")
        total_bfs_weight = sum(edge_labels_bfs.values())
        ax.set_title(
            f"BFS Tree from {start_vertex}\nTotal Weight: {total_bfs_weight:.2f}"
        )

        plt.suptitle(
            f"Tree Algorithms from {start_vertex}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

        # Create Figure 2: MST and SPT comparison
        try:
            # Create MST and SPT
            mst = graph.prim_mst(start_vertex)
            spt_tree = graph.spt(start_vertex)
            nx_mst = graph2nx(mst)
            nx_spt = graph2nx(spt_tree)

            # Create a new figure for MST vs SPT comparison
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

            # Plot MST in first subplot
            ax1 = axes2[0]
            nx2ax(nx_mst, ax1, seed=42, show_weights=True, pos=pos)
            # Highlight start vertex
            nx.draw_networkx_nodes(
                nx_mst,
                pos,
                ax=ax1,
                nodelist=[start_vertex],
                node_color="lightsteelblue",
                node_size=700,
                edgecolors="black",
            )
            # Calculate total weight of MST
            edge_labels_mst = nx.get_edge_attributes(nx_mst, "weight")
            total_mst_weight = sum(edge_labels_mst.values())
            ax1.set_title(
                f"Prim's MST from {start_vertex}\nTotal Weight: {total_mst_weight:.2f}"
            )

            # Plot SPT in second subplot
            ax2 = axes2[1]
            nx2ax(nx_spt, ax2, seed=42, show_weights=True, pos=pos)
            # Highlight start vertex
            nx.draw_networkx_nodes(
                nx_spt,
                pos,
                ax=ax2,
                nodelist=[start_vertex],
                node_color="lightsteelblue",
                node_size=700,
                edgecolors="black",
            )
            # Calculate total weight of SPT
            edge_labels_spt = nx.get_edge_attributes(nx_spt, "weight")
            total_spt_weight = sum(edge_labels_spt.values())
            ax2.set_title(
                f"Shortest Path Tree from {start_vertex}\nTotal Weight: {total_spt_weight:.2f}"
            )

            plt.suptitle(
                "Minimum Spanning Tree vs Shortest Path Tree",
                fontsize=16,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"\nCould not create MST vs SPT comparison figure: {e}")

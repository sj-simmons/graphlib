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
    # Import the same function used in search.py to create the example graph
    from graph import twenty_

    # Create the same graph as in search.py
    graph = twenty_(UndirectedGraph())

    # Get DFS and BFS trees
    start_vertex = "N0"
    dfs_tree = graph.dfs_tree(start_vertex)
    bfs_tree = graph.bfs_tree(start_vertex)

    # Visualize if matplotlib and networkx are available
    has_viz_libs = True
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

    except ImportError as e:
        print(f"\nVisualization libraries not available: {e}")
        print("To visualize the trees, install matplotlib and networkx:")
        print("pip install matplotlib networkx")
        has_viz_libs = False

    print("Original graph (20-node example):")
    print(graph)

    # Demonstrate dfs_tree
    if not has_viz_libs:
        print("\n" + "=" * 50)
        print(f"DFS Tree starting from {start_vertex}:")
        print(dfs_tree)

    # Print some statistics about the DFS tree
    print(f"DFS Tree vertices: {len(dfs_tree)}")
    print(f"DFS Tree edges: {len(dfs_tree.get_edges())}")

    # Demonstrate bfs_tree
    if not has_viz_libs:
        print("\n" + "=" * 50)
        print(f"BFS Tree starting from {start_vertex}:")
        print(bfs_tree)

    # Print some statistics about the BFS tree
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

    # Test Prim's MST
    print("\n" + "=" * 50)
    print("Testing Prim's Minimum Spanning Tree:")
    try:
        mst = graph.prim_mst(start_vertex)
        print(f"MST vertices: {len(mst)}")
        print(f"MST edges: {len(mst.get_edges())}")
        print("MST edges with weights:")
        for u, v, w in mst.get_edges():
            print(f"  {u} -- {v} : {w}")

        # Calculate total weight of MST
        total_weight = sum(w for _, _, w in mst.get_edges())
        print(f"Total MST weight: {total_weight}")

        # Check if all vertices are reachable in the original graph
        all_vertices = graph.get_vertices()
        reachable_from_start = set()

        # Simple reachability check using DFS
        stack = [start_vertex]
        visited = set([start_vertex])
        while stack:
            current = stack.pop()
            reachable_from_start.add(current)
            for neighbor in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        print(f"\nTotal vertices in original graph: {len(all_vertices)}")
        print(f"Vertices reachable from {start_vertex}: {len(reachable_from_start)}")
        print(f"MST includes {len(mst)} vertices")

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
        print("SPT edges with weights:")
        for u, v, w in spt_tree.get_edges():
            print(f"  {u} -- {v} : {w}")

        # Calculate total weight of SPT
        total_spt_weight = sum(w for _, _, w in spt_tree.get_edges())
        print(f"Total SPT weight: {total_spt_weight}")

        print(f"\nSPT includes {len(spt_tree)} vertices")
        if len(spt_tree) == len(reachable_from_start):
            print("✓ SPT includes all reachable vertices")
        else:
            print(
                "⚠ SPT may not include all reachable vertices (graph may be disconnected)"
            )

    except ValueError as e:
        print(f"Error computing SPT: {e}")

    # Check if all vertices are reachable
    all_vertices = graph.get_vertices()
    reachable_from_start = set()

    # Simple reachability check using DFS
    stack = [start_vertex]
    visited = set([start_vertex])
    while stack:
        current = stack.pop()
        reachable_from_start.add(current)
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    print(f"\nTotal vertices in original graph: {len(all_vertices)}")
    print(f"Vertices reachable from {start_vertex}: {len(reachable_from_start)}")

    if has_viz_libs:

        # Create a figure to display original graph, DFS tree, and BFS tree
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Convert graphs to networkx for visualization
        nx_original = graph.to_networkx()
        nx_dfs = dfs_tree.to_networkx()
        nx_bfs = bfs_tree.to_networkx()

        # Use a consistent layout for all subplots
        pos = nx.spring_layout(nx_original, seed=42)

        # Plot original graph
        ax = axes[0]
        nx.draw(
            nx_original,
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
        # Highlight start vertex
        nx.draw_networkx_nodes(
            nx_original,
            pos,
            ax=ax,
            nodelist=[start_vertex],
            node_color="green",
            node_size=700,
        )
        # Add edge weights to original graph
        edge_labels_original = nx.get_edge_attributes(nx_original, "weight")
        nx.draw_networkx_edge_labels(
            nx_original,
            pos,
            ax=ax,
            edge_labels=edge_labels_original,
            font_size=6,
            font_color="black",
        )
        # Calculate total weight of original graph
        total_original_weight = sum(edge_labels_original.values())
        ax.set_title(
            f"Original Graph with Edge Weights\nTotal Weight: {total_original_weight:.2f}"
        )

        # Plot DFS tree
        ax = axes[1]
        nx.draw(
            nx_dfs,
            pos,
            ax=ax,
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            font_size=8,
            font_weight="bold",
            edge_color="blue",
            width=2,
        )
        # Highlight start vertex
        nx.draw_networkx_nodes(
            nx_dfs,
            pos,
            ax=ax,
            nodelist=[start_vertex],
            node_color="green",
            node_size=700,
        )
        # Add edge weights to DFS tree
        edge_labels_dfs = nx.get_edge_attributes(nx_dfs, "weight")
        nx.draw_networkx_edge_labels(
            nx_dfs,
            pos,
            ax=ax,
            edge_labels=edge_labels_dfs,
            font_size=6,
            font_color="darkblue",
        )
        # Calculate total weight of DFS tree
        total_dfs_weight = sum(edge_labels_dfs.values())
        ax.set_title(
            f"DFS Tree with Edge Weights\nTotal Weight: {total_dfs_weight:.2f}"
        )

        # Plot BFS tree
        ax = axes[2]
        nx.draw(
            nx_bfs,
            pos,
            ax=ax,
            with_labels=True,
            node_color="lightcoral",
            node_size=500,
            font_size=8,
            font_weight="bold",
            edge_color="red",
            width=2,
        )
        # Highlight start vertex
        nx.draw_networkx_nodes(
            nx_bfs,
            pos,
            ax=ax,
            nodelist=[start_vertex],
            node_color="green",
            node_size=700,
        )
        # Add edge weights to BFS tree
        edge_labels_bfs = nx.get_edge_attributes(nx_bfs, "weight")
        nx.draw_networkx_edge_labels(
            nx_bfs,
            pos,
            ax=ax,
            edge_labels=edge_labels_bfs,
            font_size=6,
            font_color="darkred",
        )
        # Calculate total weight of BFS tree
        total_bfs_weight = sum(edge_labels_bfs.values())
        ax.set_title(
            f"BFS Tree with Edge Weights\nTotal Weight: {total_bfs_weight:.2f}"
        )

        plt.suptitle(
            f"DFS and BFS Trees from {start_vertex}",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()

        # Create a second figure for MST and SPT
        try:
            mst = graph.prim_mst(start_vertex)
            spt_tree = graph.spt(start_vertex)

            fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Convert to networkx
            nx_mst = mst.to_networkx()
            nx_spt = spt_tree.to_networkx()

            # Plot MST
            ax = axes[0]
            nx.draw(
                nx_mst,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightgreen",
                node_size=500,
                font_size=8,
                font_weight="bold",
                edge_color="darkgreen",
                width=3,
            )
            # Highlight start vertex
            nx.draw_networkx_nodes(
                nx_mst,
                pos,
                ax=ax,
                nodelist=[start_vertex],
                node_color="green",
                node_size=700,
            )
            # Add edge weights
            edge_labels_mst = nx.get_edge_attributes(nx_mst, "weight")
            nx.draw_networkx_edge_labels(
                nx_mst,
                pos,
                ax=ax,
                edge_labels=edge_labels_mst,
                font_size=8,
                font_color="darkred",
            )
            # Calculate total weight of MST
            total_mst_weight = sum(edge_labels_mst.values())
            ax.set_title(
                f"Prim's MST from {start_vertex}\nTotal Weight: {total_mst_weight:.2f}"
            )

            # Plot SPT
            ax = axes[1]
            nx.draw(
                nx_spt,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightblue",
                node_size=500,
                font_size=8,
                font_weight="bold",
                edge_color="blue",
                width=3,
            )
            # Highlight start vertex
            nx.draw_networkx_nodes(
                nx_spt,
                pos,
                ax=ax,
                nodelist=[start_vertex],
                node_color="green",
                node_size=700,
            )
            # Add edge weights
            edge_labels_spt = nx.get_edge_attributes(nx_spt, "weight")
            nx.draw_networkx_edge_labels(
                nx_spt,
                pos,
                ax=ax,
                edge_labels=edge_labels_spt,
                font_size=8,
                font_color="darkblue",
            )
            # Calculate total weight of SPT
            total_spt_weight = sum(edge_labels_spt.values())
            ax.set_title(
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
            print(f"\nCould not visualize MST or SPT: {e}")
            plt.show()

from collections import deque
import random
from typing import Any, Dict, List, Optional, Tuple, Set, Union, TypeVar


class UndirectedGraph_:
    def __init__(self) -> None:
        """
        Initialize an empty undirected graph.
        The graph is represented as an adjacency list using a dictionary.
        """
        self.graph: Dict[Any, Dict[Any, Union[int, float]]] = {}

    def add_vertex(self, vertex: Any) -> None:
        """
        Add a vertex to the graph.

        Args:
            vertex: The vertex to add (can be any hashable type)
        """
        if vertex not in self.graph:
            self.graph[vertex] = {}

    def add_edge(
        self, vertex1: Any, vertex2: Any, weight: Union[int, float] = 1
    ) -> None:
        """
        Add an undirected edge between two vertices.

        Args:
            vertex1: First vertex
            vertex2: Second vertex
            weight: Weight for the edge (default = 1 for unweighted graphs)
        """
        # Add vertices if they don't exist
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)

        # Add edge in both directions
        self.graph[vertex1][vertex2] = weight
        self.graph[vertex2][vertex1] = weight

    def has_vertex(self, vertex: Any) -> bool:
        """
        Check if a vertex exists in the graph.

        Args:
            vertex: The vertex to check

        Returns:
            bool: True if vertex exists, False otherwise
        """
        return vertex in self.graph

    def has_edge(self, vertex1: Any, vertex2: Any) -> bool:
        """
        Check if an edge exists between two vertices.

        Args:
            vertex1: First vertex
            vertex2: Second vertex

        Returns:
            bool: True if edge exists, False otherwise
        """
        if vertex1 in self.graph and vertex2 in self.graph:
            return vertex2 in self.graph[vertex1]
        return False

    def get_neighbors(self, vertex: Any) -> List[Any]:
        """
        Get all neighbors of a vertex.

        Args:
            vertex: The vertex

        Returns:
            list: List of neighboring vertices
        """
        if vertex in self.graph:
            return list(self.graph[vertex].keys())
        return []

    def get_vertices(self) -> List[Any]:
        """
        Get all vertices in the graph.

        Returns:
            list: List of all vertices
        """
        return list(self.graph.keys())

    def get_edges(self) -> List[Tuple[Any, Any, Union[int, float]]]:
        """
        Get all edges in the graph with their weights.

        Returns:
            list: List of edges as tuples (vertex1, vertex2, weight)
        """
        edges = []
        visited = set()

        for vertex in self.graph:
            for neighbor, weight in self.graph[vertex].items():
                # Use sorted tuple to ensure edge appears only once in undirected graph
                edge_key = tuple(sorted((vertex, neighbor)))
                if edge_key not in visited:
                    edges.append((vertex, neighbor, weight))
                    visited.add(edge_key)

        return edges

    def get_weight(self, vertex1: Any, vertex2: Any) -> Optional[Union[int, float]]:
        """
        Get the weight of an edge between two vertices.

        Args:
            vertex1: First vertex
            vertex2: Second vertex

        Returns:
            The weight of the edge, or None if no edge exists
        """
        if self.has_edge(vertex1, vertex2):
            return self.graph[vertex1][vertex2]
        return None

    def is_empty(self) -> bool:
        """
        Check if the graph is empty.

        Returns:
            bool: True if graph is empty, False otherwise
        """
        return len(self.graph) == 0

    def __str__(self) -> str:
        """
        String representation of the graph.
        """
        result = "Undirected Graph:\n"
        for vertex in self.graph:
            neighbors = [
                f"{neighbor}(w:{weight})"
                for neighbor, weight in self.graph[vertex].items()
            ]
            result += f"{vertex}: {', '.join(neighbors)}\n"
        return result

    def __repr__(self) -> str:
        """
        Representation of the graph.
        """
        return f"UndirectedGraph_({len(self.graph)} vertices, {len(self.get_edges())} edges)"

    def __len__(self) -> int:
        """
        Get the number of vertices in the graph.
        """
        return len(self.graph)


T = TypeVar("T", bound="UndirectedGraph_")


def complete_(
    graph: T,
    n: int = 10,
    weight_range: Tuple[Union[int, float], Union[int, float]] = (1, 10),
    seed: Optional[int] = None,
) -> T:
    """
    Generate a complete graph with n nodes (K_n).

    In a complete graph, every pair of distinct vertices is connected by a unique edge.

    Args:
        graph: An empty instance of a subclass of UndirectedGraph_ to populate
        n: Number of nodes in the graph
        weight_range: Tuple (min_weight, max_weight) for edge weights
        seed: Random seed for reproducibility

    Returns:
        T: The populated complete graph

    Raises:
        ValueError: If parameters are invalid
        AssertionError: If graph is not empty
    """
    assert len(graph) == 0, "You probably wanted to start with an empty graph!"

    if n <= 0:
        raise ValueError("n must be positive")
    if weight_range[0] > weight_range[1]:
        raise ValueError("min_weight must be <= max_weight")

    # Initialize random number generator
    rng = random.Random(seed)

    # Add vertices
    for i in range(n):
        graph.add_vertex(i)

    # Add edges between every pair of vertices
    for i in range(n):
        for j in range(i + 1, n):
            # Generate random weight within the specified range
            weight = round(rng.uniform(weight_range[0], weight_range[1]), 2)
            graph.add_edge(i, j, weight)

    return graph


def watts_strogatz_(
    graph: T,
    n: int = 20,
    k: int = 4,
    beta: float = 0.3,
    weight_range: Tuple[Union[int, float], Union[int, float]] = (1, 10),
    seed: Optional[int] = None,
) -> T:
    """
    Generate a Watts-Strogatz small-world graph.

    The graph starts as a ring lattice where each node is connected to its k nearest neighbors
    (k/2 on each side). Then, with probability beta, each edge is rewired to a random node.

    Args:
        graph: An empty instance of a subclass of UndirectedGraph_ to populate
        n: Number of nodes in the graph
        k: Each node is connected to k nearest neighbors in ring topology (must be even)
        beta: Probability of rewiring each edge (0 <= beta <= 1)
        weight_range: Tuple (min_weight, max_weight) for edge weights
        seed: Random seed for reproducibility

    Returns:
        T: The populated Watts-Strogatz small-world graph

    Raises:
        ValueError: If parameters are invalid
        AssertionError: If graph is not empty
    """
    assert len(graph) == 0, "You probably wanted to start with an empty graph!"

    if n <= 0:
        raise ValueError("n must be positive")
    if k <= 0 or k % 2 != 0:
        raise ValueError("k must be a positive even integer")
    if k >= n:
        raise ValueError("k must be less than n")
    if beta < 0 or beta > 1:
        raise ValueError("beta must be between 0 and 1")
    if weight_range[0] > weight_range[1]:
        raise ValueError("min_weight must be <= max_weight")

    # Initialize random number generator
    rng = random.Random(seed)

    # Add vertices
    for i in range(n):
        graph.add_vertex(i)

    # Track which edges exist to avoid duplicates
    edges_set: Set[Tuple[int, ...]] = set()

    # First, create the regular ring lattice
    for node in range(n):
        for j in range(1, k // 2 + 1):
            neighbor = (node + j) % n
            # Sort to ensure undirected edge representation is consistent
            edge = (node, neighbor) if node < neighbor else (neighbor, node)
            if edge not in edges_set:
                edges_set.add(edge)

    # Now rewire edges with probability beta
    rewired_edges_set: Set[Tuple[int, ...]] = set()

    for u, v in edges_set:
        if rng.random() < beta:
            # Choose a new random node to connect to u
            # The new node must be different from u and not already connected to u
            possible_nodes = []
            for i in range(n):
                if i != u:
                    edge = (u, i) if u < i else (i, u)
                    if edge not in edges_set and edge not in rewired_edges_set:
                        possible_nodes.append(i)

            if possible_nodes:
                new_v = rng.choice(possible_nodes)
                # Remove old edge (u, v) and add new edge (u, new_v)
                rewired_edges_set.add((u, new_v) if u < new_v else (new_v, u))
                # Don't add the original edge
                continue

        # Keep the original edge
        rewired_edges_set.add((u, v))

    # Add all edges to the graph with random weights
    for u, v in rewired_edges_set:
        weight = round(rng.uniform(weight_range[0], weight_range[1]), 2)
        graph.add_edge(u, v, weight)

    return graph


def planar_(
    graph: T,
    n: int = 20,
    remove_probability: float = 0.0,
    weight_range: Tuple[Union[int, float], Union[int, float]] = (1, 10),
    seed: Optional[int] = None,
) -> T:
    """
    Generate the dual of a maximal planar graph using Delaunay triangulation of n random points,
    with optional random edge removal applied to the dual graph.

    A maximal planar graph is a planar graph to which no more edges can be added
    without violating planarity. Delaunay triangulation of points in the plane
    produces a maximal planar graph. This function returns the dual of that graph,
    then removes edges from the dual with probability remove_probability while
    ensuring the dual remains connected.

    The dual is always cubic (3-regular) and, by Brook's Theorem, every cubic planar graph
    besides K_4 is 3-colorable.

    Args:
        graph: An empty instance of a subclass of UndirectedGraph_ to populate
        n: Number of nodes in the original graph
        remove_probability: Probability of removing each edge from the dual graph
                          (0 <= remove_probability <= 1)
        weight_range: Tuple (min_weight, max_weight) for edge weights
        seed: Random seed for reproducibility

    Returns:
        T: The dual of the maximal planar graph, with edges possibly removed but still connected

    Raises:
        ValueError: If parameters are invalid
        AssertionError: If graph is not empty
    """
    assert len(graph) == 0, "You probably wanted to start with an empty graph!"

    if n <= 0:
        raise ValueError("n must be positive")
    if remove_probability < 0 or remove_probability > 1:
        raise ValueError("remove_probability must be between 0 and 1")
    if weight_range[0] > weight_range[1]:
        raise ValueError("min_weight must be <= max_weight")

    # Initialize random number generator
    rng = random.Random(seed)

    # Generate n random points in [0, 1] x [0, 1]
    points: List[Tuple[float, float]] = [(rng.random(), rng.random()) for _ in range(n)]

    # Handle small n cases
    if n <= 3:
        if n == 1:
            return graph
        elif n == 2:
            # For 2 points, the dual has 2 faces (inner and outer), but we'll create a simple dual
            graph.add_vertex(0)
            graph.add_vertex(1)
            weight = round(rng.uniform(weight_range[0], weight_range[1]), 2)
            graph.add_edge(0, 1, weight)
            return graph
        elif n == 3:
            # For 3 points, triangle: dual has 1 inner face connected to outer face
            # We'll create one vertex for the inner face
            graph.add_vertex(0)
            return graph

    # For n > 3, implement Delaunay triangulation (maximal planar graph)
    # Find bounding box
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    dx = max_x - min_x
    dy = max_y - min_y
    dmax = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    super_tri = [
        (mid_x - 20 * dmax, mid_y - dmax),
        (mid_x + 20 * dmax, mid_y - dmax),
        (mid_x, mid_y + 20 * dmax),
    ]

    triangles = []
    triangles.append((n, n + 1, n + 2))

    def in_circumcircle(
        p: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
    ) -> bool:
        ax, ay = a
        bx, by = b
        cx, cy = c
        px, py = p

        d11 = ax - px
        d12 = ay - py
        d13 = d11 * d11 + d12 * d12

        d21 = bx - px
        d22 = by - py
        d23 = d21 * d21 + d22 * d22

        d31 = cx - px
        d32 = cy - py
        d33 = d31 * d31 + d32 * d32

        det = (
            d11 * d22 * d33
            + d12 * d23 * d31
            + d13 * d21 * d32
            - d13 * d22 * d31
            - d11 * d23 * d32
            - d12 * d21 * d33
        )

        return det > 0

    # Add points one by one
    for i in range(n):
        point = points[i]
        bad_triangles = []

        for tri in triangles:
            a_idx, b_idx, c_idx = tri
            point_a: Tuple[float, float] = (
                points[a_idx] if a_idx < n else super_tri[a_idx - n]
            )
            point_b: Tuple[float, float] = (
                points[b_idx] if b_idx < n else super_tri[b_idx - n]
            )
            point_c: Tuple[float, float] = (
                points[c_idx] if c_idx < n else super_tri[c_idx - n]
            )

            if in_circumcircle(point, point_a, point_b, point_c):
                bad_triangles.append(tri)

        polygon_edges: List[Tuple[int, int]] = []
        for tri in bad_triangles:
            a, b, c = tri
            tri_edges: List[Tuple[int, int]] = [(a, b), (b, c), (c, a)]
            for edge in tri_edges:
                shared = False
                for other_tri in bad_triangles:
                    if other_tri == tri:
                        continue
                    other_a, other_b, other_c = other_tri
                    other_edges = [
                        (other_a, other_b),
                        (other_b, other_c),
                        (other_c, other_a),
                    ]
                    if edge in other_edges or (edge[1], edge[0]) in other_edges:
                        shared = True
                        break
                if not shared:
                    polygon_edges.append(edge)

        for tri in bad_triangles:
            if tri in triangles:
                triangles.remove(tri)

        for edge in polygon_edges:
            new_tri = (edge[0], edge[1], i)
            triangles.append(new_tri)

    # Remove triangles containing super-triangle vertices
    final_triangles: List[Tuple[int, int, int]] = []
    for tri in triangles:
        a_idx, b_idx, c_idx = tri
        if a_idx < n and b_idx < n and c_idx < n:
            final_triangles.append(tri)

    # Build edge to triangles mapping for the maximal planar graph
    edge_to_triangles: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for tri in final_triangles:
        a, b, c = tri
        edges = [
            (a, b) if a < b else (b, a),
            (b, c) if b < c else (c, b),
            (a, c) if a < c else (c, a),
        ]
        for edge in edges:
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(tri)

    # Build the dual graph of the maximal planar graph
    triangle_to_id: Dict[Tuple[int, int, int], int] = {}
    for i, tri in enumerate(final_triangles):
        triangle_to_id[tri] = i

    # Add vertices for each triangle
    for i in range(len(final_triangles)):
        graph.add_vertex(i)

    # Add edges in the dual
    dual_edges: List[Tuple[int, int, float]] = []
    for edge, tris in edge_to_triangles.items():
        if len(tris) == 2:
            tri1, tri2 = tris
            id1 = triangle_to_id[tri1]
            id2 = triangle_to_id[tri2]
            weight = round(rng.uniform(weight_range[0], weight_range[1]), 2)
            graph.add_edge(id1, id2, weight)
            dual_edges.append((id1, id2, weight))

    # Now, remove edges from the dual with probability remove_probability
    # First, collect all edges
    edges_to_consider: List[Tuple[int, int, float]] = []
    for u in graph.graph:
        for v, w in graph.graph[u].items():
            if u < v:  # To avoid duplicates
                edges_to_consider.append((u, v, w))

    # Remove edges with probability remove_probability
    edges_to_remove: List[Tuple[int, int]] = []
    for u, v, w in edges_to_consider:
        if rng.random() < remove_probability:
            edges_to_remove.append((u, v))

    # Remove the edges
    for u, v in edges_to_remove:
        if v in graph.graph[u]:
            del graph.graph[u][v]
        if u in graph.graph[v]:
            del graph.graph[v][u]

    # Ensure the dual graph remains connected
    # Build adjacency list
    adj: Dict[int, Set[int]] = {vertex: set() for vertex in graph.graph}
    for u in graph.graph:
        for v in graph.graph[u]:
            adj[u].add(v)

    # Check connectivity using BFS
    visited = set()
    if len(graph.graph) > 0:
        start = next(iter(graph.graph.keys()))
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    # If not connected, add edges back until connected
    # We'll add edges from the original dual edges that were removed
    while len(graph.graph) > 0 and len(visited) != len(graph.graph):
        # Find unvisited vertices
        unvisited = [v for v in graph.graph.keys() if v not in visited]
        # For each unvisited vertex, find a path to connect it
        # We'll add back an edge from the original dual edges if possible
        added = False
        for u in list(visited):
            for v in unvisited:
                # Check if this edge was in the original dual
                # We can check by seeing if they were connected in the original dual
                # Since we stored dual_edges, we can check
                for uu, vv, ww in dual_edges:
                    if (u == uu and v == vv) or (u == vv and v == uu):
                        # Add this edge back
                        graph.add_edge(u, v, ww)
                        adj[u].add(v)
                        adj[v].add(u)
                        visited.add(v)
                        added = True
                        break
                if added:
                    break
            if added:
                break
        # If we couldn't add back an original edge, add any edge
        if not added and unvisited:
            u = next(iter(visited))
            v = unvisited[0]
            weight = round(rng.uniform(weight_range[0], weight_range[1]), 2)
            graph.add_edge(u, v, weight)
            adj[u].add(v)
            adj[v].add(u)
            visited.add(v)

    return graph


def rb_graph_(
    graph: T,
    n: int = 20,
    d: int = 3,
    p1: float = 0.5,
    p2: float = 0.5,
    seed: Optional[int] = None,
) -> T:
    """
    Generate a random graph using the RB (random with balanced structure) model.

    This creates a graph where edges represent constraints between variables.
    The graph is generated as the constraint graph of an RB model CSP instance.

    Args:
        graph: An empty instance of a subclass of UndirectedGraph_ to populate
        n: Number of vertices (variables) in the graph
        d: Domain size for each variable (not directly used for graph structure,
           but affects RB model parameters)
        p1: Constraint density (0 ≤ p1 ≤ 1). Determines edge density.
           Higher p1 = more edges.
        p2: Constraint tightness (0 ≤ p2 ≤ 1). Not directly used for graph structure,
           but part of RB model.
        seed: Random seed for reproducibility

    Returns:
        T: The populated random graph based on RB model

    Raises:
        ValueError: If parameters are invalid
        AssertionError: If graph is not empty
    """
    assert len(graph) == 0, "You probably wanted to start with an empty graph!"

    if n <= 0:
        raise ValueError("n must be positive")
    if d <= 0:
        raise ValueError("d must be positive")
    if p1 < 0 or p1 > 1:
        raise ValueError("p1 must be between 0 and 1")
    if p2 < 0 or p2 > 1:
        raise ValueError("p2 must be between 0 and 1")

    # Set random seed if provided
    if seed is not None:
        import random

        random.seed(seed)

    # Calculate total number of possible edges between distinct vertices
    total_possible_edges = n * (n - 1) // 2
    num_edges = int(p1 * total_possible_edges)

    # Add vertices
    for i in range(n):
        graph.add_vertex(i)

    if num_edges > 0:
        # Generate all possible vertex pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Randomly select pairs to have edges
        import random

        edge_pairs = random.sample(all_pairs, num_edges)

        # Add edges with random weights between 1 and 10
        for v1, v2 in edge_pairs:
            weight = round(random.uniform(1, 10), 2)
            graph.add_edge(v1, v2, weight)

    return graph


def erdos_renyi_(
    graph: T,
    n: int = 20,
    p: float = 0.5,
    weight_range: Tuple[Union[int, float], Union[int, float]] = (1, 10),
    seed: Optional[int] = None,
) -> T:
    """
    Generate a random Erdos-Renyi graph G(n, p).

    In G(n, p) each possible edge between distinct vertices is included with
    independent probability p.

    Args:
        graph: An empty instance of a subclass of UndirectedGraph_ to populate
        n: Number of nodes in the graph
        p: Probability for edge inclusion (0 <= p <= 1)
        weight_range: Tuple (min_weight, max_weight) for edge weights
        seed: Random seed for reproducibility

    Returns:
        T: The populated Erdos-Renyi graph

    Raises:
        ValueError: If parameters are invalid
        AssertionError: If graph is not empty
    """
    assert len(graph) == 0, "You probably wanted to start with an empty graph!"

    if n <= 0:
        raise ValueError("n must be positive")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")
    if weight_range[0] > weight_range[1]:
        raise ValueError("min_weight must be <= max_weight")

    # Initialize random number generator
    rng = random.Random(seed)

    # Add vertices
    for i in range(n):
        graph.add_vertex(i)

    # For each possible unordered pair, add edge with probability p
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                weight = round(rng.uniform(weight_range[0], weight_range[1]), 2)
                graph.add_edge(i, j, weight)

    return graph


def barabasi_albert_(
    graph: T,
    n: int = 100,
    m: int = 2,
    weight_range: Tuple[Union[int, float], Union[int, float]] = (1, 10),
    seed: Optional[int] = None,
) -> T:
    """
    Generate a Barabási–Albert scale-free network graph.

    The graph grows through preferential attachment: new nodes are more likely
    to connect to nodes that already have many connections.

    Args:
        graph: An empty instance of a subclass of UndirectedGraph_ to populate
        n: Total number of nodes in the final graph
        m: Number of edges to attach from a new node to existing nodes (m < n)
        weight_range: Tuple (min_weight, max_weight) for edge weights
        seed: Random seed for reproducibility

    Returns:
        T: The populated Barabási–Albert graph

    Raises:
        ValueError: If parameters are invalid
        AssertionError: If graph is not empty
    """
    assert len(graph) == 0, "You probably wanted to start with an empty graph!"

    if n <= 0:
        raise ValueError("n must be positive")
    if m <= 0:
        raise ValueError("m must be positive")
    if m >= n:
        raise ValueError("m must be less than n")
    if weight_range[0] > weight_range[1]:
        raise ValueError("min_weight must be <= max_weight")

    # Initialize random number generator
    rng = random.Random(seed)

    # Start with a complete graph of m+1 nodes
    # This ensures each initial node has at least m edges
    initial_nodes = m + 1

    # Add initial nodes
    for i in range(initial_nodes):
        graph.add_vertex(i)

    # Create initial complete graph among first m+1 nodes
    for i in range(initial_nodes):
        for j in range(i + 1, initial_nodes):
            weight = round(rng.uniform(weight_range[0], weight_range[1]), 2)
            graph.add_edge(i, j, weight)

    # Track degrees for preferential attachment
    degrees = [initial_nodes - 1] * initial_nodes  # Each node has m edges initially

    # Add remaining nodes with preferential attachment
    for new_node in range(initial_nodes, n):
        graph.add_vertex(new_node)

        # Create list of existing nodes weighted by their degree
        # (preferential attachment: higher degree = higher probability)
        node_list = []
        for node_idx, degree in enumerate(degrees):
            node_list.extend([node_idx] * degree)

        # Select m distinct nodes to connect to
        selected_nodes: Set[int] = set()
        attempts = 0
        max_attempts = m * 10  # Prevent infinite loops

        while len(selected_nodes) < m and attempts < max_attempts:
            if node_list:  # Ensure node_list is not empty
                selected_node = rng.choice(node_list)
                if selected_node not in selected_nodes:
                    selected_nodes.add(selected_node)
            attempts += 1

        # If we couldn't find enough distinct nodes, add random ones
        while len(selected_nodes) < m:
            available_nodes = list(range(new_node))
            random_node = rng.choice(available_nodes)
            selected_nodes.add(random_node)

        # Connect new node to selected nodes
        for existing_node in selected_nodes:
            weight = round(rng.uniform(weight_range[0], weight_range[1]), 2)
            graph.add_edge(new_node, existing_node, weight)

            # Update degrees
            degrees[existing_node] += 1

        # Add degree entry for the new node
        degrees.append(m)

    return graph


def twenty_(graph: T, weighted: bool = True, more_edges: bool = True) -> T:
    """
    Generate a 20-node graph with multiple paths from N0 to N19.

    This graph is designed to demonstrate differences between search algorithms:
    - DFS tends to follow the deep path (N0→N3→N4→N5→N6→N7→N19) with light initial edges
      but an expensive final hop
    - BFS explores breadth-first and may find alternative paths
    - UCS finds the optimal cost path considering edge weights

    Args:
        graph: An empty instance of a subclass of UndirectedGraph_ to populate
        weighted: If True, edges have random weights; if False, all edges have weight 1
        more_edges: If True, adds additional cross-connections making the graph more complex

    Returns:
        T: The populated 20-node graph

    Raises:
        AssertionError: If graph is not empty
    """
    assert len(graph) == 0, "You probably wanted to start with an empty graph!"

    nodes: List[str] = [f"N{i}" for i in range(20)]
    for node in nodes:
        graph.add_vertex(node)

    # Create some direct but heavy connections
    graph.add_edge("N0", "N1", random.randint(5, 10) if weighted else 1)
    graph.add_edge("N0", "N2", random.randint(5, 10) if weighted else 1)
    graph.add_edge(
        "N0", "N3", random.randint(1, 3) if weighted else 1
    )  # Light edge that DFS will take

    # Create branching paths
    # Path 1: Light but deep (DFS will likely take this)
    graph.add_edge("N3", "N4", random.randint(1, 3) if weighted else 1)
    graph.add_edge("N4", "N5", random.randint(1, 3) if weighted else 1)
    graph.add_edge("N5", "N6", random.randint(1, 3) if weighted else 1)
    graph.add_edge("N6", "N7", random.randint(1, 3) if weighted else 1)
    graph.add_edge(
        "N7", "N19", random.randint(50, 100) if weighted else 1
    )  # Expensive final hop

    # Path 2: Shorter but heavier edges (BFS might find this)
    graph.add_edge("N1", "N8", random.randint(10, 20) if weighted else 1)
    graph.add_edge("N8", "N9", random.randint(10, 20) if weighted else 1)
    graph.add_edge("N9", "N19", random.randint(10, 20) if weighted else 1)

    # Path 3: Alternative path
    graph.add_edge("N2", "N10", random.randint(15, 25) if weighted else 1)
    graph.add_edge("N10", "N11", random.randint(15, 25) if weighted else 1)
    graph.add_edge("N11", "N19", random.randint(15, 25) if weighted else 1)

    # Add cross-connections to make it more complex
    if more_edges:
        graph.add_edge("N4", "N12", random.randint(5, 15) if weighted else 1)
        graph.add_edge("N12", "N13", random.randint(5, 15) if weighted else 1)
        graph.add_edge("N13", "N19", random.randint(30, 40) if weighted else 1)
        graph.add_edge("N5", "N14", random.randint(20, 30) if weighted else 1)

    graph.add_edge("N14", "N19", random.randint(20, 30) if weighted else 1)
    graph.add_edge("N6", "N15", random.randint(10, 20) if weighted else 1)
    graph.add_edge("N15", "N16", random.randint(10, 20) if weighted else 1)
    graph.add_edge("N16", "N19", random.randint(10, 20) if weighted else 1)

    # Add some random connections between nodes
    additional_edges: List[Tuple[str, str, Union[int, float]]] = [
        ("N1", "N17", random.randint(5, 15) if weighted else 1),
        ("N17", "N18", random.randint(5, 15) if weighted else 1),
        ("N18", "N19", random.randint(5, 15) if weighted else 1),
        ("N8", "N14", random.randint(25, 35) if weighted else 1),
        ("N10", "N15", random.randint(20, 30) if weighted else 1),
        ("N12", "N16", random.randint(15, 25) if weighted else 1),
        ("N9", "N13", random.randint(10, 20) if weighted else 1),
    ]

    for v1, v2, weight in additional_edges:
        graph.add_edge(v1, v2, weight)

    return graph


HAS_NX_MPL = True
try:
    import matplotlib.pyplot as plt
    import networkx as nx

    def graph2nx(graph: T) -> "nx.Graph":
        """
        Convert a (subclass of) UndirectedGraph to a networkx.Graph object.

        Returns:
            networkx.Graph: A networkx graph representation
        """

        nx_graph = nx.Graph()

        # Add all vertices
        for vertex in graph.graph:
            nx_graph.add_node(vertex)

        # Add all edges with weights
        for vertex in graph.graph:
            for neighbor, weight in graph.graph[vertex].items():
                # Add edge only once (undirected)
                if not nx_graph.has_edge(vertex, neighbor):
                    nx_graph.add_edge(vertex, neighbor, weight=weight)

        return nx_graph

    def colormap(nx_graph, coloring) -> List:
        """
        Create a color map for nodes based on a coloring dictionary.

        Args:
            nx_graph: A networkx graph
            coloring: A dictionary mapping nodes to color indices (integers)
                     If None or empty, returns a default color list

        Returns:
            List of colors for each node in the graph
        """
        color_map = []
        nodes = list(nx_graph.nodes())

        # If coloring is None or empty, use a default color
        if not coloring:
            return ["lightgray"] * len(nodes)

        # Get all unique color indices
        unique_colors = set(coloring.values())
        num_unique_colors = max(1, len(unique_colors))

        # Choose appropriate colormap based on number of nodes
        cmap = plt.cm.Set2 if len(nodes) <= 50 else plt.cm.tab10  # type: ignore

        for node in nodes:
            # Get color index for this node, default to 0 if not found
            color_idx = coloring.get(node, 0)
            # Normalize to [0, 1] range
            # Handle the case when there's only one unique color
            if num_unique_colors == 1:
                normalized_idx = 0.0
            else:
                # Map color_idx to [0, 1] range
                # First, find the position of color_idx among sorted unique colors
                sorted_colors = sorted(unique_colors)
                # Find the index of color_idx in sorted_colors
                try:
                    idx_position = sorted_colors.index(color_idx)
                    normalized_idx = idx_position / (num_unique_colors - 1)
                except ValueError:
                    # If color_idx is not in unique_colors (shouldn't happen with .get(node, 0))
                    normalized_idx = 0.0
            color = cmap(normalized_idx)
            color_map.append(color)

        return color_map

    def nx2ax(
        nx_graph: "nx.Graph",
        ax,
        seed=42,
        show_weights: bool = True,
        pos=None,
        num_nodes_thresh=50,  # num of nodes to not use node labels, shrink nodes, ...
        coloring: Optional[Dict[Any, int]] = None,
    ):
        """
        Draw a networkx graph on a matplotlib axis.

        Args:
            nx_graph: The networkx graph to draw
            ax: Matplotlib axis to draw on
            seed: Random seed for layout generation
            show_weights: Whether to display edge weights
            pos: Precomputed node positions (if None, will use spring layout)
            num_nodes_thresh: Threshold for node display options
            coloring: Optional dictionary mapping nodes to color indices
                     If provided, nodes will be colored according to these indices
        """
        num_nodes = len(nx_graph.nodes())

        # Create a layout for the nodes if not provided
        if pos is None:
            pos = nx.spring_layout(nx_graph, seed=seed)

        if num_nodes <= num_nodes_thresh:
            max_label_len = max(len(str(node)) for node in list(nx_graph))
            node_size = 300 + max(max_label_len - 2, 0) * 300
        else:
            node_size = 20 if num_nodes > 180 else 40

        # Plot graph
        nx.draw(
            nx_graph,
            pos,
            ax=ax,
            with_labels=True if num_nodes <= num_nodes_thresh else False,
            node_color=colormap(nx_graph, coloring) if coloring else "lightgray",
            node_size=node_size,
            font_size=10 if num_nodes <= num_nodes_thresh else 8,
            font_weight="bold",
            edge_color="gray",
            width=2 if num_nodes <= num_nodes_thresh else 0.5,
            edgecolors="black",
            alpha=1 if num_nodes <= num_nodes_thresh else 0.7,
        )

        # Draw edge labels (weights) if requested
        if num_nodes <= num_nodes_thresh and show_weights:
            edge_labels = nx.get_edge_attributes(nx_graph, "weight")
            # Format weights to 1 decimal place for cleaner display
            formatted_edge_labels = {}
            for (u, v), weight in edge_labels.items():
                if isinstance(weight, int):
                    formatted_edge_labels[(u, v)] = f"{weight}"
                else:
                    formatted_edge_labels[(u, v)] = f"{weight:.1f}"
            nx.draw_networkx_edge_labels(
                nx_graph,
                pos,
                ax=ax,
                edge_labels=formatted_edge_labels,
                font_size=10,
                font_color="firebrick",
                bbox=dict(alpha=0.7, facecolor="white", edgecolor="none"),
            )

        return node_size

except ImportError as e:
    print(f"Required GUI visualization libraries not found: {e}")
    print("\nPlease consider installing these libraries:")
    print("pip install matplotlib networkx")
    HAS_NX_MPL = False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo creating various graphs.")
    parser.add_argument("-show", help="Toggle showing graphs", action="store_false")
    args = parser.parse_args()

    def create_and_print_graph(
        graph_type: str, graph, expected_edges: Optional[int] = None
    ) -> None:
        """Helper to create and print graph statistics."""
        print(f"\n{graph_type}:")
        print(f"  Number of vertices: {len(graph)}")
        print(f"  Number of edges: {len(graph.get_edges())}")
        if expected_edges:
            print(f"  Expected edges: {expected_edges}")

    # Create and display various graph types
    graphs = {}

    # Complete graph
    graphs["complete"] = complete_(UndirectedGraph_(), n=8)
    create_and_print_graph("Complete graph K_8", graphs["complete"], 8 * 7 // 2)

    # 20-node graph
    graphs["twenty"] = twenty_(UndirectedGraph_())
    create_and_print_graph("20-node graph", graphs["twenty"])

    # Erdos-Renyi graph
    n_er, p_er = 30, 0.2
    graphs["erdos_renyi"] = erdos_renyi_(UndirectedGraph_(), n=n_er, p=p_er)
    create_and_print_graph(
        f"Erdos-Renyi graph (n={n_er}, p={p_er})",
        graphs["erdos_renyi"],
        int(p_er * n_er * (n_er - 1) / 2),
    )

    # Watts-Strogatz graph
    n_ws, k = 30, 6
    graphs["watts_strogatz"] = watts_strogatz_(UndirectedGraph_(), n=n_ws, k=k)
    create_and_print_graph(
        f"Watts-Strogatz graph (n={n_ws}, k={k})", graphs["watts_strogatz"]
    )

    # Barabási–Albert graph
    n_ba, m = 30, 3
    graphs["barabasi_albert"] = barabasi_albert_(UndirectedGraph_(), n=n_ba, m=m)
    create_and_print_graph(
        f"Barabási–Albert graph (n={n_ba}, m={m})", graphs["barabasi_albert"]
    )

    # Small planar graph
    n_planar_small = 20
    graphs["planar_small"] = planar_(
        UndirectedGraph_(), n=n_planar_small, remove_probability=0.02
    )
    create_and_print_graph(
        f"Small planar graph (n={n_planar_small})", graphs["planar_small"]
    )

    # RB model graphs
    n_rb, p1_rb, d_rb = 30, 0.3, 3
    graphs["rb_model"] = rb_graph_(UndirectedGraph_(), n=n_rb, d=d_rb, p1=p1_rb)
    create_and_print_graph(
        f"RB model graph (n={n_rb}, p1={p1_rb})",
        graphs["rb_model"],
        int(p1_rb * n_rb * (n_rb - 1) / 2),
    )

    if args.show and HAS_NX_MPL:

        def create_figure(
            title: str,
            subplot_configs: List[Dict[str, Any]],
            figsize: Tuple[int, int] = (16, 8),
        ) -> None:
            """Helper to create a figure with multiple subplots."""
            num_subplots = len(subplot_configs)
            fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
            axes = axes if num_subplots > 1 else [axes]

            for ax, config in zip(axes, subplot_configs):
                graph = config["graph"]
                title_text = config["title"]
                try_layout = config.get("try_planar", False)

                g_nx = graph2nx(graph)
                try:
                    if try_layout:
                        nx2ax(g_nx, ax, pos=nx.planar_layout(g_nx))
                    else:
                        nx2ax(g_nx, ax)
                except:
                    print(f"Note: {title_text} is not planar")
                    nx2ax(g_nx, ax)

                ax.set_title(title_text)
                ax.axis("off")

            plt.tight_layout()
            plt.show()

        # Figure 1: Complete and 20-node graphs
        create_figure(
            "Basic Graphs",
            [
                {"graph": graphs["complete"], "title": "Complete Graph K_8"},
                {"graph": graphs["twenty"], "title": "20-node Graph"},
            ],
        )

        # Figure 2: Random graphs
        create_figure(
            "Random Graph Models",
            [
                {
                    "graph": graphs["erdos_renyi"],
                    "title": f"Erdos-Renyi (n={n_er}, p={p_er})",
                },
                {
                    "graph": graphs["watts_strogatz"],
                    "title": f"Watts-Strogatz (n={n_ws}, k={k})",
                },
                {
                    "graph": graphs["barabasi_albert"],
                    "title": f"Barabási–Albert (n={n_ba}, m={m})",
                },
            ],
            figsize=(18, 8),
        )

        # Create large graphs for Figure 3
        print("\nGenerating large graphs for visualization...")

        large_graphs = {
            "erdos_renyi_large": erdos_renyi_(UndirectedGraph_(), n=100, p=0.2),
            "watts_strogatz_large": watts_strogatz_(UndirectedGraph_(), n=200, k=8),
            "barabasi_albert_large": barabasi_albert_(UndirectedGraph_(), n=200, m=3),
        }

        for name, graph in large_graphs.items():
            print(
                f"{name.replace('_', ' ').title()}: {len(graph)} vertices, {len(graph.get_edges())} edges"
            )

        # Figure 3: Large random graphs
        create_figure(
            "Large Random Graphs",
            [
                {
                    "graph": large_graphs["erdos_renyi_large"],
                    "title": "Large Erdos-Renyi (n=100, p=0.2)",
                },
                {
                    "graph": large_graphs["watts_strogatz_large"],
                    "title": "Large Watts-Strogatz (n=200, k=8)",
                },
                {
                    "graph": large_graphs["barabasi_albert_large"],
                    "title": "Large Barabási–Albert (n=200, m=3)",
                },
            ],
            figsize=(18, 8),
        )

        # Figure 4: Planar graphs
        print("\nGenerating planar graphs for visualization...")

        n_planar_large = 100
        planar_large = planar_(
            UndirectedGraph_(), n=n_planar_large, remove_probability=0.3
        )
        print(
            f"Large planar graph: {len(planar_large)} vertices, {len(planar_large.get_edges())} edges"
        )

        create_figure(
            "Planar Graphs",
            [
                {
                    "graph": graphs["planar_small"],
                    "title": f"Small Planar (n={n_planar_small})",
                    "try_planar": True,
                },
                {
                    "graph": planar_large,
                    "title": f"Large Planar (n={n_planar_large}, 30% edges removed)",
                    "try_planar": True,
                },
            ],
        )

        # Figure 5: RB model graphs
        print("\nGenerating RB model graphs for visualization...")

        n_rb2, p1_rb2 = 60, 0.5
        rb_model_2 = rb_graph_(UndirectedGraph_(), n=n_rb2, d=3, p1=p1_rb2)
        print(
            f"Second RB model graph: {len(rb_model_2)} vertices, {len(rb_model_2.get_edges())} edges"
        )

        create_figure(
            "RB Model Graphs",
            [
                {
                    "graph": graphs["rb_model"],
                    "title": f"RB Model (n={n_rb}, p1={p1_rb})",
                    "try_planar": True,
                },
                {
                    "graph": rb_model_2,
                    "title": f"RB Model (n={n_rb2}, p1={p1_rb2})",
                    "try_planar": True,
                },
            ],
        )

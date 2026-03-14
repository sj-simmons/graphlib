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
            edge = tuple(sorted((node, neighbor)))
            if edge not in edges_set:
                edges_set.add(edge)

    # Now rewire edges with probability beta
    rewired_edges_set: Set[Tuple[int, ...]] = set()

    for u, v in edges_set:
        if rng.random() < beta:
            # Choose a new random node to connect to u
            # The new node must be different from u and not already connected to u
            possible_nodes = [
                i
                for i in range(n)
                if i != u
                and tuple(sorted((u, i))) not in edges_set
                and tuple(sorted((u, i))) not in rewired_edges_set
            ]

            if possible_nodes:
                new_v = rng.choice(possible_nodes)
                # Remove old edge (u, v) and add new edge (u, new_v)
                rewired_edges_set.add(tuple(sorted((u, new_v))))
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
    points = [(rng.random(), rng.random()) for _ in range(n)]

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

    def in_circumcircle(p, a, b, c):
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
            a = points[a_idx] if a_idx < n else super_tri[a_idx - n]
            b = points[b_idx] if b_idx < n else super_tri[b_idx - n]
            c = points[c_idx] if c_idx < n else super_tri[c_idx - n]

            if in_circumcircle(point, a, b, c):
                bad_triangles.append(tri)

        polygon_edges = []
        for tri in bad_triangles:
            tri_edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for edge in tri_edges:
                shared = False
                for other_tri in bad_triangles:
                    if other_tri == tri:
                        continue
                    other_edges = [
                        (other_tri[0], other_tri[1]),
                        (other_tri[1], other_tri[2]),
                        (other_tri[2], other_tri[0]),
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
    final_triangles = []
    for tri in triangles:
        a_idx, b_idx, c_idx = tri
        if a_idx < n and b_idx < n and c_idx < n:
            final_triangles.append(tri)

    # Build edge to triangles mapping for the maximal planar graph
    edge_to_triangles = {}
    for tri in final_triangles:
        a, b, c = tri
        for edge in [
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((a, c))),
        ]:
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(tri)

    # Build the dual graph of the maximal planar graph
    triangle_to_id = {}
    for i, tri in enumerate(final_triangles):
        triangle_to_id[tri] = i

    # Add vertices for each triangle
    for i in range(len(final_triangles)):
        graph.add_vertex(i)

    # Add edges in the dual
    dual_edges = []
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
    edges_to_consider = []
    for u in graph.graph:
        for v, w in graph.graph[u].items():
            if u < v:  # To avoid duplicates
                edges_to_consider.append((u, v, w))

    # Remove edges with probability remove_probability
    edges_to_remove = []
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
    adj = {vertex: set() for vertex in graph.graph}
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
    while len(visited) != len(graph.graph) and len(graph.graph) > 0:
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
        selected_nodes = set()
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

    def colormap(nx_graph, coloring):
        color_map = []
        nodes = nx_graph.nodes()
        cmap = plt.cm.Set2 if len(nodes) <= 50 else plt.cm.tab10
        for node in nodes:
            color_idx = coloring.get(str(node), 0)
            color = cmap(color_idx / max(1, len(set(coloring.values())) - 1))
            color_map.append(color)
        return color_map

    def nx2ax(
        nx_graph: "nx.Graph",
        ax,
        seed=42,
        show_weights: bool = True,
        pos=None,
        num_nodes_thresh=50,  # num of nodes to not use node labels, shrink nodes, ...
        coloring=None,
    ):
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

    # Test complete graph
    print("Testing complete graph:")
    graph = complete_(UndirectedGraph_(), n=8)
    print(f"Complete graph K_8: {len(graph)} vertices, {len(graph.get_edges())} edges")
    print(f"Expected edges for K_8: {8 * 7 // 2} (n*(n-1)/2)")

    # Test 20-node graph
    graph20 = twenty_(UndirectedGraph_())
    print("\nGraph Information for 20-node graph:")
    print(f"Number of vertices: {len(graph20)}")
    print(f"Number of edges: {len(graph20.get_edges())}")

    # Test Erdos-Renyi graph
    n_er = 30
    p_er = 0.2
    graph_er = erdos_renyi_(UndirectedGraph_(), n=n_er, p=p_er)
    print(f"\nErdos-Renyi graph (n={n_er}, p={p_er}):")
    print(f"Number of vertices: {len(graph_er)}")
    print(f"Number of edges: {len(graph_er.get_edges())}")
    expected_edges_er = int(p_er * n_er * (n_er - 1) / 2)
    print(f"Expected average edges: {expected_edges_er}")

    # Test Watts-Strogatz graph
    n_ws, k = 30, 6
    graph_ws = watts_strogatz_(UndirectedGraph_(), n=n_ws, k=k)
    print(f"\nWatts-Strogatz graph (n={n_ws}, k={k}):")
    print(f"Number of vertices: {len(graph_ws)}")
    print(f"Number of edges: {len(graph_ws.get_edges())}")

    # Test Barabási–Albert graph
    n_ba, m = 30, 3
    graph_ba = barabasi_albert_(UndirectedGraph_(), n=n_ba, m=m)
    print(f"\nBarabási–Albert graph (n={n_ba}, m={m}):")
    print(f"Number of vertices: {len(graph_ba)}")
    print(f"Number of edges: {len(graph_ba.get_edges())}")

    # Test small planar graph (maximal planar, no edge removal)
    n_planar_small = 20
    graph_planar_small = planar_(
        UndirectedGraph_(), n=n_planar_small, remove_probability=0.02
    )
    print(
        f"\nSmall planar graph (n={n_planar_small}: {len(graph_planar_small)} nodes, maximal):"
    )
    print(f"Number of vertices: {len(graph_planar_small)}")
    print(f"Number of edges: {len(graph_planar_small.get_edges())}")

    # Test RB model graphs
    n_rb = 30
    p1_rb = 0.3
    d_rb = 3

    graph_rb = rb_graph_(UndirectedGraph_(), n=n_rb, d=d_rb, p1=p1_rb)
    print(f"\nRB model graph (n={n_rb}, p1={p1_rb}):")
    print(f"Number of vertices: {len(graph_rb)}")
    print(f"Number of edges: {len(graph_rb.get_edges())}")
    expected_edges = int(p1_rb * n_rb * (n_rb - 1) / 2)
    print(f"Expected edges: {expected_edges}")

    if args.show and HAS_NX_MPL:
        # Figure 1: Complete graph and 20-node graph
        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8))

        # Complete graph
        ax1 = axes1[0]
        nx2ax(graph2nx(graph), ax1)
        ax1.set_title("Complete Graph K_8")
        ax1.axis("off")

        # 20-node graph
        ax2 = axes1[1]
        nx2ax(graph2nx(graph20), ax2)
        ax2.set_title("20-node Graph")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

        # Figure 2: Watts-Strogatz and Barabási–Albert graphs
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 8))

        # Erdos-Renyi graph
        graph_er = erdos_renyi_(UndirectedGraph_(), n=n_er, p=p_er)
        ax3 = axes2[0]
        nx2ax(graph2nx(graph_er), ax3)
        ax3.set_title(f"Erdos-Renyi (n={n_er}, p={p_er})")
        ax3.axis("off")

        # Watts-Strogatz graph
        ax3 = axes2[1]
        nx2ax(graph2nx(graph_ws), ax3)
        ax3.set_title(f"Watts-Strogatz (n={n_ws}, k={k})")
        ax3.axis("off")

        # Barabási–Albert graph
        ax4 = axes2[2]
        nx2ax(graph2nx(graph_ba), ax4)
        ax4.set_title(f"Barabási–Albert (n={n_ba}, m={m})")
        ax4.axis("off")

        plt.tight_layout()
        plt.show()

        # Figure 3: Large graphs without labels using nx2ax
        print("\nGenerating large graph for Figure 3...\n")

        # Create large Erdos-Renyi graph
        n_er_large, p_large = 100, 0.2
        graph_er_large = erdos_renyi_(UndirectedGraph_(), n=n_er_large, p=p_large)
        print(f"Large Erdos-Renyi graph (n={n_er_large}, p={p_large}):")
        print(f"Number of vertices: {len(graph_er_large)}")
        print(f"Number of edges: {len(graph_er_large.get_edges())}")

        # Create large Watts-Strogatz graph
        n_ws_large, k_large = 200, 8
        graph_ws_large = watts_strogatz_(UndirectedGraph_(), n=n_ws_large, k=k_large)
        print(f"\nLarge Watts-Strogatz graph (n={n_ws_large}, k={k_large}):")
        print(f"Number of vertices: {len(graph_ws_large)}")
        print(f"Number of edges: {len(graph_ws_large.get_edges())}")

        # Create large Barabási–Albert graph
        n_ba_large, m_large = 200, 3
        graph_ba_large = barabasi_albert_(UndirectedGraph_(), n=n_ba_large, m=m_large)
        print(f"\nLarge Barabási–Albert graph (n={n_ba_large}, m={m_large}):")
        print(f"Number of vertices: {len(graph_ba_large)}")
        print(f"Number of edges: {len(graph_ba_large.get_edges())}")

        # Create Figure 3
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 8))

        # Large Erdos-Renyi graph
        ax5 = axes3[0]
        nx2ax(graph2nx(graph_er_large), ax5)
        ax5.set_title(f"Large Erdos-Renyi (n={n_er_large}, p={p_large})")
        ax5.axis("off")

        # Large Watts-Strogatz graph
        ax5 = axes3[1]
        nx2ax(graph2nx(graph_ws_large), ax5)
        ax5.set_title(f"Large Watts-Strogatz (n={n_ws_large}, k={k_large})")
        ax5.axis("off")

        # Large Barabási–Albert graph
        ax6 = axes3[2]
        nx2ax(graph2nx(graph_ba_large), ax6)
        ax6.set_title(f"Large Barabási–Albert (n={n_ba_large}, m={m_large})")
        ax6.axis("off")

        plt.tight_layout()
        plt.show()

        # Figure 4: Planar graphs of different sizes
        print("\nGenerating planar graph for Figure 4...\n")

        # Create large planar graph with edge removal
        n_planar_large = 100
        graph_planar_large = planar_(
            UndirectedGraph_(), n=n_planar_large, remove_probability=0.3
        )
        print(
            f"Large planar graph (using n={n_planar_large}: {len(graph_planar_large)} nodes, with 30% edge removal):"
        )
        print(f"Number of vertices: {len(graph_planar_large)}")
        print(f"Number of edges: {len(graph_planar_large.get_edges())}")

        fig4, axes4 = plt.subplots(1, 2, figsize=(16, 8))

        # Small planar graph (with labels and weights)
        ax7 = axes4[0]
        g = graph2nx(graph_planar_small)
        try:
            nx2ax(g, ax7, pos=nx.planar_layout(g))
        except:
            print("not planar")
            nx2ax(g, ax7)
        ax7.set_title(f"Small Planar Graph (n={n_planar_small}, maximal)")
        ax7.axis("off")

        # Figure 5: Large planar graph (without labels)
        ax8 = axes4[1]
        g = graph2nx(graph_planar_large)
        try:
            nx2ax(g, ax8, pos=nx.planar_layout(g))
        except:
            print("not planar")
            nx2ax(g, ax8)
        ax8.set_title(
            f"Large Planar Graph (using n={n_planar_large}, 30% edges removed)"
        )
        ax8.axis("off")

        plt.tight_layout()
        plt.show()

        # Create Figure 5: Visualize RB model graphs
        print("\nGenerating RB model graph for Figure 5...\n")
        fig5, axes5 = plt.subplots(1, 2, figsize=(16, 8))
        ax9 = axes5[0]
        g_nx = graph2nx(graph_rb)
        try:
            nx2ax(g_nx, ax9, pos=nx.planar_layout(g_nx))
        except:
            print("not planar")
            nx2ax(g_nx, ax9)
        ax9.set_title(f"RB Model Graph (n={n_rb}, p1={p1_rb})")
        ax9.axis("off")

        ax10 = axes5[1]
        n_rb = 60
        p1_rb = 0.5
        d_rb = 3
        graph_rb_2 = rb_graph_(UndirectedGraph_(), n=n_rb, d=d_rb, p1=p1_rb)
        print(f"RB model graph (n={n_rb}, p1={p1_rb}):")
        print(f"Number of vertices: {len(graph_rb_2)}")
        print(f"Number of edges: {len(graph_rb_2.get_edges())}")
        expected_edges = int(p1_rb * n_rb * (n_rb - 1) / 2)
        print(f"Expected edges: {expected_edges}")
        g_nx = graph2nx(graph_rb_2)
        try:
            nx2ax(g_nx, ax10, pos=nx.planar_layout(g_nx))
        except:
            print("not planar")
            nx2ax(g_nx, ax10)
        ax10.set_title(f"RB Model Graph (n={n_rb}, p1={p1_rb})")
        ax10.axis("off")

        plt.tight_layout()
        plt.show()

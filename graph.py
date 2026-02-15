from collections import deque
import random
from typing import Any, Dict, List, Optional, Tuple, Set, Union, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import networkx as nx


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

    def nx2ax(nx_graph: "nx.Graph", ax, seed=42, show_weights: bool = True, pos=None):

        # Create a layout for the nodes if not provided
        if pos is None:
            pos = nx.spring_layout(nx_graph, seed=seed)

        # Plot graph
        nx.draw(
            nx_graph,
            pos,
            ax=ax,
            with_labels=True,
            node_color="lightgray",
            node_size=400 + max(len(str(node)) for node in list(nx_graph)) * 100,
            font_size=10,
            font_weight="bold",
            edge_color="gray",
            width=2,
            edgecolors="black",
        )

        # Draw edge labels (weights) if requested
        if show_weights:
            edge_labels = nx.get_edge_attributes(nx_graph, "weight")
            # Format weights to 1 decimal place for cleaner display
            formatted_edge_labels = {}
            for (u, v), weight in edge_labels.items():
                if isinstance(weight, int):
                    formatted_edge_labels[(u, v)] = f"{weight:.1f}"
                else:
                    formatted_edge_labels[(u, v)] = f"{weight}"
            nx.draw_networkx_edge_labels(
                nx_graph,
                pos,
                ax=ax,
                edge_labels=formatted_edge_labels,
                font_size=10,
                font_color="firebrick",
                bbox=dict(alpha=0.7, facecolor="white", edgecolor="none"),
            )

        return pos

except ImportError as e:
    print(f"Required GUI visualization libraries not found: {e}")
    print("\nPlease consider installing these libraries:")
    print("pip install matplotlib networkx")
    HAS_NX_MPL = False

if __name__ == "__main__":

    # Test complete graph
    print("Testing complete graph:")
    graph = complete_(UndirectedGraph_(), n=8)
    print(f"Complete graph K_8: {len(graph)} vertices, {len(graph.get_edges())} edges")
    print(f"Expected edges for K_8: {8 * 7 // 2} (n*(n-1)/2)")

    if HAS_NX_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # Complete graph
        ax1 = axes[0]
        nx2ax(graph2nx(graph), ax1, seed=42, show_weights=True)
        ax1.set_title("Complete Graph K_8")
        ax1.axis("off")

    # Test 20-node graph
    graph = twenty_(UndirectedGraph_())
    print("\nGraph Information for 20-node graph:")
    print(f"Number of vertices: {len(graph)}")
    print(f"Number of edges: {len(graph.get_edges())}")

    if HAS_NX_MPL:
        # 20-node graph
        ax2 = axes[1]
        nx2ax(graph2nx(graph), ax2, seed=42, show_weights=True)
        ax2.set_title("20-node Graph")
        ax2.axis("off")

    # Test Watts-Strogatz graph
    n, k = 24, 6
    graph = watts_strogatz_(UndirectedGraph_(), n=n, k=k)
    print(f"\nWatts-Strogatz graph (n={n}, k={k}):")
    print(f"Number of vertices: {len(graph)}")
    print(f"Number of edges: {len(graph.get_edges())}")

    if HAS_NX_MPL:
        # Watts-Strogatz graph
        ax3 = axes[2]
        nx2ax(graph2nx(graph), ax3, seed=42, show_weights=True)
        ax3.set_title(f"Watts-Strogatz (n={n}, k={k})")
        ax3.axis("off")

        plt.tight_layout()
        plt.show()

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

    def to_networkx(self) -> "nx.Graph":
        """
        Convert this UndirectedGraph to a networkx.Graph object.

        Returns:
            networkx.Graph: A networkx graph representation
        """
        import networkx as nx

        nx_graph = nx.Graph()

        # Add all vertices
        for vertex in self.graph:
            nx_graph.add_node(vertex)

        # Add all edges with weights
        for vertex in self.graph:
            for neighbor, weight in self.graph[vertex].items():
                # Add edge only once (undirected)
                if not nx_graph.has_edge(vertex, neighbor):
                    nx_graph.add_edge(vertex, neighbor, weight=weight)

        return nx_graph


T = TypeVar("T", bound=UndirectedGraph_)


def twenty_(graph: T, weighted: bool = True) -> T:

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
    if weighted:
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


if __name__ == "__main__":

    graph = twenty_(UndirectedGraph_())

    print(graph)
    print(graph.graph)

    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        # Convert our graph to networkx format
        nx_graph = graph.to_networkx()

        # Create a layout for the nodes
        pos = nx.spring_layout(nx_graph, seed=42)  # Fixed seed for reproducibility

        # Draw the graph
        plt.figure(figsize=(12, 8))

        # Draw nodes
        nx.draw_networkx_nodes(nx_graph, pos, node_size=500, node_color="lightblue")

        # Draw edges with weights
        nx.draw_networkx_edges(nx_graph, pos, width=2, alpha=0.7, edge_color="gray")

        # Draw node labels
        nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight="bold")

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(nx_graph, "weight")
        nx.draw_networkx_edge_labels(
            nx_graph, pos, edge_labels=edge_labels, font_size=8
        )

        # Display the plot
        plt.title("A 20-node Graph")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Also print some basic information
        print("\nGraph Information:")
        print(f"Number of vertices: {len(graph)}")
        print(f"Number of edges: {len(graph.get_edges())}")

    except ImportError as e:
        print(f"Required GUI visualization libraries not found: {e}")
        print("\nPlease consider installing these libraries:")
        print("pip install matplotlib networkx")

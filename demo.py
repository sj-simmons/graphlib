"""
Example script demonstrating how to use the to_networkx() method
to visualize UndirectedGraph objects with matplotlib subplots.
"""

import matplotlib.pyplot as plt
import networkx as nx
from graph import twenty_
from search import UndirectedGraph  # Import UndirectedGraph class from search.py


def visualize_graph():
    """
    Create and visualize a graph using matplotlib subplots.
    """
    # Create a graph using the provided twenty_() function
    # First create an empty UndirectedGraph instance
    graph = UndirectedGraph()
    graph_data = twenty_(graph, weighted=True)

    # Convert to networkx graph
    nx_graph = graph_data.to_networkx()

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()  # Flatten for easier indexing

    # 1. Basic graph layout
    ax1 = axes[0]
    pos = nx.spring_layout(nx_graph, seed=42)  # Fixed seed for reproducibility
    nx.draw(
        nx_graph,
        pos,
        ax=ax1,
        with_labels=True,
        node_color="lightblue",
        node_size=700,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    ax1.set_title("Basic Graph Visualization (Spring Layout)")

    # 2. Graph with edge weights
    ax2 = axes[1]
    nx.draw(
        nx_graph,
        pos,
        ax=ax2,
        with_labels=True,
        node_color="lightgreen",
        node_size=700,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    edge_labels = nx.get_edge_attributes(nx_graph, "weight")
    nx.draw_networkx_edge_labels(
        nx_graph, pos, ax=ax2, edge_labels=edge_labels, font_size=8
    )
    ax2.set_title("Graph with Edge Weights")

    # 3. Different layout (circular)
    ax3 = axes[2]
    pos_circular = nx.circular_layout(nx_graph)
    nx.draw(
        nx_graph,
        pos_circular,
        ax=ax3,
        with_labels=True,
        node_color="lightcoral",
        node_size=700,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    ax3.set_title("Circular Layout")

    # 4. Highlight shortest path using UCS
    ax4 = axes[3]
    nx.draw(
        nx_graph,
        pos,
        ax=ax4,
        with_labels=True,
        node_color="lightyellow",
        node_size=700,
        font_size=10,
        font_weight="bold",
        edge_color="lightgray",
    )

    # Find shortest path using UCS
    start_vertex = "N0"
    goal_vertex = "N19"
    path, total_weight = graph_data.ucs(
        start_vertex, goal_vertex
    )  # Use ucs method from UndirectedGraph class

    if path:
        # Highlight the path edges
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            nx_graph, pos, ax=ax4, edgelist=path_edges, edge_color="red", width=3
        )
        # Highlight the path nodes
        nx.draw_networkx_nodes(
            nx_graph, pos, ax=ax4, nodelist=path, node_color="red", node_size=800
        )
        ax4.set_title(f"UCS Shortest Path (Weight: {total_weight:.1f})")
    else:
        ax4.set_title("No Path Found")

    plt.suptitle(
        "UndirectedGraph Visualization Examples", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # Print some graph information
    print("Graph Information:")
    print(f"Number of vertices: {len(graph_data)}")
    print(f"Number of edges: {len(graph_data.get_edges())}")
    if path:
        print(f"Shortest path from {start_vertex} to {goal_vertex}: {path}")
        print(f"Total weight: {total_weight}")

    return nx_graph


def save_visualization():
    """
    Create and save a visualization to a file.
    """
    # Create a graph using the provided twenty_() function
    graph = UndirectedGraph()
    graph_data = twenty_(graph, weighted=True)

    nx_graph = graph_data.to_networkx()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left subplot: Basic visualization
    pos = nx.spring_layout(nx_graph, seed=42)
    nx.draw(
        nx_graph,
        pos,
        ax=ax1,
        with_labels=True,
        node_color="skyblue",
        node_size=600,
        font_size=9,
        edge_color="gray",
    )
    ax1.set_title("Graph Structure")

    # Right subplot: With edge weights
    nx.draw(
        nx_graph,
        pos,
        ax=ax2,
        with_labels=True,
        node_color="lightgreen",
        node_size=600,
        font_size=9,
        edge_color="gray",
    )
    edge_labels = nx.get_edge_attributes(nx_graph, "weight")
    nx.draw_networkx_edge_labels(
        nx_graph, pos, ax=ax2, edge_labels=edge_labels, font_size=7
    )
    ax2.set_title("Graph with Edge Weights")

    plt.tight_layout()
    plt.savefig("graph_visualization.png", dpi=150, bbox_inches="tight")
    print("Visualization saved to 'graph_visualization.png'")


def simple_visualization():
    """
    A simple one-plot visualization for quick viewing.
    """
    # Create a graph using the provided twenty_() function
    graph = UndirectedGraph()
    graph_data = twenty_(graph, weighted=True)

    nx_graph = graph_data.to_networkx()

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(nx_graph, seed=42)

    # Draw the graph
    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=700,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        width=1.5,
    )

    # Add edge weights
    edge_labels = nx.get_edge_attributes(nx_graph, "weight")
    nx.draw_networkx_edge_labels(
        nx_graph, pos, edge_labels=edge_labels, font_size=8, font_color="darkred"
    )

    plt.title("UndirectedGraph Visualization", fontsize=14, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    print("UndirectedGraph Visualization Examples")
    print("=" * 50)
    print("\nChoose an option:")
    print("1. Run full visualization with 4 subplots")
    print("2. Save visualization to file")
    print("3. Run simple visualization")

    try:
        choice = input("\nEnter your choice (1, 2, or 3): ").strip()

        if choice == "1":
            visualize_graph()
        elif choice == "2":
            save_visualization()
        elif choice == "3":
            simple_visualization()
        else:
            print("Invalid choice. Running full visualization by default.")
            visualize_graph()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Falling back to simple visualization...")
        simple_visualization()

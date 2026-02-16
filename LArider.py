from informed_search import UndirectedGraph
from graph import HAS_NX_MPL

cities = {
    "Northridge": [("Westwood", 25), ("Downtown", 30), ("Hillside", 15)],
    "Westwood": [
        ("Northridge", 25),
        ("Beverly", 8),
        ("Santa Monica", 12),
        ("Downtown", 20),
    ],
    "Santa Monica": [("Westwood", 12), ("Venice", 5), ("LAX", 10)],
    "Venice": [("Santa Monica", 5), ("Marina", 7), ("Culver", 9)],
    "Marina": [("Venice", 7), ("LAX", 8), ("Playa", 6)],
    "LAX": [("Santa Monica", 10), ("Marina", 8), ("Inglewood", 5), ("Hawthorne", 7)],
    "Beverly": [("Westwood", 8), ("Hollywood", 6), ("Century City", 4)],
    "Hollywood": [("Beverly", 6), ("Downtown", 12), ("Silver Lake", 8)],
    "Downtown": [
        ("Northridge", 30),
        ("Westwood", 20),
        ("Hollywood", 12),
        ("East LA", 8),
        ("Commerce", 10),
    ],
    "Century City": [("Beverly", 4), ("West LA", 7), ("Culver", 9)],
    "Culver": [("Century City", 9), ("Venice", 9), ("Inglewood", 6)],
    "Inglewood": [("Culver", 6), ("LAX", 5), ("Hawthorne", 4), ("Gardena", 8)],
    "Hawthorne": [("LAX", 7), ("Inglewood", 4), ("Gardena", 5), ("Torrance", 11)],
    "Gardena": [
        ("Inglewood", 8),
        ("Hawthorne", 5),
        ("Torrance", 7),
        ("Long Beach", 15),
    ],
    "Torrance": [("Hawthorne", 11), ("Gardena", 7), ("Long Beach", 10)],
    "Long Beach": [("Gardena", 15), ("Torrance", 10)],
    "East LA": [("Downtown", 8), ("Commerce", 6)],
    "Commerce": [("Downtown", 10), ("East LA", 6)],
    "Hillside": [("Northridge", 15)],
    "West LA": [("Century City", 7), ("Santa Monica", 9)],
    "Silver Lake": [("Hollywood", 8), ("Downtown", 9)],
    "Playa": [("Marina", 6), ("LAX", 9)],
}

distances_to_silver_lake = {
    "Silver Lake": 0.0,
    "Northridge": 18.11,
    "Westwood": 10.16,
    "Santa Monica": 13.48,
    "Venice": 13.4,
    "Marina": 12.73,
    "LAX": 12.78,
    "Beverly": 7.51,
    "Hollywood": 3.37,
    "Downtown": 2.83,
    "Century City": 8.66,
    "Culver": 8.49,
    "Inglewood": 9.86,
    "Hawthorne": 12.97,
    "Gardena": 13.88,
    "Torrance": 17.8,
    "Long Beach": 22.31,
    "East LA": 7.09,
    "Commerce": 8.68,
    "Hillside": 7.23,
    "West LA": 10.15,
    "Playa": 12.98,
}

cities_graph = UndirectedGraph()

for city in cities.keys():
    cities_graph.add_vertex(city)

for city1 in cities.keys():
    for city2, distance in cities[city1]:
        cities_graph.add_edge(city1, city2, distance)

if __name__ == "__main__":

    start_city = "Playa"
    path, distance = cities_graph.greedy(
        start_city, "Silver Lake", distances_to_silver_lake
    )
    print("Shortest route from", start_city, "to Silver Lake:")
    print(" ", " -> ".join(path))
    print("Total distance:", distance)

    if HAS_NX_MPL:

        import networkx as nx
        import matplotlib.pyplot as plt

        # Create a NetworkX graph
        G = nx.Graph()

        # Get all vertices and edges from the UndirectedGraph
        vertices = cities_graph.get_vertices()
        edges = cities_graph.get_edges()

        # Add vertices (nodes) to NetworkX graph
        G.add_nodes_from(vertices)

        # Add edges to NetworkX graph with weights
        for vertex1, vertex2, weight in edges:
            G.add_edge(vertex1, vertex2, weight=weight)

        # Set up the figure
        plt.figure(figsize=(16, 12))

        # Choose a layout algorithm
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Identify edges in the found path
        path_edges = []
        for i in range(len(path) - 1):
            # Sort to match edge representation (since graph is undirected)
            edge = tuple(sorted([path[i], path[i + 1]]))
            path_edges.append(edge)

        # Draw all edges first
        all_edges = list(G.edges())
        # Separate edges into path and non-path
        non_path_edges = [edge for edge in all_edges if edge not in path_edges]

        # Draw non-path edges
        if non_path_edges:
            non_path_weights = [G[u][v]["weight"] for u, v in non_path_edges]
            # Normalize weights for line width visualization
            min_weight = min(non_path_weights) if non_path_weights else 1
            max_weight = max(non_path_weights) if non_path_weights else 1
            if max_weight > min_weight:
                line_widths = [
                    (w - min_weight) / (max_weight - min_weight) * 3 + 1
                    for w in non_path_weights
                ]
            else:
                line_widths = [2] * len(non_path_weights)

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=non_path_edges,
                width=line_widths,
                edge_color="gray",
                alpha=0.7,
                style="solid",
            )

        # Draw path edges with highlight
        if path_edges:
            path_weights = [G[u][v]["weight"] for u, v in path_edges]
            # Make path edges thicker and colored
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=path_edges,
                width=5,
                edge_color="firebrick",
                alpha=0.9,
                style="solid",
            )

        # Draw nodes as invisible points (just for positioning)
        # We'll use bbox for the visual representation
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color="none",  # Invisible
            node_size=0,  # No size
            edgecolors="none",  # No border
        )

        # Draw labels with a bounding box - This creates the "rectangular box" look
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=10,
            font_weight="bold",
            font_color="darkblue",
            bbox=dict(
                facecolor="lightblue",
                edgecolor="darkblue",
                boxstyle="round,pad=0.5",  # Use 'square' for sharp corners
                linewidth=2,
                alpha=0.9,
            ),
        )

        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=9,
            font_color="red",
            font_weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="red", alpha=0.8
            ),
        )

        # Customize the plot
        plt.title(
            f"LA Cities Transportation Network\nHighlighted path from {start_city} to Silver Lake (in red)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.axis("off")

        # Add a legend for edge weights
        plt.text(
            0.02,
            0.02,
            f"Edge Thickness ≈ Distance\nMin Distance: {min_weight} units\nMax Distance: {max_weight} units",
            transform=plt.gca().transAxes,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightyellow",
                edgecolor="gold",
                alpha=0.9,
                linewidth=2,
            ),
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

        # Print some graph statistics
        print("\n" + "=" * 50)
        print("GRAPH STATISTICS")
        print("=" * 50)
        print(f"Number of cities (vertices): {G.number_of_nodes()}")
        print(f"Number of roads (edges): {G.number_of_edges()}")
        print(
            f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}"
        )

        # Calculate total road length
        total_distance = sum(weight for _, _, weight in cities_graph.get_edges())
        print(f"Total road length: {total_distance} units")

        # Find the city with most connections
        degrees = dict(G.degree())
        max_degree_city = max(degrees, key=degrees.get)
        print(
            f"Most connected city: {max_degree_city} ({degrees[max_degree_city]} connections)"
        )

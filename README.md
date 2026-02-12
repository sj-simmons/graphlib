# Graph Search and Visualization Project

This project implements an undirected graph data structure with search algorithms, tree algorithms, and visualization capabilities.

## Files Overview

### graph.py

**UndirectedGraph_ class** - Base class for undirected graphs
- `__init__()` - Initialize empty graph
- `add_vertex(vertex)` - Add a vertex to the graph
- `add_edge(vertex1, vertex2, weight=1)` - Add undirected edge between vertices
- `has_vertex(vertex)` - Check if vertex exists
- `has_edge(vertex1, vertex2)` - Check if edge exists
- `get_neighbors(vertex)` - Get all neighbors of a vertex
- `get_vertices()` - Get all vertices in the graph
- `get_edges()` - Get all edges with weights
- `get_weight(vertex1, vertex2)` - Get weight of edge between vertices
- `is_empty()` - Check if graph is empty
- `__str__()` - String representation
- `__repr__()` - Representation
- `__len__()` - Number of vertices
- `to_networkx()` - Convert to networkx.Graph object

**Helper function:**
- `twenty_(graph, weighted=True)` - Create a 20-node test graph with random weights

### uninformed_search.py

**UndirectedGraph class** - Extends UndirectedGraph_ with uninformed search algorithms
- Inherits all methods from UndirectedGraph_
- `dfs(start_vertex, goal_vertex)` - Depth-First Search (iterative)
- `bfs(start_vertex, goal_vertex)` - Breadth-First Search
- `ucs_(start_vertex, goal_vertex)` - Uniform Cost Search (basic implementation)
- `ucs(start_vertex, goal_vertex)` - Uniform Cost Search (optimized implementation)

### informed_search.py

**UndirectedGraph class** - Extends UndirectedGraph_ with informed search algorithms
- Inherits all methods from UndirectedGraph_
- `greedy(start_vertex, goal_vertex, heuristic)` - Greedy best-first search (similar to greedy_search in uninformed_search.py)

### tree.py

**UndirectedGraph class** - Extends UndirectedGraph_ with tree algorithms
- Inherits all methods from UndirectedGraph_
- `dfs_tree(start_vertex)` - Generate a Depth-First Search tree
- `bfs_tree(start_vertex)` - Generate a Breadth-First Search tree
- `prim_mst(start_vertex=None)` - Find Minimum Spanning Tree using Prim's algorithm
- `spt(start_vertex=None)` - Find Shortest Path Tree using Dijkstra's algorithm

### demo.py

**Visualization functions** - Demonstrate graph visualization using matplotlib and networkx
- `visualize_graph()` - Create 2x2 subplot visualization with different layouts
- `save_visualization()` - Save visualization to file
- `simple_visualization()` - Simple one-plot visualization

## Usage

Run `demo.py` to see interactive visualization options:
```bash
python demo.py
```

Run `uninformed_search.py` directly to see search algorithm comparisons:
```bash
python uninformed_search.py
```

Run `tree.py` directly to see tree algorithm demonstrations:
```bash
python tree.py
```

Run `graph.py` directly to see basic graph visualization:
```bash
python graph.py
```

## Dependencies

- matplotlib
- networkx

Install with:
```bash
pip install matplotlib networkx
```

## Example

The project includes a 20-node test graph with weighted edges to demonstrate different algorithm behaviors:
- DFS tends to explore deep paths
- BFS finds shortest path by number of edges
- UCS finds minimum weight path
- Prim's algorithm finds Minimum Spanning Tree
- Dijkstra's algorithm finds Shortest Path Tree
- Greedy search uses heuristics to guide search

## Class Hierarchy

All `UndirectedGraph` classes in different files inherit from the base `UndirectedGraph_` class in `graph.py`:
- `uninformed_search.UndirectedGraph` adds search algorithms
- `informed_search.UndirectedGraph` adds informed search
- `tree.UndirectedGraph` adds tree algorithms

Each extension provides specialized functionality while maintaining the core graph operations.

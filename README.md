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

**Graph generation functions:**
- `complete_(graph, n=10, weight_range=(1, 10), seed=None)` - Generate a complete graph K_n
- `watts_strogatz_(graph, n=20, k=4, beta=0.3, weight_range=(1, 10), seed=None)` - Generate a Watts-Strogatz small-world graph
- `twenty_(graph, weighted=True, more_edges=True)` - Create a 20-node test graph with multiple paths

**Visualization functions:**
- `graph2nx(graph)` - Convert UndirectedGraph to networkx.Graph object
- `nx2ax(nx_graph, ax, seed=42, show_weights=True, pos=None)` - Draw networkx graph on matplotlib axis
- `HAS_NX_MPL` - Boolean flag indicating if matplotlib and networkx are available

### uninformed_search.py

**UndirectedGraph class** - Extends UndirectedGraph_ with uninformed search algorithms
- Inherits all methods from UndirectedGraph_
- `dfs(start_vertex, goal_vertex)` - Depth-First Search (iterative)
- `bfs(start_vertex, goal_vertex)` - Breadth-First Search
- `ucs_(start_vertex, goal_vertex)` - Uniform Cost Search (basic implementation)
- `ucs(start_vertex, goal_vertex)` - Uniform Cost Search (optimized implementation)
- `all_simple_paths(start_vertex, goal_vertex)` - Find all simple paths between vertices

### informed_search.py

**UndirectedGraph class** - Extends UndirectedGraph_ with informed search algorithms
- Inherits all methods from UndirectedGraph_
- `greedy(start_vertex, goal_vertex, heuristic)` - Greedy best-first search using heuristic estimates
- `astar(start_vertex, goal_vertex, heuristic)` - A* search algorithm using f(n) = g(n) + h(n)

### tree.py

**UndirectedGraph class** - Extends UndirectedGraph_ with tree algorithms
- Inherits all methods from UndirectedGraph_
- `dfs_tree(start_vertex)` - Generate a Depth-First Search tree
- `bfs_tree(start_vertex)` - Generate a Breadth-First Search tree
- `prim_mst(start_vertex=None)` - Find Minimum Spanning Tree using Prim's algorithm
- `spt(start_vertex=None)` - Find Shortest Path Tree using Dijkstra's algorithm

### LArider.py

**LA Cities Transportation Network** - A practical example using the graph search algorithms
- Creates a graph of LA cities with real-world distances
- Demonstrates greedy search with heuristic distances to Silver Lake
- Includes visualization of the transportation network
- Shows path finding in a real-world scenario

### tests.py

**Test suite** - Automated tests for graph algorithms
- Tests `all_simple_paths()` method against networkx implementation
- Validates graph generation functions (complete, Watts-Strogatz, 20-node)
- Ensures algorithm correctness through comparison with established library

## Usage

Run `uninformed_search.py` directly to see search algorithm comparisons:
```bash
python uninformed_search.py
```

Run `informed_search.py` directly to see informed search algorithm comparisons:
```bash
python informed_search.py
```

Run `tree.py` directly to see tree algorithm demonstrations:
```bash
python tree.py
```

Run `graph.py` directly to see basic graph visualization:
```bash
python graph.py
```

Run `LArider.py` to see the LA cities transportation network example:
```bash
python LArider.py
```

## Testing

Run the test suite to verify algorithm correctness:
```bash
python tests.py
```

The tests compare the implementation against networkx for validation.

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
- Greedy search uses heuristic estimates to guide search
- A* search combines actual cost and heuristic estimates
- Prim's algorithm finds Minimum Spanning Tree
- Dijkstra's algorithm finds Shortest Path Tree

## Class Hierarchy

All `UndirectedGraph` classes in different files inherit from the base `UndirectedGraph_` class in `graph.py`:
- `uninformed_search.UndirectedGraph` adds uninformed search algorithms (DFS, BFS, UCS)
- `informed_search.UndirectedGraph` adds informed search algorithms (Greedy, A*)
- `tree.UndirectedGraph` adds tree algorithms (DFS tree, BFS tree, Prim's MST, Shortest Path Tree)

Each extension provides specialized functionality while maintaining the core graph operations.

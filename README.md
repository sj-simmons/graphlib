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
- `barabasi_albert_(graph, n=100, m=2, weight_range=(1, 10), seed=None)` - Generate a Barabási–Albert scale-free network graph
- `planar_(graph, n=20, remove_probability=0.0, weight_range=(1, 10), seed=None)` - Generate the dual of a maximal planar graph using Delaunay triangulation
- `rb_graph_(graph, n=20, d=3, p1=0.5, p2=0.5, seed=None)` - Generate a random graph using the RB (random with balanced structure) model
- `erdos_renyi_(graph, n=20, p=0.5, weight_range=(1, 10), seed=None)` - Generate a random Erdos-Renyi graph G(n, p)
- `twenty_(graph, weighted=True, more_edges=True)` - Create a 20-node test graph with multiple paths

**Visualization functions:**
- `graph2nx(graph)` - Convert UndirectedGraph to networkx.Graph object
- `colormap(nx_graph, coloring)` - Helper function to create a color map for nodes based on a coloring dictionary
- `nx2ax(nx_graph, ax, seed=42, show_weights=True, pos=None, num_nodes_thresh=50, coloring=None)` - Draw networkx graph on matplotlib axis with optional node coloring and adaptive node sizing
- `HAS_NX_MPL` - Boolean flag indicating if matplotlib and networkx are available

### uninformed_search.py

**UndirectedGraph class** - Extends UndirectedGraph_ with uninformed search algorithms
- Inherits all methods from UndirectedGraph_
- `dfs(start_vertex: Any, goal_vertex: Any) -> Tuple[Optional[List[Any]], Union[int, float]]` - Depth-First Search (iterative) that returns the path and total weight. Returns `(None, 0)` if start or goal vertex doesn't exist.
- `bfs(start_vertex: Any, goal_vertex: Any) -> Tuple[Optional[List[Any]], Union[int, float]]` - Breadth-First Search that returns the shortest path (by number of edges) and total weight. Returns `(None, 0)` if start or goal vertex doesn't exist.
- `ucs_(start_vertex: Any, goal_vertex: Any) -> Tuple[Optional[List[Any]], Union[int, float]]` - Basic Uniform Cost Search (Dijkstra's algorithm for non-negative weights) that returns the minimum weight path and total weight. Returns `(None, 0)` if start or goal vertex doesn't exist.
- `ucs(start_vertex: Any, goal_vertex: Any) -> Tuple[Optional[List[Any]], Union[int, float]]` - Optimized Uniform Cost Search using priority queue with additional visited set and cost tracking. Returns `(None, 0)` if start or goal vertex doesn't exist.
- `all_simple_paths(start_vertex: Any, goal_vertex: Any) -> List[List[Any]]` - Find all simple paths (paths without cycles) between vertices. Returns empty list if no paths exist or if either vertex is not in the graph.

**Algorithm details:**
- All search methods return a tuple: `(path, total_weight)` where `path` is a list of vertices and `total_weight` is the sum of edge weights along the path.
- `dfs` uses an iterative stack implementation (not recursive) to avoid recursion depth limits.
- `bfs` uses a `deque` for efficient queue operations.
- Both `ucs_` and `ucs` use priority queues (`heapq`) for Uniform Cost Search, but `ucs` includes additional optimizations with visited set and cost tracking.
- `all_simple_paths` finds ALL simple paths (no cycles) using iterative DFS and can grow exponentially with graph size.

**Search algorithm return values:**
- If path exists: `([vertex1, vertex2, ..., vertexN], total_weight)`
- If no path exists or invalid input: `(None, 0)` for search methods, `[]` for `all_simple_paths`

### informed_search.py

**UndirectedGraph class** - Extends `uninformed_search.UndirectedGraph` with informed search algorithms
- Inherits all methods from `uninformed_search.UndirectedGraph`
- `greedy(start_vertex, goal_vertex, heuristic)` - Greedy best-first search using heuristic estimates. Always expands the node that appears closest to the goal according to the provided heuristic, without considering the cost incurred so far. Returns a tuple `(path, total_weight)` where `path` is a list of vertices from start to goal (or `None` if no path exists) and `total_weight` is the sum of edge weights (or 0 if no path).
- `astar(start_vertex, goal_vertex, heuristic)` - A* search algorithm using f(n) = g(n) + h(n), where g(n) is the actual cost from start to node n, and h(n) is the heuristic estimate from node n to the goal. Returns a tuple `(path, total_weight)` where `path` is a list of vertices from start to goal (or `None` if no path exists) and `total_weight` is the sum of edge weights (or 0 if no path).
- `astar2(start_vertex, goal_vertex, heuristic)` - A* search with additional efficiency metrics tracking. Returns a tuple `(path, total_weight, efficiency_metrics)` where:
  - `path`: List of vertices from start to goal, or `None` if no path exists
  - `total_weight`: Sum of edge weights along the found path, or 0 if no path
  - `efficiency_metrics`: Dictionary with keys:
    * `'nodes_expanded'`: number of nodes popped from the frontier
    * `'nodes_visited'`: number of unique nodes visited
    * `'frontier_max_size'`: maximum size of the frontier during search
    * `'path_length'`: number of vertices in the found path (0 if none)

**Algorithm details:**
- All methods require a `heuristic` dictionary that maps vertices to estimated distances to the goal
- For vertices not in the heuristic dictionary, `float("inf")` is used as default
- The `heuristic` parameter is a dictionary, not a function
- `astar2` is identical to `astar` but collects and returns additional search efficiency metrics
- Both `astar` and `astar2` use the formula f(n) = g(n) + h(n) where:
  - g(n): actual cost from start to node n
  - h(n): heuristic estimate from node n to goal
- All methods return `(None, 0)` (or `(None, 0, efficiency_metrics)` for `astar2`) if start or goal vertex doesn't exist

**Example usage:**
```python
# Create a heuristic dictionary for all vertices
heuristic = {}
for vertex in graph.get_vertices():
    # Simple heuristic: difference in vertex indices (just for example)
    heuristic[vertex] = abs(vertex - goal_vertex)

# Run greedy search
path, weight = graph.greedy(start_vertex, goal_vertex, heuristic)

# Run A* with efficiency metrics
path, weight, metrics = graph.astar2(start_vertex, goal_vertex, heuristic)
print(f"Nodes expanded: {metrics['nodes_expanded']}")
```

**Main demo features:**
When running `python informed_search.py`, the program:
- Creates Barabási-Albert scale-free network graphs with 24 and 300 nodes
- Defines start and goal vertices
- Creates a simple heuristic based on node index differences multiplied by 5
- Runs greedy search, A* search with heuristic, and A* with zero heuristic (equivalent to UCS)
- Compares results with uninformed algorithms (DFS, BFS) from the parent class
- Displays visualizations comparing paths found by different algorithms (for smaller graphs)
- Prints detailed efficiency metrics comparing A* with zero vs non-zero heuristics

### tree.py

**UndirectedGraph class** - Extends `graph.UndirectedGraph_` with tree algorithms
- Inherits all methods from `graph.UndirectedGraph_`
- `dfs_tree(start_vertex)` - Generate a Depth-First Search tree as a new `UndirectedGraph` object. Raises `ValueError` if `start_vertex` is not in the graph.
- `bfs_tree(start_vertex)` - Generate a Breadth-First Search tree as a new `UndirectedGraph` object. Raises `ValueError` if `start_vertex` is not in the graph.
- `prim_mst(start_vertex=None)` - Find Minimum Spanning Tree using Prim's algorithm. Returns a new `UndirectedGraph` object representing the MST. If `start_vertex` is `None`, uses the first vertex from `get_vertices()`. Raises `ValueError` if the graph is empty or if `start_vertex` is provided and not in the graph. Note: For a disconnected graph, the MST will only include vertices reachable from the start vertex.
- `spt(start_vertex=None)` - Find Shortest Path Tree (SPT) from `start_vertex` to all reachable vertices using Dijkstra's algorithm. Returns a new `UndirectedGraph` object representing the SPT. If `start_vertex` is `None`, uses the first vertex from `get_vertices()`. Raises `ValueError` if the graph is empty or if `start_vertex` is provided and not in the graph. Note: For a disconnected graph, the SPT will only include vertices reachable from the start vertex.

### csp.py

**Constraint class** - Abstract base class for constraints in Constraint Satisfaction Problems
- `__init__(variables: List[V])` - Initialize a constraint with the list of variables it involves
- `is_satisfied(assignment: Dict[V, D]) -> bool` - Abstract method to check if the constraint is satisfied given a partial or complete assignment. Must be implemented by subclasses
- `get_conflicted_variables(assignment: Dict[V, D]) -> Set[V]` - Abstract method to return the set of variables involved in constraint violations. Must be implemented by subclasses

**CSP class** - Generic Constraint Satisfaction Problem solver with backtracking and heuristics
- `__init__() -> None` - Initialize an empty CSP with no variables or constraints
- `add_variable(variable: V, domain: List[D]) -> None` - Add a variable to the CSP with its domain of possible values
- `add_constraint(constraint: Constraint[V, D]) -> None` - Add a constraint to the CSP and connect it to all variables it involves
- `get_constraints_for_variable(variable: V) -> List[Constraint[V, D]]` - Get all constraints that involve a specific variable
- `is_consistent(variable: V, value: D, assignment: Dict[V, D]) -> bool` - Check if assigning a value to a variable is consistent with all constraints given the current partial assignment
- `is_complete(assignment: Dict[V, D]) -> bool` - Check if an assignment covers all variables in the CSP
- `select_unassigned_variable(assignment: Dict[V, D]) -> Optional[V]` - Select the next variable to assign using the Minimum Remaining Values (MRV) heuristic
- `order_domain_values(variable: V, assignment: Dict[V, D]) -> List[D]` - Order domain values for a variable using the Least Constraining Value (LCV) heuristic
- `forward_check(variable: V, value: D, assignment: Dict[V, D]) -> bool` - Forward checking algorithm: remove inconsistent values from future variables' domains after an assignment. Returns False if any domain becomes empty
- `solve(use_forward_checking: bool = False) -> Optional[Dict[V, D]]` - Main solving method using backtracking search with optional forward checking. Returns a solution assignment or None if no solution exists
- `reset_domains() -> None` - Reset all variable domains to their original values
- `get_all_solutions(limit: Optional[int] = None) -> List[Dict[V, D]]` - Find all solutions to the CSP, optionally limited to a maximum number
- `get_backtrack_count() -> int` - Return the number of backtracks performed during the last solve attempt

**Search algorithm details:**
- The CSP solver implements a generic backtracking search algorithm
- Uses Minimum Remaining Values (MRV) heuristic for variable selection by default
- Uses Least Constraining Value (LCV) heuristic for value ordering by default
- Optional forward checking can be enabled to prune inconsistent values from future variables' domains
- Maintains a backtrack counter to measure search complexity
- Can find single solutions or enumerate all solutions (with optional limit)
- Generic implementation works with any variable type (V) and domain value type (D)
- Constraints must inherit from the `Constraint` abstract base class and implement the required methods

**Usage example:**
```python
from csp import CSP, Constraint

# Define a custom constraint
class MyConstraint(Constraint[str, int]):
    def __init__(self, var1: str, var2: str):
        super().__init__([var1, var2])
        self.var1 = var1
        self.var2 = var2
    
    def is_satisfied(self, assignment: Dict[str, int]) -> bool:
        if self.var1 not in assignment or self.var2 not in assignment:
            return True
        return assignment[self.var1] != assignment[self.var2]
    
    def get_conflicted_variables(self, assignment: Dict[str, int]) -> Set[str]:
        if self.var1 in assignment and self.var2 in assignment:
            if assignment[self.var1] == assignment[self.var2]:
                return {self.var1, self.var2}
        return set()

# Create and solve a CSP
csp = CSP[str, int]()
csp.add_variable("A", [1, 2, 3])
csp.add_variable("B", [1, 2, 3])
csp.add_constraint(MyConstraint("A", "B"))

solution = csp.solve(use_forward_checking=True)
if solution:
    print(f"Solution: {solution}, Backtracks: {csp.get_backtrack_count()}")
```

**Note:** The `csp.py` module provides a foundation for implementing various CSP problems. See `coloring.py` and `nqueens.py` for examples of concrete CSP implementations.

### coloring.py

**Graph Coloring Constraint Satisfaction Problem** - Implementation of graph coloring as a CSP

**GraphColoringConstraint class** - Constraint for ensuring adjacent nodes have different colors
- `__init__(node1: str, node2: str)` - Initialize a constraint between two adjacent nodes in the graph
- `is_satisfied(assignment: Dict[str, int]) -> bool` - Check if the constraint is satisfied given the current assignment. Returns True if either node is unassigned or if they have different colors
- `get_conflicted_variables(assignment: Dict[str, int]) -> Set[str]` - Return the set of nodes that are involved in conflicts (both assigned and have the same color)

**GraphColoringCSP class** - Graph coloring problem formulated as a CSP
- `__init__(nodes: List[str], edges: List[tuple], num_colors: int)` - Initialize a graph coloring CSP
  - Variables: graph nodes (strings)
  - Domains: colors 0 to num_colors-1 for each node
  - Constraints: `GraphColoringConstraint` for each edge, ensuring adjacent nodes have different colors

**Usage example:**
```python
from coloring import GraphColoringCSP

# Define a graph with nodes and edges
nodes = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'A')]

# Create and solve the graph coloring problem with 3 colors
csp = GraphColoringCSP(nodes, edges, 3)
solution = csp.solve(use_forward_checking=True)

if solution:
    print(f"Graph coloring solution found:")
    for node, color in solution.items():
        print(f"  Node {node}: Color {color}")
    print(f"Backtracks required: {csp.get_backtrack_count()}")
    
    # Check if we can color with fewer colors
    for k in range(2, 4):
        csp_k = GraphColoringCSP(nodes, edges, k)
        if csp_k.solve():
            print(f"Graph is {k}-colorable")
            break
else:
    print("No valid coloring found with 3 colors")
```

**Problem formulation:**
- Variables represent graph nodes (vertices)
- Each variable's domain is the set of available colors (0 to num_colors-1)
- Binary constraints ensure that adjacent nodes (connected by an edge) have different colors
- The problem is to find an assignment of colors to all nodes that satisfies all constraints
- Graph coloring is NP-complete and serves as a benchmark for CSP algorithms

**Algorithm features:**
- Uses the generic backtracking search from the base CSP class in `csp.py`
- Supports all CSP heuristics: Minimum Remaining Values (MRV), Least Constraining Value (LCV), and forward checking
- Can find a single valid coloring or enumerate all possible colorings
- The implementation demonstrates how to map a graph problem to a CSP formulation

**Graph coloring applications:**
- Map coloring (Four Color Theorem)
- Register allocation in compiler design
- Scheduling problems
- Frequency assignment in wireless networks
- Sudoku puzzles (as a graph coloring problem)

**Example:**
For a triangle graph (3 nodes all connected), at least 3 colors are needed:
```
Nodes: ['A', 'B', 'C']
Edges: [('A', 'B'), ('B', 'C'), ('C', 'A')]
Solution with 3 colors: {'A': 0, 'B': 1, 'C': 2}
```

**Note:** The `coloring.py` module shows how to use the generic CSP framework from `csp.py` to solve a classic constraint satisfaction problem. For more advanced coloring algorithms, see `coloring2.py`, `coloring_gnn.py`, and `coloring_pinn.py`.

### nqueens.py

**N-Queens Constraint Satisfaction Problem** - Implementation of the classic N-Queens puzzle as a CSP

**NQueensConstraint class** - Specialized constraint for the N-Queens problem
- `__init__(variables: List[int])` - Initialize constraint for a list of variables (queens)
- `is_satisfied(assignment: Dict[int, int]) -> bool` - Check if the constraint is satisfied given the current assignment. Returns True if no two queens attack each other
- `get_conflicted_variables(assignment: Dict[int, int]) -> Set[int]` - Return the set of queens (variables) that are involved in conflicts (attacking each other)

**NQueensCSP class** - Specialized CSP for the N-Queens problem
- `__init__(n: int)` - Initialize an N-Queens CSP with n queens on an n×n board
  - Variables: queens numbered 0 to n-1 (representing rows)
  - Domains: column positions 0 to n-1 for each queen
  - Constraints: No two queens can be in the same row, column, or diagonal

**Functions:**
- `show_solution(solution: Dict[int, int]) -> None` - Display the N-Queens solution in a readable format. Prints a chessboard representation with queens marked

**Problem formulation:**
- Each variable represents a queen in a specific row (row i has queen i)
- Each queen's domain is the set of columns (0 to n-1) where it can be placed
- Binary constraints ensure no two queens attack each other:
  - No two queens in the same column
  - No two queens on the same diagonal (both main and anti-diagonals)
- The CSP uses the generic CSP solver from `csp.py` with specialized constraints

**Usage example:**
```python
from nqueens import NQueensCSP, show_solution

# Create and solve the 8-Queens problem
csp = NQueensCSP(8)
solution = csp.solve(use_forward_checking=True)

if solution:
    print(f"Found a solution for {len(solution)}-Queens:")
    show_solution(solution)
    print(f"Backtracks required: {csp.get_backtrack_count()}")
else:
    print(f"No solution found for {n}-Queens")

# Find all solutions (warning: exponential growth!)
all_solutions = csp.get_all_solutions()
print(f"Total solutions: {len(all_solutions)}")
```

**Algorithm features:**
- Uses the generic backtracking search from the base CSP class
- Supports all CSP heuristics (MRV, LCV, forward checking)
- Can find a single solution or enumerate all solutions
- The constraint implementation efficiently checks for diagonal conflicts using the property: |row1 - row2| == |col1 - col2|
- The `show_solution` function provides visual representation of the board

**N-Queens problem details:**
- The classic N-Queens problem places N queens on an N×N chessboard so that no two queens threaten each other
- A solution requires that no two queens share the same row, column, or diagonal
- The number of solutions grows exponentially with N, making it a good benchmark for CSP algorithms
- The implementation demonstrates how to map a real-world problem to a CSP formulation

**Example output for 4-Queens:**
```
Solution found for 4-Queens:
. Q . .
. . . Q
Q . . .
. . Q .

Row 0: column 1
Row 1: column 3  
Row 2: column 0
Row 3: column 2
```

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

The program will:
- Create a Watts-Strogatz graph with 30 nodes
- Define start and goal vertices
- Run DFS, BFS, and UCS algorithms
- Display visualizations comparing the paths found by each algorithm (if matplotlib and networkx are available)
- Print path information including number of steps and total weight for each algorithm

**Search algorithm output format:**
```
DFS: 5 steps, total weight 12.5
BFS: 3 steps, total weight 15.2  
UCS: 4 steps, total weight 10.8
```

Run `informed_search.py` directly to see informed search algorithm comparisons:
```bash
python informed_search.py
```

The program will:
- Create Barabási-Albert scale-free network graphs with 24 and 300 nodes
- Define start and goal vertices  
- Create a simple heuristic based on node index differences
- Run Greedy search, A* search, and A* with efficiency metrics
- Compare results with uninformed algorithms (DFS, BFS) and A* with zero heuristic (equivalent to UCS)
- Display visualizations comparing paths found by different algorithms for the 24-node graph (if matplotlib and networkx are available)
- Print detailed efficiency metrics comparing A* with zero vs non-zero heuristics

**Example output includes:**
- Path length and total weight for each algorithm: DFS, BFS, A* (zero heuristic), Greedy, A* (non-zero heuristic)
- Efficiency metrics: nodes expanded, nodes visited, frontier size
- Comparison of heuristic performance vs zero heuristic
- For the 300-node graph, only textual output is provided (no visualization)

Run `tree.py` directly to see tree algorithm demonstrations:
```bash
python tree.py
```

The program will:
- Create a 20-node graph with multiple paths using the `twenty_()` function
- Generate DFS and BFS trees from a starting vertex
- Generate a Minimum Spanning Tree (MST) using Prim's algorithm
- Generate a Shortest Path Tree (SPT) using Dijkstra's algorithm
- Display visualizations comparing these trees (if matplotlib and networkx are available)

Run `graph.py` directly to see demonstrations of various graph types and their visualizations:
```bash
python graph.py
```

The program will generate and display:
- Basic graphs (complete, 20-node)
- Random graph models (Erdos-Renyi, Watts-Strogatz, Barabási–Albert)
- Large random graphs
- Planar graphs
- RB model graphs

Use the `-show` flag to suppress display (e.g., `python graph.py -show` will not show the graphs). Note: The `-show` flag actually toggles showing graphs - when provided, graphs are NOT shown.

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
- DFS tends to explore deep paths, returning a path and its total weight
- BFS finds shortest path by number of edges, returning a path and its total weight  
- UCS finds minimum weight path, returning a path and its total weight
- Greedy search uses heuristic estimates to guide search
- A* search combines actual cost and heuristic estimates
- A*2 search provides efficiency metrics in addition to path finding
- Prim's algorithm finds Minimum Spanning Tree
- Dijkstra's algorithm finds Shortest Path Tree

**Example usage:**
```python
graph = UndirectedGraph()
# ... add vertices and edges ...
path, weight = graph.dfs(start_vertex, goal_vertex)
if path:
    print(f"Path: {path}, Total weight: {weight}")
else:
    print("No path found")
```

## Class Hierarchy

- `graph.UndirectedGraph_` (base class with core graph operations)
- `uninformed_search.UndirectedGraph` extends `graph.UndirectedGraph_` (adds DFS, BFS, UCS search algorithms)
- `informed_search.UndirectedGraph` extends `uninformed_search.UndirectedGraph` (adds Greedy, A* informed search algorithms)
- `tree.UndirectedGraph` extends `graph.UndirectedGraph_` (adds tree algorithms: DFS tree, BFS tree, Prim's MST, Shortest Path Tree)

Each extension provides specialized functionality while maintaining the core graph operations. Note that `tree.py` imports directly from `graph.UndirectedGraph_`, not from `uninformed_search.UndirectedGraph`.

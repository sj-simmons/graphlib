from graph import twenty_, graph2nx, watts_strogatz_, complete_
from uninformed_search import UndirectedGraph
import networkx
import math

# Test all_simple_paths

graph = twenty_(UndirectedGraph(), weighted=False)
nodes = list(graph.graph.keys())
for i, start in enumerate(nodes):
    for goal in nodes[i + 1 :]:
        graphlibpaths = graph.all_simple_paths(start, goal)
        joined_paths = ["".join(path) for path in graphlibpaths]
        assert len(joined_paths) == len(set(joined_paths))
        numgraphlibpaths = len(graphlibpaths)
        numnxgraphpaths = len(
            list(networkx.all_simple_paths(graph2nx(graph), source=start, target=goal))
        )
        assert (
            numgraphlibpaths == numnxgraphpaths
        ), f"graphlib found {len(numgraphlibpaths)} paths, networkx found {len(numnxgraphpaths)}"
        # print(numgraphlibpaths)

graph = watts_strogatz_(UndirectedGraph(), n=10, k=8)
nodes = list(graph.graph.keys())
for i, start in enumerate(nodes):
    for goal in nodes[i + 1 :]:
        numgraphlibpaths = len(graph.all_simple_paths(start, goal))
        numnxgraphpaths = len(
            list(networkx.all_simple_paths(graph2nx(graph), source=start, target=goal))
        )
        assert (
            numgraphlibpaths == numnxgraphpaths
        ), f"graphlib found {len(numgraphlibpaths)} paths, networkx found {len(numnxgraphpaths)}"
        # print(numgraphlibpaths)

n = 8
graph = complete_(UndirectedGraph(), n)
nodes = list(graph.graph.keys())
for i, start in enumerate(nodes):
    for goal in nodes[i + 1 :]:
        numgraphlibpaths = len(graph.all_simple_paths(start, goal))
        assert numgraphlibpaths == math.floor(math.factorial(n - 2) * math.e)
        # print(numgraphlibpaths)

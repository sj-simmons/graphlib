class GraphColoringConstraint(Constraint[str, int]):
    """Constraint for graph coloring."""

    def __init__(self, node1: str, node2: str):
        super().__init__([node1, node2])
        self.node1 = node1
        self.node2 = node2

    def is_satisfied(self, assignment: Dict[str, int]) -> bool:
        """Check if adjacent nodes have different colors."""
        if self.node1 not in assignment or self.node2 not in assignment:
            return True
        return assignment[self.node1] != assignment[self.node2]

    def get_conflicted_variables(self, assignment: Dict[str, int]) -> Set[str]:
        """Return nodes that are in conflict."""
        if self.node1 in assignment and self.node2 in assignment:
            if assignment[self.node1] == assignment[self.node2]:
                return {self.node1, self.node2}
        return set()


class GraphColoringCSP(CSP[str, int]):
    """Graph coloring problem as a CSP."""

    def __init__(self, nodes: List[str], edges: List[tuple], num_colors: int):
        super().__init__()

        # Add variables (nodes) with domains (colors)
        for node in nodes:
            self.add_variable(node, list(range(num_colors)))

        # Add constraints for edges
        for node1, node2 in edges:
            self.add_constraint(GraphColoringConstraint(node1, node2))

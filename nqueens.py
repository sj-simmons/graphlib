from csp import Constraint, CSP
from typing import List, Dict, Set


class NQueensConstraint(Constraint[int, int]):
    """Constraint for N-Queens problem."""

    def __init__(self, variables: List[int]):
        super().__init__(variables)
        self.n = len(variables)

    def is_satisfied(self, assignment: Dict[int, int]) -> bool:
        """Check if queens don't attack each other."""
        for i in range(self.n):
            if i not in assignment:
                continue
            for j in range(i + 1, self.n):
                if j not in assignment:
                    continue
                # Same column or diagonal
                if assignment[i] == assignment[j] or abs(
                    assignment[i] - assignment[j]
                ) == abs(i - j):
                    return False
        return True

    def get_conflicted_variables(self, assignment: Dict[int, int]) -> Set[int]:
        """Return queens that are in conflict."""
        conflicted = set()
        for i in range(self.n):
            if i not in assignment:
                continue
            for j in range(i + 1, self.n):
                if j not in assignment:
                    continue
                if assignment[i] == assignment[j] or abs(
                    assignment[i] - assignment[j]
                ) == abs(i - j):
                    conflicted.add(i)
                    conflicted.add(j)
        return conflicted


class NQueensCSP(CSP[int, int]):
    """N-Queens problem as a CSP."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

        # Add variables (rows) with domains (columns)
        for i in range(n):
            self.add_variable(i, list(range(n)))

        # Add constraints
        self.add_constraint(NQueensConstraint(list(range(n))))


def show_solution(solution):
    for row in range(n):
        line = ""
        for col in range(n):
            if solution[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line.rstrip())


n = 8
nqueens = NQueensCSP(n)
solution = nqueens.solve(use_forward_checking=False)

# print an ascii chess board with queens positioned on it
if solution is None:
    print("No solution found")
else:
    if n <= 8:
        print(solution)
    print("backtracks: ", nqueens.get_backtrack_count())
    if n <= 30:
        show_solution(solution)

if n <= 8:
    solutions = nqueens.get_all_solutions()
    print("total number of solutions:", len(solutions))
    for solution in solutions:
        show_solution(solution)
        q = input("quit? ")
        if len(q) and q[0].lower() == "q":
            break

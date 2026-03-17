from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, List, Optional, Set, Any
import copy

# Type variables for generic implementation
V = TypeVar("V")  # Variable type
D = TypeVar("D")  # Domain value type


class Constraint(Generic[V, D], ABC):
    """Abstract base class for constraints."""

    def __init__(self, variables: List[V]):
        self.variables = variables

    @abstractmethod
    def is_satisfied(self, assignment: Dict[V, D]) -> bool:
        """Check if the constraint is satisfied given the current assignment."""
        pass

    @abstractmethod
    def get_conflicted_variables(self, assignment: Dict[V, D]) -> Set[V]:
        """Return variables that are involved in conflicts."""
        pass


class CSP(Generic[V, D]):
    """Base class for Constraint Satisfaction Problems."""

    def __init__(self):
        self.variables: List[V] = []
        self.domains: Dict[V, List[D]] = {}
        self.constraints: Dict[V, List[Constraint[V, D]]] = {}
        self._current_assignment: Dict[V, D] = {}
        # Add backtracking counter
        self.backtrack_count: int = 0

    def add_variable(self, variable: V, domain: List[D]) -> None:
        """Add a variable with its domain."""
        self.variables.append(variable)
        self.domains[variable] = domain.copy()
        self.constraints[variable] = []

    def add_constraint(self, constraint: Constraint[V, D]) -> None:
        """Add a constraint and connect it to relevant variables."""
        for var in constraint.variables:
            if var not in self.constraints:
                self.constraints[var] = []
            self.constraints[var].append(constraint)

    def get_constraints_for_variable(self, variable: V) -> List[Constraint[V, D]]:
        """Get all constraints involving a specific variable."""
        return self.constraints.get(variable, [])

    def is_consistent(self, variable: V, value: D, assignment: Dict[V, D]) -> bool:
        """Check if assigning value to variable is consistent with all constraints."""
        # Create a temporary assignment to test
        temp_assignment = assignment.copy()
        temp_assignment[variable] = value

        # Check all constraints involving this variable
        for constraint in self.get_constraints_for_variable(variable):
            if not constraint.is_satisfied(temp_assignment):
                return False
        return True

    def is_complete(self, assignment: Dict[V, D]) -> bool:
        """Check if assignment covers all variables."""
        return len(assignment) == len(self.variables)

    def select_unassigned_variable(self, assignment: Dict[V, D]) -> Optional[V]:
        """Select the next variable to assign (MRV heuristic by default)."""
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None

        # Default: Minimum Remaining Values (MRV) heuristic
        return min(unassigned, key=lambda var: len(self.domains[var]))

    def order_domain_values(self, variable: V, assignment: Dict[V, D]) -> List[D]:
        """Order domain values for a variable (LCV heuristic by default)."""

        # Default: Least Constraining Value heuristic
        def count_conflicts(value: D) -> int:
            conflicts = 0
            temp_assignment = assignment.copy()
            temp_assignment[variable] = value

            for constraint in self.get_constraints_for_variable(variable):
                conflicts += len(constraint.get_conflicted_variables(temp_assignment))
            return conflicts

        return sorted(self.domains[variable], key=count_conflicts)

    def forward_check(self, variable: V, value: D, assignment: Dict[V, D]) -> bool:
        """
        Forward checking: remove inconsistent values from future variables' domains.
        Returns False if any domain becomes empty.
        """
        # This is a basic forward checking - override for more sophisticated versions
        for future_var in self.variables:
            if future_var in assignment:
                continue

            # Check if any value in future_var's domain is consistent
            consistent_values = []
            for val in self.domains[future_var]:
                temp_assignment = assignment.copy()
                temp_assignment[future_var] = val

                is_consistent = True
                for constraint in self.get_constraints_for_variable(future_var):
                    if not constraint.is_satisfied(temp_assignment):
                        is_consistent = False
                        break

                if is_consistent:
                    consistent_values.append(val)

            if not consistent_values:
                return False

            # In a real implementation, you'd need to restore domains during backtracking
            # This simplified version doesn't handle restoration
            self.domains[future_var] = consistent_values

        return True

    def solve(self, use_forward_checking: bool = False) -> Optional[Dict[V, D]]:
        """
        Main solving method using backtracking search.
        Returns a solution assignment or None if no solution exists.
        """
        # Reset backtrack counter before starting search
        self.backtrack_count = 0
        return self._backtrack({}, use_forward_checking)

    def _backtrack(
        self, assignment: Dict[V, D], use_forward_checking: bool
    ) -> Optional[Dict[V, D]]:
        """Backtracking search algorithm."""
        if self.is_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        if var is None:
            return None

        # Save current domain state for restoration
        saved_domains = None
        if use_forward_checking:
            saved_domains = copy.deepcopy(self.domains)

        for value in self.order_domain_values(var, assignment):
            if self.is_consistent(var, value, assignment):
                # Make assignment
                assignment[var] = value

                # Apply forward checking if requested
                if use_forward_checking:
                    if not self.forward_check(var, value, assignment):
                        # Restore domains and try next value
                        self.domains = saved_domains
                        del assignment[var]
                        # Increment backtrack counter (failed forward check)
                        self.backtrack_count += 1
                        continue

                # Recursive call
                result = self._backtrack(assignment, use_forward_checking)
                if result is not None:
                    return result

                # Undo assignment and restore domains
                del assignment[var]
                if use_forward_checking and saved_domains:
                    self.domains = saved_domains
                # Increment backtrack counter (recursive call failed)
                self.backtrack_count += 1

        # Increment backtrack counter (exhausted all values for this variable)
        self.backtrack_count += 1
        return None

    def get_all_solutions(self, limit: int = None) -> List[Dict[V, D]]:
        """Find all solutions (or up to limit solutions)."""
        self.__init__(self.n)  # resets order of self.domain[var] for all var
        solutions = []
        self._find_all_solutions({}, solutions, limit)
        return solutions

    def _find_all_solutions(
        self, assignment: Dict[V, D], solutions: List[Dict[V, D]], limit: int = None
    ):
        """Helper method to find all solutions."""
        if limit is not None and len(solutions) >= limit:
            return

        if self.is_complete(assignment):
            solutions.append(assignment.copy())
            return

        var = self.select_unassigned_variable(assignment)
        if var is None:
            return

        for value in self.order_domain_values(var, assignment):
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                self._find_all_solutions(assignment, solutions, limit)
                del assignment[var]

                if limit is not None and len(solutions) >= limit:
                    return

    def get_backtrack_count(self) -> int:
        """Return the number of backtracks performed in the last solve attempt."""
        return self.backtrack_count

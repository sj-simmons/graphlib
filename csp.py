from typing import Any, Dict, List, Set, Optional, Tuple
import graph


class CSP:
    """
    Constraint Satisfaction Problem solver for graph coloring problems.
    This can be extended to other CSPs, but is designed for graph coloring.
    """

    def __init__(self, graph: graph.UndirectedGraph_, domain: List[Any] = None):
        """
        Initialize a CSP for the given graph.

        Args:
            graph: An undirected graph (from graph.py)
            domain: List of possible values for each variable (e.g., colors)
                   Default is [0, 1, 2] for 3-coloring
        """
        self.graph = graph
        # Get vertices and sort by degree (highest first) to improve search efficiency
        # even when no heuristics are used. This helps color constrained variables first,
        # reducing the branching factor early in the search.
        vertices = list(graph.get_vertices())
        # Compute degrees
        degrees = {v: len(graph.get_neighbors(v)) for v in vertices}
        # Sort vertices by degree in descending order
        self.variables = sorted(vertices, key=lambda v: degrees[v], reverse=True)

        if domain is None:
            self.domain = [0, 1, 2]  # Default: 3-coloring
        else:
            self.domain = domain

        self.constraints = self._build_constraints()

    def _build_constraints(self) -> Dict[Any, Set[Any]]:
        """
        Build constraints from graph edges.
        For graph coloring: adjacent vertices must have different colors.

        Returns:
            Dictionary mapping each variable to its constrained neighbors
        """
        constraints = {}
        for vertex in self.variables:
            constraints[vertex] = set(self.graph.get_neighbors(vertex))
        return constraints

    def is_consistent(
        self, variable: Any, value: Any, assignment: Dict[Any, Any]
    ) -> bool:
        """
        Check if assigning value to variable is consistent with current assignment.

        Args:
            variable: The variable being assigned
            value: The value being assigned
            assignment: Current partial assignment

        Returns:
            True if assignment is consistent, False otherwise
        """
        for neighbor in self.constraints[variable]:
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True

    def backtracking_search(
        self,
        use_mrv: bool = False,
        use_degree: bool = False,
        use_lcv: bool = False,
        use_forward_checking: bool = False,
    ) -> Optional[Dict[Any, Any]]:
        """
        Perform backtracking search to find a solution.

        Args:
            use_mrv: Use Minimum Remaining Values heuristic
            use_degree: Use Degree heuristic (tie-breaker for MRV)
            use_lcv: Use Least Constraining Value heuristic
            use_forward_checking: Use forward checking to prune domains

        Returns:
            Complete assignment if solution found, None otherwise
        """
        if use_forward_checking:
            # Initialize domains for forward checking
            domains = {var: set(self.domain) for var in self.variables}
            return self._backtracking_with_fc({}, domains, use_mrv, use_degree, use_lcv)
        else:
            return self._backtracking({}, use_mrv, use_degree, use_lcv)

    def _backtracking(
        self, assignment: Dict[Any, Any], use_mrv: bool, use_degree: bool, use_lcv: bool
    ) -> Optional[Dict[Any, Any]]:
        """
        Recursive backtracking without forward checking.
        """
        # If assignment is complete, return it
        if len(assignment) == len(self.variables):
            return assignment

        # Select unassigned variable
        var = self._select_unassigned_variable(assignment, use_mrv, use_degree)
        if var is None:
            return None

        # Order domain values
        values = self._order_domain_values(var, assignment, use_lcv)

        for value in values:
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                result = self._backtracking(assignment, use_mrv, use_degree, use_lcv)
                if result is not None:
                    return result
                del assignment[var]

        return None

    def _backtracking_with_fc(
        self,
        assignment: Dict[Any, Any],
        domains: Dict[Any, Set[Any]],
        use_mrv: bool,
        use_degree: bool,
        use_lcv: bool,
    ) -> Optional[Dict[Any, Any]]:
        """
        Recursive backtracking with forward checking.
        """
        # If assignment is complete, return it
        if len(assignment) == len(self.variables):
            return assignment

        # Select unassigned variable
        var = self._select_unassigned_variable_fc(
            assignment, domains, use_mrv, use_degree
        )
        if var is None:
            return None

        # Order domain values
        values = self._order_domain_values_fc(var, assignment, domains, use_lcv)

        for value in values:
            if self.is_consistent(var, value, assignment):
                assignment[var] = value

                # Save current domains for backtracking
                old_domains = {v: set(domains[v]) for v in domains}

                # Perform forward checking
                if self._forward_check(var, value, assignment, domains):
                    result = self._backtracking_with_fc(
                        assignment, domains, use_mrv, use_degree, use_lcv
                    )
                    if result is not None:
                        return result

                # Restore domains and remove assignment
                for v in domains:
                    domains[v] = old_domains[v]
                del assignment[var]

        return None

    def _select_unassigned_variable(
        self, assignment: Dict[Any, Any], use_mrv: bool, use_degree: bool
    ) -> Any:
        """
        Select an unassigned variable using heuristics.
        """
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None

        if not use_mrv:
            return unassigned[0]

        # MRV: Choose variable with fewest legal values
        # Count legal values for each unassigned variable
        mrv_candidates = []
        min_legal = float("inf")

        for var in unassigned:
            legal_count = 0
            for value in self.domain:
                # Track checks only in stats version
                if hasattr(self, "stats") and "checks" in self.stats:
                    self.stats["checks"] += 1
                if self.is_consistent(var, value, assignment):
                    legal_count += 1
            if legal_count == 0:
                # If no legal values, the current assignment is inconsistent
                # Return this variable to trigger immediate backtracking
                return var
            if legal_count < min_legal:
                min_legal = legal_count
                mrv_candidates = [var]
            elif legal_count == min_legal:
                mrv_candidates.append(var)

        # Sort candidates for deterministic behavior
        if len(mrv_candidates) > 1:
            # Always use degree as a tie-breaker, even when use_degree is False
            # This is generally a good heuristic and prevents getting stuck
            # Sort by degree (highest first), then by variable name for determinism
            mrv_candidates.sort(
                key=lambda v: (
                    -len([n for n in self.constraints[v] if n not in assignment]),
                    str(v),
                )
            )

        return mrv_candidates[0] if mrv_candidates else unassigned[0]

    def _select_unassigned_variable_fc(
        self,
        assignment: Dict[Any, Any],
        domains: Dict[Any, Set[Any]],
        use_mrv: bool,
        use_degree: bool,
    ) -> Any:
        """
        Select unassigned variable for forward checking version.
        """
        unassigned = [v for v in self.variables if v not in assignment]
        if not unassigned:
            return None

        if not use_mrv:
            return unassigned[0]

        # MRV: Choose variable with fewest legal values
        # Count legal values from domain that are consistent with assignment
        mrv_vars = []
        min_legal = float("inf")

        for var in unassigned:
            # Count legal values
            legal_count = 0
            for value in domains[var]:
                if self.is_consistent(var, value, assignment):
                    legal_count += 1
            if legal_count == 0:
                # No legal values, current assignment is inconsistent
                # Return this variable to trigger immediate backtracking
                return var
            if legal_count < min_legal:
                min_legal = legal_count
                mrv_vars = [var]
            elif legal_count == min_legal:
                mrv_vars.append(var)

        # Sort for deterministic behavior
        if len(mrv_vars) > 1:
            # Always use degree as a tie-breaker, even when use_degree is False
            # This is generally a good heuristic and prevents getting stuck
            # Sort by degree (highest first), then by variable name for determinism
            mrv_vars.sort(
                key=lambda v: (
                    -len([n for n in self.constraints[v] if n not in assignment]),
                    str(v),
                )
            )

        return mrv_vars[0] if mrv_vars else unassigned[0]

    def _order_domain_values(
        self, var: Any, assignment: Dict[Any, Any], use_lcv: bool
    ) -> List[Any]:
        """
        Order domain values for a variable.
        """
        # First get all consistent values
        consistent_values = [
            value for value in self.domain if self.is_consistent(var, value, assignment)
        ]

        if not use_lcv:
            # Even without LCV, use a simple heuristic: try colors that are least used
            # in the current assignment among neighbors
            color_counts = {color: 0 for color in self.domain}
            for neighbor in self.constraints[var]:
                if neighbor in assignment:
                    color = assignment[neighbor]
                    if color in color_counts:
                        color_counts[color] += 1
            # Sort by count (least used first) to balance colors
            consistent_values.sort(key=lambda color: color_counts[color])
            return consistent_values

        if not consistent_values:
            return consistent_values

        # For LCV, we want to count how many options remain for neighbors after assigning this value
        # A better approach for graph coloring: count how many neighbors would still have this value available
        # But actually, we want to choose the value that eliminates the fewest options from neighbors
        # So we need to count for each neighbor how many of their consistent values would remain
        # This is expensive, so let's use a simpler heuristic:
        # Count how many unassigned neighbors currently have this value as a consistent option
        # We can precompute for each color how many neighbors can take it
        color_counts = {color: 0 for color in self.domain}

        # Count for each neighbor how many colors they can take
        for neighbor in self.constraints[var]:
            if neighbor not in assignment:
                for color in self.domain:
                    if self.is_consistent(neighbor, color, assignment):
                        color_counts[color] += 1

        # Now for each consistent value, the score is the number of neighbors that can take that color
        # Higher score means more neighbors can take this color, so it's less constraining
        value_scores = []
        for value in consistent_values:
            # The score is the number of neighbors that can still use this color
            # Actually, we want least constraining, which means higher score
            score = color_counts[value]
            value_scores.append((value, score))

        # Sort by highest score (least constraining first)
        value_scores.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in value_scores]

    def _order_domain_values_fc(
        self,
        var: Any,
        assignment: Dict[Any, Any],
        domains: Dict[Any, Set[Any]],
        use_lcv: bool,
    ) -> List[Any]:
        """
        Order domain values for forward checking version.
        """
        # Get consistent values from current domain
        consistent_values = [
            value
            for value in domains[var]
            if self.is_consistent(var, value, assignment)
        ]

        if not use_lcv:
            # Even without LCV, use a simple heuristic: try colors that are least used
            # in the current assignment among neighbors
            color_counts = {color: 0 for color in self.domain}
            for neighbor in self.constraints[var]:
                if neighbor in assignment:
                    color = assignment[neighbor]
                    if color in color_counts:
                        color_counts[color] += 1
            # Sort by count (least used first) to balance colors
            consistent_values.sort(key=lambda color: color_counts[color])
            return consistent_values

        # For LCV, count how many neighbors have this value in their domain
        # More neighbors having the value means it's less constraining (they can still use it)
        value_scores = []
        for value in consistent_values:
            count = 0
            for neighbor in self.constraints[var]:
                if neighbor not in assignment:
                    if value in domains[neighbor]:
                        count += 1
            # Higher count means less constraining
            value_scores.append((value, count))

        # Sort by highest count (least constraining first)
        value_scores.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in value_scores]

    def _forward_check(
        self,
        var: Any,
        value: Any,
        assignment: Dict[Any, Any],
        domains: Dict[Any, Set[Any]],
    ) -> bool:
        """
        Perform forward checking after assigning value to var.
        Returns False if any domain becomes empty.
        """
        for neighbor in self.constraints[var]:
            if neighbor not in assignment:
                if value in domains[neighbor]:
                    domains[neighbor].remove(value)
                    if len(domains[neighbor]) == 0:
                        return False
        return True

    def solve(
        self,
        use_mrv: bool = True,
        use_degree: bool = True,
        use_lcv: bool = True,
        use_forward_checking: bool = True,
        max_backtracks: Optional[int] = None,
    ) -> Tuple[Optional[Dict[Any, Any]], Dict[str, int]]:
        """
        Solve the CSP and return solution with statistics.

        Args:
            use_mrv: Use Minimum Remaining Values heuristic
            use_degree: Use Degree heuristic (tie-breaker for MRV)
            use_lcv: Use Least Constraining Value heuristic
            use_forward_checking: Use forward checking to prune domains
            max_backtracks: Maximum number of backtracks allowed before bailing out.
                           If None, there is no limit.

        Returns:
            Tuple of (solution, stats) where stats contains:
            - 'assignments': number of variable assignments attempted
            - 'backtracks': number of backtracks
            - 'checks': number of consistency checks
        """
        # Statistics tracking
        self.stats = {"assignments": 0, "backtracks": 0, "checks": 0}
        self.max_backtracks = max_backtracks

        # Call the appropriate backtracking function
        if use_forward_checking:
            domains = {var: set(self.domain) for var in self.variables}
            solution = self._backtracking_with_fc_stats(
                {}, domains, use_mrv, use_degree, use_lcv
            )
        else:
            solution = self._backtracking_stats({}, use_mrv, use_degree, use_lcv)

        return solution, self.stats

    def _backtracking_stats(
        self, assignment: Dict[Any, Any], use_mrv: bool, use_degree: bool, use_lcv: bool
    ) -> Optional[Dict[Any, Any]]:
        """
        Recursive backtracking without forward checking with statistics.
        """
        # Check if we've exceeded the maximum allowed backtracks
        if (
            self.max_backtracks is not None
            and self.stats["backtracks"] >= self.max_backtracks
        ):
            return None

        # If assignment is complete, return it
        if len(assignment) == len(self.variables):
            return assignment

        # Select unassigned variable
        var = self._select_unassigned_variable(assignment, use_mrv, use_degree)
        if var is None:
            return None

        # Order domain values
        values = self._order_domain_values(var, assignment, use_lcv)

        # If there are no values, backtrack immediately
        if not values:
            self.stats["backtracks"] += 1
            if (
                self.max_backtracks is not None
                and self.stats["backtracks"] >= self.max_backtracks
            ):
                return None
            return None

        for value in values:
            # Increment checks counter
            self.stats["checks"] += 1
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                self.stats["assignments"] += 1
                result = self._backtracking_stats(
                    assignment, use_mrv, use_degree, use_lcv
                )
                if result is not None:
                    return result
                # Backtrack
                del assignment[var]
                self.stats["backtracks"] += 1
                # Check again after incrementing backtracks
                if (
                    self.max_backtracks is not None
                    and self.stats["backtracks"] >= self.max_backtracks
                ):
                    return None

        return None

    def _backtracking_with_fc_stats(
        self,
        assignment: Dict[Any, Any],
        domains: Dict[Any, Set[Any]],
        use_mrv: bool,
        use_degree: bool,
        use_lcv: bool,
    ) -> Optional[Dict[Any, Any]]:
        """
        Recursive backtracking with forward checking and statistics.
        """
        # Check if we've exceeded the maximum allowed backtracks
        if (
            self.max_backtracks is not None
            and self.stats["backtracks"] >= self.max_backtracks
        ):
            return None

        # If assignment is complete, return it
        if len(assignment) == len(self.variables):
            return assignment

        # Select unassigned variable
        var = self._select_unassigned_variable_fc(
            assignment, domains, use_mrv, use_degree
        )
        if var is None:
            return None

        # Order domain values
        values = self._order_domain_values_fc(var, assignment, domains, use_lcv)

        # If there are no values, backtrack immediately
        if not values:
            self.stats["backtracks"] += 1
            if (
                self.max_backtracks is not None
                and self.stats["backtracks"] >= self.max_backtracks
            ):
                return None
            return None

        for value in values:
            self.stats["checks"] += 1
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                self.stats["assignments"] += 1

                # Save current domains for backtracking
                old_domains = {v: set(domains[v]) for v in domains}

                # Perform forward checking
                if self._forward_check(var, value, assignment, domains):
                    result = self._backtracking_with_fc_stats(
                        assignment, domains, use_mrv, use_degree, use_lcv
                    )
                    if result is not None:
                        return result

                # Restore domains and remove assignment
                for v in domains:
                    domains[v] = old_domains[v]
                del assignment[var]
                self.stats["backtracks"] += 1
                # Check again after incrementing backtracks
                if (
                    self.max_backtracks is not None
                    and self.stats["backtracks"] >= self.max_backtracks
                ):
                    return None

        return None

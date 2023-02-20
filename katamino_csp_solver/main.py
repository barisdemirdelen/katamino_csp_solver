import itertools

import matplotlib.pyplot as plt
from constraint import Problem
from katamino_csp_solver.constraints import (
    ValidAssignmentConstraint,
    AllDifferentPieceConstraint,
    OnlyLastOrdersCanBeUnusedConstraint,
    FieldMustBeFullConstraint,
    NoHolesOfXConstraint,
)
from katamino_csp_solver.field import Field
from katamino_csp_solver.pieces import (
    PIECES,
    get_shape_permutations,
    PieceConfig,
)


def define_variables(problem: Problem, max_y: int) -> None:
    for order in range(len(PIECES)):
        domain = []
        if max_y < 12:
            # We can also not use a piece
            domain.append(PieceConfig(order, -1, -1, -1))

        for idx in range(len(PIECES)):
            for rot, rev in get_shape_permutations(idx):
                # In this order, use a piece with index idx, rotation rot and
                # optionally reverse the piece
                domain.append(PieceConfig(order, idx, rot, int(rev)))
        problem.addVariable(order, domain)


def define_constraints(problem: Problem, max_x: int, max_y: int) -> None:
    for i, j in itertools.combinations(range(len(PIECES)), 2):
        problem.addConstraint(AllDifferentPieceConstraint(), [i, j])

    for i in range(len(PIECES)):
        if max_y < 12:
            problem.addConstraint(OnlyLastOrdersCanBeUnusedConstraint(), range(i + 1))
        problem.addConstraint(ValidAssignmentConstraint(max_x, max_y), range(i + 1))
        problem.addConstraint(NoHolesOfXConstraint(max_x, max_y, 3), range(i + 1))

    problem.addConstraint(FieldMustBeFullConstraint(max_x, max_y))


def solve(
        max_x: int, max_y: int, verbose: bool = True, show_grid: bool = True
) -> dict[int, PieceConfig]:
    """
    This function creates a Constraint Satisfaction Problem (CSP) definition.
    We create our variables as our placement order, and for a solution we have to
    assign a PieceConfig to each order variable, that satisfies all the constraints.
    PieceConfig contains information about which piece and in which configuration is placed
    into the field.
    Placement always occurs at the minimum (x, y) coordinate available.
    After the problem is well-defined, a generic CSP solver can solve the Katamino problem.

    :param max_x: x dimension of the board
    :param max_y: y dimension of the board
    :param verbose: If we should print the solution
    :param show_grid: If we should show an image containing the solution
    :return: A solution dictionary containing the placement order as keys and PieceConfig as values
    """

    problem = Problem()

    define_variables(problem, max_y=max_y)
    define_constraints(problem, max_x=max_x, max_y=max_y)

    solution = problem.getSolution()
    assert solution is not None

    field = Field(max_x, max_y)
    field.place_pieces(solution.values(), place_label=True)

    if verbose:
        print(field.grid)
        print(solution)

    if show_grid:
        plt.imshow(field.grid, cmap="inferno", aspect="equal")
        plt.show()

    return solution


if __name__ == "__main__":
    solve(5, 12)

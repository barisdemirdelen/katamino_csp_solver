import itertools

import matplotlib.pyplot as plt
from constraint import Problem

from field import Field
from katamino_solver.constraints import (
    ValidAssignmentConstraint,
    AllDifferentPieceConstraint,
    OnlyLastOrdersCanBeUnusedConstraint,
    FieldMustBeFullConstraint,
    NoHolesOfXConstraint,
)
from pieces import (
    PIECES,
    get_shape_permutations,
    PieceConfig,
)


def define_variables(problem: Problem, max_y: int) -> None:
    for order in range(len(PIECES)):
        domain = []
        if max_y < 12:
            domain.append(PieceConfig(order, -1, -1, -1))
        for idx in range(len(PIECES)):
            for rot, rev in get_shape_permutations(idx):
                domain.append(PieceConfig(order, idx, rot, int(rev)))
        problem.addVariable(order, domain)


def define_constraints(problem: Problem, max_x: int, max_y: int) -> None:
    for i, j in itertools.combinations(range(len(PIECES)), 2):
        problem.addConstraint(AllDifferentPieceConstraint(), [i, j])

    if max_y < 12:
        problem.addConstraint(OnlyLastOrdersCanBeUnusedConstraint())

    for i in range(len(PIECES)):
        if max_y < 12:
            problem.addConstraint(OnlyLastOrdersCanBeUnusedConstraint(), range(i + 1))
        problem.addConstraint(ValidAssignmentConstraint(max_x, max_y), range(i + 1))
        problem.addConstraint(NoHolesOfXConstraint(max_x, max_y, 3), range(i + 1))

    problem.addConstraint(FieldMustBeFullConstraint(max_x, max_y))


def solve(max_x, max_y, verbose=True, show_grid=True):
    problem = Problem()

    define_variables(problem, max_y=max_y)
    define_constraints(problem, max_x=max_x, max_y=max_y)

    solution = problem.getSolution()
    assert solution is not None

    field = Field(max_x, max_y)
    list(field.place_pieces(solution.values(), place_label=True))

    if verbose:
        print(field.grid)
        print(solution)

    if show_grid:
        plt.imshow(field.grid, cmap="inferno", aspect="equal")
        plt.show()

    return solution


if __name__ == "__main__":
    solve(5, 12)

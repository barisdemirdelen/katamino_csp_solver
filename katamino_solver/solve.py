import itertools

import matplotlib.pyplot as plt
import numpy as np
from constraint import Problem

from katamino_solver.constraints import (
    ValidAssignmentConstraint,
    AllDifferentPieceConstraint,
    OnlyLastOrdersCanBeUnusedConstraint,
    FieldMustBeFullConstraint,
    NoHolesOfXConstraint,
)
from pieces import (
    pieces,
    get_shape_permutations,
    get_rotated_shape,
    place_number,
    PieceConfig,
    get_min_x_y,
    get_shape_min_x_y,
)


def get_coords(max_x, max_y):
    coords = []
    for x in range(max_x):
        for y in range(max_y):
            coords.append((x, y))
    return coords


def solve(max_x, max_y):
    problem = Problem()

    for order in range(len(pieces)):
        domain = []
        if max_y < 12:
            domain.append(PieceConfig(order, -1, -1, -1))
        for idx in range(len(pieces)):
            for rot, rev in get_shape_permutations(idx):
                domain.append(PieceConfig(order, idx, rot, int(rev)))
        problem.addVariable(order, domain)

    for i, j in itertools.combinations(range(len(pieces)), 2):
        problem.addConstraint(AllDifferentPieceConstraint(), [i, j])

    # problem.addConstraint(AllDifferentPieceConstraint())
    if max_y < 12:
        problem.addConstraint(OnlyLastOrdersCanBeUnusedConstraint())

    for i in range(len(pieces)):
        if max_y < 12:
            problem.addConstraint(OnlyLastOrdersCanBeUnusedConstraint(), range(i + 1))
        problem.addConstraint(ValidAssignmentConstraint(max_x, max_y), range(i + 1))
        problem.addConstraint(NoHolesOfXConstraint(max_x, max_y, 3), range(i + 1))

    #
    # for i in range(4):
    #     combinations = itertools.combinations(range(len(pieces)), i + 1)
    #     for combination in combinations:
    #         problem.addConstraint(NoHolesOfXConstraint(max_x, max_y, 3), combination)
    #
    problem.addConstraint(FieldMustBeFullConstraint(max_x, max_y))

    solution = problem.getSolution()
    assert solution is not None

    field = np.zeros((max_y, max_x), dtype=np.int8)
    for order, piece_no, rotation, reverse in solution.values():
        if piece_no == -1:
            break
        shape = get_rotated_shape(piece_no, rotation, reverse)
        x, y = get_min_x_y(field)
        if x is None:
            break

        shape_min_x, shape_min_y = get_shape_min_x_y(piece_no, rotation, reverse)

        place_number(field, shape, x - shape_min_x, y - shape_min_y, piece_no)

    print(field)
    print(solution)

    plt.imshow(field, cmap="inferno", aspect="equal")
    plt.show()

    return solution


if __name__ == "__main__":
    solve(5, 12)

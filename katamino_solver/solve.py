import itertools

import matplotlib.pyplot as plt
import numpy as np
from constraint import Problem

from katamino_solver.constraints import PieceWithinCoordsConstraint, \
    PiecesNotConflictingConstraint, TotalAmountOfGridsMustMatchConstraint, NoHolesOfXConstraint
from katamino_solver.pieces import pieces, get_shape_permutations, get_rotated_shape, place_number


def get_coords(max_x, max_y):
    coords = []
    for x in range(max_x):
        for y in range(max_y):
            coords.append((x, y))
    return coords


def solve(max_x, max_y):
    problem = Problem()
    coords = get_coords(max_x, max_y)

    for i in range(len(pieces)):
        domain = [(i, -1, -1, -1, -1)] if max_y < 12 else []
        for x, y in coords:
            for rot, rev in get_shape_permutations(i):
                domain.append((i, x, y, rot, int(rev)))
        problem.addVariable(i, domain)

    for i, _ in enumerate(pieces):
        problem.addConstraint(PieceWithinCoordsConstraint(max_x, max_y), [i])

    for i, j in itertools.combinations(range(len(pieces)), 2):
        problem.addConstraint(PiecesNotConflictingConstraint(max_x, max_y), [i, j])

    for i in range(4):
        combinations = itertools.combinations(range(len(pieces)), i + 1)
        for combination in combinations:
            problem.addConstraint(NoHolesOfXConstraint(max_x, max_y, 3), combination)

    problem.addConstraint(TotalAmountOfGridsMustMatchConstraint(max_x, max_y),
                          list(range(len(pieces))))

    solution = problem.getSolution()
    assert solution is not None

    field = np.zeros((max_y, max_x), dtype=np.int8)
    for piece_no, x, y, rotation, reverse in solution.values():
        if x == -1:
            continue
        shape = get_rotated_shape(piece_no, rotation, reverse)
        place_number(field, shape, x, y, piece_no)

    print(field)
    print(solution)

    plt.imshow(field, cmap='inferno', aspect='equal')
    plt.show()

    return solution


if __name__ == "__main__":
    solve(5, 12)

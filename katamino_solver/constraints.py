import numba
import numpy as np

from katamino_solver.pieces import pieces, get_rotated_shape, place


class FieldConstraint:
    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y


class PieceWithinCoordsConstraint(FieldConstraint):

    def __call__(self, piece_config):
        piece_no, x, y, rotation, reverse = piece_config
        if x == -1:
            return True
        shape = get_rotated_shape(piece_no, rotation, reverse)

        return x + shape.shape[1] <= self.max_x and y + shape.shape[0] <= self.max_y


class PiecesNotConflictingConstraint(FieldConstraint):

    def __call__(self, piece1_config, piece2_config):
        piece_no1, x1, y1, rotation1, reverse1 = piece1_config
        if x1 == -1:
            return True
        shape1 = get_rotated_shape(piece_no1, rotation1, reverse1)

        piece_no2, x2, y2, rotation2, reverse2 = piece2_config
        if x2 == -1:
            return True
        shape2 = get_rotated_shape(piece_no2, rotation2, reverse2)

        field = np.zeros((self.max_y, self.max_x), dtype=np.int16)

        place(field, shape1, x1, y1)
        place(field, shape2, x2, y2)

        return np.max(field) == 1


class TotalAmountOfGridsMustMatchConstraint(FieldConstraint):
    def __call__(self, *piece_configs):
        total_count = 0
        for piece_no, x, y, rotation, reverse in piece_configs:
            if x == -1:
                continue
            piece = pieces[piece_no]
            total_count += np.sum(piece)

        return total_count == self.max_x * self.max_y


@numba.njit
def get_neighbours(field, x, y, max_x, max_y):
    if x - 1 >= 0 and field[y, x - 1] == 0:
        yield y, x - 1
    if y - 1 >= 0 and field[y - 1, x] == 0:
        yield y - 1, x
    if x + 1 < max_x and field[y, x + 1] == 0:
        yield y, x + 1
    if y + 1 < max_y and field[y + 1, x] == 0:
        yield y + 1, x


@numba.njit
def search_neighbours(field, x, y, max_x, max_y, visited, current_count):
    for ny, nx in get_neighbours(field, x, y, max_x, max_y):
        if (ny, nx) in visited:
            continue
        visited.add((ny, nx))
        current_count = search_neighbours(field, nx, ny, max_x, max_y, visited, current_count) + 1
    return current_count


class NoHolesOfXConstraint(FieldConstraint):
    def __init__(self, max_x, max_y, max_hole_zeros):
        super().__init__(max_x, max_y)
        self.max_hole_zeros = max_hole_zeros

    def __call__(self, *piece_configs):
        field = np.zeros((self.max_y, self.max_x), dtype=np.int8)
        for piece_no, x, y, rotation, reverse in piece_configs:
            if x == -1:
                continue
            shape = get_rotated_shape(piece_no, rotation, reverse)
            place(field, shape, x, y)

        zero_ys, zero_xs = np.where(field == 0)
        visited = {(-1, -1)}
        for y, x in zip(zero_ys, zero_xs):
            if (y, x) in visited:
                continue
            count = search_neighbours(field, x, y, self.max_x, self.max_y, visited, 1)
            if count <= self.max_hole_zeros:
                return False
        return True

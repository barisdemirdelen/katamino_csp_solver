import numba
import numpy as np

from katamino_solver.pieces import (
    get_rotated_shape,
    place,
    PieceConfig,
    create_field,
    get_min_x_y,
    get_shape_min_x_y,
)


class FieldConstraint:
    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y


class AllDifferentPieceConstraint:
    # @lru_cache(maxsize=None)
    def __call__(self, *piece_configs: list[PieceConfig]):
        used_pieces = set()
        for piece_config in piece_configs:
            order, piece_no, rotation, reverse = piece_config
            if piece_no == -1:
                continue
            if piece_no in used_pieces:
                return False
            used_pieces.add(piece_no)
        return True


class OnlyLastOrdersCanBeUnusedConstraint:
    # @lru_cache(maxsize=None)
    def __call__(self, *piece_configs: list[PieceConfig]):
        unused_found = False
        for piece_config in piece_configs:
            order, piece_no, rotation, reverse = piece_config
            if piece_no == -1:
                unused_found = True
            elif unused_found:
                return False
        return True


class FieldMustBeFullConstraint(FieldConstraint):
    # @lru_cache(maxsize=None)
    def __call__(self, *piece_configs: list[PieceConfig]):
        field = create_field(self.max_x, self.max_y)
        for piece_config in piece_configs:
            order, piece_no, rotation, reverse = piece_config
            if piece_no == -1:
                break

            shape = get_rotated_shape(piece_no, rotation, reverse)

            x, y = get_min_x_y(field)
            if x is None:
                break

            shape_min_x, shape_min_y = get_shape_min_x_y(piece_no, rotation, reverse)

            if (
                x - shape_min_x < 0
                or y - shape_min_y < 0
                or x - shape_min_x + shape.shape[1] > self.max_x
                or y - shape_min_y + shape.shape[0] > self.max_y
            ):
                return False

            place(field, shape, x - shape_min_x, y - shape_min_y)

        return np.all(field == 1)


class ValidAssignmentConstraint(FieldConstraint):
    # @lru_cache(maxsize=None)
    def __call__(self, *piece_configs: list[PieceConfig]):
        field = create_field(self.max_x, self.max_y)
        for piece_config in piece_configs:
            order, piece_no, rotation, reverse = piece_config
            if piece_no == -1:
                break

            shape = get_rotated_shape(piece_no, rotation, reverse)

            x, y = get_min_x_y(field)
            if x is None:
                break

            shape_min_x, shape_min_y = get_shape_min_x_y(piece_no, rotation, reverse)

            if (
                x - shape_min_x < 0
                or y - shape_min_y < 0
                or x - shape_min_x + shape.shape[1] > self.max_x
                or y - shape_min_y + shape.shape[0] > self.max_y
            ):
                return False

            place(field, shape, x - shape_min_x, y - shape_min_y)

        return np.max(field) == 1


@numba.njit(cache=True)
def get_neighbours(field, x, y, max_x, max_y):
    if x - 1 >= 0 and field[y, x - 1] == 0:
        yield y, x - 1
    if y - 1 >= 0 and field[y - 1, x] == 0:
        yield y - 1, x
    if x + 1 < max_x and field[y, x + 1] == 0:
        yield y, x + 1
    if y + 1 < max_y and field[y + 1, x] == 0:
        yield y + 1, x


@numba.njit(cache=True)
def search_neighbours(field, x, y, max_x, max_y, visited, current_count):
    for ny, nx in get_neighbours(field, x, y, max_x, max_y):
        if (ny, nx) in visited:
            continue
        visited.add((ny, nx))
        current_count = (
            search_neighbours(field, nx, ny, max_x, max_y, visited, current_count) + 1
        )
    return current_count


class NoHolesOfXConstraint(FieldConstraint):
    def __init__(self, max_x, max_y, max_hole_zeros):
        super().__init__(max_x, max_y)
        self.max_hole_zeros = max_hole_zeros

    def __call__(self, *piece_configs):
        field = create_field(self.max_x, self.max_y)
        for order, piece_no, rotation, reverse in piece_configs:
            if order == -1:
                break
            shape = get_rotated_shape(piece_no, rotation, reverse)
            x, y = get_min_x_y(field)
            if x is None:
                break

            shape_min_x, shape_min_y = get_shape_min_x_y(piece_no, rotation, reverse)

            if (
                x - shape_min_x < 0
                or y - shape_min_y < 0
                or x - shape_min_x + shape.shape[1] > self.max_x
                or y - shape_min_y + shape.shape[0] > self.max_y
            ):
                return False

            place(field, shape, x - shape_min_x, y - shape_min_y)

        visited = {(-1, -1)}
        zero_locs = np.where(field == 0)
        if not zero_locs or len(zero_locs[0]) == 0:
            return True

        for y, x in zip(*zero_locs):
            if (y, x) in visited:
                continue
            count = search_neighbours(field, x, y, self.max_x, self.max_y, visited, 1)
            if count <= self.max_hole_zeros:
                return False
        return True

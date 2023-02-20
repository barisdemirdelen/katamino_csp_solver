import abc
from collections.abc import Generator

import numpy as np

from katamino_csp_solver.field import Field, InvalidPlacementError
from katamino_csp_solver.pieces import (
    PieceConfig,
)


class PieceOrderConstraint(abc.ABC):
    def __call__(self, *piece_configs: PieceConfig):
        return self.evaluate(list(piece_configs))

    @abc.abstractmethod
    def evaluate(self, piece_configs: list[PieceConfig]) -> bool:
        pass


class FieldConstraint(abc.ABC):
    def __init__(self, max_x, max_y):
        self.max_x = max_x
        self.max_y = max_y

    def create_field(self):
        return Field(self.max_x, self.max_y)

    def __call__(self, *piece_configs: PieceConfig):
        field = self.create_field()
        try:
            field.place_pieces(piece_configs)
        except InvalidPlacementError:
            return False
        return self.evaluate(field)

    @abc.abstractmethod
    def evaluate(self, field: Field) -> bool:
        pass


class AllDifferentPieceConstraint(PieceOrderConstraint):
    """
    All pieces used in an order must be different.
    Same piece cannot be used twice.
    """

    def evaluate(self, piece_configs: list[PieceConfig]) -> bool:
        used_pieces = set()
        for piece_config in piece_configs:
            if piece_config.piece_index == -1:
                continue
            if piece_config.piece_index in used_pieces:
                return False
            used_pieces.add(piece_config.piece_index)
        return True


class OnlyLastOrdersCanBeUnusedConstraint(PieceOrderConstraint):
    """
    If an order is empty, next orders should also be empty.
    For example, we cannot have a situation where
      order 1 is a piece, order 2 is empty but then order 3 is also a piece.
    """

    def evaluate(self, piece_configs: list[PieceConfig]) -> bool:
        unused_found = False
        for piece_config in piece_configs:
            if piece_config.piece_index == -1:
                unused_found = True
            elif unused_found:
                return False
        return True


class FieldMustBeFullConstraint(FieldConstraint):
    """
    A valid solution means there are no empty squares in the field
    """

    def evaluate(self, field: Field) -> bool:
        return np.all(field.grid == 1)


class ValidAssignmentConstraint(FieldConstraint):
    """
    A valid solution means a maximum of only one piece
      can be assigned to a coordinate
    """

    def evaluate(self, field: Field) -> bool:
        return np.max(field.grid) == 1


class NoHolesOfXConstraint(FieldConstraint):
    """
    When assigning, make sure we don't create any holes larger than X
    By default, a neighbouring square hole means we already cannot fill
      in this square.
    """

    def __init__(self, max_x, max_y, max_hole_zeros=3):
        super().__init__(max_x, max_y)
        self.max_hole_zeros = max_hole_zeros

    def _get_neighbours(
            self, grid: np.ndarray, x: int, y: int
    ) -> Generator[tuple[int, int], None, None]:
        if x - 1 >= 0 and grid[y, x - 1] == 0:
            yield y, x - 1
        if y - 1 >= 0 and grid[y - 1, x] == 0:
            yield y - 1, x
        if x + 1 < self.max_x and grid[y, x + 1] == 0:
            yield y, x + 1
        if y + 1 < self.max_y and grid[y + 1, x] == 0:
            yield y + 1, x

    def _search_neighbours(
            self,
            grid: np.ndarray,
            x: int,
            y: int,
            visited: set[tuple[int, int]],
            current_count: int = 1,
    ):
        for ny, nx in self._get_neighbours(grid, x, y):
            if (ny, nx) in visited:
                continue
            visited.add((ny, nx))
            current_count = (
                    self._search_neighbours(grid, nx, ny, visited, current_count) + 1
            )
        return current_count

    def evaluate(self, field: Field) -> bool:
        visited = set()
        zero_locs = np.where(field.grid == 0)
        if not zero_locs or len(zero_locs[0]) == 0:
            return True

        for y, x in zip(*zero_locs):
            if (y, x) in visited:
                continue
            count = self._search_neighbours(field.grid, x, y, visited)
            if count <= self.max_hole_zeros:
                return False
        return True

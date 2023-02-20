from collections.abc import Iterable

import numpy as np

from pieces import PieceConfig, get_rotated_shape, get_shape_min_x_y
from util import get_min_x_y


class InvalidPlacementError(Exception):
    pass


class Field:
    def __init__(self, max_x: int, max_y: int) -> None:
        self.max_x = max_x
        self.max_y = max_y
        self.grid = np.zeros((max_y, max_x), dtype=np.int8)

    def place_pieces(
        self, piece_configs: Iterable[PieceConfig], place_label: bool = False
    ) -> None:
        for piece_config in piece_configs:
            if piece_config.order == -1:
                return
            shape = get_rotated_shape(
                piece_config.piece_index, piece_config.rotation, piece_config.reverse
            )
            x, y = get_min_x_y(self.grid)
            if x is None:
                return

            shape_min_x, shape_min_y = get_shape_min_x_y(
                piece_config.piece_index, piece_config.rotation, piece_config.reverse
            )

            if (
                x - shape_min_x < 0
                or y - shape_min_y < 0
                or x - shape_min_x + shape.shape[1] > self.max_x
                or y - shape_min_y + shape.shape[0] > self.max_y
            ):
                raise InvalidPlacementError

            if place_label:
                self.place_label(
                    shape, x - shape_min_x, y - shape_min_y, piece_config.piece_index
                )
            else:
                self.place(shape, x - shape_min_x, y - shape_min_y)

    def place(self, shape: np.ndarray, x: int, y: int) -> None:
        self.grid[y : y + shape.shape[0], x : x + shape.shape[1]] += shape

    def place_label(self, shape: np.ndarray, x: int, y: int, label: int):
        self.grid[y : y + shape.shape[0], x : x + shape.shape[1]] += shape * (label + 1)

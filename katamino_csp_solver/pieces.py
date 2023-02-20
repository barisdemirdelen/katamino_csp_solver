from collections import namedtuple
from collections.abc import Generator
from functools import lru_cache

import numpy as np

from katamino_csp_solver.util import get_min_x_y

PIECES = [
    np.array([[1, 1, 1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]

PieceConfig = namedtuple("PieceConfig", ["order", "piece_index", "rotation", "reverse"])


@lru_cache(maxsize=None)
def get_rotated_shape(shape_no: int, rotate_count: int, reverse: bool) -> np.ndarray:
    shape = PIECES[shape_no]
    rotated_shape = np.rot90(shape, k=rotate_count, axes=(0, 1))
    if reverse:
        rotated_shape = np.fliplr(rotated_shape)
    return rotated_shape


def get_shape_permutations(shape_no: int) -> Generator[tuple[int, bool], None, None]:
    shapes = set()
    for reverse in [False, True]:
        for rotate in range(4):
            current_shape = get_rotated_shape(
                shape_no, rotate_count=rotate, reverse=reverse
            )
            shape_str = str(current_shape)
            if shape_str not in shapes:
                shapes.add(shape_str)
                yield rotate, reverse


@lru_cache(maxsize=None)
def get_shape_min_x_y(
    shape_no: int, rotate_count: int, reverse: bool
) -> tuple[int, int]:
    shape = get_rotated_shape(shape_no, rotate_count, reverse)
    x, y = get_min_x_y(shape, where=1)
    return x, y

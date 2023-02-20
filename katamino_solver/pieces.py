import typing as t
from collections import namedtuple
from functools import lru_cache

import numba
import numpy as np

pieces = [
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
]

PieceConfig = namedtuple("PieceConfig", ["order", "piece_index", "rotation", "reverse"])


@lru_cache(maxsize=None)
def get_rotated_shape(shape_no: int, rotate_count: int, reverse: bool) -> np.ndarray:
    shape = pieces[shape_no]
    rotated_shape = np.rot90(shape, k=rotate_count, axes=(0, 1))
    if reverse:
        rotated_shape = np.fliplr(rotated_shape)
    return rotated_shape


def get_shape_permutations(shape_no: int) -> t.Generator[tuple[int, bool], None, None]:
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


@numba.njit(cache=True)
def place(field, shape, x, y):
    field[y : y + shape.shape[0], x : x + shape.shape[1]] += shape


def place_number(field, shape, x, y, number):
    field[y : y + shape.shape[0], x : x + shape.shape[1]] += shape * (number + 1)


@numba.njit(cache=True)
def create_field(max_x: int, max_y: int):
    return np.zeros((max_y, max_x), dtype=np.int8)


@numba.njit(cache=True)
def get_min_x_y(field, where=0):
    zero_locs = np.where(field == where)
    if not zero_locs or len(zero_locs[0]) == 0:
        return None, None
    y, x = min(zip(*zero_locs))
    return x, y


@lru_cache(maxsize=None)
def get_shape_min_x_y(
    shape_no: int, rotate_count: int, reverse: bool
) -> tuple[int, int]:
    shape = get_rotated_shape(shape_no, rotate_count, reverse)
    x, y = get_min_x_y(shape, where=1)
    return x, y

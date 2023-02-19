import typing as t
from functools import lru_cache

import numpy as np

pieces = [
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1],
              [1, 1, 1],
              [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0, 0],
              [1, 1, 0, 0],
              [0, 1, 1, 0],
              [0, 0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0],
              [0, 1, 1],
              [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 0],
              [0, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0, 0],
              [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0, 0],
              [1, 0, 0, 0],
              [1, 0, 0, 0],
              [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0],
              [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0],
              [1, 1, 1],
              [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0, 0],
              [1, 1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0],
              [1, 1, 1],
              [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1],
              [1, 1, 1]], dtype=bool),
]


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
            current_shape = get_rotated_shape(shape_no, rotate_count=rotate, reverse=reverse)
            shape_str = str(current_shape)
            if shape_str not in shapes:
                shapes.add(shape_str)
                yield rotate, reverse


def place(field, shape, x, y):
    field[y:y + shape.shape[0], x:x + shape.shape[1]] += shape


def place_number(field, shape, x, y, number):
    field[y:y + shape.shape[0], x:x + shape.shape[1]] += shape * (number + 1)

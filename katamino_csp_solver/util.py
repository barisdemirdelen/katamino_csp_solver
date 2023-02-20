import numpy as np


def get_min_x_y(grid: np.ndarray, where: int = 0) -> tuple[int | None, int | None]:
    zero_locs = np.where(grid == where)
    if not zero_locs or len(zero_locs[0]) == 0:
        return None, None
    y, x = min(zip(*zero_locs))
    return x, y

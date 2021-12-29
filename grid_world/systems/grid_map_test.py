import numpy as np

from systems.grid_map import (
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
    DIRECTION_UP,
    MAP_FLOOR,
    MAP_WALL,
    GridMap,
    closest_coords,
    direction_to_delta,
    normalize_position,
)


def test_direction_to_delta():
    cases = [
        (DIRECTION_LEFT, (-1, 0)),
        (DIRECTION_RIGHT, (1, 0)),
        (DIRECTION_UP, (0, -1)),
        (DIRECTION_DOWN, (0, 1)),
        (-1, (0, 0)),
    ]

    for arg, exp in cases:
        out = direction_to_delta(arg)
        assert out == exp


def test_closest_coords():
    cases = [
        ((0, 0, 1, 1), [(0, 0)]),
        ((0, 0, 2, 2), [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ((0, 1, 2, 2), [(0, 1), (0, 0), (1, 1), (1, 0)]),
        (
            (1, 1, 3, 3),
            [(1, 1), (0, 1), (1, 0), (1, 2), (2, 1), (0, 0), (0, 2), (2, 0), (2, 2)],
        ),
        ((0, 0, 2, 3), [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (1, 2)]),
    ]

    for arg, exp in cases:
        out = closest_coords(*arg)
        assert out == exp


def test_normalize_position():
    cases = [
        ((0, 0, 3, 3), (0, 0)),
        ((-1, -1, 3, 3), (2, 2)),
        ((3, 3, 3, 3), (0, 0)),
        ((42, 42, 5, 5), (2, 2)),
        ((-42, -42, 5, 5), (3, 3)),
    ]

    for arg, exp in cases:
        x, y, h, w = arg
        out = normalize_position(x, y, w, h)
        assert out == exp


def test_GridMap_constructor():
    cases = [
        (GridMap(np.array([[MAP_FLOOR]]), 0, 0), (0, 0)),
        (GridMap(np.array([[MAP_FLOOR, MAP_WALL]]), 0, 0), (0, 0)),
        (GridMap(np.array([[MAP_FLOOR, MAP_WALL]]), 0, 1), (0, 0)),
        (
            GridMap(
                np.array(
                    [
                        [MAP_FLOOR, MAP_WALL, MAP_WALL],
                        [MAP_WALL, MAP_WALL, MAP_FLOOR],
                        [MAP_WALL, MAP_FLOOR, MAP_FLOOR],
                    ]
                ),
                1,
                1,
            ),
            (1, 2),
        ),
        (
            GridMap(
                np.array(
                    [
                        [MAP_FLOOR, MAP_WALL, MAP_WALL],
                        [MAP_WALL, MAP_WALL, MAP_FLOOR],
                        [MAP_WALL, MAP_FLOOR, MAP_FLOOR],
                    ]
                ),
                0,
                1,
            ),
            (0, 0),
        ),
    ]

    for grid, exp in cases:
        assert (grid.x, grid.y) == exp


def test_GridMap_move():
    layout = np.array(
        [
            [MAP_WALL, MAP_FLOOR],
            [MAP_FLOOR, MAP_FLOOR],
        ]
    )

    cases = [
        ((0, 1), [(0, 1), (1, 1), (0, 1), (1, 1)]),
        ((1, 0), [(1, 1), (1, 0), (1, 1), (1, 0)]),
        ((1, 1), [(1, 0), (0, 1), (1, 0), (0, 1)]),
    ]

    for arg, exps in cases:
        init_x, init_y = arg
        for direction, exp in zip(
            [DIRECTION_UP, DIRECTION_RIGHT, DIRECTION_DOWN, DIRECTION_LEFT], exps
        ):
            grid = GridMap(layout, init_x=init_x, init_y=init_y)
            grid.move(direction)
            assert (grid.x, grid.y) == exp

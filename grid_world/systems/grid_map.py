from typing import List, Tuple

import numpy as np

MAP_FLOOR = 0
MAP_WALL = 1
MAP_CHEESE = 2

DIRECTION_UP = 0
DIRECTION_RIGHT = 1
DIRECTION_DOWN = 2
DIRECTION_LEFT = 3

# グリッドを表現する行列の列をX軸、行をY軸として上下左右の方向指定を移動量に変換する。
def direction_to_delta(direction: int) -> Tuple[int, int]:
    pass


# 大きさ (W, H) マスのグリッドの全座標について現在の (X, Y) 座標から近い順に返す。
def closest_coords(
    current_x: int,
    current_y: int,
    w: int,
    h: int,
) -> List[Tuple[int, int]]:
    pass


# 現在地 (X, Y) を大きさ (W, H) について正規化する。Xはmod(W)、Yはmod(H)となるようにする。
def normalize_position(x, y, w, h):
    pass


# グリッドマップはトーラスの展開図を表現する。
class GridMap:
    layout: np.ndarray
    x: int
    y: int

    # 初期化の際に layout について初期位置 (init_x, init_y) が壁の場合は最も近い地面を探す。
    def __init__(self, layout: np.ndarray, init_x: int = 0, init_y: int = 0):
        pass

    # 指定された方向に、移動先が壁でなければ移動してタイルの種類を返す。
    def move(self, direction: int) -> int:
        pass

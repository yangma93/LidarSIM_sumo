# -*- coding: utf-8 -*-
"""智能网联车 (CAV) 车载 LiDAR 配置与选择."""
from dataclasses import dataclass

from .config import LidarSpec

# 车型代表外廓 (长, 宽, 高) [m]; 用于复杂配置的默认安装高度与 UI 绘制
VEHICLE_CLASSES = {
    "car": (4.6, 1.85, 1.5),
    "suv": (4.8, 1.90, 1.7),
    "van": (5.3, 2.00, 2.0),
    "truck": (9.0, 2.50, 3.2),
    "bus": (11.0, 2.55, 3.2),
}


def default_sensor_height(vtype: str = "car") -> float:
    """车顶中心上方 0.2 m 的默认安装高度 (离地)."""
    return float(VEHICLE_CLASSES.get(vtype, VEHICLE_CLASSES["car"])[2]) + 0.2


@dataclass
class CavLidar:
    """车端 LiDAR: 绑定车辆 vid, 每帧根据车辆位姿动态计算传感器位姿.

    ``local_pos`` 为车辆坐标系下的安装位置 (x 前 / y 左 / z 离地高度, m);
    未设置时沿用旧行为: 车顶中心上方 ``height_offset`` 处.
    ``local_attitude`` 为 (roll, pitch, yaw) 传感器相对车体的姿态 (rad).
    """
    spec: LidarSpec
    vid: int
    height_offset: float = 0.2   # 车顶中心上方高度 [m] (local_pos 为 None 时使用)
    self_clear: float = 2.5      # 剔除自车体回波的距离阈值 [m] (避免污染 L-shape)
    local_pos: tuple = None      # (x, y, z) 车辆坐标系, z 为离地高度
    local_attitude: tuple = (0.0, 0.0, 0.0)   # (roll, pitch, yaw) rad


def select_cav_vids(traj, t_start: float, t_end: float,
                    n_cav: int = None,
                    id_list: list[int] = None,
                    max_length: float = 8.0) -> list[int]:
    """在指定时间窗内筛选可被设为 CAV 的车辆 vid.

    规则:
      - 车长 < max_length [m];
      - 若在 id_list 中, 校验存在且满足长度约束后按列表顺序保留;
      - 否则从时间窗内所有满足长度约束的个体中按 vid 升序取前 n_cav 个;
      - 总数硬上限 10 (与需求一致).
    """
    if id_list is not None:
        out = []
        for vid in id_list:
            tr = traj.tracks.get(vid)
            if tr is None:
                print(f"[CAV] 警告: vid {vid} 在轨迹中不存在, 跳过")
                continue
            if tr.length() >= max_length:
                print(f"[CAV] 警告: vid {vid} 车长 {tr.length():.1f}m >= {max_length}m, 跳过")
                continue
            out.append(vid)
        if n_cav is not None:
            out = out[:n_cav]
        return out[:10]
    eligible = []
    for vid, tr in traj.tracks.items():
        if tr.length() >= max_length:
            continue
        # 在时间窗内至少出现一次
        if not (t_start <= tr.t[-1] and t_end >= tr.t[0]):
            continue
        eligible.append(vid)
    eligible.sort()
    n = n_cav if n_cav is not None else 0
    n = max(0, min(n, len(eligible), 10))
    return eligible[:n]

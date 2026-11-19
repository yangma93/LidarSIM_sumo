# -*- coding: utf-8 -*-
"""在线交通轨迹: 把 SUMO 每一步的车辆快照接入 ``SceneBuilder``.

``SceneBuilder`` 只需要一个具备 ``tracks`` / ``active_at`` / ``vtype_dims`` 的
轨迹对象. 这里实现一个增量式容器, 每步追加/更新车辆轨迹, 使真实 SUMO 动态
交通可以复用既有的车路协同感知仿真链路, 而无需区分"离线文件导入"与
"在线联仿"两种模式.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VehicleSnapshot:
    """SUMO / TraCI 单帧车辆状态."""
    sid: str                 # SUMO 原始字符串 id
    vid: int                 # 内部整数 id
    vtype: int
    x: float
    y: float
    z: float
    angle_deg: float         # SUMO: 0=北, 顺时针
    speed: float             # m/s
    length: float
    width: float


class LiveTrack:
    """增量轨迹; 与 ``trajectory.Track`` 的 ``pose_at`` 接口保持一致."""

    def __init__(self, vid: int, vtype: int, max_hist: int = 1200):
        self.vid = vid
        self.vtype = vtype
        self.max_hist = int(max_hist)
        self.t = np.zeros(0, np.float64)
        self.front = np.zeros((0, 2), np.float64)
        self.rear = np.zeros((0, 2), np.float64)
        self.z = np.zeros(0, np.float64)
        self.speed = np.zeros(0, np.float64)
        self.width = 1.8
        self.length_m = 5.0
        self.ground_fn = None

    def update(self, t: float, x: float, y: float, z: float,
               angle_deg: float, speed: float, length: float, width: float):
        ang = np.deg2rad(float(angle_deg))
        d = np.array([np.sin(ang), np.cos(ang)])     # SUMO 0=北 顺时针
        L = max(float(length), 0.5)
        self.length_m = L
        self.width = max(float(width), 0.3)
        c = np.array([float(x), float(y)])
        if len(self.t) and abs(self.t[-1] - t) < 1e-6:
            self.t[-1] = t
            self.front[-1] = c + d * L / 2
            self.rear[-1] = c - d * L / 2
            self.z[-1] = z
            self.speed[-1] = speed
        else:
            self.t = np.append(self.t, t)
            self.front = np.vstack([self.front, c + d * L / 2])
            self.rear = np.vstack([self.rear, c - d * L / 2])
            self.z = np.append(self.z, z)
            self.speed = np.append(self.speed, speed)
        if len(self.t) > self.max_hist:
            keep = len(self.t) - self.max_hist
            self.t = self.t[keep:]
            self.front = self.front[keep:]
            self.rear = self.rear[keep:]
            self.z = self.z[keep:]
            self.speed = self.speed[keep:]

    def length(self):
        """与 trajectory.Track.length() 接口一致."""
        return float(self.length_m)

    def pose_at(self, tq: np.ndarray):
        tq = np.atleast_1d(np.asarray(tq, np.float64))
        if len(self.t) == 0:
            n = len(tq)
            return (np.zeros((n, 3)), np.zeros(n),
                    np.zeros((n, 2)), np.zeros((n, 2)))
        tq = np.clip(tq, self.t[0], self.t[-1])
        yaw_raw = np.arctan2(self.front[:, 1] - self.rear[:, 1],
                             self.front[:, 0] - self.rear[:, 0])
        yaw_u = np.unwrap(yaw_raw) if len(yaw_raw) > 1 else yaw_raw
        fu = np.stack([np.interp(tq, self.t, self.front[:, i]) for i in range(2)], 1)
        ru = np.stack([np.interp(tq, self.t, self.rear[:, i]) for i in range(2)], 1)
        yaw = np.interp(tq, self.t, yaw_u) if len(yaw_u) > 1 else np.full(len(tq), yaw_u[0])
        pos = (fu + ru) * 0.5
        return pos, yaw, fu, ru

    def speed_at(self, t: float) -> float:
        if len(self.t) == 0:
            return 0.0
        return float(np.interp(t, self.t, self.speed))


class OnlineTrajectories:
    """在线轨迹容器, 逐步接收 SUMO 快照."""

    def __init__(self):
        self.tracks: dict[int, LiveTrack] = {}
        self.sid_to_vid: dict[str, int] = {}
        self.vid_to_sid: dict[int, str] = {}
        self.vtype_map: dict[str, int] = {}
        self._active: list[int] = []
        self._t = 0.0
        self.source_format = "sumo-live"
        self.vid_map: dict[int, str] = {}
        self._max_hist = 1200
        self._last_seen: dict[int, float] = {}
        self._max_age = 120.0

    def reset(self):
        self.tracks.clear()
        self.sid_to_vid.clear()
        self.vid_to_sid.clear()
        self.vtype_map.clear()
        self._active.clear()
        self._last_seen.clear()
        self._t = 0.0

    def update(self, t: float, snapshots: list[VehicleSnapshot]):
        self._t = float(t)
        active = []
        for s in snapshots:
            tr = self.tracks.get(s.vid)
            if tr is None:
                tr = LiveTrack(s.vid, s.vtype, self._max_hist)
                self.tracks[s.vid] = tr
            tr.vtype = s.vtype
            tr.update(t, s.x, s.y, s.z, s.angle_deg, s.speed, s.length, s.width)
            self.sid_to_vid[s.sid] = s.vid
            self.vid_to_sid[s.vid] = s.sid
            active.append(s.vid)
            self._last_seen[s.vid] = t
        self._active = active
        # 长时间未出现的车辆释放轨迹, 防止联仿长时间运行内存单调增长
        stale = [v for v, ts in self._last_seen.items() if t - ts > self._max_age]
        for v in stale:
            self.tracks.pop(v, None)
            self._last_seen.pop(v, None)
            sid = self.vid_to_sid.pop(v, None)
            if sid is not None:
                self.sid_to_vid.pop(sid, None)

    def active_at(self, t: float) -> np.ndarray:
        return np.asarray(self._active, np.int64)

    @property
    def frame_times(self) -> np.ndarray:
        return np.asarray([self._t], np.float64)

    @property
    def current_time(self) -> float:
        return self._t

    def vtype_dims(self) -> dict[int, tuple[float, float]]:
        out: dict[int, list] = {}
        for tr in self.tracks.values():
            out.setdefault(tr.vtype, []).append((tr.length(), tr.width))
        return {k: (float(np.median([a for a, _ in v])),
                    float(np.median([b for _, b in v])))
                for k, v in out.items()}

    def speed_of(self, vid: int) -> float:
        tr = self.tracks.get(vid)
        return tr.speed_at(self._t) if tr is not None else 0.0

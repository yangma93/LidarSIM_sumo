# -*- coding: utf-8 -*-
"""CAV 复杂配置: 多车型 + 多传感器安装位姿 + 分车型渗透率.

该模块不依赖 Qt, 供主界面 / SUMO 联仿 / 命令行共用. 生成的 ``CavLidar``
与现有简单配置完全一致地进入 ``LidarSimulator._cav_rays``, 因此求交、
运动伪影、强度与漏检逻辑不做任何分叉.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

from .cav import CavLidar, VEHICLE_CLASSES, default_sensor_height

# 车型 -> (长, 宽, 高) [m]; 仅用于 UI 绘图与默认安装高度
VEHICLE_CLASS_LABELS = {
    "car": ("轿车", "Car"),
    "suv": ("SUV", "SUV"),
    "van": ("面包车", "Van"),
    "truck": ("卡车", "Truck"),
    "bus": ("客车", "Bus"),
}


def hash01(key) -> float:
    """稳定散列 -> [0,1), 用于渗透率的确定性抽样."""
    return (zlib.crc32(str(key).encode("utf-8")) % 10000) / 10000.0


@dataclass
class CavSensor:
    """车辆坐标系下的单个车端传感器 (x 前 / y 左 / z 离地高度)."""
    preset: str = "vlp16"
    x: float = 0.0
    y: float = 0.0
    z: float = 1.6
    roll: float = 0.0      # deg
    yaw: float = 0.0       # deg (相对车头)
    pitch: float = 0.0     # deg

    def pose_local(self):
        return (float(self.x), float(self.y), float(self.z))

    def attitude_rad(self):
        return (math.radians(self.roll), math.radians(self.pitch),
                math.radians(self.yaw))

    def copy(self) -> "CavSensor":
        return CavSensor(**vars(self))


@dataclass
class CavTypeConfig:
    """一种 CAV 车型的传感器组合与渗透率."""
    vtype: str = "car"
    penetration: float = 0.1
    sensors: list = field(default_factory=list)

    def ensure_sensors(self, n: int):
        n = max(1, int(n))
        while len(self.sensors) < n:
            self.sensors.append(CavSensor(
                preset="vlp16", z=default_sensor_height(self.vtype)))
        del self.sensors[n:]

    def copy(self) -> "CavTypeConfig":
        return CavTypeConfig(vtype=self.vtype, penetration=self.penetration,
                             sensors=[s.copy() for s in self.sensors])


@dataclass
class CavComplexConfig:
    """CAV 复杂配置总入口."""
    enabled: bool = False
    types: list = field(default_factory=lambda: [CavTypeConfig(
        vtype="car", penetration=0.1,
        sensors=[CavSensor(preset="vlp16", z=default_sensor_height("car"))])])

    # ---- 类型管理 ----
    def copy(self) -> "CavComplexConfig":
        return CavComplexConfig(enabled=self.enabled,
                                types=[t.copy() for t in self.types])

    def ensure_types(self, n: int):
        n = max(1, int(n))
        used = [t.vtype for t in self.types]
        while len(self.types) < n:
            cls = next((c for c in VEHICLE_CLASSES if c not in used), "car")
            used.append(cls)
            self.types.append(CavTypeConfig(
                vtype=cls, penetration=0.1,
                sensors=[CavSensor(preset="vlp16", z=default_sensor_height(cls))]))
        del self.types[n:]

    def type_for(self, vtype: str):
        for t in self.types:
            if t.vtype == vtype:
                return t
        return None

    # ---- 仿真接入 ----
    def cav_lidars_for(self, vid, vtype: str, self_clear: float = 2.5,
                       force: bool = False):
        """按车型渗透率决定该个体是否配置车端雷达, 返回 CavLidar 列表."""
        if not self.enabled:
            return []
        cfg = self.type_for(vtype)
        if cfg is None or not cfg.sensors:
            return []
        if not force and hash01(vid) >= float(np_clip(cfg.penetration)):
            return []
        out = []
        for s in cfg.sensors:
            out.append(CavLidar(
                spec=make_spec(s.preset), vid=int(vid),
                height_offset=0.0, self_clear=float(self_clear),
                local_pos=s.pose_local(), local_attitude=s.attitude_rad()))
        return out

    def summary(self) -> str:
        if not self.enabled:
            return "未启用"
        items = []
        for t in self.types:
            items.append(f"{t.vtype} {t.penetration:.2f}×{len(t.sensors)}")
        return "启用: " + ", ".join(items)

    # ---- 序列化 (SUMO 联仿线程) ----
    def to_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "types": [{
                "vtype": t.vtype,
                "penetration": float(t.penetration),
                "sensors": [vars(s) for s in t.sensors],
            } for t in self.types],
        }

    @classmethod
    def from_dict(cls, data) -> "CavComplexConfig":
        cfg = cls(enabled=False, types=[])
        if not data:
            return cfg
        cfg.enabled = bool(data.get("enabled", False))
        for item in data.get("types", []) or []:
            sensors = []
            for s in item.get("sensors", []) or []:
                sensors.append(CavSensor(
                    preset=s.get("preset", "vlp16"),
                    x=float(s.get("x", 0.0)), y=float(s.get("y", 0.0)),
                    z=float(s.get("z", 1.6)),
                    roll=float(s.get("roll", 0.0)),
                    yaw=float(s.get("yaw", 0.0)),
                    pitch=float(s.get("pitch", 0.0))))
            cfg.types.append(CavTypeConfig(
                vtype=item.get("vtype", "car"),
                penetration=float(item.get("penetration", 0.1)),
                sensors=sensors or [CavSensor(
                    preset="vlp16",
                    z=default_sensor_height(item.get("vtype", "car")))]))
        if not cfg.types:
            cfg.ensure_types(1)
        return cfg


def np_clip(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def make_spec(preset: str):
    from .config import make_lidar
    return make_lidar(preset)

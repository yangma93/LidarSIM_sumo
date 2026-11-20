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

    def ensure_sensors(self, n: int, preset: str = None):
        """调整传感器数量.

        ``preset`` 为新传感器继承的型号: 未指定时沿用最后一个传感器的型号
        (而不是一律 vlp16) —— 否则"再加一个 Livox"会悄悄变成 vlp16, 与用户
        在界面上的配置意图不符.
        """
        n = max(1, int(n))
        base = preset or (self.sensors[-1].preset if self.sensors else "vlp16")
        z = default_sensor_height(self.vtype)
        while len(self.sensors) < n:
            self.sensors.append(CavSensor(preset=base, z=z))
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
            # 新车型继承上一种车型的传感器组合与渗透率, 避免"加了一个车型又
            # 回到单个 vlp16"; 用户可在界面上再按车型改成不同雷达.
            src = self.types[-1] if self.types else None
            sensors = ([s.copy() for s in src.sensors] if src and src.sensors
                       else [CavSensor(preset="vlp16",
                                       z=default_sensor_height(cls))])
            self.types.append(CavTypeConfig(
                vtype=cls, penetration=(src.penetration if src else 0.1),
                sensors=sensors))
        del self.types[n:]

    def type_for(self, vtype: str):
        for t in self.types:
            if t.vtype == vtype:
                return t
        return None

    # ---- 仿真接入 ----
    def lidars_for_type(self, vid, cfg: "CavTypeConfig", force: bool = False):
        """按给定车型配置生成传感器列表 (不含车型/渗透率判定).

        多传感器时附加 ``tag`` (#1/#2...), 使输出点云名
        ``livox_horizon#2@cav12`` 能区分同车的不同雷达, 便于联仿反馈与排查.
        """
        if cfg is None or not cfg.sensors:
            return []
        force = bool(force)
        if not force and hash01(vid) >= float(np_clip(cfg.penetration)):
            return []
        multi = len(cfg.sensors) > 1
        out = []
        for i, s in enumerate(cfg.sensors):
            out.append(CavLidar(
                spec=make_spec(s.preset), vid=int(vid),
                height_offset=0.0,
                local_pos=s.pose_local(),
                local_attitude=s.attitude_rad(),
                tag=(f"#{i + 1}" if multi else "")))
        return out

    def cav_lidars_for(self, vid, vtype: str, force: bool = False):
        """按车型渗透率决定该个体是否配置车端雷达, 返回 CavLidar 列表."""
        if not self.enabled:
            return []
        return self.lidars_for_type(vid, self.type_for(vtype), force=force)

    def summary(self) -> str:
        """状态栏/按钮旁的一行摘要 (含型号, 便于确认复杂配置是否真的生效)."""
        if not self.enabled:
            return "未启用"
        items = []
        for t in self.types:
            models = "+".join(dict.fromkeys(s.preset for s in t.sensors))
            items.append(f"{t.vtype} {t.penetration:.2f}×{len(t.sensors)}({models})")
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

# -*- coding: utf-8 -*-
"""点云语义类别编码 (自适应背景数据集).

不同点云数据集的语义编码完全不同 (例如 SemanticKITTI 1..19,
Waymo 类 100100.., 城市模型 6..41). 因此:

* 背景点原样保留数据集自带的编码;
* 仿真前景目标 (车辆等) 的编码基址根据背景编码自适应: 取
  ``max(FOREGROUND_SEM_BASE, 背景最大类别 + 1)``, 再依次为
  car/suv/van/truck/bus/cyclist/ped/other 分配 ``基址+1 .. 基址+8``,
  保证与背景标签严格不重叠;
* 未启用语义或背景点云没有语义属性时, 所有点默认类别 0 (unlabeled).
"""
import numpy as np

# 前景目标语义编码缺省基址 (背景类别通常 < 100)
FOREGROUND_SEM_BASE = 100

# 前景类别顺序: 编码 = base + 1 + index
OBJECT_ORDER = ("car", "suv", "van", "truck", "bus", "cyclist", "ped", "other")

OBJECT_SEMANTIC = {c: FOREGROUND_SEM_BASE + i + 1
                   for i, c in enumerate(OBJECT_ORDER)}

SEMANTIC_NAMES = {
    0: "unlabeled",
    **{FOREGROUND_SEM_BASE + i + 1: c for i, c in enumerate(OBJECT_ORDER)},
}


def foreground_base(background) -> int:
    """根据背景语义编码确定前景编码基址 (严格大于背景最大值)."""
    if background is None:
        return FOREGROUND_SEM_BASE
    arr = np.asarray(background)
    if arr.size == 0:
        return FOREGROUND_SEM_BASE
    valid = arr[arr >= 0]
    if valid.size == 0:
        return FOREGROUND_SEM_BASE
    return max(FOREGROUND_SEM_BASE, int(valid.max()) + 1)


def object_semantic_class(cls, base: int = FOREGROUND_SEM_BASE) -> int:
    """车型类别名 -> 前景语义编码 (未识别类别记为 other)."""
    name = str(cls or "").strip().lower()
    idx = OBJECT_ORDER.index(name) if name in OBJECT_ORDER else len(OBJECT_ORDER) - 1
    return int(base) + idx + 1


def vtype_label(traj, vtype) -> str:
    """取 vType 的原始名称; 兼容 {int: str} 与 {str: int} 两种映射."""
    vm = getattr(traj, "vtype_map", None) or {}
    val = vm.get(vtype)
    if isinstance(val, str):
        return val
    if isinstance(val, int):
        inv = {v: k for k, v in vm.items() if isinstance(k, str)}
        return inv.get(val, "")
    return ""


def vehicle_class(traj, vtype, mesh_name: str = "") -> str:
    """推断车辆类别: 优先 SUMO/VISSIM 车型名, 否则按网格模型名."""
    from .cav import class_from_label
    from .meshes import class_of_mesh
    cls = class_from_label(vtype_label(traj, vtype))
    if cls:
        return cls
    try:
        return class_of_mesh(mesh_name)
    except Exception:                                             # noqa: BLE001
        return "car"


def align_semantics(sem, keep: np.ndarray):
    """按布尔掩码裁剪语义数组; sem 为 None 时保持 None."""
    if sem is None:
        return None
    return np.asarray(sem)[keep]

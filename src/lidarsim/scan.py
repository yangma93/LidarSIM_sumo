# -*- coding: utf-8 -*-
"""LiDAR 扫描模式生成与传感器位姿.

位姿约定 [x y z roll yaw pitch] (世界坐标系, 单位 m / rad):
  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
旋转扫描: 方位角列在扫描周期内匀速旋转, 每列光束带独立时间戳,
对运动车辆按光束时刻插值位姿 -> 自然产生运动伪影 (rolling shutter).
"""
import numpy as np

from .config import LidarSpec


def pose_matrix(pose) -> np.ndarray:
    x, y, z, roll, yaw, pitch = [float(v) for v in pose]
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    return R


def scan_pattern(spec: LidarSpec, pose, phase: float = 0.0):
    """生成一帧扫描的所有光束.

    返回 origins (R,3), dirs (R,3) 单位向量, t_frac (R,) 光束在扫描周期内的
    相对时刻 [0,1) (用于车辆位姿插值 -> 运动伪影).
    ``phase`` 仅对非重复扫描有效, 用于帧间图案漂移.
    """
    R_l = pose_matrix(pose)
    origin = np.asarray(pose[:3], np.float64)
    if spec.scan_mode != "rotary":
        n = max(1, spec.n_rays)
        u = (np.arange(n, dtype=np.float64) + 0.5) / n
        hfov = np.radians(spec.fov_h_deg)
        vmid = np.radians(0.5 * (spec.vfov_min_deg + spec.vfov_max_deg))
        vspan = np.radians(spec.vfov_max_deg - spec.vfov_min_deg)
        ph = 2.0 * np.pi * float(phase)
        az = -0.5 * hfov + hfov * (0.5 + 0.5 * np.sin(
            2.0 * np.pi * 11.0 * u + ph))
        el = vmid - 0.5 * vspan + vspan * (0.5 + 0.5 * np.sin(
            2.0 * np.pi * 7.0 * np.sqrt(2.0) * u + 1.7 * ph))
        ct = np.cos(el)
        dirs = np.stack([ct * np.cos(az), ct * np.sin(az), np.sin(el)], -1)
        dirs_w = dirs @ R_l.T
        origins = np.broadcast_to(origin, dirs_w.shape).copy()
        return (origins.astype(np.float32), dirs_w.astype(np.float32),
                np.ascontiguousarray(u, np.float32))
    n_az = int(round(spec.fov_h_deg / spec.h_res_deg)) if spec.fov_h_deg < 360 \
        else int(round(360.0 / spec.h_res_deg))
    if spec.fov_h_deg < 360:
        az = np.linspace(-spec.fov_h_deg / 2, spec.fov_h_deg / 2, n_az, endpoint=False)
    else:
        az = np.arange(n_az) * spec.h_res_deg  # [0,360)
    elev = np.linspace(spec.vfov_min_deg, spec.vfov_max_deg, spec.channels) * np.pi / 180
    az_r = az * np.pi / 180
    # 网格: 每列 (方位) x 每通道
    AZ, EL = np.meshgrid(az_r, elev, indexing="ij")     # (n_az, ch)
    ct = np.cos(EL)
    dirs = np.stack([ct * np.cos(AZ), ct * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
    dirs_w = dirs @ R_l.T
    # 光束时刻: 方位角在扫描周期内线性推进.
    # 扇形雷达 (fov_h_deg < 360) 的方位列原本从 -FOV/2 起算, 直接除以 FOV 会得到
    # [-0.5, 0.5) 的负时刻 —— 一半光束的位姿被外推/钳制到帧起点, rolling shutter
    # (运动伪影) 失效, t_beam 也不再满足 [0,1) 的约定. 这里平移到 [0,1).
    if spec.fov_h_deg < 360:
        t_frac_col = (AZ[:, 0] + 0.5 * (spec.fov_h_deg * np.pi / 180)) / \
            (spec.fov_h_deg * np.pi / 180)
    else:
        t_frac_col = az_r / (2 * np.pi)
    t_frac = np.repeat(t_frac_col, spec.channels)
    # float32 舍入可能把最后一列推到 1.0 (与下一帧首列同一时刻), 兜底压回 [0,1)
    t_frac = np.clip(t_frac, 0.0, np.nextafter(1.0, 0.0))
    origins = np.broadcast_to(origin, dirs_w.shape).copy()
    return (origins.astype(np.float32), dirs_w.astype(np.float32),
            np.ascontiguousarray(t_frac, np.float32))

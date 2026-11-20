# -*- coding: utf-8 -*-
"""帧场景重构: 环境点云 + 轨迹驱动的网格模型 -> 融合三维场景."""
from dataclasses import dataclass, field

import numpy as np

from .pointcloud import EnvCloud, GroundModel, estimate_normals
from .trajectory import Trajectories, Track
from .meshes import ModelLibrary, Mesh
from .semantics import vehicle_class


def voxel_downsample(points: np.ndarray, voxel: float, values: list[np.ndarray] = None,
                     return_index: bool = False):
    """体素降采样, values 为逐点标量属性 (体素内取均值).

    ``return_index=True`` 时额外返回每个体素代表点的原始索引 (体素内首点),
    用于让离散属性 (如语义类别) 与降采样后的点一一对齐.
    """
    ij = np.floor(points / voxel).astype(np.int64)
    key = np.stack([ij[:, 0], ij[:, 1], ij[:, 2]], 1)
    uniq, idx, inv = np.unique(key, axis=0, return_index=True,
                               return_inverse=True)
    center = (uniq + 0.5) * voxel
    out = [center.astype(np.float32)]
    inv64 = inv.astype(np.int64)
    cnt = np.bincount(inv64, minlength=len(uniq)).astype(np.float64)
    for v in (values or []):
        s = np.bincount(inv64, v.astype(np.float64), minlength=len(uniq))
        out.append((s / cnt).astype(np.float32))
    if return_index:
        return out, idx
    return out


@dataclass
class VehicleInstance:
    vid: int
    vtype: int
    track: Track
    mesh: Mesh
    pos: np.ndarray        # (3,) 车底几何中心 (地面高程已抬升)
    yaw: float
    length: float
    width: float
    height: float
    refl: float
    ground_fn: object = None   # callable(xy (m,2)) -> (m,) 地面高程
    cls: str = "car"           # 车型类别 (car/suv/van/truck/bus/cyclist/ped), 语义编码用

    def obb(self) -> np.ndarray:
        """真值 OBB 8 角点 (8,3)."""
        L, W, H = self.length / 2, self.width / 2, self.height
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        local = np.array([[sx * L, sy * W, sz * H]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)])
        return self.pos + local @ R.T

    def pose_at(self, tq: np.ndarray):
        """给定一束时间戳 -> (pos(m,3), yaw(m,)) 用于运动伪影仿真."""
        p, yaw, _, _ = self.track.pose_at(tq)
        if self.ground_fn is not None:
            z = self.ground_fn(p)
        else:
            z = np.full(len(p), self.pos[2])
        return np.stack([p[:, 0], p[:, 1], z], 1), yaw


@dataclass
class SceneFrame:
    t: float
    env: EnvCloud                       # 仿真用环境 (降采样+法向)
    env_viz: EnvCloud = None            # 可视化用环境 (原始)
    vehicles: list = field(default_factory=list)


class SceneBuilder:
    """初始化一次 (轨迹/环境/地面/模型库), 之后按帧构建融合场景."""

    def __init__(self, traj: Trajectories, env: EnvCloud, models: ModelLibrary,
                 ground_cell: float = 1.5, ground_pctl: float = 5.0,
                 ground_tol: float = 0.30, env_voxel: float = 0.10,
                 normal_voxel: float = 0.25, verbose: bool = True):
        self.traj = traj
        self.models = models
        self.verbose = verbose
        # 地面模型 (基于原始环境点云)
        self.ground = GroundModel(env.points, cell=ground_cell, pctl=ground_pctl)
        self.ground_tol = ground_tol
        # 仿真用环境: 体素降采样 + 反射率均值 + 法向
        (pts, refl), vidx = voxel_downsample(
            env.points, env_voxel,
            [env.reflectance if env.reflectance is not None
             else np.full(len(env.points), 0.2, np.float32)],
            return_index=True)
        normals = estimate_normals(pts, normal_voxel)
        # 语义类别按体素代表点继承 (离散属性不能取均值), 保证与降采样点对齐
        sem_sim = None
        if env.semantics is not None and len(env.semantics) == len(env.points):
            sem_sim = np.asarray(env.semantics)[vidx].astype(np.int32)
        self.env_sim = EnvCloud(points=pts, reflectance=refl, normals=normals,
                                semantics=sem_sim, rgb=env.rgb)
        self.env_full = env
        # vType -> 网格
        vmap = models.match_vtypes(traj.vtype_dims())
        self.vtype_mesh = vmap
        if verbose:
            print("[SceneBuilder] vType->模型匹配:")
            for vt, m in sorted(vmap.items()):
                print(f"    vType {vt:4d} -> {m.name:10s} dims(L,W,H)=({m.dims[0]:.2f},{m.dims[1]:.2f},{m.dims[2]:.2f}) refl={m.refl:.2f}")
            g0 = float(self.ground.grid.min()); g1 = float(self.ground.grid.max())
            print(f"[SceneBuilder] 地面高程范围: {g0:.2f} ~ {g1:.2f} m; 仿真环境点 {len(pts)}")
        # 车辆轨迹注入地面高程函数
        for tr in traj.tracks.values():
            tr.ground_height = lambda xy, _g=self.ground: _g.height(xy)

    def build(self, t: float) -> SceneFrame:
        vids = self.traj.active_at(t)
        vehicles = []
        for v in vids:
            tr = self.traj.tracks[v]
            mesh = self.vtype_mesh.get(tr.vtype)
            if mesh is None:          # 在线联仿: 运行中新出现的车辆类型按需匹配
                mesh = self.models.match(tr.length(), tr.width)
                self.vtype_mesh[tr.vtype] = mesh
            (p, yaw, fr, rr) = tr.pose_at(np.array([t]))
            pos2 = p[0]
            gz = float(self.ground.height(pos2[None, :])[0])
            veh = VehicleInstance(
                vid=v, vtype=tr.vtype, track=tr, mesh=mesh,
                pos=np.array([pos2[0], pos2[1], gz]), yaw=float(yaw[0]),
                length=float(tr.length()), width=tr.width, height=float(mesh.dims[2]),
                refl=mesh.refl, ground_fn=self.ground.height,
                cls=vehicle_class(self.traj, tr.vtype, mesh.name))
            vehicles.append(veh)
        return SceneFrame(t=t, env=self.env_sim, env_viz=None, vehicles=vehicles)

    def build_with_viz(self, t: float) -> SceneFrame:
        f = self.build(t)
        f.env_viz = self.env_full
        return f

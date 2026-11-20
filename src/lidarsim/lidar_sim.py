# -*- coding: utf-8 -*-
"""多 LiDAR 帧级点云仿真: 光线生成 -> GPU 求交 -> 物理强度 -> 噪声/漏检."""
from dataclasses import dataclass

import numpy as np

from .config import LidarSpec, SimParams
from .scan import scan_pattern
from .raycast import EnvGrid, get_xp, ray_aabb_one, raycast_vehicle
from .scene import SceneBuilder, SceneFrame
from .weather import WeatherConfig, generate_clutter
from .cav import CavLidar
from .semantics import object_semantic_class, foreground_base


@dataclass
class LidarPose:
    spec: LidarSpec
    pose: list          # [x y z roll yaw pitch] 世界坐标系


@dataclass
class SimCloud:
    lidar_name: str
    origin: np.ndarray       # (3,)
    xyz: np.ndarray          # (N,3) 世界系
    intensity: np.ndarray    # (N,) DN [0,255]
    label: np.ndarray        # (N,) int 0=环境, >0=车辆 vid (真值关联)
    t_beam: np.ndarray       # (N,) 光束帧内相对时刻
    range_m: np.ndarray      # (N,)
    semantics: np.ndarray = None   # (N,) int 语义类别 (None=未启用)
    semantic_base: int = 0         # 前景目标语义编码基址 (自适应背景编码)


def compute_intensity(ranges, cos_inc, refl, spec: LidarSpec, rng,
                      ext_extra: float = 0.0):
    """物理强度模型 (漫反射扩展目标雷达方程 + 传感器近场压缩):

        P_r ∝ rho * |cos(theta_inc)| * exp(-2*(alpha_atm+ext_extra)*R) / (1 + (R/R0)^2)

    R0 为传感器近场压缩参考距离 (真实接收链路 AGC 的等效模型);
    ext_extra 为雨/雪等引入的附加消光系数 [1/m] (WeatherConfig.sigma_ext);
    I_DN = clip(gain * P_r, 0, 1) * 255 + 高斯噪声.
    漏检: 归一化接收功率低于阈值 (掠射/超远) 或随机丢失.
    """
    R0 = spec.range_ref
    atm = np.exp(-2.0 * (spec.alpha_atm + ext_extra) * ranges)
    p = refl * np.abs(cos_inc) * atm / (1.0 + (ranges / R0) ** 2)
    dn = np.clip(spec.gain * p, 0.0, 1.0) * 255.0
    dn = dn + rng.normal(0.0, spec.noise_intensity, len(dn))
    keep = p > spec.dropout_thr
    keep &= rng.random(len(dn)) > spec.dropout_random
    return np.clip(dn, 0, 255).astype(np.float32), keep


def scan_phase(t: float) -> float:
    """非重复扫描的帧间相位 (黄金比例步进 -> 多帧覆盖不重复)."""
    return (float(t) * 0.6180339887498949) % 1.0


class LidarSimulator:
    def __init__(self, builder: SceneBuilder, lidars: list[LidarPose],
                 params: SimParams = None, cav_lidars: list[CavLidar] = None,
                 semantics: bool = False):
        self.builder = builder
        self.lidars = lidars
        self.cav_lidars = cav_lidars or []
        self.params = params or SimParams()
        # 点云语义: 环境点继承背景点云语义 (无则 0); 前景目标编码基址按背景
        # 标签自适应 (max(100, 背景最大类别+1)), 保证与任意数据集编码不重叠
        self.semantics = bool(semantics)
        env_sem = getattr(builder.env_sim, "semantics", None)
        self.env_sem_np = (np.asarray(env_sem, np.int32)
                           if env_sem is not None
                           and len(env_sem) == len(builder.env_sim.points)
                           else None)
        self.sem_fg_base = foreground_base(self.env_sem_np)
        self.xp, self.gpu = get_xp(self.params.device)
        xp = self.xp
        self.grid = EnvGrid(builder.env_sim.points, self.params.grid_cell, xp, self.gpu)
        env = builder.env_sim
        self.env_refl_np = (env.reflectance if env.reflectance is not None
                            else np.full(len(env.points), 0.2, np.float32))
        self.env_nrm_np = env.normals
        self.rng = np.random.default_rng(self.params.seed)
        self._patterns = [scan_pattern(l.spec, l.pose) for l in lidars]
        self._pat_gpu = [(xp.asarray(o), xp.asarray(d)) for o, d, _ in self._patterns] \
            if self.gpu else None
        self._mesh_cache = {}
        # 车端雷达: 零位姿扫描模式按 spec 缓存 (逐帧重建网格纯属浪费)
        self._zero_pat_cache: dict[tuple, tuple] = {}
        # 三角面法向 CPU 副本 (强度计算用), 按网格名缓存
        self._mesh_nrm_cpu: dict[str, np.ndarray] = {}
        # 路侧雷达环境求交缓存: li -> (t_env, pt_idx)
        # 路侧雷达位姿/扫描方向固定, 环境点云不变 -> 环境命中深度图每帧相同, 可复用.
        # CAV 位姿逐帧变化, 不进入该缓存.
        self._env_cache: dict[int, tuple] = {}
        # CAV 上一帧环境深度图 (逐束): (vid, tag) -> {"t_env", "o", "d"}
        # 同一车辆可能装多个雷达, 必须按传感器区分, 否则会拿另一雷达的深度图
        # 当本帧求交上界 (启用 cav_cap_enable 时会漏检).
        self._cav_state: dict[tuple, dict] = {}
        # 传感器净空检查: 位姿嵌入遮挡物时给出警告 (视线将被拦截)
        for lp in lidars:
            near = self.grid.points_in_cell(np.asarray(lp.pose[:3], np.float64))
            if len(near):
                dmin = float(np.linalg.norm(
                    near - np.asarray(lp.pose[:3]), axis=1).min())
                print(f"[LidarSimulator] 警告: 雷达 '{lp.spec.name}' 位姿 "
                      f"({lp.pose[0]:.1f},{lp.pose[1]:.1f},{lp.pose[2]:.1f}) "
                      f"距最近环境点仅 {dmin:.2f} m —— 可能嵌入树冠/建筑等遮挡物, "
                      f"大部分视线将被拦截, 建议调整位姿 (如移至路面上方)")

    # ------------------------------------------------------------------
    def set_cav_lidars(self, cavs):
        """在线联仿: CAV 集合逐帧变化, 允许运行时更新车端雷达列表."""
        self.cav_lidars = list(cavs or [])

    def simulate_frame(self, t: float) -> tuple[SceneFrame, list[SimCloud]]:
        scene = self.builder.build(t)
        clouds = []
        for li, lp in enumerate(self.lidars):
            clouds.append(self._simulate_one(scene, li, lp, t))
        clouds.extend(self._simulate_cavs(self.cav_lidars, scene, t))
        return scene, clouds

    def _simulate_one(self, scene: SceneFrame, li: int, lp: LidarPose,
                      t: float) -> SimCloud:
        if lp.spec.scan_mode != "rotary":
            # 非重复扫描: 每帧图案相位漂移, 不能复用环境深度图缓存
            o_np, d_np, tf_np = scan_pattern(lp.spec, lp.pose,
                                             phase=scan_phase(t))
            od_gpu = ((self.xp.asarray(o_np), self.xp.asarray(d_np))
                      if self.gpu else None)
            return self._simulate_pattern(scene, lp.spec, lp.spec.name, t,
                                          o_np, d_np, tf_np,
                                          np.asarray(lp.pose[:3], np.float32),
                                          od_gpu, env_hit=None)
        o_np, d_np, tf_np = self._patterns[li]
        od_gpu = self._pat_gpu[li] if self.gpu else None
        # 路侧雷达: 环境求交结果 (深度图) 可跨帧复用
        if li not in self._env_cache:
            o, d = od_gpu if self.gpu else (o_np, d_np)
            self._env_cache[li] = self._env_raycast(lp.spec, o, d, len(o_np))
        env_hit = self._env_cache[li]
        return self._simulate_pattern(scene, lp.spec, lp.spec.name, t,
                                      o_np, d_np, tf_np,
                                      np.asarray(lp.pose[:3], np.float32), od_gpu,
                                      env_hit=env_hit)

    def _env_raycast(self, spec: LidarSpec, o, d, R: int):
        """对环境点云求交 (光线-点, 束半径判据), 返回 (t_env, pt_idx)."""
        return self.grid.raycast(
            o, d, spec.range_min, np.full(R, spec.range_max, np.float32),
            float(np.tan(np.radians(spec.beam_divergence_mrad) / 2)))

    # ------------------------------------------------------------------
    # 车端雷达: 多 CAV 合批
    # ------------------------------------------------------------------
    def _cav_pattern(self, spec: LidarSpec, roll: float, yaw_s: float,
                     pitch: float, t: float, frame_cache: dict):
        """传感器体坐标系下的基准扫描方向 (零位移, 含安装姿态).

        旋转/转镜雷达的光束图案与帧无关 -> 实例级缓存;
        非重复扫描随黄金比例相位逐帧漂移 -> 帧内缓存. 同一帧内所有型号与
        安装姿态相同的传感器 (不管装在哪辆车上) 共用一份图案, 多车多雷达
        时不再重复生成数万条光束.
        """
        key = (spec.channels, spec.vfov_min_deg, spec.vfov_max_deg,
               spec.fov_h_deg, spec.h_res_deg, spec.scan_mode,
               int(spec.points_per_sec),
               round(roll, 6), round(yaw_s, 6), round(pitch, 6))
        att = [0.0, 0.0, 0.0, roll, yaw_s, pitch]
        if spec.scan_mode != "rotary":
            ph = round(scan_phase(t), 9)
            fkey = key + (ph,)
            pat = frame_cache.get(fkey)
            if pat is None:
                _, d0, tf_np = scan_pattern(spec, att, phase=ph)
                pat = (d0, tf_np)
                frame_cache[fkey] = pat
            return pat
        pat = self._zero_pat_cache.get(key)
        if pat is None:
            _, d0, tf_np = scan_pattern(spec, att)
            pat = (d0, tf_np)
            self._zero_pat_cache[key] = pat
        return pat

    def _cav_rays(self, cav: CavLidar, veh, t: float, pat_cache: dict,
                  pose_cache: dict):
        """单 CAV 一帧光线 (逐光束位姿插值 -> 运动伪影).

        返回 (o_np, d_np, tf_np, origin, beam_abs_t, scan_T).
        ``local_pos`` 指定车辆坐标系安装点; ``local_attitude`` 为传感器
        相对车体的姿态, 与旋转扫描/非重复扫描使用完全相同的求交与强度逻辑.

        ``pat_cache`` / ``pose_cache`` 为帧内共享缓存: 同一车辆上扫描周期相同
        的多个传感器共享一份逐光束位姿插值结果 (veh.pose_at), 避免对同一批
        光束时刻反复插值.
        """
        spec = cav.spec
        scan_T = 1.0 / spec.rate_hz
        roll, pitch, yaw_s = [float(v) for v in (cav.local_attitude or (0, 0, 0))]
        d0, tf_np = self._cav_pattern(spec, roll, yaw_s, pitch, t, pat_cache)
        if cav.local_pos is not None:
            lx, ly, lz = [float(v) for v in cav.local_pos]
        else:
            lx, ly, lz = 0.0, 0.0, float(veh.height) + float(cav.height_offset)
        # 同一车辆 + 同一光束时序 -> 位姿插值结果可复用
        pkey = (int(veh.vid), round(scan_T, 9), id(tf_np))
        cached = pose_cache.get(pkey)
        if cached is None:
            beam_t = t + tf_np.astype(np.float64) * scan_T
            pos, yaw = veh.pose_at(beam_t)           # (R,3) 逐光束位姿
            p0, y0 = veh.pose_at(np.array([t]))      # 帧起始位姿 (输出原点)
            cached = (beam_t, pos, yaw, p0, y0)
            pose_cache[pkey] = cached
        beam_t, pos, yaw, p0, y0 = cached
        cy, sy = np.cos(yaw), np.sin(yaw)            # 车载: 车辆 yaw
        ox = pos[:, 0] + lx * cy - ly * sy
        oy = pos[:, 1] + lx * sy + ly * cy
        oz = pos[:, 2] + lz
        o_np = np.stack([ox, oy, oz], 1).astype(np.float32)
        d_np = np.stack([d0[:, 0] * cy - d0[:, 1] * sy,
                         d0[:, 0] * sy + d0[:, 1] * cy,
                         d0[:, 2]], 1).astype(np.float32)
        cy0, sy0 = float(np.cos(y0[0])), float(np.sin(y0[0]))
        origin = np.array([p0[0, 0] + lx * cy0 - ly * sy0,
                           p0[0, 1] + lx * sy0 + ly * cy0,
                           p0[0, 2] + lz], np.float32)
        return o_np, d_np, tf_np, origin, beam_t, scan_T

    def _cav_env_cap(self, cav: CavLidar, o_np, d_np) -> np.ndarray:
        """本帧环境求交的逐束 t_max 上界 (上一帧环境深度 + 运动/指向余量).

        环境静态: 上一帧该束在 t_prev 命中时, 本帧同束 (位姿位移 |Δo|、指向
        变化 Δθ) 的环境最近命中不会远远超过 t_prev, 故取
            cap = t_prev + gain * (|Δo| + t_prev * Δθ) + pad
        作为上界; 上一帧未命中、指向变化过大或光束数不一致时回落到 range_max.
        """
        spec = cav.spec
        R = len(o_np)
        full = np.full(R, spec.range_max, np.float32)
        st = self._cav_state.get(cav.key)
        if not getattr(self.params, "cav_cap_enable", False) or st is None \
                or len(st["t_env"]) != R:
            return full
        t_prev = st["t_env"]
        do = np.linalg.norm(o_np - st["o"], axis=1).astype(np.float32)
        dot = np.clip((d_np * st["d"]).sum(1), -1.0, 1.0)
        dang = np.arccos(dot).astype(np.float32)
        cap = (t_prev + np.float32(self.params.cav_cap_gain) *
               (do + t_prev * dang) + np.float32(self.params.cav_cap_pad))
        ok = np.isfinite(t_prev) & \
            (dang <= np.radians(self.params.cav_cap_max_ang_deg))
        cap = np.where(ok, cap, np.float32(spec.range_max))
        return np.minimum(cap, np.float32(spec.range_max)).astype(np.float32)

    def _simulate_cavs(self, cavs, scene: SceneFrame, t: float) -> list[SimCloud]:
        """多 CAV 合批: 所有车端光线拼接后一次环境求交 + 一次 AABB 粗筛,
        每辆车一次光线-三角求交, 再按雷达切片输出点云.

        与逐 CAV 调用逐束等价 (同一光线集合、同一判据), 合批只是减少
        小 kernel 提交次数与 AABB 候选矩阵的重复构建.
        """
        entries = []
        pat_cache, pose_cache = {}, {}
        for cav in cavs:
            veh = next((v for v in scene.vehicles if v.vid == cav.vid), None)
            if veh is None:
                continue                    # 该帧车辆未激活
            o_np, d_np, tf_np, origin, beam_t, scan_T = self._cav_rays(
                cav, veh, t, pat_cache, pose_cache)
            entries.append((cav, o_np, d_np, tf_np, origin, beam_t, scan_T))
        if not entries:
            self._cav_state.clear()
            return []

        xp = self.xp
        O = np.concatenate([e[1] for e in entries])
        D = np.concatenate([e[2] for e in entries])
        caps = np.concatenate([self._cav_env_cap(e[0], e[1], e[2]) for e in entries])
        rmax = np.concatenate([np.full(len(e[1]), e[0].spec.range_max, np.float32)
                               for e in entries])
        rmins = np.concatenate([np.full(len(e[1]), e[0].spec.range_min, np.float32)
                                for e in entries])
        btans = np.concatenate(
            [np.full(len(e[1]),
                     np.float32(np.tan(np.radians(e[0].spec.beam_divergence_mrad) / 2)),
                     np.float32) for e in entries])
        beam_abs = np.concatenate([e[5] for e in entries])

        if self.gpu:
            Og, Dg = xp.asarray(O), xp.asarray(D)
            t_env, pt_idx = self.grid.raycast(Og, Dg, xp.asarray(rmins),
                                              xp.asarray(caps), xp.asarray(btans))
        else:
            Og, Dg = O, D
            t_env, pt_idx = self.grid.raycast(Og, Dg, rmins, caps, btans)
        # 运动包围盒按最大扫描周期取 (仅粗筛, 取超集不改变求交结果)
        scan_T = max(e[6] for e in entries)
        veh_t, veh_slot, veh_tri, veh_yaw = self._vehicle_raycast(
            Og, Dg, t_env, rmax, rmins, beam_abs, scene, t, scan_T)

        t_env_np = xp.asnumpy(t_env) if self.gpu else t_env
        clouds, state = [], {}
        off = 0
        for cav, o_np, d_np, tf_np, origin, _bt, _st in entries:
            R = len(o_np)
            sl = slice(off, off + R)
            off += R
            # 帧末记录环境深度图, 供下一帧 t_max 上界使用 (按传感器而非车辆区分,
            # 同一车辆多个雷达各自维护自己的深度图)
            state[cav.key] = {"t_env": np.array(t_env_np[sl], np.float32),
                              "o": o_np, "d": d_np}
            name = (f"{cav.spec.name}{cav.tag}@cav{cav.vid}")
            cloud = self._finalize_pattern(
                scene, cav.spec, name, tf_np, origin,
                o_np, d_np, t_env[sl], pt_idx[sl],
                veh_t[sl], veh_slot[sl], veh_tri[sl], veh_yaw[sl])
            # 自车体回波: 车端雷达只可能打到自己车身 (label == 自身 vid), 这些都是
            # 遮挡回波而非目标. 旧实现只剔除 range < self_clear 的近场部分, 大型车
            # (卡车/客车) 或偏置安装的传感器会把远处的自身车厢留在点云里, 污染
            # L-shape 拟合与 IoU; 这里按标签整体剔除.
            # 注意 label==0 表示环境点 (vid 也可以从 0 开始), 必须用 label>0 限定,
            # 否则 vid=0 的 CAV 会把整片环境点误删.
            if len(cloud.xyz):
                keep = ~((cloud.label == cav.vid) & (cloud.label > 0))
                if not keep.all():
                    cloud.xyz = cloud.xyz[keep]
                    cloud.intensity = cloud.intensity[keep]
                    cloud.label = cloud.label[keep]
                    cloud.t_beam = cloud.t_beam[keep]
                    cloud.range_m = cloud.range_m[keep]
                    if cloud.semantics is not None:
                        cloud.semantics = cloud.semantics[keep]
            clouds.append(cloud)
        self._cav_state = state
        return clouds

    # ------------------------------------------------------------------
    def _vehicle_raycast(self, o, d, t_env, rmax, rmin, beam_abs_t,
                         scene: SceneFrame, t: float, scan_T: float):
        """光线-三角形求交: AABB 粗筛 + 逐车辆 Möller–Trumbore, 逐光线取最近命中.

        o/d 为已拼接的全部光线 (xp 数组), t_env 为环境命中距离,
        rmax 为逐光线距离上限 (环境未命中时的兜底), rmin 为逐光线最近距离,
        beam_abs_t 为逐光线绝对时刻.
        """
        xp = self.xp
        R = len(t_env)
        veh_t = xp.full(R, np.inf, xp.float32)
        veh_slot = xp.full(R, -1, xp.int32)
        veh_tri = xp.full(R, -1, xp.int32)
        veh_yaw = xp.zeros(R, xp.float32)
        vehicles = scene.vehicles
        if not vehicles or R == 0:
            return veh_t, veh_slot, veh_tri, veh_yaw
        t_cap = xp.where(xp.isfinite(t_env), t_env, xp.asarray(rmax))
        rmin_xp = xp.asarray(rmin, xp.float32)
        bb_min, bb_max = self._vehicle_aabbs(vehicles, t, scan_T)
        bb_min, bb_max = xp.asarray(bb_min), xp.asarray(bb_max)
        # 逐车 slab 测试: 与 (R,V) 批量矩阵逐元素同判据, 但不生成大临时矩阵
        for vi, veh in enumerate(vehicles):
            m = ray_aabb_one(o, d, bb_min[vi], bb_max[vi], t_cap)
            if self.gpu:
                ray_ids = xp.asnumpy(xp.flatnonzero(m))
            else:
                ray_ids = np.flatnonzero(m)
            if len(ray_ids) == 0:
                continue
            tq = beam_abs_t[ray_ids]
            pos_np, yaw_np = veh.pose_at(tq)          # CPU 插值 (运动伪影)
            tv, tn = self._mesh_arrays(veh.mesh)
            pos_f = pos_np.astype(np.float32)
            yaw_f = yaw_np.astype(np.float32)
            if self.gpu:
                # GPU 分支: 命中判定/散射全部留在显存, 每车仅 1 次索引上传,
                # 不再逐车回传 th/ti/veh_t (原实现每车 3 次 D2H + 4 次 H2D)
                rid = xp.asarray(ray_ids)
                prev = veh_t[rid]
                cap_r = xp.where(xp.isfinite(prev),
                                 xp.minimum(t_cap[rid], prev), t_cap[rid])
                th, ti = raycast_vehicle(o[rid], d[rid], cap_r, tv, tn,
                                         xp.asarray(pos_f), xp.asarray(yaw_f),
                                         rmin_xp[rid])
                yaw_dev = xp.asarray(yaw_f)
                better = xp.isfinite(th) & (th < prev)
                sel = rid[better]
                veh_t[sel] = th[better]
                veh_slot[sel] = vi
                veh_tri[sel] = ti[better]
                veh_yaw[sel] = yaw_dev[better]
            else:
                cap_r = t_cap[ray_ids]
                th, ti = raycast_vehicle(o[ray_ids], d[ray_ids], cap_r, tv, tn,
                                         pos_f, yaw_f, rmin_xp[ray_ids])
                better = np.isfinite(th) & (th < veh_t[ray_ids])
                rid = ray_ids[better]
                veh_t[rid] = th[better]
                veh_slot[rid] = vi
                veh_tri[rid] = ti[better]
                veh_yaw[rid] = yaw_f[better]
        return veh_t, veh_slot, veh_tri, veh_yaw

    # ------------------------------------------------------------------
    def _simulate_pattern(self, scene: SceneFrame, spec: LidarSpec, name: str,
                          t: float, o_np, d_np, tf_np, origin: np.ndarray,
                          od_gpu=None, env_hit=None) -> SimCloud:
        xp = self.xp
        if self.gpu:
            o, d = od_gpu if od_gpu is not None else (xp.asarray(o_np),
                                                      xp.asarray(d_np))
        else:
            o, d = o_np, d_np
        R = len(o_np)
        scan_T = 1.0 / spec.rate_hz
        beam_abs_t = t + tf_np.astype(np.float64) * scan_T   # 各光束绝对时刻

        # ---- 1) 环境求交 (光线-点, 束半径判据) ----
        # 路侧雷达: env_hit 为跨帧缓存结果.
        if env_hit is not None:
            t_env, pt_idx = env_hit
        else:
            t_env, pt_idx = self.grid.raycast(
                o, d, spec.range_min, np.full(R, spec.range_max, np.float32),
                float(np.tan(np.radians(spec.beam_divergence_mrad) / 2)))
        # ---- 2) 车辆求交 (光线-三角形, 逐光线位姿 -> 运动伪影) ----
        rmax = np.full(R, spec.range_max, np.float32)
        rmin = np.full(R, spec.range_min, np.float32)
        veh = self._vehicle_raycast(o, d, t_env, rmax, rmin, beam_abs_t,
                                    scene, t, scan_T)
        # ---- 3) 融合/强度/输出 ----
        return self._finalize_pattern(scene, spec, name, tf_np, origin,
                                      o_np, d_np, t_env, pt_idx, *veh)

    def _finalize_pattern(self, scene: SceneFrame, spec: LidarSpec, name: str,
                          tf_np, origin: np.ndarray, o_np, d_np,
                          t_env, pt_idx, veh_t, veh_slot, veh_tri,
                          veh_yaw) -> SimCloud:
        """最近命中融合 -> 入射角/反射率 -> 物理强度 -> 输出点云."""
        xp = self.xp
        env_closer = t_env <= veh_t
        t_hit = xp.where(env_closer, t_env, veh_t)
        idx = xp.nonzero(xp.isfinite(t_hit))[0]
        wcfg = getattr(self.params, "weather", None)
        w_on = wcfg is not None and wcfg.active()
        sigma = wcfg.sigma_ext() if w_on else 0.0
        vehicles = scene.vehicles
        if len(idx) == 0:
            cloud = SimCloud(name, origin,
                             np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                             np.zeros(0, np.int64), np.zeros(0, np.float32),
                             np.zeros(0, np.float32),
                             np.zeros(0, np.int32) if self.semantics else None,
                             self.sem_fg_base if self.semantics else 0)
            if w_on:   # 即使无真实命中, 降水杂波仍可能存在
                r_full = (xp.asnumpy(t_hit) if self.gpu else t_hit).astype(np.float64)
                cl = generate_clutter(wcfg, spec, o_np, d_np, tf_np, r_full, self.rng)
                if cl is not None:
                    cloud = self._append_clutter(cloud, cl)
            return cloud
        g = (lambda a: xp.asnumpy(a) if self.gpu else a)
        if self.gpu:
            # 一次打包回传 (idx / 距离 / 环境标记 / 点号 / 车槽 / 三角 / 偏航)
            # 取代 7 次独立的 D2H 同步
            blob = xp.asnumpy(xp.stack([
                idx.astype(xp.int32),
                t_hit[idx].view(xp.int32),
                env_closer[idx].astype(xp.int32),
                pt_idx[idx], veh_slot[idx], veh_tri[idx],
                veh_yaw[idx].view(xp.int32)]))
            idx_np = blob[0].astype(np.int64)
            th_np = blob[1].view(np.float32).astype(np.float64)
            ec_np = blob[2].astype(bool)
            pt_np, slot_np, tri_np = blob[3], blob[4], blob[5]
            yaw_np = blob[6].view(np.float32)
        else:
            idx_np = g(idx)
            th_np = g(t_hit[idx]).astype(np.float64)
            ec_np = g(env_closer[idx])
            pt_np = g(pt_idx[idx])
            slot_np = g(veh_slot[idx])
            tri_np = g(veh_tri[idx])
            yaw_np = g(veh_yaw[idx])

        # ---- 4) 入射角/反射率 ----
        n = len(idx_np)
        cos_inc = np.zeros(n, np.float64)
        refl = np.zeros(n, np.float64)
        em = ec_np.copy()
        em &= pt_np >= 0
        if em.any():
            pn = pt_np[em]
            nrm = self.env_nrm_np[pn]
            cos_inc[em] = np.abs((nrm * d_np[idx_np[em]]).sum(1))
            refl[em] = self.env_refl_np[pn]
        vm = ~ec_np
        vm &= slot_np >= 0
        for s in np.unique(slot_np[vm]):
            m = vm & (slot_np == s)
            veh = vehicles[int(s)]
            tn = self._mesh_normals(veh.mesh)[tri_np[m]]      # 面法向 (车辆系)
            c, sn = np.cos(yaw_np[m]), np.sin(yaw_np[m])
            nw = np.stack([c * tn[:, 0] - sn * tn[:, 1],
                           sn * tn[:, 0] + c * tn[:, 1], tn[:, 2]], 1)
            cos_inc[m] = np.abs((nw * d_np[idx_np[m]]).sum(1))
            refl[m] = veh.refl

        # ---- 5) 物理强度 + 噪声/漏检 (雨/雪时附加消光) ----
        inten, keep = compute_intensity(th_np, cos_inc, refl, spec, self.rng,
                                        ext_extra=sigma)
        xyz = o_np[idx_np] + th_np[:, None].astype(np.float32) * d_np[idx_np]
        label = np.where(ec_np, 0,
                         np.array([vehicles[int(s)].vid if s >= 0 else 0
                                   for s in slot_np], np.int64))
        # ---- 5b) 语义类别: 背景继承环境点云语义, 前景按车型对象属性编码 ----
        sem = None
        if self.semantics:
            sem = np.zeros(n, np.int32)          # 背景无语义时默认 0 (unlabeled)
            if em.any() and self.env_sem_np is not None:
                sem[em] = self.env_sem_np[pt_np[em]]
            for s in np.unique(slot_np[vm]):
                veh = vehicles[int(s)]
                sem[vm & (slot_np == s)] = object_semantic_class(
                    getattr(veh, "cls", ""), self.sem_fg_base)
        cloud = SimCloud(
            lidar_name=name,
            origin=origin,
            xyz=xyz[keep].astype(np.float32),
            intensity=inten[keep],
            label=label[keep],
            t_beam=tf_np[idx_np][keep].astype(np.float32),
            range_m=th_np[keep].astype(np.float32),
            semantics=(sem[keep] if sem is not None else None),
            semantic_base=(self.sem_fg_base if self.semantics else 0))
        # ---- 6) 雨/雪杂波 (label=-1) ----
        if w_on:
            r_full = (xp.asnumpy(t_hit) if self.gpu else t_hit).astype(np.float64)
            cl = generate_clutter(wcfg, spec, o_np, d_np, tf_np, r_full, self.rng)
            if cl is not None:
                cloud = self._append_clutter(cloud, cl)
        return cloud

    @staticmethod
    def _append_clutter(cloud: SimCloud, cl) -> SimCloud:
        cx, ci, ct, cr = cl
        cloud.xyz = np.concatenate([cloud.xyz, cx])
        cloud.intensity = np.concatenate([cloud.intensity, ci])
        cloud.label = np.concatenate(
            [cloud.label, np.full(len(cx), -1, np.int64)])
        cloud.t_beam = np.concatenate([cloud.t_beam, ct])
        cloud.range_m = np.concatenate([cloud.range_m, cr])
        if cloud.semantics is not None:
            # 天气杂波不属于背景环境也不是前景目标 -> 未标注
            cloud.semantics = np.concatenate(
                [cloud.semantics, np.zeros(len(cx), np.int32)])
        return cloud

    # ------------------------------------------------------------------
    def _mesh_normals(self, mesh):
        """网格三角面法向 (CPU), 按网格名缓存, 避免每帧重算. """
        a = self._mesh_nrm_cpu.get(mesh.name)
        if a is None:
            a = mesh.tri_normals()
            self._mesh_nrm_cpu[mesh.name] = a
        return a

    def _mesh_arrays(self, mesh):
        key = mesh.name
        if key not in self._mesh_cache:
            tv = mesh.tri_verts.astype(np.float32)
            tn = self._mesh_normals(mesh)
            self._mesh_cache[key] = ((self.xp.asarray(tv), self.xp.asarray(tn))
                                     if self.gpu else (tv, tn))
        return self._mesh_cache[key]

    def _vehicle_aabbs(self, vehicles, t, scan_T):
        """扫描周期内车辆的运动包围盒 (AABB 粗筛)."""
        mins, maxs = [], []
        for veh in vehicles:
            p0, _ = veh.pose_at(np.array([t]))
            p1, _ = veh.pose_at(np.array([t + scan_T]))
            half = max(veh.length, veh.width) * 0.5 + 0.2
            c = np.concatenate([p0, p1])
            mn = c.min(0) - half
            mx = c.max(0) + half
            mn[2] = min(p0[0, 2], veh.pos[2]) - 0.1
            mx[2] = max(p1[0, 2], veh.pos[2]) + veh.height + 0.1
            mins.append(mn)
            maxs.append(mx)
        return np.stack(mins).astype(np.float32), np.stack(maxs).astype(np.float32)

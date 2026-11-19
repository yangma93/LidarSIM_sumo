# -*- coding: utf-8 -*-
"""SUMO 联合仿真: TraCI 连接 / 动态交通接入 / 车路协同感知反馈.

数据流
------
SUMO 每步给出全网车辆状态 -> ``OnlineTrajectories`` -> ``SceneBuilder``
-> ``LidarSimulator`` (路侧 LiDAR + CAV 车端 LiDAR) -> 感知结果
-> TraCI 写回 SUMO (检测目标着色 / 自定义参数 / 可选闭环减速控制).

本模块不依赖 Qt, 便于在后台线程或命令行中复用.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
import zlib
from dataclasses import dataclass, field, replace

import numpy as np

from .config import SimParams, make_lidar
from .cav import CavLidar
from .cav_config import CavComplexConfig, hash01
from .meshes import class_of_mesh
from .lidar_sim import LidarSimulator, LidarPose
from .lshape import evaluate_frame
from .scene import SceneBuilder
from .roadnet import find_sumo_binary, describe_sumo_env
from .traffic_live import OnlineTrajectories, VehicleSnapshot

MAX_CAV = 10


# ---------------------------------------------------------------------------
# 环境检查
# ---------------------------------------------------------------------------
def traci_available() -> bool:
    try:
        import traci  # noqa: F401
        return True
    except Exception:                                             # noqa: BLE001
        return False


def traci_message() -> str:
    if traci_available():
        return "TraCI 可用"
    return ("未安装 traci (pip install eclipse-sumo); "
            + describe_sumo_env())


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class CosimConfig:
    sumo_binary: str = ""
    config_path: str = ""            # .sumocfg (优先)
    net_path: str = ""               # 仅给定 net.xml 时使用
    use_gui: bool = True
    step_length: float = 0.1
    port: int = 8813
    sim_every: int = 1               # 每 N 个 SUMO 步做一次 LiDAR 感知
    cav_mode: str = "penetration"    # none | ids | penetration
    cav_ids: list = field(default_factory=list)
    penetration: float = 0.1
    cav_preset: str = "vlp16"
    cav_height: float = 0.2
    lidars: list = field(default_factory=list)   # [(preset, [x,y,z,roll,yaw,pitch])]
    feedback: bool = True
    control: bool = False
    ttc_thr: float = 2.5
    viz: bool = True
    min_pts: int = 5
    # 协同感知 (L-shape) 反馈: IoU >= iou_thr 视为已感知 (SUMO 绿色),
    # 否则视为未感知 (SUMO 红色)
    iou_thr: float = 0.3
    show_gt: bool = False        # 三维可视化绘制 GT BBox
    show_det: bool = True        # 三维可视化绘制感知 (拟合) BBox
    cav_complex: dict = field(default_factory=dict)   # CAV 复杂配置 (CavComplexConfig.to_dict)
    algo_params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TraCI 桥
# ---------------------------------------------------------------------------
class SumoBridge:
    """对 traci 的薄封装 (支持 sumo / sumo-gui, 可复用于连接测试)."""

    def __init__(self, cfg: CosimConfig):
        self.cfg = cfg
        self.connected = False
        self.ended = False
        self.binary = ""
        self.log_path = os.path.join(
            tempfile.gettempdir(), f"lidarsim_sumo_{int(cfg.port)}.log")
        self._traci = None
        self._proc = None

    def _launch_args(self) -> list[str]:
        binary = self.cfg.sumo_binary.strip()
        if binary:
            # 勾选/未勾选 SUMO 界面时, 若手动指定的是另一种可执行文件,
            # 自动切换到同目录下匹配的 sumo / sumo-gui
            base = os.path.basename(binary).lower()
            gui_names = ("sumo-gui", "sumo-gui.exe")
            plain_names = ("sumo", "sumo.exe")
            if (self.cfg.use_gui and base in plain_names) or \
                    (not self.cfg.use_gui and base in gui_names):
                want = "sumo-gui" if self.cfg.use_gui else "sumo"
                name = want + (".exe" if base.endswith(".exe") else "")
                cand = os.path.join(os.path.dirname(binary), name)
                if os.path.exists(cand):
                    binary = cand
        if not binary:
            binary = find_sumo_binary(self.cfg.use_gui)
        if not binary:
            raise FileNotFoundError(
                "未找到 SUMO 可执行文件。请安装 Eclipse SUMO "
                "(https://eclipse.dev/sumo/) 或设置 SUMO_HOME / 手动指定路径。\n"
                + describe_sumo_env())
        self.binary = binary
        args = [binary]
        if self.cfg.config_path:
            args += ["-c", self.cfg.config_path]
        elif self.cfg.net_path:
            args += ["-n", self.cfg.net_path]
        else:
            raise ValueError("请指定 SUMO 配置文件 (.sumocfg) 或路网文件 (.net.xml)")
        args += ["--step-length", f"{float(self.cfg.step_length):g}",
                 "--quit-on-end", "--no-warnings", "--no-step-log", "true",
                 "--ignore-route-errors", "true",
                 "--duration-log.statistics", "false",
                 "--collision.action", "none",
                 "--time-to-teleport", "-1",
                 "--seed", "42",
                 "--log", self.log_path]
        if self.cfg.use_gui:
            # sumo-gui 缺省会停在"等待手动点击开始"的状态, 不响应 TraCI 命令,
            # 导致 traci.start 在 getVersion 上永久阻塞. --start 让它立即进入
            # 由 TraCI 驱动的仿真循环.
            args += ["--start", "--delay", "0"]
        return args

    def log_tail(self, n: int = 4000) -> str:
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()[-n:]
        except Exception:                                         # noqa: BLE001
            pass
        return ""

    def _sumo_version(self) -> str:
        if not self.binary:
            return ""
        try:
            cp = subprocess.run([self.binary, "--version"], capture_output=True,
                                text=True, timeout=10)
            txt = (cp.stdout or "") + (cp.stderr or "")
            for line in txt.splitlines():
                if line.strip():
                    return line.strip()
        except Exception:                                         # noqa: BLE001
            pass
        return ""

    @staticmethod
    def _force_close(wait_timeout: float = 3.0):
        """关闭残留 TraCI 连接; 对 SUMO 进程做有限等待后强制结束.

        traci.close(True) 内部是无超时的 process.wait(), 一旦 sumo-gui 停在
        启动界面不退出, 就会永久阻塞调用线程 (GUI 线程即表现为整个程序卡死).
        """
        import traci
        try:
            conn = traci.getConnection("default")
        except Exception:                                         # noqa: BLE001
            conn = None
        proc = getattr(conn, "_process", None) if conn is not None else None
        try:
            traci.close(False)
        except Exception:                                         # noqa: BLE001
            pass
        if proc is not None:
            try:
                proc.wait(timeout=wait_timeout)
            except Exception:                                     # noqa: BLE001
                try:
                    proc.kill()
                except Exception:                                 # noqa: BLE001
                    pass

    def connect(self, num_retries: int = 60) -> dict:
        import traci
        self._traci = traci
        # 清理可能残留的连接, 避免端口被上一次测试/仿真占用
        self._force_close()
        # 自行启动进程并持有其句柄, 这样即使握手失败也能在 close() 中强制结束,
        # 不会留下僵死的 sumo-gui. remote-port 由这里显式附加.
        args = self._launch_args() + ["--remote-port", str(int(self.cfg.port))]
        try:
            self._proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
        except Exception as e:                                    # noqa: BLE001
            raise RuntimeError(f"启动 SUMO 失败: {e}")
        try:
            raw = traci.init(port=int(self.cfg.port), numRetries=int(num_retries),
                             proc=self._proc)
        except Exception as e:                                    # noqa: BLE001
            tail = self.log_tail()
            self.close()
            raise RuntimeError(f"{e}" + (f"\nSUMO 日志:\n{tail}" if tail else ""))
        self.connected = True
        self.ended = False
        # TraCI 返回值: 新版为 (api_version, "SUMO x.y.z"), 旧版可能是 dict
        if isinstance(raw, dict):
            ver = raw
        elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
            ver = {"api": raw[0], "version": str(raw[1])}
        elif raw:
            ver = {"version": str(raw)}
        else:
            ver = {}
        if not ver.get("version"):
            ver["version"] = self._sumo_version()
        return ver

    def step(self):
        if not self.connected:
            raise RuntimeError("SUMO 尚未连接")
        traci = self._traci
        try:
            traci.simulationStep()
        except Exception as e:                                    # noqa: BLE001
            if "closed" in str(e).lower():
                self.ended = True
            raise
        t = float(traci.simulation.getTime())
        snaps = []
        for sid in traci.vehicle.getIDList():
            try:
                x, y, z = traci.vehicle.getPosition3D(sid)
                ang = float(traci.vehicle.getAngle(sid))
                spd = float(traci.vehicle.getSpeed(sid))
                L = float(traci.vehicle.getLength(sid))
                W = float(traci.vehicle.getWidth(sid))
                typ = traci.vehicle.getTypeID(sid)
            except Exception:                                     # noqa: BLE001
                continue
            snaps.append(dict(sid=sid, x=x, y=y, z=z, angle=ang,
                              speed=spd, length=L, width=W, vtype=typ))
        try:
            colliding = list(traci.simulation.getCollidingVehiclesIDList())
        except Exception:                                         # noqa: BLE001
            colliding = []
        return t, snaps, colliding

    # ---- 反馈 ----
    def set_color(self, sid: str, rgba):
        try:
            self._traci.vehicle.setColor(sid, tuple(int(v) for v in rgba))
        except Exception:                                         # noqa: BLE001
            pass

    def set_param(self, sid: str, key: str, value: str):
        try:
            self._traci.vehicle.setParameter(sid, key, value)
        except Exception:                                         # noqa: BLE001
            pass

    def slow_down(self, sid: str, speed: float, duration: float = 0.5):
        try:
            self._traci.vehicle.slowDown(sid, max(0.0, float(speed)), float(duration))
        except Exception:                                         # noqa: BLE001
            pass

    def close(self):
        traci = self._traci
        proc = self._proc
        self.connected = False
        self._traci = None
        self._proc = None
        if traci is not None:
            try:
                traci.close(False)      # 不等待进程, 避免 sumo-gui 阻塞调用线程
            except Exception:                                     # noqa: BLE001
                pass
        if proc is not None:
            try:
                proc.wait(timeout=4)
            except Exception:                                     # noqa: BLE001
                try:
                    proc.kill()
                except Exception:                                 # noqa: BLE001
                    pass
        time.sleep(0.1)


def test_connection(cfg: CosimConfig) -> tuple[bool, str]:
    """MCP 式连接测试: 启动 -> 读取版本 -> 关闭, 返回 (成功, 描述).

    使用独立端口, 避免测试连接占用正式联仿端口导致后续 FatalTraCIError.
    测试固定使用无界面 sumo: 不需要额外窗口, 也不会被 sumo-gui 的启动流程阻塞.
    """
    if not traci_available():
        return False, traci_message()
    probe = replace(cfg, port=int(cfg.port) + 37, use_gui=False)
    bridge = SumoBridge(probe)
    try:
        ver = bridge.connect(num_retries=10)
        ver_s = ""
        if isinstance(ver, dict):
            ver_s = ver.get("version", "") or ver.get("sumo", "")
        return True, f"连接成功: {ver_s or 'SUMO 版本未知'} (测试端口 {probe.port})"
    except Exception as e:                                        # noqa: BLE001
        tail = bridge.log_tail()
        return False, f"连接失败: {e}" + (f"\nSUMO 日志:\n{tail}" if tail else "")
    finally:
        bridge.close()


# ---------------------------------------------------------------------------
# 感知结果
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    vid: int
    sid: str
    dist: float
    ttc: float
    bearing_deg: float
    n_points: int


@dataclass
class PerceptionReport:
    ego_vid: int
    ego_sid: str
    detections: list = field(default_factory=list)

    def summary(self) -> str:
        items = [f"{d.sid}({d.dist:.1f}m,TTC={d.ttc:.1f}s)" for d in self.detections[:5]]
        return f"ego {self.ego_sid}: {len(self.detections)} targets " + \
               ("; ".join(items) if items else "-")


def _parse_cav_from_cloud_name(name: str):
    """``vlp16@cav12`` -> 12; 非车端雷达返回 None."""
    if "@cav" not in name:
        return None
    try:
        return int(name.rsplit("@cav", 1)[1])
    except ValueError:
        return None


class CosimSession:
    """一次联仿会话: 初始化仿真器 -> 循环 step -> 感知 -> 反馈."""

    def __init__(self, cfg: CosimConfig, env, models,
                 logger=None):
        self.cfg = cfg
        self.env = env
        self.models = models
        self.log = logger or (lambda msg: None)
        self.bridge = SumoBridge(cfg)
        self.traffic = OnlineTrajectories()
        self.builder = None
        self.sim = None
        self.vid_map: dict[str, int] = {}
        self.vtype_map: dict[str, int] = {}
        self.cav_cfg = CavComplexConfig.from_dict(self.cfg.cav_complex)
        self._last_status = None
        # 上次写入 SUMO 的车辆颜色, 避免每步重复 setColor
        self._colors: dict[str, tuple] = {}

    # ---- 初始化 ----
    def open(self):
        if self.env is None:
            raise ValueError("联仿需要环境点云 (真实道路背景)")
        if self.models is None:
            raise ValueError("联仿需要车辆网格模型 (车辆点云仿真)")
        ver = self.bridge.connect()
        vstr = ver.get("version", "") if isinstance(ver, dict) else ""
        self.log(f"SUMO 已连接 (版本 {vstr or '未知'}, 端口 {self.cfg.port})")
        self._check_coordinate_space()
        params = SimParams(device="auto", seed=0)
        for k, v in (self.cfg.algo_params or {}).items():
            if hasattr(params, k):
                setattr(params, k, v)
        ground_cell = getattr(params, "ground_cell", 1.5)
        self.builder = SceneBuilder(
            self.traffic, self.env, self.models, verbose=False,
            ground_cell=ground_cell,
            ground_pctl=getattr(params, "ground_pctl", 5.0),
            ground_tol=getattr(params, "ground_tol", 0.3),
            env_voxel=getattr(params, "env_voxel", 0.1),
            normal_voxel=getattr(params, "normal_voxel", 0.25))
        lidars = [LidarPose(spec=make_lidar(p), pose=list(pose))
                  for p, pose in (self.cfg.lidars or [])]
        self.sim = LidarSimulator(self.builder, lidars, params, cav_lidars=[])
        self.log(f"感知链路就绪: 路侧雷达 {len(lidars)} 个, "
                 f"CAV 模式 {self.cfg.cav_mode}, 可视化 {'开' if self.cfg.viz else '关'}")

    # ---- 坐标一致性检查 ----
    def _check_coordinate_space(self):
        """SUMO 路网边界与点云包围盒应位于同一坐标空间."""
        try:
            b = self.bridge._traci.simulation.getNetBoundary()
            (x1, y1), (x2, y2) = b
            p = np.asarray(self.env.points, np.float64)
            lo, hi = p.min(0), p.max(0)
            overlap = not (x2 < lo[0] - 5.0 or x1 > hi[0] + 5.0 or
                           y2 < lo[1] - 5.0 or y1 > hi[1] + 5.0)
            msg = (f"SUMO net bbox x[{min(x1, x2):.1f},{max(x1, x2):.1f}] "
                   f"y[{min(y1, y2):.1f},{max(y1, y2):.1f}] | "
                   f"cloud x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}]")
            if overlap:
                self.log("坐标空间匹配: " + msg)
            else:
                self.log("警告: SUMO 路网与点云坐标范围不重叠, 感知结果将为空; "
                         "请在底图绘制时保持原始点云坐标并关闭坐标归一化. " + msg)
        except Exception as e:                                    # noqa: BLE001
            self.log(f"坐标一致性检查跳过: {e}")

    # ---- CAV 选择 ----
    def _snap_class(self, s) -> str:
        """按车辆外廓匹配网格, 推断车型类别 (用于复杂配置)."""
        try:
            mesh = self.models.match(float(s["length"]), float(s["width"]))
            return class_of_mesh(mesh.name)
        except Exception:                                         # noqa: BLE001
            return "car"

    def _select_cav_sids(self, snaps) -> list[str]:
        mode = self.cfg.cav_mode
        if mode == "none":
            return []
        sids = [s["sid"] for s in snaps]
        if mode == "ids":
            wanted = {str(x).strip() for x in (self.cfg.cav_ids or []) if str(x).strip()}
            return [s for s in sids if s in wanted][:MAX_CAV]
        # 复杂配置: 分车型渗透率 (与主界面/离线一致)
        if self.cav_cfg.enabled:
            out = []
            for s in snaps:
                cfg = self.cav_cfg.type_for(self._snap_class(s))
                if cfg is None:
                    continue
                key = self.vid_map.get(s["sid"], s["sid"])
                if hash01(key) < float(cfg.penetration):
                    out.append(s["sid"])
            return out[:MAX_CAV]
        p = float(np.clip(self.cfg.penetration, 0.0, 1.0))
        if p <= 0:
            return []
        out = [s for s in sids
               if (zlib.crc32(s.encode("utf-8")) % 10000) / 10000.0 < p]
        return out[:MAX_CAV]

    def _snapshots(self, t, raw) -> list[VehicleSnapshot]:
        snaps = []
        for s in raw:
            sid = s["sid"]
            if sid not in self.vid_map:
                vid = len(self.vid_map)
                self.vid_map[sid] = vid
            if s["vtype"] not in self.vtype_map:
                self.vtype_map[s["vtype"]] = len(self.vtype_map)
            snaps.append(VehicleSnapshot(
                sid=sid, vid=self.vid_map[sid], vtype=self.vtype_map[s["vtype"]],
                x=s["x"], y=s["y"], z=s["z"], angle_deg=s["angle"],
                speed=s["speed"], length=s["length"], width=s["width"]))
        return snaps

    # ---- 感知 ----
    def _perceive(self, scene, clouds, cav_vids) -> list[PerceptionReport]:
        veh_by_vid = {v.vid: v for v in scene.vehicles}
        merged: dict[int, PerceptionReport] = {}
        for cloud in clouds:
            ego = _parse_cav_from_cloud_name(cloud.lidar_name)
            if ego is None:
                continue
            labels = cloud.label[cloud.label > 0]
            u, cnt = (np.unique(labels, return_counts=True) if len(labels)
                      else (np.zeros(0, np.int64), np.zeros(0, np.int64)))
            ego_veh = veh_by_vid.get(ego)
            ego_sid = self.traffic.vid_to_sid.get(ego, str(ego))
            ego_speed = self.traffic.speed_of(ego)
            rep = merged.get(ego)
            if rep is None:
                rep = PerceptionReport(ego_vid=ego, ego_sid=ego_sid)
                merged[ego] = rep
            by_vid = {d.vid: d for d in rep.detections}
            for vid, n in zip(u, cnt):
                if int(n) < int(self.cfg.min_pts):
                    continue
                tv = veh_by_vid.get(int(vid))
                if tv is None:
                    continue
                rel = tv.pos[:2] - (ego_veh.pos[:2] if ego_veh is not None else cloud.origin[:2])
                dist = float(np.linalg.norm(rel))
                bearing = float(math.degrees(math.atan2(rel[1], rel[0])))
                ttc = float("inf")
                if ego_veh is not None:
                    fy = ego_veh.yaw
                    fwd = rel[0] * math.cos(fy) + rel[1] * math.sin(fy)
                    lat = abs(-rel[0] * math.sin(fy) + rel[1] * math.cos(fy))
                    if fwd > 0.0 and lat < 4.0:
                        closing = ego_speed - self.traffic.speed_of(int(vid))
                        if closing > 0.2:
                            ttc = fwd / closing
                old = by_vid.get(int(vid))
                if old is not None:
                    # 同一 CAV 多传感器融合: 保留最近距离 / 最大点数
                    old.n_points = max(old.n_points, int(n))
                    if dist < old.dist:
                        old.dist = dist
                    continue
                d = Detection(
                    vid=int(vid),
                    sid=self.traffic.vid_to_sid.get(int(vid), str(vid)),
                    dist=dist, ttc=ttc,
                    bearing_deg=bearing, n_points=int(n))
                by_vid[int(vid)] = d
                rep.detections.append(d)
            rep.detections.sort(key=lambda d: d.dist)
        return list(merged.values())

    def _feedback(self, reports, ev):
        """把协同感知结果写回 SUMO.

        协同感知口径: ``evaluate_frame`` 会把同一目标上来自全部路侧雷达与
        CAV 车端雷达的点云融合后做 L-shape 拟合, 再与真值框计算 3D IoU.
        IoU >= ``iou_thr`` 的目标在 SUMO 中标记为绿色 (已感知), 其余标红
        (未感知). 颜色仅在变化时写入, 避免每步重复 TraCI 调用.
        """
        if not self.cfg.feedback:
            return
        thr = float(self.cfg.iou_thr)
        iou_by_vid = {ins.vid: float(ins.iou3d) for ins in ev.instances}
        iou_by_sid = {self.traffic.vid_to_sid.get(v, ""): i for v, i in iou_by_vid.items()}
        perceived = set()
        for ins in ev.instances:
            if ins.matched and ins.iou3d >= thr:
                sid = self.traffic.vid_to_sid.get(ins.vid)
                if sid:
                    perceived.add(sid)
        for _vid, sid in list(self.traffic.vid_to_sid.items()):
            col = (self.PERCEIVED_RGBA if sid in perceived
                   else self.NOT_PERCEIVED_RGBA)
            if self._colors.get(sid) != col:
                self.bridge.set_color(sid, col)
                # 协同感知结果写回 SUMO, 供上层控制算法按需读取
                self.bridge.set_param(sid, "lidarsim.iou",
                                      f"{iou_by_sid.get(sid, 0.0):.3f}")
                self.bridge.set_param(sid, "lidarsim.perceived",
                                      "1" if sid in perceived else "0")
                self._colors[sid] = col
        if not reports:
            return
        for rep in reports:
            payload = {
                "t": round(self.traffic.current_time, 3),
                "ego": rep.ego_sid,
                "iou_thr": round(thr, 3),
                "targets": [{"id": d.sid, "dist": round(d.dist, 2),
                             "ttc": (None if not math.isfinite(d.ttc) else round(d.ttc, 2)),
                             "bearing": round(d.bearing_deg, 1),
                             "iou": round(iou_by_vid.get(d.vid, 0.0), 3)}
                            for d in rep.detections[:16]],
            }
            self.bridge.set_param(rep.ego_sid, "lidarsim.perception",
                                  json.dumps(payload, ensure_ascii=False))
            if self.cfg.control and rep.detections:
                threat = min((d for d in rep.detections if math.isfinite(d.ttc)),
                             key=lambda d: d.ttc, default=None)
                if threat is not None and threat.ttc <= self.cfg.ttc_thr:
                    ego_speed = self.traffic.speed_of(rep.ego_vid)
                    self.bridge.slow_down(rep.ego_sid, ego_speed * 0.5, 0.5)

    PERCEIVED_RGBA = (30, 200, 60, 255)       # IoU >= 阈值: 绿色
    NOT_PERCEIVED_RGBA = (225, 45, 45, 255)   # IoU < 阈值: 红色

    def _emit_status(self, on_status, t, raw, reports, ev):
        """节流上报状态.

        联仿循环可能远快于 GUI 重绘; 若每一步都跨线程 emit 状态, Qt 事件队列
        会被 status/frame 信号淹没而表现为界面卡死. 这里限制最小上报间隔.
        """
        if on_status is None:
            return
        now = time.time()
        if now - float(self._last_status or 0.0) < 0.15:
            return
        self._last_status = now
        thr = float(self.cfg.iou_thr)
        n_gt = len(ev.instances)
        n_seen = sum(1 for ins in ev.instances if ins.matched and ins.iou3d >= thr)
        if reports:
            msg = " | ".join(r.summary() for r in reports[:2])
            on_status(2, f"t={t:.1f}s 协同感知 {n_seen}/{n_gt} | {msg}")
        else:
            on_status(2, f"t={t:.1f}s 数据流正常 ({len(raw)} 车, "
                         f"协同感知 {n_seen}/{n_gt})")

    # ---- 主循环 ----
    def run(self, stop_event, on_frame=None, on_status=None, on_log=None):
        """阻塞运行, 直到 stop_event 置位或 SUMO 结束."""
        report_log = on_log or self.log
        idx = 0
        last_collide = []
        try:
            if self.builder is None:
                self.open()
            while not stop_event.is_set():
                try:
                    t, raw, colliding = self.bridge.step()
                except Exception as e:                            # noqa: BLE001
                    tail = self.bridge.log_tail()
                    if self.bridge.ended or "closed by sumo" in str(e).lower():
                        report_log("SUMO 进程已结束: " + str(e) +
                                   (f"\nSUMO 日志(末尾):\n{tail}" if tail else ""))
                        if on_status:
                            on_status(1, "SUMO 仿真结束 (进程退出)")
                        break
                    raise
                try:
                    remaining = int(self.bridge._traci.simulation.getMinExpectedNumber())
                except Exception:                                 # noqa: BLE001
                    remaining = 1
                if not raw and remaining <= 0:
                    if on_status:
                        on_status(1, "SUMO 仿真结束")
                    break
                snaps = self._snapshots(t, raw)
                self.traffic.update(t, snaps)
                cav_sids = self._select_cav_sids(raw)
                cavs = []
                raw_by_sid = {s["sid"]: s for s in raw}
                for sid in cav_sids:
                    vid = self.vid_map.get(sid)
                    if vid is None:
                        continue
                    if self.cav_cfg.enabled:
                        cls = self._snap_class(raw_by_sid[sid])
                        cavs.extend(self.cav_cfg.cav_lidars_for(
                            vid, cls, force=(self.cfg.cav_mode == "ids")))
                    else:
                        cavs.append(CavLidar(spec=make_lidar(self.cfg.cav_preset),
                                             vid=vid,
                                             height_offset=self.cfg.cav_height))
                self.sim.set_cav_lidars(cavs)
                if colliding and colliding != last_collide:
                    report_log(f"SUMO 碰撞车辆: {colliding}")
                    last_collide = list(colliding)
                if idx % max(1, int(self.cfg.sim_every)) == 0:
                    if self.cfg.viz or self.cfg.feedback or self.sim.lidars or cavs:
                        scene, clouds = self.sim.simulate_frame(t)
                        reports = self._perceive(scene, clouds, cavs)
                        # 协同感知: 多雷达/CAV 点云按目标融合 -> L-shape 拟合 -> IoU
                        ev = evaluate_frame(clouds, scene,
                                            min_points=max(1, int(self.cfg.min_pts)))
                        self._feedback(reports, ev)
                        if on_frame is not None:
                            on_frame((scene, clouds, t, reports, cav_sids, ev))
                        self._emit_status(on_status, t, raw, reports, ev)
                idx += 1
        except Exception as e:                                    # noqa: BLE001
            if on_status:
                on_status(0, f"联仿中断: {e}")
            report_log(f"联仿异常: {type(e).__name__}: {e}")
            raise
        finally:
            self.close()

    def close(self):
        was_open = self.bridge.connected or self.bridge._traci is not None
        self.bridge.close()
        if was_open:
            self.log("SUMO 连接已关闭")

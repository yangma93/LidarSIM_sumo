# -*- coding: utf-8 -*-
"""SUMO 微观路网: 数据模型 / 几何辅助 / 明文 XML 与 net.xml 读写.

设计目标
--------
* 在真实环境点云上描摹道路中心线, 生成可直接交给 SUMO netconvert 的
  ``nodes.nod.xml`` / ``edges.edg.xml`` / ``connections.con.xml``;
* 节点可携带高程 z (由点云地面模型赋值), 保证 SUMO 坐标空间与点云一致;
* 提供路面多边形 / 车道线 / 方向箭头 / 交叉口轮廓等纯几何结果, 供 GUI 绘制;
* 读取已有 ``*.net.xml`` 回填编辑器, 便于二次修改.

坐标约定与 ``lidarsim`` 其它模块完全一致: x/y/z 为世界系 (米), z 向上.
SUMO 的 net.xml 同样是右手系 (x 东, y 北, z 上), 因此无需坐标变换.
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

_DEFAULT_SPEED = 13.89        # 50 km/h
_DEFAULT_LANE_WIDTH = 3.2


# ---------------------------------------------------------------------------
# 几何辅助
# ---------------------------------------------------------------------------
def polyline_length(points: np.ndarray) -> float:
    p = np.asarray(points, np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def point_at_fraction(points: np.ndarray, frac: float):
    """折线上按弧长比例取点与切向 (单位向量)."""
    p = np.asarray(points, np.float64)
    if len(p) < 2:
        return (p[0] if len(p) else np.zeros(3)), np.array([1.0, 0.0, 0.0])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-9:
        return p[0], np.array([1.0, 0.0, 0.0])
    target = float(np.clip(frac, 0.0, 1.0)) * total
    acc = 0.0
    for i, s in enumerate(seg):
        if acc + s >= target or i == len(seg) - 1:
            k = (target - acc) / max(s, 1e-9)
            k = float(np.clip(k, 0.0, 1.0))
            pos = p[i] + (p[i + 1] - p[i]) * k
            d = p[i + 1] - p[i]
            return pos, (d / max(np.linalg.norm(d), 1e-9))
        acc += s
    d = p[-1] - p[-2]
    return p[-1], d / max(np.linalg.norm(d), 1e-9)


def offset_polyline(points: np.ndarray, offset: float) -> np.ndarray:
    """折线向左偏移 offset 米 (用相邻线段法向内插, 端点用端段法向)."""
    p = np.asarray(points, np.float64)
    if len(p) < 2:
        return p.copy()
    d = np.diff(p, axis=0)
    ln = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(ln, 1e-9)
    n = np.stack([-d[:, 1], d[:, 0]], 1)          # 左法向 (逐线段)
    vn = np.empty((len(p), 2), np.float64)        # 逐顶点法向
    vn[0] = n[0]
    vn[-1] = n[-1]
    if len(p) > 2:
        vn[1:-1] = n[:-1] + n[1:]
        vln = np.linalg.norm(vn[1:-1], axis=1, keepdims=True)
        vn[1:-1] = vn[1:-1] / np.maximum(vln, 1e-9)
    return p[:, :2] + vn * float(offset)


def road_polygon(points: np.ndarray, width: float) -> np.ndarray:
    """道路中心线 -> 路面多边形 (左侧正向 + 右侧逆向)."""
    h = max(float(width), 0.2) * 0.5
    left = offset_polyline(points, h)
    right = offset_polyline(points, -h)
    return np.vstack([left, right[::-1]])


def arrow_polygons(points: np.ndarray, width: float, size: float = 2.2,
                   reverse: bool = False, offset: float = 0.0,
                   fractions=(0.35, 0.68)):
    """沿道路生成方向箭头三角面片列表 (每个 (3,2))."""
    out = []
    p = np.asarray(points, np.float64)
    for f in fractions:
        pos, d = point_at_fraction(p, f)
        if reverse:
            d = -d
        nrm = np.array([-d[1], d[0]])
        c = pos[:2] + nrm * offset
        tri = np.array([c + d[:2] * size * 0.6,
                        c - d[:2] * size * 0.45 + nrm * size * 0.38,
                        c - d[:2] * size * 0.45 - nrm * size * 0.38])
        out.append(tri)
    return out


def convex_hull(xy: np.ndarray) -> np.ndarray:
    """Andrew monotone chain; 返回逆时针凸包顶点."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(xy, np.float64)})
    if len(pts) <= 2:
        return np.asarray(pts, np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-9:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-9:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], np.float64)


def chaikin(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin 角切平滑, 保持首尾端点不变."""
    p = np.asarray(points, np.float64)
    for _ in range(max(0, int(iterations))):
        if len(p) < 3:
            break
        q = [p[0]]
        for i in range(len(p) - 1):
            q.append(0.75 * p[i] + 0.25 * p[i + 1])
            q.append(0.25 * p[i] + 0.75 * p[i + 1])
        q.append(p[-1])
        p = np.asarray(q, np.float64)
    return p


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass
class RoadNode:
    id: str
    x: float
    y: float
    z: float = 0.0
    kind: str = "priority"      # priority | traffic_light | right_before_left | unregulated
    radius: float = 0.0         # 交叉口轮廓半径 (0 = 自动)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], np.float64)


@dataclass
class RoadEdge:
    id: str
    from_id: str
    to_id: str
    shape: list = field(default_factory=list)   # [(x,y,z), ...] 含两端点
    lanes: int = 2
    lane_width: float = _DEFAULT_LANE_WIDTH
    speed: float = _DEFAULT_SPEED
    priority: int = 1
    bidir: bool = False
    allow: str = ""
    name: str = ""

    @property
    def width(self) -> float:
        return max(1, int(self.lanes)) * float(self.lane_width)

    def direction_lanes(self) -> tuple[int, int]:
        """返回 (正向车道数, 反向车道数).

        双向道路的 ``lanes`` 是双向合计 (与编辑器绘制/箭头一致),
        导出 SUMO 时拆成两条相反方向的道路; 单向道路反向车道数为 0.
        """
        n = max(1, int(self.lanes))
        if not self.bidir:
            return n, 0
        fwd = max(1, n // 2)
        rev = max(1, n - fwd)
        return fwd, rev

    def points(self) -> np.ndarray:
        return np.asarray(self.shape, np.float64).reshape(-1, 3)

    def xy(self) -> np.ndarray:
        return self.points()[:, :2]

    def reverse(self):
        self.shape = list(reversed(self.shape))
        self.from_id, self.to_id = self.to_id, self.from_id

    def total_length(self) -> float:
        return polyline_length(self.xy())


@dataclass
class RoadConnection:
    from_edge: str
    to_edge: str
    from_lane: int = 0
    to_lane: int = 0


class RoadNetwork:
    """节点 / 边 / 连接 容器, 提供编辑与 SUMO 导出."""

    def __init__(self):
        self.nodes: dict[str, RoadNode] = {}
        self.edges: dict[str, RoadEdge] = {}
        self.connections: list[RoadConnection] = []
        self._nid = 0
        self._eid = 0

    # ---- 基础增删 ----
    def new_node_id(self) -> str:
        while f"n{self._nid}" in self.nodes:
            self._nid += 1
        self._nid += 1
        return f"n{self._nid - 1}"

    def new_edge_id(self) -> str:
        while f"e{self._eid}" in self.edges:
            self._eid += 1
        self._eid += 1
        return f"e{self._eid - 1}"

    def add_node(self, x: float, y: float, z: float = 0.0,
                 kind: str = "priority", nid: str = None) -> RoadNode:
        nid = nid or self.new_node_id()
        node = RoadNode(id=nid, x=float(x), y=float(y), z=float(z), kind=kind)
        self.nodes[nid] = node
        return node

    def nearest_node(self, x: float, y: float, max_dist: float = 3.0):
        best, bd = None, max_dist
        for n in self.nodes.values():
            d = math.hypot(n.x - x, n.y - y)
            if d <= bd:
                best, bd = n, d
        return best

    def add_edge(self, points, from_id: str = None, to_id: str = None, **attrs) -> RoadEdge:
        p = np.asarray(points, np.float64)
        if len(p) < 2:
            raise ValueError("道路至少需要两个点")
        if p.shape[1] == 2:
            p = np.column_stack([p, np.zeros(len(p))])
        if from_id is None:
            n = self.add_node(p[0, 0], p[0, 1], p[0, 2])
            from_id = n.id
        if to_id is None:
            n = self.add_node(p[-1, 0], p[-1, 1], p[-1, 2])
            to_id = n.id
        # 端点以节点坐标为准
        p[0] = [self.nodes[from_id].x, self.nodes[from_id].y, self.nodes[from_id].z]
        p[-1] = [self.nodes[to_id].x, self.nodes[to_id].y, self.nodes[to_id].z]
        e = RoadEdge(id=self.new_edge_id(), from_id=from_id, to_id=to_id,
                     shape=[tuple(map(float, q)) for q in p], **attrs)
        self.edges[e.id] = e
        return e

    def remove_edge(self, eid: str):
        self.edges.pop(eid, None)
        self.connections = [c for c in self.connections
                            if c.from_edge != eid and c.to_edge != eid]
        used = {e.from_id for e in self.edges.values()} | {e.to_id for e in self.edges.values()}
        for nid in [n for n in self.nodes if n not in used]:
            self.nodes.pop(nid, None)

    def remove_node(self, nid: str):
        for eid in [e.id for e in self.edges.values()
                    if e.from_id == nid or e.to_id == nid]:
            self.remove_edge(eid)
        self.nodes.pop(nid, None)

    # ---- 几何同步 ----
    def sync_endpoints(self, edge: RoadEdge):
        if edge.from_id in self.nodes and edge.shape:
            n = self.nodes[edge.from_id]
            edge.shape[0] = (n.x, n.y, n.z)
        if edge.to_id in self.nodes and edge.shape:
            n = self.nodes[edge.to_id]
            edge.shape[-1] = (n.x, n.y, n.z)

    def incident_edges(self, nid: str) -> list[RoadEdge]:
        return [e for e in self.edges.values() if e.from_id == nid or e.to_id == nid]

    # ---- 编辑操作 ----
    def split_edge(self, eid: str, seg_index: int, frac: float,
                   z: float = None) -> str:
        """在第 seg_index 段的 frac 比例处插入节点并一分为二, 返回新节点 id."""
        e = self.edges[eid]
        p = e.points()
        seg_index = int(np.clip(seg_index, 0, len(p) - 2))
        frac = float(np.clip(frac, 0.05, 0.95))
        q = p[seg_index] + (p[seg_index + 1] - p[seg_index]) * frac
        nz = float(q[2] if z is None else z)
        n = self.add_node(q[0], q[1], nz, kind=e.kind if hasattr(e, "kind") else "priority")
        left = np.vstack([p[:seg_index + 1], q])
        right = np.vstack([q, p[seg_index + 1:]])
        attrs = dict(lanes=e.lanes, lane_width=e.lane_width, speed=e.speed,
                     priority=e.priority, bidir=e.bidir, allow=e.allow, name=e.name)
        # 先建两条新边再删除旧边: remove_edge 会清理孤立节点, 需保证端点仍被引用
        self.add_edge(left, e.from_id, n.id, **attrs)
        self.add_edge(right, n.id, e.to_id, **attrs)
        self.remove_edge(eid)
        return n.id

    def split_edge_at_point(self, eid: str, x: float, y: float,
                            z: float = None) -> str:
        """按最近的折线段与投影位置切分边."""
        e = self.edges[eid]
        p = e.xy()
        best = (0, 0.5, 1e18)
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            d = b - a
            L2 = float(d @ d)
            k = 0.0 if L2 < 1e-12 else float(np.clip(((np.array([x, y]) - a) @ d) / L2, 0.02, 0.98))
            q = a + d * k
            dist = float(np.hypot(q[0] - x, q[1] - y))
            if dist < best[2]:
                best = (i, k, dist)
        return self.split_edge(eid, best[0], best[1], z)

    # ---- 形状编辑 / 平滑 / 自动连接 ----
    def nearest_on_edge(self, eid: str, x: float, y: float):
        """返回 (seg_index, frac, point2d, dist) 的最近投影."""
        e = self.edges[eid]
        p = e.xy()
        best = (0, 0.5, p[0], 1e18)
        for i in range(len(p) - 1):
            a, b = p[i], p[i + 1]
            d = b - a
            L2 = float(d @ d)
            k = 0.0 if L2 < 1e-12 else float(np.clip(((np.array([x, y]) - a) @ d) / L2, 0.0, 1.0))
            q = a + d * k
            dist = float(np.hypot(q[0] - x, q[1] - y))
            if dist < best[3]:
                best = (i, k, q, dist)
        return best

    def insert_vertex(self, eid: str, x: float, y: float,
                      z: float = None) -> int:
        """在道路最近位置插入一个形状点, 返回新点索引."""
        e = self.edges[eid]
        i, k, q, _ = self.nearest_on_edge(eid, x, y)
        p = e.points()
        zz = float(z if z is not None else
                   p[i, 2] + (p[i + 1, 2] - p[i, 2]) * k)
        e.shape.insert(i + 1, (float(q[0]), float(q[1]), zz))
        return i + 1

    def smooth_edge(self, eid: str, iterations: int = 2) -> int:
        """对道路形状做 Chaikin 平滑 (可作为曲线), 返回新形状点数."""
        e = self.edges[eid]
        p = e.points()
        if len(p) < 3:
            return len(p)
        seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
        s_old = np.concatenate([[0.0], np.cumsum(seg)])
        smooth_xy = chaikin(p[:, :2], iterations)
        seg_new = np.linalg.norm(np.diff(smooth_xy, axis=0), axis=1)
        s_new = np.concatenate([[0.0], np.cumsum(seg_new)])
        z_new = np.interp(s_new, s_old, p[:, 2]) if s_old[-1] > 1e-9 else p[0, 2]
        e.shape = [(float(a), float(b), float(c))
                   for (a, b), c in zip(smooth_xy, z_new)]
        # 端点严格对齐节点
        self.sync_endpoints(e)
        return len(e.shape)

    @staticmethod
    def _endpoint_info(e: RoadEdge, which: str):
        p = e.points()
        if which == "from":
            pos = p[0]
            nid = e.from_id
            out = p[0] - p[min(1, len(p) - 1)]      # 从端点向外
            into = p[min(1, len(p) - 1)] - p[0]     # 进入道路方向
        else:
            pos = p[-1]
            nid = e.to_id
            out = p[-1] - p[max(-2, -len(p))]
            into = p[max(-2, -len(p))] - p[-1]
        return {"pos": pos, "nid": nid, "out": out, "into": into, "which": which}

    def connect_edges(self, e1: RoadEdge, e2: RoadEdge, z_fn=None,
                      samples: int = 24):
        """在两条道路最近的端点间生成平滑连接段 (宽度/车道数渐变).

        返回 (new_edge_ids, node_start, node_end); 若无可行端点返回 ([], None, None).
        """
        if e1 is e2 or e1.id not in self.edges or e2.id not in self.edges:
            return [], None, None
        best = None
        for a in ("from", "to"):
            i1 = self._endpoint_info(e1, a)
            for b in ("from", "to"):
                i2 = self._endpoint_info(e2, b)
                if i1["nid"] == i2["nid"]:
                    continue
                if any((e.from_id, e.to_id) == (i1["nid"], i2["nid"]) or
                       (e.from_id, e.to_id) == (i2["nid"], i1["nid"])
                       for e in self.edges.values()):
                    continue
                d = float(np.linalg.norm(i2["pos"][:2] - i1["pos"][:2]))
                if best is None or d < best[0]:
                    best = (d, i1, i2)
        if best is None:
            return [], None, None
        dist, a, b = best
        p1 = np.asarray(a["pos"][:2], np.float64)
        p2 = np.asarray(b["pos"][:2], np.float64)
        d1 = np.asarray(a["out"][:2], np.float64)
        d2 = np.asarray(b["into"][:2], np.float64)
        d1 /= max(np.linalg.norm(d1), 1e-9)
        d2 /= max(np.linalg.norm(d2), 1e-9)
        k = max(dist / 3.0, 2.0)
        c1 = p1 + d1 * k
        c2 = p2 - d2 * k
        ts = np.linspace(0.0, 1.0, max(6, int(samples)))
        curve = ((1 - ts) ** 3)[:, None] * p1 + \
                (3 * (1 - ts) ** 2 * ts)[:, None] * c1 + \
                (3 * (1 - ts) * ts ** 2)[:, None] * c2 + \
                (ts ** 3)[:, None] * p2
        if z_fn is not None:
            zs = np.asarray(z_fn(curve), np.float64).ravel()
        else:
            zs = np.linspace(a["pos"][2], b["pos"][2], len(ts))
        lanes1, lanes2 = int(e1.lanes), int(e2.lanes)
        nseg = 1 if lanes1 == lanes2 else min(4, abs(lanes1 - lanes2) + 1)
        lane_w = (float(e1.lane_width) + float(e2.lane_width)) * 0.5
        speed = min(float(e1.speed), float(e2.speed))
        priority = max(int(e1.priority), int(e2.priority))

        node_start, node_end = a["nid"], b["nid"]
        prev_nid = node_start
        ids = []
        splits = np.linspace(0, len(ts) - 1, nseg + 1).astype(int)
        for j in range(nseg):
            i0, i1 = splits[j], splits[j + 1]
            sub = np.column_stack([curve[i0:i1 + 1], zs[i0:i1 + 1]])
            if j == nseg - 1:
                nxt = node_end
            else:
                q = sub[-1]
                nxt = self.add_node(q[0], q[1], q[2]).id
            lanes_j = int(round(lanes1 + (lanes2 - lanes1) *
                               ((j + 0.5) / nseg)))
            e = self.add_edge(sub, prev_nid, nxt, lanes=max(1, lanes_j),
                              lane_width=lane_w, speed=speed,
                              priority=priority)
            ids.append(e.id)
            prev_nid = nxt
        return ids, node_start, node_end

    # ---- 交叉口创建 ----
    @staticmethod
    def _seg_intersection(p1, p2, p3, p4):
        r = p2 - p1
        s = p4 - p3
        den = r[0] * s[1] - r[1] * s[0]
        if abs(den) < 1e-10:
            return None
        q = p3 - p1
        t = (q[0] * s[1] - q[1] * s[0]) / den
        u = (q[0] * r[1] - q[1] * r[0]) / den
        if -1e-6 <= t <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6:
            return p1 + t * r, float(t), float(u), t
        return None

    def find_crossing_near(self, x: float, y: float, tol: float = 8.0):
        """找到距 (x,y) 最近的两条道路几何交点; 返回 (e1, e2, point)."""
        ids = list(self.edges)
        best = None
        for ii in range(len(ids)):
            e1 = self.edges[ids[ii]]
            p1 = e1.xy()
            for jj in range(ii + 1, len(ids)):
                e2 = self.edges[ids[jj]]
                p2 = e2.xy()
                # 已共享节点则无需创建
                if ({e1.from_id, e1.to_id} & {e2.from_id, e2.to_id}):
                    continue
                for a in range(len(p1) - 1):
                    for b in range(len(p2) - 1):
                        hit = self._seg_intersection(p1[a], p1[a + 1],
                                                     p2[b], p2[b + 1])
                        if hit is None:
                            continue
                        q = hit[0]
                        d = float(np.hypot(q[0] - x, q[1] - y))
                        if d <= tol and (best is None or d < best[0]):
                            best = (d, e1, e2, q)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def merge_nodes(self, a: str, b: str):
        """把节点 b 并入 a (b 被删除), 用于交叉口."""
        if a == b or a not in self.nodes or b not in self.nodes:
            return
        na = self.nodes[a]
        for e in list(self.edges.values()):
            if e.from_id == b:
                e.from_id = a
                if e.shape:
                    e.shape[0] = (na.x, na.y, na.z)
            if e.to_id == b:
                e.to_id = a
                if e.shape:
                    e.shape[-1] = (na.x, na.y, na.z)
            if e.from_id == e.to_id:          # 合并后形成自环 -> 删除
                self.remove_edge(e.id)
        self.nodes.pop(b, None)

    def create_intersection(self, e1: RoadEdge, e2: RoadEdge,
                            x: float, y: float, z: float = None,
                            kind: str = "priority") -> str:
        """在 e1/e2 的几何交点处各分割一次并合并节点, 返回交叉口节点 id."""
        n1 = self.split_edge_at_point(e1.id, x, y, z)
        # e2 可能因共享节点被 remove_edge 清理, 重新取
        e2b = self.edges.get(e2.id)
        if e2b is None:
            return n1
        n2 = self.split_edge_at_point(e2b.id, x, y, z)
        self.merge_nodes(n1, n2)
        if n1 in self.nodes:
            self.nodes[n1].kind = kind
        return n1

    def set_elevation(self, ground_fn, samples_per_edge: int = 0):
        """用地面高程函数给节点和形状点赋 z.

        ground_fn(xy (n,2)) -> (n,) 高程.
        """
        if not self.nodes:
            return
        xy = np.array([[n.x, n.y] for n in self.nodes.values()], np.float64)
        z = np.asarray(ground_fn(xy), np.float64).ravel()
        for n, zz in zip(self.nodes.values(), z):
            if np.isfinite(zz):
                n.z = float(zz)
        for e in self.edges.values():
            p = e.points()
            gz = np.asarray(ground_fn(p[:, :2]), np.float64).ravel()
            for i in range(1, len(p) - 1):
                if np.isfinite(gz[i]):
                    p[i, 2] = gz[i]
            p[0, 2] = self.nodes[e.from_id].z
            p[-1, 2] = self.nodes[e.to_id].z
            e.shape = [tuple(map(float, q)) for q in p]

    # ---- 可视化几何 ----
    def junction_outline(self, nid: str, use_radius: bool = True) -> np.ndarray:
        node = self.nodes.get(nid)
        if node is None:
            return np.zeros((0, 2))
        edges = self.incident_edges(nid)
        if use_radius and node.radius > 0:
            t = np.linspace(0, 2 * np.pi, 25)[:-1]
            return np.stack([node.x + node.radius * np.cos(t),
                             node.y + node.radius * np.sin(t)], 1)
        pts = [[node.x, node.y]]
        auto_r = 3.0
        for e in edges:
            p = e.xy()
            if e.from_id == nid and len(p) > 1:
                q = p[1]
            elif e.to_id == nid and len(p) > 1:
                q = p[-2]
            else:
                continue
            d = q - np.array([node.x, node.y])
            ln = float(np.linalg.norm(d))
            if ln < 1e-6:
                continue
            auto_r = max(auto_r, min(e.width * 0.7 + 1.5, 12.0))
            pts.append([node.x + d[0] / ln * auto_r, node.y + d[1] / ln * auto_r])
        return convex_hull(np.asarray(pts, np.float64))

    def bounds(self):
        pts = []
        for n in self.nodes.values():
            pts.append([n.x, n.y, n.z])
        for e in self.edges.values():
            pts.extend(e.points().tolist())
        if not pts:
            return np.zeros(3), np.zeros(3)
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        return p.min(0), p.max(0)

    # ------------------------------------------------------------------
    # SUMO 明文 XML 导出
    # ------------------------------------------------------------------
    def write_plain_xml(self, out_dir: str, basename: str = "roadnet") -> dict:
        """写出 nodes/edges/connections/types 四个明文文件, 返回路径字典."""
        os.makedirs(out_dir, exist_ok=True)
        paths = {
            "nodes": os.path.join(out_dir, f"{basename}.nod.xml"),
            "edges": os.path.join(out_dir, f"{basename}.edg.xml"),
            "conns": os.path.join(out_dir, f"{basename}.con.xml"),
            "types": os.path.join(out_dir, f"{basename}.typ.xml"),
            "netccfg": os.path.join(out_dir, f"{basename}.netccfg"),
        }
        root = ET.Element("nodes")
        for n in self.nodes.values():
            ET.SubElement(root, "node", {
                "id": n.id, "x": _f(n.x), "y": _f(n.y), "z": _f(n.z),
                "type": n.kind})
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(paths["nodes"], encoding="utf-8", xml_declaration=True)

        root = ET.Element("edges")

        def emit_edge(e: RoadEdge, eid: str, from_id: str, to_id: str, shape,
                      lanes: int, spread: str = "center"):
            attrs = {
                "id": eid, "from": from_id, "to": to_id,
                "numLanes": str(max(1, int(lanes))),
                "width": _f(e.lane_width),
                "speed": _f(e.speed), "priority": str(int(e.priority)),
                # 编辑器把折线当作道路中心线绘制; netconvert 默认 spreadType=right
                # 会把车道整体推到折线右侧, 导致导出后与点云底图错位.
                "spreadType": spread,
            }
            if e.allow:
                attrs["allow"] = e.allow
            if e.name:
                attrs["name"] = e.name
            if len(shape) > 2:
                attrs["shape"] = " ".join(f"{_f(p[0])},{_f(p[1])}" for p in shape)
            ET.SubElement(root, "edge", attrs)

        for e in self.edges.values():
            nf, nr = e.direction_lanes()
            if e.bidir:
                # 双向道路: 两条相反方向的边共用中心线, SUMO 默认的 right
                # 会把它们分别铺在中心线两侧, 正好对应编辑器里的双向路面.
                emit_edge(e, e.id, e.from_id, e.to_id, e.shape, nf, "right")
                emit_edge(e, f"{e.id}_r", e.to_id, e.from_id,
                          list(reversed(e.shape)), nr, "right")
            else:
                emit_edge(e, e.id, e.from_id, e.to_id, e.shape, nf)
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(paths["edges"], encoding="utf-8", xml_declaration=True)

        root = ET.Element("connections")
        for c in self.connections:
            ET.SubElement(root, "connection", {
                "from": c.from_edge, "to": c.to_edge,
                "fromLane": str(int(c.from_lane)), "toLane": str(int(c.to_lane))})
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(paths["conns"], encoding="utf-8", xml_declaration=True)

        root = ET.Element("routes")
        ET.SubElement(root, "vType", {
            "id": "lidarsim_car", "vClass": "passenger",
            "length": "5.0", "width": "1.8", "maxSpeed": _f(_DEFAULT_SPEED)})
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(paths["types"], encoding="utf-8", xml_declaration=True)

        cfg = ET.Element("configuration")
        inp = ET.SubElement(cfg, "input")
        ET.SubElement(inp, "node-files", {"value": os.path.basename(paths["nodes"])})
        ET.SubElement(inp, "edge-files", {"value": os.path.basename(paths["edges"])})
        if self.connections:
            ET.SubElement(inp, "connection-files", {"value": os.path.basename(paths["conns"])})
        out = ET.SubElement(cfg, "output")
        ET.SubElement(out, "output-file", {"value": f"{basename}.net.xml"})
        proc = ET.SubElement(cfg, "processing")
        ET.SubElement(proc, "no-turnarounds", {"value": "true"})
        ET.SubElement(proc, "offset.disable-normalization", {"value": "true"})
        ET.SubElement(proc, "junctions.corner-detail", {"value": "5"})
        ET.indent(cfg, space="  ")
        ET.ElementTree(cfg).write(paths["netccfg"], encoding="utf-8", xml_declaration=True)
        return paths

    def write_demo_traffic(self, out_dir: str, net_file: str,
                           n_flow: int = 6, veh_per_hour: int = 360,
                           begin: float = 0.0, end: float = 3600.0,
                           seed: int = 42) -> str:
        """生成可直接运行的示例交通流 routes.rou.xml."""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "routes.rou.xml")
        edges = [e for e in self.edges.values() if e.total_length() > 5.0]
        root = ET.Element("routes")
        ET.SubElement(root, "vType", {
            "id": "car", "vClass": "passenger", "length": "5.0", "width": "1.8",
            "maxSpeed": _f(_DEFAULT_SPEED), "speedFactor": "1.0"})
        if len(edges) >= 2:
            rng = np.random.default_rng(seed)
            by_from: dict[str, list] = {}
            for e in edges:
                by_from.setdefault(e.from_id, []).append(e)
            pairs = []
            for a in edges:
                outs = [b for b in by_from.get(a.to_id, []) if b.id != a.id]
                if outs:
                    pairs.append((a, outs[int(rng.integers(len(outs)))]))
            if not pairs:                       # 无直接衔接: 退化到随机起终点
                for _ in range(len(edges)):
                    a = edges[int(rng.integers(len(edges)))]
                    b = edges[int(rng.integers(len(edges)))]
                    if a.id != b.id:
                        pairs.append((a, b))
                        break
            if not pairs:
                pairs = [(edges[0], edges[0])]
            for i in range(max(1, n_flow)):
                a, b = pairs[i % len(pairs)]
                ET.SubElement(root, "flow", {
                    "id": f"flow{i}", "type": "car", "begin": _f(begin),
                    "end": _f(end), "vehsPerHour": str(int(veh_per_hour)),
                    "from": a.id, "to": b.id, "departLane": "best",
                    "departSpeed": "max"})
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        return path

    def write_sumo_config(self, out_dir: str, net_file: str, route_file: str,
                          begin: float = 0.0, end: float = 3600.0,
                          step_length: float = 0.1, name: str = "lidarsim.sumocfg") -> str:
        path = os.path.join(out_dir, name)
        root = ET.Element("configuration")
        inp = ET.SubElement(root, "input")
        ET.SubElement(inp, "net-file", {"value": os.path.basename(net_file)})
        ET.SubElement(inp, "route-files", {"value": os.path.basename(route_file)})
        tm = ET.SubElement(root, "time")
        ET.SubElement(tm, "begin", {"value": _f(begin)})
        ET.SubElement(tm, "end", {"value": _f(end)})
        ET.SubElement(tm, "step-length", {"value": _f(step_length)})
        rep = ET.SubElement(root, "report")
        ET.SubElement(rep, "no-step-log", {"value": "true"})
        ET.SubElement(rep, "no-warnings", {"value": "true"})
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        return path

    # ------------------------------------------------------------------
    # netconvert / net.xml
    # ------------------------------------------------------------------
    @staticmethod
    def run_netconvert(netconvert: str, netccfg: str, timeout: int = 120):
        """调用 netconvert 生成 net.xml; 返回 (ok, 标准输出+错误)."""
        try:
            cp = subprocess.run([netconvert, "-c", netccfg],
                                capture_output=True, text=True, timeout=timeout)
            log = (cp.stdout or "") + (cp.stderr or "")
            return cp.returncode == 0, log.strip()
        except FileNotFoundError:
            return False, f"netconvert 不存在: {netconvert}"
        except subprocess.TimeoutExpired:
            return False, "netconvert 执行超时"
        except Exception as e:                                    # noqa: BLE001
            return False, f"netconvert 调用失败: {e}"

    def export_scenario(self, out_dir: str, basename: str = "roadnet",
                        netconvert: str = None, demo_traffic: bool = True,
                        begin: float = 0.0, end: float = 3600.0,
                        step_length: float = 0.1) -> dict:
        """明文 XML + (可选) net.xml + 示例交通流 + sumocfg 一次性导出."""
        paths = self.write_plain_xml(out_dir, basename)
        result = {"plain": paths, "net": "", "routes": "", "cfg": "",
                  "log": "", "netconvert": netconvert or find_netconvert()}
        net_file = os.path.join(out_dir, f"{basename}.net.xml")
        if result["netconvert"]:
            ok, log = self.run_netconvert(result["netconvert"], paths["netccfg"])
            result["log"] = log
            if ok and os.path.exists(net_file):
                result["net"] = net_file
        if result["net"] and demo_traffic:
            result["routes"] = self.write_demo_traffic(out_dir, net_file,
                                                       begin=begin, end=end)
            result["cfg"] = self.write_sumo_config(out_dir, net_file,
                                                   result["routes"], begin=begin,
                                                   end=end, step_length=step_length)
        return result

    @classmethod
    def from_net_xml(cls, path: str) -> "RoadNetwork":
        """读取 SUMO net.xml (只保留普通节点/边, 忽略 internal)."""
        root = ET.parse(path).getroot()
        net = cls()
        for j in root.findall("junction"):
            jid = j.get("id", "")
            if jid.startswith(":") or j.get("type") == "internal":
                continue
            try:
                net.add_node(float(j.get("x", 0)), float(j.get("y", 0)),
                             float(j.get("z", 0)), kind="priority", nid=jid)
            except (TypeError, ValueError):
                continue
        for e in root.findall("edge"):
            if e.get("function") == "internal":
                continue
            lanes = e.findall("lane")
            if not lanes:
                continue
            lane_pts = []
            for ln in lanes:
                pts = []
                for tok in ln.get("shape", "").split():
                    v = tok.split(",")
                    if len(v) >= 2:
                        pts.append([float(v[0]), float(v[1]),
                                    float(v[2]) if len(v) > 2 else 0.0])
                if len(pts) >= 2:
                    lane_pts.append(np.asarray(pts, np.float64))
            if not lane_pts:
                continue
            # net.xml 的 edge shape 不是车道组中心线 (默认位于左侧边界);
            # 取各车道中心线的平均, 还原后重新导出才不会偏移.
            if all(len(q) == len(lane_pts[0]) for q in lane_pts):
                pts = np.mean(np.stack(lane_pts, 0), axis=0)
            else:
                pts = lane_pts[len(lane_pts) // 2]
            if len(pts) < 2:
                continue
            from_id, to_id = e.get("from"), e.get("to")
            if from_id not in net.nodes or to_id not in net.nodes:
                continue
            try:
                lanes_n = len(lanes)
                width = float(lanes[0].get("width", _DEFAULT_LANE_WIDTH))
                speed = float(lanes[0].get("speed", _DEFAULT_SPEED))
                edge = net.add_edge(pts, from_id, to_id, lanes=lanes_n,
                                    lane_width=width, speed=speed,
                                    name=e.get("name", ""))
                eid = e.get("id", "")
                if eid and eid != edge.id:
                    net.edges.pop(edge.id, None)
                    edge.id = eid
                    net.edges[eid] = edge
            except Exception:                                     # noqa: BLE001
                continue
        # 保证 id 计数器不冲突
        net._nid = len(net.nodes) + 1
        net._eid = len(net.edges) + 1
        return net

    def clear(self):
        self.nodes.clear()
        self.edges.clear()
        self.connections.clear()
        self._nid = self._eid = 0


# ---------------------------------------------------------------------------
# SUMO 工具定位
# ---------------------------------------------------------------------------
def find_sumo_home() -> str:
    home = os.environ.get("SUMO_HOME", "")
    if home and os.path.isdir(home):
        return home
    for p in (r"C:\Program Files (x86)\Eclipse\Sumo",
              r"C:\Program Files\Eclipse\Sumo",
              r"C:\Eclipse\Sumo"):
        if os.path.isdir(p):
            return p
    try:
        import importlib.util as ilu
        spec = ilu.find_spec("sumo")
        if spec and spec.submodule_search_locations:
            base = list(spec.submodule_search_locations)[0]
            if os.path.isdir(os.path.join(base, "bin")):
                return base
    except Exception:                                             # noqa: BLE001
        pass
    return ""


def find_sumo_binary(gui: bool = False) -> str:
    name = "sumo-gui" if gui else "sumo"
    exe = name + (".exe" if os.name == "nt" else "")
    w = shutil.which(name) or shutil.which(exe)
    if w:
        return w
    home = find_sumo_home()
    if home:
        for p in (os.path.join(home, "bin", exe), os.path.join(home, "bin", name)):
            if os.path.exists(p):
                return p
    return ""


def find_netconvert() -> str:
    exe = "netconvert" + (".exe" if os.name == "nt" else "")
    w = shutil.which("netconvert") or shutil.which(exe)
    if w:
        return w
    home = find_sumo_home()
    if home:
        p = os.path.join(home, "bin", exe)
        if os.path.exists(p):
            return p
    return ""


def describe_sumo_env() -> str:
    home = find_sumo_home()
    sumo = find_sumo_binary(False)
    netc = find_netconvert()
    parts = [f"SUMO_HOME={'未设置' if not home else home}",
             f"sumo={'未找到' if not sumo else sumo}",
             f"netconvert={'未找到' if not netc else netc}"]
    return "; ".join(parts)


def _f(v: float) -> str:
    """紧凑数值格式, 避免 XML 中出现过长小数."""
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"

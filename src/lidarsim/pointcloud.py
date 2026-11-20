# -*- coding: utf-8 -*-
"""环境点云加载 (PLY / H5), 地面提取与高程插值, 反射率与法向估计."""
from dataclasses import dataclass, field
import numpy as np

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False

from scipy.spatial import cKDTree


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------
_PLY_TYPE = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "ushort": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _load_ply(path: str):
    with open(path, "rb") as f:
        head = f.read(65536)
        hend = head.find(b"end_header")
        if hend < 0:
            raise ValueError(f"PLY 头部解析失败: {path}")
        lines = head[:hend].decode("ascii", "ignore").strip().splitlines()
        fmt, nvert, props = None, 0, []
        for ln in lines:
            tk = ln.split()
            if not tk:
                continue
            if tk[0] == "format":
                fmt = tk[1]
            elif tk[0] == "element" and tk[1] == "vertex":
                nvert = int(tk[2])
            elif tk[0] == "property":
                props.append((tk[2], _PLY_TYPE[tk[1]]))
        off = hend + len(b"end_header")
        while off < len(head) and head[off:off + 1] in (b"\n", b"\r"):
            off += 1
        dt = np.dtype(props)
        if fmt == "ascii":
            f.seek(off)
            raw = np.loadtxt(f, max_rows=nvert)
            arr = np.empty(nvert, dtype=dt)
            for i, name in enumerate(dt.names):
                arr[name] = raw[:, i]
            return arr
        f.seek(off)
        return np.fromfile(f, dtype=dt, count=nvert)


def _load_h5(path: str):
    """读取 H5 点云: 递归查找 (N,3) 坐标 + 强度 + 语义/实例标签.

    不同数据集的键名与形状不同 (coords/points/xyz, semantics/labels/class,
    1D 或 (N,1) 或 one-hot (N,C)), 这里做自适应识别.
    """
    import h5py

    with h5py.File(path, "r") as f:
        items = []

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append((name, obj.shape, obj.dtype, obj))

        f.visititems(visit)

        def find(keys, ndim=None, cols=None):
            cands = []
            for name, shape, dtype, ds in items:
                base = name.rsplit("/", 1)[-1].lower()
                if base in keys:
                    cands.append((len(name), name, shape, ds))
            cands.sort(key=lambda t: t[0])      # 优先浅层/短路径
            for _, name, shape, ds in cands:
                if ndim is not None and len(shape) != ndim:
                    continue
                if cols is not None and (len(shape) < 2 or shape[1] != cols):
                    continue
                return ds, name
            return None, None

        pts, pname = find(("coords", "points", "xyz", "pos"), ndim=2, cols=3)
        if pts is None:                          # 任意 (N,3) 数据集
            for name, shape, dtype, ds in items:
                if len(shape) == 2 and shape[1] == 3:
                    pts, pname = ds, name
                    break
        if pts is None:
            raise ValueError(f"H5 中未找到 (N,3) 坐标数据集: {path}")
        n = int(pts.shape[0])

        def read_vec(ds):
            a = ds[()]
            if a.ndim == 2:
                if a.shape[0] == n and a.shape[1] == 1:
                    a = a[:, 0]
                elif a.shape[1] == n:
                    a = a.T
                    a = a[:, 0]
                elif a.shape[0] == n:
                    a = np.argmax(a, 1)          # one-hot -> 类别索引
            return np.asarray(a).ravel()

        inten, _ = find(("intensity", "reflectance", "intensities"))
        inten = read_vec(inten) if inten is not None and inten.shape[0] == n else None
        if inten is not None and inten.size != n:
            inten = None

        # 语义标签: 名称自适应, 且必须与点数一致
        sem, sname = None, ""
        sem_keys = ("semantics", "semantic", "sem", "sem_label", "semantic_label",
                    "semantic_id", "class", "classes", "class_id",
                    "label", "labels", "seg", "segmentation", "seg_labels")
        sem_ds, sname = find(sem_keys)
        if sem_ds is not None:
            try:
                arr = read_vec(sem_ds).astype(np.int64)
                if arr.size == n:
                    sem = arr
                else:
                    sname = ""
            except Exception:                                  # noqa: BLE001
                sem, sname = None, ""

        inst, _ = find(("instances", "instance", "instance_ids", "instance_id"))
        inst = read_vec(inst) if inst is not None and inst.shape[0] == n else None
        if inst is not None and inst.size != n:
            inst = None
        # 立即读出为 ndarray, 避免文件关闭后 h5py 数据集失效
        return np.asarray(pts[()]), inten, sem, inst, pname, sname


# PLY 语义属性名优先级 (instance 是实例 id, 不作为语义)
_PLY_SEM_KEYS = ("semantic", "semantics", "sem_label", "semantic_label",
                 "semantic_id", "class", "class_id", "classes",
                 "label", "labels")


def _ply_semantic_field(names):
    low = {n.lower(): n for n in names}
    for key in _PLY_SEM_KEYS:
        if key in low:
            return low[key]
    for lname, real in low.items():
        if "semantic" in lname:
            return real
    return None


def _as_labels(values, n: int):
    """任意数值类型 -> (n,) int32 语义标签; 长度不符返回 None."""
    a = np.asarray(values).ravel()
    if a.size != n:
        return None
    if a.dtype.kind == "f":
        a = np.rint(np.nan_to_num(a, nan=0.0))
    try:
        return a.astype(np.int32)
    except (TypeError, ValueError):
        return None


@dataclass
class EnvCloud:
    """环境点云 (世界坐标系, 与轨迹/传感器位姿同一坐标系)."""
    points: np.ndarray            # (N,3) float32
    reflectance: np.ndarray       # (N,) float32 [0,1]
    normals: np.ndarray = None    # (N,3) float32
    semantics: np.ndarray = None  # (N,) int
    rgb: np.ndarray = None        # (N,3) uint8
    extras: dict = field(default_factory=dict)


def load_env(path: str) -> EnvCloud:
    """加载环境点云; 自适应识别不同数据集的语义标签编码 (PLY/H5)."""
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "ply":
        arr = _load_ply(path)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
        names = arr.dtype.names
        sem_field = _ply_semantic_field(names)
        sem = _as_labels(arr[sem_field], len(xyz)) if sem_field else None
        rgb = None
        if all(k in names for k in ("red", "green", "blue")):
            rgb = np.stack([arr["red"], arr["green"], arr["blue"]], 1).astype(np.uint8)
        refl = None
        if rgb is not None:  # 灰度 -> 反射率 (沥青灰度~0.15 -> rho~0.15)
            gray = rgb.astype(np.float32).mean(1) / 255.0
            refl = np.clip(0.04 + 0.75 * gray, 0.04, 0.85).astype(np.float32)
        extras = {k: arr[k] for k in names
                  if k not in ("x", "y", "z", "red", "green", "blue")}
        if sem is not None:
            extras["semantic_field"] = sem_field
        return EnvCloud(points=xyz, reflectance=refl, semantics=sem, rgb=rgb,
                        extras=extras)
    elif ext == "h5":
        pts, inten, sem, inst, pname, sname = _load_h5(path)
        pts = pts.astype(np.float32)
        refl = None
        if inten is not None:
            refl = np.clip(inten.astype(np.float32) / 65535.0, 0.02, 0.95)
        sem = _as_labels(sem, len(pts)) if sem is not None else None
        extras = {"instances": inst, "points_field": pname}
        if sem is not None:
            extras["semantic_field"] = sname
        return EnvCloud(points=pts, reflectance=refl, semantics=sem, extras=extras)
    raise ValueError(f"不支持的环境点云格式: {path} (支持 .ply / .h5)")


def make_flat_env(traj, margin: float = 25.0, cell: float = 0.5,
                  z: float = 0.0, refl: float = 0.15,
                  max_points: int = 1_500_000) -> EnvCloud:
    """道路环境缺省: 轨迹包围盒范围内 z=z 的水平点阵.

    用于未导入环境点云的情形 —— 所有车辆个体位于同一水平面上, 无其他障碍物.
    返回结构与 load_env 一致 (points + reflectance), 因此 ray-casting CSR 网格、
    地面模型 (GroundModel) 与 POG 全流程无需改动即可正常运行.
    """
    p = np.vstack([np.asarray(traj.front), np.asarray(traj.rear)])
    lo = p.min(0) - margin
    hi = p.max(0) + margin
    ext = np.maximum(hi - lo, cell)
    if (ext[0] / cell) * (ext[1] / cell) > max_points:   # 控制总点数
        cell = float(np.sqrt(ext[0] * ext[1] / max_points))
        cell = max(cell, 0.1)
    xs = np.arange(lo[0], hi[0] + cell * 0.5, cell)
    ys = np.arange(lo[1], hi[1] + cell * 0.5, cell)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1).astype(np.float32)
    return EnvCloud(points=pts,
                    reflectance=np.full(len(pts), refl, np.float32),
                    extras={"flat": True, "cell": cell})


def _polygon_area(xy: np.ndarray) -> float:
    p = np.asarray(xy, np.float64)
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], 1)) -
                           np.dot(p[:, 1], np.roll(p[:, 0], 1))))


def _points_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """向量化 ray-casting 多边形内点判定 (xy (n,2), poly (m,2))."""
    x = xy[:, 0]
    y = xy[:, 1]
    inside = np.zeros(len(xy), bool)
    p = np.asarray(poly, np.float64)
    j = len(p) - 1
    for i in range(len(p)):
        xi, yi = p[i]
        xj, yj = p[j]
        cond = ((yi > y) != (yj > y)) & \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def make_roadnet_env(net, cell: float = 0.5, z: float = 0.0,
                     refl: float = 0.15, max_points: int = 2_000_000) -> EnvCloud:
    """由 SUMO 路网 (net.xml / RoadNetwork) 的道路多边形生成地面点阵.

    没有真实背景点云时, 用路面覆盖多边形在 z=z 平面生成规则点阵, 使
    ray-cast / 地面模型 / POG / 联仿全流程与真实点云完全一致.
    """
    from .roadnet import road_polygon

    edges = [e for e in getattr(net, "edges", {}).values()
             if len(getattr(e, "shape", []) or []) >= 2]
    if not edges:
        raise ValueError("路网中没有可用道路, 无法生成点阵")
    polys = []
    area = 0.0
    for e in edges:
        p = e.points()[:, :2]
        if len(p) < 2:
            continue
        poly = road_polygon(p, float(e.width))
        polys.append(poly)
        area += _polygon_area(poly)
    if not polys:
        raise ValueError("路网道路几何为空, 无法生成点阵")
    # 按总面积自适应网格, 控制总点数
    cell = max(0.1, float(cell))
    if area > 0:
        cell = max(cell, float(np.sqrt(area / max(1, int(max_points)))))
    xs_all, ys_all = [], []
    for poly in polys:
        lo = poly.min(0) - cell
        hi = poly.max(0) + cell
        xs = np.arange(lo[0], hi[0] + cell * 0.5, cell)
        ys = np.arange(lo[1], hi[1] + cell * 0.5, cell)
        if xs.size == 0 or ys.size == 0:
            continue
        X, Y = np.meshgrid(xs, ys)
        cand = np.stack([X.ravel(), Y.ravel()], 1)
        m = _points_in_polygon(cand, poly)
        if m.any():
            xs_all.append(cand[m, 0])
            ys_all.append(cand[m, 1])
    if not xs_all:
        raise ValueError("路面多边形内未生成任何点")
    x = np.concatenate(xs_all)
    y = np.concatenate(ys_all)
    # 重叠路面去重 (同一栅格只保留一个点)
    key = np.stack([np.floor(x / cell), np.floor(y / cell)], 1).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    x, y = x[idx], y[idx]
    if len(x) > max_points:
        sel = np.linspace(0, len(x) - 1, int(max_points)).astype(np.int64)
        x, y = x[sel], y[sel]
    pts = np.stack([x, y, np.full(len(x), float(z))], 1).astype(np.float32)
    return EnvCloud(points=pts,
                    reflectance=np.full(len(pts), float(refl), np.float32),
                    extras={"roadnet": True, "cell": float(cell),
                            "n_edges": len(polys)})


def suggest_roi(points, z_lo_pad: float = 0.5, z_hi_pad: float = 4.0):
    """由点云包围盒给出 POG 默认 ROI; 近平面环境 (缺省道路) 竖向补齐.

    缺省平坦点云的 z 跨度 ≈ 0, 若直接用包围盒, 体素在 z 方向只有 1 层, 高于地面的
    车辆点会被裁掉. 故竖向上下各留余量.
    """
    pts = np.asarray(points, np.float64)
    lo, hi = pts.min(0).copy(), pts.max(0).copy()
    if hi[2] - lo[2] < 1.0:
        lo[2] -= z_lo_pad
        hi[2] += z_hi_pad
    return lo, hi


# --------------------------------------------------------------------------
# 地面提取 + 高程插值
# --------------------------------------------------------------------------
class GroundModel:
    """基于 2D 网格局部最低分位数的地面高程场, 双线性插值查询 G(x,y)."""

    def __init__(self, points: np.ndarray, cell: float = 1.0, pctl: float = 5.0):
        p = np.asarray(points[:, :2], np.float64)
        z = points[:, 2].astype(np.float64)
        self.cell = float(cell)
        self.min_xy = p.min(0) - cell
        ij = np.floor((p - self.min_xy) / cell).astype(np.int64)
        nx, ny = ij[:, 0].max() + 1, ij[:, 1].max() + 1
        flat = ij[:, 0] * ny + ij[:, 1]
        # 每 cell 的低分位数
        order = np.lexsort((z, flat))
        f_s, z_s = flat[order], z[order]
        bounds = np.searchsorted(f_s, np.arange(nx * ny + 1))
        n = bounds[-1]
        cnt = np.diff(bounds)
        k = np.maximum(1, np.ceil(np.maximum(cnt, 1) * pctl / 100.0)).astype(np.int64)
        first = bounds[:-1]
        take = np.minimum(first + k, n) - 1
        zg = np.full(nx * ny, np.nan)
        valid = cnt > 0
        # 低分位数 z: 取排序后第 k 个
        zg[valid] = z_s[np.clip(take[valid], 0, n - 1)]
        # 修补空洞: 迭代邻域均值
        Z = zg.reshape(nx, ny)
        good = np.isfinite(Z)
        for _ in range(200):
            if good.all():
                break
            Zf = np.where(good, Z, 0.0)
            gf = good.astype(np.float32)
            acc = np.zeros_like(Z)
            gcnt = np.zeros_like(Z)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    zs = np.roll(np.roll(Zf, dy, 0), dx, 1)
                    gs = np.roll(np.roll(gf, dy, 0), dx, 1)
                    acc += zs * gs
                    gcnt += gs
            fill = (~good) & (gcnt > 0)
            Z[fill] = acc[fill] / gcnt[fill]
            good |= fill
        # 轻度平滑
        for _ in range(2):
            Zp = np.pad(Z, 1, mode="edge")
            Z = (Zp[:-2, :-2] + Zp[:-2, 1:-1] + Zp[:-2, 2:] +
                 Zp[1:-1, :-2] + Zp[1:-1, 1:-1] * 2 + Zp[1:-1, 2:] +
                 Zp[2:, :-2] + Zp[2:, 1:-1] + Zp[2:, 2:]) / 10.0
        self.nx, self.ny = nx, ny
        self.grid = Z.astype(np.float32)

    def height(self, xy: np.ndarray) -> np.ndarray:
        """双线性插值地面高程; 网格外取最近值."""
        xy = np.atleast_2d(np.asarray(xy, np.float64))
        g = np.asarray(xy - self.min_xy) / self.cell - 0.5
        gx = np.clip(g[:, 0], 0, self.nx - 1.001)
        gy = np.clip(g[:, 1], 0, self.ny - 1.001)
        x0 = np.floor(gx).astype(np.int64); y0 = np.floor(gy).astype(np.int64)
        fx = gx - x0; fy = gy - y0
        x1 = np.minimum(x0 + 1, self.nx - 1); y1 = np.minimum(y0 + 1, self.ny - 1)
        Z = self.grid
        h = (Z[x0, y0] * (1 - fx) * (1 - fy) + Z[x1, y0] * fx * (1 - fy) +
             Z[x0, y1] * (1 - fx) * fy + Z[x1, y1] * fx * fy)
        return h.astype(np.float64)

    def ground_mask(self, points: np.ndarray, tol: float = 0.3) -> np.ndarray:
        g = self.height(points[:, :2])
        return (points[:, 2] - g) < tol


def estimate_normals(points: np.ndarray, voxel: float = 0.25) -> np.ndarray:
    """体素级法向估计: 体素内 PCA 求法向 (向量化批处理), 再按体素映射回所有点.

    返回的法向未定向 (使用 |cos| 入射角, 定向无关紧要).
    """
    ij = np.floor(points / voxel).astype(np.int64)
    pts = points - points.mean(0)          # 中心化, 避免大坐标下方差精度损失
    key = np.stack([ij[:, 0], ij[:, 1], ij[:, 2]], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    nv = len(uniq)
    inv64 = inv.astype(np.int64)
    n = np.bincount(inv64, minlength=nv).astype(np.float64)
    sx = np.bincount(inv64, pts[:, 0], minlength=nv)
    sy = np.bincount(inv64, pts[:, 1], minlength=nv)
    sz = np.bincount(inv64, pts[:, 2], minlength=nv)
    sxx = np.bincount(inv64, pts[:, 0] ** 2, minlength=nv)
    syy = np.bincount(inv64, pts[:, 1] ** 2, minlength=nv)
    szz = np.bincount(inv64, pts[:, 2] ** 2, minlength=nv)
    sxy = np.bincount(inv64, pts[:, 0] * pts[:, 1], minlength=nv)
    sxz = np.bincount(inv64, pts[:, 0] * pts[:, 2], minlength=nv)
    syz = np.bincount(inv64, pts[:, 1] * pts[:, 2], minlength=nv)
    cov = np.empty((nv, 3, 3))
    mean = np.stack([sx, sy, sz], 1) / n[:, None]
    s2 = np.stack([sxx, syy, szz], 1) / n[:, None]
    diag = s2 - mean ** 2
    off = np.stack([sxy / n - mean[:, 0] * mean[:, 1],
                    sxz / n - mean[:, 0] * mean[:, 2],
                    syz / n - mean[:, 1] * mean[:, 2]], 1)
    cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2] = diag[:, 0], diag[:, 1], diag[:, 2]
    cov[:, 0, 1] = cov[:, 1, 0] = off[:, 0]
    cov[:, 0, 2] = cov[:, 2, 0] = off[:, 1]
    cov[:, 1, 2] = cov[:, 2, 1] = off[:, 2]
    w, v = np.linalg.eigh(cov)          # 批量 3x3 特征分解, 最小特征向量为法向
    normals = v[:, :, 0].astype(np.float32)
    normals[~np.isfinite(normals).all(1)] = (0, 0, 1)
    return normals[inv]

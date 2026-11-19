# -*- coding: utf-8 -*-
"""微观仿真底图绘制窗口: 在真实点云上描摹 SUMO 路网并导出.

界面结构
--------
* 顶部工具栏: 导出 SUMO XML / 导入环境点云 / 底色 / 着色 / 色图 / 网格 /
  点大小 / 高度过滤 (相对路面);
* 左侧工具栏: 选择移动 / 新建道路 / 分割道路 / 删除 / 交叉口 / 形状编辑 /
  高程赋值 / 适应视图;
* 右侧属性栏: 车道数 / 车道宽 / 限速 / 优先级 / 方向 / 交叉口类型与轮廓半径;
* 中央视图: 俯视点云背景 + 路面多边形 + 车道线 + 交通方向箭头.
"""
from __future__ import annotations

import math
import os

import numpy as np

import matplotlib

from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (QAction, QActionGroup, QBrush, QColor, QFont, QImage,
                         QLinearGradient, QPainter, QPen, QPixmap, QPolygonF)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGraphicsEllipseItem, QGraphicsItem,
    QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
    QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMessageBox, QPushButton, QSpinBox, QToolBar, QVBoxLayout, QWidget,
)

from lidarsim.pointcloud import GroundModel, load_env
from lidarsim.viz import semantic_rgba
from lidarsim.roadnet import (RoadNetwork, arrow_polygons, find_netconvert,
                              offset_polyline, road_polygon)

_LANG = "zh"


def set_language(lang: str):
    global _LANG
    _LANG = "en" if lang == "en" else "zh"


def L(zh: str, en: str) -> str:
    return en if _LANG == "en" else zh


BG_KEYS = [("black", "黑底", "Black"), ("white", "白底", "White"),
           ("gray", "灰底", "Gray"), ("gradient", "渐变", "Gradient")]
COLOR_KEYS = [("rgb", "RGB", "RGB"), ("intensity", "强度 (intensity)", "Intensity"),
              ("height", "高度", "Height"),
              ("semantic", "语义类别", "Semantic class")]
CMAP_KEYS = [("gray", "灰度", "Gray"), ("magma", "深紫-珊瑚-琥珀", "Magma"),
             ("turbo", "Turbo", "Turbo"), ("viridis", "Viridis", "Viridis"),
             ("jet", "Jet", "Jet")]


# ---------------------------------------------------------------------------
# 点云俯视栅格化
# ---------------------------------------------------------------------------
def render_pointcloud_image(points, reflectance=None, rgb=None, ground=None,
                            bg: str = "black", color_by: str = "intensity",
                            cmap: str = "turbo", hmin: float = 0.0,
                            hmax: float = 8.0, point_size: int = 2,
                            max_px: int = 1800, semantics=None, stride: int = 1):
    """把三维点云栅格化为俯视 QImage.

    stride: 原始点云抽稀步长 (1=不抽稀), 用于提升底图刷新与交互流畅度.
    height 过滤的参考面是地面高程 (ground), 而不是点云最低点.
    返回 dict(image, bounds, res, kept, total) 或 None.
    """
    step = max(1, int(stride))
    points = points[::step]
    if reflectance is not None:
        reflectance = np.asarray(reflectance)[::step]
    if rgb is not None:
        rgb = np.asarray(rgb)[::step]
    if semantics is not None:
        semantics = np.asarray(semantics)[::step]
    p = np.asarray(points, np.float32)
    if len(p) == 0:
        return None
    ok = np.isfinite(p).all(1)
    p = p[ok]
    if len(p) == 0:
        return None
    xy = p[:, :2].astype(np.float64)
    z = p[:, 2].astype(np.float64)
    if ground is not None:
        g = np.asarray(ground.height(xy), np.float64).ravel()
    else:
        g = np.full(len(p), float(z.min()))
    h = z - g
    keep = (h >= float(hmin)) & (h <= float(hmax))
    if keep.sum() < 10:
        return None
    p = p[keep]; xy = xy[keep]; h = h[keep]; z = z[keep]
    refl = None if reflectance is None else np.asarray(reflectance, np.float32)[ok][keep]
    rgb_k = None if rgb is None else np.asarray(rgb, np.uint8)[ok][keep]
    sem_k = (None if semantics is None or len(semantics) != len(points)
             else np.asarray(semantics)[ok][keep])

    lo = xy.min(0); hi = xy.max(0)
    ext = np.maximum(hi - lo, 0.5)
    res = max(float(ext.max()) / float(max_px), 0.03)
    nx = max(2, int(math.ceil(ext[0] / res)) + 1)
    ny = max(2, int(math.ceil(ext[1] / res)) + 1)
    ix = np.clip(((xy[:, 0] - lo[0]) / res).astype(np.int64), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / res).astype(np.int64), 0, ny - 1)
    iy_img = ny - 1 - iy                     # 图像第 0 行对应世界 y 最大处

    img = np.zeros((ny, nx, 4), np.uint8)
    # ---- 背景 ----
    if bg == "gradient":
        ramp = np.linspace(0.16, 0.92, ny, dtype=np.float32)[:, None]
        base = np.stack([ramp * 0.75, ramp * 0.82, ramp * 0.95], 2)
        img[:, :, :3] = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    else:
        v = {"black": 8, "white": 248, "gray": 120}.get(bg, 8)
        img[:, :, :3] = v
    img[:, :, 3] = 255

    if color_by == "semantic" and sem_k is not None:
        flat = iy_img * nx + ix
        # 每个像素取最高点的类别, 避免俯视时被地面点覆盖
        order = np.argsort(z)
        sbuf = np.zeros(nx * ny, np.int64)
        present = np.zeros(nx * ny, bool)
        sbuf[flat[order]] = sem_k[order]
        present[flat] = True
        img_s = sbuf.reshape(ny, nx)
        mask = present.reshape(ny, nx)
        rgba = np.zeros((ny, nx, 4), np.float64)
        rgba[:, :, 3] = np.where(mask, 255.0, 0.0)
        rgba[:, :, :3] = semantic_rgba(img_s.ravel())[:, :3].reshape(ny, nx, 3) * 255.0
    elif color_by == "rgb" and rgb_k is not None:
        acc = np.zeros((ny, nx, 3), np.float64)
        cnt = np.zeros((ny, nx), np.float64)
        flat = iy_img * nx + ix
        for c in range(3):
            acc[:, :, c] = np.bincount(flat, rgb_k[:, c].astype(np.float64),
                                       minlength=nx * ny).reshape(ny, nx)
        cnt = np.bincount(flat, minlength=nx * ny).reshape(ny, nx)
        mask = cnt > 0
        col = np.zeros((ny, nx, 3), np.float64)
        col[mask] = acc[mask] / cnt[mask, None]
        rgba = np.dstack([col, mask.astype(np.float64) * 255.0])
    else:
        if color_by == "height":
            vals = h
        else:
            vals = (np.asarray(refl, np.float64) * 255.0 if refl is not None
                    else np.full(len(p), 140.0))
        flat = iy_img * nx + ix
        buf = np.full(nx * ny, -np.inf, np.float64)
        np.maximum.at(buf, flat, vals)
        img_v = buf.reshape(ny, nx)
        mask = np.isfinite(img_v)
        rgba = np.zeros((ny, nx, 4), np.float64)
        rgba[:, :, 3] = np.where(mask, 255.0, 0.0)
        if mask.any():
            vv = img_v[mask]
            lo_v, hi_v = np.percentile(vv, [2, 98])
            if hi_v - lo_v < 1e-9:
                hi_v = lo_v + 1.0
            norm = np.clip((img_v - lo_v) / (hi_v - lo_v), 0, 1)
            cm = matplotlib.colormaps.get_cmap(cmap)
            rgb_map = (cm(norm)[:, :, :3] * 255.0)
            rgba[:, :, :3] = rgb_map

    # ---- 点大小 (栅格膨胀) ----
    if int(point_size) > 1 and rgba[:, :, 3].any():
        try:
            from scipy.ndimage import maximum_filter
            s = int(point_size) | 1
            rgba[:, :, 3] = maximum_filter(rgba[:, :, 3], size=s)
            for c in range(3):
                rgba[:, :, c] = maximum_filter(rgba[:, :, c], size=s)
        except Exception:                                         # noqa: BLE001
            pass

    # ---- 合成到背景 ----
    a = np.clip(rgba[:, :, 3:4] / 255.0, 0, 1)
    out = (rgba[:, :, :3] * a + img[:, :, :3] * (1 - a)).astype(np.uint8)
    out = np.dstack([out, np.full((ny, nx), 255, np.uint8)])
    out = np.ascontiguousarray(out)
    image = QImage(out.data, nx, ny, 4 * nx, QImage.Format.Format_RGBA8888).copy()
    return {"image": image, "bounds": (float(lo[0]), float(lo[1]),
                                       float(hi[0]), float(hi[1])),
            "res": res, "kept": int(keep.sum()), "total": int(len(ok))}


# ---------------------------------------------------------------------------
# 图元
# ---------------------------------------------------------------------------
class NodeItem(QGraphicsEllipseItem):
    def __init__(self, node, view: "RoadNetView", selected=False):
        r = 1.6
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.node = node
        self.view = view
        self.setPos(node.x, -node.y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setData(0, ("node", node.id))
        self.setZValue(20)
        self.setToolTip(f"{node.id}  ({node.x:.1f}, {node.y:.1f}, {node.z:.2f})")
        self.setBrush(QBrush(QColor("#f7f9fc")))
        self.setPen(QPen(QColor("#1565c0"), 0))
        if selected:
            self.setSelected(True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged \
                and not self.view._suppress:
            self.view.editor.node_moved(self.node, value)
        return super().itemChange(change, value)


class VertexItem(QGraphicsEllipseItem):
    """道路中间形状点 (可拖动)."""

    def __init__(self, eid: str, idx: int, pos, view: "RoadNetView"):
        r = 1.1
        super().__init__(-r, -r, 2 * r, 2 * r)
        self.eid, self.idx, self.view = eid, idx, view
        self.setPos(pos[0], -pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setData(0, ("vertex", eid, idx))
        self.setZValue(25)
        self.setBrush(QBrush(QColor("#ffb300")))
        self.setPen(QPen(QColor("#4e2b00"), 0))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged \
                and not self.view._suppress:
            self.view.editor.vertex_moved(self.eid, self.idx, value)
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# 视图
# ---------------------------------------------------------------------------
class RoadNetView(QGraphicsView):
    cursor_info = pyqtSignal(float, float)

    def __init__(self, editor: "RoadNetEditorWindow", parent=None):
        super().__init__(parent)
        self.editor = editor
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.tool = "select"
        self._draw_pts = []
        self._draw_nodes = []
        self._suppress = False
        self._panning = False
        self._pan_start = None
        self._preview = None
        self._backdrop = None
        self._backdrop_item = None
        self._backdrop_bounds = None
        self._connect_src = None

    # ---- 坐标 ----
    @staticmethod
    def to_scene(x, y):
        return QPointF(float(x), -float(y))

    @staticmethod
    def to_world(pt: QPointF):
        return float(pt.x()), float(-pt.y())

    # ---- 背景绘制 (网格) ----
    def drawBackground(self, painter: QPainter, rect: QRectF):
        bg = self.editor.bg_key()
        if bg == "gradient":
            grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            grad.setColorAt(0.0, QColor(20, 24, 32))
            grad.setColorAt(1.0, QColor(120, 160, 210))
            painter.fillRect(rect, QBrush(grad))
        else:
            painter.fillRect(rect, QColor({"black": "#0b0d12", "white": "#fbfcfe",
                                          "gray": "#787878"}.get(bg, "#0b0d12")))
        if not self.editor.chk_grid.isChecked():
            return
        # 依据缩放选择网格间距
        scale = self.transform().m11()
        step = 50.0
        for cand in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500):
            if cand * scale >= 28:
                step = float(cand)
                break
        pen = QPen(QColor(255, 255, 255, 38), 0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        x0 = math.floor(rect.left() / step) * step
        y0 = math.floor(rect.top() / step) * step
        x = x0
        while x <= rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += step
        y = y0
        while y <= rect.bottom():
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += step
        # 坐标标注 (世界坐标)
        painter.setPen(QPen(QColor(230, 236, 245, 150), 0))
        f = QFont("Arial", 8)
        painter.setFont(f)
        x = x0
        while x <= rect.right():
            painter.drawText(QPointF(x + 3, rect.top() + 13), f"{x:.0f}")
            x += step
        y = y0
        while y <= rect.bottom():
            painter.drawText(QPointF(rect.left() + 3, y - 4), f"{-y:.0f}")
            y += step

    # ---- 交互 ----
    def set_tool(self, tool: str):
        self.tool = tool
        self._cancel_draw()
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag
                         if tool == "select" else QGraphicsView.DragMode.NoDrag)

    def _cancel_draw(self):
        self._draw_pts = []
        self._draw_nodes = []
        if self._preview is not None:
            self._scene.removeItem(self._preview)
            self._preview = None

    def _pick(self, pos):
        for it in self._scene.items(pos):
            d = it.data(0)
            if d:
                return d, it
        return None, None

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        pos = ev.position().toPoint()
        # 自动连接: 已选定源道路, 左键点击目标道路完成连接
        if ev.button() == Qt.MouseButton.LeftButton and self._connect_src:
            d, _ = self._pick(self.mapToScene(pos))
            if d and d[0] == "edge" and d[1] != self._connect_src:
                self._do_connect(self._connect_src, d[1])
                return
        if self.tool == "road" and ev.button() == Qt.MouseButton.LeftButton:
            self._add_draw_point(self.mapToScene(pos))
            return
        if ev.button() == Qt.MouseButton.RightButton:
            if self.tool == "road":
                self._finish_road()
            else:
                super().mousePressEvent(ev)
            return
        if ev.button() == Qt.MouseButton.LeftButton and self.tool == "select" \
                and self.editor._shape_eid:
            d, _ = self._pick(self.mapToScene(pos))
            if d and d[0] == "edge" and d[1] == self.editor._shape_eid:
                x, y = self.to_world(self.mapToScene(pos))
                idx = self.editor.net.insert_vertex(
                    d[1], x, y, self.editor.ground_z(x, y))
                self.editor.rebuild_network()
                self.editor.show_vertices(d[1])
                self.editor.log(L(f"已插入形状点 #{idx}, 可继续拖动调整曲线",
                                  f"Vertex #{idx} inserted; drag to shape the curve"))
                return
        if ev.button() == Qt.MouseButton.LeftButton and self.tool in ("split", "delete", "junction"):
            d, it = self._pick(self.mapToScene(ev.position().toPoint()))
            if self.tool == "split" and d and d[0] == "edge":
                x, y = self.to_world(self.mapToScene(ev.position().toPoint()))
                z = self.editor.ground_z(x, y)
                self.editor.net.split_edge_at_point(d[1], x, y, z)
                self.editor.rebuild_network()
                self.editor.log(L("道路已分割", "Edge split"))
                return
            if self.tool == "delete" and d:
                if d[0] == "edge":
                    self.editor.net.remove_edge(d[1])
                elif d[0] == "node":
                    self.editor.net.remove_node(d[1])
                self.editor.rebuild_network()
                self.editor.log(L("已删除选中对象", "Selected object deleted"))
                return
            if self.tool == "junction":
                if d and d[0] == "node":
                    self._scene.clearSelection()
                    self.editor.select_node(d[1])
                    return
                x, y = self.to_world(self.mapToScene(ev.position().toPoint()))
                hit = self.editor.net.find_crossing_near(x, y, tol=8.0)
                if hit is not None:
                    e1, e2, q = hit
                    nid = self.editor.net.create_intersection(
                        e1, e2, float(q[0]), float(q[1]),
                        self.editor.ground_z(float(q[0]), float(q[1])),
                        kind=self.editor.cb_jtype.currentText())
                    self.editor.rebuild_network()
                    self.editor.select_node(nid)
                    self.editor.log(L(f"已创建交叉口 {nid}: {e1.id} x {e2.id}",
                                      f"Junction {nid} created: {e1.id} x {e2.id}"))
                    return
                self.editor.log(L("未找到交叉的两条道路 (点击交叉点附近)",
                                  "No crossing edges found near click"))
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        scene_pt = self.mapToScene(ev.position().toPoint())
        wx, wy = self.to_world(scene_pt)
        self.cursor_info.emit(wx, wy)
        if self._panning and self._pan_start is not None:
            d = ev.position() - self._pan_start
            self._pan_start = ev.position()
            s = max(abs(self.transform().m11()), 1e-6)
            # 只平移视图变换, 不修改任何世界坐标
            self.translate(d.x() / s, d.y() / s)
            return
        if self.tool == "road" and self._draw_pts:
            self._update_preview(scene_pt)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        if self.tool == "road":
            self._finish_road()
            return
        d, it = self._pick(self.mapToScene(ev.position().toPoint()))
        if d and d[0] == "edge":
            self.editor.show_vertices(d[1])
            return
        super().mouseDoubleClickEvent(ev)

    def contextMenuEvent(self, ev):
        """右键集成: 形状编辑 / 插入节点 / 分割 / 平滑 / 自动连接 / 删除."""
        if self.tool == "road":
            return
        scene_pt = self.mapToScene(ev.pos())
        d, _ = self._pick(scene_pt)
        wx, wy = self.to_world(scene_pt)
        z = self.editor.ground_z(wx, wy)
        menu = QMenu(self)
        if d and d[0] == "edge":
            eid = d[1]
            a = menu.addAction(L("形状编辑 (可拖动/点击加点)", "Edit shape (drag / click to add)"))
            a.triggered.connect(lambda _=False, e=eid: self.editor.show_vertices(e))
            a = menu.addAction(L("在点击处插入形状点", "Insert vertex here"))
            a.triggered.connect(lambda _=False, e=eid, x=wx, y=wy, zz=z:
                                self._insert_vertex(e, x, y, zz))
            a = menu.addAction(L("在此处分割路段", "Split edge here"))
            a.triggered.connect(lambda _=False, e=eid, x=wx, y=wy, zz=z:
                                self._split_edge(e, x, y, zz))
            a = menu.addAction(L("平滑为曲线", "Smooth into curve"))
            a.triggered.connect(lambda _=False, e=eid: self._smooth_edge(e))
            a = menu.addAction(L("从此道路自动连接...", "Auto-connect from this edge..."))
            a.triggered.connect(lambda _=False, e=eid: self._start_connect(e))
            menu.addSeparator()
            a = menu.addAction(L("删除该路段", "Delete edge"))
            a.triggered.connect(lambda _=False, e=eid: self._delete_edge(e))
        elif d and d[0] == "vertex":
            eid, idx = d[1], d[2]
            a = menu.addAction(L("删除该形状点", "Delete vertex"))
            a.triggered.connect(lambda _=False, e=eid, i=idx: self.editor.delete_vertex(e, i))
            a = menu.addAction(L("平滑为曲线", "Smooth into curve"))
            a.triggered.connect(lambda _=False, e=eid: self._smooth_edge(e))
        elif d and d[0] == "node":
            nid = d[1]
            a = menu.addAction(L("编辑交叉口", "Edit junction"))
            a.triggered.connect(lambda _=False, n=nid: self.editor.select_node(n))
            menu.addSeparator()
            a = menu.addAction(L("删除节点及关联路段", "Delete node and incident edges"))
            a.triggered.connect(lambda _=False, n=nid: self._delete_node(n))
        else:
            a = menu.addAction(L("结束形状编辑", "Finish shape editing"))
            a.triggered.connect(self.editor.clear_shape_edit)
            a = menu.addAction(L("取消自动连接", "Cancel auto-connect"))
            a.triggered.connect(self._cancel_connect)
            a = menu.addAction(L("全局视图", "Fit all"))
            a.triggered.connect(lambda: self.editor.view.fit_all())
        menu.exec(ev.globalPos())

    # ---- 右键菜单动作 ----
    def _insert_vertex(self, eid, x, y, z):
        idx = self.editor.net.insert_vertex(eid, x, y, z)
        self.editor.rebuild_network()
        self.editor.show_vertices(eid)
        self.editor.log(L(f"已插入形状点 #{idx}", f"Vertex #{idx} inserted"))

    def _split_edge(self, eid, x, y, z):
        self.editor.net.split_edge_at_point(eid, x, y, z)
        self.editor.rebuild_network()
        self.editor.log(L("道路已分割", "Edge split"))

    def _smooth_edge(self, eid):
        n = self.editor.net.smooth_edge(eid, iterations=2)
        self.editor.rebuild_network()
        self.editor.show_vertices(eid)
        self.editor.log(L(f"已平滑为曲线 ({n} 个形状点)",
                          f"Smoothed into a curve ({n} vertices)"))

    def _delete_edge(self, eid):
        self.editor.net.remove_edge(eid)
        self.editor.rebuild_network()
        self.editor.log(L("已删除路段", "Edge deleted"))

    def _delete_node(self, nid):
        self.editor.net.remove_node(nid)
        self.editor.rebuild_network()
        self.editor.log(L("已删除节点", "Node deleted"))

    def _start_connect(self, eid):
        self._connect_src = eid
        self.editor.status_hint(
            L(f"自动连接: 已选择 {eid}, 左键点击目标道路",
              f"Auto-connect: {eid} selected, left-click the target edge"))

    def _cancel_connect(self):
        self._connect_src = None
        self.editor.status_hint(L("已取消自动连接", "Auto-connect cancelled"))

    def _do_connect(self, src_eid, dst_eid):
        e1 = self.editor.net.edges.get(src_eid)
        e2 = self.editor.net.edges.get(dst_eid)
        self._connect_src = None
        if e1 is None or e2 is None:
            return
        z_fn = self.editor.ground.height if self.editor.ground is not None else None
        ids, n0, n1 = self.editor.net.connect_edges(e1, e2, z_fn=z_fn)
        if not ids:
            self.editor.log(L("未找到可连接的端点 (可能已连接或端点重合)",
                              "No connectable endpoints (already connected or coincident)"))
            return
        self.editor.rebuild_network(select_edges=ids)
        self.editor.log(L(f"已自动连接 {src_eid} -> {dst_eid}, 生成 {len(ids)} 段(宽度渐变)",
                          f"Auto-connected {src_eid} -> {dst_eid} with {len(ids)} segment(s)"))

    def wheelEvent(self, ev):
        k = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(k, k)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.tool == "road":
                self._finish_road()
                return
        if ev.key() == Qt.Key.Key_Escape:
            if self.tool == "road":
                self._cancel_draw()
                self.editor.status_hint(L("已取消绘制", "Drawing cancelled"))
                return
        if ev.key() == Qt.Key.Key_Delete:
            sel = [it.data(0) for it in self._scene.selectedItems() if it.data(0)]
            for item in sel:
                if item[0] == "edge":
                    self.editor.net.remove_edge(item[1])
                elif item[0] == "node":
                    self.editor.net.remove_node(item[1])
            if sel:
                self.editor.rebuild_network()
                return
        super().keyPressEvent(ev)

    # ---- 画路 ----
    def _add_draw_point(self, scene_pt: QPointF):
        wx, wy = self.to_world(scene_pt)
        node = self.editor.net.nearest_node(wx, wy, 2.5)
        if node is not None:
            pt = (node.x, node.y, node.z)
            nid = node.id
        else:
            z = self.editor.ground_z(wx, wy)
            pt = (wx, wy, z)
            nid = None
        self._draw_pts.append(pt)
        self._draw_nodes.append(nid)
        self.editor.status_hint(
            L(f"绘制中: {len(self._draw_pts)} 个点 (右键/双击完成, Esc 取消)",
              f"Drawing: {len(self._draw_pts)} points (right-click/double-click to finish, Esc to cancel)"))
        self._update_preview(scene_pt)

    def _update_preview(self, scene_pt: QPointF):
        from PyQt6.QtGui import QPainterPath
        path = QPainterPath()
        first = self._draw_pts[0]
        path.moveTo(first[0], -first[1])
        for p in self._draw_pts[1:]:
            path.lineTo(p[0], -p[1])
        wx, wy = self.to_world(scene_pt)
        path.lineTo(wx, -wy)
        if self._preview is None:
            self._preview = QGraphicsPathItem()
            pen = QPen(QColor("#ffb300"), 0)
            pen.setCosmetic(True)
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._preview.setPen(pen)
            self._preview.setZValue(30)
            self._scene.addItem(self._preview)
        self._preview.setPath(path)

    def _finish_road(self):
        if len(self._draw_pts) < 2:
            self._cancel_draw()
            return
        pts = np.asarray(self._draw_pts, np.float64)
        start_id = self._draw_nodes[0]
        end_id = self._draw_nodes[-1]
        try:
            e = self.editor.add_road(pts, start_id, end_id)
        except Exception as ex:                                   # noqa: BLE001
            QMessageBox.warning(self.editor, L("绘制失败", "Draw failed"), str(ex))
            self._cancel_draw()
            return
        self._cancel_draw()
        self.editor.rebuild_network(select_edges=[e.id])
        # 创建后自动进入选择/形状编辑状态, 便于立即拖动节点与插入形状点
        self.editor._set_tool("select")
        self.editor.show_vertices(e.id)
        self.editor.log(L(f"新建道路 {e.id} ({e.lanes} 车道, 宽 {e.width:.1f} m); "
                          f"可拖动节点或点击道路插入形状点",
                          f"New edge {e.id} ({e.lanes} lanes, width {e.width:.1f} m); "
                          f"drag nodes or click the edge to add vertices"))

    # ---- 背景 ----
    def set_backdrop(self, result):
        if self._backdrop_item is not None:
            self._scene.removeItem(self._backdrop_item)
            self._backdrop_item = None
        self._backdrop = result
        if result is None:
            return
        pm = QPixmap.fromImage(result["image"])
        item = QGraphicsPixmapItem(pm)
        x0, _, _, y1 = result["bounds"]
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setPos(x0, -y1)
        item.setScale(result["res"])
        item.setZValue(-100)
        item.setData(0, None)
        self._scene.addItem(item)
        self._backdrop_item = item
        self._backdrop_bounds = result["bounds"]

    def reset_scene_rect(self):
        r = QRectF(-500, -500, 1000, 1000)
        if self._backdrop_bounds:
            x0, y0, x1, y1 = self._backdrop_bounds
            r = r.united(QRectF(x0, -y1, max(x1 - x0, 1), max(y1 - y0, 1)))
        if self.editor.net.nodes:
            lo, hi = self.editor.net.bounds()
            r = r.united(QRectF(lo[0], -hi[1], max(hi[0] - lo[0], 1),
                                max(hi[1] - lo[1], 1)))
        self._scene.setSceneRect(r)
        return r

    def fit_all(self):
        r = self.reset_scene_rect()
        self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
        self.scale(0.9, 0.9)

    def fit_backdrop(self):
        """一键将点云底图铺满视图 (仅按点云范围取景, 保持纵横比)."""
        if not self._backdrop_bounds:
            self.fit_all()
            return
        x0, y0, x1, y1 = self._backdrop_bounds
        r = QRectF(float(x0), float(-y1),
                   max(float(x1 - x0), 1.0), max(float(y1 - y0), 1.0))
        self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio)
        self.scale(0.995, 0.995)


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class RoadNetEditorWindow(QMainWindow):
    def __init__(self, env=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(L("微观仿真底图绘制 - SUMO 路网", "Microscopic Base Map Editor - SUMO Network"))
        self.resize(1280, 860)
        self.net = RoadNetwork()
        self.env = None
        self.ground = None
        self._net_items = []
        self._edge_groups = {}
        self._junction_groups = {}
        self._edge_items = {}
        self._vertex_items = []
        self._selected_edge = None
        self._shape_eid = None

        self.view = RoadNetView(self)
        self.setCentralWidget(self.view)
        self.view.cursor_info.connect(self._on_cursor)

        self._build_top_toolbar()
        self._build_left_toolbar()
        self._build_attr_dock()
        self.statusBar().showMessage(L("就绪: 导入点云后沿道路中心线点击绘制", "Ready: import a cloud then click along road centerlines"))
        self.view.scene().selectionChanged.connect(self._on_selection)
        if env is not None:
            self.set_env(env)

    # ---- 顶部工具栏 ----
    def _build_top_toolbar(self):
        # 第一行: 文件与关键视图操作 (确保"点云铺满"始终可见)
        tb = QToolBar(L("底图文件", "Base map file"), self)
        tb.setObjectName("roadnet_toolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        act = QAction(L("导出 SUMO XML...", "Export SUMO XML..."), self)
        act.triggered.connect(self.export_sumo)
        tb.addAction(act)
        act = QAction(L("导入环境点云...", "Import env cloud..."), self)
        act.triggered.connect(self.pick_env)
        tb.addAction(act)
        act = QAction(L("载入 net.xml...", "Load net.xml..."), self)
        act.triggered.connect(self.load_net_xml)
        tb.addAction(act)
        tb.addSeparator()
        act = QAction(L("点云铺满视图", "Fit cloud view"), self)
        act.setToolTip(L("仅按点云范围取景, 一键铺满绘图窗口 (Ctrl+Shift+F)",
                         "Fit the point-cloud extent to fill the view (Ctrl+Shift+F)"))
        act.setShortcut("Ctrl+Shift+F")
        act.triggered.connect(lambda: self.view.fit_backdrop())
        tb.addAction(act)
        act = QAction(L("全局视图", "Fit all"), self)
        act.triggered.connect(lambda: self.view.fit_all())
        tb.addAction(act)

        # 第二行: 显示选项
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        tb = QToolBar(L("底图显示", "Base map display"), self)
        tb.setObjectName("roadnet_display")
        tb.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        tb.addWidget(QLabel(L("底色", "BG")))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems([L(zh, en) for _, zh, en in BG_KEYS])
        self.bg_combo.currentIndexChanged.connect(self._refresh_backdrop)
        tb.addWidget(self.bg_combo)
        tb.addWidget(QLabel(L("着色", "Color")))
        self.color_combo = QComboBox()
        self.color_combo.addItems([L(zh, en) for _, zh, en in COLOR_KEYS])
        self.color_combo.currentIndexChanged.connect(self._refresh_backdrop)
        tb.addWidget(self.color_combo)
        tb.addWidget(QLabel(L("色图", "Map")))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([L(zh, en) for _, zh, en in CMAP_KEYS])
        self.cmap_combo.setCurrentIndex(2)
        self.cmap_combo.currentIndexChanged.connect(self._refresh_backdrop)
        tb.addWidget(self.cmap_combo)
        self.chk_grid = QCheckBox(L("网格", "Grid"))
        self.chk_grid.setChecked(True)
        self.chk_grid.stateChanged.connect(lambda _=0: self.view.viewport().update())
        tb.addWidget(self.chk_grid)
        tb.addWidget(QLabel(L("点大小", "Point size")))
        self.point_size = QComboBox()
        self.point_size.addItems(["1", "2", "3", "4", "6", "8"])
        self.point_size.setCurrentText("2")
        self.point_size.currentIndexChanged.connect(self._refresh_backdrop)
        tb.addWidget(self.point_size)
        tb.addWidget(QLabel(L("抽稀", "Decimate")))
        self.stride_combo = QComboBox()
        self.stride_combo.addItems(["1/1", "1/2", "1/4", "1/8", "1/16"])
        self.stride_combo.setCurrentIndex(2)
        self.stride_combo.setToolTip(L("点云抽稀步长, 越大绘制越流畅",
                                       "Point stride; larger = faster interaction"))
        self.stride_combo.currentIndexChanged.connect(self._refresh_backdrop)
        tb.addWidget(self.stride_combo)
        tb.addWidget(QLabel(L("离地高度", "Height above ground")))
        self.h_min = QDoubleSpinBox()
        self.h_min.setRange(-5.0, 100.0); self.h_min.setDecimals(2)
        self.h_min.setSingleStep(0.5); self.h_min.setValue(0.0)
        self.h_min.valueChanged.connect(self._refresh_backdrop)
        tb.addWidget(self.h_min)
        tb.addWidget(QLabel("~"))
        self.h_max = QDoubleSpinBox()
        self.h_max.setRange(-5.0, 100.0); self.h_max.setDecimals(2)
        self.h_max.setSingleStep(0.5); self.h_max.setValue(8.0)
        self.h_max.valueChanged.connect(self._refresh_backdrop)
        tb.addWidget(self.h_max)
        self.chk_arrows = QCheckBox(L("方向箭头", "Arrows"))
        self.chk_arrows.setChecked(True)
        self.chk_arrows.stateChanged.connect(lambda _=0: self.rebuild_network())
        tb.addWidget(self.chk_arrows)
        act = QAction(L("刷新底图", "Refresh"), self)
        act.triggered.connect(self._refresh_backdrop)
        tb.addAction(act)

    # ---- 左侧工具栏 ----
    def _build_left_toolbar(self):
        tb = QToolBar(L("绘制工具", "Tools"), self)
        tb.setObjectName("roadnet_tools")
        tb.setMovable(False)
        tb.setOrientation(Qt.Orientation.Vertical)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb)
        grp = QActionGroup(self)
        grp.setExclusive(True)
        self._tool_acts = {}
        tools = [
            ("select", "选择/移动", "Select / Move"),
            ("road", "新建道路", "New road"),
            ("split", "分割道路", "Split edge"),
            ("delete", "删除对象", "Delete"),
            ("junction", "交叉口", "Junction"),
        ]
        for key, zh, en in tools:
            a = QAction(L(zh, en), self)
            a.setCheckable(True)
            if key == "select":
                a.setChecked(True)
            a.triggered.connect(lambda _=False, k=key: self._set_tool(k))
            grp.addAction(a)
            tb.addAction(a)
            self._tool_acts[key] = a
        tb.addSeparator()
        a = QAction(L("形状编辑", "Edit shape"), self)
        a.setToolTip(L("选中道路后显示并拖动中间形状点; 双击道路亦可",
                       "Show draggable vertices for the selected edge; double-click an edge too"))
        a.triggered.connect(self._edit_selected_shape)
        tb.addAction(a)
        a = QAction(L("根据点云赋高程", "Elevation from cloud"), self)
        a.triggered.connect(self.assign_elevation)
        tb.addAction(a)
        a = QAction(L("适应视图", "Fit view"), self)
        a.triggered.connect(lambda: self.view.fit_all())
        tb.addAction(a)
        a = QAction(L("清空路网", "Clear network"), self)
        a.triggered.connect(self.clear_network)
        tb.addAction(a)

    # ---- 属性栏 ----
    def _build_attr_dock(self):
        dock = QDockWidget(L("道路属性", "Road attributes"), self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea |
                             Qt.DockWidgetArea.RightDockWidgetArea)
        w = QWidget(); lay = QVBoxLayout(w)
        fl = QFormLayout()
        self.sp_lanes = QSpinBox(); self.sp_lanes.setRange(1, 8); self.sp_lanes.setValue(2)
        self.sp_lanes.setToolTip(L("双向道路填双向合计车道数, 导出时自动拆分",
                                   "For bidirectional roads enter the total lane count; "
                                   "it is split on export"))
        fl.addRow(L("车道数", "Lanes"), self.sp_lanes)
        self.sp_lw = QDoubleSpinBox(); self.sp_lw.setRange(1.5, 6.0)
        self.sp_lw.setDecimals(2); self.sp_lw.setSingleStep(0.1); self.sp_lw.setValue(3.2)
        fl.addRow(L("车道宽 (m)", "Lane width (m)"), self.sp_lw)
        self.sp_speed = QDoubleSpinBox(); self.sp_speed.setRange(5.0, 160.0)
        self.sp_speed.setDecimals(1); self.sp_speed.setSingleStep(5.0); self.sp_speed.setValue(50.0)
        fl.addRow(L("限速 (km/h)", "Speed limit (km/h)"), self.sp_speed)
        self.sp_priority = QSpinBox(); self.sp_priority.setRange(1, 20); self.sp_priority.setValue(1)
        fl.addRow(L("优先级", "Priority"), self.sp_priority)
        self.cb_dir = QComboBox()
        self.cb_dir.addItems([L("正向", "Forward"), L("反向", "Reverse"), L("双向", "Bidirectional")])
        fl.addRow(L("交通方向", "Direction"), self.cb_dir)
        self.cb_jtype = QComboBox()
        self.cb_jtype.addItems(["priority", "traffic_light", "right_before_left", "unregulated"])
        fl.addRow(L("交叉口类型", "Junction type"), self.cb_jtype)
        self.sp_jradius = QDoubleSpinBox(); self.sp_jradius.setRange(0.0, 40.0)
        self.sp_jradius.setDecimals(1); self.sp_jradius.setSingleStep(0.5); self.sp_jradius.setValue(0.0)
        fl.addRow(L("交叉口轮廓半径 (0=自动)", "Junction radius (0=auto)"), self.sp_jradius)
        lay.addLayout(fl)

        self.lbl_info = QLabel(L("未选中对象", "Nothing selected"))
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#1565c0;")
        lay.addWidget(self.lbl_info)
        row = QHBoxLayout()
        self.btn_apply = QPushButton(L("应用到选中", "Apply to selection"))
        self.btn_apply.clicked.connect(self.apply_attrs)
        row.addWidget(self.btn_apply)
        lay.addLayout(row)
        self.lbl_stats = QLabel("")
        lay.addWidget(self.lbl_stats)
        lay.addStretch(1)
        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.dock_attrs = dock

    # ---- 语言 ----
    def set_language(self, lang: str):
        set_language(lang)

    # ---- 环境点云 ----
    def set_env(self, env):
        self.env = env
        self.ground = None
        if env is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()
            try:
                # 地面模型也适当抽稀, 避免百万级点云阻塞窗口
                gstep = max(1, len(env.points) // 300000)
                self.ground = GroundModel(env.points[::gstep], cell=1.0, pctl=5.0)
                self.statusBar().showMessage(
                    L(f"环境点云: {len(env.points)} 点, 地面模型已建立 (抽样 1/{gstep})",
                      f"Env cloud: {len(env.points)} pts, ground model ready (1/{gstep})"))
            except Exception as e:                                # noqa: BLE001
                self.log(L(f"地面模型构建失败: {e}", f"Ground model failed: {e}"))
            finally:
                QApplication.restoreOverrideCursor()
        self._refresh_backdrop(fit=True)

    def stride(self) -> int:
        """当前点云抽稀步长 (1,2,4,8,16)."""
        return 1 << max(0, self.stride_combo.currentIndex())

    def pick_env(self):
        f, _ = QFileDialog.getOpenFileName(self, L("选择环境点云", "Select env cloud"),
                                           "", "Point Cloud (*.ply *.h5)")
        if not f:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            env = load_env(f)
        except Exception as e:                                    # noqa: BLE001
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, L("导入失败", "Import failed"), str(e))
            return
        QApplication.restoreOverrideCursor()
        self.set_env(env)
        self.log(L(f"导入点云: {os.path.basename(f)} ({len(env.points)} 点)",
                   f"Cloud imported: {os.path.basename(f)} ({len(env.points)} pts)"))

    def ground_z(self, x: float, y: float) -> float:
        if self.ground is None:
            return 0.0
        try:
            return float(self.ground.height(np.array([[x, y]], np.float64))[0])
        except Exception:                                         # noqa: BLE001
            return 0.0

    def bg_key(self) -> str:
        return BG_KEYS[self.bg_combo.currentIndex()][0]

    def color_key(self) -> str:
        return COLOR_KEYS[self.color_combo.currentIndex()][0]

    def cmap_key(self) -> str:
        return CMAP_KEYS[self.cmap_combo.currentIndex()][0]

    def _refresh_backdrop(self, *_, fit: bool = False):
        if self.env is None:
            self.log(L("请先导入环境点云", "Import an environment cloud first"))
            return
        if self.color_key() == "semantic":
            sem = getattr(self.env, "semantics", None)
            if sem is None or len(sem) != len(self.env.points):
                self.log(L("当前点云无语义类别属性, 该着色选项不生效",
                           "Point cloud has no semantic labels; option has no effect"))
                return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            res = render_pointcloud_image(
                self.env.points, self.env.reflectance, self.env.rgb,
                self.ground, bg=self.bg_key(), color_by=self.color_key(),
                cmap=self.cmap_key(), hmin=self.h_min.value(),
                hmax=self.h_max.value(), point_size=int(self.point_size.currentText()),
                semantics=getattr(self.env, "semantics", None),
                stride=self.stride())
        except Exception as e:                                    # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self.log(L(f"底图生成失败: {e}", f"Backdrop failed: {e}"))
            return
        QApplication.restoreOverrideCursor()
        self.view.set_backdrop(res)
        if res is not None:
            self.log(L(f"底图: 保留 {res['kept']}/{res['total']} 点, 分辨率 {res['res']:.2f} m/px",
                       f"Backdrop: {res['kept']}/{res['total']} pts, {res['res']:.2f} m/px"))
        if fit:
            self.view.fit_all()
        self.view.viewport().update()

    # ---- 路网操作 ----
    def add_road(self, pts: np.ndarray, start_id: str, end_id: str):
        return self.net.add_edge(
            pts, start_id, end_id,
            lanes=self.sp_lanes.value(), lane_width=self.sp_lw.value(),
            speed=self.sp_speed.value() / 3.6,
            priority=self.sp_priority.value(),
            bidir=(self.cb_dir.currentIndex() == 2))

    def clear_network(self):
        if self.net.edges and QMessageBox.question(
                self, L("确认", "Confirm"),
                L("清空全部路网?", "Clear the whole road network?")) \
                != QMessageBox.StandardButton.Yes:
            return
        self.net.clear()
        self.rebuild_network()
        self.log(L("路网已清空", "Network cleared"))

    def rebuild_network(self, select_edges=None):
        self._suppress = True
        keep_sel = [it.data(0) for it in self.view.scene().selectedItems() if it.data(0)]
        for it in list(self._net_items) + list(self._vertex_items):
            self.view.scene().removeItem(it)
        self._net_items = []
        self._vertex_items = []
        self._edge_groups = {}
        self._junction_groups = {}
        self._edge_items = {}
        sc = self.view.scene()

        # 交叉口轮廓 (先画, 位于路面之下)
        for nid in self.net.nodes:
            outline = self.net.junction_outline(nid)
            if len(outline) >= 3:
                poly = QPolygonF([QPointF(x, -y) for x, y in outline])
                item = QGraphicsPolygonItem(poly)
                item.setBrush(QBrush(QColor(70, 78, 92, 220)))
                item.setPen(QPen(QColor(180, 190, 205, 200), 0))
                item.setZValue(2)
                item.setData(0, ("junction", nid))
                sc.addItem(item)
                self._junction_groups[nid] = [item]
                self._net_items.append(item)

        # 道路
        for eid, e in self.net.edges.items():
            self._build_edge_group(eid, e)

        # 节点 (最上层)
        for nid, n in self.net.nodes.items():
            item = NodeItem(n, self.view)
            sc.addItem(item)
            self._net_items.append(item)

        self._suppress = False
        if select_edges:
            for eid in select_edges:
                if eid in self._edge_items:
                    self._edge_items[eid].setSelected(True)
        elif keep_sel:
            for kind, *rest in keep_sel:
                if kind == "edge" and rest and rest[0] in self._edge_items:
                    self._edge_items[rest[0]].setSelected(True)
        self._update_stats()
        self.view.reset_scene_rect()
        if self._shape_eid in self.net.edges:
            self.show_vertices(self._shape_eid)

    def _build_edge_group(self, eid, e):
        sc = self.view.scene()
        items = []
        p = e.xy()
        if len(p) < 2:
            return
        poly = road_polygon(p, e.width)
        if len(poly) == 0:
            return
        qpoly = QPolygonF([QPointF(x, -y) for x, y in poly])
        item = QGraphicsPolygonItem(qpoly)
        color = QColor(190, 198, 210, 165)
        if self._selected_edge == eid:
            color = QColor(255, 196, 90, 210)
        item.setBrush(QBrush(color))
        pen = QPen(QColor(245, 248, 255, 220), 0)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(4)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setData(0, ("edge", eid))
        item.setToolTip(f"{eid}: {e.from_id}->{e.to_id}, {e.lanes} lanes, "
                        f"W={e.width:.1f}m, L={e.total_length():.1f}m")
        sc.addItem(item)
        items.append(item)
        self._edge_items[eid] = item

        # 车道分隔线
        h = e.width / 2
        if e.lanes > 1:
            for i in range(1, int(e.lanes)):
                off = -h + i * e.lane_width
                line = offset_polyline(p, off)
                for k in range(len(line) - 1):
                    from PyQt6.QtWidgets import QGraphicsLineItem
                    li = QGraphicsLineItem(line[k][0], -line[k][1],
                                           line[k + 1][0], -line[k + 1][1])
                    lp = QPen(QColor(240, 240, 240, 170), 0, Qt.PenStyle.DashLine)
                    lp.setCosmetic(True)
                    li.setPen(lp)
                    li.setZValue(5)
                    sc.addItem(li)
                    items.append(li)
        if e.bidir:
            from PyQt6.QtWidgets import QGraphicsLineItem
            center = offset_polyline(p, 0.0)
            for k in range(len(center) - 1):
                li = QGraphicsLineItem(center[k][0], -center[k][1],
                                       center[k + 1][0], -center[k + 1][1])
                lp = QPen(QColor(250, 205, 60, 210), 0)
                lp.setCosmetic(True)
                li.setPen(lp)
                li.setZValue(5)
                sc.addItem(li)
                items.append(li)

        # 方向箭头
        if self.chk_arrows.isChecked():
            arrow_color = QColor(255, 214, 64, 235)
            for fwd, off in ((False, -e.width * 0.22), (True, e.width * 0.22)):
                if fwd and not e.bidir:
                    continue
                for tri in arrow_polygons(p, e.width, size=max(2.5, e.width * 0.42),
                                          reverse=fwd, offset=off):
                    a = QGraphicsPolygonItem(
                        QPolygonF([QPointF(x, -y) for x, y in tri]))
                    a.setBrush(QBrush(arrow_color))
                    a.setPen(QPen(QColor(90, 70, 0, 200), 0))
                    a.setZValue(7)
                    a.setData(0, ("edge", eid))
                    sc.addItem(a)
                    items.append(a)
        self._edge_groups[eid] = items
        self._net_items.extend(items)

    def _refresh_edge(self, eid):
        old = self._edge_groups.get(eid, [])
        for it in old:
            self.view.scene().removeItem(it)
            if it in self._net_items:
                self._net_items.remove(it)
        self._build_edge_group(eid, self.net.edges[eid])
        # 交叉口轮廓依赖相邻道路方向
        for nid in {self.net.edges[eid].from_id, self.net.edges[eid].to_id}:
            for it in self._junction_groups.get(nid, []):
                self.view.scene().removeItem(it)
                if it in self._net_items:
                    self._net_items.remove(it)
            self._junction_groups.pop(nid, None)
            outline = self.net.junction_outline(nid)
            if len(outline) >= 3:
                item = QGraphicsPolygonItem(
                    QPolygonF([QPointF(x, -y) for x, y in outline]))
                item.setBrush(QBrush(QColor(70, 78, 92, 220)))
                item.setPen(QPen(QColor(180, 190, 205, 200), 0))
                item.setZValue(2)
                item.setData(0, ("junction", nid))
                self.view.scene().addItem(item)
                self._junction_groups[nid] = [item]
                self._net_items.append(item)

    def node_moved(self, node, scene_pos: QPointF):
        try:
            node.x = float(scene_pos.x())
            node.y = float(-scene_pos.y())
            if self.ground is not None:
                node.z = self.ground_z(node.x, node.y)
            for e in self.net.incident_edges(node.id):
                self.net.sync_endpoints(e)
                self._refresh_edge(e.id)
            for item in self.view.scene().items():
                if item.data(0) == ("node", node.id) and hasattr(item, "setToolTip"):
                    item.setToolTip(f"{node.id}  ({node.x:.1f}, {node.y:.1f}, {node.z:.2f})")
            # 交叉口轮廓由 _refresh_edge 按受影响节点刷新
            self._update_stats()
        except Exception as e:                                    # noqa: BLE001
            self.log(L(f"节点更新失败: {e}", f"Node update failed: {e}"))

    def vertex_moved(self, eid: str, idx: int, scene_pos: QPointF):
        e = self.net.edges.get(eid)
        if e is None or not (0 < idx < len(e.shape) - 1):
            return
        x, y = float(scene_pos.x()), float(-scene_pos.y())
        z = self.ground_z(x, y) if self.ground is not None else e.shape[idx][2]
        e.shape[idx] = (x, y, z)
        self._refresh_edge(eid)
        self._update_stats()

    def show_vertices(self, eid: str):
        """进入形状编辑: 显示中间形状点; 点击道路即可继续插入节点."""
        for it in self._vertex_items:
            self.view.scene().removeItem(it)
        self._vertex_items = []
        e = self.net.edges.get(eid)
        if e is None:
            self._shape_eid = None
            return
        self._shape_eid = eid
        self._selected_edge = eid
        for i in range(1, len(e.shape) - 1):
            it = VertexItem(eid, i, e.shape[i][:2], self.view)
            self.view.scene().addItem(it)
            self._vertex_items.append(it)
        if len(e.shape) <= 2:
            self.log(L(f"{eid}: 形状编辑中 — 拖动白色节点调整端点, 点击道路插入形状点",
                       f"{eid}: shape editing — drag white nodes, click the edge to add vertices"))
        else:
            self.log(L(f"{eid}: 拖动橙色点编辑形状, 点击道路继续加点, 右键可删除/平滑",
                       f"{eid}: drag orange vertices, click to add, right-click to delete/smooth"))

    def clear_shape_edit(self):
        for it in self._vertex_items:
            self.view.scene().removeItem(it)
        self._vertex_items = []
        self._shape_eid = None
        self.status_hint(L("已结束形状编辑", "Shape editing finished"))

    def delete_vertex(self, eid: str, idx: int):
        e = self.net.edges.get(eid)
        if e is None or not (0 < idx < len(e.shape) - 1):
            return
        e.shape.pop(idx)
        self._refresh_edge(eid)
        self.show_vertices(eid)
        self._update_stats()
        self.log(L(f"{eid}: 已删除形状点 #{idx}", f"{eid}: vertex #{idx} removed"))

    # ---- 选择 / 属性 ----
    def select_node(self, nid: str):
        self._selected_edge = None
        n = self.net.nodes[nid]
        self.lbl_info.setText(f"{L('交叉口', 'Junction')} {nid}: "
                              f"({n.x:.1f}, {n.y:.1f}, {n.z:.2f}), "
                              f"{L('关联道路', 'edges')} {len(self.net.incident_edges(nid))}")
        self.cb_jtype.setCurrentText(n.kind)
        self.sp_jradius.setValue(n.radius)
        for it in self.view.scene().items():
            if it.data(0) == ("node", nid):
                it.setSelected(True)

    def _on_selection(self):
        sel = [it.data(0) for it in self.view.scene().selectedItems() if it.data(0)]
        edge_ids = [d[1] for d in sel if d and d[0] == "edge"]
        node_ids = [d[1] for d in sel if d and d[0] == "node"]
        if edge_ids:
            eid = edge_ids[0]
            self._selected_edge = eid
            e = self.net.edges.get(eid)
            if e is not None:
                self.lbl_info.setText(
                    f"{eid}: {e.from_id} -> {e.to_id}\n"
                    f"{L('长度', 'Length')} {e.total_length():.1f} m, "
                    f"{e.lanes} {L('车道', 'lanes')} x {e.lane_width:.1f} m")
                self.sp_lanes.setValue(int(e.lanes))
                self.sp_lw.setValue(float(e.lane_width))
                self.sp_speed.setValue(float(e.speed) * 3.6)
                self.sp_priority.setValue(int(e.priority))
                self.cb_dir.setCurrentIndex(2 if e.bidir else 0)
        elif node_ids:
            self.select_node(node_ids[0])
        else:
            self._selected_edge = None
            self.lbl_info.setText(L("未选中对象", "Nothing selected"))
        self._update_stats()

    def _edit_selected_shape(self):
        if self._selected_edge:
            self.show_vertices(self._selected_edge)
        else:
            self.log(L("请先选中一条道路", "Select an edge first"))

    def apply_attrs(self):
        sel = [it.data(0)[1] for it in self.view.scene().selectedItems()
               if it.data(0) and it.data(0)[0] == "edge"]
        if not sel and self._selected_edge:
            sel = [self._selected_edge]
        if not sel:
            self.log(L("未选中道路", "No edge selected"))
            return
        for eid in sel:
            e = self.net.edges.get(eid)
            if e is None:
                continue
            e.lanes = int(self.sp_lanes.value())
            e.lane_width = float(self.sp_lw.value())
            e.speed = float(self.sp_speed.value()) / 3.6
            e.priority = int(self.sp_priority.value())
            idx = self.cb_dir.currentIndex()
            if idx == 1 and not e.bidir:
                e.reverse()
                e.bidir = False
            elif idx == 2:
                e.bidir = True
            elif idx == 0:
                e.bidir = False
            self._refresh_edge(eid)
        for nid in [it.data(0)[1] for it in self.view.scene().selectedItems()
                    if it.data(0) and it.data(0)[0] == "node"]:
            n = self.net.nodes.get(nid)
            if n is not None:
                n.kind = self.cb_jtype.currentText()
                n.radius = float(self.sp_jradius.value())
        self.rebuild_network(select_edges=sel)
        self.log(L(f"已更新 {len(sel)} 条道路属性", f"Updated {len(sel)} edge(s)"))

    def assign_elevation(self):
        if self.ground is None:
            QMessageBox.information(self, L("提示", "Info"),
                                    L("请先导入环境点云", "Import an environment cloud first"))
            return
        self.net.set_elevation(self.ground.height)
        self.rebuild_network()
        self.log(L("已根据点云地面高程赋值道路节点/形状点",
                   "Elevation assigned from point-cloud ground model"))

    def _set_tool(self, key: str):
        if key != "select":
            self.clear_shape_edit()
            self.view._connect_src = None
        self.view.set_tool(key)
        self._tool_acts[key].setChecked(True)
        hint = {
            "select": L("选择/拖拽节点; 点击道路加点; 右键: 形状/分割/平滑/自动连接",
                        "Select/drag nodes; click edge to add vertex; right-click: shape/split/smooth/connect"),
            "road": L("左键依次点击中心线, 右键/双击完成", "Left-click centerline points, right-click/double-click to finish"),
            "split": L("点击道路待分割位置", "Click an edge where to split"),
            "delete": L("点击道路或节点删除", "Click an edge or node to delete"),
            "junction": L("点击节点编辑交叉口, 或点击交叉处创建", "Click a node or a crossing to create/edit a junction"),
        }.get(key, "")
        self.status_hint(hint)

    def status_hint(self, text: str):
        self.statusBar().showMessage(text, 6000)

    def _update_stats(self):
        self.lbl_stats.setText(
            L(f"节点 {len(self.net.nodes)} | 道路 {len(self.net.edges)} | "
              f"总长 {sum(e.total_length() for e in self.net.edges.values()):.0f} m",
              f"Nodes {len(self.net.nodes)} | Edges {len(self.net.edges)} | "
              f"Length {sum(e.total_length() for e in self.net.edges.values()):.0f} m"))

    def _on_cursor(self, x: float, y: float):
        z = self.ground_z(x, y) if self.ground is not None else 0.0
        self.statusBar().showMessage(
            L(f"x={x:.2f}  y={y:.2f}  地面 z={z:.2f} m",
              f"x={x:.2f}  y={y:.2f}  ground z={z:.2f} m"))

    # ---- 导入/导出 ----
    def load_net_xml(self):
        f, _ = QFileDialog.getOpenFileName(self, L("载入 SUMO net.xml", "Load SUMO net.xml"),
                                           "", "SUMO network (*.net.xml)")
        if not f:
            return
        try:
            net = RoadNetwork.from_net_xml(f)
        except Exception as e:                                    # noqa: BLE001
            QMessageBox.critical(self, L("载入失败", "Load failed"), str(e))
            return
        self.net = net
        self.rebuild_network()
        self.view.fit_all()
        self.log(L(f"载入 net.xml: {len(net.nodes)} 节点, {len(net.edges)} 道路",
                   f"net.xml loaded: {len(net.nodes)} nodes, {len(net.edges)} edges"))

    def export_sumo(self):
        if len(self.net.edges) < 1:
            QMessageBox.information(self, L("提示", "Info"),
                                    L("请先绘制至少一条道路", "Draw at least one edge first"))
            return
        out = QFileDialog.getExistingDirectory(self, L("选择导出目录", "Select output folder"))
        if not out:
            return
        netc = find_netconvert()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            res = self.net.export_scenario(out, "roadnet", netconvert=netc,
                                           demo_traffic=True)
        finally:
            QApplication.restoreOverrideCursor()
        lines = [L("明文 XML 已导出:", "Plain XML exported:"),
                 "  " + os.path.basename(res["plain"]["nodes"]),
                 "  " + os.path.basename(res["plain"]["edges"]),
                 "  " + os.path.basename(res["plain"]["conns"]),
                 "  " + os.path.basename(res["plain"]["netccfg"])]
        if res["net"]:
            lines += ["", L("netconvert 生成:", "netconvert output:"),
                      "  " + os.path.basename(res["net"])]
            if res["routes"]:
                lines += ["", L("示例交通流:", "Demo traffic:"),
                          "  " + os.path.basename(res["routes"]),
                          "  " + os.path.basename(res["cfg"])]
        else:
            lines += ["", L("未找到 netconvert, 请手动执行:",
                            "netconvert not found; run manually:"),
                      f'  netconvert -c "{res["plain"]["netccfg"]}"']
        if res["log"]:
            lines += ["", L("netconvert 日志 (末尾):", "netconvert log (tail):"),
                      res["log"][-1200:]]
        QMessageBox.information(self, L("导出完成", "Export finished"), "\n".join(lines))
        self.log(L(f"导出到 {out}", f"Exported to {out}"))

    def log(self, msg: str):
        self.statusBar().showMessage(msg, 8000)

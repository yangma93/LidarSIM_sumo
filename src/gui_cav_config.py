# -*- coding: utf-8 -*-
"""CAV 复杂配置窗口: 多车型 / 多传感器 / 俯视-侧视-正视图与拖拽微调.

窗口左侧为功能区, 右侧为绘图区: 主视图 (俯视) 最大, 右侧上下排列侧视与正视.
传感器覆盖范围以扇形/圆形表示 (80% 透明填充), 俯视图与侧视图展示.
"""
from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath, QPen,
                         QPolygonF)
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsLineItem, QGraphicsPathItem, QGraphicsPolygonItem,
    QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QVBoxLayout, QWidget,
)

from lidarsim.cav import VEHICLE_CLASSES, default_sensor_height
from lidarsim.cav_config import (CavComplexConfig, CavSensor, CavTypeConfig,
                                 VEHICLE_CLASS_LABELS)
from lidarsim.config import LIDAR_PRESETS, make_lidar
from lidarsim.meshes import ModelLibrary

_LIDAR_NAMES = list(LIDAR_PRESETS.keys())
_CLASS_ORDER = ["car", "suv", "van", "truck", "bus"]
_COLORS = [QColor(220, 55, 55), QColor(45, 105, 220), QColor(35, 165, 85),
           QColor(225, 160, 25), QColor(165, 60, 200), QColor(20, 170, 190),
           QColor(200, 90, 40), QColor(120, 120, 120)]
# 传感器局部坐标系 (x 前 / y 左 / z 上)
_AXIS_COLORS = {"x": QColor(210, 45, 45), "y": QColor(30, 150, 65),
                "z": QColor(40, 95, 210)}


def _class_label(c: str) -> str:
    zh, en = VEHICLE_CLASS_LABELS.get(c, (c, c))
    return f"{zh} ({en})"


# ---------------------------------------------------------------------------
# 二维绘图视图
# ---------------------------------------------------------------------------
class CoverageView(QGraphicsView):
    sensor_selected = pyqtSignal(int)
    sensor_moved = pyqtSignal(int, dict)

    def __init__(self, mode: str, title: str, parent=None):
        super().__init__(parent)
        self.mode = mode            # top | side | front
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor(250, 251, 253)))
        self.setMouseTracking(True)
        self.setMinimumSize(180, 140)
        self.setToolTip(title)
        self._mesh_item = None
        self._dyn_items = []
        self._sensors = []
        self._selected = -1
        self._cover = 20.0
        self._dims = VEHICLE_CLASSES["car"]
        self._drag = None
        self._mesh_key = None

    # ---- 坐标映射 ----
    def _to_scene(self, h: float, v: float):
        return float(h), -float(v)

    def _from_scene(self, p):
        return float(p.x()), -float(p.y())

    def _sensor_hv(self, s: CavSensor):
        if self.mode == "top":
            return s.x, s.y
        if self.mode == "side":
            return s.x, s.z
        return s.y, s.z

    # ---- 数据 ----
    def set_data(self, mesh, vtype: str, sensors, selected: int,
                 cover: float):
        self._sensors = list(sensors or [])
        self._selected = int(selected)
        self._cover = max(2.0, float(cover))
        self._dims = VEHICLE_CLASSES.get(vtype, VEHICLE_CLASSES["car"])
        key = (vtype, getattr(mesh, "name", None), self.mode)
        if key != self._mesh_key:
            self._build_mesh(mesh, vtype)
            self._mesh_key = key
        self._rebuild_dynamic()
        self._fit()

    def _build_mesh(self, mesh, vtype: str):
        if self._mesh_item is not None:
            self.scene().removeItem(self._mesh_item)
            self._mesh_item = None
        path = QPainterPath()
        if mesh is not None and getattr(mesh, "tris", None) is not None \
                and len(mesh.tris):
            verts = np.asarray(mesh.verts, np.float64)
            tris = np.asarray(mesh.tris, np.int64)
            if len(tris) > 8000:
                tris = tris[:: max(1, len(tris) // 8000)]
            for tri in tris:
                p3 = verts[tri]
                if self.mode == "top":
                    hv = p3[:, :2]
                elif self.mode == "side":
                    hv = p3[:, [0, 2]]
                else:
                    hv = p3[:, [1, 2]]
                path.moveTo(float(hv[0, 0]), -float(hv[0, 1]))
                path.lineTo(float(hv[1, 0]), -float(hv[1, 1]))
                path.lineTo(float(hv[2, 0]), -float(hv[2, 1]))
                path.closeSubpath()
            path.setFillRule(Qt.FillRule.WindingFill)
        else:
            L, W, H = self._dims
            if self.mode == "top":
                x0, x1, y0, y1 = -L / 2, L / 2, -W / 2, W / 2
            elif self.mode == "side":
                x0, x1, y0, y1 = -L / 2, L / 2, 0.0, H
            else:
                x0, x1, y0, y1 = -W / 2, W / 2, 0.0, H
            path.addRect(min(x0, x1), -max(y0, y1), abs(x1 - x0), abs(y1 - y0))
        item = QGraphicsPathItem(path)
        item.setBrush(QBrush(QColor(205, 212, 222, 235)))
        pen = QPen(QColor(95, 110, 130), 0)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(-10)
        self.scene().addItem(item)
        self._mesh_item = item

    def _rebuild_dynamic(self):
        sc = self.scene()
        for it in self._dyn_items:
            try:
                sc.removeItem(it)
            except Exception:
                pass
        self._dyn_items = []
        # 地面参考线 (侧视/正视图)
        if self.mode in ("side", "front"):
            path = QPainterPath()
            path.moveTo(-200.0, 0.0)
            path.lineTo(200.0, 0.0)
            it = QGraphicsPathItem(path)
            pen = QPen(QColor(150, 160, 175), 0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            it.setPen(pen)
            it.setZValue(-20)
            sc.addItem(it)
            self._dyn_items.append(it)
        # 每个传感器的覆盖区 + 标记
        for i, s in enumerate(self._sensors):
            col = _COLORS[i % len(_COLORS)]
            if self.mode != "front":
                for poly in self._coverage_polys(s):
                    pts = QPolygonF([self._qpt(h, v) for h, v in poly])
                    it = QGraphicsPolygonItem(pts)
                    fill = QColor(col)
                    fill.setAlpha(51)                  # 80% 透明
                    it.setBrush(QBrush(fill))
                    pen = QPen(QColor(col), 0)
                    pen.setCosmetic(True)
                    it.setPen(pen)
                    it.setZValue(-5 + i)
                    sc.addItem(it)
                    self._dyn_items.append(it)
            h, v = self._sensor_hv(s)
            r = max(0.18, self._cover * 0.012)
            ell = QGraphicsEllipseItem(h - r, -(v + r), 2 * r, 2 * r)
            fill = QColor(col)
            fill.setAlpha(255 if i == self._selected else 200)
            ell.setBrush(QBrush(fill))
            pen = QPen(QColor(30, 30, 30), 0)
            pen.setCosmetic(True)
            if i == self._selected:
                pen.setWidth(2)
            ell.setPen(pen)
            ell.setZValue(10 + i)
            sc.addItem(ell)
            self._dyn_items.append(ell)
        self._add_axes()

    # ---- 传感器局部坐标系 ----
    @staticmethod
    def _axes_dirs(s: CavSensor):
        """由 yaw/pitch/roll 计算局部 x/y/z 轴在世界系中的单位方向."""
        cy, sy = math.cos(math.radians(s.yaw)), math.sin(math.radians(s.yaw))
        cp, sp = math.cos(math.radians(s.pitch)), math.sin(math.radians(s.pitch))
        cr, sr = math.cos(math.radians(s.roll)), math.sin(math.radians(s.roll))
        return {
            "x": (cy * cp, sy * cp, -sp),
            "y": (cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
            "z": (cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr),
        }

    def _proj(self, d):
        if self.mode == "top":
            return float(d[0]), float(d[1])
        if self.mode == "side":
            return float(d[0]), float(d[2])
        return float(d[1]), float(d[2])

    def _add_axes(self):
        if not self._sensors:
            return
        sc = self.scene()
        L, W, H = self._dims
        # 坐标轴长度按视图取景范围自适应, 避免正视图中轴过长压小车身
        if self.mode == "top":
            alen = max(0.8, min(self._cover * 0.22, max(L, W) * 0.6))
        elif self.mode == "side":
            alen = max(0.8, min(self._cover * 0.22, max(L, H) * 0.6))
        else:
            alen = max(0.45, min(W * 0.35, 1.2))
        for i, s in enumerate(self._sensors):
            h0, v0 = self._sensor_hv(s)
            lab = (i == self._selected)
            for name, d in self._axes_dirs(s).items():
                a, b = self._proj(d)
                n = math.hypot(a, b)
                col = _AXIS_COLORS[name]
                if n < 1e-3:                     # 轴垂直于视图平面: 画小圆示意
                    if not lab:
                        continue
                    rr = alen * 0.10
                    it = QGraphicsEllipseItem(h0 - rr, -(v0 + rr), 2 * rr, 2 * rr)
                    pen = QPen(col, 0)
                    pen.setCosmetic(True)
                    it.setPen(pen)
                    it.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                    it.setZValue(16)
                    sc.addItem(it)
                    self._dyn_items.append(it)
                    continue
                h1 = h0 + alen * a / n
                v1 = v0 + alen * b / n
                ln = QGraphicsLineItem(h0, -v0, h1, -v1)
                pen = QPen(col, 2 if lab else 1)
                pen.setCosmetic(True)
                ln.setPen(pen)
                ln.setZValue(15)
                sc.addItem(ln)
                self._dyn_items.append(ln)
                if lab:
                    txt = QGraphicsSimpleTextItem(name)
                    font = QFont()
                    font.setBold(True)
                    font.setPointSizeF(8.5)
                    txt.setFont(font)
                    txt.setBrush(QBrush(col))
                    txt.setFlag(
                        QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
                        True)
                    txt.setPos(h1, -v1)
                    txt.setZValue(17)
                    sc.addItem(txt)
                    self._dyn_items.append(txt)

    def _qpt(self, h, v):
        from PyQt6.QtCore import QPointF
        return QPointF(float(h), -float(v))

    def _coverage_polys(self, s: CavSensor):
        spec = make_lidar(s.preset)
        r = min(float(spec.range_max), float(self._cover))
        hfov = math.radians(spec.fov_h_deg)
        vmin = math.radians(spec.vfov_min_deg)
        vmax = math.radians(spec.vfov_max_deg)
        polys = []
        if self.mode == "top":
            cx, cy = s.x, s.y
            if spec.fov_h_deg >= 360:
                ang = np.linspace(0, 2 * np.pi, 97)
                polys.append(np.stack([cx + r * np.cos(ang),
                                       cy + r * np.sin(ang)], 1))
            else:
                yaw = math.radians(s.yaw)
                ang = np.linspace(yaw - hfov / 2, yaw + hfov / 2, 65)
                px = np.concatenate([[cx], cx + r * np.cos(ang)])
                py = np.concatenate([[cy], cy + r * np.sin(ang)])
                polys.append(np.stack([px, py], 1))
        elif self.mode == "side":
            cx = s.x
            pitch = math.radians(s.pitch)
            a0, a1 = vmin + pitch, vmax + pitch
            polys.append(self._sector_xz(cx, s.z, r, a0, a1))
            if spec.fov_h_deg >= 360:
                polys.append(self._sector_xz(cx, s.z, r,
                                             math.pi - a1, math.pi - a0))
        return polys

    @staticmethod
    def _sector_xz(cx, cz, r, a0, a1, n=65):
        ang = np.linspace(a0, a1, n)
        px = np.concatenate([[cx], cx + r * np.cos(ang)])
        pz = np.concatenate([[cz], cz + r * np.sin(ang)])
        return np.stack([px, pz], 1)

    def _fit(self):
        L, W, H = self._dims
        if self.mode == "top":
            x0, x1 = -self._cover - L / 2, self._cover + L / 2
            y0, y1 = -self._cover - W / 2, self._cover + W / 2
        elif self.mode == "side":
            x0, x1 = -self._cover - L / 2, self._cover + L / 2
            y0, y1 = -0.5, self._cover
        else:
            x0, x1 = -W * 0.75, W * 0.75
            y0, y1 = -0.5, max(H, max([s.z for s in self._sensors] or [H])) + 0.5
        m = max(x1 - x0, y1 - y0, 1.0) * 0.06
        self.setSceneRect(x0 - m, -(y1 + m), (x1 - x0) + 2 * m, (y1 - y0) + 2 * m)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ---- 拖拽 ----
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._sensors:
            p = self.mapToScene(ev.position().toPoint())
            h, v = self._from_scene(p)
            best, bd = -1, 1e18
            for i, s in enumerate(self._sensors):
                sh, sv = self._sensor_hv(s)
                d = (sh - h) ** 2 + (sv - v) ** 2
                if d < bd:
                    best, bd = i, d
            tol = max(0.5, self._cover * 0.05)
            if best >= 0 and math.sqrt(bd) <= tol:
                s = self._sensors[best]
                self.sensor_selected.emit(best)
                self._selected = best
                self._drag = {
                    "idx": best, "shift": bool(
                        ev.modifiers() & Qt.KeyboardModifier.ShiftModifier),
                    "press_h": h, "press_v": v,
                    "x": s.x, "y": s.y, "z": s.z,
                    "roll": s.roll, "yaw": s.yaw, "pitch": s.pitch,
                }
                ev.accept()
                return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            super().mouseMoveEvent(ev)
            return
        p = self.mapToScene(ev.position().toPoint())
        h, v = self._from_scene(p)
        d = self._drag
        i = d["idx"]
        s = self._sensors[i]
        up = {}
        if self.mode == "top":
            if d["shift"]:
                ang = math.degrees(math.atan2(v - s.y, h - s.x))
                up["yaw"] = ang
            else:
                up["x"] = d["x"] + (h - d["press_h"])
                up["y"] = d["y"] + (v - d["press_v"])
        elif self.mode == "side":
            if d["shift"]:
                up["pitch"] = math.degrees(math.atan2(v - s.z, h - s.x))
            else:
                up["z"] = max(-0.5, min(6.0, d["z"] + (v - d["press_v"])))
        else:  # front
            if d["shift"]:
                up["roll"] = math.degrees(math.atan2(v - s.z, h - s.y))
            else:
                up["y"] = d["y"] + (h - d["press_h"])
                up["z"] = max(-0.5, min(6.0, d["z"] + (v - d["press_v"])))
        if up:
            self.sensor_moved.emit(i, up)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self._drag = None
        super().mouseReleaseEvent(ev)


# ---------------------------------------------------------------------------
# 单个传感器参数行
# ---------------------------------------------------------------------------
class SensorRow(QWidget):
    changed = pyqtSignal()
    selected = pyqtSignal()

    def __init__(self, idx: int, sensor: CavSensor, vtype: str, parent=None):
        super().__init__(parent)
        self.idx = idx
        self.sensor = sensor
        self.vtype = vtype
        self._updating = False
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 3, 2, 3)
        root.setSpacing(3)

        head = QHBoxLayout()
        self.lbl = QLabel(f"LiDAR {idx + 1}")
        self.lbl.setStyleSheet("font-weight:600;color:#1565c0;")
        self.preset = QComboBox()
        self.preset.addItems(_LIDAR_NAMES)
        self.preset.setCurrentText(sensor.preset)
        self.preset.currentIndexChanged.connect(self._on_preset_changed)
        self.btn_sel = QPushButton("选中")
        self.btn_sel.setCheckable(True)
        self.btn_sel.clicked.connect(self.selected.emit)
        head.addWidget(self.lbl)
        head.addWidget(self.preset, 1)
        head.addWidget(self.btn_sel)
        root.addLayout(head)

        self.pos = {}
        root.addLayout(self._row(("x", "y", "z"), -50.0, 50.0, 2, 0.05, "pos"))
        self.att = {}
        root.addLayout(self._row(("roll", "yaw", "pitch"), -180.0, 180.0, 1,
                                 1.0, "att"))
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("位置预设"))
        self.pos_preset = QComboBox()
        self.pos_preset.addItems(["自定义", "车顶中心", "车顶前部", "车顶后部", "车头"])
        self.pos_preset.currentIndexChanged.connect(self._apply_preset)
        preset_row.addWidget(self.pos_preset, 1)
        root.addLayout(preset_row)
        self._sync_from_sensor()

    def _row(self, names, lo, hi, dec, step, kind):
        lay = QHBoxLayout()
        lay.setSpacing(3)
        store = self.pos if kind == "pos" else self.att
        suffix = " m" if kind == "pos" else "°"
        for n in names:
            cap = QLabel(n)
            cap.setMinimumWidth(30)
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(dec)
            sp.setSingleStep(step)
            sp.setSuffix(suffix)
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(lambda _v, k=n, w=kind: self._on_spin(k, w))
            lay.addWidget(cap)
            lay.addWidget(sp, 1)
            store[n] = sp
        return lay

    def _on_preset_changed(self, _i):
        if self._updating:
            return
        self.sensor.preset = self.preset.currentText()
        self.changed.emit()

    def _on_spin(self, name, kind):
        if self._updating:
            return
        if kind == "pos":
            setattr(self.sensor, name, float(self.pos[name].value()))
        else:
            setattr(self.sensor, name, float(self.att[name].value()))
        self.changed.emit()

    def _sync_from_sensor(self):
        self._updating = True
        s = self.sensor
        for n in ("x", "y", "z"):
            self.pos[n].setValue(float(getattr(s, n)))
        for n in ("roll", "yaw", "pitch"):
            self.att[n].setValue(float(getattr(s, n)))
        self.preset.setCurrentText(s.preset)
        self._updating = False

    def update_from_sensor(self, s: CavSensor):
        self.sensor = s
        self._sync_from_sensor()

    def set_selected(self, on: bool):
        self.btn_sel.setChecked(bool(on))
        self.setStyleSheet("background:#e8f1ff;" if on else "")

    def _apply_preset(self, i):
        if self._updating or i <= 0:
            return
        L, W, H = VEHICLE_CLASSES.get(self.vtype, VEHICLE_CLASSES["car"])
        z = default_sensor_height(self.vtype)
        pos = {
            1: (0.0, 0.0, z),
            2: (0.3 * L, 0.0, z),
            3: (-0.3 * L, 0.0, z),
            4: (0.35 * L, 0.0, 0.6 * H),
        }.get(i)
        if pos:
            self._updating = True
            self.pos["x"].setValue(pos[0])
            self.pos["y"].setValue(pos[1])
            self.pos["z"].setValue(pos[2])
            self._updating = False
            self.sensor.x, self.sensor.y, self.sensor.z = pos
            self.changed.emit()


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class CavConfigDialog(QDialog):
    """CAV 复杂配置 (独立窗口, 主界面与 SUMO 联仿共用)."""

    def __init__(self, parent=None, models: ModelLibrary = None,
                 config: CavComplexConfig = None, on_changed=None):
        super().__init__(parent)
        self.setWindowTitle("CAV 复杂配置")
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.resize(1280, 860)
        self.models = models
        self.config = (config or CavComplexConfig()).copy()
        if not self.config.types:
            self.config.ensure_types(1)
        self.on_changed = on_changed
        self._building = False
        self._sel = 0
        self._rows = []
        self._build_ui()
        self._load_type(0)

    # ---- UI ----
    def _build_ui(self):
        root = QHBoxLayout(self)
        left = QWidget()
        left.setFixedWidth(430)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(6)

        self.btn_toggle = QPushButton()
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(self.config.enabled)
        self.btn_toggle.clicked.connect(self._toggle_enabled)
        ll.addWidget(self.btn_toggle)

        gb = QGroupBox("车型与渗透率")
        gl = QVBoxLayout(gb)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("CAV 类型数量"))
        self.cb_types = QComboBox()
        self.cb_types.addItems([str(i) for i in range(1, 6)])
        self.cb_types.setCurrentIndex(max(0, len(self.config.types) - 1))
        self.cb_types.currentIndexChanged.connect(
            lambda i: self._on_types_changed(i + 1))
        r1.addWidget(self.cb_types)
        r1.addWidget(QLabel("当前配置"))
        self.cb_type = QComboBox()
        self.cb_type.currentIndexChanged.connect(self._load_type)
        r1.addWidget(self.cb_type, 1)
        gl.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("车型"))
        self.cb_class = QComboBox()
        for c in _CLASS_ORDER:
            self.cb_class.addItem(_class_label(c), c)
        self.cb_class.currentIndexChanged.connect(self._on_class_changed)
        r2.addWidget(self.cb_class, 1)
        r2.addWidget(QLabel("渗透率"))
        self.sp_pen = QDoubleSpinBox()
        self.sp_pen.setRange(0.0, 1.0)
        self.sp_pen.setDecimals(2)
        self.sp_pen.setSingleStep(0.01)
        self.sp_pen.valueChanged.connect(self._on_pen_changed)
        r2.addWidget(self.sp_pen)
        gl.addLayout(r2)
        ll.addWidget(gb)

        gb2 = QGroupBox("传感器")
        sl = QVBoxLayout(gb2)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("传感器数量"))
        self.sp_n = QSpinBox()
        self.sp_n.setRange(1, 8)
        self.sp_n.valueChanged.connect(self._on_n_changed)
        r3.addWidget(self.sp_n)
        r3.addWidget(QLabel("覆盖显示范围 (m)"))
        self.sp_cover = QDoubleSpinBox()
        self.sp_cover.setRange(3.0, 100.0)
        self.sp_cover.setDecimals(0)
        self.sp_cover.setSingleStep(1.0)
        self.sp_cover.setValue(15.0)
        self.sp_cover.valueChanged.connect(lambda _v: self._refresh_views())
        r3.addWidget(self.sp_cover)
        r3.addStretch(1)
        sl.addLayout(r3)

        self.rows_host = QWidget()
        self.rows_lay = QVBoxLayout(self.rows_host)
        self.rows_lay.setContentsMargins(0, 0, 0, 0)
        self.rows_lay.setSpacing(2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.rows_host)
        sl.addWidget(scroll, 1)
        ll.addWidget(gb2, 1)

        tip = QLabel("拖拽: 俯视图左键移动位置, Shift+左键调整水平朝向;\n"
                     "侧视图左键调整高度, Shift+左键调整俯仰; 正视图可调 y/z。")
        tip.setStyleSheet("color:#607d9b;")
        tip.setWordWrap(True)
        ll.addWidget(tip)

        row = QHBoxLayout()
        self.btn_default = QPushButton("恢复默认")
        self.btn_default.clicked.connect(self._restore_default)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("确定")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._accept)
        row.addWidget(self.btn_default)
        row.addStretch(1)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_ok)
        ll.addLayout(row)
        root.addWidget(left)

        right = QWidget()
        grid = QGridLayout(right)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)
        self.v_top = CoverageView("top", "俯视图")
        self.v_side = CoverageView("side", "侧视图")
        self.v_front = CoverageView("front", "正视图")
        # 右侧: 侧视图 + 正视图上下等分, 与主视图同一网格行 => 高度一致
        side_col = QWidget()
        sc_lay = QVBoxLayout(side_col)
        sc_lay.setContentsMargins(0, 0, 0, 0)
        sc_lay.setSpacing(4)
        for title, view in (("侧视图", self.v_side), ("正视图", self.v_front)):
            box = QVBoxLayout()
            box.setSpacing(2)
            box.addWidget(QLabel(title))
            box.addWidget(view, 1)
            sc_lay.addLayout(box, 1)
        grid.addWidget(QLabel("俯视图 (主视图)"), 0, 0)
        grid.addWidget(self.v_top, 1, 0)
        grid.addWidget(side_col, 1, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)
        for v in (self.v_top, self.v_side, self.v_front):
            v.sensor_selected.connect(self._select_sensor)
            v.sensor_moved.connect(self._on_drag)
        root.addWidget(right, 1)
        self._refresh_type_combo()
        self._refresh_toggle()

    # ---- 状态 ----
    def _toggle_enabled(self):
        self.config.enabled = bool(self.btn_toggle.isChecked())
        self._refresh_toggle()

    def _refresh_toggle(self):
        on = bool(self.btn_toggle.isChecked())
        self.btn_toggle.setText("✓ 已启用复杂配置 (点击停用)" if on
                                else "启用复杂配置 (当前未启用)")
        self.btn_toggle.setStyleSheet(
            "text-align:left;padding:6px;font-weight:600;"
            + ("background:#e3f2e8;color:#1b5e20;" if on
               else "background:#f2f3f5;color:#555;"))

    def _cur_type(self) -> CavTypeConfig:
        return self.config.types[self.cb_type.currentIndex()]

    def _refresh_type_combo(self):
        self._building = True
        cur = self.cb_type.currentIndex()
        self.cb_type.clear()
        for i, t in enumerate(self.config.types):
            self.cb_type.addItem(f"类型 {i + 1}: {t.vtype}", i)
        self.cb_type.setCurrentIndex(max(0, min(cur, len(self.config.types) - 1)))
        self._building = False

    def _load_type(self, idx: int):
        if self._building or not self.config.types:
            return
        idx = max(0, min(int(idx), len(self.config.types) - 1))
        t = self.config.types[idx]
        self._building = True
        ci = self.cb_class.findData(t.vtype)
        self.cb_class.setCurrentIndex(max(0, ci))
        self.sp_pen.setValue(float(t.penetration))
        t.ensure_sensors(max(1, len(t.sensors)))
        self.sp_n.setValue(len(t.sensors))
        self._building = False
        self._sel = 0
        self._rebuild_rows()
        self._refresh_views()

    def _on_types_changed(self, n):
        if self._building:
            return
        self.config.ensure_types(n)
        self._refresh_type_combo()
        self._load_type(self.cb_type.currentIndex())

    def _on_class_changed(self, _i):
        if self._building:
            return
        t = self._cur_type()
        t.vtype = self.cb_class.currentData() or "car"
        # 默认高度跟随新车型, 仅当传感器仍在默认高度时
        for s in t.sensors:
            s.z = default_sensor_height(t.vtype)
        self._refresh_type_combo()
        self._load_type(self.cb_type.currentIndex())

    def _on_pen_changed(self, v):
        if self._building:
            return
        self._cur_type().penetration = float(v)

    def _on_n_changed(self, n):
        if self._building:
            return
        self._cur_type().ensure_sensors(n)
        self._sel = min(self._sel, len(self._cur_type().sensors) - 1)
        self._rebuild_rows()
        self._refresh_views()

    def _rebuild_rows(self):
        while self.rows_lay.count():
            it = self.rows_lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        self._rows = []
        t = self._cur_type()
        for i, s in enumerate(t.sensors):
            row = SensorRow(i, s, t.vtype)
            row.changed.connect(self._refresh_views)
            row.selected.connect(lambda i=i: self._select_sensor(i))
            self.rows_lay.addWidget(row)
            self._rows.append(row)
        self.rows_lay.addStretch(1)
        self._select_sensor(self._sel, emit=False)

    def _select_sensor(self, idx: int, emit=True):
        if not self._rows:
            return
        self._sel = max(0, min(int(idx), len(self._rows) - 1))
        for i, r in enumerate(self._rows):
            r.set_selected(i == self._sel)
        self._refresh_views()

    def _on_drag(self, idx: int, up: dict):
        t = self._cur_type()
        if not (0 <= idx < len(t.sensors)):
            return
        s = t.sensors[idx]
        for k, v in up.items():
            setattr(s, k, float(v))
        if idx < len(self._rows):
            row = self._rows[idx]
            row._updating = True
            for k in ("x", "y", "z"):
                if k in up:
                    row.pos[k].setValue(float(up[k]))
            for k in ("roll", "yaw", "pitch"):
                if k in up:
                    row.att[k].setValue(float(up[k]))
            row._updating = False
        self._refresh_views()

    def _refresh_views(self):
        if not self.config.types:
            return
        t = self._cur_type()
        mesh = None
        if self.models is not None:
            try:
                mesh = self.models.by_class(t.vtype, VEHICLE_CLASSES.get(t.vtype))
            except Exception:
                mesh = None
        cover = float(self.sp_cover.value())
        for v in (self.v_top, self.v_side, self.v_front):
            v.set_data(mesh, t.vtype, t.sensors, self._sel, cover)

    def _restore_default(self):
        self.config = CavComplexConfig()
        self._building = True
        self.btn_toggle.setChecked(self.config.enabled)
        self.cb_types.setCurrentIndex(len(self.config.types) - 1)
        self._building = False
        self._refresh_type_combo()
        self._refresh_toggle()
        self._load_type(0)

    def _accept(self):
        self.config.enabled = bool(self.btn_toggle.isChecked())
        if self.on_changed is not None:
            try:
                self.on_changed(self.config)
            except Exception:
                pass
        self.accept()

    def set_context(self, models, config):
        """外部重复打开时同步最新模型库与配置对象."""
        self.models = models
        self.config = (config or CavComplexConfig()).copy()
        if not self.config.types:
            self.config.ensure_types(1)
        self._building = True
        self.btn_toggle.setChecked(self.config.enabled)
        self.cb_types.setCurrentIndex(len(self.config.types) - 1)
        self._building = False
        self._refresh_toggle()
        self._refresh_type_combo()
        self._load_type(min(self.cb_type.currentIndex(), len(self.config.types) - 1))

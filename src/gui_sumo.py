# -*- coding: utf-8 -*-
"""SUMO 联合仿真控制面板与状态指示灯."""
from __future__ import annotations

import os
import threading

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from lidarsim.config import LIDAR_PRESETS
from lidarsim.cav_config import CavComplexConfig
from lidarsim.roadnet import find_sumo_binary, find_netconvert
from lidarsim.sumo_cosim import (CosimConfig, CosimSession, test_connection,
                                 traci_available, traci_message)

_LANG = "zh"


def set_language(lang: str):
    global _LANG
    _LANG = "en" if lang == "en" else "zh"


def L(zh: str, en: str) -> str:
    return en if _LANG == "en" else zh


# ---------------------------------------------------------------------------
# 状态灯
# ---------------------------------------------------------------------------
class LinkLight(QWidget):
    """红/黄/绿 + 灰 数据流状态灯."""

    COLORS = {0: "#e53935", 1: "#fbc02d", 2: "#2e7d32", 3: "#9e9e9e"}
    NAMES = {0: ("未连接", "Disconnected"), 1: ("已连接/等待", "Connected / waiting"),
             2: ("数据流正常", "Data flowing"), 3: ("空闲", "Idle")}

    def __init__(self, compact: bool = False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(5)
        self.dot = QLabel()
        self.dot.setFixedSize(12, 12)
        lay.addWidget(self.dot)
        self.text = QLabel()
        if not compact:
            lay.addWidget(self.text)
        else:
            self.text.hide()
        self._level = 3
        self.set_state(3)

    def set_state(self, level: int, text: str = ""):
        self._level = int(level)
        color = self.COLORS.get(self._level, "#9e9e9e")
        self.dot.setStyleSheet(
            f"background:{color}; border-radius:6px; border:1px solid rgba(0,0,0,0.25);")
        name = text or L(*self.NAMES.get(self._level, ("未知", "Unknown")))
        self.text.setText(name)
        self.setToolTip(name)


# ---------------------------------------------------------------------------
# 后台联仿线程
# ---------------------------------------------------------------------------
class SumoWorker(QThread):
    frame = pyqtSignal(object)
    status = pyqtSignal(int, str)
    logmsg = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: CosimConfig, env, models, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.env = env
        self.models = models
        self.stop_event = threading.Event()
        self.session = None
        self._ack = None

    def run(self):
        try:
            self.session = CosimSession(self.cfg, self.env, self.models,
                                        logger=self.logmsg.emit)
            self.session.open()
            self.status.emit(1, L("SUMO 已连接，仿真启动", "SUMO connected, simulation started"))
            self.session.run(
                self.stop_event,
                on_frame=self._emit_frame,
                on_status=self.status.emit,
                on_log=self.logmsg.emit)
        except Exception as e:                                    # noqa: BLE001
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self.session = None

    def _emit_frame(self, payload):
        """发射一帧并阻塞等待 GUI 绘制完成.

        联仿线程在没有可视化时可能每秒推进上千步; 若只 emit 不等待, Qt 事件
        队列会积压到卡死. 等待 GUI 回执后 SUMO 才会执行下一步, 从而让 SUMO
        与三维可视化严格同步 (锁步).
        """
        self._ack = threading.Event()
        try:
            self.frame.emit(payload)
            self._ack.wait(timeout=2.0)
        finally:
            self._ack = None

    def frame_ack(self):
        """GUI 线程绘制完成后回执."""
        ev = self._ack
        if ev is not None:
            ev.set()

    def stop(self):
        self.stop_event.set()
        self.frame_ack()          # 尽快解除锁步等待


class TestWorker(QThread):
    """后台连接测试, 避免 traci 启动/关闭期间阻塞 Qt 事件循环."""

    done = pyqtSignal(bool, str)

    def __init__(self, cfg: CosimConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            ok, msg = test_connection(self.cfg)
        except Exception as e:                                    # noqa: BLE001
            ok, msg = False, f"{type(e).__name__}: {e}"
        self.done.emit(ok, msg)


# ---------------------------------------------------------------------------
# 控制面板
# ---------------------------------------------------------------------------
class SumoSimDialog(QDialog):
    def __init__(self, parent=None, env=None, models=None, algo_params=None,
                 lidar_configs=None, on_frame=None, on_log=None, on_status=None,
                 get_lidars=None, on_need_lidar=None, default_net="",
                 cav_config=None, on_cav_config=None):
        super().__init__(parent)
        # 独立窗口标题栏: 提供最小化按钮
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint, True)
        self.setWindowTitle(L("SUMO 联合仿真", "SUMO Co-simulation"))
        self.resize(780, 860)
        self.env = env
        self.models = models
        self.algo_params = dict(algo_params or {})
        self.lidar_configs = list(lidar_configs or [])
        self.get_lidars = get_lidars
        self.on_need_lidar = on_need_lidar
        self.on_frame = on_frame
        self.on_log = on_log
        self.on_status = on_status
        self.on_cav_config = on_cav_config
        self.cav_config = cav_config or CavComplexConfig()
        self.default_net = default_net
        self.worker = None
        self.test_worker = None
        self._viz_seen = 0
        self._build_ui()
        self._lidar_timer = QTimer(self)
        self._lidar_timer.setInterval(1000)
        self._lidar_timer.timeout.connect(self._poll_lidars)
        self._refresh_sumo_paths()
        if self.chk_lidars.isChecked() and not self._has_lidar_config():
            QTimer.singleShot(0, self._request_lidar_config)

    # ---- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)
        # SUMO 配置
        gb = QGroupBox(L("1. SUMO 场景", "1. SUMO scenario"))
        fl = QFormLayout(gb)
        row = QHBoxLayout()
        self.ed_bin = QLineEdit()
        self.ed_bin.setPlaceholderText(L("留空则自动搜索 sumo / sumo-gui",
                                         "Empty = auto-detect sumo / sumo-gui"))
        b = QPushButton("...")
        b.setFixedWidth(34)
        b.clicked.connect(lambda: self._browse(self.ed_bin, True))
        row.addWidget(self.ed_bin, 1); row.addWidget(b)
        fl.addRow(L("SUMO 可执行文件", "SUMO binary"), row)
        row = QHBoxLayout()
        self.ed_cfg = QLineEdit()
        b = QPushButton("...")
        b.setFixedWidth(34)
        b.clicked.connect(lambda: self._browse(self.ed_cfg, False))
        row.addWidget(self.ed_cfg, 1); row.addWidget(b)
        fl.addRow(L("SUMO 配置文件 (.sumocfg)", "SUMO config (.sumocfg)"), row)
        row = QHBoxLayout()
        self.ed_net = QLineEdit()
        b = QPushButton("...")
        b.setFixedWidth(34)
        b.clicked.connect(lambda: self._browse(self.ed_net, False))
        row.addWidget(self.ed_net, 1); row.addWidget(b)
        fl.addRow(L("或路网文件 (.net.xml)", "or network (.net.xml)"), row)
        if self.default_net:
            self.ed_net.setText(self.default_net)
        self.chk_gui = QCheckBox(L("启动 SUMO 仿真界面 (sumo-gui, 与主界面同步)",
                                   "Launch SUMO GUI (sumo-gui, synced with main view)"))
        self.chk_gui.setChecked(False)
        self.chk_gui.setToolTip(L(
            "勾选: 打开 sumo-gui 窗口, 随联仿逐步刷新, 与主界面可视化同步;\n"
            "不勾选: 使用 sumo 在后台静默运行.",
            "Checked: open sumo-gui and refresh step by step together with the main view;\n"
            "unchecked: run sumo headless in the background."))
        fl.addRow(L("SUMO 界面", "SUMO GUI"), self.chk_gui)
        self.chk_viz = QCheckBox(L("在主界面 3D 视图动态展示", "Visualize in main 3D view"))
        self.chk_viz.setChecked(True)
        row_viz = QHBoxLayout()
        row_viz.addWidget(self.chk_viz)
        self.chk_sep_win = QCheckBox(L("单独窗口", "Separate window"))
        self.chk_sep_win.setChecked(True)
        self.chk_sep_win.setToolTip(L(
            "勾选: 在新窗口动态展示 3D 点云 (黑底, turbo, intensity 着色, 不绘制静态环境点云);\n"
            "取消: 在主界面 3D 视图中展示.",
            "Checked: show the 3D cloud in a separate window (black bg, turbo, intensity;\n"
            "no static environment cloud); unchecked: show in the main 3D view."))
        row_viz.addWidget(self.chk_sep_win)
        row_viz.addStretch(1)
        fl.addRow(L("主界面显示", "Main view"), row_viz)

        # 联仿可视化工作栏: IoU 阈值 / GT BBox / 感知 BBox
        vizbox = QGroupBox(L("联仿可视化", "Co-sim visualization"))
        vl = QHBoxLayout(vizbox)
        vl.addWidget(QLabel(L("IoU 阈值", "IoU threshold")))
        self.sp_iou = QDoubleSpinBox()
        self.sp_iou.setRange(0.0, 1.0)
        self.sp_iou.setDecimals(2)
        self.sp_iou.setSingleStep(0.05)
        self.sp_iou.setValue(0.30)
        self.sp_iou.setToolTip(L(
            "同一目标的多源点云融合后做 L-shape 拟合, IoU >= 阈值视为已感知 (SUMO 绿色),\n"
            "否则视为未感知 (SUMO 红色).",
            "Points on the same target from all roadside LiDARs/CAVs are fused and fitted\n"
            "with L-shape; IoU >= threshold = perceived (green in SUMO), otherwise red."))
        vl.addWidget(self.sp_iou)
        self.chk_gt = QCheckBox(L("绘制 GT BBox", "Draw GT BBox"))
        self.chk_gt.setChecked(False)
        self.chk_det = QCheckBox(L("绘制感知 BBox", "Draw perceived BBox"))
        self.chk_det.setChecked(True)
        vl.addWidget(self.chk_gt)
        vl.addWidget(self.chk_det)
        vl.addStretch(1)
        fl.addRow(vizbox)
        row = QHBoxLayout()
        self.sp_step = QDoubleSpinBox(); self.sp_step.setRange(0.02, 1.0)
        self.sp_step.setDecimals(3); self.sp_step.setSingleStep(0.05); self.sp_step.setValue(0.1)
        row.addWidget(QLabel(L("步长 (s)", "Step (s)")))
        row.addWidget(self.sp_step)
        self.sp_every = QSpinBox(); self.sp_every.setRange(1, 100); self.sp_every.setValue(1)
        row.addWidget(QLabel(L("每 N 步做一次感知", "Perceive every N steps")))
        row.addWidget(self.sp_every)
        self.sp_port = QSpinBox(); self.sp_port.setRange(1000, 65535); self.sp_port.setValue(8813)
        row.addWidget(QLabel(L("端口", "Port")))
        row.addWidget(self.sp_port)
        row.addStretch(1)
        fl.addRow(L("时序", "Timing"), row)
        root.addWidget(gb)

        # 感知
        gb2 = QGroupBox(L("2. 车路协同感知", "2. Cooperative perception"))
        fl2 = QFormLayout(gb2)
        self.cb_cav = QComboBox()
        self.cb_cav.addItems([L("无 CAV", "No CAV"), L("指定 SUMO 车辆 ID", "SUMO vehicle IDs"),
                              L("CAV 渗透率", "CAV penetration")])
        self.cb_cav.setCurrentIndex(2)
        fl2.addRow(L("CAV 选择", "CAV selection"), self.cb_cav)
        self.ed_ids = QLineEdit()
        self.ed_ids.setPlaceholderText(L("逗号/空格分隔, 如 veh0, veh3",
                                         "comma/space separated, e.g. veh0, veh3"))
        fl2.addRow(L("CAV ID 列表", "CAV IDs"), self.ed_ids)
        self.sp_pen = QDoubleSpinBox(); self.sp_pen.setRange(0.0, 1.0)
        self.sp_pen.setDecimals(2); self.sp_pen.setSingleStep(0.05); self.sp_pen.setValue(0.15)
        fl2.addRow(L("渗透率", "Penetration"), self.sp_pen)
        row = QHBoxLayout()
        self.cb_preset = QComboBox(); self.cb_preset.addItems(LIDAR_PRESETS)
        self.sp_height = QDoubleSpinBox(); self.sp_height.setRange(0.0, 3.0)
        self.sp_height.setDecimals(2); self.sp_height.setSingleStep(0.05); self.sp_height.setValue(0.2)
        self.sp_minpts = QSpinBox(); self.sp_minpts.setRange(1, 100); self.sp_minpts.setValue(5)
        row.addWidget(self.cb_preset); row.addWidget(QLabel(L("车顶高度", "roof offset")))
        row.addWidget(self.sp_height); row.addWidget(QLabel(L("最少点数", "min points")))
        row.addWidget(self.sp_minpts)
        self.btn_cav_complex = QPushButton(L("复杂配置", "Complex config"))
        self.btn_cav_complex.clicked.connect(self._open_cav_complex)
        row.addWidget(self.btn_cav_complex)
        row.addStretch(1)
        fl2.addRow(L("车端 LiDAR", "Onboard LiDAR"), row)
        self.lbl_cav_complex = QLabel()
        self.lbl_cav_complex.setWordWrap(True)
        self.lbl_cav_complex.setStyleSheet("color:#607d9b;")
        fl2.addRow(L("复杂配置", "Complex config"), self.lbl_cav_complex)
        self.lbl_lidars = QLabel()
        self.lbl_lidars.setWordWrap(True)
        fl2.addRow(L("路侧 LiDAR", "Roadside LiDAR"), self.lbl_lidars)
        self.chk_lidars = QCheckBox(L("启用主界面路侧 LiDAR", "Enable roadside LiDAR from main window"))
        self.chk_lidars.setChecked(bool(self.lidar_configs))
        self.chk_lidars.stateChanged.connect(self._on_lidar_toggle)
        row_l = QHBoxLayout()
        row_l.addWidget(self.chk_lidars)
        self.btn_refresh_lidar = QPushButton(L("刷新雷达配置", "Refresh LiDAR"))
        self.btn_refresh_lidar.clicked.connect(self._manual_refresh_lidars)
        row_l.addWidget(self.btn_refresh_lidar)
        row_l.addStretch(1)
        fl2.addRow("", row_l)
        root.addWidget(gb2)

        # 反馈
        gb3 = QGroupBox(L("3. 感知反馈 SUMO", "3. Perception feedback"))
        fl3 = QFormLayout(gb3)
        row = QHBoxLayout()
        self.chk_feedback = QCheckBox(L("检测结果写回 (颜色 / 自定义参数)",
                                        "Write detections back (color / parameters)"))
        self.chk_feedback.setChecked(True)
        row.addWidget(self.chk_feedback)
        row.addStretch(1)
        fl3.addRow(L("反馈", "Feedback"), row)
        row = QHBoxLayout()
        self.chk_control = QCheckBox(L("闭环速度控制 (TTC 触发减速)",
                                       "Closed-loop speed control (TTC braking)"))
        row.addWidget(self.chk_control)
        self.sp_ttc = QDoubleSpinBox(); self.sp_ttc.setRange(0.2, 20.0)
        self.sp_ttc.setDecimals(1); self.sp_ttc.setSingleStep(0.5); self.sp_ttc.setValue(2.5)
        row.addWidget(QLabel(L("TTC 阈值 (s)", "TTC threshold (s)")))
        row.addWidget(self.sp_ttc); row.addStretch(1)
        fl3.addRow(L("控制", "Control"), row)
        root.addWidget(gb3)

        # 状态
        row = QHBoxLayout()
        self.light = LinkLight()
        row.addWidget(self.light)
        self.lbl_state = QLabel(L("未连接", "Disconnected"))
        row.addWidget(self.lbl_state, 1)
        root.addLayout(row)

        row = QHBoxLayout()
        self.btn_env = QPushButton(L("环境信息", "Environment"))
        self.btn_env.clicked.connect(self._show_env)
        self.btn_test = QPushButton(L("连接测试", "Test connection"))
        self.btn_test.clicked.connect(self._test)
        self.btn_start = QPushButton(L("开始联仿", "Start"))
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton(L("停止", "Stop"))
        self.btn_stop.clicked.connect(self._stop)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_env)
        row.addStretch(1)
        row.addWidget(self.btn_test)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        root.addLayout(row)

        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        root.addWidget(self.log_text, 1)
        self._update_lidar_label()
        self._refresh_cav_complex_label()

    def _open_cav_complex(self):
        if self.on_cav_config is None:
            return
        try:
            self.on_cav_config()
        except Exception as e:                                    # noqa: BLE001
            self._log(f"打开复杂配置失败: {e}")

    def set_cav_config(self, cfg):
        self.cav_config = cfg or CavComplexConfig()
        self._refresh_cav_complex_label()

    def _refresh_cav_complex_label(self):
        on = bool(getattr(self.cav_config, "enabled", False))
        if on:
            self.lbl_cav_complex.setText(
                L(f"{self.cav_config.summary()} (按车型渗透率自动选择 CAV)",
                  f"{self.cav_config.summary()} (per-type penetration)"))
        else:
            self.lbl_cav_complex.setText(
                L("未启用: 使用上方简单的车端 LiDAR 配置",
                  "Disabled: using the simple onboard LiDAR settings above"))
        for w in (self.cb_preset, self.sp_height, self.sp_pen):
            w.setEnabled(not on)

    def _update_lidar_label(self):
        n = len(self.lidar_configs)
        if n and self.chk_lidars.isChecked():
            if not self._has_lidar_config():
                self.lbl_lidars.setText(
                    L("主界面雷达仍为默认全零位姿 -> 视为未配置, 请设置位置/姿态",
                      "Roadside LiDAR still at default zero pose -> unconfigured; set position/attitude"))
                return
            names = ", ".join(f"{p}@{i + 1}" for i, (p, _) in enumerate(self.lidar_configs))
            self.lbl_lidars.setText(L(f"沿用主界面 {n} 个路侧雷达: {names}",
                                      f"Using {n} roadside LiDAR(s) from main window: {names}"))
        elif n:
            self.lbl_lidars.setText(L("已关闭路侧感知 (仅 CAV 车端)",
                                      "Roadside perception disabled (CAV onboard only)"))
        else:
            self.lbl_lidars.setText(L("主界面未配置路侧雷达 -> 无路侧感知 (仅 CAV 车端)",
                                      "No roadside LiDAR configured -> CAV onboard only"))

    # ---- 路侧雷达配置联动 ----
    def _has_lidar_config(self) -> bool:
        """是否已在主界面实际配置雷达 (默认全零位姿视为未配置)."""
        if not self.lidar_configs:
            return False
        default_preset = next(iter(LIDAR_PRESETS), "")
        for preset, pose in self.lidar_configs:
            if any(abs(float(v)) > 1e-6 for v in pose):
                return True
            if preset and preset != default_preset:
                return True
        return len(self.lidar_configs) > 1

    def _sync_lidar_configs(self):
        if self.get_lidars is None:
            return
        try:
            self.lidar_configs = list(self.get_lidars() or [])
        except Exception as e:                                    # noqa: BLE001
            self._log(L(f"读取主界面雷达配置失败: {e}",
                        f"Failed to read roadside LiDAR config: {e}"))

    def refresh_lidars(self, configs=None):
        """主界面配置变化后由外部调用, 实时更新。"""
        had = self._has_lidar_config()
        if configs is not None:
            self.lidar_configs = list(configs)
        else:
            self._sync_lidar_configs()
        self._update_lidar_label()
        if self.chk_lidars.isChecked() and self._has_lidar_config():
            self._lidar_timer.stop()
            if not had:
                self._log(L(f"已自动获取 {len(self.lidar_configs)} 个路侧雷达配置",
                            f"Received {len(self.lidar_configs)} roadside LiDAR config(s)"))

    def _on_lidar_toggle(self, _state=0):
        if self.chk_lidars.isChecked():
            self._sync_lidar_configs()
            if not self._has_lidar_config():
                self._request_lidar_config()
            else:
                self._lidar_timer.stop()
        else:
            self._lidar_timer.stop()
        self._update_lidar_label()

    def _request_lidar_config(self):
        self._log(L("尚未配置路侧雷达, 请在主界面设置; 窗口将自动监听配置变化",
                    "No roadside LiDAR configured; configure it in the main window; listening..."))
        if self.on_need_lidar is not None:
            try:
                self.on_need_lidar()
            except Exception:                                     # noqa: BLE001
                pass
        if not self._lidar_timer.isActive():
            self._lidar_timer.start()

    def _poll_lidars(self):
        self._sync_lidar_configs()
        self._update_lidar_label()
        if self._has_lidar_config():
            self._lidar_timer.stop()
            self._log(L(f"检测到主界面路侧雷达配置 ({len(self.lidar_configs)} 个), 已自动载入",
                        f"Roadside LiDAR config detected ({len(self.lidar_configs)}), loaded"))

    def _manual_refresh_lidars(self):
        self._sync_lidar_configs()
        self._update_lidar_label()
        if self.chk_lidars.isChecked() and not self._has_lidar_config():
            self._request_lidar_config()
        else:
            self._log(L(f"当前路侧雷达: {len(self.lidar_configs)} 个",
                        f"Roadside LiDAR: {len(self.lidar_configs)}"))

    def _refresh_sumo_paths(self):
        if not self.ed_bin.text().strip():
            p = find_sumo_binary(False)
            if p:
                self.ed_bin.setText(p)
        if not self.ed_net.text().strip():
            nc = find_netconvert()
            if nc:
                self.ed_net.setPlaceholderText(
                    L(f"netconvert 可用: {os.path.basename(nc)}",
                      f"netconvert available: {os.path.basename(nc)}"))

    def _browse(self, edit: QLineEdit, exe: bool):
        if exe:
            f, _ = QFileDialog.getOpenFileName(self, L("选择 SUMO 可执行文件", "Select SUMO binary"),
                                               "", "Executable (*.exe);;All (*)")
        else:
            f, _ = QFileDialog.getOpenFileName(
                self, L("选择场景文件", "Select scenario file"), "",
                "SUMO (*.sumocfg *.net.xml);;All (*)")
        if f:
            edit.setText(f)

    # ---- 配置 ----
    def _config(self) -> CosimConfig:
        self._sync_lidar_configs()
        mode = ("none", "ids", "penetration")[self.cb_cav.currentIndex()]
        ids = [t for t in self.ed_ids.text().replace(",", " ").split() if t]
        return CosimConfig(
            sumo_binary=self.ed_bin.text().strip(),
            config_path=self.ed_cfg.text().strip(),
            net_path=self.ed_net.text().strip(),
            use_gui=self.chk_gui.isChecked(),
            step_length=float(self.sp_step.value()),
            port=int(self.sp_port.value()),
            sim_every=int(self.sp_every.value()),
            cav_mode=mode, cav_ids=ids,
            penetration=float(self.sp_pen.value()),
            cav_preset=self.cb_preset.currentText(),
            cav_height=float(self.sp_height.value()),
            lidars=[(p, list(pose)) for p, pose in self.lidar_configs]
            if self.chk_lidars.isChecked() else [],
            feedback=self.chk_feedback.isChecked(),
            control=self.chk_control.isChecked(),
            ttc_thr=float(self.sp_ttc.value()),
            viz=(self.chk_viz.isChecked() or self.chk_sep_win.isChecked()),
            min_pts=int(self.sp_minpts.value()),
            iou_thr=float(self.sp_iou.value()),
            show_gt=self.chk_gt.isChecked(),
            show_det=self.chk_det.isChecked(),
            cav_complex=self.cav_config.to_dict() if self.cav_config.enabled else {},
            algo_params=dict(self.algo_params))

    def _validate(self):
        if not self.ed_cfg.text().strip() and not self.ed_net.text().strip():
            QMessageBox.warning(self, L("提示", "Info"),
                                L("请指定 .sumocfg 或 .net.xml", "Specify .sumocfg or .net.xml"))
            return False
        for p in (self.ed_cfg.text().strip(), self.ed_net.text().strip()):
            if p and not os.path.exists(p):
                QMessageBox.warning(self, L("提示", "Info"),
                                    L(f"文件不存在: {p}", f"File not found: {p}"))
                return False
        if self.env is None:
            QMessageBox.warning(self, L("提示", "Info"),
                                L("联仿需要先在主界面加载环境点云", "Load an env cloud in the main window first"))
            return False
        if self.models is None:
            QMessageBox.warning(self, L("提示", "Info"),
                                L("联仿需要车辆网格模型", "Vehicle mesh models are required"))
            return False
        return True

    # ---- 操作 ----
    def _show_env(self):
        from lidarsim.roadnet import describe_sumo_env
        txt = (f"traci: {traci_message()}\n{describe_sumo_env()}\n"
               f"env points: {0 if self.env is None else len(self.env.points)}")
        QMessageBox.information(self, L("环境信息", "Environment"), txt)

    def _test(self):
        if self.test_worker is not None and self.test_worker.isRunning():
            return
        if not self._validate():
            return
        if not traci_available():
            self._set_light(0, traci_message())
            QMessageBox.critical(self, L("连接失败", "Connection failed"), traci_message())
            return
        self.btn_test.setEnabled(False)
        self.btn_start.setEnabled(False)
        self._set_light(1, L("正在测试连接...", "Testing connection..."))
        self._log(L("正在测试 SUMO 连接 (后台无界面运行)...",
                    "Testing SUMO connection in the background (headless)..."))
        self.test_worker = TestWorker(self._config(), self)
        self.test_worker.done.connect(self._on_test_done)
        self.test_worker.start()

    def _on_test_done(self, ok: bool, msg: str):
        self.btn_test.setEnabled(True)
        self.btn_start.setEnabled(True)
        self._log(msg)
        if ok:
            self._set_light(1, L("连接成功，等待启动", "Connected, waiting to start"))
            QMessageBox.information(self, L("连接成功", "Connected"), msg)
        else:
            self._set_light(0, msg)
            QMessageBox.critical(self, L("连接失败", "Connection failed"), msg)

    def _start(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self._sync_lidar_configs()
        if self.chk_lidars.isChecked() and not self._has_lidar_config():
            self._request_lidar_config()
            QMessageBox.information(
                self, L("配置路侧雷达", "Configure roadside LiDAR"),
                L("尚未检测到已配置的路侧雷达。已切换到主界面, 请在 LiDAR 配置中设置位置/姿态; "
                  "配置完成后本窗口会自动读取, 然后再次点击“开始联仿”。",
                  "No roadside LiDAR is configured. The main window is shown; set position/attitude "
                  "in the LiDAR panel. This dialog will pick it up automatically, then click Start again."))
            return
        if not self._validate():
            return
        if not traci_available():
            QMessageBox.critical(self, L("无法启动", "Cannot start"), traci_message())
            return
        cfg = self._config()
        self.worker = SumoWorker(cfg, self.env, self.models, self)
        self.worker.frame.connect(self._on_frame)
        self.worker.status.connect(self._on_status)
        self.worker.logmsg.connect(self._on_log)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()
        self._set_light(1, L("连接中...", "Connecting..."))
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_test.setEnabled(False)
        self._log(L("启动 SUMO 联仿...", "Starting SUMO co-simulation..."))

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()
            self._log(L("已请求停止", "Stop requested"))
            self.btn_stop.setEnabled(False)

    # ---- 信号 ----
    def _on_frame(self, payload):
        try:
            # 主界面视图与单独窗口是两种并列的可视化出口:
            # 只要任一开启就转发帧, 避免勾选“单独窗口”却因主界面视图未勾选而收不到数据
            if ((self.chk_viz.isChecked() or self.chk_sep_win.isChecked())
                    and self.on_frame is not None):
                self.on_frame(payload)
            self._viz_seen += 1
        finally:
            # 绘制完成 -> 放行联仿线程推进下一步 (SUMO 与三维可视化锁步同步)
            if self.worker is not None:
                self.worker.frame_ack()

    def _on_status(self, level: int, text: str):
        self._set_light(level, text)
        if self.on_status is not None:
            self.on_status(level, text)

    def _on_log(self, msg: str):
        self._log(msg)
        if self.on_log is not None:
            self.on_log(msg)

    def _on_failed(self, msg: str):
        self._set_light(0, L("联仿失败", "Simulation failed"))
        self._log(L(f"联仿失败: {msg}", f"Co-sim failed: {msg}"))
        QMessageBox.critical(self, L("联仿失败", "Co-simulation failed"), msg)
        if self.on_status is not None:
            self.on_status(0, msg)

    def _on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_test.setEnabled(True)
        self.worker = None

    def _set_light(self, level: int, text: str = ""):
        self.light.set_state(level, text)
        self.lbl_state.setText(text or self.light.toolTip())

    def _log(self, msg: str):
        self.log_text.append(msg)

    def closeEvent(self, ev):
        if self.test_worker is not None and self.test_worker.isRunning():
            if not self.test_worker.wait(12000):
                self._log(L("连接测试仍在进行，请稍候再关闭窗口",
                            "Connection test still running, please close later"))
                ev.ignore()
                return
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(8000):
                self._log(L("联仿正在停止，请稍候再关闭窗口",
                            "Co-simulation is stopping, please close later"))
                ev.ignore()
                return
        super().closeEvent(ev)

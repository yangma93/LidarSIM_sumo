# -*- coding: utf-8 -*-
"""传感器与仿真参数配置."""
from dataclasses import dataclass, field


@dataclass
class LidarSpec:
    """激光雷达参数 (机械旋转 / 转镜 / 非重复扫描).

    fov_h_deg=360 表示 360 度旋转扫描; 否则为以 yaw 为中心的扇形扫描.
    h_res_deg: 水平角分辨率 (每个发射列之间的方位角间隔).
    scan 时序: 每列光束的时间戳按方位角在 [t0, t0+T_scan] 内线性分布,
    由此在仿真中自然产生车辆运动伪影 (rolling shutter).
    """
    name: str = "vlp16"
    channels: int = 16
    vfov_min_deg: float = -15.0
    vfov_max_deg: float = 15.0
    fov_h_deg: float = 360.0
    h_res_deg: float = 0.2          # 水平分辨率
    rate_hz: float = 10.0           # 扫描帧率
    range_min: float = 0.5
    range_max: float = 120.0
    beam_divergence_mrad: float = 3.0   # 光束发散角 -> 束半径 = t*tan(div/2)
    noise_range_m: float = 0.02         # 测距高斯噪声 sigma
    noise_intensity: float = 4.0        # 强度高斯噪声 sigma (DN)
    alpha_atm: float = 2e-4             # 大气消光系数 1/m (双程 exp(-2*alpha*R))
    gain: float = 2.2                   # 强度增益 (DN 标定)
    range_ref: float = 35.0             # 近场压缩参考距离 R0: P ∝ 1/(1+(R/R0)^2)
    dropout_thr: float = 0.002          # 归一化接收功率低于该值时丢弃回波
    dropout_random: float = 0.005       # 随机漏检比例
    # 扫描模式: rotary=机械旋转/转镜 (方位列 x 通道网格);
    #           non_repetitive=非重复扫描 (Livox 类 rosette/Lissajous 图案).
    scan_mode: str = "rotary"
    # 非重复扫描点频 (点/秒); 单帧光束数 = points_per_sec / rate_hz.
    points_per_sec: int = 0

    @property
    def n_rays(self) -> int:
        if self.scan_mode != "rotary" and self.points_per_sec > 0:
            return max(1, int(round(self.points_per_sec /
                                     max(self.rate_hz, 1e-6))))
        steps = int(round(self.fov_h_deg / self.h_res_deg)) if self.fov_h_deg < 360 \
            else int(round(360.0 / self.h_res_deg))
        return max(steps, 1) * self.channels


LIDAR_PRESETS = {
    "vlp16":  dict(name="vlp16",  channels=16, vfov_min_deg=-15.0,  vfov_max_deg=15.0,  h_res_deg=0.2,  rate_hz=10.0, range_max=100.0),
    "hdl32":  dict(name="hdl32",  channels=32, vfov_min_deg=-30.67, vfov_max_deg=10.67, h_res_deg=0.33, rate_hz=10.0, range_max=100.0),
    "os1_64": dict(name="os1_64", channels=64, vfov_min_deg=-16.6,  vfov_max_deg=16.6,  h_res_deg=0.35, rate_hz=10.0, range_max=120.0),
    "pandar40": dict(name="pandar40", channels=40, vfov_min_deg=-25.0, vfov_max_deg=15.0, h_res_deg=0.2, rate_hz=10.0, range_max=200.0),
    # 128 线 (RoboSense RS-Ruby 级别); h_res 取 0.4° 兼顾单帧光束规模 (≈11.5 万束)
    "rs128":  dict(name="rs128",  channels=128, vfov_min_deg=-25.0, vfov_max_deg=15.0, h_res_deg=0.4, rate_hz=10.0, range_max=200.0),
    # ---- 车规级新增型号 (公开参数; h_res 为控制单帧光束规模做了适度稀疏) ----
    # 禾赛 ATX: 256 线, 120°x20°, 0.08°(H)x0.05°(V), 230m@10%, 3.84M pts/s.
    "hesai_atx": dict(name="hesai_atx", channels=256, vfov_min_deg=-10.0, vfov_max_deg=10.0,
                      fov_h_deg=120.0, h_res_deg=0.2, rate_hz=10.0, range_max=230.0,
                      beam_divergence_mrad=1.0, range_ref=45.0),
    # 禾赛 AT128: 128 线, 120°x25.4°, 0.1°(H)x0.2°(V), 200m@10%, 1.536M pts/s.
    "hesai_at128": dict(name="hesai_at128", channels=128, vfov_min_deg=-12.7, vfov_max_deg=12.7,
                        fov_h_deg=120.0, h_res_deg=0.2, rate_hz=10.0, range_max=200.0,
                        beam_divergence_mrad=1.0, range_ref=45.0),
    # 禾赛 OT128: 128 线机械旋转, 360°x40°(-25°~+15°), 0.1°(H)x0.125°(V), 200m@10%, 3.456M pts/s.
    "hesai_ot128": dict(name="hesai_ot128", channels=128, vfov_min_deg=-25.0, vfov_max_deg=15.0,
                        h_res_deg=0.4, rate_hz=10.0, range_max=200.0,
                        beam_divergence_mrad=1.0, range_ref=45.0),
    # 速腾聚创 EMX: 192 线, 120°x20°(最大水平 140°), 0.08°(H)x0.10°(V), 200m@10%, 2.88M pts/s, 最高 20Hz.
    "robosense_emx": dict(name="robosense_emx", channels=192, vfov_min_deg=-10.0, vfov_max_deg=10.0,
                          fov_h_deg=120.0, h_res_deg=0.25, rate_hz=10.0, range_max=200.0,
                          beam_divergence_mrad=1.0, range_ref=45.0),
    # 览沃 Livox Horizon: 非重复扫描, 81.7°x25.1°, 260m@80%, 240k pts/s (单回波), 10Hz.
    "livox_horizon": dict(name="livox_horizon", channels=64,
                          vfov_min_deg=-12.55, vfov_max_deg=12.55, fov_h_deg=81.7,
                          h_res_deg=0.4, rate_hz=10.0, range_max=260.0,
                          scan_mode="non_repetitive", points_per_sec=240_000,
                          beam_divergence_mrad=1.5, range_ref=50.0,
                          noise_range_m=0.02),
}


def normalize_preset(preset) -> str:
    """把配置里的型号字符串规范化 (去空白 / 小写), 未知型号原样返回."""
    return str(preset or "vlp16").strip().lower()


def make_lidar(preset: str = "vlp16", **overrides) -> LidarSpec:
    """按预设名构造 LidarSpec.

    型号名先做规范化 (strip + lower). 未知型号不再静默回退到 vlp16 —— 静默
    回退会让"车端配了 Livox 却按 vlp16 仿真"这种问题完全无迹可寻, 这里给出
    明确的告警行, 便于从 GUI/日志中定位配置错误.
    """
    key = normalize_preset(preset)
    if key not in LIDAR_PRESETS:
        print(f"[config] 未知雷达预设 '{preset}', 已回退 vlp16; "
              f"可用预设: {', '.join(LIDAR_PRESETS)}")
        key = "vlp16"
    kw = dict(LIDAR_PRESETS[key])
    kw.update(overrides)
    kw["name"] = overrides.get("name", key)
    return LidarSpec(**kw)


@dataclass
class SimParams:
    """环境/场景构建参数."""
    # 地面提取
    ground_cell: float = 1.5        # 地面高程网格 cell (m)
    ground_tol: float = 0.30        # 判定为地面点的竖向容差 (m)
    ground_pctl: float = 5.0        # 每 cell 取分位数作为局部最低面
    # 环境点云
    env_voxel: float = 0.10         # ray-casting 用环境点云体素降采样尺寸 (m)
    normal_voxel: float = 0.25      # 法向估计用的体素尺寸 (m)
    # ray-casting 加速网格
    grid_cell: float = 0.6          # 环境点 CSR 网格 cell (m)
    # GPU/CPU
    device: str = "auto"            # auto | gpu | cpu
    max_pairs_per_chunk: int = 120_000_000
    seed: int = 0
    # CAV 帧间环境求交上界: 用上一帧环境命中距离 t_prev 作为本帧 t_max 的参考,
    # 上界 = t_prev + gain*(|Δo| + t_prev*Δθ) + pad; 上一帧未命中/无历史时用 range_max.
    # 属于近似 (掠射束可能漏掉更远的命中), 默认关闭以保证逐束结果与原实现一致.
    cav_cap_enable: bool = False
    cav_cap_gain: float = 2.0       # 位移/角度项放大系数
    cav_cap_pad: float = 2.0        # 额外常数余量 (m)
    cav_cap_max_ang_deg: float = 2.0  # 单束指向变化超过该值时回落到 range_max
    # 强度物理模型
    refl_env_floor: float = 0.04
    refl_env_ceil: float = 0.85
    # 天气 (雨/雪), WeatherConfig 实例或 None; 见 weather.py
    weather: object = None

# LidarSIM 点云仿真加速改进说明（v1 → v4）

本文档逐项说明当前 `lidarsim` 包（下称 **v4**）相对于改进前基线（下称 **v1**）所做的全部改动。两条硬性约束贯穿所有改动：

1. **精度无损**：不改变光束集合、不改变求交判据、不改变随机数使用方式，逐帧输出的点云与 v1 逐元素一致；
2. **接口不变**：`SceneBuilder` / `LidarSimulator` / `SimParams` 的对外用法保持兼容，GUI 与上层实验脚本无需修改。

- v1：`%TEMP%\lidarsim_orig_20261118\{raycast,lidar_sim,config}.py`（2026-11-18 改进前的原始备份）
- v4：当前仓库 `lidarsim/`（在上一轮 A/B/C/D 优化之后再合并 B6/B7/C10/C11/D12/A2 的最终版本，其中 B7 因实测无收益且有浮点差异被弃用）

---

## 0. 改动总览

| 编号 | 模块 | 改动 | 目的 | 精度影响 |
|------|------|------|------|----------|
| A2 | `raycast.EnvGrid` / `raycast` | 扁平体素索引改 int32；新增单字节占用表 `occ`；占用判定只读 `offs` 的选中项 | 降低采样阶段的显存带宽与随机访存 | 无（判据同 `offs>0`） |
| A2b | `raycast.raycast` | 按逐束 `t_max` 升序重排后分块，块内采样步数 `S` 收缩到块内最长光线 | 采样点数从“全场最长光线”降到“本块最长光线” | 无（采样网格仍是 `{0,step,…}` 前缀，仅改变块的组成） |
| A2c | `raycast._reduce_nearest` | GPU 端 `atomicMin` 打包键 `(float位模式<<32)|点号` 逐束归约；失败回退 `lexsort+unique` | 替代对全部候选点的排序去重 | 无（取最小距离，等距时取点号小者，与排序去重结果相同） |
| A2d | `raycast.raycast` | 分块大小 4096 → 8192；`r_min` / `beam_tan` 支持逐束数组 | 减少 kernel 提交次数；为多雷达合批提供接口 | 无 |
| B6 | `raycast.ray_aabb_one` + `lidar_sim._vehicle_raycast` | 逐车 slab 测试，只把候选光线索引回传主机；取代 `(R,V)` 布尔矩阵 + 整矩阵 `asnumpy` | 车辆粗筛不再生成 `R×V` 大临时矩阵，也不再整表 D2H | 无（与 `ray_aabb` 单列判据逐元素相同） |
| B6b | `raycast.mt_hit` | 核函数 `r_min` 由标量改为逐光线数组 | 支持多雷达（不同 `range_min`）合批 | 无（同一标量时逐元素相同） |
| C10 | `lidar_sim._finalize_pattern` | 7 个主机数组一次 `stack` 打包回传，取代 7 次独立 D2H 同步 | 去掉每帧 6 次多余的设备同步 | 无（仅打包方式变化） |
| C11 | `lidar_sim._mesh_normals` | 三角面法向按网格名缓存（CPU） | 避免每帧重复 `cross/normalize` 数万三角面 | 无（纯缓存） |
| D12 | `lidar_sim._cav_rays` | CAV 零位姿扫描模式按 `spec` 缓存 | 避免每帧重建车端扫描方向/光束时刻 | 无（纯缓存） |
| E1 | `lidar_sim._simulate_cavs` | 多 CAV 合批：一次环境求交 + 一次车辆粗筛 + 每车一次三角求交 | 减少小 kernel 提交与重复 AABB 构建 | 无（光线集合与判据不变） |
| E2 | `lidar_sim._cav_env_cap` | 用上一帧环境深度作为本帧 `t_max` 上界（默认关闭） | 可选的车端环境求交加速 | **近似**（掠射束可能漏检），故默认 `False` |

> 曾尝试但未保留：B7（`mt_hit` 内直接 `atomicMin` 出逐束最近距离）。实测车辆点出现 ≤ 3.8×10⁻⁶ m（float32 1~2 ULP）的差异，而环境求交与整帧耗时均无改善，按“不折损精度”的验收标准整段删除。

---

## 1. 环境点云求交（`raycast.py`）

环境求交是单帧耗时的主要构成：每条光线沿距离以 `step = grid_cell * 0.5` 采样，命中体素后展开候选点做精确判据。v1 的实现有三处浪费：

1. 采样步数 `S` 由**全束最大** `range_max` 决定，所有光线都要走满 100~200 m；
2. 扁平体素索引用 int64，采样三轴与索引全程 64 位；
3. 每个采样点都读 `offsets[flat]`、`offsets[flat+1]` 判断占用，随机访存两次；
4. 命中归约用 `lexsort + unique` 对全部候选点排序。

### 1.1 块级采样上限收缩（A2b）

```python
# v1
S = int(np.ceil(float(t_max.max()) / step)) + 1
t_off = xp.arange(S) * step

# v4
order = np.argsort(tm, kind="stable")      # 逐束 t_max 升序
...
for s0 in range(0, R, chunk):
    S = int(np.ceil(float(tm_s[s1 - 1]) / step)) + 1   # 只覆盖本块最长光线
```

- 光线按 `t_max` 升序 **重排后再分块**，因此块内上确界就是块末元素；
- 采样网格仍是 `{0, step, 2step, …}` 的**前缀**，不会出现“半格偏移”，每条光线的采样点集合与 v1 完全相同；
- 环境求交中绝大多数光线的 `t_max = range_max`，但最后一块（以及车辆候选光线）会明显缩短，减少无效采样。

### 1.2 int32 扁平索引 + 单字节占用表（A2）

```python
i32 = self.ncell < 2 ** 31 - 1
flat = (c[..., 0] * fny + c[..., 1]) * fnz + c[..., 2]          # int32
...
sel = (ts >= RMIN) & (ts < TMAX)
sel = sel & (occ[flat] > 0)          # 只做一次字节比较
c_sel = offs[f_sel + 1] - offs[f_sel]  # 只对选中项读 offsets
```

- `nx*ny*nz < 2^31` 时扁平索引用 int32，采样阶段 3 个 int32 中间量代替 int64，显存带宽近似减半；
- 新增 `occ = (counts > 0).astype(uint8)`：占用判定从“两次 offsets 读 + 减法”降为“一次 uint8 读”，`offsets` 只对真正非空的采样点读取；
- 体素数超过 64M 时自动回落到 `offsets` 直接判定（`use_occ=False`），不改变结果。

### 1.3 GPU `atomicMin` 打包归约（A2c）

```cuda
// 一线程一个 (ray, point) 候选对
unsigned long long k = ((unsigned long long)__float_as_uint(along[i]) << 32)
                       | (unsigned long long)pt[i];
atomicMin(&out[ray[i]], k);
```

- 非负 float32 的位模式与数值同序，把“距离”放在高 32 位、“点号”放在低 32 位，`atomicMin` 一次得到 `(最小距离, 该距离下最小点号)`；
- 与 v1 的 `lexsort + unique` 结果逐束一致：距离相同则取点号小者；
- 核函数编译失败时自动回退到原有排序实现，CPU（NumPy）后端仍走原路径。

### 1.4 其它

- 分块 4096 → 8192，减少 kernel 提交与 Python 循环次数；
- `r_min` / `beam_tan` 允许逐束数组，为“多雷达一次求交”提供接口（同一雷达时等价于标量）；
- 候选点展开后按 `O[s0 + pair_ray]` 取光线，而不是索引整个 `O`，减少一次大数组 gather。

---

## 2. 车辆三角网格求交（`raycast.py` + `lidar_sim.py`）

### 2.1 逐车 slab 粗筛（B6）

```python
# v1: 一次生成 (R, V) 布尔矩阵, 再整表回传
cand = ray_aabb(o, d, bb_min, bb_max, t_cap)
cand_np = xp.asnumpy(cand) if self.gpu else cand
for vi, veh in enumerate(vehicles):
    ray_ids = np.flatnonzero(cand_np[:, vi])

# v4: 逐车单列 slab, 只回传候选索引
for vi, veh in enumerate(vehicles):
    m = ray_aabb_one(o, d, bb_min[vi], bb_max[vi], t_cap)
    ray_ids = xp.asnumpy(xp.flatnonzero(m)) if self.gpu else np.flatnonzero(m)
```

- `ray_aabb_one` 与 `ray_aabb` 的单列判据逐元素相同：`(tlo <= thi) & (thi > 0) & (tlo < t_cap)`；
- 不再生成 `R×V` 的 bool 中间矩阵，也不再每帧回传整张 `R×V` 表；只回传真正命中的光线索引（通常只占很小比例）。

### 2.2 逐束 `range_min`（B6b）

`mt_hit` 核函数签名由 `const float r_min` 改为 `const float* rmin`，判据 `tt > rmin[r]`。同一雷达传入常量数组时，结果与 v1 的标量版本逐位一致；不同量程雷达合批时才体现差异。

### 2.3 三角面法向缓存（C11）

```python
# v1: 每帧、每个车槽重新计算
tn = veh.mesh.tri_normals()[tri_np[m]]

# v4: 按网格名缓存
tn = self._mesh_normals(veh.mesh)[tri_np[m]]
```

网格模型在 `ModelLibrary` 中只加载一次，三角面法向对同一网格恒定不变，缓存不影响任何数值。

---

## 3. 帧级流程（`lidar_sim.py`）

### 3.1 单次打包 D2H（C10）

v1 在 `_finalize_pattern` 中依次对 `idx / t_hit[idx] / env_closer[idx] / pt_idx[idx] / veh_slot[idx] / veh_tri[idx] / veh_yaw[idx]` 做 7 次 `cupy.asnumpy`，每次都是一次隐式设备同步。v4 先 `stack` 成一个 int32 包，再一次性回传：

```python
blob = xp.asnumpy(xp.stack([
    idx.astype(xp.int32), t_hit[idx].view(xp.int32),
    env_closer[idx].astype(xp.int32), pt_idx[idx],
    veh_slot[idx], veh_tri[idx], veh_yaw[idx].view(xp.int32)]))
idx_np = blob[0].astype(np.int64)
th_np = blob[1].view(np.float32).astype(np.float64)
```

float32 用位模式 `view(int32)` 搬运，回主机后再 `view(float32)`，数值完全一致。

### 3.2 CAV 扫描模式缓存（D12）

```python
key = (spec.channels, spec.vfov_min_deg, spec.vfov_max_deg,
       spec.fov_h_deg, spec.h_res_deg)
pat = self._zero_pat_cache.get(key)
if pat is None:
    _, d0, tf_np = scan_pattern(spec, [0, 0, 0, 0, 0, 0])
    self._zero_pat_cache[key] = (d0, tf_np)
```

车端雷达每帧只需要“传感器体坐标系下的方向 + 光束相对时刻”，与车辆位姿无关；按传感器参数缓存后，每帧只做一次旋转/平移。

### 3.3 多 CAV 合批（E1）

v1 对每个 CAV 各调用一次 `_simulate_pattern`，即每个 CAV 一次环境求交、一次 `(R,V)` 粗筛、一次 `_finalize_pattern`。v4 的 `_simulate_cavs` 把本帧所有 CAV 的光线拼接：

1. 所有光线一次 `EnvGrid.raycast`；
2. 所有光线一次车辆粗筛（逐车 slab）；
3. 每辆车一次 `raycast_vehicle`（候选光线是全体 CAV 光线的子集）；
4. 按 CAV 切片，逐雷达做强度/输出。

光线集合、判据、随机数使用顺序均不变，因此逐雷达点云与逐 CAV 调用完全相同；差别只是减少了小 kernel 提交次数和重复的 AABB 构建。

### 3.4 CAV 环境求交上界（E2，默认关闭）

`_cav_env_cap` 用上一帧该束的环境命中距离 `t_prev` 作为本帧 `t_max`：

```
cap = t_prev + gain * (|Δo| + t_prev * Δθ) + pad
```

环境静态、车辆位姿逐帧连续，因此该上界在绝大多数光束上是安全的；但**掠射光束**可能在本帧命中比上界更远的点，属于近似。为保证逐束结果与 v1 完全一致，`SimParams.cav_cap_enable` 默认为 `False`，且未命中 / 指向变化超过 `cav_cap_max_ang_deg` 的光束自动回落到 `range_max`。

---

## 4. 精度等价性说明

| 改动 | 为什么逐元素一致 |
|------|------------------|
| 按 `t_max` 排序分块 | 只改变“哪些光线一起处理”，每条光线的采样点集合仍是 `{0, step, …}` 前缀；求交判据与取值不变 |
| int32 扁平索引 | `ncell < 2^31` 时 int32 与 int64 索引值相同，运算结果相同 |
| `occ` 占用表 | `occ[flat] > 0` ⇔ `offsets[flat+1] > offsets[flat]`，两判据同真同假 |
| `atomicMin` 打包键 | 非负 float 位模式保序；等距时低 32 位点号决出唯一结果，与排序去重一致 |
| 逐车 slab | 判据逐元素相同，仅去掉未命中列 |
| 逐束 `r_min` | 同一雷达时所有元素相等，等价于标量 |
| 单次打包 D2H | 只改传输方式，`view` 不改变位模式 |
| 法向 / 扫描模式缓存 | 缓存对象在场景生命周期内恒定 |
| 多 CAV 合批 | 光线并集相同、每车候选集相同、每束判据相同 |

实测校验（见第 5 节）：同一帧、同一随机种子下，v1 与 v4 输出点云的坐标数组 `max|Δ| = 0`，点数完全相同。

---

## 5. 实测性能

> 基准设置：`Results/scene_001..004` 各取 `traffic_01`，每个场景只挂 1 个静态雷达（scene_001/003 为 hdl32，scene_002/004 为 vlp16，位姿取自各自的 `lidar_pose_static.json`）；每场景随机抽取 500 帧；两版共用同一 `SceneBuilder` 与同一随机种子；`SimParams` 默认参数（`grid_cell=0.6`、GPU 后端）。
>
> 计时口径：**cold** —— 每帧仿真前清空路侧雷达的环境深度图缓存，计时覆盖“环境点云求交 + 车辆三角面求交 + 强度计算”的完整单帧流程。v1 与 v4 都带有该缓存，这里统一清空是为了比较求交算法本身，而不是比较场景级的跨帧复用。

### 5.1 测试环境

| 项目 | 配置 |
|------|------|
| CPU / GPU | RTX 5070 Laptop |
| Python / CuPy / NumPy | 3.14.6 / 14.2.0 / 2.5.4 |
| 场景 | `scene_001 ~ scene_004` 的 `traffic_01`（whu 城市点云，环境点 100 万 ~ 173 万） |
| 雷达 | 每个场景使用 1 个静态雷达：scene_001/003 = hdl32（静态位姿 1），scene_002/004 = vlp16（静态位姿 2） |
| 帧采样 | 每场景从 21 600 帧中随机抽取 500 帧（前 8 帧用于预热，不参与统计），两版共用同一批帧与同一随机种子 |
| 环境网格 | `grid_cell = 0.6 m`，`env_voxel = 0.10 m`，GPU 后端 |

### 5.2 单帧平均耗时（cold 口径）

| 场景 | 雷达 | v1 / ms | v4 / ms | 加速比 |
|------|------|---------|---------|--------|
| scene_001 | hdl32 | 73.52 ± 0.66 | 44.97 ± 0.53 | **1.63×** |
| scene_002 | vlp16 | 49.23 ± 0.74 | 34.17 ± 0.63 | **1.44×** |
| scene_003 | hdl32 | 89.45 ± 1.07 | 67.07 ± 1.12 | **1.33×** |
| scene_004 | vlp16 | 62.66 ± 0.80 | 50.31 ± 0.86 | **1.25×** |
| 合计 | — | 68.72 | 49.13 | **1.40×** |

（每场景 n = 492 帧；表中 ± 为 95% 置信区间；图中误差棒为 ±1 标准差，以便直观看到帧间波动）

### 5.3 规律

环境点云更密、三角面负担更重的场景（scene_003/004）中，两版共有的车辆求交耗时占比上升，环境求交的固定收益被稀释，加速比由 1.6× 收敛到 1.25×。

### 5.4 精度校验

每个场景的前 3 帧做了逐点比对：v1 与 v4 输出点云的坐标数组形状相同、`max|Δ| = 0.000000 m`，点数完全相同（4 个场景共 12 组帧对全部为 0）。

### 5.5 与求解三角面数量的关系

按每帧实际参与 Möller–Trumbore 求解的 `(光线, 三角面)` 对数分箱统计：

| 光线-三角面对 | v1 / ms | v4 / ms |
|---------------|---------|---------|
| < 0.3 M | 63.1 | 40.5 |
| 0.3 ~ 1 M | 65.0 | 49.6 |
| 1 ~ 2 M | 81.6 | 62.4 |
| 2 ~ 3 M | 88.9 | 67.1 |
| 3 ~ 4 M | 95.4 | 71.4 |
| 4 ~ 6 M | 93.4 | 71.2 |

两条曲线都随三角面负担增加而上升（v1 约 63 → 95 ms，v4 约 41 → 71 ms），v4 全程保持约 20 ~ 24 ms 的优势。不同场景的环境点云规模与雷达型号不同，曲线的绝对水平同时包含环境求交的固定开销，因此该图用于比较**同一难度下两种方法的耗时**，而不是单纯的环境开销。

### 5.6 适用边界：路侧雷达热路径

若不清空环境深度图缓存（同一静态雷达连续回放），单帧只剩“车辆求交 + 强度”：

| 场景 | v1 / ms | v4 / ms |
|------|---------|---------|
| scene_001 | 9.3 | 11.7 |
| scene_002 | 10.0 | 13.7 |
| scene_003 | 25.7 | 32.9 |
| scene_004 | 21.8 | 29.7 |

此时 v4 反而每帧慢约 2.4 ~ 7.2 ms。原因是 B6 的逐车 slab 需要为每辆车单独回传候选光线索引（每帧约 20~30 次小 D2H），而 v1 是一次性回传整张 `(R,V)` 表。该差异只影响“环境深度图已缓存”的纯热路径；完整单帧路径（环境求交占主导）仍是 v4 明显更快。若后续只针对路侧静态雷达回放，可考虑把逐车候选合并成一次回传以消除这部分开销。

### 5.7 图件

- `figures/fig_v1_v4_runtime_bar.png`：4 个场景的单帧平均耗时（柱状图，横轴为 scene_001 ~ scene_004，误差棒为 ±1 标准差）；
- `figures/fig_v1_v4_tri_pairs.png`：单帧耗时随光线-三角面对数量的变化（折线 + 95% 置信带）。

---

## 6. 复现方式

- 基准脚本：`%TEMP%\lidarsim_bench\bench_one.py`（运行时以 `lidarsim_v1` / `lidarsim_v4` 两个包副本并存，仅替换 `raycast.py`、`lidar_sim.py`、`config.py`）；
- 逐帧结果：`%TEMP%\lidarsim_bench\bench_results_one.csv`；
- 车辆三角面负担：以模块级 `raycast_vehicle` 包装器统计每帧实际求解的 `(光线, 三角面)` 对数，不修改算法本身。

---

## 7. 结论

- v4 在不改变任何一帧点云的前提下，降低了环境求交的采样量、显存带宽与归约开销，并去掉了车辆粗筛的全矩阵回传与 `_finalize_pattern` 的多次同步；
- 改进集中在**完整单帧仿真**路径（环境求交占比最大），对“路侧雷达环境深度图已缓存”的纯热路径收益有限；
- 所有改动均可在 CPU（NumPy）后端回退，未被 GPU 强绑定。

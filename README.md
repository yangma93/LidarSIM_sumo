# LidarSIM — Multi-LiDAR Simulation & Evaluation for Vehicle–Road Cooperative Perception

**Physics-based LiDAR simulation on real environment point clouds, with SUMO co-simulation and a built-in cooperative-perception evaluation pipeline.**

LidarSIM renders roadside and on-board LiDAR point clouds against real road environment
point clouds. It supports ten sensor presets, physically based intensity, rolling-shutter
motion artefacts, rain/snow clutter, multi-LiDAR placement comparison, closed-loop SUMO
co-simulation, L-shape cooperative evaluation, probabilistic occupancy grids and OpenCOOD
annotation export.

<p align="center">
  <a href="html/index.html"><img src="html/assets/Lidar%20sim.gif" width="760" alt="LidarSIM point-cloud simulation"></a>
</p>

<p align="center">
  <a href="html/index.html">Project page</a> ·
  <a href="#features">Features</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#citation">Citation</a>
</p>

---

## Overview

Roadside LiDARs complement on-board sensors at intersections and curves, while on-board
LiDARs provide dense observations of the ego neighbourhood. Evaluating perception
algorithms for such vehicle–road systems requires point clouds that respect real scene
geometry, sensor mounting, scanning order and weather. LidarSIM provides this simulation
stack together with the evaluation and data-export utilities used in our studies.

The simulator shares one kernel for offline replay and online SUMO co-simulation:

```
trajectories ──► SceneBuilder ──► LidarSimulator ──► SimCloud[]
environment cloud ──► (optional)      │
                                      ├─ L-shape fitting + 3D IoU / Dev / Comp
                                      ├─ probabilistic occupancy grid (POG)
                                      └─ OpenCOOD (OPV2V-style) export
```

## Features

| Area | Capability |
|---|---|
| Sensors | 10 presets: mechanical spinning, rotating-mirror and non-repetitive scanning; 100–260 m range |
| Deployment | Arbitrary roadside poses and per-vehicle CAV sensors with local mounting offsets |
| Physics | Reflectance × incidence intensity model, atmospheric extinction, beam divergence, dropout |
| Motion | Per-beam timestamps and rolling-shutter artefacts under dynamic vehicle poses |
| Weather | Rain / snow particle spectra with extinction, backscatter and labelled clutter returns |
| Sumo | Lock-step TraCI co-simulation with IoU-driven colour and parameter feedback |
| Evaluation | L-shape detection, 3D IoU / BEV IoU / Dev / Comp, per-vType statistics, CSV export |
| Data | POG (NPZ / MAT) and OpenCOOD-format datasets (PCD + YAML) |
| Acceleration | v1 → v4 kernel optimisation: **1.40×** mean speed-up with bit-identical clouds |

## Repository layout

```
LidarSIM_sumo/
├── src/
│   ├── lidarsim/                  # Pure-Python simulation kernel (no Qt imports)
│   │   ├── config.py              #   10 LiDAR presets and simulation parameters
│   │   ├── trajectory.py          #   Vissim / SUMO FCD trajectories
│   │   ├── traffic_live.py        #   online SUMO trajectory container
│   │   ├── pointcloud.py          #   environment clouds, ground model, road-network points
│   │   ├── meshes.py              #   vehicle mesh library and vType matching
│   │   ├── scene.py               #   frame-level scene reconstruction
│   │   ├── scan.py / raycast.py   #   beam generation and GPU/CPU ray casting
│   │   ├── lidar_sim.py           #   multi-LiDAR frame simulation
│   │   ├── weather.py             #   rain / snow extinction and clutter
│   │   ├── lshape.py              #   L-shape fitting and 3D IoU evaluation
│   │   ├── pog.py                 #   probabilistic occupancy grid
│   │   ├── opencood_export.py     #   OpenCOOD (OPV2V-style) export
│   │   ├── roadnet.py             #   SUMO road-network model and net.xml I/O
│   │   ├── sumo_cosim.py          #   TraCI bridge and co-simulation loop
│   │   ├── cav.py, cav_config.py  #   CAV sensor descriptions and complex configurations
│   │   └── conflict.py, viz.py    #   conflict / TTC analysis and visualisation
│   ├── gui.py                     # PyQt6 main window
│   ├── gui_sumo.py                # SUMO co-simulation panel
│   ├── gui_cav_config.py          # CAV multi-sensor configuration window
│   ├── gui_roadnet.py             # point-cloud basemap / road-network editor
│   └── simulate.py                # command-line entry point
├── html/
│   ├── index.html                 # Bilingual (中文 / EN) project page
│   └── assets/                    # Images and videos used by the page
├── requirements.txt
├── pyproject.toml
├── README.md
├── LICENSE
└── .github/workflows/pages.yml    # GitHub Pages deployment of html/
```

## Installation

Python 3.10 or newer is required. A CUDA-capable GPU is strongly recommended; the
ray-casting kernel falls back to CPU (NumPy) when CuPy is unavailable.

```bash
git clone <repository-url>
cd LidarSIM_sumo
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

Optional GPU acceleration (choose the wheel matching `nvidia-smi`):

```bash
pip install cupy-cuda12x      # CUDA 12.x
# pip install cupy-cuda11x    # CUDA 11.x
```

Optional SUMO co-simulation: install Eclipse SUMO (`sumo`, `sumo-gui`, `netconvert`)
and the TraCI Python package:

```bash
pip install eclipse-sumo
```

> The STL vehicle mesh library is **not** bundled with this repository. Place your
> `agent mesh models/` folder next to `src/` or pass its path with `--models-dir`.

## Data preparation

| Input | Format | Notes |
|---|---|---|
| Traffic trajectories | Vissim `.txt` (15 columns) or SUMO FCD `.xml` | Loaded by `lidarsim.trajectory` |
| Environment point cloud | `.ply` (x/y/z, optional RGB/semantic) or `.h5` | Optional; `net.xml` can generate road-surface points |
| Vehicle meshes | `.stl` / `.ply` / `.obj` | One mesh per vehicle class or vType |
| SUMO scenario | `.sumocfg` + `net.xml` | Required only for co-simulation |

All paths are passed explicitly on the command line or through the GUI, so no global
environment variables are needed.

## Quick start

### Graphical interface

```bash
cd src
python gui.py
```

The interface provides data loading, scene and point-cloud rendering, the SUMO basemap
editor, CAV sensor configuration and the SUMO co-simulation panel.

### Command line

```bash
# 1) Render one fused scene
python simulate.py scene --traj data.txt --env env.ply \
    --models-dir "agent mesh models" --t 10.0 --save scene.png

# 2) Simulate multi-LiDAR point clouds with ground truth
python simulate.py lidar --traj data.txt --env env.ply --t 10.0 \
    --lidar "0,0,6,0,0,0:vlp16" --cav-n 2 --out cloud.npz --ply cloud.ply

# 3) Build a probabilistic occupancy grid
python simulate.py pog --traj data.txt --env env.ply --t1 0 --t2 20 \
    --lidar "0,0,6,0,0,0:vlp16" --delta 0.4 --thr 0.7 --out pog --mat pog

# 4) Export an OpenCOOD training set
python simulate.py opencood --traj data.txt --env env.ply \
    --lidar "0,0,6,0,0,0:vlp16" --cav-n 3 --i1 0 --i2 100 --out opencood_out
```

### SUMO co-simulation

1. Open `src/gui.py`.
2. Choose **SUMO 联仿** in the tools menu, or import `net.xml` from the file menu when no
   background point cloud is available.
3. Select the SUMO scenario, roadside LiDARs, CAV mode (none / ids / penetration /
   complex configuration) and the feedback threshold.

Each STEP advances SUMO, simulates the configured sensors, fits L-shaped targets, computes
cooperative 3D IoU, and writes the result back to SUMO as green / red vehicle colours plus
`lidarsim.iou` and `lidarsim.perceived` custom parameters. Rendering is lock-step: SUMO
advances only after the frame has been drawn.

## Performance

The v4 kernel keeps every point cloud element-wise identical to v1 (`max|Δ| = 0`) while
reducing environment sampling, memory traffic and reduction overhead:

| Scene | LiDAR | v1 / ms | v4 / ms | Speed-up |
|---|---|---|---|---|
| scene_001 | hdl32 | 73.52 | 44.97 | 1.63× |
| scene_002 | vlp16 | 49.23 | 34.17 | 1.44× |
| scene_003 | hdl32 | 89.45 | 67.07 | 1.33× |
| scene_004 | vlp16 | 62.66 | 50.31 | 1.25× |
| **Total** | — | **68.72** | **49.13** | **1.40×** |

See [`src/lidarsim/LidarSIM_improvement.md`](src/lidarsim/LidarSIM_improvement.md) for the
per-change rationale and equivalence argument.

## Project page

The black, Apple-inspired bilingual project page is [`html/index.html`](html/index.html).
It is fully static: all images and videos live in `html/assets/`. To publish it, enable
GitHub Pages with **Source: GitHub Actions**; `.github/workflows/pages.yml` uploads the
`html/` folder on every push to `main`.

## Citation

If you use LidarSIM in your research, please cite the related publications:

1. *IEEE Transactions on Intelligent Transportation Systems*, 2023.
   DOI: [10.1109/TITS.2023.3286384](https://doi.org/10.1109/TITS.2023.3286384)
2. *Computer-Aided Civil and Infrastructure Engineering*, 2024.
   DOI: [10.1111/mice.13167](https://doi.org/10.1111/mice.13167)
3. *Transportation Research Part C*, 2026.
   DOI: [10.1016/j.trc.2025.105473](https://doi.org/10.1016/j.trc.2025.105473)

## Contact

* Yang Ma — mayang@hfut.edu.cn
* Hefei University of Technology

## License

Released under the [MIT License](LICENSE). Third-party datasets and vehicle mesh models
remain subject to their original licenses.

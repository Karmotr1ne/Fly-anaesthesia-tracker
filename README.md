# FATer — Fly Anaesthesia Tracker

果蝇麻醉实验的桌面采集与分析工作台，整合实验协议、相机录像、气体流量控制、轨迹追踪和行为评估，也支持离线分析已有视频。

A desktop workbench for fruit fly anaesthesia experiments, integrating protocols, video acquisition, gas-flow control, trajectory tracking and behavioural evaluation. Existing videos can also be analysed offline.

## v1.0 改进 / What’s new

相较旧版 `python_suite`，本版将视频分析扩展为完整实验流程：

Compared with the earlier `python_suite`, v1.0 extends video analysis into an integrated experimental workflow:

| 功能 / Feature | 改进 / Improvement |
| --- | --- |
| 采集控制 / Acquisition | Basler 相机预览、PFS 配置、独立录像、同步实验及分阶段 MFC 控制。 / Basler preview, PFS settings, video-only recording, synchronized experiments and staged MFC control. |
| 实验归档 / Session management | 按实验保存视频、遥测、协议、相机配置及分析参数。 / Stores video, telemetry, protocols, camera settings and analysis parameters per session. |
| 识别追踪 / Tracking | 在背景差分基础上增加面积、形状和边缘评分，以及逐舱室丢失重捕获；支持批任务、标定复用和背景缓存。 / Adds area, shape and edge scoring plus per-chamber reacquisition to background subtraction; supports batch tasks, calibration reuse and background caching. |
| 评估导出 / Evaluation | 区分原始检测与清洗轨迹；GUI 与 CLI 共用评估服务，支持汇总导出及临时文件提交保护。 / Separates raw detections from cleaned trajectories; shares evaluation between GUI and CLI, with summary export and staged output commits. |

采用经典计算机视觉，无需下载深度学习模型；追踪流程按每舱室一个目标设计。以上为实现变化，未作准确率或性能提升的量化声明。

Uses classical computer vision without model downloads, with one target per chamber. These are implementation changes, not measured accuracy or performance claims.

## 安装与启动 / Install & launch

需要支持当前依赖的 Python 环境；源码语法要求 Python 3.10+，完整版本兼容性尚未验证。以下命令在 Windows PowerShell 的项目根目录执行。

Use a Python environment supported by the dependencies. Source syntax requires Python 3.10+; a full compatibility matrix has not been validated. Run these commands from the project root in Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install PyQt5 numpy pandas scipy opencv-python matplotlib

# 离线分析 / Offline analysis
.\.venv\Scripts\python.exe main.py --no-hardware
```

新版使用 **PyQt5**。`--no-hardware` 跳过启动时的硬件连接，不生成模拟数据。实际采集另需安装相机驱动及硬件接口：

The current GUI uses **PyQt5**. `--no-hardware` skips hardware connections at startup; it does not generate simulated data. Acquisition additionally requires camera drivers and hardware interfaces:

```powershell
.\.venv\Scripts\python.exe -m pip install pyserial pypylon
.\.venv\Scripts\python.exe main.py
```

## 基础配置 / Basic configuration

| 项目 / Item | 默认值与设置 / Defaults & setup |
| --- | --- |
| 相机 / Camera | Basler + pylon 驱动及 `pypylon`；先确认相机可被 pylon 工具识别。 / Basler with pylon drivers and `pypylon`; verify detection in pylon tools first. |
| PFS | 在 Acquisition → Browse 选择适合相机的 `.pfs`；默认引用的 PFS 未随源码提供。 / Select a camera-compatible `.pfs` via Acquisition → Browse; the default referenced file is not bundled. |
| 帧率 / Frame rate | 录像默认 30 fps；核对相机实际输出、录像和分析 FPS。 / Recording defaults to 30 fps; verify camera, recording and analysis rates. |
| 串口 / Serial | 默认 `COM3`、19200 baud、8N1；按实际端口修改并重连。 / Defaults to `COM3`, 19200 baud, 8N1; select the actual port and reconnect. |
| MFC | A = O2，B = Anaesthetics，C = N2；单位 sccm，设定精度 0.1。B 的默认气体标签为 CO2，须核对实际接气和标定。 / A = O2, B = Anaesthetics, C = N2; sccm with 0.1 setpoint precision. B defaults to a CO2 label; verify gas connections and calibration. |
| 协议 / Protocol | 在 Protocol 页面建立或导入 `.fv`，核对阶段时长、流量和混合气体起始时间。 / Create or import `.fv` protocols; check stage durations, flows and mix-gas onset. |
| 数据 / Storage | 默认写入 `workspace/sessions/`；确保空间充足并单独备份，该目录被 Git 忽略。 / Defaults to `workspace/sessions/`; allow sufficient disk space and back up separately, as Git ignores this directory. |

硬件默认值见 `core/protocol/flowvision_model.py`，MFC 地址映射见 `core/hardware/protocol_scheduler.py`。当前串口实现使用字母地址和 ASCII 指令，其他型号需确认协议兼容。

Hardware defaults are in `core/protocol/flowvision_model.py`; MFC mappings are in `core/hardware/protocol_scheduler.py`. The serial implementation uses letter addresses and ASCII commands; verify compatibility with other devices.

## 快速上手 / Quick start

### 已有视频 / Existing videos

1. **导入 / Import** — 以 `--no-hardware` 启动，在 Tracking 导入视频并确认预览、尺寸和 FPS。 / Launch with `--no-hardware`, import videos in Tracking and verify preview, dimensions and FPS.
2. **标定 / Calibrate** — 设置 Rows、Columns 和编号顺序，框选首个舱室后使用 Auto-Snap Grid，再检查所有 ROI。 / Set rows, columns and numbering order; mark the first chamber, use Auto-Snap Grid and inspect all ROIs.
3. **检查识别 / Inspect detection** — 查看背景和识别预览。默认从前 0%–30% 区间抽取 60 帧；若静止目标进入背景，调整区间并 Rebuild Background。 / Inspect background and detection previews. The default samples 60 frames from the first 0%–30%; adjust the interval and rebuild if stationary targets enter the background.
4. **追踪 / Track** — 勾选任务并点击 Track，检查轨迹后，将生成的 Position CSV 加入 Evaluation。 / Select tasks and click Track; inspect trajectories, then load the generated Position CSV into Evaluation.
5. **评估 / Evaluate** — 核对气体起始时间与行为阈值，点击 Run Evaluation on Selected，按需导出图表和汇总。 / Check gas onset and behavioural thresholds, run evaluation and export plots or summaries as needed.

### 完整采集 / Full acquisition

1. **准备 / Prepare** — 正常启动，在 Protocol 配置阶段；在 Acquisition 选择 PFS、确认串口及设备状态。 / Launch normally, configure stages in Protocol, then select PFS and verify serial and device status in Acquisition.
2. **运行 / Run** — 选择 Record Video Only 或 Start Synchronized Experiment，观察画面、流量和日志。 / Choose Record Video Only or Start Synchronized Experiment; monitor video, flows and logs.
3. **结束 / Finish** — 点击 Stop Experiment，确认视频与日志保存；应用会尝试将视频加入 Tracking，继续标定和评估。 / Stop the experiment and verify saved video and logs; the application attempts to queue the recording in Tracking for calibration and evaluation.

单独录像需要相机就绪，同步实验还需要串口就绪。协议暂停保留当前流量，仅冻结计时，并不关闭供气。

Video-only recording requires a ready camera; synchronized experiments also require a ready serial connection. Pausing a protocol freezes its timer while retaining the current flow; it does not shut off gas.

## 数据与分析 / Data & analysis

```text
workspace/sessions/Session_<timestamp>_<flow>Anes_<duration>s/
├── raw/       # 原始视频 / Original video
├── logs/      # 遥测、协议、相机与分析配置 / Telemetry, protocol, camera & analysis settings
└── analysis/  # 检测、轨迹、评估、汇总与可选图表 / Detections, trajectories, evaluation, summaries & optional plots
```

- **输入 / Input** — `*_raw.csv` 是原始检测，`*_position.csv` 是清洗轨迹，评估优先使用后者。 / `*_raw.csv` contains raw detections; `*_position.csv` contains cleaned trajectories and is the preferred evaluation input.
- **输出 / Output** — `*_evaluated.csv` 为逐帧评估，`*_summary.csv` 为汇总；追踪和评估参数保存在 `logs/`。 / `*_evaluated.csv` stores frame-level evaluation; `*_summary.csv` stores summaries. Tracking and evaluation settings are saved in `logs/`.
- **单位 / Units** — 坐标为像素，速度为 **px/s**，高度阈值为归一化值；不能直接把速度当作 mm/s。 / Coordinates are in pixels, speed in **px/s**, and height thresholds are normalized; speed is not calibrated to mm/s.
- **阈值 / Thresholds** — 评估引擎默认气体起始 300 s、静止窗口 120 s、静止速度 0.1 px/s；应按实际实验调整。缺失检测不等于静止。 / Engine defaults are 300 s gas onset, a 120 s stillness window and 0.1 px/s stillness speed; adjust to the experiment. Missing detections do not imply stillness.

CLI 与桌面端使用同一评估服务 / CLI and GUI share the evaluation service:

```powershell
.\.venv\Scripts\python.exe -m core.analysis.cli --help
.\.venv\Scripts\python.exe -m core.analysis.cli --csv "D:\data\example_position.csv" --fps 30 --gas-onset 300
.\.venv\Scripts\python.exe -m core.analysis.cli --dir "D:\data\sessions" --summary-only
```

## 常见问题 / Troubleshooting

| 问题 / Issue | 检查 / Check |
| --- | --- |
| 缺少依赖 / Missing imports | 在 VS Code 选择 `.venv` 解释器，并用同一解释器安装依赖。 / Select the `.venv` interpreter in VS Code and install dependencies with that interpreter. |
| 相机或串口离线 / Hardware offline | 检查驱动、PFS、COM、设备地址及其他程序占用；纯分析使用 `--no-hardware`。 / Check drivers, PFS, COM, device addresses and competing applications; use `--no-hardware` for offline analysis. |
| 误识别或丢失 / False or missing detections | 检查 ROI、光照、背景采样和预览，再调整参数。 / Inspect ROIs, lighting, background sampling and previews before tuning parameters. |
| Raw 未清洗 / Uncleaned Raw input | 先运行 Tracking，再使用 Position CSV 评估。 / Run Tracking first and evaluate the Position CSV. |

文档依据源码整理，尚未进行完整硬件联调；正式实验前请检查代表性视频与输出结果。

This guide is based on source inspection, without full hardware validation. Inspect representative videos and outputs before running a full experimental batch.

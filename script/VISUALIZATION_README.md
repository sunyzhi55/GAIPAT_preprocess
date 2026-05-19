# GAIPAT 任务可视化脚本

## 脚本名称
`visualize_task_gazepoints.py`

## 功能
将 GAIPAT 实验中的积木指令和参与者眼动数据可视化为视频。

### 可视化内容
- **蓝色实线**：积木起始位置（origin）
- **红色虚线**：积木目标位置（destination）
- **绿色散点**：参与者在桌面上的注视点（gaze points），随时间累积显示

## 基本用法

```bash
python visualize_task_gazepoints.py \
    --task car \
    --subject-id 69907732 \
    --output output.mp4
```

## 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--repo-root` | 否 | 脚本所在目录 | 仓库根目录 |
| `--task` | 是 | - | 任务类型：`car`, `tb`, `house`, `sc`, `tc`, `tsb` |
| `--subject-id` | 是 | - | 参与者 ID |
| `--output` | 是 | - | 输出视频文件路径 |
| `--fps` | 否 | 30 | 视频帧率（帧/秒） |
| `--duration` | 否 | 无限制 | 最大视频时长（秒），默认使用全部数据 |

## 示例

### 示例 1：生成 car 任务的 30 秒视频
```bash
python visualize_task_gazepoints.py \
    --repo-root d:\code\gaipat \
    --task car \
    --subject-id 69907732 \
    --output d:\code\gaipat\processed\car_viz.mp4 \
    --duration 30
```

### 示例 2：生成完整的 house 任务视频（所有数据）
```bash
python visualize_task_gazepoints.py \
    --task house \
    --subject-id 69907732 \
    --output house_viz.mp4
```

### 示例 3：生成 24 FPS 的低帧率视频
```bash
python visualize_task_gazepoints.py \
    --task tc \
    --subject-id 69907732 \
    --output tc_viz.mp4 \
    --fps 24
```

## 输入数据

脚本需要以下输入文件：

1. **指令文件** (`setup/instructions_{task}.csv`)
   - 包含每个积木的起始和目标位置信息

2. **注视点数据** (`participants/{subject_id}/{task}/table/gazepoints.csv`)
   - 包含时间戳和眼动坐标 (x, y)

## 输出

- 生成一个 MP4 视频文件，包含：
  - 积木位置的矢量图形
  - 随时间推进的注视点轨迹
  - 实时时间戳显示
  - 累积注视点数显示

## 技术细节

- **坐标系**：相对坐标（0-1 范围），对应桌面工作区
- **分辨率**：800×800 像素（根据第一帧自动确定）
- **编码**：MP4v 或 MJPEG（根据系统支持自动选择）
- **颜色编码**：
  - 起始位置：蓝色
  - 目标位置：红色
  - 注视点：绿色

## 依赖

- Python 3.9+
- opencv-python (`cv2`)
- matplotlib
- numpy
- PIL

## 常见问题

### Q: 视频文件很大
A: 可以通过 `--duration` 参数限制时长，或使用外部工具进行压缩。

### Q: 脚本运行很慢
A: 脚本需要为每一帧绘制图表，这可能需要一些时间。对于长时间的视频，请耐心等待。可以尝试减少帧率 `--fps` 来加快速度。

### Q: 视频中没有注视点显示
A: 检查 `participants/{subject_id}/{task}/table/gazepoints.csv` 文件是否存在且有效数据。某些参与者或任务可能没有眼动数据记录。

## 输出示例

已生成的示例视频：
- `processed/visualization_69907732_car.mp4` (2.93 MB, 20 秒)
- `processed/visualization_69907732_house.mp4` (2.28 MB, 15 秒)

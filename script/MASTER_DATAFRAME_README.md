# GAIPAT Master DataFrame Pipeline - 使用文档

## 脚本

`gaipat_master_dataframe_pipeline.py` - 完整的多模态数据融合流水线

## 概述

该脚本实现了需求规范中的核心三大逻辑：

1. **逻辑 1：时间戳并集对齐（Logic 1: Union Timeline Alignment）**
   - 合并 table 和 screen 注视点的时间戳
   - 保留 NaN 值，不进行坐标插值
   - 对离散控制信号（event）进行前向填充

2. **逻辑 2：认知焦点状态机（Logic 2: Active Target FSM）**
   - 生成 `active_target_type` 列，取值 ['SCREEN', 'SRT', 'DST', 'BLINK']
   - SCREEN：屏幕注视有效
   - SRT：桌面注视有效（寻找积木阶段）
   - DST：拿起积木后（对准装配阶段）
   - BLINK：两个注视都无效

3. **逻辑 3：坐标单位转换和物理对齐**
   - 相对坐标 → 物理厘米单位
   - 生成 block 的起始和目标位置坐标

## 使用方式

### 处理单个参与者

```bash
python gaipat_master_dataframe_pipeline.py \
    --repo-root d:\code\gaipat \
    --subject-id 69907732
```

### 处理所有参与者
```bash
python script/gaipat_master_dataframe_pipeline.py --repo-root /root/autodl-tmp/shenxy/XDU/Dataset/gaipat/
```

```bash
python gaipat_master_dataframe_pipeline.py \
    --repo-root d:\code\gaipat
```

### 指定输出目录

```bash
python gaipat_master_dataframe_pipeline.py \
    --repo-root d:\code\gaipat \
    --output-dir d:\code\gaipat\my_output
```

## 输出数据格式

### 输出位置

```
processed/master_dataframes/{subject_id}/{task}_master.csv
```

### 列定义

| 列名 | 类型 | 说明 |
|------|------|------|
| `timestamp` | int64 | 时间戳（毫秒，并集后） |
| `subject_id` | str | 参与者 ID |
| `task` | str | 任务名称 (car, house, sc, tb, tc, tsb) |
| `event` | str | 事件类型（前向填充） |
| `block_id` | int16 | 当前操作的积木 ID |
| `active_target_type` | category | 认知焦点 (SCREEN/SRT/DST/BLINK) |
| `step_id` | int | 指令步骤 ID |

#### 注视点数据

| 列名 | 单位 | 说明 |
|------|------|------|
| `gaze_x_table_rel`, `gaze_y_table_rel` | [0,1] | 桌面相对坐标 |
| `gaze_x_table_cm`, `gaze_y_table_cm` | cm | 桌面物理坐标 |
| `gaze_x_screen_rel`, `gaze_y_screen_rel` | [0,1] | 屏幕相对坐标 |
| `gaze_x_screen_cm`, `gaze_y_screen_cm` | cm | 屏幕物理坐标 |
| `gaze_x_screen_px`, `gaze_y_screen_px` | px | 屏幕像素坐标 |

#### 目标积木坐标

每个积木的 4 个角落（0=左上, 1=右上, 2=右下, 3=左下）

| 列名 | 说明 |
|------|------|
| `target_x_srt0~3`, `target_y_srt0~3` | 积木起始位置（Source） |
| `target_x_dst0~3`, `target_y_dst0~3` | 积木目标位置（Destination） |

## 核心实现特性

### ✅ 已实现

- [x] 时间戳并集合并
- [x] 坐标无插值处理（保留 NaN）
- [x] 离散信号前向填充
- [x] 多模态坐标转换
- [x] 物理单位对齐（cm, pixels）
- [x] 认知焦点状态机
- [x] 内存优化（float32 for coordinates, int16 for labels）
- [x] 向量化操作（无 iterrows()）
- [x] 完整的错误处理和日志

### 📊 测试结果

处理参与者 69907732 的所有 6 个任务：

```
car:   1918 rows ✓
tb:    1547 rows ✓
house: 1571 rows ✓
sc:    1411 rows ✓
tc:    1236 rows ✓
tsb:   2238 rows ✓
```

## 环境常量

脚本使用的物理常量（来自需求规范）：

```python
显示器 (Dell P2416D):
  - 物理尺寸: 52.7 × 29.6 cm
  - 分辨率: 2560 × 1440 px

LEGO 装配表面:
  - 物理尺寸: 76.0 × 38.0 cm

坐标转换:
  - 表面: 相对 [0,1] → cm [0, 76/38]
  - 屏幕: 相对 [0,1] → cm [0, 52.7/29.6] → px [0, 2560/1440]
```

## 适配说明

该实现经过以下适配以符合实际 GAIPAT 数据结构：

1. **数据位置**：events, states 等文件位于 `{task}/table/` 和 `{task}/screen/` 目录
2. **时间戳来源**：来自 gazepoints.csv 和 events.csv 的并集
3. **Block ID 映射**：从 instructions_{task}.csv 中提取
4. **状态机简化**：基于注视点有效性和事件进行判定

## 后续扩展

建议在此基础上添加的功能：

1. **逻辑 3 完整版**：片段级前瞻标签（is_success）
2. **事件富化**：解析 events.csv 中的更多动作细节
3. **AOI 映射**：从 slides_{task}.csv 提取屏幕兴趣区域
4. **异常检测**：识别数据质量问题和 outliers
5. **特征工程**：计算注视稳定性、扫视速度等高级指标

## 文件输出示例

```
processed/master_dataframes/
├── 69907732/
│   ├── car_master.csv (1918 rows × 33 cols)
│   ├── house_master.csv (1571 rows × 33 cols)
│   ├── ...
└── pipeline.log
```

## 性能

- 单参与者单任务处理时间：~50ms
- 整体流水线（不含 I/O）：线性时间复杂度
- 内存占用：通过 float32 和 category 优化

## 故障排查

### 警告：CSV 中缺少文件
→ 检查 participants/{id}/{task}/table/ 和 screen/ 目录是否存在

### 空 DataFrame
→ 检查该 subject/task 的数据完整性

### 不正确的 block_id
→ 检查 instructions_{task}.csv 是否正确解析

# GAIPAT 偏差序列可视化脚本

## 脚本

`06_visualize_deviation_sequences.py` - 将第三步和第四步生成的偏差 CSV 文件可视化为 PNG 图像，横轴为时间或归一化进度，纵轴为偏差。

## 功能说明

该脚本会分别扫描：

- 第三步输出目录中的所有偏差 CSV；
- 第四步输出目录中的所有偏差 CSV，包括 `arithmetic_mean` 和 `dba_mean` 两个方法目录下的文件。

对每个可绘制文件，脚本会：

- 画出所有原始点；
- 用一条平滑曲线连接这些点；
- 保留原始文件名，只把扩展名改为 `.png`；
- 将第三步和第四步的图像分别保存到不同文件夹。

## 默认输入输出

- 第三步输入目录：`<repo-root>/block_deviation_sequences`
- 第四步输入目录：`<repo-root>/mean_deviation_sequences`
- 输出基目录：`<repo-root>/deviation_visualizations`

输出结构示例：

```text
deviation_visualizations/
  ├── step3/
  │   ├── 10383544_sc_12_grasp_3_1.png
  │   └── ...
  └── step4/
      ├── arithmetic_mean/
      │   ├── mean_deviation_sequence.png
      │   └── normalized_sequences/
      │       └── ...
      └── dba_mean/
          ├── mean_deviation_sequence.png
          └── normalized_sequences/
              └── ...
```

## 运行方式

### 1. 使用默认路径

```bash
python preprocess_script/06_visualize_deviation_sequences.py
```

### 2. 指定仓库根目录

```bash
python preprocess_script/06_visualize_deviation_sequences.py --repo-root D:\code\gaze_attn
```

### 3. 指定输入和输出目录

```bash
python preprocess_script/06_visualize_deviation_sequences.py \
    --repo-root D:\code\gaze_attn \
    --step3-dir D:\code\gaze_attn\block_deviation_sequences \
    --step4-dir D:\code\gaze_attn\mean_deviation_sequences \
    --output-dir D:\code\gaze_attn\deviation_visualizations
```

## 绘图规则

- 第三步文件优先使用 `deviation_cm` 作为纵轴；如果没有，则回退到 `deviation_dst_cm` 或 `deviation_srt_cm`。
- 第四步文件优先使用 `arithmetic_mean_deviation_dst_cm`、`dba_deviation_dst_cm`、`deviation_cm`、`deviation_dst_cm` 等可用偏差列。
- 横轴优先使用 `normalized_progress`，否则使用 `timestamp`。
- 平滑曲线优先使用 SciPy 的样条插值，若环境中不可用则自动回退到线性插值。

## 依赖

该脚本依赖 `numpy`、`pandas`、`matplotlib`，并建议安装 `scipy` 以获得更平滑的曲线效果。

```bash
pip install matplotlib numpy pandas scipy
```
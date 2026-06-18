# GAIPAT 第五步：DTW 专注力分数与相关性散点图

## 脚本

`05_compute_dtw_adf_score.py` - 基于第 4 步生成的标准模板与归一化序列，计算每个片段到模板的 DTW 距离，并使用 `scipy.stats.pearsonr` 计算 DTW 距离与成功标签的相关性，同时输出散点图。

## 依赖

```bash
pip install fastdtw scipy matplotlib
```

## 功能说明

该脚本默认读取第 4 步输出目录中的两个方法文件夹：

- `arithmetic_mean`
- `dba_mean`

每个方法文件夹内的 `normalized_sequences/` 会被逐个扫描，然后将每个归一化序列与对应方法的 `mean_deviation_sequence.csv` 中的标准模板计算 DTW 距离。

输入文件名沿用第 3 步的新格式，包含 `event` 字段，但 DTW 计算本身只读取序列值，不额外依赖事件类型。

输出的 DTW 距离可以直接作为 ADF 分数使用，距离越小，表示该片段与标准专注模板越接近。

随后脚本会把 DTW 距离与 `label` 做 Pearson 相关分析，并生成散点图，用于观察成功 / 失败样本在模板距离上的分布差异。

## 默认输入输出

- 输入目录：`<repo-root>/mean_deviation_sequences`
- 输出位置：直接写回每个方法子目录

每个方法子目录会生成以下文件：

- `dtw_scatter_data.csv`：每条序列的 DTW 距离与元信息；
- `dtw_pearson_summary.csv`：Pearson 相关结果摘要；
- `dtw_distance_scatter.png`：相关性散点图。

## 运行方式

### 1. 使用默认路径

```bash
python preprocess_script/05_compute_dtw_adf_score.py
```

### 2. 指定仓库根目录

```bash
python preprocess_script/05_compute_dtw_adf_score.py --repo-root /root/autodl-tmp/shenxy/XDU/Dataset/gaipat
```

### 3. 只处理某一种模板

```bash
python preprocess_script/05_compute_dtw_adf_score.py --method arithmetic_mean
```

## 输出字段说明

`dtw_scatter_data.csv` 包含：

- `subject_id`
- `task`
- `step_id`
- `block_id`
- `label`
- `is_success`
- `sequence_length`
- `dtw_distance`
- `template_column`
- `sequence_file`

`dtw_pearson_summary.csv` 包含：

- `sequence_count`
- `pearson_r`
- `pearson_p_value`
- `mean_distance_label_0`
- `mean_distance_label_1`

## 说明

- 这里使用 `fastdtw` 计算 DTW 距离；
- `pearsonr` 的相关对象是 `dtw_distance` 与二值标签 `label`；
- 散点图中对标签轴做了轻微 jitter 处理，以避免点重叠。
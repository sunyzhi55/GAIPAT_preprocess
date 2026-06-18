# GAIPAT 第五步：相对 ADF 距离计算脚本使用说明

## 脚本

`05_compute_relative_adf_score.py` - 基于第三步输出的偏差序列，先对每个 `subject_id` 的完美成功片段拟合个人 Gamma 分布，再将该受试者的所有样本广播到自己的均值基线之上，得到相对 ADF 距离。

## 核心思路

对于每个受试者：

1. 只取其所有 `label = 1` 的成功片段；
2. 从这些片段中抽取高频距离数据，优先使用 `gaze_target_distance`，兼容 `deviation_cm` 和 `deviation_dst_cm`；
3. 使用 `scipy.stats.gamma.fit()` 拟合个人专注状态的 Gamma 分布；
4. 用拟合参数计算个人均值基线 `μ_individual`；
5. 将该受试者的所有样本都减去自己的基线，得到：

```text
ADF_Relative_Distance = gaze_target_distance - μ_individual
```

这表示“此刻相对于他自己平时专注状态的偏离程度”。

## 依赖

```bash
pip install scipy matplotlib pandas numpy
```

## 输入输出

- 输入目录：`<repo-root>/block_deviation_sequences`
- 输出目录：`<repo-root>/relative_adf_sequences`

输出文件包括：

- `subject_gamma_baselines.csv`：每个受试者的 Gamma 拟合参数与均值基线；
- `relative_adf_distance_data.csv`：逐样本的相对 ADF 距离结果；
- `relative_adf_distance_summary.csv`：整体统计摘要；
- `subject_relative_adf_summary.csv`：每个受试者的相对 ADF 统计摘要；
- `relative_adf_distance_distribution.png`：相对 ADF 距离分布图；
- `subject_relative_adf_plots/`：每个受试者单独的相对 ADF 分布图；
- `relative_adf.log`：运行日志。

## 输出字段

`subject_gamma_baselines.csv` 包含：

- `subject_id`
- `file_count`
- `success_file_count`
- `valid_frame_count`
- `success_valid_frame_count`
- `gamma_shape`
- `gamma_loc`
- `gamma_scale`
- `individual_mean_baseline`
- `fit_status`
- `fit_sample_count`

`relative_adf_distance_data.csv` 包含：

- `subject_id`
- `task`
- `step_id`
- `event`
- `block_id`
- `label`
- `is_success`
- `frame`
- `timestamp`
- `gaze_target_distance`
- `individual_mean_baseline`
- `adf_relative_distance`
- `gamma_shape`
- `gamma_loc`
- `gamma_scale`
- `fit_status`
- `distance_column`
- `sequence_length`
- `sequence_file`

`subject_relative_adf_summary.csv` 包含：

- `subject_id`
- `row_count`
- `success_row_count`
- `failure_row_count`
- `mean_relative_distance`
- `median_relative_distance`
- `mean_relative_distance_success`
- `mean_relative_distance_failure`

## 运行方式

### 1. 使用默认路径

```bash
python preprocess_script/05_compute_relative_adf_score.py
```

### 2. 指定仓库根目录

```bash
python preprocess_script/05_compute_relative_adf_score.py --repo-root D:\code\gaze_attn
```

### 3. 指定输入和输出目录

```bash
python preprocess_script/05_compute_relative_adf_score.py \
    --repo-root D:\code\gaze_attn \
    --input-dir D:\code\gaze_attn\block_deviation_sequences \
    --output-dir D:\code\gaze_attn\relative_adf_sequences
```

## 说明

- 该脚本只用成功片段拟合每个受试者的 Gamma 基线；
- 但输出会广播到该受试者的所有有效样本，因此失败片段也会参与相对距离分布；
- 如果某个受试者没有任何成功片段，脚本会保留该受试者的行，但相对距离会是 `NaN`，并在日志中提示。
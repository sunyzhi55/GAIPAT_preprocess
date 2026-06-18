# GAIPAT 第四步：切片级偏差序列聚类说明

## 脚本

`04_cluster_deviation_sequences.py` 接在第三步 `03_compute_block_deviation_sequences.py` 之后运行。它读取第三步导出的 `.jsonl` 偏差序列文件，把每个 JSONL 文件视为一个切片级样本，提取固定长度特征，并按照 `grasp` 与 `release` 两类事件分别训练独立的 GMM 聚类模型。

## 输入

默认输入目录：

```text
<repo-root>/block_deviation_sequences
```

输入文件名必须符合第三步输出格式：

```text
[subject_id]_[task]_[step]_[event]_[block_id]_[label].jsonl
```

其中：

- `event` 只能是 `grasp` 或 `release`。
- `label == 1` 表示原始切片来自成功样本。
- `label == 0` 表示原始切片来自失败样本。

JSONL 每一行至少需要包含：

- `timestamp`
- `event`
- `deviation_cm`

脚本只使用有限数值的 `deviation_cm`。空值、非数值和无穷值会被跳过。

## 特征

脚本严格基于原始物理偏差序列 `deviation_cm` 提取以下 10 个切片级特征：

- `distance_median`
- `distance_q10`
- `distance_q90`
- `distance_std`
- `distance_slope`
- `diff_abs_mean`
- `distance_lag1_autocorr`
- `near_ratio_tau`
- `longest_near_run_ratio`
- `time_to_first_near_ratio`

`near_ratio_tau`、`longest_near_run_ratio` 和 `time_to_first_near_ratio` 默认使用 `2.0 cm` 作为接近目标阈值，可通过 `--near-threshold-cm` 修改。

## 聚类策略

脚本流程为：

1. 读取第三步 JSONL，生成原始特征表 `cluster_raw_features.csv`。
2. 按 `event` 拆分为 `grasp` 和 `release` 两组。
3. 对每组特征分别使用 `PowerTransformer(method="yeo-johnson", standardize=True)` 进行变换。
4. 对每组变换后的特征分别训练 `GaussianMixture`。
5. 输出每个切片的聚类结果和 GMM 软概率。
6. 根据硬规则将所有 `label == 0` 的样本最终覆盖为 `Distracted`。

注意：失败样本仍会参与 GMM 拟合，但最终导出的可解释标签必须被覆盖为 `Distracted`，用于满足 `cluster.md` 中的金标准约束。

## 输出

默认输出目录：

```text
<repo-root>/cluster_results
```

主要输出文件：

- `cluster_raw_features.csv`：未缩放的原始 10 维特征表。
- `cluster_assignments.csv`：每个切片的最终聚类分配结果。
- `cluster_summary.csv`：按 `event / cluster_id / cluster_name` 汇总的统计结果。
- `grasp_cluster_probabilities.csv`：`grasp` 模型的软概率。
- `release_cluster_probabilities.csv`：`release` 模型的软概率。
- `grasp_gmm_model.pkl`：`grasp` 的 Yeo-Johnson 变换器和 GMM 模型。
- `release_gmm_model.pkl`：`release` 的 Yeo-Johnson 变换器和 GMM 模型。
- `cluster_mean_curves.csv`：每个最终簇的平均偏差曲线。
- `{grasp,release}_cluster_mean_curves.png`：平均偏差曲线图。若服务器未安装 `matplotlib`，会自动跳过 PNG。
- `cluster.log`：运行日志。

## 运行方式

### 1. 默认路径运行

```bash
python preprocess_script/04_cluster_deviation_sequences.py
```

### 2. 指定输入输出目录

```bash
python preprocess_script/04_cluster_deviation_sequences.py \
    --repo-root /root/autodl-tmp/shenxy/XDU/Dataset/gaipat-main \
    --input-dir /root/autodl-tmp/shenxy/XDU/Dataset/gaipat-main/block_deviation_sequences \
    --output-dir /root/autodl-tmp/shenxy/XDU/Dataset/gaipat-main/cluster_results
```

### 3. 只生成原始特征表

本模式不需要 `scikit-learn`，适合先检查第三步 JSONL 是否可正常解析。

```bash
python preprocess_script/04_cluster_deviation_sequences.py --features-only
```

### 4. 调整 GMM 簇数

```bash
python preprocess_script/04_cluster_deviation_sequences.py --n-components 4
```

如果某个事件的样本数小于请求的簇数，脚本会自动把该事件的簇数降到可用样本数。

### 5. 不生成图片

```bash
python preprocess_script/04_cluster_deviation_sequences.py --no-plots
```

## 依赖

完整聚类需要：

```bash
pip install scikit-learn
```

如需导出 PNG 曲线图，还需要：

```bash
pip install matplotlib
```

项目已有前三步脚本依赖的 `numpy` 和 `pandas` 仍然需要保留。

## 结果解释建议

`cluster_name` 是根据成功样本的簇均值特征自动生成的启发式名称：

- `Focused`：接近目标比例高、距离中位数低、波动小。
- `Searching`：距离斜率更偏负，表示逐渐靠近目标。
- `Wandering`：波动较大，最长接近目标持续段偏短。
- `Noise`：一阶差分偏大，自相关偏低。
- `Distracted`：所有原始 `label == 0` 的失败样本都会被硬覆盖成该标签。

最终判断聚类是否合理时，不建议只看数学指标。更重要的是查看 `cluster_mean_curves.csv` 或 PNG 曲线，确认每个簇的平均偏差曲线是否具有清晰的物理含义。

# GAIPAT 第四步：平均偏差序列计算脚本使用说明

## 脚本

`04_compute_mean_deviation_sequence.py` - 对第三步输出的 `deviation_dst_cm` 序列做时序归一化，并分别保存算术平均和 DBA 两种平均偏差序列。

```bash
pip install tslearn
```

## 功能说明

该脚本按照 `get_mean_deviation.md` 的思路实现三件事：

1. 算术平均版本先统计成功序列长度分布，再自动选择固定长度 `L`（默认中位数）；
2. DBA 版本先用 `dtw_barycenter_averaging` 自动得到自己的 `L`；
3. 两种版本各自独立做 progress-based resampling；
4. 两种版本都只使用 `label = 1` 的成功序列；
5. 两种版本分别独立计算、独立选择 L、独立保存到两个不同的文件夹。

注意：

- `label = 1` 和 `label = 0` 的文件都会进行归一化输出；
- 但计算平均偏差序列时，只使用成功序列；
- 优先使用第三步新生成的 `deviation_cm` 作为输入；如果旧文件仍只有 `deviation_dst_cm`，则自动回退兼容。
- 输入切片文件名现在包含 `event` 字段，但这里的均值/DBA 计算仍按每条切片独立处理，不需要额外分支逻辑。

## 默认输入输出

- 输入目录：`<repo-root>/block_deviation_sequences`
- 输出基目录：`<repo-root>/mean_deviation_sequences`

输出会分为两个互相独立的方法文件夹；每个方法文件夹内部都会有自己的：

- `mean_deviation_sequence.csv`：该方法的平均偏差模板；
- `length_statistics.csv`：该方法的长度统计；
- `processing_summary.csv`：该方法的处理汇总；

最终会得到两个互相独立的结果文件夹：

- `arithmetic_mean`
- `dba_mean`

### 输出文件

归一化后的单文件结果会保留原文件名，例如：

```text
[subject_id]_[task]_[step]_[block_id]_[label].csv
```

另外还会生成：

- `mean_deviation_sequence.csv`：分别位于 `arithmetic_mean` 和 `dba_mean` 子目录中；
- `normalized_sequences/`：分别位于 `arithmetic_mean` 和 `dba_mean` 子目录中，包含所有归一化后的单文件 CSV；
- `length_statistics.csv`：分别位于两个结果子目录中；
- `processing_summary.csv`：分别位于两个结果子目录中；
- `mean_deviation.log`：运行日志，保存在基目录中。

## 导出字段

每个归一化后的单文件 CSV 只保留以下列：

- `normalized_progress`
- `deviation_dst_cm`

平均偏差序列文件包含：

- `normalized_progress`
- `arithmetic_mean_deviation_dst_cm` 或 `dba_deviation_dst_cm`
- `std_deviation_dst_cm`
- `sequence_count`

## 固定长度 L 的选择

脚本会先统计成功序列的有效长度，然后分别为两个方法选择各自的 `L`：

- 算术平均版本：默认使用中位数自动选择 `L`；
- DBA 版本：默认使用 `dtw_barycenter_averaging` 自动得到 `L`。
- `--arithmetic-target-length`
- `--dba-target-length`

其中，`--target-length` 只作用于算术平均版本，DBA 版本默认走自己的自动长度。

## 运行方式

### 1. 使用默认路径

```bash
python preprocess_script/04_compute_mean_deviation_sequence.py
```

### 2. 指定仓库根目录

```bash
python preprocess_script/04_compute_mean_deviation_sequence.py --repo-root /root/autodl-tmp/shenxy/XDU/Dataset/gaipat
```

### 3. 指定输入和输出目录

```bash
python preprocess_script/04_compute_mean_deviation_sequence.py \
    --repo-root D:\code\gaze_attn \
    --input-dir D:\code\gaze_attn\block_deviation_sequences \
    --output-dir D:\code\gaze_attn\mean_deviation_sequences
```

### 4. 手动指定固定长度 L

```bash
python preprocess_script/04_compute_mean_deviation_sequence.py --arithmetic-target-length 100 --dba-target-length 120
```

## 依赖

该脚本依赖 `numpy`、`pandas` 和 `tslearn`。

```bash
pip install tslearn
```

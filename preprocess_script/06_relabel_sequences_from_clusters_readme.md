# GAIPAT 第六步：根据聚类结果重打标签并复制 JSONL

## 脚本

`06_relabel_sequences_from_clusters.py` 用于读取聚类输出目录中的 `cluster_assignments.csv`，根据聚类结果重新定义样本标签，并把原始 JSONL 文件复制到新的输出目录。

输出文件名仍保持：

```text
[subject_id]_[task]_[step]_[event]_[block_id]_[label].jsonl
```

其中最后的 `label` 会被替换为新的标签。

## 标签规则

### release

| cluster_name | 新标签 | 含义 |
|---|---:|---|
| `Focused` | `1` | 专注 |
| `Distracted` | `0` | 分心 |
| `Wandering` | `0` | 分心 |
| `Searching` | `2` | 丢弃 |

### grasp

| cluster_name | 新标签 | 含义 |
|---|---:|---|
| `Focused` | `1` | 专注 |
| `Distracted` | `0` | 分心 |
| `Wandering` | `2` | 丢弃 |
| `Searching` | `3` | 丢弃 |

## 输出结构

默认只复制新标签为 `0` 和 `1` 的可训练样本。新标签为 `2` 或 `3` 的丢弃样本不会复制，只会记录到审计表。

```text
relabelled_sequences/
  grasp/
    *_0.jsonl
    *_1.jsonl
  release/
    *_0.jsonl
    *_1.jsonl
  relabel_audit.csv
  relabel_summary.csv
  relabel_sequences.log
```

如果使用 `--copy-discarded`，丢弃样本也会复制：

```text
relabelled_sequences/
  grasp/
    discarded/
      *_2.jsonl
      *_3.jsonl
  release/
    discarded/
      *_2.jsonl
```

## 输入路径

脚本需要 `cluster_assignments.csv`。可以通过两种方式指定：

### 1. 指定聚类结果文件夹

```bash
--cluster-results-dir /path/to/cluster_results
```

脚本会读取：

```text
/path/to/cluster_results/cluster_assignments.csv
```

### 2. 直接指定 CSV

```bash
--cluster-assignments /path/to/cluster_assignments.csv
```

## 源 JSONL 文件定位

脚本优先支持两种定位方式：

1. 如果传入 `--source-dir`，则按 `sequence_id + ".jsonl"` 在该目录查找。
2. 如果不传 `--source-dir`，则使用 `cluster_assignments.csv` 中的 `source_file` 列。

建议在你已经做完长度归一化后，使用归一化后的目录作为源目录，例如：

```text
normalized_deviation_jsonl_sequences/arithmetic_mean/normalized_sequences
```

这样重打标签后的文件就是长度一致的版本。

## 推荐运行方式

### 使用 arithmetic 归一化结果重打标签

```bash
python preprocess_script/06_relabel_sequences_from_clusters.py \
    --cluster-results-dir /root/autodl-tmp/shenxy/Data/gaipat/cluster_results \
    --source-dir /root/autodl-tmp/shenxy/Data/gaipat/normalized_deviation_jsonl_sequences/arithmetic_mean/normalized_sequences \
    --output-dir /root/autodl-tmp/shenxy/Data/gaipat/relabelled_sequences
```

### 复制丢弃样本用于人工检查

```bash
python preprocess_script/06_relabel_sequences_from_clusters.py \
    --cluster-results-dir /root/autodl-tmp/shenxy/Data/gaipat/cluster_results \
    --source-dir /root/autodl-tmp/shenxy/Data/gaipat/normalized_deviation_jsonl_sequences/arithmetic_mean/normalized_sequences \
    --output-dir /root/autodl-tmp/shenxy/Data/gaipat/relabelled_sequences \
    --copy-discarded
```

## 输出审计文件

### `relabel_audit.csv`

逐样本记录：

- `sequence_id`
- `event`
- `cluster_name`
- `old_label`
- `new_label`
- `action`
- `source_file`
- `output_file`
- `reason`

其中：

- `action == copied`：已复制。
- `action == discarded`：标签为 2/3，默认未复制。
- `action == skipped`：源文件缺失、命名异常或规则缺失。

### `relabel_summary.csv`

按 `event / cluster_name / new_label / action` 汇总数量。运行后应检查该表是否符合预期数量。

## 注意

脚本只复制和重命名文件，不修改 JSONL 文件内容。如果你的 JSONL 内部也保存了旧 `label` 字段，而你希望内部字段也同步更新，需要另写一个内容重写版本。当前版本保持原始内容不变，方便追溯。 

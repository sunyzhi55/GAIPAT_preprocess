# GAIPAT 切片级无监督聚类方案

## 1. 目标

本文档用于说明如何基于 `03_compute_block_deviation_sequences.py` 产生的切片结果，对每个切片片段进行无监督聚类，以发现潜在的注视状态模式，并为后续“专注 / 分心”定义提供结构化依据。

本方案针对当前项目的约束做了专门设计：

- 聚类输入来自第三步 `compute_block_deviation_sequences` 的切片结果。
- 每个切片是一个变长时序序列。
- 主要使用“注视点与目标之间的距离”作为核心信息源。
- 成功和失败片段都必须参加聚类。
- 原始失败片段在聚类后仍必须保留为“分心”或“非专注”样本，不能被聚类结果改写。

---

## 2. 为什么不直接用原有 `is_success` 作为注意标签

当前流程中，`is_success` 来自操作结果，即某一步最终是否成功完成。它适合作为行为结果标签，但不适合作为瞬时注意状态标签。原因有三点：

1. `is_success` 是结果变量，不是过程变量。它混合了视觉注意、手部控制、空间修正、任务策略和偶然误差。
2. 同一个成功片段内部也可能同时存在“高度专注”“任务相关搜索”“短暂偏离”三种状态。
3. 如果把整个失败片段等价为全程分心，会把部分“曾经专注但最终执行失败”的时段也错误归入分心。

因此，本项目的聚类目标不应是“复现 `is_success`”，而应是发现切片内部的潜在 gaze-distance 模式。

---

## 3. 聚类单位

### 3.1 主聚类单位

建议将每个切片文件视作一个样本进行聚类。

当前切片命名格式为：

```text
[subject_id]_[task]_[step]_[event]_[block_id]_[label].csv
```

其中：

- `subject_id`：参与者编号
- `task`：拼接任务名称
- `step`：步骤编号
- `event`：`grasp` 或 `release`
- `block_id`：当前积木编号
- `label`：由原流程得到的结果标签，`1=成功`，`0=失败`

### 3.2 为什么不用单帧聚类

不建议对单帧直接聚类，原因如下：

- 单帧噪声大，容易受采样、眨眼、信号缺失影响。
- 专注状态具有时间连续性，仅凭单帧很难稳定判断。
- 你的数据已经天然按行为事件切成片段，继续利用片段级结构更合理。

因此，建议对“每个切片提取一个固定长度特征向量”，再进行样本级聚类。

---

## 4. 事件分层策略

## 4.1 不建议把 `grasp` 和 `release` 混在一起聚类

建议对以下两类切片分别聚类：

- `grasp`：关注从源位置搜索并抓取积木
- `release`：关注将积木移动到目标位置并完成放置

原因：

- 两类事件的目标语义不同。
- 同样的距离模式在 `grasp` 和 `release` 中含义不同。
- 混合聚类会让簇同时包含“找源物体”和“对齐目标位置”两类机制，降低可解释性。

### 4.2 推荐实践

先分别建立两个聚类模型：

- `grasp_cluster_model`
- `release_cluster_model`

如果后续需要统一分析，可以在聚类结束后再比较两个事件空间中的簇分布。

---

## 5. 输入数据建议

聚类的基础输入不应直接使用第三步切片中的原始坐标，而应优先使用由第三步或其等价逻辑得到的“目标距离序列”。

`03_compute_block_deviation_sequences.py` 生成的距离序列
   - `deviation_cm`

如果直接从第三步切片做聚类，建议先补一个中间表，把每条切片转换成对应的距离序列，再做特征提取。这样后续实验会更稳定，也更容易复现实验。

---

## 6. 特征设计原则

由于每个切片长度不等，因此不建议直接做零填充后用普通 KMeans。更合理的方式是先把变长距离序列压缩成固定长度特征表示。

特征应满足以下原则：

- 以目标距离为核心
- 保留时间结构信息
- 可解释
- 尽量适应跨对象、跨参与者差异
- 能兼容成功与失败片段共同建模

---

## 7. 推荐特征组

以下特征建议以每个切片为单位提取。

设距离序列为：

```text
d = [d1, d2, d3, ..., dL]
```

其中 `L` 为该切片长度。

### 7.1 距离水平特征

这组特征反映切片整体距离目标是近还是远。

- `distance_mean`
- `distance_median`
- `distance_min`
- `distance_max`
- `distance_q10`
- `distance_q25`
- `distance_q75`
- `distance_q90`

作用：

- 区分“整体贴近目标”与“整体远离目标”的样本
- 为后续聚类提供最基础的强信号

### 7.2 距离波动特征

这组特征反映 gaze 是否稳定。

- `distance_std`
- `distance_mad`
- `distance_iqr`
- `distance_cv`

其中：

- `MAD`：median absolute deviation
- `IQR`：四分位距
- `CV`：变异系数，可定义为 `std / (mean + eps)`

作用：

- 高专注片段往往表现为“近且稳”
- 游移、搜索、分心片段往往表现为“远或波动大”

### 7.3 距离趋势特征

这组特征反映 gaze 是否在逐渐靠近目标。

- `distance_slope`
- `distance_start_end_gap`
- `distance_first_half_mean`
- `distance_second_half_mean`
- `distance_half_gap`
- `distance_argmin_ratio`

定义建议：

- `distance_slope`：对 `d ~ t` 做线性回归得到斜率
- `distance_start_end_gap`：前 10% 均值减后 10% 均值
- `distance_half_gap`：前半段均值减后半段均值
- `distance_argmin_ratio`：最小距离首次出现位置除以序列长度

作用：

- 区分“从远处逐渐收敛到目标”的搜索型片段
- 区分“始终没有贴近目标”的偏离型片段

### 7.4 距离变化率特征

这组特征反映 gaze 的跳动程度。

设一阶差分序列为：

```text
Δd = [d2-d1, d3-d2, ..., dL-d(L-1)]
```

推荐特征：

- `diff_abs_mean`
- `diff_abs_std`
- `diff_abs_max`
- `diff_mean`
- `diff_std`
- `distance_lag1_autocorr`

作用：

- 区分平稳跟踪与频繁跳变
- 自相关高的片段更可能属于稳定监控状态

### 7.5 持续贴近目标特征

这是本方案最重要的一组特征。

专注不应只由“平均距离小”定义，更应体现为“在较长时间内持续贴近任务目标”。

建议先定义两个归一化阈值：

- `tau_near = 0.25 * object_diag`
- `tau_mid = 0.50 * object_diag`

其中 `object_diag` 为目标多边形对角线长度。

然后计算：

- `near_ratio_tau1 = p(d < tau_near)`
- `near_ratio_tau2 = p(d < tau_mid)`
- `longest_near_run_tau1`
- `longest_near_run_tau2`
- `near_run_count_tau1`
- `near_run_count_tau2`
- `time_to_first_near_tau1`
- `time_to_first_near_tau2`
- `mean_near_run_length_tau1`

再做长度归一化：

- `longest_near_run_tau1_ratio = longest_near_run_tau1 / L`
- `time_to_first_near_tau1_ratio = time_to_first_near_tau1 / L`

作用：

- 区分“快速进入目标并保持”的片段
- 区分“偶尔扫到目标但很快离开”的片段
- 区分“从未真正贴近目标”的片段

### 7.6 长度与质量特征

这些特征不直接表示专注，但对于聚类稳定性很重要。

- `seq_len`
- `valid_gaze_ratio`
- `table_confidence_mean`
- `table_confidence_low_ratio`

如果使用的是第三步切片，则可以从：

- `table_confidence_right`
- `table_confidence_left`
- `screen_confidence_right`
- `screen_confidence_left`

中构造质量特征。

作用：

- 避免低质量样本形成“假分心簇”
- 支持后续剔除或标注低置信度簇

---

## 8. 最小可用特征集合

如果只做第一版聚类，建议先用以下 10 个特征：

- `distance_median`
- `distance_q10`
- `distance_q90`
- `distance_std`
- `distance_slope`
- `diff_abs_mean`
- `distance_lag1_autocorr`
- `near_ratio_tau1`
- `longest_near_run_tau1_ratio`
- `time_to_first_near_tau1_ratio`

原因：

- 数量少，便于调试和解释
- 同时覆盖水平、稳定性、趋势、持续性四个维度
- 对变长序列比较友好

---

## 9. 增强特征方案

如果基础特征聚类效果一般，建议增加以下增强方式。

### 9.1 重采样形状特征

将每个距离序列重采样为固定长度，例如 32 或 64 点：

```text
[d1, ..., dL] -> [r1, r2, ..., r32]
```

然后：

- 直接把重采样后的值作为形状特征
- 或进一步提取低维统计特征

优点：

- 保留更多整体轮廓信息
- 适合后续做 DTW 距离比较

### 9.2 通用时间序列特征

可选地引入 `catch22` 一类通用时间序列特征：

- 自相关相关特征
- 分布不对称特征
- 波动结构特征
- 复杂度特征

适用场景：

- 当手工特征无法稳定分出有解释力的簇时
- 当希望比较不同特征库效果时

### 9.3 DTW 到原型距离

先从样本中选出若干原型序列或 medoid，再计算每个样本到这些原型的 DTW 距离：

- `dtw_to_proto_1`
- `dtw_to_proto_2`
- `dtw_to_proto_3`
- ...

然后将这些 DTW 距离和手工特征拼接起来做聚类。

优点：

- 对变长时序友好
- 保留时序形状信息
- 比直接对全序列做聚类更容易解释

---

## 10. 是否要加入原始坐标特征

如果你的研究重点严格限制为“以目标距离为主”，那么二维坐标可以不作为第一版输入。

但从建模角度看，以下增强是有价值的：

- `dx = gaze_x - target_center_x`
- `dy = gaze_y - target_center_y`
- `angle_to_target`
- `radial_velocity`

原因：

- 同样的距离值可能对应完全不同的空间行为
- 有的人是在目标周围稳定观察
- 有的人是在远处横向扫视

这两种情况仅靠距离未必能区分

因此建议：

- 第一版：只用距离类特征
- 第二版：加入 `dx/dy` 派生特征做对照实验

---

## 11. 归一化建议

由于不同目标积木尺寸、不同任务和不同参与者可能带来尺度差异，建议对距离做尺度归一化。

### 11.1 目标尺度归一化

推荐方式：

```text
normalized_distance = raw_distance / (target_object_diagonal + eps)
```

或：

```text
normalized_distance = raw_distance / (target_object_width + eps)
```

推荐优先使用对角线尺度，因为它更稳定地反映目标空间尺度。


---

## 12. 聚类算法推荐

### 12.1 第一推荐：手工特征 + GMM

优点：

- 适合连续特征
- 可以输出软概率
- 便于观察簇间分布差异

适用：

- 你希望聚类结果具有统计可解释性
- 希望后续做簇概率分析

### 12.2 第二推荐：手工特征 + HDBSCAN

优点：

- 对非球形簇更友好
- 可以自动识别噪声点
- 不强制每个样本都进入某个簇

适用：

- 你怀疑存在异常切片
- 你不确定最佳簇数

### 12.3 第三推荐：DTW / soft-DTW 距离 + k-medoids

优点：

- 对变长时序天然友好
- 更保留序列形状

缺点：

- 解释性较弱
- 实现和调参成本更高
- 与手工特征方案相比，更难解释“为什么这个簇代表专注”

### 12.4 不建议直接使用的方法

- 原始序列零填充后直接 KMeans
- 单一均值特征后二类 KMeans
- 不区分 `grasp/release` 的全局混合聚类

---

## 13. 聚类流程建议

建议按以下流程组织实验。

### 第一步：准备样本表

从第三步切片文件出发，生成一个“切片级样本清单表”，每行对应一个切片，至少包含：

- `subject_id`
- `task`
- `step_id`
- `event`
- `block_id`
- `label`
- `file_path`

### 第二步：构造距离序列

对于每个切片：使用deviation_cm

并统一保存为：

- `distance_sequence`

### 第三步：提取固定长度特征

对每个切片输出一行特征：

- 元信息列
- 上述手工统计特征
- 可选的重采样形状特征

形成：

```text
cluster_feature_table.csv
```

### 第四步：分事件聚类

分别对：

- `event == grasp`
- `event == release`

做独立聚类。

### 第五步：解释每个簇

对每个簇统计：

- 样本数
- 成功比例
- 失败比例
- 平均距离曲线
- 平均序列长度
- 持续贴近目标比例

然后人工解释每个簇的含义，例如：

- 簇 A：快速贴近并稳定保持
- 簇 B：搜索后逐渐收敛
- 簇 C：长时间远离目标
- 簇 D：低质量 / 高噪声

---

## 14. 关于失败样本的强约束

这是本方案的核心约束之一。

### 14.1 聚类阶段

所有切片都参加聚类：

- 成功样本参加
- 失败样本也参加

这样做的目的是让簇结构能看到完整行为分布，而不是只看到成功样本的局部模式。

### 14.2 最终标签阶段

聚类完成后，必须保留以下规则：

- 原始 `label == 0` 的切片，最终仍归入“分心 / 非专注”
- 聚类只能用于进一步细分成功样本，或解释失败样本的内部结构
- 聚类不能把原始失败样本重新改判成“专注”

### 14.3 推荐的最终使用方式

最终标签建议分两层：

#### 第一层：硬规则层

- `failure -> distracted`
- `success -> unresolved`

#### 第二层：簇解释层

对于 `success` 样本，再根据聚类簇做细分：

- `focused_success`
- `searching_success`
- `unstable_success`

这样既满足你的约束，也能从成功样本内部挖出注意状态结构。

---

## 15. 如何判断聚类效果好不好

不要只看轮廓系数。

建议同时看三类指标。

### 15.1 内部指标

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

### 15.2 稳定性指标

- 不同随机种子下簇划分是否稳定
- 不同特征子集下簇结构是否稳定
- 不同参与者子集下是否能得到相似簇

### 15.3 可解释性指标

对每个簇检查：

- 平均距离曲线是否有明确形状差异
- `failure` 比例是否显著不同
- 是否能对应明确行为模式

如果一个簇无法解释，即使内部指标好，也不建议直接用于标签定义。

---

## 16. 与当前项目脚本的衔接建议

### 16.1 现有数据链路

当前项目已有如下处理链：

1. `01_extract_unified_data_pipeline.py`
2. `02_slice_dataframes.py`
3. `03_compute_block_deviation_sequences.py`
4. `04_compute_mean_deviation_sequence.py`
5. `05_compute_dtw_adf_score.py`
6. `05_compute_relative_adf_score.py`

### 16.2 聚类方案建议插入的位置

推荐把聚类放在第三步之后。

- 输入：`block_deviation_sequences/*.csv`
- 过程：直接提取距离特征并聚类
- 优点：逻辑更清晰，复用已有距离计算
- 缺点：形式上不是“只用第三步结果”


---

## 17. 推荐的第一版实验设计

为避免一开始实验过重，建议第一版按以下方案执行：

### 17.1 样本

- 使用所有切片
- `grasp` 和 `release` 分开
- 成功、失败都参加

### 17.2 特征

仅使用“最小可用特征集合”的 10 个特征：

- `distance_median`
- `distance_q10`
- `distance_q90`
- `distance_std`
- `distance_slope`
- `diff_abs_mean`
- `distance_lag1_autocorr`
- `near_ratio_tau1`
- `longest_near_run_tau1_ratio`
- `time_to_first_near_tau1_ratio`

### 17.3 标准化

- 对数值特征做 z-score 标准化
- 距离先按目标对角线归一化

### 17.4 聚类

- 先试 `GMM(k=3,4,5)`
- 再试 `HDBSCAN`
- 比较结果可解释性

### 17.5 输出

至少输出：

- `cluster_feature_table.csv`
- `cluster_assignments.csv`
- `cluster_summary.csv`
- 每个簇的平均距离曲线图

---

## 18. 预期聚类结果解释

在第一版实验中，合理的簇通常可能呈现为以下几类：

1. **Focused / Stable**
   - 距离低
   - 波动小
   - 快速贴近目标并保持

2. **Task-related Search**
   - 初期距离大
   - 后期逐渐收敛
   - 波动中等

3. **Unstable / Wandering**
   - 距离中高
   - 波动大
   - 贴近目标时间短

4. **Low-quality / Noise**
   - 有效帧少
   - 置信度低
   - 曲线不稳定

注意：

- 这些簇不是先验真值
- 需要结合样本回看和统计表做人类解释

---

## 19. 文献依据

以下文献为本方案提供理论和方法参考。

1. Maxence Grand, Damien Pellier, Francis Jambon. *GAIPAT - Dataset on Human Gaze and Actions for Intent Prediction in Assembly Tasks*. arXiv, 2025.  
   链接：<https://arxiv.org/abs/2503.11186>

2. Marco Cuturi, Mathieu Blondel. *Soft-DTW: a Differentiable Loss Function for Time-Series*. Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 2017, pp. 894-903.  
   链接：<https://arxiv.org/abs/1703.01541>

3. Carl H. Lubba, Sarab S. Sethi, Philip Knaute, Simon R. Schultz, Ben D. Fulcher, Nick S. Jones. *catch22: CAnonical Time-series CHaracteristics*. Data Mining and Knowledge Discovery, 2019.  
   链接：<https://arxiv.org/abs/1901.10200>

4. Ben D. Fulcher, Max A. Little, Nick S. Jones. *Highly comparative time-series analysis: the empirical structure of time series and their methods*. Journal of the Royal Society Interface, 2013.  
   链接：<https://arxiv.org/abs/1304.1209>

5. Lei Shi, Cosmin Copot, Steve Vanlanduit. *What Are You Looking at? Detecting Human Intention in Gaze based Human-Robot Interaction*. arXiv, 2019.  
   链接：<https://arxiv.org/abs/1909.07953>

6. Bo Yang, Jian Huang, Xiaolong Li, Xinxing Chen, Caihua Xiong, Yasuhisa Hasegawa. *Natural grasp intention recognition based on gaze fixation in human-robot interaction*. arXiv, 2020.  
   链接：<https://arxiv.org/abs/2012.08703>

---

## 20. 最终建议

对于当前项目，推荐的聚类思路可以总结为：

- 使用切片级样本，而不是单帧样本
- 使用距离序列派生的固定长度特征，而不是直接零填充原始序列
- 对 `grasp` 和 `release` 分别聚类
- 让成功和失败样本都参加聚类
- 但原始失败样本的最终语义保持为“分心 / 非专注”
- 聚类主要用于发现成功样本内部的不同注意模式，以及解释失败样本的行为结构

如果后续需要实现代码，建议新增一个独立脚本，负责：

1. 从第三步结果构建距离特征表
2. 执行聚类
3. 输出簇分配与统计结果
4. 生成簇平均曲线与可视化图


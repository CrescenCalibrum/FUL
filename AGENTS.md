# AGENTS.md — 联邦遗忘实验项目上下文与协作规范

> 用途：供 Codex、Claude Code 等代码助手在本仓库中开发、排错、复现实验时阅读。本文是**此前讨论与已上传诊断结果的上下文快照**，不是对当前本地仓库的自动审计。修改代码前必须查看真实源码和最近一次 `runs/<run_id>/config.json`，如有冲突，以实际仓库与运行记录为准。

## 1. 研究目标与范围

- 课题：**差分隐私约束下联邦遗忘性能恢复与可验证机制研究及系统实现**（计算机技术专业硕士）。
- 研究内容一（当前实验主线）：**DP 类原型驱动的联邦遗忘性能恢复**。由可插拔遗忘算子生成 `M_unlearned`；使用类级特征均值、对角方差、类频等统计重建分类头，力求低开销恢复保留集性能，并维持遗忘有效性。
- 研究内容二（后续拓展）：**结构化 marker + 不变量 + 统计检验的可验证联邦遗忘**，关注已遗忘客户端标记退化、保留客户端标记稳定、置信区间与伪造遗忘检测。
- 研究内容三（系统演示方向）：电力设备故障诊断/预测性运维平台；本仓库首先是算法仿真实验底座，并非已建成业务系统。
- 关键研究问题：DP 噪声、非 IID、遗忘算子、特征表示漂移、原型失配、连续遗忘次数如何影响恢复上限、时间和存储开销。
- 开题报告：source\侯磊-2024140875-差分隐私约束下联邦遗忘性能恢复与可验证机制研究及系统实现.pdf
- 小论文：source\Differentially_Private_Class-Prototype-Driven_Performance_Recovery_for_Federated_Unlearning.pdf

## 2. 实验环境与项目惯例

- 目标环境：Ubuntu 22.04、Miniconda、Python 3.10、CUDA 11.8、PyTorch、Flower、NumPy、Pandas。
- 早期基座采用 Flower `NumPyClient`、`FedAvg`、`fl.simulation.start_simulation`；实际 Flower 版本及 API 以仓库依赖为准，不要盲目升级为 `ServerApp`/`ClientApp`。
- 已讨论/使用数据集：Fashion-MNIST、CIFAR-10；模型包括 SmallCNN、CIFAR 版 ResNet18（32×32 输入通常使用 3×3 stride=1 的 conv1 并移除初始 maxpool）。**早期参考 `build_model()` 只支持 SmallCNN，当前仓库是否真正支持 `--model resnet18_cifar` 必须核实**。
- 数据划分：IID 和 Dirichlet label-skew 非 IID，主要参数 `alpha`、`seed`、`num_clients`。
- 曾规划 CIFAR-10 + ResNet18-CIFAR、20/50 客户端、50–100 联邦轮次；**这属于实验规划，不等同于已全部完成**。

### 此前参考工程布局（必须与当前仓库核对）

```text
fulbase/
  data.py                    # 数据集、客户端划分及 DataLoader
  models.py                  # 模型与分类头/特征提取器
  train.py                   # 客户端本地训练与评估
  utils.py                   # 参数列表与模型 state_dict 转换
  flower/{client,sim,strategy}.py
  dp/prototype.py            # 裁剪、类统计、加噪、原型库
  unlearning/{base,registry,history_subtract,federaser_lite,pga,retrain_oracle}.py
  recovery/{federaser_replay,prototype_lda}.py
  experiment/{pipeline,io}.py
scripts/
  run_pipeline.py
  analyze_single_run_overhead.py       # 如果当前仓库已加入
  diagnose_feature_drift.py            # 已编写独立诊断脚本
runs/<run_id>/
```

## 3. 现有实验链路

```text
加载数据与固定客户端划分
  -> Flower FedAvg 训练获得 M_full
  -> 保存每轮历史更新与 model_full
  -> 构建用于恢复的类级特征原型
  -> 选择遗忘算子并产生 M_unlearned
  -> **必须记录并评估“遗忘后、恢复前”模型**
  -> 分别执行恢复方法/历史回放对照
  -> 保存模型、指标、配置和方法日志
  -> 可选：离线表示漂移诊断、资源开销统计
```

### 已纳入早期参考代码的遗忘算子

| `unlearner` | 原理 | 重要限制 |
| --- | --- | --- |
| `history_subtract` | 从模型扣除待遗忘客户端的历史加权更新 | 启发式近似，不保证等价于 Retrain；需确认是否真的有非零模型变化 |
| `federaser_lite` | 过滤遗忘客户端后，回放保留客户端的历史更新 | **FedEraser-style 简化回放**，不是未经核对就可称作原论文完整复现 |
| `pga` | 在遗忘客户端数据上受约束梯度上升；支持 `head`/`last`/`all` | 需要目标客户端数据/参与；遗忘效果必须另行验证 |
| `retrain_oracle` | 保留数据重新训练的参照 | 早期代码是**中心化 retained-data 重训简化版**，不等于严格 Flower/FedAvg 重训；论文参照需实现一致的 FL Retrain |

### 恢复方法与公平性

- `none`：记录 `M_unlearned`，这是所有恢复对比不可省略的起点。
- `prototype_lda`：利用 DP 类原型重建分类头，输入是对应算子的 `M_unlearned`；理论上只改分类头。
- `federaser_replay`：早期代码直接**从初始模型与训练历史做回放**，没有以该算子的 `M_unlearned` 为恢复起点。它应标记为**独立历史回放/重建对照**，**不能**宣称是针对每一种 `M_unlearned` 的同口径后处理；不同算子下其结果可能重复。
- 待加入的改进：Mean-only、Mean+Var+Prior、校准、严格 Retrain、可验证 marker。不要宣称已经实现。

## 4. DP 原型与隐私边界：必须诚实标注

此前参考 `dp/prototype.py` 的实际顺序：

1. 从**正常联邦训练结束后的完整模型**提取中间特征；逐样本按 `clip_norm=C` 做 L2 裁剪。
2. 每个客户端每类统计 `sum(z)`、`sum(z²)`、`count`。
3. 向 `sum` 添加标准差 `dp_sigma*C` 的高斯噪声；向 `sumsq` 添加标准差 `dp_sigma*C²` 的噪声；向 count 添加标准差 `dp_sigma` 的噪声。
4. 进程内加总统计；由 noisy sums/counts 得到均值、对角方差与类频，保存 `retained_dp_prototypes.pt`。
5. 早期代码在收集时使用 `exclude_clients=forget_ids`，即**在知道本次遗忘请求后重新访问保留客户端数据**，与课题目标“训练期形成统计、遗忘期不重新读取原始数据”仍有落差。

**严禁错误表述**：

- 目前这不是 DP-SGD：训练 `train.py` 仍是普通 SGD，`dp_sigma` 只作用于原型统计。
- 不能仅凭噪声倍率声称满足某个 `(epsilon, delta)`：没有完整敏感度分析、DP 会计与组合分析。
- 目前“安全聚合”只是模拟各客户端统计相加，不是密码学 SA。
- 历史回放存储每客户端完整参数，与“严格 SA 下服务器看不到单客户端更新”的设定冲突。可将其作为非 SA 历史法对照，并明确标注能力边界。
- 若连续遗忘，原型必须随累计保留集合、权限与版本正确更新；不能简单一直使用含已遗忘客户端贡献的全局总原型。
- 原型基于 `M_full` 的特征空间；如果 PGA 等改变 backbone，需要分析与 `M_unlearned` 的**原型失配**。

## 5. 训练历史快照、checkpoint 与资源开销

此前 `flower/strategy.py` 的历史记录：

```python
round_item = {
    "round": t,
    "global_before": [...],  # 本轮广播前全局模型完整参数
    "clients": [
        {"cid": cid, "num_examples": n,
         "params_after": [...], "metrics": {...}}
        # 仅本轮成功返回的参与客户端
    ]
}
```

- `params_after` 是**客户端训练后的完整模型参数**，不是梯度；可以由 `params_after - global_before` 得到历史更新。
- 参考实现训练时将每轮历史放在内存，训练结束时一次 `torch.save` 到 `history.pt`，有内存、磁盘开销风险。
- 每轮记录 1 份 `global_before` 和 K 份客户端模型时，粗略存储复杂度为 `O(T*K*P)`（T=轮数、K=每轮成功参与客户端数、P=模型参数量），还要算缓冲区和序列化开销。
- `models/model_*.pt` 是阶段 checkpoint，与 `history.pt` 的训练历史不同。
- 资源比较时分开记录：`recovery_time_sec`、`model_storage_mb`、`aux_storage_mb`、`total_storage_mb`；FedEraser-style 的辅助存储通常是 `history.pt`，Prototype-LDA 是 `retained_dp_prototypes.pt`。
- 当前脚本若没有记录真实恢复耗时应写 NaN，**不能用文件时间戳假装算法耗时**；不要重复计数共用 `history.pt`。

## 6. PGA 更新范围及已知陷阱

- 早期 `pga_layers=head` 按名称匹配 `classifier` 或 `fc.`；`last` 还匹配以 `layer4` 开头的参数；`all` 允许全部权重更新。
- 在早期 SmallCNN 上，`head` 和 `last` 都通常只选到 `classifier.*`，**不会自动选到 `fc1/fc2`**；在标准 ResNet18 命名下 `last` 预期为 `layer4 + fc`。
- **检查真实 `model.named_parameters()`**，不可假设参数命名与 torchvision 一致。
- 只设置 `requires_grad=False` **不会冻结 BatchNorm `running_mean/running_var` 等缓冲区**；若想仅更新所选参数，应明确处理 BN 状态，并对比前后模型的全部 `state_dict()` 键。
- PGA 的投影中心应是本次遗忘前的模型；投影只针对允许更新的浮点参数更合理，不能把 BN 的计数器缓冲区直接纳入 L2 投影。
- 使用 `itertools.cycle(dataloader)` 可能缓存 batch，注意内存。

## 7. 结果文件约定与指标

一次实验通常输出：

```text
runs/<run_id>/
  config.json
  metrics.csv
  history.pt
  retained_dp_prototypes.pt
  models/model_full.pt
  models/model_<unlearner>_pre_recovery.pt
  models/model_<unlearner>_federaser_recovery.pt
  models/model_<unlearner>_proto_lda_recovery.pt
  diagnostics/feature_drift_recovery.csv  # 运行诊断脚本后
  diagnostics/feature_drift_vs_recovery_gain.png
```

`metrics.csv` 中主要字段：`stage`, `unlearner`, `recovery`, `all_acc`, `retain_acc`, `forget_acc`, `retain_loss`, `forget_loss`, `retain_num_examples`, `forget_num_examples`，以及**若已加计时**才有 `recovery_time_sec`。

- `full_after_fl` = 正常联邦训练完成；`<algo>_pre_recovery` = 遗忘后恢复前；`<algo>_proto_lda_recovery` = 应用原型恢复后。
- 同一次实验中，如果 `All = Retain ∪ Forget` 且互不交叉：

  `all_acc = (N_r*retain_acc + N_f*forget_acc)/(N_r+N_f)`。

- 客户端级遗忘的 `forget_acc` **不是忘记与否的充分证据**：各客户端可共享标签与分布；应结合目标客户端成员推断、marker、与对应 Retrain 模型差距等证据。
- 论文比较优先报告恢复前/恢复后 `retain_acc`、遗忘有效性、与 Retrain 的差距、计算/存储/通信开销；不能只报告 `all_acc`。

## 8. 已经得到的一次表示漂移诊断结果（真实已上传 CSV）

数据来源：一次 `feature_drift_recovery.csv`；固定诊断参考集合为 **800 个保留客户端测试样本**。这是**单次运行**，不代表多随机种子的统计结论。

| Unlearner | Relative L2 drift | CKA | Class-center drift | `retain_acc`：恢复前 → 后 | Gain (百分点) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `history_subtract` | 0.000000 | 1.000000 | 0.000000 | 72.2801% → 71.7881% | -0.4919 |
| `federaser_lite` | 0.000884 | 0.999999 | 0.000412 | 72.2801% → 71.7996% | -0.4805 |
| `pga` | 0.140320 | 0.985812 | 0.132422 | 69.4543% → 69.1568% | -0.2974 |

其他关键发现：

- 原始完整模型 `retain_acc_full≈72.2801%`。
- 三组 `post_recovery_relative_l2_change=0`、`post_recovery_one_minus_cka=0`：Prototype-LDA 重建在该次运行中未改变 backbone 特征。
- `prototype_mismatch_full≈0.754141`：原型与原始模型测试集类中心的**基准失配较大**；检查原型统计时使用特征裁剪，而诊断提特征时未按同规则裁剪、DP 噪声、训练/测试分布差异及 LDA 假设。
- `excess_prototype_mismatch`：History Subtract 0，FedEraser-Lite 0.0000132，PGA 0.004218。
- 三组完整保留集 `retain_acc_gain` **都为负**；当前结果**尚未证明** Prototype-LDA 带来精度恢复。不能将样本子集上的偶然 +0.5 个百分点写成完整测试集恢复成功。
- History Subtract 漂移精确为零、FedEraser-Lite 接近原模型，需核对是否真正更改参数及是否有效遗忘；“漂移大”也不等于“忘得充分”。

诊断指标：

- `relative_l2_drift = ||Z_unlearned-Z_full||_F / (||Z_full||_F + eps)`：**主特征漂移指标**。
- `cosine_drift`：特征方向变化；`cka`、`one_minus_cka`：表示结构相似/变化。
- `mean_class_center_drift`：类别中心相对位移。
- `prototype_mismatch_full/unlearned`：类中心与保存原型的距离；`excess_prototype_mismatch`：遗忘额外造成的失配。
- `retain_acc_gain = retain_acc_recovered - retain_acc_pre`；`fraction_of_full_gap_closed` 只是补回与**原始 Full** 差距的比例，**不是基于 Retrain 的严格恢复率**；分母为零时 NaN 合理。

## 9. 计划补充的实验（未视为完成）

1. **恢复有效性排障（最高优先级）**：先做 `dp_sigma=0`，复核原型特征裁剪和分类头维度/先验/方差；用只更新分类头的 PGA-head 建立“无特征漂移”的可恢复基准；比较 Mean-only / LDA / 校准。
2. **表示漂移—恢复效果**：固定同一 `M_full`、保留/遗忘客户端、DP 原型与参考数据；比较 PGA-head、PGA-last、PGA-all，及不同 `pga_steps`、投影半径；绘制 `relative_l2_drift` 或原型失配 vs `retain_acc_gain`/到 Retrain 差距。报告不同 seed 的重复实验，勿用 3 个算法点做强因果断言。
3. **连续遗忘**：按步骤累积忘记不同客户端；每一步记录当前 `M_unlearned_pre_recovery`、恢复模型及**累计遗忘集合的有效性**；不同恢复策略必须维护独立的连续模型轨迹。FedEraser-style 回放需从头排除累计遗忘集合；不能把它当成基于上一步 `M_recovered` 的增量更新。
4. **隐私与非 IID 消融**：`dp_sigma`、Dirichlet `alpha`、遗忘比例、客户端规模、不同随机种子；只有建立隐私会计后才能报告明确的 DP 预算。
5. **研究内容二**：客户端/群组 marker、间隔及类中心关系不变量、统计检验/置信区间、假遗忘和过度遗忘检测。

## 10. 给代码助手的执行规范

1. **先检查当前文件树和真实函数签名**。本文路径和参数名来自之前参考工程；遇到不一致不得直接覆盖正在使用的版本。
2. 优先使用**新增独立脚本**分析某次 `--run-dir runs/<run_id>` 的记录，不要默认递归扫描全部 `runs/*` 或重新训练 Flower。
3. 保持 `M_full`、`M_unlearned_pre_recovery`、`M_recovered` 三阶段及其结果文件；不要把恢复前评估合并掉。
4. 所有方法的比较必须复用**相同数据划分、遗忘集合、Full Model、模型结构、随机种子**；改变某一变量时说明其他变量是否固定。
5. 不要将示例 `federaser_lite/replay` 直接描述为论文原版 FedEraser；不要把中心化 retrain 当标准 FL Retrain；不要声称实现了密码学 SA 或经会计证明的 DP。
6. 明确标注哪些指标是**测得的**、哪些是理论/预期；旧日志缺耗时字段就保留缺失值，严禁捏造时间、精度、隐私预算或图表数据。
7. 审计存储时注意 `history.pt` 很大，Ray 并发/显存/内存/磁盘压力；先在小规模 smoke test 验证，再扩大轮次、客户端数量。
8. `torch.load(..., weights_only=False)` 仅允许加载本地可信模型文件；不要加载来源不明的 pickle/checkpoint。
9. 修改 PGA / 原型路径时同时增加验证：加载 checkpoint 前后参数、BN buffer 的变化、每层可训练参数列表、原型特征维度/裁剪的一致性。
10. 对后续连续遗忘和研究内容二，应先给出实验协议、评估定义及合法数据访问约束，再修改代码；将未实现功能明确标记为 TODO。

## 11. 本地常用命令（以当前仓库支持的选项为准）

```bash
# 小规模遗忘算子功能检查
python scripts/run_pipeline.py \
  --dataset fashionmnist --num-clients 10 --num-rounds 3 \
  --forget-clients 0 --unlearner all --pga-steps 20 \
  --retrain-epochs 2 --run-name unlearner_smoke_test

# 对某次完整实验离线诊断表示漂移；不重新训练
python scripts/diagnose_feature_drift.py \
  --run-dir runs/<run_id> --per-class 80 --batch-size 128 --device auto

# 如果已放入开销统计脚本，针对单次实验而不是递归扫描所有实验
python scripts/analyze_single_run_overhead.py --run-dir runs/<run_id>
```

变更实验逻辑后，至少确认：代码可导入/编译、smoke test 能跑、三个模型阶段都有输出、ACC 加权恒等式成立、Full checkpoint 与实际 FedAvg 输出一致、诊断所用特征层与 DP 原型采集层一致。

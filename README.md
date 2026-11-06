# ful_flower_base：联邦遗忘统一实验底座

2026-11-07 已将37个系统设备模块集中到 [`fulbase/system/`](fulbase/system/README.md)，原通用实验实现留在原位置。公共Python导入名保持兼容；训练、数据、模型和历史验收记录没有因整理目录被重写。

2026-11-07 已将系统开发新增的 Python 脚本集中到 [`scripts/system/`](scripts/system/README.md)。6 个原实验入口保留在 `scripts/` 根目录；系统启动器、Worker 和测试已使用新位置，历史源码证据另按原哈希保存。

2026-11-06 已完成C8：Windows/CPU集成回归、成组备份/空库恢复、构建产物部署、页面真实合成smoke及重启验收。原库/历史业务和研究结论保持；长路径和代理断连问题已修复。详见[最终验收](docs/system-design/c8-delivery-acceptance.md)、[部署演示手册](apps/deploy/README.md)和[统一进度](docs/system-design/PROGRESS.md)。后续重点为演示彩排与论文材料整理。

2026-11-06 完成 C7.17 保留侧参照校准失配定位与研究结论收口：404在三个既有批次上的保留客户端1/3差距持续，客户端1集中在类别0/1，客户端3还存在较大类内波动；原区间超界属于等效证据不足。该条件由Full与Retained-FL决定，不能通过调PGA直接修复。原严格2/3、历史开发3/3、恢复增益0及参照成本边界分别保留。39Python、57非MySQL、6MySQL、10752实际输出/36统计/3矩阵独立复算与API/页面/下载/重启通过；15旧导出及业务保持，V17诊断7份。C7已声明演示范围工程与证据收口，正式研究结论仍不足。 详见[验收记录](docs/system-design/c7-17-closure-acceptance.md)及[统一进度](docs/system-design/PROGRESS.md)。普通启动方式保持；报告无需Worker。在“设备标记校准”打开C7.12父记录，下方查看收口诊断。下方旧阶段保留原验收日期。

系统当前状态、已验收里程碑、遗留问题和后续开发日志统一记录在 [系统开发进度](docs/system-design/PROGRESS.md)。后续系统开发须按 [AGENTS.md](AGENTS.md) 在任务结束前同步更新该文件。

研究点二的结构化 marker、严格 FL Retrain、伪造对照与统计审计已加入独立入口，运行方法和当前边界见 [研究点二实现说明](docs/verification.md)。默认审计阈值仍需开发校准。

系统演示已完成 A1–A6 和 B1–B5：任务队列、Worker、合成 smoke 页面闭环及异常恢复，并支持执行上下文、模型版本/血缘和独立 TRAIN_FULL。管理员可在“独立训练与模型”登记上下文，只训练并保存 Initial → Full，不自动遗忘、恢复或审计。启动与真实验收见 [B1 说明](docs/system-design/b1-train-full.md)，故障恢复见 [A6 验收](docs/system-design/a6-task-recovery.md)。B2 可在页面选择已发布 Full，独立执行 PGA-head 并生成 Pre、恢复前评估及 Full → Pre 血缘，不重训 Full。见 [B2 实现与实测](docs/system-design/b2-pga-unlearning.md)。B3 已接入独立原型构建和 RECOVER，原型可复用，保留 Initial → Full → Pre → Recovered，详见 [B3 实现与实测](docs/system-design/b3-prototype-recovery.md)。B4 已支持独立开发审计，分别保存 Pre/Recovered 报告、原始观测和不足原因，见 [B4 实现与实测](docs/system-design/b4-independent-audit.md)。B5 已增加统一操作工作台、按血缘查询的三阶段比较和三种独立方法对照，见 [B5 使用与兼容验收](docs/system-design/b5-unified-operations.md)。其他方法的原型恢复/审计仍按特征兼容边界拦截；设备算法另行适配。

这个工程用于先跑通：

```text
Flower FedAvg 正常训练
→ 黑盒遗忘算子 history_subtract
→ 记录“遗忘后、恢复前”模型结果
→ FedEraser-style replay 恢复 baseline
→ DP prototype + LDA 分类头重建（你的方法）
→ 统一输出 metrics.csv 与模型文件
```

当前代码是“最小可运行底座”，重点是把实验链路、目录结构、接口和结果记录打通；后续可替换遗忘算子、DP 加噪机制、结构化 marker 验证模块、更多 baseline。

---

## 1. 环境安装

现有环境：Ubuntu 22.04 + Miniconda/conda3 + Python 3.10 + CUDA 11.8。

```bash
conda create -n ful_flower python=3.10 -y
conda activate ful_flower

# CUDA 11.8 版本 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Flower 与其他依赖
pip install -r requirements.txt
pip install -e .
```

检查：

```bash
python - <<'PY'
import torch, flwr
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('flwr:', flwr.__version__)
PY
```

---

## 2. 最小运行

默认使用 Fashion-MNIST、10 个客户端、Dirichlet 非 IID、5 轮联邦训练、忘记 client 0。

```bash
python scripts/run_pipeline.py \
  --dataset fashionmnist \
  --num-clients 10 \
  --num-rounds 5 \
  --local-epochs 1 \
  --partition dirichlet \
  --alpha 0.5 \
  --forget-clients 0 \
  --dp-sigma 0.0 \
  --run-name fm10_debug
```

如果服务器 CPU 核数不够，可以降低并发：

```bash
python scripts/run_pipeline.py --num-clients 10 --client-cpus 2 --run-name low_parallel
```

说明：多客户端仿真默认让客户端在 CPU 上训练，避免 Ray/Flower 多个 worker 同时抢同一块 GPU 导致 OOM。评估和原型计算会在可用时使用 CUDA。若你确定显存足够，可尝试：

```bash
python scripts/run_pipeline.py --client-device cuda --client-gpus 0.1 --run-name gpu_clients
```

---

## 3. 结果在哪里看

每次运行会生成一个目录：

```text
runs/YYYYMMDD-HHMMSS_你的run-name/
├── config.json
├── metrics.csv
├── history.pt
├── retained_dp_prototypes.pt
└── models/
    ├── model_full.pt
    ├── model_unlearned_pre_recovery.pt
    ├── model_federaser_recovery.pt
    └── model_proto_lda_recovery.pt
```

查看指标：

```bash
cat runs/*fm10_debug*/metrics.csv
```

重点看 `metrics.csv` 中 4 行：

- `full_after_fl`：正常 FL 完成后的模型；
- `unlearned_pre_recovery`：遗忘之后、恢复之前的模型；
- `federaser_recovery`：FedEraser-style 历史回放恢复 baseline；
- `prototype_lda_recovery`：你的 DP 类原型 + LDA 头重建方法。

重点列：

- `retain_acc`：保留客户端测试集准确率；
- `forget_acc`：被遗忘客户端测试集准确率；
- `all_acc`：总测试准确率；
- `retain_loss / forget_loss`：保留/遗忘集合损失。

---

## 4. 后续替换点

### 替换遗忘算子

当前占位实现：

```text
fulbase/unlearning/history_subtract.py
```

后续可以新增：

```text
fulbase/unlearning/pga.py
fulbase/unlearning/federaser_unlearn.py
fulbase/unlearning/quickdrop.py
```

然后在 `fulbase/experiment/pipeline.py` 的 `[4/6] Black-box unlearning operator` 部分替换调用。

### 替换恢复 baseline

当前 FedEraser baseline：

```text
fulbase/recovery/federaser_replay.py
```

你的方法：

```text
fulbase/recovery/prototype_lda.py
fulbase/dp/prototype.py
```

### 替换 DP 加噪 / 安全聚合仿真

当前 DP 类原型统计位置：

```text
fulbase/dp/prototype.py
```

当前做的是“本地裁剪 + 噪声 + 聚合”的仿真。真实系统中可把 `compute_client_feature_stats` 放到 Flower 客户端侧，把 noisy stats 作为自定义消息或额外记录上传；服务器只保留聚合后的 noisy 原型。

---

## 5. 建议的第一批实验命令

### IID sanity check

```bash
python scripts/run_pipeline.py --partition iid --num-rounds 3 --forget-clients 0 --run-name iid_sanity
```

### 非 IID 主实验

```bash
python scripts/run_pipeline.py --partition dirichlet --alpha 0.5 --num-rounds 10 --forget-clients 0 --run-name noniid_a05
```

### DP 噪声对比

```bash
for s in 0.0 0.5 1.0 2.0; do
  python scripts/run_pipeline.py --dp-sigma $s --num-rounds 10 --forget-clients 0 --run-name dp_sigma_$s
done
```

### 遗忘多个客户端

```bash
python scripts/run_pipeline.py --num-clients 20 --num-rounds 10 --forget-clients 0,1,2 --run-name forget_3_clients
```


## C7.12 语义保护与参照对齐（2026-11-06）

2026-11-06 完成 C7.12 保护约束语义校准与目标对齐：严格/方向保护×间隔/参照目标四组对照，预声明参照辅助组接受全新校准与留出，16/16＋10/10满足原开发门槛。Pre参照差距0.179064、同时区间[0.111842,0.246287]，原±0.25不变；单seed、人工标记、依赖重训参照且区间接近边界，正式遗忘仍不足。自然三阶段ACC100%、恢复增益0，未发布业务模型。

页面 `/#/equipment-calibrations?run=776e7a41-f5b0-4d98-9374-0ccc8f235084`。新增协议 `cwru_marker_semantic_v12`，复用V16研究表，数据库保持V17；ADMIN登记，ADMIN/AUDITOR项目读取，OPERATOR拒绝，下载核验原字节。工程实现包括四组算法、冻结执行与新样本协议、独立数值复核、后端来源/算术/统计验证和页面。八旧研究及两诊断导出、165旧清单/工件和业务记录保持。

下一步建议 **C7.13：冻结候选独立复验与参照依赖评估**，先冻结独立训练种子、新观察及汇总规则；若修改目标或比例边界求解器，另立协议，不能用本次留出调参或默认继承pass。


## C7.13 独立复验（2026-11-06）

2026-11-06 完成 C7.13 冻结候选独立复验与参照依赖评估：101/202/303三次新训练、每次两组目标、全新校准/隔离留出，原求解器和历史结果保持。按预声明48/60/6项比较族，仅202参照辅助组满足全部条件（1/3）；101校准先决条件不足，303目标参照等效不足。自然三阶段ACC均100%、恢复增益0；额外参照训练约110–142秒，仍是需要Retained-FL的研究对照。29 Python、42后端非MySQL、6 MySQL及独立数值/API/页面/下载/重启验收通过。

页面在C7.12父研究记录下展示独立复验，可登记、查看六组及Pre/Recovered、导出冻结JSON；管理员写、审计员只读。原始128工件及99项私有证据绑定，新增1份V17研究诊断，业务任务/模型保持。202中断恢复的原型初始耗时缺失已显式标注。

下一步建议 **C7.14：跨训练校准稳定性与目标对齐泛化诊断**，分别诊断101保留侧校准与303目标泛化/停止问题；修改候选必须另立协议和全新样本，不改写本轮结果。

## C7.14 泛化诊断（2026-11-06）

2026-11-06 完成 C7.14 跨训练校准稳定性与目标对齐泛化诊断：保留C7.13三个训练重复及两目标模型，新增三批共享诊断探针，完成逐类/尺度/交叉离差和停止分解。303训练差距已约0.301，101/303参照方向被保留比例下界截断；101校准上界0.250992显示边界敏感。原1/3复验结论保持，无新训练、候选或审计。45 Python、37后端非MySQL、6 MySQL、32256实际输出独立复算及API/页面/下载/重启通过。

在父C7.12研究页面查看新增“跨训练校准稳定性与目标对齐泛化诊断”区块。来源登记复用V17，API为`/api/v1/equipment-calibrations/{calibration}/generalization-diagnostics`；ADMIN登记、AUDITOR只读、OPERATOR拒绝。原报告与业务模型保持。

下一步建议 **C7.15：比例边界联合可行方向与新候选验证**。先在训练侧验证纳入比例边界后是否存在可行下降方向；若形成新候选，先冻结协议，再使用全新校准和留出，保持原预算与阈值。101参照校准稳定性仍单独保留。

## C7.15 联合比例方向候选（2026-11-06）

2026-11-06 完成 C7.15 比例边界联合可行方向与新候选验证：复用C7.13三个Full/参照/训练探针及原型，新增联合比例边界求解器；101/303旧停止点存在原预算下的下降见证。新候选87/110/127步、双进程精确重复；全新校准和隔离留出下新组3/3满足、原组同批1/3，原阈值±0.25不变。33792输出、84行自然指标、554步轨迹独立复算通过；56个不同Python用例、49非MySQL、6MySQL、API/页面/下载/重启通过。保留旧结果与13份历史导出，V17诊断增至5，业务保持。参照依赖、自然恢复增益0及101校准边界敏感仍保留。

页面入口为设备标记训练与校准中的C7.12父研究，新增联合方向候选面板；接口为`/api/v1/equipment-calibrations/{parent}/joint-candidates`，管理员登记、审计员只读。

下一步建议 **C7.16：联合方向候选跨训练独立复验与参照成本边界**。冻结本轮候选，在全新训练重复和全新角色上验证，单列保留侧校准稳定性；现有三个Full上的开发通过不能替代跨新训练或正式遗忘证明。

## C7.16 跨新训练复验与参照成本（2026-11-06）

2026-11-06 完成 C7.16 联合方向候选跨训练独立复验与参照成本边界：冻结求解器，在404/505/606全新训练上生成六组候选、新校准和真实隔离留出。联合组留出均10/10，但404保留客户端1/3校准不足，严格复验2/3，旧组同批1/3；原±0.25保持。参照训练142–150秒、联合操作约7秒，必需计算149–158秒，比直接采用同一重训参照多约4.9%，没有整体加速结论。67Python、65非MySQL、6MySQL、33792输出/548步复算及API/页面/下载/重启通过；14旧导出与业务保持，V17诊断增至6，正式结论不足。

报告接口：`/api/v1/equipment-calibrations/{parent}/transfer-studies`；ADMIN可登记；ADMIN和AUDITOR可查询、导出；OPERATOR不可访问。前端在设备标记训练与校准父研究中展示六组阶段、全部不足原因和真实参照成本；冻结JSON下载校验SHA。

下一步建议 **C7.17：保留侧参照校准失配定位与研究结论收口**。先分解404的客户端1/3校准不足，保持原阈值与全部本轮结果；明确参照辅助研究对照的成本和使用边界。已有留出仅作事后解释，若修改候选须另立协议并使用全新样本，不能改写本轮2/3。

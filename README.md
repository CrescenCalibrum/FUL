# ful_flower_base：联邦遗忘统一实验底座

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

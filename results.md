# 实验结果

## 1、主实验

| stage            | all_acc | retain_acc | forget_acc | retain_gain | retrain_gap |
| ---------------- | ------- | ---------- | ---------- | ----------- | ----------- |
| Pre-Recovery     | 0.6861  | 0.6945     | 0.6274     | 0           | 0.0283      |
| FedEraser Replay | 0.7066  | 0.7179     | 0.6282     | 0.0234      | 0.0049      |
| DP Prototype     | 0.7045  | 0.7152     | 0.6307     | 0.0207      | 0.0076      |
| Retrain          | 0.7106  | 0.7228     | 0.6258     | 0.0283      | 0           |

## 2、不同遗忘方法

PGA

| stage            | retain_acc |
| ---------------- | ---------- |
| Pre-Recovery     | 0.6945     |
| FedEraser Replay | 0.7179     |
| DP Prototype     | 0.7152     |
| Retrain          | 0.7228     |

FedEraser

| stage            | retain_acc |
| ---------------- | ---------- |
| Pre-Recovery     | 0.6974     |
| FedEraser Replay | 0.7186     |
| DP Prototype     | 0.7143     |
| Retrain          | 0.7228     |



## 3、不同噪声强度

DP Prototype

| DP σ | retain_acc |
| ---- | ---------- |
| 0    | 0.7178     |
| 0.25 | 0.7152     |
| 0.5  | 0.7113     |
| 1.0  | 0.6989     |

FedEraser Replay

| DP σ | retain_acc |
| ---- | ---------- |
| 0    | 0.7235     |
| 0.25 | 0.7179     |
| 0.5  | 0.7092     |
| 1.0  | 0.6931     |

## 4、遗忘后的表征漂移

| PGA Steps | Feature Drift | Cosine Drift | CKA     | \(A_{\mathrm{retain}}^{\mathrm{pre}}\) | \(A_{\mathrm{retain}}^{\mathrm{stale}}\) | \(A_{\mathrm{retain}}^{\mathrm{matched}}\) |
| --------- | ------------- | ------------ | ------- | -------------------------------------- | ---------------------------------------- | ------------------------------------------ |
| 50        | 0.040319      | 0.0090       | 0.98581 | 0.7038                                 | 0.7173                                   | 0.7181                                     |
| 100       | 0.088451      | 0.0158       | 0.97419 | 0.6945                                 | 0.7152                                   | 0.7187                                     |

## 5、效率对比

| Unlearning | Recovery         | Time       | Total Storage |
| ---------- | ---------------- | ---------- | ------------- |
| PGA        | FedEraser Replay | 7.281 s    | 5.75 GB       |
| PGA        | DP Prototype     | 1.093 s    | 63.41 MB      |
| FedEraser  | FedEraser Replay | 5.219 s    | 5.75 GB       |
| FedEraser  | DP Prototype     | 0.984 s    | 63.89 MB      |
| Retrain    | —                | 106.62 min | —             |

## 6、消融实验

| method                 | retain_acc |
| ---------------------- | ---------- |
| Pre-Recovery           | 0.6945     |
| Prototype              | 0.7108     |
| Prototype + LDA Only   | 0.7141     |
| Prototype + LDA + Bias | 0.7152     |
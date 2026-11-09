# 研究点二主实验数据表契约

2026-11-09；主实验v1，尚无新benchmark结果。不填模拟实测值。

| 表 | 行单位 | 指标 | 来源与汇总规则 |
| --- | --- | --- | --- |
| 方法任务性能 | cohort×seed×method | retain/forget/all ACC、到Full/Retrain差距、3pp/3.5pp状态、阶段耗时 | campaign/tables/benchmark_metrics.csv，来源checkpoint_manifest与metrics.csv；45绑定不丢行 |
| 联合marker与消融 | cohort×seed×method | 主分层NI+anchor、分层等效、pooled、accuracy、几何、Full资格、失败/不足 | 新audit analysis/summary.csv和report.json；不把不足计作成功检出 |
| 客户端与类别 | seed×method×client×class | family配对间隔差、区间、类内/类间方差、marker ACC | 新audit统计明细；family不是训练重复 |
| 开销 | source/run×stage×operation | 实测时间、checkpoint字节、辅助历史存储 | benchmark_metrics的已记录计时、shared_auxiliary_storage.csv每seed单列history/initial；共用工件不重复加总、缺失留空 |

最终表格由真实结果生成；development43–45与confirmation46–47各自汇总，保留逐seed行。主比较40请求、45阶段绑定，不称45独立训练模型。

# 赛题与模型

## 任务

百度 2026 CTI 生成式推荐广告排序推理性能优化赛道要求参赛者在给定模型、权重和测试数据上输出 CTR 概率，在保证 AUC 与 PCOC 的同时压缩端到端推理时延。

官方页面：[AI Studio Competition 1461](https://aistudio.baidu.com/competition/detail/1461)

## 模型结构

```text
28 路变长 sparse feature
        ↓
Embedding pooling × 28
        ↓
RepEncoder: LayerNorm(14336) + Linear(14336 → 512)
        ↓
Transformer Encoder × 8
  ├─ causal self-attention
  └─ Top-2 Sparse MoE, 8 experts
        ↓
Linear(512 → 1) + sigmoid
        ↓
按 pred_mask 收集 logid 与概率
```

输入既包含用户历史序列，也包含待预测曝光。每个 slot 的 sign 数量可变，用户序列长度也可变，因此性能问题不只是矩阵乘法，还包括大量小 tensor、动态 shape、padding、H2D 和 Python 调度。

## 指标

线上记录使用以下计分关系：

```text
score_latency = (300 - latency) / 300

score_model = (
    (AUC - 0.65) * 1000
    + (0.15 - abs(PCOC - 1)) / 0.15 * 10
) / 360

score_all = score_latency * 70 + score_model * 30
```

- AUC 衡量排序能力，越高越好。
- PCOC 是预测点击率与真实点击率之比，越接近 1 越好。
- 时延占总分权重 70%，但不能通过少算输入、跳层或把真实计算移出计时区获得。

## 计时边界

最终使用的端到端口径覆盖：

1. batch 搬到 GPU；
2. 模型 forward；
3. sigmoid；
4. pred_mask 筛选；
5. GPU 到 CPU；
6. Python list 收集。

这也是本项目强调“布局、搬运和调度”的原因。只 profile `model(batch)` 会漏掉推荐系统推理中很大一部分固定开销。

## 环境

最终提交面向 A800、CUDA 12.6.3、Ubuntu 20.04、PyTorch 2.6.0 和 Triton 3.2.0。准确依赖见 `online_best/requirements.txt`。

## 基线口径

- `baseline/`：官方原版代码，赛后确认时延约 230s。
- 22.72s：初赛阶段已优化版本，之后作为决赛参考基线。
- `online_best/`：D150c，最终确认时延 9.63042s。

三者不能混为同一基线。

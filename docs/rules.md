# 合规边界与工程风险

本页用于帮助读者区分“跑得更快”和“少跑了计算”。具体以比赛最终官方规则为准。

## 不应采用

- 对输入 sampling、subset、截断或跳 batch；
- 减少 Transformer 层数、Attention 层或有效 expert 数；
- 把真实 forward、H2D 或全量输入处理搬到计时区外；
- 缓存 logid 到概率的最终预测结果；
- 篡改计时、异步逃逸或输出常量；
- 依赖未初始化内存或错误输出顺序获得偶然成绩。

## 通常属于工程优化

- FP16/BF16/INT8 等经过精度验证的低精度路径；
- Triton/CUDA fused kernel；
- packed sparse layout、pinned memory 和白名单 H2D；
- FlashAttention/SDPA 和内存 buffer 复用；
- 保持 Top-K 语义的 MoE grouped/sparse kernel；
- 在真实计时路径中执行的动态 batching；
- 数学等价的 bias、scale 或权重融合。

## 本项目中的敏感点

### Cached layout prebuild

CPU metadata/layout 的预构建是否计时取决于 evaluator 边界。公开学习时应把真实业务计时定义写清楚，不能通过改变计时口径宣称加速。

### MoE dim_ff pruning

`online_best` 设置 `_MOE_PRUNE_DIM_FF = 192`。它减少有效 FFN channel，属于近似压缩而不是纯算子等价改写。虽然该版本线上 AUC/PCOC 达标，但其他比赛可能明确禁止此做法。迁移前应：

1. 阅读目标规则；
2. 保留未裁剪对照；
3. 做逐版本 AUC/PCOC 回归；
4. 在项目说明中明确披露。

### Batch grouping

Grouping 必须保持所有 token、pred_mask 和 logid，合并成本也应计入真实推理路径。输出顺序不能依赖偶然的 batch 排列。

## 公开仓库边界

本仓库只发布代码与技术总结，不发布模型权重、数据、标签、cached batches、平台凭据和浏览器会话。官方 baseline 和赛题文字的权利仍归原权利方所有，参见根目录 `NOTICE.md`。

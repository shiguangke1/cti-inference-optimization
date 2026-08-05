# 从 230s 到 9.63s：有效优化路径

这份文档不按版本号复述实验，而是按推理系统的真实数据流解释最终有效的优化。核心结论是：23.88 倍加速不是来自一个神奇 kernel，而是连续消除“小 tensor + Python 循环 + 重复搬运 + 动态调度”造成的系统性损耗。

## 1. 先识别 baseline 的开销形态

官方 baseline 有四个明显特征：

1. 28 个 sparse slot 分别保存 values 和 offsets；
2. RepEncoder 在 Python 中逐 slot pooling；
3. Attention 手写 score、causal mask、softmax 和 value matmul；
4. MoE 在 Python 中逐 expert 运行，并产生较大的 dense 中间 tensor。

对应代码入口：

- [`baseline/make_collate_fn`](../baseline/infer.py#L185)
- [`baseline/move_batch_to_device`](../baseline/infer.py#L233)
- [`baseline/RepEncoder`](../baseline/infer.py#L244)
- [`baseline/scaled_dot_product`](../baseline/infer.py#L269)
- [`baseline/SMoE`](../baseline/infer.py#L311)

这类工作负载往往同时受 kernel launch、PCIe 事务、显存带宽和 Python 调度限制。只调一个 GEMM 参数不会产生数量级提升。

## 2. Flat packed sparse layout

### 问题

baseline 每个 batch 持有 28 组 `(values, offsets)`。递归搬运会产生大量小 H2D，Embedding pooling 也要重复进入 Python 和框架算子。

### 做法

`online_best` 把所有 slot 组织成三个连续 tensor：

```text
_flat_values       所有 slot 的 sign id
_flat_row_offsets  每个 token 在各 slot 中的行边界
_val_offsets       每个 slot 在 flat_values 中的起点
```

布局在 [`make_collate_fn`](../online_best/infer.py#L395) 和 [`_prebuild_cached_batch_layout`](../online_best/infer.py#L139) 中建立。

### 为什么有效

- 小 tensor 数量从每 batch 数十个降到少数连续 tensor；
- H2D 事务减少；
- 28 个 slot 可以由同一个 GPU kernel 处理；
- 后续算子可以依赖稳定的 offsets 接口，不再解析 Python tuple。

### 可迁移经验

推荐系统的第一优化对象通常不是网络层，而是 sparse feature 的物理布局。先把“许多变长小表”变成“一个 packed buffer + 边界描述”，再谈 kernel。

## 3. 白名单 H2D 与紧凑 dtype

### 问题

baseline 的递归 `move_batch_to_device` 会搬运标签、原始 slot tuple 和中间元数据，其中很多字段并不参与 forward。

### 做法

[`move_model_inputs_to_device`](../online_best/infer.py#L742) 只搬运模型必需字段：flat values、offsets、user offsets 和 attention metadata。CPU 侧索引尽量保持 int32/pinned memory，GPU 侧需要时再转换。

### 为什么有效

- 减少 PCIe 字节数和 transaction 数；
- 避免重复搬运已经被 flat layout 替代的字段；
- non-blocking copy 更容易与 GPU 工作重叠；
- batch 的 Python 结构更小，遍历成本更低。

### 可迁移经验

不要把“一个 batch 字典”默认等同于“模型输入”。先写出严格的输入白名单，再决定每个 tensor 的 CPU dtype、pin 策略和 GPU dtype。

## 4. Fused Embedding 与 RepEncoder

### 问题

28 路 EmbeddingBag 是典型的 memory-bound 推荐系统算子。baseline 逐 slot lookup、segment reduction、拼接，框架调度成本和中间 tensor 写回都很高。

### 做法

- [`fused_all_slots_embedding_kernel`](../online_best/infer.py#L794) 一次完成 28 路 lookup 与 pooling；
- [`layernorm_forward_kernel`](../online_best/infer.py#L830) 处理 RepEncoder 的 14336 维 LayerNorm；
- BF16 降低 embedding 输出和主干激活带宽。

### 为什么有效

合并 slot 后，kernel 可以沿连续 buffer 读取，并直接写出 `[tokens, 28 × 512]` 表示，省掉多次 launch、segment_reduce 和 cat。

### 可迁移经验

Embedding 优化的优先级通常是：布局合并 > launch 合并 > 访存合并 > 数值量化。不要一开始就上 INT8；若 pooling 很短，额外 dequant 开销可能比节省的带宽更贵。

## 5. Packed SDPA Attention

### 问题

baseline 手写 attention 会显式生成 score 和 causal mask，并执行 masked_fill、softmax 和 matmul。变长序列还会引入 padding 与 scatter/gather。

### 做法

[`scaled_dot_product`](../online_best/infer.py#L1133) 使用 PyTorch `scaled_dot_product_attention`，把用户序列按 offsets 描述并走 packed 处理。最终配置使用单桶边界 `(4096,)`，同时复用 attention scratch buffer。

### 为什么有效

- SDPA 可以选择 fused/FlashAttention 后端；
- 不显式持久化完整 attention score 与 mask；
- 单桶 identity path 避免多桶 scatter/gather；
- buffer 复用减少 8 层中的重复分配。

### 可迁移经验

变长 attention 不能只追求 padding 最少。桶越多，重排和 gather 越多；序列较短、层数较多时，简单稳定的 packed 路径可能胜过复杂分桶。

## 6. Packed Sparse MoE

### 问题

baseline 按 expert 循环并用通用 bmm/einsum 计算。Top-2 路由实际只激活少量 expert-token 对，但 dense 中间表示仍带来较大开销。

### 做法

`online_best` 用 Triton kernel 完成：

1. Top-2 gate；
2. route count 与 prefix；
3. token-expert pack；
4. sparse fc1/fc2；
5. 加权 reduce。

主要实现位于 [`moe_route_count_kernel`](../online_best/infer.py#L1206) 到 [`fused_top2_gate_kernel`](../online_best/infer.py#L1625)，模型入口是 [`SMoE`](../online_best/infer.py#L1854)。

### 为什么有效

- 计算和内存访问围绕真实激活路由组织；
- 移除 Python expert loop；
- 减少 zero-work 和 dense 中间 tensor；
- 将多个小 gate/pack 操作融合为较少 kernel。

### 关于 dim_ff=192

最终代码还设置 `_MOE_PRUNE_DIM_FF = 192`。这是有精度与合规风险的压缩，不属于纯数学等价优化。它在本次线上结果中通过了 AUC/PCOC 检验，但迁移到其他比赛或生产系统前，必须重新确认规则并做离线回归。

## 7. 计时区内 lazy batch grouping

### 问题

即使单 batch forward 已经很快，2039 次循环中的 H2D、kernel launch、sigmoid、mask、CPU copy 和 `tolist` 仍会累积成明显固定成本。

### 做法

D150c 在真实计时路径中按长度和资源预算组合相邻 cached batches：

```text
bucket_size    = 128
max_users      = 512
max_attn_slots = 300000
max_tokens     = 300000
max_preds      = 300000
```

核心逻辑位于 [`_timed_lazy_ensure_group_plan`](../online_best/infer.py#L512)、[`_timed_lazy_merge_group_cpu`](../online_best/infer.py#L581) 和 [`_timed_lazy_make_group_runtime_batch`](../online_best/infer.py#L634)。

### 为什么有效

- 减少 DataLoader/缓存 batch 的循环次数；
- 减少 H2D 和后处理调用次数；
- 预算上限约束 attention shape，避免为减少 batch 数而让 padding 爆炸；
- 保留 logid 与 pred_mask，最终输出仍能正确对齐。

### 可迁移经验

动态 batching 的目标不是“batch 越大越好”，而是同时控制 users、tokens、attention slots 和预测数。只有把合并成本放在真实计时路径中测量，收益才可信。

## 8. 低风险收尾优化

- BF16：降低激活与权重带宽，同时适配 A800；
- inference mode、TF32 和后端 autotune：减少框架开销并使用硬件快路径；
- [`final_linear_clamp_kernel`](../online_best/infer.py#L862)：融合最终 linear、clamp 与偏置；
- PCOC scale 折入 bias：避免额外逐元素缩放；
- 输出收集按 CPU/GPU 分工，保持 logid 对齐。

这些优化单项不一定大，但位于每 batch 或每层重复路径中，叠加后有价值。

## 9. 已证伪、无需重复堆叠的方向

| 方向 | 观察 | 工程结论 |
|---|---|---|
| Dataset/import-path grouping | 142 的 9.77292s 变为 D146 的 10.01961s | 没命中线上 cached-batch 主路径，且增加组织成本 |
| pred-only final head | 相比 J113 慢约 0.36-0.40s | final head 不是瓶颈，gather 成本更高 |
| group3 pairmerge | J113 12.3268s 变为 12.89069s | attention shape 膨胀超过减少循环的收益 |
| embedding int32 GPU 路径 | J113 12.3268s 变为 J117 12.98822s | cast/索引开销抵消传输节省 |
| selective cached repcat | 云端 official-like 与 142 持平或略慢 | 缓存复用没有形成可测收益 |
| torch.compile/复杂 varlen | 未进入最终版本 | 动态 shape、重编译和重排成本过高 |

负结果的共同点是：优化了局部操作，却增加了 shape 管理、gather、cast 或 Python 调度。端到端测试必须优先于局部算子直觉。

## 10. 推荐的优化工作流

1. 先按 move、forward、collect 三段 profile；
2. 统计 batch 数、用户长度、tokens、pred 数和 pooling length；
3. 每次只改一个数据流环节；
4. 使用线上相同计时边界；
5. 同时检查时延、AUC、PCOC 和输出 logid；
6. 至少重复运行，区分真实收益与 GPU 噪声；
7. 只有稳定正收益才进入最终代码。

这套方法比保存大量版本文件更有复用价值：版本会过时，数据流和测量纪律不会。

# 复现指南

## 1. 环境

最终代码面向：

```text
GPU: NVIDIA A800 80GB
OS: Ubuntu 20.04
CUDA: 12.6.3
Python: 3.10
PyTorch: 2.6.0
Triton: 3.2.0
```

安装依赖：

```bash
python -m pip install -r online_best/requirements.txt
```

比赛容器可能已经预装 CUDA/PyTorch。不要在不了解驱动兼容性的情况下重复安装整套 NVIDIA wheel。

## 2. 外部资产

仓库不提供以下文件：

```text
ckpt.pt
dataset/
├── test.csv
├── label_data.txt          # 可选，仅本地评估
├── history/*.csv
└── cached_batches/*.pt     # 可选，线上主要路径的本地复现
```

推荐目录：

```text
online_best/
├── infer.py
├── build_env.sh
├── requirements.txt
├── ckpt.pt
└── dataset/
```

也可以通过 `--ckpt` 指向外部 checkpoint。

## 3. 运行最终版本

```bash
python online_best/infer.py --ckpt /path/to/ckpt.pt
```

`infer.py` 默认从 `online_best/dataset/` 读取数据，并在当前工作目录生成 `predict.txt`。若存在 `cached_batches/shard_*.pt` 会优先走缓存路径，否则从 CSV 构造 DataLoader。

## 4. Profile

```bash
python tools/profile_cached_batches.py \
  --module online_best.infer \
  --ckpt /path/to/ckpt.pt \
  --dataset /path/to/dataset
```

该脚本分别统计：

- move：CPU 到 GPU；
- forward：模型与 sigmoid；
- collect：mask、GPU 到 CPU 和 list；
- CUDA kernel 热点分类。

它需要真实 cached shards，且结果必须标注运行环境、batch 数和同步方式。

## 5. 对比 baseline

源码结构比较：

```bash
git diff --no-index baseline/infer.py online_best/infer.py
```

性能比较必须使用同一 checkpoint、同一数据、同一计时范围和相同预热策略。

## 6. 打包

PowerShell：

```powershell
.\tools\package.ps1 online_best
```

Bash：

```bash
bash tools/package.sh online_best
```

输出位于 `dist/`，ZIP 根层包含：

```text
infer.py
build_env.sh
requirements.txt
```

## 7. 验证清单

提交或部署前至少确认：

1. `predict.txt` 的 logid 数量与待预测样本一致；
2. logid 顺序或映射符合 evaluator 约定；
3. AUC、PCOC 在允许范围；
4. 计时包含 move、forward、sigmoid、mask、CPU copy 和 list；
5. 没有截断序列、跳过 batch 或读取预生成预测；
6. 多次运行的时延收益稳定。

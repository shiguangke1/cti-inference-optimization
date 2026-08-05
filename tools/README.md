# 工具

本目录只保留能够直接服务最终版本的工具。

## Profile

```bash
python tools/profile_cached_batches.py \
  --module online_best.infer \
  --ckpt /path/to/ckpt.pt \
  --dataset /path/to/dataset
```

输出 move、forward、collect 和 CUDA kernel 分类数据。该工具依赖真实 GPU、checkpoint 和 cached batch shards。

## 打包

```powershell
.\tools\package.ps1 online_best
```

或：

```bash
bash tools/package.sh online_best
```

生成的 ZIP 位于 `dist/`，且不会被 Git 跟踪。

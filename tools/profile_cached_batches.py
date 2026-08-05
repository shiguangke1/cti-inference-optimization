import argparse, time, importlib, sys
from pathlib import Path
from collections import defaultdict
import torch


def load_infer(mod):
    if mod.endswith(".py"):
        mod = Path(mod).stem
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module(mod)


def load_batches(cache_dir, max_batches):
    shard_files = sorted(Path(cache_dir).glob("shard_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    out = []
    for sf in shard_files:
        out.extend(torch.load(sf, weights_only=False))
        if len(out) >= max_batches:
            break
    return out[:max_batches]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="online_best.infer")
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--nprofile", type=int, default=150)
    ap.add_argument("--nwarmup", type=int, default=15)
    ap.add_argument("--nkernel", type=int, default=80)
    args = ap.parse_args()

    print(f"[ENV] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
    inf = load_infer(args.module)
    need = args.nwarmup + args.nprofile
    batches = load_batches(Path(args.dataset) / "cached_batches", need)
    print(f"[DATA] loaded {len(batches)} batches (need {need}) module={args.module}")
    model, dev = inf.load_model(ckpt_path=Path(args.ckpt))

    def run_one(batch):
        bg = inf.move_batch_to_device(batch, dev)
        logits, _ = model(bg)
        logits = logits.squeeze(-1)
        probs = logits.sigmoid_()
        pm = bg["pred_mask"].bool()
        _ = probs[pm].detach().float().cpu().tolist()

    # ---------- warmup ----------
    with torch.inference_mode():
        for b in batches[:args.nwarmup]:
            run_one(b)
        torch.cuda.synchronize()

    prof_set = batches[args.nwarmup:args.nwarmup + args.nprofile]
    n = len(prof_set)

    # ---------- 分段 CUDA-event 计时 (加 sync, 会禁用重叠, 用于阶段归属) ----------
    mv = fw = cl = 0.0
    with torch.inference_mode():
        for b in prof_set:
            s = time.time()
            bg = inf.move_batch_to_device(b, dev)
            torch.cuda.synchronize(); m = time.time()
            logits, _ = model(bg); logits = logits.squeeze(-1); probs = logits.sigmoid_()
            torch.cuda.synchronize(); f = time.time()
            pm = bg["pred_mask"].bool(); _ = probs[pm].detach().float().cpu().tolist()
            c = time.time()
            mv += m - s; fw += f - m; cl += c - f
    tot = mv + fw + cl
    print("\n" + "=" * 70)
    print(f"[TOP-LEVEL over {n} batches] (含 sync, move 因禁用重叠偏大)")
    print(f"  move   ={mv:.3f}s ({mv/tot*100:4.1f}%)  {mv/n*1000:.2f} ms/batch  ~{mv/n*2039:.1f}s@2039")
    print(f"  forward={fw:.3f}s ({fw/tot*100:4.1f}%)  {fw/n*1000:.2f} ms/batch  ~{fw/n*2039:.1f}s@2039")
    print(f"  collect={cl:.3f}s ({cl/tot*100:4.1f}%)  {cl/n*1000:.2f} ms/batch  ~{cl/n*2039:.1f}s@2039")

    # ---------- torch.profiler kernel 级 ----------
    from torch.profiler import profile, ProfilerActivity
    kn = min(args.nkernel, n)
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for b in prof_set[:kn]:
                run_one(b)
            torch.cuda.synchronize()
    print("\n" + "=" * 70)
    print(f"[KERNEL TABLE over {kn} batches] sort=cuda_time_total")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # ---------- 按阶段聚合 kernel CUDA 时间 ----------
    STAGE_RULES = [
        ("embedding", ("fused_all_slots_embedding", "embedding")),
        ("layernorm", ("layernorm_forward", "layer_norm", "native_layer_norm")),
        ("attention", ("scaled_dot_product", "sdpa", "flash", "_varlen", "attention", "bmm")),
        ("moe_gate",  ("fused_top2_gate", "moe_route", "moe_prefix", "moe_top2")),
        ("moe_ffn",   ("moe_sparse_fc1", "moe_sparse_fc2", "moe_")),
        ("final_head",("final_linear_clamp",)),
        ("gemm_mm",   ("addmm", "mm", "gemm", "cublas", "matmul")),
        ("h2d_copy",  ("copy", "memcpy", "htod", "to_")),
        ("elementwise",("sigmoid", "add", "mul", "cat", "index", "elementwise", "vectorized")),
    ]
    stage_us = defaultdict(float)
    other_us = 0.0
    for evt in prof.key_averages():
        cu = getattr(evt, "cuda_time_total", 0) or getattr(evt, "device_time_total", 0)
        if cu <= 0:
            continue
        name = evt.key.lower()
        placed = False
        for stage, keys in STAGE_RULES:
            if any(k in name for k in keys):
                stage_us[stage] += cu
                placed = True
                break
        if not placed:
            other_us += cu
    total_us = sum(stage_us.values()) + other_us
    print("\n" + "=" * 70)
    print(f"[STAGE BREAKDOWN of GPU kernel time over {kn} batches]  total_cuda={total_us/1e3:.1f}ms")
    for stage, us in sorted(stage_us.items(), key=lambda x: -x[1]):
        print(f"  {stage:12s} {us/1e3:8.1f} ms  ({us/total_us*100:5.1f}%)  {us/kn:7.1f} us/batch")
    print(f"  {'OTHER':12s} {other_us/1e3:8.1f} ms  ({other_us/total_us*100:5.1f}%)")


if __name__ == "__main__":
    main()

"""Measure model complexity and PyTorch inference speed.

The script is intended for fair, single-GPU comparisons of trained weights.
It reports parameters, GFLOPs, checkpoint size, peak CUDA memory and batch-1
forward latency. It supports both the project's RT-DETR and YOLO checkpoints.
"""

import argparse
import csv
import gc
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import torch
from ultralytics import RTDETR, YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="+",
        help="One or more trained .pt files (best.pt is recommended).",
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "rtdetr", "yolo"),
        default="auto",
        help="Model loader. Use yolo for a group of YOLO checkpoints.",
    )
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 autocast/model weights. Only valid on CUDA.",
    )
    parser.add_argument(
        "--output",
        default="runs/complexity/complexity_results.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def resolve_device(value):
    if value.lower() == "cpu" or not torch.cuda.is_available():
        if value.lower() != "cpu":
            print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(f"cuda:{value.split(',')[0]}")


def choose_loader(path, kind):
    if kind != "auto":
        return RTDETR if kind == "rtdetr" else YOLO
    # Check the full path because trained checkpoints are usually all named
    # ``best.pt``; their parent run directory carries the model name.
    name = str(path).lower()
    return RTDETR if "rtdetr" in name or "rt-detr" in name else YOLO


def count_flops(core, dummy):
    """Return multiply-add FLOPs as GFLOPs, or NaN when thop cannot trace it."""
    try:
        import thop

        profiled = deepcopy(core).eval()
        with torch.no_grad():
            flops, _ = thop.profile(profiled, inputs=(dummy,), verbose=False)
        return float(flops * 2.0 / 1e9)
    except Exception as error:
        print(f"  GFLOPs unavailable: {type(error).__name__}: {error}")
        return float("nan")


def forward_once(core, dummy, use_half):
    if use_half:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return core(dummy)
    return core(dummy)


def measure_latency(core, dummy, device, warmup, iterations, use_half):
    with torch.inference_mode():
        for _ in range(warmup):
            forward_once(core, dummy, use_half)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    samples = []
    with torch.inference_mode():
        for _ in range(iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward_once(core, dummy, use_half)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            else:
                start = time.perf_counter()
                forward_once(core, dummy, use_half)
                samples.append((time.perf_counter() - start) * 1000.0)

    peak_allocated = 0.0
    peak_reserved = 0.0
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20

    average = mean(samples)
    return {
        "latency_ms_mean": average,
        "latency_ms_std": stdev(samples) if len(samples) > 1 else 0.0,
        "latency_ms_p50": sorted(samples)[len(samples) // 2],
        "latency_ms_p95": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)],
        "fps_batch1": 1000.0 / average if average > 0 else 0.0,
        "peak_cuda_allocated_mib": peak_allocated,
        "peak_cuda_reserved_mib": peak_reserved,
    }


def measure_one(path, kind, device, imgsz, warmup, iterations, use_half):
    loader = choose_loader(path, kind)
    print(f"\nLoading: {path} ({loader.__name__})")
    wrapper = loader(path)
    if hasattr(wrapper, "fuse"):
        try:
            wrapper.fuse()
        except Exception as error:
            print(f"  Fuse skipped: {type(error).__name__}: {error}")

    core = wrapper.model.to(device).eval()
    if use_half:
        if device.type != "cuda":
            raise ValueError("--half requires a CUDA device")
        core = core.half()
    dummy = torch.zeros((1, 3, imgsz, imgsz), device=device)
    if use_half:
        dummy = dummy.half()

    params = sum(parameter.numel() for parameter in core.parameters())
    trainable = sum(parameter.numel() for parameter in core.parameters() if parameter.requires_grad)
    flops = count_flops(core, dummy)
    timing = measure_latency(core, dummy, device, warmup, iterations, use_half)
    size_mib = Path(path).stat().st_size / 2**20
    result = {
        "model": path,
        "loader": loader.__name__,
        "imgsz": imgsz,
        "precision_mode": "FP16" if use_half else "FP32",
        "parameters": params,
        "parameters_m": params / 1e6,
        "trainable_parameters": trainable,
        "gflops": flops,
        "checkpoint_mib": size_mib,
        **timing,
    }
    print(
        f"  Params: {params / 1e6:.3f}M | GFLOPs: {flops:.3f} | "
        f"Weight: {size_mib:.2f} MiB"
    )
    print(
        f"  Latency: {timing['latency_ms_mean']:.3f} +/- "
        f"{timing['latency_ms_std']:.3f} ms | "
        f"FPS: {timing['fps_batch1']:.2f} | "
        f"Peak VRAM: {timing['peak_cuda_allocated_mib']:.2f} MiB"
    )
    del wrapper, core, dummy
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    device = resolve_device(args.device)
    rows = [
        measure_one(
            path,
            args.kind,
            device,
            args.imgsz,
            args.warmup,
            args.iterations,
            args.half,
        )
        for path in args.models
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {output.resolve()}")


if __name__ == "__main__":
    main()

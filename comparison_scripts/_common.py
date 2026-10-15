"""Shared settings for the independent YOLO comparison experiments."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "NEU-DET" / "data.yaml"
RUNS = ROOT / "runs" / "train"


def train(model_yaml: str, run_name: str) -> None:
    import ultralytics

    imported_from = Path(ultralytics.__file__).resolve()
    local_package = (ROOT / "ultralytics").resolve()
    try:
        imported_from.relative_to(local_package)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            f"Imported the RT-DETR local package: {imported_from}. "
            "Run this script in the separate official Ultralytics environment."
        )

    from ultralytics import YOLO

    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")

    model = YOLO(model_yaml)
    model.train(
        data=str(DATA),
        pretrained=False,
        cache=True,
        imgsz=640,
        epochs=300,
        batch=8,
        workers=8,
        device=0,
        amp=True,
        seed=0,
        deterministic=True,
        patience=50,
        project=str(RUNS),
        name=run_name,
        exist_ok=False,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=1.0,
        momentum=0.9,
        weight_decay=1e-4,
        warmup_epochs=12.7,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        close_mosaic=0,
        cos_lr=False,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )

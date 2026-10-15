"""Evaluate a trained detector once on an independent test split."""

import argparse
import csv
import json
from pathlib import Path

from ultralytics import RTDETR, YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to best.pt")
    parser.add_argument("data", help="Dataset data.yaml containing a test entry")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", required=True, help="Unique output name under runs/test")
    parser.add_argument("--yolo", action="store_true", help="Load the checkpoint with YOLO instead of RTDETR")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cls = YOLO if args.yolo else RTDETR
    model = model_cls(args.model)
    metrics = model.val(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project="runs/test",
        name=args.name,
        exist_ok=False,
        plots=True,
    )

    box = metrics.box
    save_dir = Path(metrics.save_dir)
    class_indices = list(getattr(box, "ap_class_index", range(len(box.p))))
    names = metrics.names
    rows = []
    for metric_index, class_index in enumerate(class_indices):
        class_index = int(class_index)
        rows.append({
            "class_id": class_index,
            "class_name": names[class_index],
            "precision": float(box.p[metric_index]),
            "recall": float(box.r[metric_index]),
            "f1": float(box.f1[metric_index]),
            "ap50": float(box.ap50[metric_index]),
            "ap75": float(box.all_ap[metric_index, 5]),
            "ap50_95": float(box.ap[metric_index]),
        })

    with (save_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "data": args.data,
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "f1": float(sum(row["f1"] for row in rows) / len(rows)),
        "map50": float(box.map50),
        "map75": float(box.map75),
        "map50_95": float(box.map),
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }
    with (save_dir / "test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=True)

    print("\nIndependent test summary")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Per-class metrics: {save_dir / 'per_class_metrics.csv'}")


if __name__ == "__main__":
    main()

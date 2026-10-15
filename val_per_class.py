"""逐类 AP 提取脚本。

用法（在云服务器上）:
    python val_per_class.py <best.pt路径> <data.yaml路径> [imgsz]

示例:
    python val_per_class.py runs/train/yolov8n-neu-baseline/weights/best.pt dataset/NEU-DET/data.yaml 640
    python val_per_class.py runs/train/rtdetr-r18-neu-global-aifi-shsa/weights/best.pt dataset/NEU-DET/data.yaml 640

注意: RT-DETR 的自定义模型(best.pt)必须在装了 GlobalEdge/SHSA 自定义模块的环境里跑;
      YOLO 模型用原版 ultralytics 即可。
"""

import sys
import warnings

warnings.filterwarnings('ignore')

from ultralytics import YOLO, RTDETR


def main():
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 640

    # 自动判断模型类型
    cls = RTDETR if 'rtdetr' in model_path.lower() else YOLO
    model = cls(model_path)

    print(f"验证模型: {model_path}")
    print(f"数据: {data_path}  imgsz={imgsz}\n")

    metrics = model.val(data=data_path, imgsz=imgsz, split='val')

    box = metrics.box
    names = box.names  # dict {class_id: name}

    ap50 = getattr(box, 'ap50', None)          # 逐类 AP@0.5
    ap = getattr(box, 'ap', None)              # 逐类 AP@0.5:0.95
    if ap is None:
        ap = getattr(box, 'maps', None)

    print(f"\n===== 逐类 AP: {model_path} =====")
    print(f"{'类别':<18}{'AP50':>9}{'AP50-95':>10}")
    for i in sorted(names):
        n = names[i]
        a50 = f"{ap50[i]:.4f}" if (ap50 is not None and i < len(ap50)) else "N/A"
        a = f"{ap[i]:.4f}" if (ap is not None and i < len(ap)) else "N/A"
        print(f"{n:<18}{a50:>9}{a:>10}")

    print(f"\n总体: mAP50={box.map50:.4f}  mAP50-95={box.map:.4f}")


if __name__ == '__main__':
    main()

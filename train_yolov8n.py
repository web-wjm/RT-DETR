import warnings

from ultralytics import YOLO


warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # 用 yolov8n.yaml 从零训练，与 RT-DETR R18 baseline 保持同一起点（均无预训练）
    model = YOLO('yolov8n.yaml')

    model.train(
        data='dataset/NEU-DET/data.yaml',
        cache=True,
        imgsz=640,
        epochs=300,
        batch=8,
        workers=8,
        device=0,
        amp=True,
        seed=0,
        patience=50,
        project='runs/train',
        name='yolov8n-neu-baseline-640',
    )

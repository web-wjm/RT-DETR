import warnings

from ultralytics import RTDETR


warnings.filterwarnings('ignore')


if __name__ == '__main__':
    model = RTDETR('models/rt-detr/rtdetr-r18.yaml')

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
        name='rtdetr-r18-neu-baseline-640',
    )

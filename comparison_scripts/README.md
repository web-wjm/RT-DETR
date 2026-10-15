# YOLO comparison experiments

These scripts are intended for an official Ultralytics environment (version >= 8.3.0), not the local RT-DETR fork.
They use the same NEU-DET split, image size, batch size, seed, optimizer, and augmentation settings as the main comparison protocol.

## Cloud setup

Replace `rtdetr` with the conda environment used for the RT-DETR project:

```bash
conda env list
conda create -n yolo_compare --clone rtdetr -y
conda activate yolo_compare
python -m pip uninstall -y ultralytics
python -m pip install -U "ultralytics>=8.3.0"
python -c "import torch, ultralytics; print(torch.__version__); print(torch.cuda.is_available()); print(ultralytics.__version__); print(ultralytics.__file__)"
```

The printed Ultralytics path must be under the new conda environment, not `RTDETR-main/ultralytics`.

Upload this directory to `/root/autodl-tmp/yolo_compare/`, then run one model at a time:

```bash
conda activate yolo_compare
cd /root/autodl-tmp/yolo_compare
python train_yolov8n_neu.py
python train_yolov8s_neu.py
python train_yolo11n_neu.py
python train_yolo11s_neu.py
```

The data YAML and output directory are resolved from the RT-DETR project at `/root/autodl-tmp/RTDETR-main`. Change `ROOT` in `_common.py` if the project is stored elsewhere.

The scripts intentionally use YAML initialization with `pretrained=False`. Do not replace the model names with `.pt` files if the results are to be compared with the current RT-DETR experiments, which were initialized from YAML.

# An RT-DETR-R18 Adaptation for Industrial Surface Defect Detection

Official code, model configurations and experimental records for the manuscript
*"An RT-DETR-R18 Adaptation for Industrial Surface Defect Detection: An Accuracy--Computational Cost Study"* (submitted to Signal, Image and Video Processing).

The study evaluates three components on an RT-DETR-R18 detector under a complete 2^3 factorial design:

- **SlimNeck (SN)** — GSConv-based feature projection and four VoVGSCSP fusion blocks with channel shuffling.
- **Inner-GIoU (IG)** — auxiliary-box IoU regression term used during training only.
- **Dual-scale AIFI (DA)** — independent AIFI encoding at the P4 and P5 levels.

GC10-DET is the primary benchmark (frozen split, seeds 0-2); NEU-DET is the secondary benchmark.

## Repository layout

```
ultralytics/                  RT-DETR framework (training, validation, dataloaders)
models/rt-detr/               model configurations used in the paper
runs/test/                    test summaries, per-class metrics and curves (39 runs)
runs/complexity/              parameter / GFLOPs / latency / peak-memory measurements
dataset/                      dataset preparation and split scripts (data not hosted)
comparison_scripts/           YOLOv8n/s, YOLO11n/s reference-detector scripts
springer期刊模板/              LaTeX manuscript, figures and revision notes
*.svg / *.png / *.pdf         editable sources of the three paper figures
draw_editable_*.py            figure generation scripts
train.py, val.py, val_per_class.py, complexity_test.py, get_FPS.py
```

## Model configurations

| Config | SN | IG | DA | YAML |
|---|---|---|---|---|
| RT-DETR-R18 (baseline) | | | | `models/rt-detr/rtdetr-r18.yaml` |
| SN | ✓ | | | `models/rt-detr/rtdetr-slimneck.yaml` |
| IG | | ✓ | | `models/rt-detr/rtdetr-r18-InnerGIoU.yaml` |
| DA | | | ✓ | `models/rt-detr/rtdetr-r18-MSAIFI-Only.yaml` |
| SN + IG | ✓ | ✓ | | `models/rt-detr/rtdetr-SlimNeck-InnerGIoU.yaml` |
| SN + DA | ✓ | | ✓ | `models/rt-detr/rtdetr-SlimNeck-MSAIFI.yaml` |
| IG + DA | | ✓ | ✓ | `models/rt-detr/rtdetr-r18-MSAIFI-InnerGIoU.yaml` |
| Combined | ✓ | ✓ | ✓ | `models/rt-detr/rtdetr-SlimNeck-InnerGIoU-MSAIFI.yaml` |

Reference detector: `models/rt-detr/rtdetr-r50.yaml`.

## Data preparation

Raw images are not hosted in this repository. Prepare GC10-DET and NEU-DET with:

```bash
python dataset/setup_gc10_det.py
python dataset/setup_neu_det.py
python dataset/split_data.py
```

Expected layout: `dataset/GC10-DET/` and `dataset/NEU-DET/` with `images/{train,val,test}` and `labels/{train,val,test}`. The frozen GC10-DET partition contains 1833/229/230 images for train/validation/test.

## Training and testing

```bash
python train.py --model models/rt-detr/rtdetr-SlimNeck-InnerGIoU-MSAIFI.yaml \
                --data dataset/data.yaml --img 800 --batch 8 --seed 0 --epochs 300

python val_per_class.py --model runs/train/<run>/weights/best.pt \
                        --data dataset/data.yaml --split test --img 800
```

Complexity measurements (parameters, GFLOPs, latency, peak memory):

```bash
bash run_gc10_complexity_full_fp32.sh
```

## Experimental results

Test outputs are provided per run under `runs/test/<config>-seed{0,1,2}/`:
`test_summary.json`, `per_class_metrics.csv`, precision/recall/F1 curves and confusion matrices.
Manuscript tables are derived from these files; `runs/complexity/` holds the FP32/FP16 cost measurements.

| Model (GC10-DET) | mAP50 | mAP50-95 |
|---|---|---|
| RT-DETR-R18 | 54.42 | 28.01 |
| Combined | 61.00 | 30.29 |

## Figures

The three manuscript figures are provided as editable SVG, PNG and PDF:
`RTDETR_SlimNeck_DualAIFI_InnerGIoU_Overall_editable.*`, `RTDETR_DualAIFI_Comparison_editable.*`
and `RTDETR_SlimNeck_KeyModules_editable.*`.

## License and attribution

The code is derived from the Ultralytics RT-DETR implementation (AGPL-3.0).

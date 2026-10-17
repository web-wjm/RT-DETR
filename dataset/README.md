## Dataset preparation

Raw images and labels are **not hosted** in this repository. They are obtained with the scripts here:

| Script | Purpose |
|---|---|
| `setup_gc10_det.py` | Download and convert GC10-DET into YOLO layout |
| `setup_neu_det.py` | Download and convert NEU-DET into YOLO layout |
| `split_data.py` | Build the frozen train/validation/test partitions |
| `check_gc10_det.py`, `check_neu_det.py` | Verify class counts and file integrity |
| `xml2txt.py`, `yolo2coco.py` | Annotation format conversion |

Expected layout:

```
dataset/
  GC10-DET/  images/{train,val,test}  labels/{train,val,test}  data.yaml
  NEU-DET/   images/{train,val,test}  labels/{train,val,test}  data.yaml
```

The GC10-DET partition used in the paper contains 1833 training, 229 validation and 230 test images (2292 in total), frozen across all seeds. See `dataset/data.yaml` for the dataset root used by training scripts.

GC10-DET source: https://github.com/lvxiaomingld/GC10-DET
NEU-DET source: https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm

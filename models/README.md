## Paper model configurations

Only the configurations used in the manuscript are retained:

| Config | SN | IG | DA | YAML |
|---|---|---|---|---|
| RT-DETR-R18 (baseline) | | | | `rtdetr-r18.yaml` |
| SN | ✓ | | | `rtdetr-slimneck.yaml` |
| IG | | ✓ | | `rtdetr-r18-InnerGIoU.yaml` |
| DA | | | ✓ | `rtdetr-r18-MSAIFI-Only.yaml` |
| SN + IG | ✓ | ✓ | | `rtdetr-SlimNeck-InnerGIoU.yaml` |
| SN + DA | ✓ | | ✓ | `rtdetr-SlimNeck-MSAIFI.yaml` |
| IG + DA | | ✓ | ✓ | `rtdetr-r18-MSAIFI-InnerGIoU.yaml` |
| Combined | ✓ | ✓ | ✓ | `rtdetr-SlimNeck-InnerGIoU-MSAIFI.yaml` |
| Reference | | | | `rtdetr-r50.yaml` |

SN = SlimNeck, IG = Inner-GIoU, DA = dual-scale AIFI.
Training and evaluation commands are given in the repository README.

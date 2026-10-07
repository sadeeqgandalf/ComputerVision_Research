# KITTI MOTS val — SAM 3 vs SAM 3.1

SAM 3 conf=0.5 · SAM 3.1 conf=0.3 (not matched). SAM 3.1 conf=0.5 pending.

## HOTA

| Class | SAM 3 | SAM 3.1 | Δ |
| :--- | ---: | ---: | ---: |
| car | 82.6 | 69.0 | -13.7 |
| pedestrian | 70.0 | 63.7 | -6.3 |

## Car diagnostics

| Metric | SAM 3 | SAM 3.1 |
| :--- | ---: | ---: |
| DetPr | 84.9 | 73.0 |
| IDSW | 4 | 46 |
| Pred / GT | 8,261 / 8,029 | 9,448 / 8,029 |
| LocA | 89.7 | 89.4 |

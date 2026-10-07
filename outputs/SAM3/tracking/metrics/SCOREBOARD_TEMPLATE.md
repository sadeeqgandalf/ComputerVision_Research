# Tracking scoreboard template (SAM 3 / SAM 3.1)

Primary suite (locked column order):

| benchmark | split | tracker | class | HOTA | DetA | AssA | DetRe | DetPr | AssRe | AssPr | LocA | sMOTSA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitti_mots | val | sam3_1_person_car | car | — | — | — | — | — | — | — | — | — |
| kitti_mots | val | sam3_1_person_car | pedestrian | — | — | — | — | — | — | — | — | — |
| mots_challenge | train | sam3_1_person | pedestrian | — | — | — | — | — | — | — | — | — |

Historical SAM 3 tables live in `outputs/SAM3/tracking/metrics/`.
New SAM 3.1 tables are written by `eval/tracking/run_hota.py` to
`outputs/SAM3.1/tracking/metrics/`.

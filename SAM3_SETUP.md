# SAM 3 / SAM 3.1 setup

## Environment

```bash
cd /Users/sadeeqadeyemi/Documents/Computervision_Research
source .venv_sam3/bin/activate   # Python 3.12, torch, sam3 editable
```

## Hugging Face gated weights

1. Request access: https://huggingface.co/facebook/sam3 (and sam3.1 if separate gate)
2. Create a read token: https://huggingface.co/settings/tokens
3. Login:

```bash
source .venv_sam3/bin/activate
hf auth login
```

## Artifact layout

```
outputs/
  benchmarks/ground_truth/     # shared GT (model-independent)
  SAM3/                        # historical base-model metrics + demos
  SAM3.1/tracking/             # current inference + HOTA (pod pull target)
    raw/<benchmark>/<split>/
    predictions/
    results/
    metrics/
  blindbench/                  # CARLA BlindBench (non-SAM)
  analysis/                    # pixel / exploratory viz
```

## BDD image demo (historical SAM3)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/run_sam3_bdd_demo.py
```

Output: `outputs/SAM3/demos/bdd/` (GT boxes left, SAM masks right).

## Tracking benchmarks (SAM 3.1 on GPU pod)

```bash
python scripts/run_sam31_hota_benchmarks.py --skip-existing
# or stepwise:
python scripts/run_sam3_kitti_mots_batch.py --split val --skip-video
python eval/tracking/score_kitti_mots.py --split val
python scripts/run_sam3_mots_challenge_batch.py --skip-video
python eval/tracking/score_mots_challenge.py
```

## Local Mac notes

- Official README assumes CUDA; this machine uses **MPS**/CPU for light demos only.
- Full KITTI / MOTS Challenge HOTA runs on the RunPod A100 with the research volume attached.
- Small patches in `sam3/model/edt.py` (cv2 EDT fallback) and `model_builder.py` (mps device) so imports work without Triton/CUDA.

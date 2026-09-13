# SAM3 on Mac (Goal 1: BDD100K)

## Environment

```bash
cd /Users/sadeeqadeyemi/Documents/Computervision_Research
source .venv_sam3/bin/activate   # Python 3.12, torch MPS, sam3 editable
```

## Hugging Face gated weights (required)

1. Request access: https://huggingface.co/facebook/sam3  
2. Wait until accepted  
3. Create a token (read access): https://huggingface.co/settings/tokens  
4. Login:

```bash
source .venv_sam3/bin/activate
hf auth login
# paste token
```

5. Download + run BDD demo:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/run_sam3_bdd_demo.py
# optional: SAM3_PROMPT="traffic sign" python scripts/run_sam3_bdd_demo.py
```

Output: `outputs/bdd_sam3_demo/` (GT boxes left, SAM3 masks right).

## Local Mac notes

- Official README assumes CUDA; this machine uses **MPS**.
- Small patches in `sam3/model/edt.py` (cv2 EDT fallback) and `model_builder.py` (mps device) so imports work without Triton/CUDA.
- `decord` notebook extra is skipped on Mac ARM.

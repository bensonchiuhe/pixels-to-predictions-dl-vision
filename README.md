# Pixels to Predictions: DL Vision Challenge

Fine-tuning [SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) with QLoRA for science visual multiple-choice QA.

**NYU ECE-GY 7123 Deep Learning — Final Project**

| Version | Key Change | Public Score |
|---------|-----------|-------------|
| V1 | 1 epoch, IMG_SIZE=224, single model | 0.70824 |
| V2 | 3 epochs, IMG_SIZE=384, Phase 2 (train+val) | 0.72434 |
| **V3** | **4-checkpoint ensemble** | **0.74044** ✓ final |

---

## Method Overview

- **Base model**: `HuggingFaceTB/SmolVLM-500M-Instruct` (Idefics3: SigLIP + SmolLM2)
- **Quantization**: 4-bit NF4 QLoRA (BitsAndBytes), float16 compute
- **LoRA**: r=16, α=32, dropout=0.05, targets `q_proj k_proj v_proj o_proj`
- **Trainable params**: 4,161,536 / 305,991,872 (1.36%) — within 5M limit ✓
- **Inference**: Log-likelihood scoring (one forward pass, argmax over answer letter logits)

### Training Curriculum

**Phase 1** — train set (3,109 samples), 3 epochs, lr=2e-4 cosine, effective batch=16  
→ Best val acc: **0.6660** (Epoch 2, saved as `best_ckpt`)

**Phase 2** — train+val combined (4,157 samples), 3 epochs, lr=1e-4 warm-start from best Phase 1  
→ Saves `phase2_ep1`, `phase2_ep2`, `phase2_ep3`

**Ensemble** — average log-likelihood scores from 4 checkpoints (`best_ckpt` + `phase2_ep1/2/3`), argmax

---

## Setup

### Requirements
- Google Colab Free Tier (T4 GPU, 15.6 GB VRAM)
- Google Drive with competition data

### Google Drive Structure
Upload your competition data to Google Drive:
```
MyDrive/
└── competition_data/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── images/
        ├── train/
        ├── val/
        └── test/
```

### Model Weights
Pre-trained checkpoints (best_ckpt, phase2_ep1/2/3) are available at:

> **[Google Drive — Model Weights](https://drive.google.com/drive/folders/1NLbOYkJJdUUgHKa-3ElkfFPFU4rEw0tZ?usp=sharing)**

Place them at `MyDrive/smolvlm_competition/` before running inference-only mode.

---

## How to Run

Open `solution_v3_ensemble.ipynb` in Google Colab with a **T4 GPU** runtime (Runtime → Change runtime type → T4 GPU).

### Full Training + Inference (from scratch, ~8–10 hours)

Run all cells from top to bottom in order.

### Inference Only (using provided weights, ~32 min)

1. Download model weights from the Drive link above → place the folder at `MyDrive/smolvlm_competition/`  
   (you should have subfolders: `best_ckpt/`, `phase2_ep1/`, `phase2_ep2/`, `phase2_ep3/`)
2. Run the **Setup** cells — Drive mount, install packages, copy data, imports, config, dataset/model loading, and inference helpers (everything before the Phase 1 training loop)
3. **Skip** the Phase 1 training cell, the Phase 2 warm-start cell
4. Run the **Ensemble Inference** cell — loads the 4 checkpoints from Drive and generates `submission.csv`
5. Run the **Download** cell to retrieve `submission.csv` locally

**Expected output**: `submission.csv` with 1,008 predictions, achieving public leaderboard accuracy of **0.74044** (V3 ensemble, final submission).

### Package Versions (auto-installed at the top of the notebook)
```
transformers==4.51.3
peft==0.14.0
bitsandbytes==0.45.5
accelerate==1.6.0
```

---

## Repository Structure

```
├── solution_v3_ensemble.ipynb      # Final notebook (V3, score 0.74044)
├── solution_v2_384px_phase2.ipynb  # V2 (score 0.72434)
├── solution_v1_baseline.ipynb      # V1 baseline (score 0.70824)
├── submission.csv                  # Final submission predictions
└── README.md
```

---

## Key Design Decisions

**Why log-likelihood over text generation?**  
Generation can fail to produce a valid letter or output verbose phrases. Log-likelihood scoring is deterministic, requires no post-processing, and uses a single forward pass per sample.

**Why include the `solution` field during training?**  
The solution field provides step-by-step reasoning that directly supports the correct answer — a form of chain-of-thought supervision. It is excluded at inference (not present in the test set).

**Why 4-checkpoint ensemble?**  
Each Phase 2 epoch represents a different point on the training trajectory. Averaging their log-likelihood scores reduces variance without any additional training.

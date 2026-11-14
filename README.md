# Ordinal Regression Heads and Loss Functions for Fine-Grained Arabic Readability Assessment

Master 2 project (Artificial Intelligence and Big Data), Department of Computer Science,
University Djilali Bounaama of Khemis Miliana, academic year 2025/2026.

- **Students:** Abderrahim Bardad, Taiba Ilyes
- **Supervisor:** Dr. Mohamed Goudjil

## What the project does

The task is to predict the readability level (1 to 19) of Arabic sentences from the
[BAREC corpus](https://huggingface.co/datasets/CAMeL-Lab/BAREC-Corpus-v1.0).
The main metric is quadratic weighted kappa (QWK).

The levels are ordered, so we compare **prediction heads and loss functions that use the order**,
inside one fixed pipeline:

```
BAREC sentence -> D3Tok input -> AraBERTv2 encoder -> prediction head + loss -> level -> metrics
```

Everything is fixed (data splits, input, encoder, training protocol of the BAREC paper),
except the prediction head and the loss function.

| Group | Methods | Why |
|---|---|---|
| 1. BAREC reproduction | CE (baseline), Regression (MSE), OLL (alpha=1.5), EMD, SOFT (beta=3) | check our pipeline against the BAREC paper |
| 2. Published systems | CORAL head (ZAI), MSE then Soft-QWK (STBW) | compare with the two shared-task papers |
| 3. New loss functions | MV, CCC, CLM, OE, RANK, PLR, CORN | our own exploration |
| 4. Additional | ListNet, DLDL-v2, truncated soft labels, length-balanced loss | reported, including negative results |

## Main results (QWK in %, development / test)

| Method | Dev | Test |
|---|---|---|
| CE (baseline) | 79.81 | 80.48 |
| Regression (MSE) | 82.88 | 83.64 |
| OLL (alpha=1.5) | 82.91 | 84.08 |
| EMD | 81.29 | 82.70 |
| SOFT (beta=3) | 79.72 | 80.31 |
| CORAL | 82.90 | **84.33** |
| MSE then Soft-QWK | 83.24 | 84.31 |
| MV | **83.50** | 84.04 |
| CCC | 83.29 | 84.30 |
| CLM | 83.06 | 83.71 |
| OE | 82.90 | 83.93 |
| RANK | 82.67 | 84.06 |
| PLR | 82.61 | 83.68 |
| CORN | 81.92 | 82.57 |
| *Group 4:* ListNet / DLDL-v2 / TSOFT / LBL | 82.38 / 81.96 / 80.81 / 75.72 | 83.20 / 83.02 / 81.84 / 78.13 |
| *Ablation:* CORAL with head learning rate 1e-3 | 83.86 | 84.58 |

For reference, the best model of the BAREC paper (regression) has 84.0 (dev) and 84.4 (test).
None of our main methods beats it; the best ones are within about 0.1 point on the test set.
All runs use one seed (42).

## Repository map

| Folder | Content |
|---|---|
| `final/gpu code/` | training scripts (Parts A to E and J to L), one GPU |
| `final/gpu code results/` | one JSON result file per method |
| `final/cpu code/` | CPU-friendly track: frozen encoder, linear heads, 5 seeds |
| `final/cpu code results/` | results of the CPU track |
| `final/papers/` | the papers the project builds on |
| `final/these nov/`, `final/these v2/`, `final/these3/`, `final/thesis/` | earlier drafts of the thesis |

The code and result folders each have their own README. The final thesis is written in LaTeX on Overleaf.

## How to reproduce a run

1. Open a Kaggle notebook with a GPU (Tesla T4) and Internet on. Use the same software as our runs:
   Python 3.12.13, PyTorch 2.10.0+cu128, Transformers 5.0.0, Datasets 5.0.0 (we used a notebook
   created in September 2026 with "Pin to original environment"). If the notebook has two GPUs,
   the scripts use only the first one.
2. Paste one script from `final/gpu code/` into one cell and run it.
   It downloads the corpus and the model at fixed revisions, trains its methods and writes
   one `results/<METHOD>.json` per method.
3. One method takes 25 to 36 minutes on one T4.

Training protocol (from the BAREC paper and its script): AdamW, learning rate 5e-5, linear decay,
no warm-up, 6 epochs, batch 64, seed 42, maximum length 512, fp16, checkpoint with the lowest
development loss.

## Sources

- Corpus: Elmadani, Habash and Taha-Thomure (2025), *A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment*, Findings of ACL 2025.
- Encoder: [aubmindlab/bert-base-arabertv2](https://huggingface.co/aubmindlab/bert-base-arabertv2) (Antoun et al., 2020).
- Shared task: Elmadani et al. (2025), *BAREC Shared Task 2025 on Arabic Readability Assessment*, ArabicNLP 2025.

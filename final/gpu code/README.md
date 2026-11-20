# Training scripts (GPU)

Each script is one self-contained Python file for a Kaggle notebook with one GPU (Tesla T4).
All scripts share the same pipeline code (data loading, AraBERTv2 model, training protocol,
metrics, checks). They differ only in the methods they train.

| Script | Methods | Group |
|---|---|---|
| `barec_exact_v3_partA.py` | CE, MSE (regression), OLL15, EMD | A: BAREC reproduction |
| `barec_exact_v3_partB.py` | SOFT2, SOFT3, SOFT4, SOFT5 | A: BAREC reproduction (SOFT3 kept) |
| `barec_exact_v3_partC.py` | CORAL, CORAL_HEADLR, SOFTQWK | B: ZAI and STBW methods (CORAL_HEADLR: additional) |
| `barec_exact_v3_partD.py` | CORN, TSOFT12, LBL10 | C (CORN) and additional |
| `barec_exact_v3_partE.py` | RANK, PLR, WKL | C (RANK, PLR); WKL not used, see below |
| `barec_exact_v3_partJ.py` | CCC, MV | C |
| `barec_exact_v3_partK.py` | OE, DLDL2 | C (OE) and additional (DLDL2) |
| `barec_exact_v3_partL.py` | LISTNET, CLM | additional (ListNet) and C (CLM) |

Parts A, B and C are identical except for the line `PART = ...`. The same is true for Parts J, K and L.
Parts J, K and L also stop at the start if the software versions differ from the reference
environment.

## Fixed settings

- Corpus `CAMeL-Lab/BAREC-Corpus-v1.0`, revision `3a336527783c4c3311a900429203bf4e25c7a80f`, D3Tok column, official splits (54,845 / 7,310 / 7,286 sentences).
- Encoder `aubmindlab/bert-base-arabertv2`, revision `97522efce17efa33036ac619802d5cec238dcad9`.
- AdamW, learning rate 5e-5, linear decay, no warm-up, weight decay 0, gradient clipping 1.0.
- 6 epochs, batch 64 (16 x 4 gradient accumulation, or a real batch of 64 with gradient checkpointing for losses computed on the whole batch), seed 42, fp16.
- Checkpoint: lowest development loss (BAREC rule). The highest-QWK epoch is also saved, for the selection ablation.

## Checks in every run

- The run stops if more than one GPU is visible or the batch is not 64.
- The number of optimiser steps of every epoch is checked (857).
- The saved checkpoint is reloaded and its development metrics must equal the training log.
- Every loss and metric is tested against an independent implementation before training.

## Note on WKL

`WKL` (weighted kappa loss) was run once in a newer software environment
(Python 3.13.15, PyTorch 2.11.0, Transformers 5.16.1). It is therefore not used in the thesis.

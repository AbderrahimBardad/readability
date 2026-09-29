# Part E: three readability losses that overlap with nothing already run

`barec_exact_v3_partE.py` is your v3 script with `PART = "E"` and three losses
added. Data, splits, model, heads, training settings, seed, metrics, checkpoint
selection, report and guards are the v3 code, unchanged. They follow BAREC
section 5.5: learning rate 5e-5, batch 64, six epochs, best epoch by evaluation
loss. In the Python code, only two v3 lines differ: the part letter and the
version date.

Part E shares no loss with Parts A, B and C, nor with the Part D you already
ran (CORN, TSOFT12, LBL10). Its arm names and its `results_backup_partE.zip`
do not collide with any of them.

| Arm | Loss | Paper | Languages in the paper | Formula taken from | Compare with |
|---|---|---|---|---|---|
| WKL | Weighted Kappa Loss | de la Torre et al. 2018; used for readability by Lim & Lee, BEA 2024 | English, Chinese | BAREC `train.py` lines 87 to 114 | CE, SOFTQWK |
| RANK | MSE + Margin Ranking Loss | Trokhymovych, Sen & Gerlach 2024 | 14 languages | The paper's formula, `max(0, -y(S1-S2)+m)` | MSE |
| PLR | MSE + Pairwise Logistic Loss | Lee & Vajjala, Findings of ACL 2022 | English, French, Spanish | The authors' code, `jlee118/NPRM`, and TF-Ranking's source | MSE, RANK |

## Notes per loss

- **WKL.** This is the exact code BAREC ships but never reported. Your v3 left it
  out by your earlier instruction. SOFTQWK in Part C is a different kappa loss on
  the regression head.
- **RANK and PLR.** Both add a pairwise term to MSE. For every pair in the batch
  where text i is harder than text j, RANK uses the hinge `max(0, 0.5 - (s_i - s_j))`.
  PLR uses the logistic `log(1 + exp(-(s_i - s_j)))`. MSE is needed because a
  ranking score alone has no scale for 19 levels. The weight λ = 1 is your own
  choice.
- **Pairs.** The original papers pair versions of the same article. BAREC has no
  such versions, so pairs are formed between sentences in the training batch.

WKL, RANK and PLR compare samples inside a batch, so they use v3's rule D3: a
true batch of 64 with gradient checkpointing. Part C uses the same rule for
SOFTQWK, and BAREC's `train.py` runs WKL at that batch.

## A check on the Part D you ran

The Length-Balanced Loss authors' run scripts set ρ = 0.6 for Cambridge, 0.4 for
OneStopEnglish and 0.8 for WeeBit. None uses ρ = 1.0. If the thesis calls
LBL10's ρ = 1.0 "the paper's value", check it against the paper.

## How to run

1. Use one Kaggle account with GPU T4 and Internet ON.
2. Paste the whole file as the first and only cell, then Save & Run All.
3. Download `results_backup_partE.zip` and unzip it next to the other parts.

The three arms take about 1.5 hours in total.

## BibTeX to add

`delatorre2018wkl` and `lee2022neural` are already in `thesis/references.bib`.

```bibtex
@inproceedings{lim-lee-2024-improving,
  title     = {Improving Readability Assessment with Ordinal Log-Loss},
  author    = {Lim, Ho Hung and Lee, John},
  booktitle = {Proceedings of the 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024)},
  year      = {2024},
  address   = {Mexico City, Mexico},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.bea-1.28/},
  pages     = {343--350}
}

@misc{trokhymovych2024readability,
  author        = {Trokhymovych, Mykola and Sen, Indira and Gerlach, Martin},
  title         = {An Open Multilingual System for Scoring Readability of {W}ikipedia},
  year          = {2024},
  eprint        = {2406.01835},
  archivePrefix = {arXiv}
}
```

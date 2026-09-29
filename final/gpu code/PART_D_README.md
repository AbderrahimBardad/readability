# Part D: three readability losses from other languages

`barec_exact_v3_partD.py` is your v3 script with `PART = "D"` and three losses
added. Data, splits, model, heads, training settings, seed, metrics, checkpoint
selection and guards are the v3 code, unchanged. Only four v3 lines were edited:
the part letter, the version date, one import and the WKL note in the header.

| Arm | Loss | Paper | Languages in the paper | Formula taken from | Compare with |
|---|---|---|---|---|---|
| LBL | Length-Balanced Loss | Li, Wang & Wu, EMNLP 2022 | English, Chinese | Authors' code, `liwb1219/zhfeat` | CE |
| WKL | Weighted Kappa Loss | de la Torre et al. 2018; used for readability by Lim & Lee, BEA 2024 | English, Chinese | BAREC `train.py` lines 87 to 114 | CE, SOFTQWK |
| RANK | MSE + Margin Ranking Loss | Trokhymovych, Sen & Gerlach 2024; Lee & Vajjala, Findings of ACL 2022 | 14 languages; English, French, Spanish | Paper's formula, `max(0, -y(S1-S2)+m)` | MSE |

## Notes per loss

- **LBL.** This is cross-entropy where each level k has the weight `c_k^ρ`. The
  value `c_k` is level k's share of `0.25 × (mean + min + median + max)` of its
  training texts' word counts. The authors tuned ρ per dataset at 0.4, 0.6 and
  0.8, so Part D uses 0.6. Lengths are counted on the D3Tok text the model reads.
- **WKL.** This is the exact code BAREC ships but never reported. Your v3 left it
  out by your earlier instruction. Part C's SOFTQWK is a different kappa loss on
  the regression head.
- **RANK.** A ranking score alone has no scale, so MSE anchors it to the 19
  levels. The weight λ = 1 is your own choice. Confirm the margin m = 0.5 in
  Trokhymovych et al. before citing it.

WKL and RANK compare samples inside a batch, so they use v3's rule D3: a true
batch of 64 with gradient checkpointing. That is also the batch BAREC's
`train.py` runs WKL at. LBL is a per-sample loss and uses v3's 16 × 4 batching.

## How to run

1. Use one Kaggle account with GPU T4 and Internet ON.
2. Paste the whole file as the first and only cell, then Save & Run All.
3. Download `results_backup_partD.zip` and unzip it next to the A, B and C results.

The three arms take about 1.5 hours in total. The JSON files have the v3 format
and the v3 checks: one GPU, batch 64, 857 steps per epoch, and reported equals
selected epoch.

## BibTeX to add

`delatorre2018wkl` and `lee2022neural` are already in `thesis/references.bib`.

```bibtex
@inproceedings{li-etal-2022-unified,
  title     = {A Unified Neural Network Model for Readability Assessment with Feature Projection and Length-Balanced Loss},
  author    = {Li, Wenbiao and Wang, Ziyang and Wu, Yunfang},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year      = {2022},
  address   = {Abu Dhabi, United Arab Emirates},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2022.emnlp-main.504/},
  pages     = {7446--7457}
}

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

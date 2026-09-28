# New losses on the v3 pipeline

`barec_newloss_v1_partA.py`, `_partB.py` and `_partC.py` are your
`barec_exact_v3` script with only the loss functions and the arm list changed.
The three files differ only in the line `PART = "A" / "B" / "C"`, like v3.

Heads, data, hyper-parameters, metrics, checkpoint selection, batch rules, the
one-GPU guard, the time guard and the report are all v3's own code. The results
go to a separate `newloss_v1` folder so they never mix with v3.

## Readability losses from the last five years

| Loss | Paper | Status |
|---|---|---|
| Ordinal Log Loss, soft labels, Earth Mover's Distance, Weighted Kappa Loss | Lim & Lee, BEA 2024 | First three already in v3; WKL left out by your choice |
| Soft labels + pairwise pre-training | Zeng et al., Findings of EMNLP 2022 | Variant of your SOFT arms; exact definition not verified |
| Length-Balanced Loss | Li, Wang & Wu, EMNLP 2022 | Formula could not be read from here |
| Pairwise ranking | Lee & Vajjala, Findings of ACL 2022; Trokhymovych et al. 2024 | **Part A** |
| Contrastive loss on relative difficulty | Zeng et al., AAAI 2024 | Formula not available |
| Ranking + sentence-document KL consistency | Zheng, Chen & Zhang 2025, arXiv 2511.21473 | Needs document-level input, so more than a loss change |

## What each part runs

| Part | Arm | Loss | Source | Compare with |
|---|---|---|---|---|
| A | `RANK` | MSE + pairwise margin ranking | Trokhymovych et al. 2024; Lee & Vajjala 2022 | v3 MSE |
| B | `RANKGAP` | MSE + ranking whose margin grows with the level gap | **new, proposed by this thesis** | v3 MSE and RANK |
| C | `CBOLL` | OLL15 with class-balanced weights | **new, proposed by this thesis**, built on Lim & Lee 2024 and Cui et al. 2019 | v3 OLL15 |

- **RANK** is the published readability loss. For every pair in the batch where
  text i is harder than text j, it adds `max(0, 0.5 - (s_i - s_j))`, plus the
  usual MSE.
- **RANKGAP** uses the margin `0.5 × (y_i - y_j)`, so texts two levels apart
  must be scored at least 1.0 apart. It targets the squeezed scale that STBW
  report, where levels 18 and 19 are predicted as 16. On adjacent levels it is
  identical to RANK, so RANK is its exact control.
- **CBOLL** multiplies each sample's OLL15 loss by a weight for its gold level.
  The weight is `(1-β)/(1-β^n)` with β = 0.999, where n is the number of training
  rows at that level. The weights are rescaled so the average weight over the
  training rows is 1. Level 18, with 67 training rows, gets a weight of about 13.
  Level 12, with 11,318 rows, gets about 0.9. This answers STBW's printed plan
  to try loss weighting. Watch macro-F1 and the extreme levels, not only QWK.

RANK and RANKGAP compare pairs inside a batch, so they use v3's rule for
batch-level losses: a true batch of 64 with gradient checkpointing, as SOFTQWK
did. CBOLL is a per-sample loss and uses v3's 16 × 4 batching.

## Claims to phrase carefully

- Call RANKGAP and CBOLL "proposed in this thesis" and "not reported by any BAREC
  2025 paper". Do not call them the first such losses anywhere.
- Confirm the margin of 0.5 in Trokhymovych et al. before citing it.
- λ = 1 and β = 0.999 are your own choices. The script lists them under
  DECLARED DEFAULTS.

## How to run

1. Open one Kaggle notebook per part, with GPU T4 and Internet ON.
2. Paste the whole file as the first and only cell, then Save & Run All.
3. Download `results_backup_partA.zip`, `_partB.zip` and `_partC.zip`.

Each arm should take about 30 to 35 minutes. The JSON files have the same format
as v3, so they go straight into your existing tables.

## BibTeX to add

`lee2022neural`, `castagnos2022ordinal`, `nazzal2025zai` and `trigui2025stbw`
are already in `thesis/references.bib`.

```bibtex
@misc{trokhymovych2024readability,
  author        = {Trokhymovych, Mykola and Sen, Indira and Gerlach, Martin},
  title         = {An Open Multilingual System for Scoring Readability of {W}ikipedia},
  year          = {2024},
  eprint        = {2406.01835},
  archivePrefix = {arXiv}
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

@inproceedings{cui2019classbalanced,
  title     = {Class-Balanced Loss Based on Effective Number of Samples},
  author    = {Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}

@inproceedings{li-etal-2022-unified,
  title     = {A Unified Neural Network Model for Readability Assessment with Feature Projection and Length-Balanced Loss},
  author    = {Li, Wenbiao and Wang, Ziyang and Wu, Yunfang},
  booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year      = {2022},
  url       = {https://aclanthology.org/2022.emnlp-main.504/},
  pages     = {7446--7457}
}

@inproceedings{zeng-etal-2022-enhancing,
  title     = {Enhancing Automatic Readability Assessment with Pre-training and Soft Labels for Ordinal Regression},
  author    = {Zeng, Jinshan and Xie, Yudong and Yu, Xianglong and Lee, John S. Y. and Zhou, Ding-Xuan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  year      = {2022},
  url       = {https://aclanthology.org/2022.findings-emnlp.334/},
  pages     = {4557--4568}
}
```

Pages for Cui et al. were left out because they were not checked. Confirm the
Trokhymovych title on arXiv.

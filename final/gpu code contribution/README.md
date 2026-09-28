# Contribution runs: three new objectives on the v3 pipeline

`barec_contrib_v1_partA.py`, `_partB.py` and `_partC.py` are one script. Only the
line `PART = "A" / "B" / "C"` differs, exactly like `barec_exact_v3`. Everything
except the head, the loss and the batch rule is copied from v3 unchanged, so each
new arm is directly comparable with your v3 table.

| Part | Arm | Objective | Paper | Matched v3 control |
|---|---|---|---|---|
| A | `CORN` | Conditional ordinal regression, 18 independent binary outputs | Shi, Cao & Raschka 2023, *Pattern Analysis and Applications* 26(3):941–955 | `CORAL` |
| B | `MSE_RANK` | MSE + pairwise margin ranking on the regression score | Trokhymovych, Sen & Gerlach 2024 (arXiv 2406.01835); Lee & Vajjala 2022 (Findings of ACL) | `MSE` |
| C | `BMSE` | Balanced MSE, BNI form, learnable noise scale | Ren, Zhang, Yu & Liu, CVPR 2022, pp. 7926–7935 | `MSE` |

## Why these three and not others

None of the three appears in the BAREC corpus paper, in the ZAI or STBW papers,
in your thesis, or in any v3 arm. Each one answers something printed in a paper
you already cite.

- **CORN.** ZAI (p. 267) writes: *"Future work includes error analysis, and
  exploring alternative ordinal objectives (e.g., CORN)"*. CORN keeps CORAL's 18
  "y > level j" tasks but removes the shared weight vector and trains task j only
  on rows with y >= j. Your CORAL run is the exact control.
- **MSE_RANK.** Pairwise ranking is the objective that readability models in
  other languages use: Wikipedia readability in 14 languages (Trokhymovych et al.
  2024) and English, French and Spanish (Lee & Vajjala 2022). It is the one pick
  that directly answers "a loss used for readability in other languages".
- **BMSE.** STBW write that level-18/19 items are predicted as 16 *"due to the
  class imbalance"* and that they *"will investigate employing data and loss
  weighting techniques"*. Balanced MSE is that loss-weighting idea for a
  regression head, which is your strongest head family.

**CDW-CE is not an arm on purpose.** Its formula `-Σ log(1-p_i)·|i-c|^α` is
exactly your `loss_oll`. Your OLL15 arm therefore already is CDW-CE with α = 1.5.

## Formulas as implemented

- **CORN:** `L = -Σ_j Σ_{i: y_i >= j} [log σ(z_ij)·1{y_i > j} + log(1-σ(z_ij))·1{y_i <= j}] / Σ_j |S_j|`.
  The level is decoded as the count of `Π_{i<=j} σ(z_i) > 0.5`. The code matches
  `coral_pytorch.losses.corn_loss` to 1e-15 in the self-test.
- **MSE_RANK:** `L = MSE + λ · mean over pairs with y_i > y_j of max(0, m - (s_i - s_j))`,
  with m = 0.5 and λ = 1. The ranking term is torch's `MarginRankingLoss`.
- **BMSE:** `L = (s-y)²/(2σ²) + logsumexp_k[log π_k - (s-k)²/(2σ²)]`, multiplied by
  `(2σ²).detach()`. Here π is the train-label frequency of each of the 19 levels
  and σ is learned. With discrete levels this sum is exact.

## Values to check before you write them in the thesis

- **Ranking margin.** The margin 0.5 comes from the research summary of
  Trokhymovych et al. Confirm it in their paper.
- **Noise-scale learning rate.** The σ learning rate of 1e-2 follows the usage
  example in the Balanced MSE repository `jiawei-ren/BalancedMSE`. Confirm it there.
- **Own choices.** λ = 1, σ init = 1.0, and CORN's train-prior bias init are the
  thesis's own choices. The script lists them under DECLARED DEFAULTS.

## How to run

1. Open one Kaggle notebook per part, with GPU T4 and Internet ON.
2. Paste the whole file as the first and only cell, then Save & Run All.
3. Download `results_backup_partA.zip`, `_partB.zip` and `_partC.zip`.

Each arm should take about 30–40 minutes. One account can run all three files
one after another if you prefer.

The printed report shows each new arm next to your v3 numbers, the difference to
its matched control, and a test-set error-distance histogram. Each JSON also
stores the dev and test predictions, so confusion matrices can be drawn later
without re-running.

With one seed, a difference of a few tenths of a QWK point is inside seed noise.
Report the far-miss shares and the error histogram alongside QWK.

## BibTeX to add

```bibtex
@article{shi2023corn,
  title   = {Deep neural networks for rank-consistent ordinal regression based on conditional probabilities},
  author  = {Shi, Xintong and Cao, Wenzhi and Raschka, Sebastian},
  journal = {Pattern Analysis and Applications},
  volume  = {26},
  number  = {3},
  pages   = {941--955},
  year    = {2023},
  doi     = {10.1007/s10044-023-01181-9}
}

@inproceedings{ren2022balanced,
  author    = {Ren, Jiawei and Zhang, Mingyuan and Yu, Cunjun and Liu, Ziwei},
  title     = {Balanced {MSE} for Imbalanced Visual Regression},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  pages     = {7926--7935}
}

@misc{trokhymovych2024readability,
  author        = {Trokhymovych, Mykola and Sen, Indira and Gerlach, Martin},
  title         = {An Open Multilingual System for Scoring Readability of {W}ikipedia},
  year          = {2024},
  eprint        = {2406.01835},
  archivePrefix = {arXiv}
}
```

The Trokhymovych title was not re-checked against arXiv, so confirm it on the
arXiv page. `lee2022neural`, `nazzal2025zai` and `trigui2025stbw` are already in
`thesis/references.bib`.

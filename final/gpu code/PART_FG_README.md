# Parts F and G: six more loss functions

`barec_exact_v3_partF.py` and `barec_exact_v3_partG.py` are the same file. They differ only in
the line `PART = "F"` / `PART = "G"`, like Parts A, B and C.

Everything except the loss is the v3 code:
- the same BAREC corpus and official splits;
- the same D3Tok input and AraBERTv2 encoder;
- the same lr 5e-5, 6 epochs, seed 42 and batch 64;
- the same lowest-dev-loss selection, metrics, guards and JSON files.

None of the six losses repeats an earlier run: not Parts A–C, not Part D (CORN, TSOFT12,
LBL10) and not Part E (WKL, RANK, PLR).

## How to run

Run them the same way as Parts D and E: one part per Kaggle account, GPU T4, Internet ON. Paste
the whole file as the first and only cell, then use Save Version → Save & Run All. Each part
writes `results_backup_partF.zip` or `results_backup_partG.zip`.

- **Part F** (about 1.5 h): MV, DLDL2, CLM. These use per-sentence losses, so they train at
  16 × 4 like CE.
- **Part G** (about 1.8 h): OE, LISTNET, CCC. These use batch losses, so they train at a true
  batch of 64 with gradient checkpointing, like SOFTQWK and Part E.

## The six losses

| Arm | Paper | Code it was checked against | What changes | Control |
|---|---|---|---|---|
| MV | Pan, Han, Shan & Chen, *Mean-Variance Loss for Deep Age Estimation from a Face*, CVPR 2018 | github.com/Herosan163/AgeEstimation | CE + 0.2 × mean loss + 0.05 × variance loss; predicts the expected level | CE |
| DLDL2 | Gao, Zhou, Wu & Geng, *Age Estimation Using Expectation of Label Distribution Learning*, IJCAI 2018 | authors' code github.com/gaobb/DLDL-v2 (`ldklexpl1`, `-lambda 1`) | KL to a Gaussian label distribution (σ = 2) + 1.0 × L1 of the expected level; predicts the expected level | SOFT2, CE |
| CLM | McCullagh, *Regression Models for Ordinal Data*, JRSS-B 1980; Vargas, Gutiérrez & Hervás-Martínez, *Cumulative link models for deep ordinal classification*, Neurocomputing 2020 | dlordinal `output_layers/clm.py` (same research group) | one score and 18 ordered thresholds, trained by maximum likelihood | CORAL |
| OE | Zhang et al., *Improving Deep Regression with Ordinal Entropy*, ICLR 2023 | authors' code github.com/needylove/OrdinalEntropy (`AgeEstimation/OrdinalEntropy.py`) | MSE + ordinal entropy of the 768-d sentence vectors | MSE |
| LISTNET | Cao, Qin, Liu, Tsai & Li, *Learning to Rank: From Pairwise Approach to Listwise Approach*, ICML 2007 | loop in the self-test | MSE + listwise ranking loss (the listwise version of RANK and PLR) | MSE |
| CCC | Lin, *A Concordance Correlation Coefficient to Evaluate Reproducibility*, Biometrics 1989; used as a training loss by Trigeorgis et al., ICASSP 2016 | loop in the self-test, and sklearn's QWK | 1 − concordance correlation of the batch | MSE, SOFTQWK |

**Why CCC is interesting.** On whole-number predictions, the concordance correlation of a batch
is exactly the quadratic weighted kappa. The self-test checks this against sklearn and they
agree to 1e-11. That makes CCC a QWK objective without the extra choices Soft-QWK needs: no
Gaussian σ and no warm-up phase.

## Self-tests (run at the start of every run)

All 22 loss checks agree to better than 1e-9:
- the nine v3 checks;
- MV against the public reimplementation;
- DLDL2 against the authors' Torch code;
- the CLM probabilities, loss and decoding against dlordinal;
- OE against the authors' function;
- LISTNET and CCC against loops;
- CCC against QWK.

## Values that are choices, not paper facts

These are all printed under DECLARED DEFAULTS in the run log.

- **MV weights 0.2 and 0.05.** These are the values in the public reimplementation. Check that
  Section 4 of the CVPR paper states the same values. The paper's PDF could not be opened from
  this environment.
- **DLDL2 σ = 2.** This is the DLDL value for data without a per-sample σ, given in levels here.
- **CLM.** It is trained with the likelihood. Vargas et al. 2020 used a QWK loss with the same
  layer. The thresholds start at 0, 1, …, 17, the library default.
- **LISTNET weight 1.0.** This is our choice, the same as RANK and PLR.
- **CCC ε = 1e-9.** It only prevents a 0/0 batch.
- **OE edge cases.** OE returns 0 for an eval batch with a single level, where the official code
  would stop with an error.

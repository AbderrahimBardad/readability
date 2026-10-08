# =============================================================================
#  barec_exact_v3_partF.py / _partG.py   PARTS F AND G -- six further losses,
#                             each from its own paper. The two files are the
#                             same file except the single line PART = "F" / "G"
#                             (as Parts A, B and C). This is
#                             barec_exact_v3_partA.py with that PART line, the six
#                             losses, their self-tests and their arms added, plus
#                             three small hooks: a per-arm decoder, a CLM head and
#                             a pooler hook for the one loss that reads features.
#                             Data, splits, D3Tok input, AraBERTv2, training
#                             arguments, seed, metrics, selection rule, batch
#                             rules, guards and report are the v3 code, unchanged.
#
#  No overlap with Parts A-C (CE, MSE, OLL15, EMD, SOFT2-5, CORAL, CORAL_HEADLR,
#  SOFTQWK), Part D (CORN, TSOFT12, LBL10) or Part E (WKL, RANK, PLR).
#
#  PART F ARMS -- 19-way heads read as distributions (one Kaggle account, ~1.5 h)
#   MV      Mean-variance loss. Pan, Han, Shan & Chen, CVPR 2018, pp. 5285-5294.
#           CE + 0.2 * mean loss + 0.05 * variance loss; predicts the expected
#           level. Head: v3's 19-way head. Control: CE.
#   DLDL2   DLDL-v2. Gao, Zhou, Wu & Geng, IJCAI 2018, pp. 712-718. KL to a
#           Gaussian label distribution (sigma = 2) + 1.0 * L1 of the expected
#           level; predicts the expected level. Transcribed from the authors'
#           code, github.com/gaobb/DLDL-v2 (train_mtnet.lua, loss ldklexpl1).
#           Head: v3's 19-way head. Control: SOFT2.
#   CLM     Cumulative link model, McCullagh, JRSS-B 42 (1980), trained by
#           maximum likelihood; deep layer of Vargas, Gutierrez & Hervas-
#           Martinez, Neurocomputing 401 (2020), pp. 48-58, transcribed from their group's
#           library dlordinal (output_layers/clm.py, logit link). One projection
#           and 18 ordered thresholds, like CORAL, but one likelihood instead of
#           18 separate binary losses. Control: CORAL.
#
#  PART G ARMS -- regression head, batch losses (one Kaggle account, ~1.8 h)
#   OE      Ordinal entropy. Zhang, Yang, Mi, Zheng & Yao, "Improving Deep
#           Regression with Ordinal Entropy", ICLR 2023 (arXiv 2301.08915).
#           MSE + (tightness - diversity) of the pooled sentence vectors,
#           transcribed from the authors' AgeEstimation/OrdinalEntropy.py
#           (github.com/needylove/OrdinalEntropy). Control: MSE.
#   LISTNET MSE + listwise top-one ranking loss. Cao, Qin, Liu, Tsai & Li,
#           ICML 2007. The listwise sibling of Part E's pairwise RANK and PLR.
#           Control: MSE.
#   CCC     1 - Lin's concordance correlation coefficient, Lin, Biometrics 45
#           (1989), pp. 255-268; maximised directly as a training objective by
#           Tzirakis, Trigeorgis, Nicolaou, Schuller & Zafeiriou, IEEE J. Sel.
#           Topics Signal Process. 11(8), 2017. On integer predictions the batch
#           CCC IS the quadratic weighted kappa (self-tested below), so this is
#           a QWK objective with no extra hyper-parameter. Controls: MSE,
#           SOFTQWK.
#   OE, LISTNET and CCC are statistics of the whole batch, so they use v3's
#   rule D3 (true batch 64, gradient checkpointing), as SOFTQWK and Part E.
#
# -----------------------------------------------------------------------------
#  barec_exact_v3_partA.py   (barec_exact_v3_partB.py and _partC.py are this
#                             same file with the single line PART = "B" / "C")
#
#  Ordinal Regression Heads and Loss Functions for Fine-Grained Arabic
#  Readability Assessment -- BAREC corpus, AraBERTv2 + D3Tok, 19 levels.
#
#  v3 = barec_exact_v2_partA/B.py plus ONLY the fixes that the 2026-09-14 audit
#  proved from the v2 result files. Losses, metrics, data loading, the 7/5/3
#  maps, hyper-parameters, the arm list and the selection RULE are unchanged.
#
#  WHAT v2 DID WRONG, AND THE FIX IN THIS FILE
#   1. The realised batch was 128, not 64. Kaggle's "GPU T4 x2" exposes two
#      GPUs; the HF Trainer then wraps the model in DataParallel and multiplies
#      the per-device batch by 2 (16 x 2 x 4 = 128; SOFTQWK 64 x 2 = 128). Every
#      v2 JSON logs 429 steps per epoch instead of 857, while recording
#      "train_batch": 64. Autocast is thread-local, so the DataParallel
#      replicas also ran their forward passes in fp32.
#      FIX: CUDA_VISIBLE_DEVICES=0 is set before torch is imported; the run
#      aborts if more than one GPU is visible; before every training run it
#      aborts unless the realised batch is 64; every JSON records the GPU count,
#      the realised batch and the observed steps per epoch (857 expected).
#   2. In 9 of 11 v2 arms the reported dev/test numbers were not the selected
#      epoch's own evaluation (up to 0.107 QWK points). The torch.save snapshot
#      reproduced the training history exactly in all 11 arms; HF's
#      save_pretrained reload did not.
#      FIX: the lowest-dev-loss epoch (same rule, strict "<" as HF) is
#      snapshotted with torch.save, and the reported dev/test come from that
#      snapshot. The reloaded dev metrics are compared with the history at that
#      epoch and stored as "selection_check". SOFTQWK phase 2 starts from the
#      phase-1 snapshot.
#   3. restore_previous_results accepted any JSON, including v2's batch-128
#      ones. FIX: it only restores v3 results recorded on one GPU at batch 64.
#   4. A commit that hits Kaggle's 12-hour limit loses its output. FIX: an arm
#      is not started unless it is expected to finish before the limit; the run
#      then stops cleanly, reports, zips, and the next commit resumes.
#   5. The CORAL projection used PyTorch's default init; the softmax and
#      regression heads get HF's normal(0, initializer_range). FIX: the CORAL
#      projection gets the same HF init, so the core arms differ in head and
#      loss only.
#   6. Values that can affect results but were undeclared are now in
#      DECLARED_DEFAULTS; comments the audit found wrong are corrected.
#
#  HOW TO RUN (one part per Kaggle account, all three in parallel)
#   Part A: CE, MSE, OLL15, EMD      Part B: SOFT2, SOFT3, SOFT4, SOFT5
#   Part C: CORAL, CORAL_HEADLR, SOFTQWK
#   Part F: MV, DLDL2, CLM          Part G: OE, LISTNET, CCC (these files)
#   - Accelerator: GPU (T4 x2 is fine -- this file uses exactly one GPU).
#   - Internet ON. Paste the whole file as the FIRST and ONLY cell.
#   - Save Version -> Save & Run All.
#   - If a commit stops early (the time guard prints why), open a new version,
#     "Add Input" -> this notebook -> the previous version's output, and run
#     again: finished v3 arms are restored and skipped.
#   - Download results_backup_partA.zip, _partB.zip and _partC.zip and unzip
#     all three into the same folder.
#
# -----------------------------------------------------------------------------
#  PROVENANCE -- github.com/CAMeL-Lab/barec_analyzer scripts/train.py (commit
#  4ed1243) and Elmadani, Habash & Taha-Thomure 2025 (arXiv 2502.13520).
#
#    set_seed(42)                                   train.py:378
#    load_dataset("CAMeL-Lab/BAREC-Corpus-v1.0")    train.py:380
#    input variant = D3Tok (a column)               train.py:383
#    label = Readability_Level_19 - 1               train.py:384
#    truncation=True, max_length=512                train.py:409
#    DataCollatorWithPadding (dynamic)              train.py:412
#    AutoModelForSequenceClassification(19)         train.py:417
#    evaluation_strategy="epoch"                    train.py:477
#    num_train_epochs=6                             train.py:478
#    train batch 64                                 train.py:479
#    eval batch 16                                  train.py:480
#    load_best_model_at_end=True                    train.py:481
#    metric_for_best_model="eval_loss"              train.py:482
#    greater_is_better=False                        train.py:483
#    save_strategy="epoch", save_total_limit=1      train.py:484, 487
#    learning_rate commented out (train.py:488) -> HF default 5e-5 = BAREC 5.5
#    everything else HF default: weight decay 0, warmup 0, linear schedule,
#    max_grad_norm 1.0, AdamW(0.9, 0.999, 1e-8). No early stopping.
#
#  CORRECTIONS TO THE PUBLISHED SCRIPT
#   - The released dataset's dev split is "dev"; train.py:388 reads
#     "validation".
#   - train.py has no Reg/MSE trainer; the MSE arm follows BAREC 5.4.
#   - train.py:374 maps "CORAL" to a plain Trainer; CORAL follows Cao et al.
#     2020 Eqs. 1, 4, 5.
#   - WKL (train.py:87-114) is not run in Parts A-C: the author's instruction.
#     Part E runs it, at the author's later request.
#
#  HARDWARE-FORCED DEVIATIONS -- nothing else
#   D1. Batch 64 as micro-batch 16 x accumulation 4 on ONE GPU. 54,845 rows
#       give 3,428 micro-batches = 857 x 4, i.e. 857 steps per epoch as with a
#       true batch of 64. Every step equals a true batch of 64 for every
#       per-sample-mean loss, except the last of each epoch (61 rows in both
#       cases; its final 13-row micro-batch is weighted slightly differently).
#   D2. fp16 autocast with fp32 master weights; every loss upcasts logits to
#       fp32 before any log / softmax / sigmoid.
#   D3. SOFTQWK's loss is a statistic of the whole batch, so accumulation is not
#       equivalent for it: it runs at a true per-device batch of 64 with
#       gradient checkpointing (changes memory and speed, not arithmetic).
# =============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # FIX 1 -- must run before torch is imported

import sys, json, gc, time, math, shutil, inspect, subprocess, platform
from datetime import datetime, timezone
from pathlib import Path

T_START = time.time()

PART = "G"   # the ONLY line that differs between the Part F and G files

SCRIPT_FAMILY = "barec_exact_v3"
SCRIPT_VERSION = f"{SCRIPT_FAMILY}_part{PART} (2026-10-08)"

# ----------------------------------------------------------------- 0. install
def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

try:
    import transformers, datasets  # noqa
except Exception:
    _pip("transformers", "datasets", "accelerate")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          TrainerCallback, set_seed)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error, f1_score

if torch.cuda.is_available() and torch.cuda.device_count() != 1:
    raise SystemExit(
        f"ABORT: {torch.cuda.device_count()} GPUs are visible. Run this file as the FIRST cell "
        f"of a fresh session so CUDA_VISIBLE_DEVICES=0 takes effect. Two GPUs double the "
        f"batch to 128 -- the v2 defect.")

# ----------------------------------------------------------------- 1. config
CONFIG = dict(
    model_name    = "aubmindlab/bert-base-arabertv2",   # BAREC 5.3
    dataset       = "CAMeL-Lab/BAREC-Corpus-v1.0",      # train.py:380
    input_var     = "D3Tok",                            # train.py:383
    label_col     = "Readability_Level_19",             # train.py:384
    max_length    = 512,                                # train.py:409
    micro_batch   = 16,                                 # D1
    grad_accum    = 4,                                  # D1 -> 64 on one GPU
    true_batch    = 64,                                 # train.py:479
    eval_batch    = 16,                                 # train.py:480
    learning_rate = 5e-5,                               # HF default; BAREC 5.5
    epochs        = 6,                                  # train.py:478
    warmup_ratio  = 0.0,                                # HF default
    weight_decay  = 0.0,                                # HF default
    max_grad_norm = 1.0,                                # HF default
    seed          = 42,                                 # train.py:378
    num_labels    = 19,
    fp16          = torch.cuda.is_available(),          # D2
)

# ---- Part F/G hyper-parameters (the only new values in these files)
MV_LAMBDA1 = 0.2         # Pan et al. 2018, weight of the mean loss
MV_LAMBDA2 = 0.05        # Pan et al. 2018, weight of the variance loss
DLDL_SIGMA = 2.0         # DLDL (Gao et al. 2017/2018): sigma = 2 where the data give none
DLDL_LAMBDA = 1.0        # Gao et al. 2018, official command "-loss ldklexpl1 -lambda 1"
OE_WEIGHT = 1.0          # Zhang et al. 2023, official README: "loss = loss + oe_loss"
LISTNET_LAMBDA = 1.0     # ours, as RANK_LAMBDA in Part E: MSE + 1.0 * ListNet
CCC_EPS = 1e-9           # ours: only guards a 0/0 batch

EXPECTED_TRAIN_BATCH = 64
KAGGLE_LIMIT_S = 12 * 3600          # Kaggle kills a commit at 43,200 s (observed on v2)
FIRST_ARM_ESTIMATE_MIN = 150        # used only until one arm has finished in this session
SAFETY_MIN = 25                     # report, zip and notebook-to-HTML conversion
KAGGLE_INPUT = Path("/kaggle/input")

DECLARED_DEFAULTS = [
    ("D1 micro-batch x accumulation", "16 x 4 on one GPU",
     "hardware: a T4 has 16 GB. 857 steps/epoch as with batch 64; the last step of each epoch has 61 rows"),
    ("D2 precision", "fp16 autocast, fp32 master weights, losses in fp32",
     "hardware: fits the Kaggle time limit. train.py runs fp32"),
    ("D3 SOFTQWK batch", "true batch 64, gradient checkpointing",
     "Soft-QWK is a batch statistic, so accumulation is not equivalent for it"),
    ("SoftQWK Gaussian sigma", 1.0,
     "STBW publishes no sigma and no phase-2 code"),
    ("SoftQWK Gaussian centre", "the continuous prediction, clamped to [0, 18]",
     "STBW say 'centered at the predicted class'; a rounded centre has zero gradient"),
    ("SoftQWK denominator epsilon", 1e-9,
     "not in STBW; <W,E> is of order 1e3 at batch 64, so it only guards a 0/0 batch"),
    ("SoftQWK phase split", "MSE epochs 1-3, SoftQWK epochs 4-6",
     "STBW report epoch 3 as their best warm-up; keeps the 6-epoch budget"),
    ("SoftQWK phase-1 checkpoint", "lowest phase-1 dev MSE",
     "BAREC's selection rule applied to phase 1; uses dev"),
    ("SoftQWK phase-2 lr", 5e-5,
     "BAREC's lr, fresh optimiser AND fresh scheduler (assigning param_group['lr'] is "
     "overwritten by the next scheduler step)"),
    ("OLL clamp", "p <= 1 - eps before log1p(-p)",
     "equal to train.py:273 wherever train.py's value is finite; avoids inf * 0 = NaN "
     "at the gold level"),
    ("SOFT clamp", "p >= tiny before log(p)",
     "equal to train.py:160 wherever train.py's value is finite"),
    ("Regression decode", "rint (half to even), then clip to [0, 18]",
     "BAREC 5.4 says only 'round'"),
    ("CORAL decode rule", "count of sigmoid(z_k) > 0.5",
     "Cao et al. 2020 Eqs. 1 and 5. ZAI's dev-tuned threshold is an ablation only"),
    ("CORAL batch reduction", "sum over the 18 thresholds, mean over the batch",
     "Cao Eq. 4 sums over examples; every train.py loss means over the batch"),
    ("CORAL bias init", "b_k = logit P(y > k) from TRAIN labels only",
     "ours; a head initialisation, not a pipeline change"),
    ("CORAL projection init", "normal(0, initializer_range = 0.02), no bias",
     "the init HF gives the softmax and regression heads"),
    ("CORAL head-lr ablation", 1e-3,
     "ours, ablation arm only; encoder stays at 5e-5; never in the core table"),
    ("ZAI threshold grid", "0.05 : 0.01 : 0.95, tuned on dev",
     "ablation only; its dev numbers leak by construction"),
    ("QWK-selected checkpoint", "highest dev QWK",
     "ZAI / STBW rule; ablation only; its dev numbers leak by construction"),
    ("far>=3 / far>=5 metrics", "share of |pred - gold| >= 3 and >= 5",
     "OURS, not a BAREC metric. Must be labelled as the thesis's own everywhere"),
    ("Kaggle split", "Parts A, B and C on three accounts, plus a 12-hour time guard",
     "OURS. Scheduling only; no arm's hyper-parameters change"),
    ("MV weights", "lambda1 = 0.2 (mean), lambda2 = 0.05 (variance)",
     "Pan et al. 2018; the values of the public reimplementation github.com/Herosan163/AgeEstimation"),
    ("MV mean-loss reduction", "mean over the batch of (m - y)^2 / 2",
     "Pan et al. 2018 Eq. for L_m (1/2N); variance loss mean over the batch"),
    ("MV / DLDL2 decode", "expected level sum_k k * p_k, rint, clip to [0, 18]",
     "both papers predict the expectation; rint as v3's regression decode"),
    ("DLDL2 label distribution", "exp(-(k - y)^2 / (2 sigma^2)), normalised over the 19 levels, sigma 2",
     "train_mtnet.lua genld; sigma = 2 is the DLDL value where no per-sample sigma exists"),
    ("DLDL2 reductions", "KL summed / (batch x 19); L1 averaged over the batch",
     "official Torch code: nn.KLDivCriterion(true) averages over all elements, nn.AbsCriterion() over the batch"),
    ("DLDL2 lambda", DLDL_LAMBDA, "official README command: -loss ldklexpl1 -lambda 1"),
    ("CLM layer", "P(y <= k) = sigmoid(theta_k - f), theta = cumsum(b, a_1^2, ..., a_17^2)",
     "dlordinal CLM (logit link, min_distance 0), Vargas et al. 2020"),
    ("CLM init", "b = 0, a = 1 (thresholds 0, 1, ..., 17); projection normal(0, 0.02), no bias",
     "dlordinal defaults; projection initialised as v3's CORAL projection (FIX 5)"),
    ("CLM loss", "negative log-likelihood of the gold level, computed in log space",
     "McCullagh 1980 maximum likelihood. Vargas et al. 2020 trained the layer with a QWK loss"),
    ("CLM decode", "argmax of the 19 level probabilities", "Vargas et al. 2020 / dlordinal"),
    ("OE", "MSE + 1.0 * (tightness - diversity)",
     "AgeEstimation/OrdinalEntropy.py of the authors; README: loss = loss + oe_loss"),
    ("OE feature", "the encoder's pooler output, before dropout, read by a forward hook",
     "the vector the regression head reads, as the official code's 'encoding'"),
    ("OE degenerate batches", "diversity 0 with one level; tightness 0 if every row is its own centre",
     "the official code raises (one level) or returns NaN there; possible only in eval batches of 16"),
    ("LISTNET", "MSE + 1.0 * ( -sum_i softmax(y)_i log softmax(s)_i ) over the batch",
     "Cao et al. 2007 top-one ListNet; lambda as RANK/PLR (ours)"),
    ("CCC", "1 - 2 cov(s, y) / (var s + var y + (mean s - mean y)^2 + 1e-9), population moments",
     "Lin 1989; on integer predictions equal to QWK (self-tested)"),
    ("OE / LISTNET / CCC batch", "true batch 64, gradient checkpointing (D3)",
     "all three are statistics of the batch"),
    ("Part F/G dev-loss selection", "each arm's own loss, on eval batches of 16",
     "BAREC's rule, as Part E applies it"),
]

SMOKE = False          # True -> 2000 train rows, 1 epoch, 12 dev/test rows per level. Plumbing only.

BASE = (Path("/kaggle/working") / "v3") if Path("/kaggle").exists() else Path("./barec_out_v3")
RESULTS = BASE / "results"
CKPT = BASE / "ckpt"
RESULTS.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

DATASET_SHA = None


# ----------------------------------------------------------------- 2. collapse maps (train.py:31-39)
BAREC_7 = {1:1,2:1,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:4,11:4,12:5,13:5,14:6,15:6,16:7,17:7,18:7,19:7}
BAREC_5 = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:2,9:2,10:2,11:2,12:3,13:3,14:4,15:4,16:5,17:5,18:5,19:5}
BAREC_3 = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:2,13:2,14:3,15:3,16:3,17:3,18:3,19:3}

K = CONFIG["num_labels"]
_LEVELS = np.arange(K)
_W = (_LEVELS[:, None] - _LEVELS[None, :]) ** 2
_D = np.abs(_LEVELS[:, None] - _LEVELS[None, :]).astype(np.float64)


def _collapse(y0, table):
    return np.array([table[int(v) + 1] for v in y0])


# ----------------------------------------------------------------- 3. metrics
def metrics_from_preds(gold0, pred0):
    gold0 = np.asarray(gold0).astype(int)
    pred0 = np.asarray(pred0).astype(int)
    d = np.abs(gold0 - pred0)
    return {
        "qwk":        float(cohen_kappa_score(gold0, pred0, weights="quadratic")),
        "acc19":      float(accuracy_score(gold0, pred0)),
        "acc19_pm1":  float((d <= 1).mean()),
        "dist":       float(mean_absolute_error(gold0, pred0)),
        "acc7":       float(accuracy_score(_collapse(gold0, BAREC_7), _collapse(pred0, BAREC_7))),
        "acc5":       float(accuracy_score(_collapse(gold0, BAREC_5), _collapse(pred0, BAREC_5))),
        "acc3":       float(accuracy_score(_collapse(gold0, BAREC_3), _collapse(pred0, BAREC_3))),
        "macro_f1":   float(f1_score(gold0, pred0, average="macro", zero_division=0)),
        "far_ge3":    float((d >= 3).mean()),      # THIS THESIS'S OWN metric
        "far_ge5":    float((d >= 5).mean()),      # THIS THESIS'S OWN metric
    }


# ----------------------------------------------------------------- 4. decoders
def decode_cls(logits):
    return np.argmax(logits, axis=1)


def decode_reg(logits):
    return np.clip(np.rint(np.asarray(logits).squeeze(-1)), 0, K - 1).astype(int)


def decode_coral(logits, t=0.5):
    return (1.0 / (1.0 + np.exp(-np.asarray(logits))) > t).sum(axis=1).astype(int)


def decode_exp(logits):
    """Parts F (MV, DLDL2): the expected level under the softmax, rounded."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return np.clip(np.rint(p @ np.arange(K)), 0, K - 1).astype(int)


def decode_clm(logits):
    """Part F (CLM): the most probable level of the cumulative link model."""
    z = torch.as_tensor(np.asarray(logits, dtype=np.float64))
    return _clm_logp(z).argmax(dim=1).numpy().astype(int)


# ----------------------------------------------------------------- 5. losses (unchanged from v2)
def _fp32(x):
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def _p(logits):
    return F.softmax(_fp32(logits), dim=1)


def _levels(like):
    return torch.arange(K, device=like.device, dtype=like.dtype)


def loss_ce(logits, labels, **kw):
    return F.cross_entropy(_fp32(logits), labels)


def loss_oll(logits, labels, alpha=1.5, **kw):
    p = _p(logits)
    d = (labels.unsqueeze(1).to(p.dtype) - _levels(p).unsqueeze(0)).abs()
    p_safe = p.clamp(max=1.0 - torch.finfo(p.dtype).eps)
    return (-torch.log1p(-p_safe) * d.pow(alpha)).sum(dim=1).mean()


def loss_emd(logits, labels, **kw):
    p = _p(logits)
    cdf_true = (_levels(p).unsqueeze(0) >= labels.unsqueeze(1)).to(p.dtype)
    return ((torch.cumsum(p, dim=1) - cdf_true) ** 2).sum(dim=1).mean()


_SOFT_DENOM = {}


def loss_soft(logits, labels, beta=3.0, **kw):
    p = _p(logits)
    key = (beta, p.device, p.dtype)
    if key not in _SOFT_DENOM:
        grid = torch.tensor(_D, dtype=p.dtype, device=p.device)
        _SOFT_DENOM[key] = torch.exp(-beta * grid).sum(dim=0)
    d = (labels.unsqueeze(1).to(p.dtype) - _levels(p).unsqueeze(0)).abs()
    target = torch.exp(-beta * d) / _SOFT_DENOM[key].unsqueeze(0)
    return (-torch.log(p.clamp(min=torch.finfo(p.dtype).tiny)) * target).sum(dim=1).mean()


def loss_mse(logits, labels, **kw):
    z = _fp32(logits).squeeze(-1)
    return F.mse_loss(z, labels.to(z.dtype))


def loss_coral(logits, labels, **kw):
    z = _fp32(logits)
    t = (labels.unsqueeze(1) > torch.arange(K - 1, device=z.device).unsqueeze(0)).to(z.dtype)
    return F.binary_cross_entropy_with_logits(z, t, reduction="none").sum(dim=1).mean()


def loss_softqwk(logits, labels, sigma=1.0, **kw):
    z = _fp32(logits)
    yhat = z.squeeze(-1).clamp(0, K - 1)
    lv = torch.arange(K, device=z.device, dtype=z.dtype).unsqueeze(0)
    g = -((lv - yhat.unsqueeze(1)) ** 2) / (2 * sigma ** 2)
    P = F.softmax(g, dim=1)
    T = F.one_hot(labels, K).to(z.dtype)
    O = T.t() @ P
    hist_t, hist_p = O.sum(dim=1), O.sum(dim=0)
    E = torch.outer(hist_t, hist_p) / yhat.shape[0]
    W = torch.tensor(_W, dtype=z.dtype, device=z.device)
    kappa = 1.0 - (W * O).sum() / ((W * E).sum() + 1e-9)
    return 1.0 - kappa


# ----------------------------------------------------------------- 5b. Part F/G losses
def loss_mv(logits, labels, l1=MV_LAMBDA1, l2=MV_LAMBDA2, **kw):
    """Mean-variance loss (Pan et al. 2018): CE + l1 * mean_N (m - y)^2 / 2
    + l2 * mean_N sum_k p_k (k - m)^2, with m = sum_k k p_k."""
    z = _fp32(logits)
    p = F.softmax(z, dim=1)
    lv = _levels(p)
    m = (p * lv.unsqueeze(0)).sum(dim=1)
    mean_loss = ((m - labels.to(p.dtype)) ** 2).mean() / 2.0
    var_loss = (p * (lv.unsqueeze(0) - m.unsqueeze(1)) ** 2).sum(dim=1).mean()
    return F.cross_entropy(z, labels) + l1 * mean_loss + l2 * var_loss


def loss_dldl2(logits, labels, sigma=DLDL_SIGMA, lam=DLDL_LAMBDA, **kw):
    """DLDL-v2 (Gao et al. 2018), as train_mtnet.lua 'ldklexpl1': KL(ld || p)
    averaged over all batch x 19 elements (nn.KLDivCriterion(true)) + lam *
    mean |sum_k k p_k - y| (nn.AbsCriterion())."""
    z = _fp32(logits)
    logp = F.log_softmax(z, dim=1)
    p = logp.exp()
    lv = _levels(p)
    y = labels.to(p.dtype)
    ld = torch.exp(-0.5 * ((lv.unsqueeze(0) - y.unsqueeze(1)) / sigma) ** 2)
    ld = ld / ld.sum(dim=1, keepdim=True)
    kl_el = torch.where(ld > 0, ld * (torch.log(ld.clamp(min=torch.finfo(ld.dtype).tiny)) - logp),
                        torch.zeros_like(ld))
    kl = kl_el.sum() / ld.numel()
    l1 = ((p * lv.unsqueeze(0)).sum(dim=1) - y).abs().mean()
    return kl + lam * l1


def _clm_logp(z):
    """log P(y = k) of the cumulative link model from z_k = theta_k - f
    (N x 18, non-decreasing in k): P(y = 0) = s(z_0), P(y = 18) = 1 - s(z_17),
    P(y = k) = s(z_k) - s(z_{k-1}) = s(z_k) s(-z_{k-1}) (1 - exp(z_{k-1} - z_k))."""
    tiny = torch.finfo(z.dtype).tiny
    first = F.logsigmoid(z[:, :1])
    last = F.logsigmoid(-z[:, -1:])
    a, b = z[:, 1:], z[:, :-1]
    mid = F.logsigmoid(a) + F.logsigmoid(-b) + torch.log((-torch.expm1(b - a)).clamp(min=tiny))
    return torch.cat([first, mid, last], dim=1)


def loss_clm(logits, labels, **kw):
    """Negative log-likelihood of the cumulative link model (McCullagh 1980)."""
    return -_clm_logp(_fp32(logits)).gather(1, labels.unsqueeze(1)).mean()


def _euclid(x, y):
    """OrdinalEntropy.py euclidean_dist; addmm_(1, -2, x, y.t()) in today's signature."""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = (xx + yy).addmm(x, y.t(), beta=1, alpha=-2)
    return dist.clamp(min=1e-12).sqrt()


def _up_triu(x):
    n = x.shape[0]
    return x[torch.triu(torch.ones(n, n, device=x.device), diagonal=1).to(torch.bool)]


def ordinal_entropy(features, gt):
    """AgeEstimation/OrdinalEntropy.py (Zhang et al. 2023) on any device:
    tightness of the normalised features around their level centres minus the
    label-distance-weighted spread of the normalised centres."""
    f = _fp32(features)
    f_c = f.shape[1]
    u_value, u_index, u_counts = torch.unique(gt, return_inverse=True, return_counts=True)
    center_f = torch.zeros(len(u_value), f_c, device=f.device, dtype=f.dtype)
    u_index = u_index.reshape(-1)
    center_f.index_add_(0, u_index, f)
    center_f = center_f / u_counts.unsqueeze(1).to(f.dtype)
    p = F.normalize(center_f, dim=1)
    if len(u_value) > 1:
        _distance = _up_triu(_euclid(p, p))
        uv = u_value.to(f.dtype).unsqueeze(1)
        _weight = _up_triu(_euclid(uv, uv))
        _weight = (_weight - torch.min(_weight)) / torch.max(_weight)
        _entropy = torch.mean(_distance * _weight)
    else:
        _entropy = f.new_zeros(())
    _features = (F.normalize(f, dim=1) - p[u_index, :]).pow(2)
    _tightness = torch.sum(_features, dim=1)
    _mask = _tightness > 0
    _tightness = torch.sqrt(_tightness[_mask]).mean() if bool(_mask.any()) else f.new_zeros(())
    return _tightness - _entropy


def loss_oe(logits, labels, feats=None, weight=OE_WEIGHT, **kw):
    """MSE + weight * ordinal entropy of the pooled sentence vectors."""
    assert feats is not None, "OE needs the pooled features (FeatArmTrainer)"
    z = _fp32(logits).squeeze(-1)
    return F.mse_loss(z, labels.to(z.dtype)) + weight * ordinal_entropy(feats, labels)


def loss_listnet(logits, labels, lam=LISTNET_LAMBDA, **kw):
    """MSE + lam * top-one ListNet (Cao et al. 2007) over the batch:
    -sum_i softmax(y)_i * log softmax(s)_i."""
    s = _fp32(logits).squeeze(-1)
    y = labels.to(s.dtype)
    listnet = -(F.softmax(y, dim=0) * F.log_softmax(s, dim=0)).sum()
    return F.mse_loss(s, y) + lam * listnet


def loss_ccc(logits, labels, **kw):
    """1 - Lin's concordance correlation coefficient of the batch (Lin 1989),
    population moments: 2 cov / (var_s + var_y + (mean_s - mean_y)^2)."""
    s = _fp32(logits).squeeze(-1)
    y = labels.to(s.dtype)
    ms, my = s.mean(), y.mean()
    vs, vy = ((s - ms) ** 2).mean(), ((y - my) ** 2).mean()
    cov = ((s - ms) * (y - my)).mean()
    return 1.0 - 2.0 * cov / (vs + vy + (ms - my) ** 2 + CCC_EPS)


# ----------------------------------------------------------------- 6. self-tests
def self_test_losses():
    torch.manual_seed(0)
    n = 37
    logits = torch.randn(n, K, dtype=torch.float64) * 3
    labels = torch.randint(0, K, (n,))
    d_matrix = [[abs(i - j) for i in range(K)] for j in range(K)]
    probas = F.softmax(logits, dim=1)
    true_labels = [K * [labels[k].item()] for k in range(n)]
    label_ids = n * [[k for k in range(K)]]
    checks = []

    lit = float(np.mean([-math.log(probas[j, labels[j]].item()) for j in range(n)]))
    checks.append(("CE", lit, float(loss_ce(logits, labels))))

    dist = [[float(d_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(K)] for j in range(n)]
    dt = torch.tensor(dist, dtype=torch.float64)
    lit = float((-torch.log(1 - probas) * dt ** 1.5).sum(dim=1).mean())
    checks.append(("OLL15", lit, float(loss_oll(logits, labels, alpha=1.5))))

    cdf_true = torch.tensor([labels[k].item() * [0.] + (K - labels[k].item()) * [1.]
                             for k in range(n)], dtype=torch.float64)
    lit = float(((torch.cumsum(probas, dim=1) - cdf_true) ** 2).sum(dim=1).mean())
    checks.append(("EMD", lit, float(loss_emd(logits, labels))))

    for beta in (2, 3, 4, 5):
        softs = [[math.exp(-beta * d_matrix[true_labels[j][i]][label_ids[j][i]])
                  / sum(math.exp(-beta * d_matrix[nn_][label_ids[j][i]]) for nn_ in range(K))
                  for i in range(K)] for j in range(n)]
        st = torch.tensor(softs, dtype=torch.float64)
        lit = float((-torch.log(probas) * st).sum(dim=1).mean())
        checks.append((f"SOFT{beta}", lit, float(loss_soft(logits, labels, beta=float(beta)))))

    proj = torch.randn(n, 1, dtype=torch.float64)
    bias = torch.sort(torch.randn(K - 1, dtype=torch.float64), descending=True).values
    z = proj + bias
    t = (labels.unsqueeze(1) > torch.arange(K - 1).unsqueeze(0)).to(torch.float64)
    lit = float(np.mean([sum(math.log1p(math.exp(-abs(z[j, k].item())))
                             + max(z[j, k].item(), 0.0) - z[j, k].item() * t[j, k].item()
                             for k in range(K - 1)) for j in range(n)]))
    checks.append(("CORAL", lit, float(loss_coral(z, labels))))

    yhat = labels.to(torch.float64) + torch.randn(n, dtype=torch.float64)
    hard = np.clip(np.rint(yhat.numpy()), 0, K - 1).astype(int)
    lit = 1.0 - cohen_kappa_score(labels.numpy(), hard, weights="quadratic")
    got = float(loss_softqwk(torch.tensor(hard, dtype=torch.float64).unsqueeze(1),
                             labels, sigma=1e-3))
    checks.append(("SoftQWK->sklearn", lit, got))

    # --- Parts F/G: MV vs github.com/Herosan163/AgeEstimation mean_variance_loss.py
    #     (cuda calls dropped) plus a literal CE loop
    p_ = F.softmax(logits, dim=1)
    a_ = torch.arange(0, K - 1 + 1, dtype=torch.float64)
    mean_ = torch.squeeze((p_ * a_).sum(1, keepdim=True), dim=1)
    mean_loss_ = ((mean_ - labels.to(torch.float64)) ** 2).mean() / 2.0
    b_ = (a_[None, :] - mean_[:, None]) ** 2
    variance_loss_ = (p_ * b_).sum(1, keepdim=True).mean()
    ce_ = float(np.mean([-math.log(probas[j, labels[j]].item()) for j in range(n)]))
    lit = ce_ + MV_LAMBDA1 * float(mean_loss_) + MV_LAMBDA2 * float(variance_loss_)
    checks.append(("MV", lit, float(loss_mv(logits, labels))))

    # --- DLDL2 vs train_mtnet.lua: genld loop, KLDivCriterion(true), AbsCriterion()
    kl_tot, l1_tot = 0.0, 0.0
    for j in range(n):
        ld = [math.exp(-0.5 * ((k - labels[j].item()) / DLDL_SIGMA) ** 2) for k in range(K)]
        sld = sum(ld)
        ld = [v / sld for v in ld]
        logp_j = [math.log(probas[j, k].item()) for k in range(K)]
        kl_tot += sum(t * (math.log(t) - x) for t, x in zip(ld, logp_j) if t > 0)
        l1_tot += abs(sum(k * probas[j, k].item() for k in range(K)) - labels[j].item())
    lit = kl_tot / (n * K) + DLDL_LAMBDA * l1_tot / n
    checks.append(("DLDL2", lit, float(loss_dldl2(logits, labels))))

    # --- decode_exp vs a loop
    lit_dec = [int(np.clip(np.rint(sum(k * probas[j, k].item() for k in range(K))), 0, K - 1))
               for j in range(n)]
    checks.append(("decode_exp", 0.0, float(np.abs(np.array(lit_dec) - decode_exp(logits.numpy())).sum())))

    # --- CLM vs dlordinal output_layers/clm.py (_convert_thresholds, _compute_z3, _clm,
    #     logit link), transcribed; probabilities, NLL and decode
    th_b = torch.randn(1, dtype=torch.float64)
    th_a = torch.rand(K - 2, dtype=torch.float64) + 0.1
    proj_ = torch.randn(n, dtype=torch.float64) * 3 + 9
    tp = torch.cat((th_b, th_a ** 2 + 0.0), dim=0)
    th = torch.sum(torch.tril(torch.ones((K - 1, K - 1), dtype=torch.float64), diagonal=0)
                   * torch.reshape(torch.tile(tp, (K - 1,)), shape=(K - 1, K - 1)), dim=(1,))
    z3 = torch.reshape(torch.tile(th, (n,)), shape=(n, -1)) - torch.transpose(
        torch.reshape(torch.tile(proj_, (K - 1,)), shape=(-1, n)), 0, 1)
    a3 = torch.cat((torch.sigmoid(z3), torch.ones((n, 1), dtype=torch.float64)), dim=1)
    a3[:, 1:] = a3[:, 1:] - a3[:, 0:-1]
    ours = _clm_logp(z3).exp()
    checks.append(("CLM probabilities", 0.0, float((ours - a3).abs().max())))
    lit = float(np.mean([-math.log(a3[j, labels[j]].item()) for j in range(n)]))
    checks.append(("CLM NLL", lit, float(loss_clm(z3, labels))))
    checks.append(("CLM decode", 0.0,
                   float(np.abs(a3.argmax(dim=1).numpy() - decode_clm(z3.numpy())).sum())))
    checks.append(("CLM cumsum", 0.0, float((torch.cumsum(tp, 0) - th).abs().max())))

    # --- OE vs AgeEstimation/OrdinalEntropy.py of the authors (.cuda() -> device,
    #     addmm_ positional signature -> keywords), on a batch with repeated levels
    def _official_oe(features, gt):
        f_n, f_c = features.size()
        u_value, u_index, u_counts = torch.unique(gt, return_inverse=True, return_counts=True)
        center_f = torch.zeros([len(u_value), f_c], dtype=features.dtype)
        u_index = u_index.squeeze()
        center_f.index_add_(0, u_index, features)
        u_counts = u_counts.unsqueeze(1)
        center_f = center_f / u_counts
        p = F.normalize(center_f, dim=1)

        def euclidean_dist(x, y):
            m_, n_ = x.size(0), y.size(0)
            xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m_, n_)
            yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n_, m_).t()
            dist = xx + yy
            dist = dist.addmm(x, y.t(), beta=1, alpha=-2)
            return dist.clamp(min=1e-12).sqrt()

        def up_triu(x):
            n_, m_ = x.shape
            return x[torch.triu(torch.ones(n_, n_), diagonal=1).to(torch.bool)]

        _distance = up_triu(euclidean_dist(p, p))
        u_value = u_value.unsqueeze(1).to(features.dtype)
        _weight = up_triu(euclidean_dist(u_value, u_value))
        _weight = ((_weight - torch.min(_weight)) / torch.max(_weight))
        _entropy = torch.mean(_distance * _weight)
        _features = F.normalize(features, dim=1)
        _features = (_features - p[u_index, :]).pow(2)
        _tightness = torch.sum(_features, dim=1)
        _tightness = torch.sqrt(_tightness[_tightness > 0]).mean()
        return _tightness - _entropy

    feats = torch.randn(n, 768, dtype=torch.float64)
    s = labels.to(torch.float64).unsqueeze(1) + torch.randn(n, 1, dtype=torch.float64) * 2
    mse = sum((float(s[i, 0]) - int(labels[i])) ** 2 for i in range(n)) / n
    checks.append(("OE official", float(_official_oe(feats, labels)), float(ordinal_entropy(feats, labels))))
    checks.append(("OE loss", mse + OE_WEIGHT * float(_official_oe(feats, labels)),
                   float(loss_oe(s, labels, feats=feats))))
    one = torch.full((n,), 7)
    oe_one = ordinal_entropy(feats, one)
    checks.append(("OE one level", 0.0, 0.0 if bool(torch.isfinite(oe_one)) else 1.0))

    # --- LISTNET vs a loop
    ey = [math.exp(int(labels[i])) for i in range(n)]
    es = [math.exp(float(s[i, 0])) for i in range(n)]
    lit = mse + LISTNET_LAMBDA * -sum((ey[i] / sum(ey)) * math.log(es[i] / sum(es)) for i in range(n))
    checks.append(("LISTNET loop", lit, float(loss_listnet(s, labels))))

    # --- CCC vs a loop, and on integer predictions vs sklearn's QWK over the 19 levels
    sv = [float(s[i, 0]) for i in range(n)]
    yv = [int(labels[i]) for i in range(n)]
    ms_, my_ = sum(sv) / n, sum(yv) / n
    vs_ = sum((v - ms_) ** 2 for v in sv) / n
    vy_ = sum((v - my_) ** 2 for v in yv) / n
    cv_ = sum((sv[i] - ms_) * (yv[i] - my_) for i in range(n)) / n
    checks.append(("CCC loop", 1.0 - 2.0 * cv_ / (vs_ + vy_ + (ms_ - my_) ** 2 + CCC_EPS),
                   float(loss_ccc(s, labels))))
    hard = np.clip(np.rint(s.numpy()[:, 0]), 0, K - 1).astype(int)
    qwk = cohen_kappa_score(labels.numpy(), hard, weights="quadratic", labels=list(range(K)))
    checks.append(("CCC == QWK ints", float(qwk),
                   1.0 - float(loss_ccc(torch.tensor(hard, dtype=torch.float64).unsqueeze(1), labels))))

    print("  loss self-test: CE/OLL15/EMD/SOFT vs train.py's literal loops; "
          "CORAL vs closed-form BCE; SoftQWK vs 1 - sklearn QWK")
    print("  Parts F/G: MV vs the public reimplementation; DLDL2 vs train_mtnet.lua; CLM vs "
          "dlordinal; OE vs the authors' OrdinalEntropy.py; LISTNET and CCC vs loops; CCC vs QWK")
    bad = []
    for name, a, b in checks:
        diff = abs(a - b)
        flag = "ok" if diff < 1e-9 else "FAIL"
        if diff >= 1e-9:
            bad.append(name)
        print(f"    {name:<18} {a:>18.12f}  {b:>18.12f}  {diff:.2e}  {flag}")
    assert not bad, f"loss mismatch: {bad}"


def self_test_metrics():
    rng = np.random.default_rng(1)
    gold = rng.integers(0, K, 500)
    pred = np.clip(gold + rng.integers(-4, 5, 500), 0, K - 1)
    m = metrics_from_preds(gold, pred)
    assert abs(m["dist"] - np.abs(gold - pred).mean()) < 1e-12
    assert abs(m["acc19_pm1"] - (np.abs(gold - pred) <= 1).mean()) < 1e-12
    assert abs(m["far_ge3"] - (np.abs(gold - pred) >= 3).mean()) < 1e-12
    perfect = metrics_from_preds(gold, gold)
    assert abs(perfect["qwk"] - 1.0) < 1e-12 and abs(perfect["acc3"] - 1.0) < 1e-12
    assert set(BAREC_7.values()) == set(range(1, 8))
    assert set(BAREC_5.values()) == set(range(1, 6))
    assert set(BAREC_3.values()) == set(range(1, 4))
    print("  metric self-test ok")


# ----------------------------------------------------------------- 7. data
def dataset_sha():
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(CONFIG["dataset"]).sha
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def load_barec():
    ds = load_dataset(CONFIG["dataset"])
    print("  splits as shipped:", {k: len(v) for k, v in ds.items()})
    dev_key = "dev" if "dev" in ds else "validation"
    assert dev_key in ds, f"no dev split; got {list(ds.keys())}"
    expect = {"train": 54845, dev_key: 7310, "test": 7286}
    for k, v in expect.items():
        assert len(ds[k]) == v, f"split {k} is {len(ds[k])}, paper Table 3 says {v}"
    iv, lc = CONFIG["input_var"], CONFIG["label_col"]
    assert iv in ds["train"].column_names, f"{iv} column missing"
    sample = ds["train"][iv][:2000]
    plus_rate = sum("+" in s for s in sample) / len(sample)
    assert plus_rate > 0.5, f"D3Tok column has '+' in only {plus_rate:.1%} of rows"
    print(f"  D3Tok '+' clitic marks present in {plus_rate:.1%} of a 2000-row sample")
    for split_key in ("train", dev_key, "test"):
        cols = ds[split_key].column_names
        checked = []
        for n_lv, table in ((7, BAREC_7), (5, BAREC_5), (3, BAREC_3)):
            col = f"Readability_Level_{n_lv}"
            if col not in cols:
                print(f"  WARNING: {split_key} has no {col}; train.py's {n_lv}-level map not cross-checked")
                continue
            a = np.array(ds[split_key][col])
            b = np.array([table[v] for v in ds[split_key][lc]])
            assert (a == b).all(), f"train.py's {n_lv}-level map disagrees with {col} on {split_key}"
            checked.append(n_lv)
        print(f"  {split_key}: collapse maps {checked} agree with the dataset's own columns")
    out = {}
    for split, key in (("train", "train"), ("dev", dev_key), ("test", "test")):
        out[split] = Dataset.from_dict({
            "text": ds[key][iv],
            "labels": [int(l) - 1 for l in ds[key][lc]],
        })
    assert set(out["dev"]["labels"]) == set(range(K)), "dev is missing a level"
    assert set(out["test"]["labels"]) == set(range(K)), "test is missing a level"
    if SMOKE:
        out["train"] = out["train"].select(range(2000))
        for s in ("dev", "test"):
            labs = np.array(out[s]["labels"])
            keep = sorted(int(i) for k in range(K) for i in np.flatnonzero(labs == k)[:12])
            out[s] = out[s].select(keep)
    return DatasetDict(out)


# ----------------------------------------------------------------- 8. model heads
class CoralModel(nn.Module):
    """AraBERTv2 pooler -> dropout -> ONE shared projection + (K-1) free biases
    (Cao et al. 2020). Dropout is read from the config exactly as
    BertForSequenceClassification reads it."""

    def __init__(self, checkpoint, bias_init=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        cfg = self.encoder.config
        p = getattr(cfg, "classifier_dropout", None)
        if p is None:
            p = cfg.hidden_dropout_prob
        self.dropout = nn.Dropout(p)
        self.proj = nn.Linear(cfg.hidden_size, 1, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=cfg.initializer_range)   # FIX 5
        b0 = torch.zeros(K - 1) if bias_init is None else torch.as_tensor(bias_init, dtype=torch.float32)
        self.cutpoints = nn.Parameter(b0.clone())
        self.config = cfg

    def gradient_checkpointing_enable(self, **kw):
        self.encoder.gradient_checkpointing_enable(**kw)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kw):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        pooled = self.dropout(out.pooler_output)
        return SequenceClassifierOutput(logits=self.proj(pooled) + self.cutpoints)


def coral_bias_init(train_labels):
    y = np.asarray(train_labels)
    ps = np.array([(y > k).mean() for k in range(K - 1)])
    ps = np.clip(ps, 1e-4, 1 - 1e-4)
    b = np.log(ps / (1 - ps))
    assert np.all(np.diff(b) <= 0), "P(y>k) must be non-increasing in k"
    return b.astype(np.float32)


class ClmModel(nn.Module):
    """Part F. AraBERTv2 pooler -> dropout -> ONE projection f (no bias) and the
    cumulative link layer of dlordinal (Vargas et al. 2020, logit link):
    theta = cumsum(b, a_1^2, ..., a_17^2), output z_k = theta_k - f, so that
    P(y <= k) = sigmoid(z_k). Thresholds start at 0, 1, ..., 17 (b = 0, a = 1)."""

    def __init__(self, checkpoint):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)
        cfg = self.encoder.config
        p = getattr(cfg, "classifier_dropout", None)
        if p is None:
            p = cfg.hidden_dropout_prob
        self.dropout = nn.Dropout(p)
        self.proj = nn.Linear(cfg.hidden_size, 1, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=cfg.initializer_range)
        self.thresholds_b = nn.Parameter(torch.zeros(1))
        self.thresholds_a = nn.Parameter(torch.ones(K - 2))
        self.config = cfg

    def gradient_checkpointing_enable(self, **kw):
        self.encoder.gradient_checkpointing_enable(**kw)

    def thresholds(self):
        return torch.cumsum(torch.cat([self.thresholds_b, self.thresholds_a ** 2]), dim=0)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kw):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        f = self.proj(self.dropout(out.pooler_output))
        return SequenceClassifierOutput(logits=self.thresholds().unsqueeze(0) - f)


# ----------------------------------------------------------------- 9. trainer and callbacks
class ArmTrainer(Trainer):
    def __init__(self, *a, loss_fn=None, loss_kwargs=None, head_lr=None, **kw):
        super().__init__(*a, **kw)
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs or {}
        self.head_lr = head_lr
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels, **self.loss_kwargs)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self, *a, **kw):
        if self.optimizer is not None or self.head_lr is None:
            return super().create_optimizer(*a, **kw)
        head, enc = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (head if n.split(".")[0] in ("proj", "cutpoints", "classifier") else enc).append(p)
        cls_, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        kwargs.pop("lr", None)
        self.optimizer = cls_([
            {"params": enc,  "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay},
            {"params": head, "lr": self.head_lr,            "weight_decay": self.args.weight_decay},
        ], lr=self.args.learning_rate, **kwargs)
        return self.optimizer


class FeatArmTrainer(ArmTrainer):
    """Part G (OE). The same model as v3's regression arm; a forward hook on the
    encoder's pooler hands its output (the vector the head's dropout and linear
    layer receive) to the loss as feats=. Used for training and dev loss alike."""

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        base = model.module if hasattr(model, "module") else model
        store = {}
        hook = base.base_model.pooler.register_forward_hook(
            lambda mod, inp, out: store.__setitem__("z", out))
        try:
            outputs = model(**inputs)
        finally:
            hook.remove()
        assert "z" in store, "pooler hook did not fire"
        loss = self.loss_fn(outputs.logits, labels, feats=store["z"], **self.loss_kwargs)
        return (loss, outputs) if return_outputs else loss


class BestSnapshots(TrainerCallback):
    """FIX 2. torch.save snapshots of the lowest-dev-loss epoch (BAREC's rule,
    strict '<' exactly like HF's load_best_model_at_end) and of the highest-
    dev-QWK epoch (ablation). The metrics HF logged at that epoch are kept so
    the reloaded snapshot can be checked against them."""

    def __init__(self, out_dir):
        self.loss_path = out_dir / "best_loss.pt"
        self.qwk_path = out_dir / "best_qwk.pt"
        self.best_loss, self.loss_epoch, self.loss_metrics = math.inf, None, None
        self.best_qwk, self.qwk_epoch, self.qwk_metrics = -math.inf, None, None

    @staticmethod
    def _save(model, path):
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        if not metrics or model is None:
            return
        ep = float(state.epoch or 0)
        loss, qwk = metrics.get("eval_loss"), metrics.get("eval_qwk")
        if loss is not None and loss < self.best_loss:
            self.best_loss, self.loss_epoch, self.loss_metrics = loss, ep, dict(metrics)
            self._save(model, self.loss_path)
        if qwk is not None and qwk > self.best_qwk:
            self.best_qwk, self.qwk_epoch, self.qwk_metrics = qwk, ep, dict(metrics)
            self._save(model, self.qwk_path)


class CutpointLogger(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        if not hasattr(model, "cutpoints"):
            return
        b = model.cutpoints.detach().float().cpu().numpy()
        spread = float(b.max() - b.min())
        ep = float(state.epoch or 0)
        self.history.append({"epoch": ep, "spread": spread,
                             "monotone": bool(np.all(np.diff(b) <= 1e-6))})
        if metrics is not None:
            metrics["eval_cutpoint_spread"] = spread
        print(f"    [CORAL] epoch {ep:.0f}  cut-point spread {spread:.3f}"
              f"  ordered={self.history[-1]['monotone']}")


# ----------------------------------------------------------------- 10. plumbing
_TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)
_TR_PARAMS = set(inspect.signature(Trainer.__init__).parameters)
_CRITICAL_ARGS = {
    "output_dir", "num_train_epochs", "per_device_train_batch_size",
    "gradient_accumulation_steps", "per_device_eval_batch_size", "learning_rate",
    "weight_decay", "lr_scheduler_type", "max_grad_norm", "save_strategy",
    "save_total_limit", "load_best_model_at_end", "metric_for_best_model",
    "greater_is_better", "seed", "fp16", "remove_unused_columns", "label_names",
    "gradient_checkpointing",
}
_DROPPED_REPORTED = set()


def make_args(out_dir, epochs, micro_bs, accum, grad_ckpt=False, **extra):
    kw = dict(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=micro_bs,
        gradient_accumulation_steps=accum,
        per_device_eval_batch_size=CONFIG["eval_batch"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],
        lr_scheduler_type="linear",
        max_grad_norm=CONFIG["max_grad_norm"],
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=CONFIG["seed"],
        data_seed=CONFIG["seed"],
        fp16=CONFIG["fp16"],
        logging_steps=100,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2,
        gradient_checkpointing=grad_ckpt,
    )
    kw.update(extra)
    kw["eval_strategy" if "eval_strategy" in _TA_PARAMS else "evaluation_strategy"] = "epoch"
    if grad_ckpt and "gradient_checkpointing_kwargs" in _TA_PARAMS:
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    dropped = sorted(k for k in kw if k not in _TA_PARAMS)
    critical = [k for k in dropped if k in _CRITICAL_ARGS]
    if critical:
        raise SystemExit(f"ABORT: installed transformers' TrainingArguments has no {critical}")
    new = [k for k in dropped if k not in _DROPPED_REPORTED]
    if new:
        _DROPPED_REPORTED.update(new)
        print(f"  note: installed transformers has no TrainingArguments {new}; "
              f"its defaults are used for them")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in _TA_PARAMS})


def tok_kwarg(tokenizer):
    return ({"processing_class": tokenizer} if "processing_class" in _TR_PARAMS
            else {"tokenizer": tokenizer})


def make_compute_metrics(decode):
    def fn(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return metrics_from_preds(p.label_ids, decode(logits))
    return fn


def predict_logits(trainer, ds):
    out = trainer.predict(ds)
    lg = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
    return np.asarray(lg), np.asarray(out.label_ids).astype(int)


def assert_realised_batch(trainer):
    """FIX 1. Abort BEFORE training unless one GPU at most and batch 64."""
    a = trainer.args
    realised = a.train_batch_size * a.gradient_accumulation_steps * a.world_size
    if a.n_gpu > 1 or realised != EXPECTED_TRAIN_BATCH:
        raise SystemExit(f"ABORT before training: n_gpu={a.n_gpu}, realised batch "
                         f"{realised} != {EXPECTED_TRAIN_BATCH}")
    return realised


def steps_per_epoch(history):
    st = [int(h["step"]) for h in history if "step" in h]
    return [st[0]] + [st[i + 1] - st[i] for i in range(len(st) - 1)] if st else []


def expected_steps(trainer, n_rows, micro, accum):
    n_batches = math.ceil(n_rows / (micro * max(1, trainer.args.n_gpu)))
    return sorted({n_batches // accum, math.ceil(n_batches / accum)})


def selection_check(history_metrics, reloaded_dev):
    """FIX 2. Reloaded snapshot vs what HF logged at the selected epoch."""
    h = history_metrics or {}
    keys = ("qwk", "acc19", "dist")
    logged = {k: h.get(f"eval_{k}") for k in keys}
    reloaded = {k: reloaded_dev.get(k) for k in keys}
    diff = {k: (abs(logged[k] - reloaded[k]) if logged[k] is not None and reloaded[k] is not None
                else None) for k in keys}
    exact = all(d is not None and d < 1e-9 for d in diff.values())
    return {"logged_at_selected_epoch": logged, "reloaded_snapshot_dev": reloaded,
            "abs_diff": diff, "exact": exact}


def provenance(trainer=None, realised=None):
    import transformers as _tf, datasets as _ds
    try:
        import accelerate as _acc
        acc_v = _acc.__version__
    except Exception:
        acc_v = None
    p = dict(family=SCRIPT_FAMILY, script=SCRIPT_VERSION, part=PART,
             utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
             python=platform.python_version(), torch=torch.__version__,
             transformers=_tf.__version__, datasets=_ds.__version__, accelerate=acc_v,
             cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
             cuda_device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
             gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
             dataset=CONFIG["dataset"], dataset_sha=DATASET_SHA, smoke=SMOKE)
    if trainer is not None:
        a = trainer.args
        p.update(n_gpu=a.n_gpu, per_device_train_batch=a.per_device_train_batch_size,
                 grad_accum=a.gradient_accumulation_steps, realised_train_batch=realised,
                 eval_batch=a.per_device_eval_batch_size, fp16=bool(a.fp16))
    return p


def _valid_v3_result(res):
    p = res.get("provenance") or {}
    return (p.get("family") == SCRIPT_FAMILY and not p.get("smoke", True)
            and (p.get("n_gpu") is not None and p.get("n_gpu") <= 1)
            and p.get("realised_train_batch") == EXPECTED_TRAIN_BATCH
            and "dev" in res and "test" in res)


def restore_previous_results():
    """FIX 3. Resume across Kaggle commits on THIS account: attach a previous
    version's output as an input. Only single-GPU, batch-64 v3 results of this
    part's arms are restored; v2 files are refused."""
    if not KAGGLE_INPUT.exists():
        print("  not running on Kaggle -- nothing to restore")
        return
    mine = {a["name"] for a in ARMS}
    took, refused = [], []
    for f in sorted(KAGGLE_INPUT.glob("**/v3/results/*.json")):
        dst = RESULTS / f.name
        if f.stem not in mine or dst.exists():
            continue
        try:
            res = json.loads(f.read_text())
        except Exception:
            refused.append((f, "unreadable"))
            continue
        if _valid_v3_result(res):
            dst.write_text(f.read_text())
            took.append(f)
        else:
            refused.append((f, "not a single-GPU, batch-64 v3 result"))
    for f in took:
        print(f"  restored {f.name}  <-  {f.parent}")
    for f, why in refused:
        print(f"  REFUSED  {f.name}  <-  {f.parent}  ({why})")
    if not took and not refused:
        print("  no previous v3 output under /kaggle/input -- starting fresh")


# ----------------------------------------------------------------- 11. arms
# Part A: CE, MSE, OLL15, EMD.  Part B: SOFT2-5.  Part C: CORAL, CORAL_HEADLR,
# SOFTQWK (the longest arms). One part per Kaggle account.
ALL_ARMS = [
    dict(part="A", name="CE",    head="cls", loss=loss_ce,   group="core", target="Table 6: QWK 80.0"),
    dict(part="A", name="MSE",   head="reg", loss=loss_mse,  group="core", target="Table 6: QWK 84.0 (Reg)"),
    dict(part="A", name="OLL15", head="cls", loss=loss_oll,  group="core", kw=dict(alpha=1.5), target="Table 6: QWK 82.8"),
    dict(part="A", name="EMD",   head="cls", loss=loss_emd,  group="core", target="Table 6: QWK 81.2"),
    dict(part="B", name="SOFT2", head="cls", loss=loss_soft, group="core", kw=dict(beta=2.0), target="Table 6: QWK 80.0"),
    dict(part="B", name="SOFT3", head="cls", loss=loss_soft, group="core", kw=dict(beta=3.0), target="Table 6: QWK 80.1"),
    dict(part="B", name="SOFT4", head="cls", loss=loss_soft, group="core", kw=dict(beta=4.0), target="Table 6: QWK 79.6"),
    dict(part="B", name="SOFT5", head="cls", loss=loss_soft, group="core", kw=dict(beta=5.0), target="Table 6: QWK 79.3"),
    dict(part="C", name="CORAL", head="coral", loss=loss_coral, group="core", target="ours (Cao et al. 2020)"),
    dict(part="C", name="CORAL_HEADLR", head="coral", loss=loss_coral, group="ablation", head_lr=1e-3,
         target="ablation: head lr 1e-3, encoder 5e-5"),
    dict(part="C", name="SOFTQWK", head="reg", loss=loss_softqwk, group="softqwk", kw=dict(sigma=1.0),
         two_phase=True, batch_statistic=True, target="ours (STBW 2025)"),
    dict(part="F", name="MV", head="cls", decode="exp", loss=loss_mv, group="core",
         kw=dict(l1=MV_LAMBDA1, l2=MV_LAMBDA2),
         target="Pan et al., CVPR 2018 (mean-variance loss); control CE"),
    dict(part="F", name="DLDL2", head="cls", decode="exp", loss=loss_dldl2, group="core",
         kw=dict(sigma=DLDL_SIGMA, lam=DLDL_LAMBDA),
         target="Gao et al., IJCAI 2018 (DLDL-v2); controls SOFT2, CE"),
    dict(part="F", name="CLM", head="clm", loss=loss_clm, group="core",
         target="McCullagh 1980; Vargas et al., Neurocomputing 2020 (deep CLM); control CORAL"),
    dict(part="G", name="OE", head="reg", loss=loss_oe, group="core", batch_statistic=True,
         features=True, kw=dict(weight=OE_WEIGHT),
         target="Zhang et al., ICLR 2023 (ordinal entropy); control MSE"),
    dict(part="G", name="LISTNET", head="reg", loss=loss_listnet, group="core", batch_statistic=True,
         kw=dict(lam=LISTNET_LAMBDA),
         target="Cao et al., ICML 2007 (ListNet); control MSE; siblings RANK, PLR"),
    dict(part="G", name="CCC", head="reg", loss=loss_ccc, group="core", batch_statistic=True,
         target="Lin 1989 (concordance correlation = QWK on integers); controls MSE, SOFTQWK"),
]
ARMS = [a for a in ALL_ARMS if a["part"] == PART]

PAPER_TABLE6 = {
    "CE":    dict(acc19=.566, acc19_pm1=.699, dist=1.14, qwk=.800),
    "EMD":   dict(acc19=.553, acc19_pm1=.703, dist=1.11, qwk=.812),
    "OLL15": dict(acc19=.473, acc19_pm1=.711, dist=1.13, qwk=.828),
    "MSE":   dict(acc19=.431, acc19_pm1=.731, dist=1.13, qwk=.840),
    "SOFT2": dict(acc19=.558, acc19_pm1=.698, dist=1.15, qwk=.800),
    "SOFT3": dict(acc19=.564, acc19_pm1=.699, dist=1.14, qwk=.801),
    "SOFT4": dict(acc19=.564, acc19_pm1=.699, dist=1.15, qwk=.796),
    "SOFT5": dict(acc19=.562, acc19_pm1=.695, dist=1.17, qwk=.793),
}


# ----------------------------------------------------------------- 12. one arm
def build_model_init(head, bias_init=None):
    ck = CONFIG["model_name"]
    if head == "cls":
        return lambda: AutoModelForSequenceClassification.from_pretrained(
            ck, num_labels=K, ignore_mismatched_sizes=True)
    if head == "reg":
        return lambda: AutoModelForSequenceClassification.from_pretrained(
            ck, num_labels=1, ignore_mismatched_sizes=True)
    if head == "coral":
        return lambda: CoralModel(ck, bias_init=bias_init)
    if head == "clm":
        return lambda: ClmModel(ck)
    raise ValueError(head)


def run_arm(arm, tok, enc, collator, bias_init):
    name = arm["name"]
    decode = {"cls": decode_cls, "reg": decode_reg, "coral": decode_coral,
              "exp": decode_exp, "clm": decode_clm}[arm.get("decode", arm["head"])]
    out_dir = CKPT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if arm.get("batch_statistic"):
        micro, accum, gckpt = CONFIG["true_batch"], 1, True
    else:
        micro, accum, gckpt = CONFIG["micro_batch"], CONFIG["grad_accum"], False

    epochs = 1 if SMOKE else CONFIG["epochs"]
    n_train = len(enc["train"])
    res = {"arm": name, "group": arm["group"], "target": arm.get("target"), "head": arm["head"],
           "micro_batch": micro, "grad_accum": accum, "gradient_checkpointing": gckpt,
           "epochs": epochs, "loss_is_batch_statistic": bool(arm.get("batch_statistic"))}

    def make_trainer(loss_fn, loss_kw, n_epochs, sub, model_init=None, model=None,
                     micro_bs=micro, accum_=accum, gckpt_=gckpt):
        sub_dir = out_dir / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
        snaps = BestSnapshots(sub_dir)
        cbs = [snaps]
        cutlog = None
        if arm["head"] == "coral":
            cutlog = CutpointLogger()
            cbs.append(cutlog)
        tr = (FeatArmTrainer if arm.get("features") else ArmTrainer)(
            args=make_args(sub_dir, n_epochs, micro_bs, accum_, grad_ckpt=gckpt_),
            model_init=model_init, model=model,
            train_dataset=enc["train"], eval_dataset=enc["dev"],
            data_collator=collator, compute_metrics=make_compute_metrics(decode),
            callbacks=cbs, loss_fn=loss_fn, loss_kwargs=loss_kw,
            head_lr=arm.get("head_lr"), **tok_kwarg(tok))
        realised = assert_realised_batch(tr)
        return tr, snaps, cutlog, realised

    if arm.get("two_phase"):
        p1_ep = epochs // 2 or 1
        p2_ep = epochs - p1_ep or 1
        print(f"  phase 1: MSE, {p1_ep} epochs, batch {CONFIG['micro_batch']}x{CONFIG['grad_accum']}")
        tr1, snaps1, _, realised1 = make_trainer(
            loss_mse, {}, p1_ep, "phase1", model_init=build_model_init("reg"),
            micro_bs=CONFIG["micro_batch"], accum_=CONFIG["grad_accum"], gckpt_=False)
        tr1.train()
        h1 = [h for h in tr1.state.log_history if "eval_loss" in h]
        res["phase1_history"] = h1
        res["phase1_selected_epoch"] = snaps1.loss_epoch
        res["phase1_realised_train_batch"] = realised1
        res["phase1_steps_per_epoch_observed"] = steps_per_epoch(h1)
        res["phase1_steps_per_epoch_expected"] = expected_steps(
            tr1, n_train, CONFIG["micro_batch"], CONFIG["grad_accum"])
        warm = torch.load(snaps1.loss_path, map_location="cpu")
        tr1.model.load_state_dict(warm)
        lg, gd = predict_logits(tr1, enc["dev"])
        res["phase1_dev"] = metrics_from_preds(gd, decode_reg(lg))
        res["phase1_selection_check"] = selection_check(snaps1.loss_metrics, res["phase1_dev"])
        del tr1
        gc.collect(); torch.cuda.empty_cache()

        print(f"  phase 2: SoftQWK, {p2_ep} epochs, true batch {micro} (D3), gradient checkpointing, "
              f"starting from the phase-1 snapshot (epoch {snaps1.loss_epoch})")
        m = build_model_init("reg")()
        m.load_state_dict(warm)
        del warm
        shutil.rmtree(out_dir / "phase1", ignore_errors=True)
        gc.collect()
        trainer, snaps, cutlog, realised = make_trainer(
            arm["loss"], arm.get("kw", {}), p2_ep, "phase2", model=m)
        trainer.train()
    else:
        trainer, snaps, cutlog, realised = make_trainer(
            arm["loss"], arm.get("kw", {}), epochs, "run",
            model_init=build_model_init(arm["head"], bias_init))
        trainer.train()

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    res["history"] = history
    res["selected_by"] = "lowest dev eval_loss over all epochs, no early stopping (train.py:481-483)"
    res["selected_epoch"] = snaps.loss_epoch
    res["realised_train_batch"] = realised
    res["steps_per_epoch_observed"] = steps_per_epoch(history)
    res["steps_per_epoch_expected"] = expected_steps(trainer, n_train, micro, accum)
    res["steps_per_epoch_ok"] = (bool(res["steps_per_epoch_observed"])
                                 and all(s in res["steps_per_epoch_expected"]
                                         for s in res["steps_per_epoch_observed"]))

    # FIX 2: the reported numbers are the selected epoch's own snapshot
    trainer.model.load_state_dict(torch.load(snaps.loss_path, map_location="cpu"))
    logits = {}
    for split in ("dev", "test"):
        lg, gd = predict_logits(trainer, enc[split])
        res[split] = metrics_from_preds(gd, decode(lg))
        logits[split] = (lg, gd)
    res["selection_check"] = selection_check(snaps.loss_metrics, res["dev"])

    if cutlog is not None:
        b = trainer.model.cutpoints.detach().float().cpu().numpy()
        res["cutpoint_history"] = cutlog.history
        res["cutpoints_selected"] = b.tolist()
        res["cutpoint_spread_selected"] = float(b.max() - b.min())
        res["cutpoint_max_move_from_init"] = float(np.abs(b - np.asarray(bias_init)).max())

    if arm["head"] == "coral":
        (dl, dg), (tl, tg) = logits["dev"], logits["test"]
        grid = np.arange(0.05, 0.96, 0.01)
        qwks = [metrics_from_preds(dg, decode_coral(dl, t))["qwk"] for t in grid]
        t_star = float(grid[int(np.argmax(qwks))])
        res["ablation_zai_threshold"] = {
            "note": "ZAI tune a single threshold on dev. Ablation only; its dev numbers leak.",
            "t": t_star,
            "dev": metrics_from_preds(dg, decode_coral(dl, t_star)),
            "test": metrics_from_preds(tg, decode_coral(tl, t_star)),
        }

    if snaps.qwk_path.exists():
        trainer.model.load_state_dict(torch.load(snaps.qwk_path, map_location="cpu"))
        abl = {"selected_by": "highest dev QWK (ZAI/STBW rule, NOT BAREC's); ablation, dev leaks",
               "selected_epoch": snaps.qwk_epoch}
        for split in ("dev", "test"):
            lg, gd = predict_logits(trainer, enc[split])
            abl[split] = metrics_from_preds(gd, decode(lg))
        abl["selection_check"] = selection_check(snaps.qwk_metrics, abl["dev"])
        res["ablation_qwk_selected"] = abl

    res["provenance"] = provenance(trainer, realised)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return res


# ----------------------------------------------------------------- 13. report
def report():
    rows = [json.loads(f.read_text()) for f in sorted(RESULTS.glob("*.json"))]
    order = [a["name"] for a in ALL_ARMS]
    rows.sort(key=lambda r: order.index(r["arm"]) if r["arm"] in order else 99)

    def line(r, split):
        m = r[split]
        return (f"{r['arm']:<14}{m['acc19']*100:>8.1f}{m['acc19_pm1']*100:>10.1f}"
                f"{m['dist']:>8.2f}{m['qwk']*100:>8.1f}{m['acc7']*100:>8.1f}"
                f"{m['acc5']*100:>7.1f}{m['acc3']*100:>7.1f}"
                f"{m['far_ge3']*100:>9.1f}{m['far_ge5']*100:>9.1f}")

    groups = [
        ("core", "CORE ARMS"),
        ("softqwk", "SOFTQWK -- different training procedure, NOT directly comparable to CORE"),
        ("ablation", "ABLATIONS -- excluded from the fair comparison"),
    ]
    for split in ("dev", "test"):
        print(f"\n=== PART {PART} -- BAREC {split.upper()} set (AraBERTv2 + D3Tok) ===")
        for gkey, glabel in groups:
            grows = [r for r in rows if r.get("group") == gkey and split in r]
            if not grows:
                continue
            print(f"\n  -- {glabel} --")
            print(f"  {'arm':<14}{'Acc19':>8}{'+-1Acc':>10}{'Dist':>8}{'QWK':>8}"
                  f"{'Acc7':>8}{'Acc5':>7}{'Acc3':>7}{'far>=3*':>9}{'far>=5*':>9}")
            for r in grows:
                print("  " + line(r, split))
        print("  * far>=3 / far>=5 are THIS THESIS'S metrics, not BAREC's.")

    print("\n=== run checks (FIX 1: one GPU, batch 64 | FIX 2: reported == selected epoch) ===")
    for r in rows:
        p = r.get("provenance", {})
        sc = r.get("selection_check", {})
        spe = r.get("steps_per_epoch_observed") or [None]
        print(f"  {r['arm']:<14} n_gpu={p.get('n_gpu')}  batch={r.get('realised_train_batch')}  "
              f"steps/epoch={spe[0]} (expected {r.get('steps_per_epoch_expected')}, "
              f"ok={r.get('steps_per_epoch_ok')})  reported==selected: {sc.get('exact')}")
        if "phase1_selection_check" in r:
            print(f"  {'':<14} phase 1: batch={r.get('phase1_realised_train_batch')}  "
                  f"reported==selected: {r['phase1_selection_check'].get('exact')}")

    print("\n=== reproduction gap on DEV, CORE arms only (ours - paper Table 6, QWK points) ===")
    for r in rows:
        if r.get("group") != "core":
            continue
        pt = PAPER_TABLE6.get(r["arm"])
        if pt and "dev" in r:
            g = (r["dev"]["qwk"] - pt["qwk"]) * 100
            print(f"  {r['arm']:<10} ours {r['dev']['qwk']*100:6.2f}   "
                  f"paper {pt['qwk']*100:6.2f}   gap {g:+6.2f}"
                  f"{'   <-- outside +-1.5' if abs(g) > 1.5 else ''}")

    for r in rows:
        if "cutpoint_spread_selected" in r:
            print(f"\n=== {r['arm']} ({r.get('group')}) cut-point spread: "
                  f"{r['cutpoint_spread_selected']:.2f}, max move from init "
                  f"{r['cutpoint_max_move_from_init']:.3f}")
        if "ablation_zai_threshold" in r:
            a = r["ablation_zai_threshold"]
            print(f"    ZAI dev-tuned threshold t={a['t']:.2f}: "
                  f"dev QWK {a['dev']['qwk']*100:.2f}, test QWK {a['test']['qwk']*100:.2f}")
        if "ablation_qwk_selected" in r:
            a = r["ablation_qwk_selected"]
            print(f"    {r['arm']} QWK-selected checkpoint (epoch {a['selected_epoch']}): "
                  f"dev QWK {a['dev']['qwk']*100:.2f}, test QWK {a['test']['qwk']*100:.2f}")
        if "phase1_dev" in r:
            print(f"    {r['arm']} phase-1 (MSE warm-up) dev QWK: {r['phase1_dev']['qwk']*100:.2f}")


# ----------------------------------------------------------------- 14. main
def main():
    global DATASET_SHA
    print("=" * 78)
    print(f"barec_exact_v3_part{PART}.py -- arms: {', '.join(a['name'] for a in ARMS)}")
    print(SCRIPT_VERSION)
    print("=" * 78)
    print("\nDECLARED DEFAULTS -- values that appear in NO paper, or that deviate from train.py:")
    for k, v, why in DECLARED_DEFAULTS:
        print(f"  - {k}: {v}\n      why: {why}")
    print(f"\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"visible GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}  "
          f"fp16: {CONFIG['fp16']}")

    print("\n[resume check -- across separate Kaggle commits, THIS account only]")
    restore_previous_results()

    print("\n[self-tests]")
    self_test_metrics()
    self_test_losses()

    set_seed(CONFIG["seed"])
    print("\n[data]")
    DATASET_SHA = dataset_sha()
    print(f"  dataset revision: {DATASET_SHA}")
    ds = load_barec()

    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    enc = ds.map(lambda b: tok(b["text"], truncation=True, max_length=CONFIG["max_length"]),
                 batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tok)
    bias_init = coral_bias_init(ds["train"]["labels"])
    print(f"  CORAL bias init: spread {bias_init.max()-bias_init.min():.2f}, "
          f"ordered={bool(np.all(np.diff(bias_init) <= 0))}")

    print(f"\n[arms -- PART {PART}]")
    arm_minutes = []
    stopped = False
    for arm in ARMS:
        f = RESULTS / f"{arm['name']}.json"
        if f.exists():
            print(f"  {arm['name']}: done, skipping")
            continue
        elapsed = time.time() - T_START
        est_min = 1.5 * max(arm_minutes) if arm_minutes else FIRST_ARM_ESTIMATE_MIN
        if elapsed + (est_min + SAFETY_MIN) * 60 > KAGGLE_LIMIT_S:
            print(f"\n  TIME GUARD (FIX 4): {elapsed/3600:.2f} h elapsed; {arm['name']} is estimated "
                  f"at {est_min:.0f} min and would not finish before the 12-hour limit. Stopping "
                  f"cleanly. Start a new version with this version's output attached as an input "
                  f"to resume.")
            stopped = True
            break
        print(f"\n--- {arm['name']}  (group={arm['group']}, {arm['target']})  "
              f"[{elapsed/3600:.2f} h elapsed]")
        try:
            res = run_arm(arm, tok, enc, collator, bias_init)
            f.write_text(json.dumps(res, indent=2))
            arm_minutes.append(res["minutes"])
            print(f"  {arm['name']}: dev QWK {res['dev']['qwk']*100:.2f}  "
                  f"test QWK {res['test']['qwk']*100:.2f}  ({res['minutes']} min)  "
                  f"batch {res['realised_train_batch']}  "
                  f"reported==selected: {res['selection_check']['exact']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            (RESULTS / f"{arm['name']}.FAILED.txt").write_text(repr(e))
            gc.collect(); torch.cuda.empty_cache()
            continue
        shutil.make_archive(str(BASE / f"results_backup_part{PART}"), "zip", RESULTS)

    report()
    shutil.make_archive(str(BASE / f"results_backup_part{PART}"), "zip", RESULTS)
    missing = [a["name"] for a in ARMS if not (RESULTS / f"{a['name']}.json").exists()]
    print(f"\nPART {PART} {'STOPPED BY TIME GUARD' if stopped else 'finished'}. "
          f"Results in {RESULTS} and {BASE}/results_backup_part{PART}.zip")
    print(f"Arms still missing in this part: {missing if missing else 'none'}")


if __name__ == "__main__":
    main()

# =============================================================================
#  barec_contrib_v1_partA.py   (barec_contrib_v1_partB.py and _partC.py are this
#                               same file with the single line PART = "B" / "C")
#
#  THE THESIS'S OWN CONTRIBUTION -- three objectives that no BAREC paper and no
#  arm of barec_exact_v3 uses, run on EXACTLY the v3 pipeline: BAREC corpus,
#  AraBERTv2 + D3Tok, 19 levels, seed 42, 6 epochs, batch 64, lr 5e-5,
#  lowest-dev-loss selection, the same metrics and the same run checks.
#
#  THE THREE NEW ARMS (one per part, one per Kaggle account)
#   Part A  CORN      Shi, Cao & Raschka 2023, Pattern Analysis and Applications
#                     26(3):941-955. Rank-consistent ordinal regression from
#                     conditional probabilities. ZAI 2025 (BAREC shared task,
#                     p. 267) names it as future work: "exploring alternative
#                     ordinal objectives (e.g., CORN)".
#                     Matched control: v3 CORAL. Both have K-1 = 18 binary tasks
#                     "y > level j"; CORN drops CORAL's single shared weight
#                     vector and trains task j only on the rows with y >= j.
#   Part B  MSE_RANK  MSE + pairwise margin ranking loss on the regression score.
#                     Pairwise ranking is the training objective of readability
#                     models in other languages: Trokhymovych, Sen & Gerlach
#                     2024 (arXiv 2406.01835; Wikipedia readability in 14
#                     languages; MarginRankingLoss, margin 0.5) and Lee &
#                     Vajjala 2022 (Findings of ACL; neural pairwise ranking,
#                     English / French / Spanish). Matched control: v3 MSE.
#   Part C  BMSE      Balanced MSE, Ren, Zhang, Yu & Liu, CVPR 2022, pp.
#                     7926-7935, in its BNI form, which is exact for a scale of
#                     19 discrete levels. Answers STBW 2025 (BAREC shared task),
#                     who find level-18/19 items predicted as 16 "due to the
#                     class imbalance" and plan "loss weighting techniques".
#                     Matched control: v3 MSE (same head built by the same call,
#                     so the same initial weights, and the same 16 x 4 batching).
#
#  WHAT IS DELIBERATELY NOT AN ARM
#   - CDW-CE (Polat et al. 2025): its formula -sum_i log(1 - p_i) |i - c|^a is
#     exactly v3's loss_oll, so v3's OLL15 already IS CDW-CE with a = 1.5.
#   - Everything except the head, the loss and the batch rule is copied from
#     barec_exact_v3 unchanged: data loading and its asserts, the 7/5/3 maps,
#     metrics, TrainingArguments, snapshot selection (FIX 2), the one-GPU and
#     batch-64 guards (FIX 1), the resume logic (FIX 3), the time guard (FIX 4).
#
#  HOW TO RUN (one part per Kaggle account, all three in parallel)
#   Part A: CORN      Part B: MSE_RANK      Part C: BMSE
#   Each arm should take about 30-40 min on one T4 (v3 arms took 25-30 min),
#   so a single account can also run the three files one after another.
#   - Accelerator: GPU (T4 x2 is fine -- this file uses exactly one GPU).
#   - Internet ON. Paste the whole file as the FIRST and ONLY cell.
#   - Save Version -> Save & Run All.
#   - Download results_backup_partA.zip, _partB.zip and _partC.zip.
#   - The report prints each new arm next to the v3 numbers (embedded below,
#     copied from the v3 result JSONs) and the difference to its control.
#
#  HARDWARE-FORCED DEVIATIONS -- the same three as v3
#   D1. Batch 64 as micro-batch 16 x accumulation 4 on ONE GPU for losses that
#       are a per-sample mean (BMSE). 857 optimiser steps per epoch.
#   D2. fp16 autocast with fp32 master weights; every loss upcasts to fp32.
#   D3. CORN and MSE_RANK are statistics of the whole batch: CORN divides by
#       the number of rows in all conditional subsets of the batch, and the
#       ranking term averages over pairs inside the batch. Accumulation is not
#       equivalent for them, so they run at a true per-device batch of 64 with
#       gradient checkpointing, exactly as SOFTQWK did in v3 (857 steps/epoch).
# =============================================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # FIX 1 -- must run before torch is imported

import sys, json, gc, time, math, shutil, inspect, subprocess, platform
from datetime import datetime, timezone
from pathlib import Path

T_START = time.time()

PART = "A"   # the ONLY line that differs between the Part A, B and C files

SCRIPT_FAMILY = "barec_contrib_v1"
SCRIPT_VERSION = f"{SCRIPT_FAMILY}_part{PART} (2026-09-28)"

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
from scipy.special import expit
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          TrainerCallback, set_seed)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error, f1_score

if torch.cuda.is_available() and torch.cuda.device_count() != 1:
    raise SystemExit(
        f"ABORT: {torch.cuda.device_count()} GPUs are visible. Run this file as the FIRST cell "
        f"of a fresh session so CUDA_VISIBLE_DEVICES=0 takes effect. Two GPUs double the "
        f"batch to 128 -- the v2 defect.")

# ----------------------------------------------------------------- 1. config (identical to v3)
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

# The only new hyper-parameters in this file (see DECLARED_DEFAULTS)
RANK_MARGIN   = 0.5       # Trokhymovych et al. 2024
RANK_LAMBDA   = 1.0       # ours: MSE and ranking terms with equal weight
BMSE_SIGMA0   = 1.0       # ours: one level
BMSE_SIGMA_LR = 1e-2      # the learning rate the Balanced MSE authors give the noise scale

EXPECTED_TRAIN_BATCH = 64
KAGGLE_LIMIT_S = 12 * 3600          # Kaggle kills a commit at 43,200 s (observed on v2)
FIRST_ARM_ESTIMATE_MIN = 150        # used only until one arm has finished in this session
SAFETY_MIN = 25                     # report, zip and notebook-to-HTML conversion
KAGGLE_INPUT = Path("/kaggle/input")

# v3 results of the same pipeline (barec_exact_v3 result JSONs, one T4, batch 64,
# transformers 5.0.0, torch 2.10.0+cu128). Used ONLY for the comparison table.
REFERENCE_FAMILY = "barec_exact_v3"
REFERENCE_TRANSFORMERS = "5.0.0"
V3_REFERENCE = {
    "CE": dict(
        dev=dict(qwk=0.7981, acc19=0.5484, acc19_pm1=0.7047, dist=1.1629, acc7=0.6516, acc5=0.6996, acc3=0.7676, macro_f1=0.4545, far_ge3=0.1560, far_ge5=0.0669),
        test=dict(qwk=0.8048, acc19=0.5472, acc19_pm1=0.7029, dist=1.1281, acc7=0.6436, acc5=0.6871, acc3=0.7481, macro_f1=0.4686, far_ge3=0.1506, far_ge5=0.0630),
    ),
    "MSE": dict(
        dev=dict(qwk=0.8288, acc19=0.3554, acc19_pm1=0.7115, dist=1.2120, acc7=0.5702, acc5=0.6443, acc3=0.7334, macro_f1=0.2213, far_ge3=0.1343, far_ge5=0.0367),
        test=dict(qwk=0.8364, acc19=0.3420, acc19_pm1=0.7170, dist=1.1860, acc7=0.5624, acc5=0.6305, acc3=0.7131, macro_f1=0.2361, far_ge3=0.1341, far_ge5=0.0288),
    ),
    "OLL15": dict(
        dev=dict(qwk=0.8291, acc19=0.4621, acc19_pm1=0.7115, dist=1.1302, acc7=0.6155, acc5=0.6762, acc3=0.7633, macro_f1=0.2764, far_ge3=0.1345, far_ge5=0.0424),
        test=dict(qwk=0.8408, acc19=0.4628, acc19_pm1=0.7208, dist=1.0743, acc7=0.6069, acc5=0.6609, acc3=0.7380, macro_f1=0.2901, far_ge3=0.1252, far_ge5=0.0360),
    ),
    "EMD": dict(
        dev=dict(qwk=0.8129, acc19=0.5427, acc19_pm1=0.7010, dist=1.1252, acc7=0.6501, acc5=0.7010, acc3=0.7663, macro_f1=0.3660, far_ge3=0.1412, far_ge5=0.0584),
        test=dict(qwk=0.8270, acc19=0.5358, acc19_pm1=0.6939, dist=1.0804, acc7=0.6400, acc5=0.6897, acc3=0.7487, macro_f1=0.3818, far_ge3=0.1333, far_ge5=0.0523),
    ),
    "CORAL": dict(
        dev=dict(qwk=0.8290, acc19=0.3415, acc19_pm1=0.6684, dist=1.2893, acc7=0.5540, acc5=0.6360, acc3=0.7300, macro_f1=0.2456, far_ge3=0.1286, far_ge5=0.0432),
        test=dict(qwk=0.8433, acc19=0.3415, acc19_pm1=0.6670, dist=1.2324, acc7=0.5409, acc5=0.6098, acc3=0.6921, macro_f1=0.2638, far_ge3=0.1206, far_ge5=0.0329),
    ),
    "SOFTQWK": dict(
        dev=dict(qwk=0.8324, acc19=0.3523, acc19_pm1=0.7233, dist=1.2506, acc7=0.5829, acc5=0.6576, acc3=0.7531, macro_f1=0.2803, far_ge3=0.1383, far_ge5=0.0473),
        test=dict(qwk=0.8431, acc19=0.3596, acc19_pm1=0.7221, dist=1.1963, acc7=0.5733, acc5=0.6436, acc3=0.7280, macro_f1=0.2896, far_ge3=0.1307, far_ge5=0.0419),
    ),
}

DECLARED_DEFAULTS = [
    ("D1 micro-batch x accumulation", "16 x 4 on one GPU (BMSE)",
     "hardware: a T4 has 16 GB. 857 steps/epoch as with batch 64; the last step of each epoch has 61 rows"),
    ("D2 precision", "fp16 autocast, fp32 master weights, losses in fp32",
     "hardware: fits the Kaggle time limit. train.py runs fp32"),
    ("D3 batch-statistic arms", "CORN and MSE_RANK: true batch 64, gradient checkpointing",
     "their loss is a statistic of the whole batch, so accumulation is not equivalent; v3's SOFTQWK rule"),
    ("CORN head", "AutoModelForSequenceClassification with 18 outputs: pooler -> dropout -> Linear(768, 18)",
     "Shi et al. 2023: K-1 independent binary outputs, no weight sharing (the one head change from CORAL)"),
    ("CORN bias init", "b_j = logit P(y > j | y >= j) from TRAIN labels only",
     "ours; the conditional analogue of v3 CORAL's prior init, so CORN and CORAL differ in head and loss only"),
    ("CORN loss reduction", "sum over tasks and subset rows / number of subset rows in the batch",
     "Shi et al. 2023 Eq. 5, identical to coral_pytorch.losses.corn_loss"),
    ("CORN decode rule", "count of cumulative products prod_{i<=j} sigmoid(z_i) > 0.5",
     "Shi et al. 2023 chain rule; coral_pytorch corn_label_from_logits"),
    ("CORN threshold ablation", "0.05 : 0.01 : 0.95 on the cumulative products, tuned on dev",
     "mirror of v3's ZAI-threshold ablation for CORAL; ablation only, its dev numbers leak"),
    ("MSE_RANK margin", RANK_MARGIN,
     "Trokhymovych et al. 2024 (MarginRankingLoss, margin 0.5)"),
    ("MSE_RANK weight lambda", RANK_LAMBDA,
     "ours; loss = MSE + lambda * ranking, equal weight, no sweep"),
    ("MSE_RANK pairs", "every ordered pair (i, j) in the batch with y_i > y_j; mean hinge over pairs",
     "equal-level pairs carry no order; a batch with a single level adds 0"),
    ("BMSE form", "BNI: exact sum over the 19 levels, prior = TRAIN label frequencies",
     "Ren et al. 2022; with discrete levels the integral is a sum, and the loss is per-sample, so D1 stays exact"),
    ("BMSE noise sigma", f"learnable, init {BMSE_SIGMA0}, lr {BMSE_SIGMA_LR}, same linear decay",
     "Ren et al.: sigma learnable; the init is ours (one level); lr as in the authors' usage example"),
    ("BMSE loss scale", "loss x (2 sigma^2).detach()",
     "the authors' scale restore; keeps the gradient on the prediction at MSE scale"),
    ("BMSE sigma floor", "sigma^2 >= 1e-6", "ours; a guard only"),
    ("Regression decode", "rint (half to even), then clip to [0, 18]",
     "BAREC 5.4 says only 'round'; used by MSE_RANK and BMSE, as by v3 MSE"),
    ("Dev-loss selection", "lowest dev loss of the arm's own objective, strict '<'",
     "BAREC's rule. BMSE's dev loss uses that epoch's sigma; batch-statistic losses use eval batches of 16"),
    ("QWK-selected checkpoint", "highest dev QWK",
     "ZAI / STBW rule; ablation only; its dev numbers leak by construction"),
    ("far>=3 / far>=5 metrics and the error histogram", "share of |pred - gold| at each distance",
     "OURS, not a BAREC metric. Must be labelled as the thesis's own everywhere"),
    ("v3 reference numbers", "copied from the barec_exact_v3 result JSONs",
     "comparison table only; a different transformers version on Kaggle can move numbers slightly"),
    ("Kaggle split", "Parts A, B and C on three accounts, plus a 12-hour time guard",
     "OURS. Scheduling only; no arm's hyper-parameters change"),
]

SMOKE = False          # True -> 2000 train rows, 1 epoch, 12 dev/test rows per level. Plumbing only.

BASE = (Path("/kaggle/working") / "contrib_v1") if Path("/kaggle").exists() else Path("./barec_out_contrib_v1")
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


def _collapse(y0, table):
    return np.array([table[int(v) + 1] for v in y0])


# ----------------------------------------------------------------- 3. metrics (identical to v3)
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


def error_histogram(gold0, pred0):
    """Share of predictions at distance |pred - gold| = 0, 1, 2, 3, 4 and >= 5 levels.
    THIS THESIS'S OWN analysis: where an objective's gains come from."""
    d = np.abs(np.asarray(gold0).astype(int) - np.asarray(pred0).astype(int))
    return {**{str(k): float((d == k).mean()) for k in range(5)}, "5+": float((d >= 5).mean())}


# ----------------------------------------------------------------- 4. decoders
def decode_reg(logits):
    return np.clip(np.rint(np.asarray(logits).squeeze(-1)), 0, K - 1).astype(int)


def decode_corn(logits, t=0.5):
    """P(y > r_j) = prod_{i<=j} f_i(x) (Shi et al. 2023); the level is the number
    of these that exceed t (coral_pytorch corn_label_from_logits, t = 0.5).
    The products never increase in j, so the decode is rank-consistent."""
    return (np.cumprod(expit(np.asarray(logits, dtype=np.float64)), axis=1) > t).sum(axis=1).astype(int)


# ----------------------------------------------------------------- 5. losses
def _fp32(x):
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def _levels(like):
    return torch.arange(K, device=like.device, dtype=like.dtype)


def _unwrap(model):
    return getattr(model, "module", model)


def loss_mse(logits, labels, **kw):
    """v3's MSE arm (BAREC 5.4). Used here inside MSE_RANK and by the self-test."""
    z = _fp32(logits).squeeze(-1)
    return F.mse_loss(z, labels.to(z.dtype))


def loss_corn(logits, labels, **kw):
    """CORN, Shi, Cao & Raschka 2023, Eq. 5. Output j (0-indexed) models
    f_j = P(y > j | y >= j) and is trained only on the conditional subset
    S_j = {rows with y >= j} (S_0 = every row); the sum of the binary log-losses
    is divided by the total number of subset rows in the batch."""
    z = _fp32(logits)                                               # (n, K-1)
    task = torch.arange(K - 1, device=z.device).unsqueeze(0)
    y = labels.unsqueeze(1)
    in_subset = (y >= task).to(z.dtype)
    target = (y > task).to(z.dtype)
    ls = F.logsigmoid(z)                                            # log f;  log(1 - f) = ls - z
    ll = ls * target + (ls - z) * (1.0 - target)
    return -(ll * in_subset).sum() / in_subset.sum()                # S_0 holds every row, so > 0


def loss_mse_rank(logits, labels, margin=RANK_MARGIN, lam=RANK_LAMBDA, **kw):
    """MSE + lam * mean over batch pairs (i, j) with y_i > y_j of
    max(0, margin - (s_i - s_j)) -- torch's MarginRankingLoss with target +1,
    the pairwise readability objective of Trokhymovych et al. 2024."""
    s = _fp32(logits).squeeze(-1)
    y = labels.to(s.dtype)
    mse = F.mse_loss(s, y)
    harder = (y.unsqueeze(1) > y.unsqueeze(0)).to(s.dtype)
    hinge = F.relu(margin - (s.unsqueeze(1) - s.unsqueeze(0)))
    rank = (hinge * harder).sum() / harder.sum().clamp(min=1.0)
    return mse + lam * rank


def loss_bmse(logits, labels, model=None, log_prior=None, **kw):
    """Balanced MSE (Ren et al. 2022), BNI form, exact for 19 discrete levels:
        L = -log N(y; s, sigma^2) + log sum_k pi_k N(k; s, sigma^2)
          = (s - y)^2 / (2 sigma^2) + logsumexp_k[log pi_k - (s - k)^2 / (2 sigma^2)]
    (the Gaussian constants cancel). pi = TRAIN label frequencies; sigma is the
    model's learnable noise_sigma. Scaled by (2 sigma^2).detach() as the authors do."""
    s = _fp32(logits)                                               # (n, 1)
    sigma = _unwrap(model).noise_sigma.to(s.dtype)
    var = sigma.pow(2).clamp(min=1e-6)
    y = labels.to(s.dtype).unsqueeze(1)
    lp = torch.as_tensor(log_prior, dtype=s.dtype, device=s.device).unsqueeze(0)
    mse_term = (s - y).pow(2) / (2 * var)
    balance = torch.logsumexp(lp - (s - _levels(s).unsqueeze(0)).pow(2) / (2 * var), dim=1, keepdim=True)
    return ((mse_term + balance) * (2 * var).detach()).mean()


# ----------------------------------------------------------------- 6. self-tests
def self_test_losses():
    torch.manual_seed(0)
    n = 37
    labels = torch.randint(0, K, (n,))
    low = torch.randint(0, 4, (n,))            # every deep CORN subset is empty
    checks = []

    # CORN vs coral_pytorch.losses.corn_loss (Raschka research group), transcribed literally
    def corn_literal(logits, y_train, num_classes):
        sets = []
        for i in range(num_classes - 1):
            label_mask = y_train > i - 1
            label_tensor = (y_train[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))
        num_examples = 0
        losses = 0.
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]
            if len(train_labels) < 1:
                continue
            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]
            loss = -torch.sum(F.logsigmoid(pred) * train_labels
                              + (F.logsigmoid(pred) - pred) * (1 - train_labels))
            losses += loss
        return losses / num_examples

    z = torch.randn(n, K - 1, dtype=torch.float64) * 2
    checks.append(("CORN", float(corn_literal(z, labels, K)), float(loss_corn(z, labels))))
    checks.append(("CORN low labels", float(corn_literal(z, low, K)), float(loss_corn(z, low))))

    # CORN vs Eq. 5 with python floats
    num, den = 0.0, 0
    for j in range(K - 1):
        for i in range(n):
            yi = int(labels[i])
            if yi >= j:
                f = 1.0 / (1.0 + math.exp(-float(z[i, j])))
                num += math.log(f) if yi > j else math.log(1.0 - f)
                den += 1
    checks.append(("CORN Eq.5 floats", -num / den, float(loss_corn(z, labels))))

    # CORN decode vs coral_pytorch corn_label_from_logits
    lit = (torch.cumprod(torch.sigmoid(z), dim=1) > 0.5).sum(dim=1).numpy()
    checks.append(("CORN decode (#diff)", 0.0, float((decode_corn(z.numpy()) != lit).sum())))
    perfect = torch.full((n, K - 1), -20.0, dtype=torch.float64)
    for i in range(n):
        perfect[i, :int(labels[i])] = 20.0
    checks.append(("CORN decode exact (#diff)", 0.0,
                   float((decode_corn(perfect.numpy()) != labels.numpy()).sum())))

    # MSE_RANK vs a double loop, and vs torch.nn.MarginRankingLoss on the same pairs
    s = labels.to(torch.float64).unsqueeze(1) + torch.randn(n, 1, dtype=torch.float64) * 2
    mse = sum((float(s[i, 0]) - int(labels[i])) ** 2 for i in range(n)) / n
    ii, jj, tot = [], [], 0.0
    for i in range(n):
        for j in range(n):
            if int(labels[i]) > int(labels[j]):
                tot += max(0.0, RANK_MARGIN - (float(s[i, 0]) - float(s[j, 0])))
                ii.append(i); jj.append(j)
    checks.append(("MSE_RANK loop", mse + RANK_LAMBDA * tot / len(ii), float(loss_mse_rank(s, labels))))
    mrl = nn.MarginRankingLoss(margin=RANK_MARGIN)(s[ii, 0], s[jj, 0], torch.ones(len(ii), dtype=torch.float64))
    checks.append(("MSE_RANK torch MRL", float(loss_mse(s, labels) + RANK_LAMBDA * mrl),
                   float(loss_mse_rank(s, labels))))
    same = torch.full((n,), 7)
    checks.append(("MSE_RANK one level", float(loss_mse(s, same)), float(loss_mse_rank(s, same))))

    # BMSE vs Ren et al.'s definition with full Gaussian densities, python floats
    class _Holder(nn.Module):
        def __init__(self, v):
            super().__init__()
            self.noise_sigma = nn.Parameter(torch.tensor(v, dtype=torch.float64))

    prior = np.random.default_rng(0).dirichlet(np.ones(K))
    logp = np.log(prior)

    def normal(x, m, v):
        return math.exp(-(x - m) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)

    for sig in (0.7, 1.0, 2.5):
        v = sig ** 2
        lit = float(np.mean([
            (-math.log(normal(int(labels[i]), float(s[i, 0]), v))
             + math.log(sum(prior[k] * normal(k, float(s[i, 0]), v) for k in range(K)))) * 2 * v
            for i in range(n)]))
        checks.append((f"BMSE sigma={sig}", lit,
                       float(loss_bmse(s, labels, model=_Holder(sig), log_prior=logp).detach())))

    h = _Holder(1.3)
    sg = s.clone().requires_grad_(True)
    loss_bmse(sg, labels, model=h, log_prior=logp).backward()
    g_ok = (h.noise_sigma.grad is not None and bool(torch.isfinite(h.noise_sigma.grad))
            and float(h.noise_sigma.grad) != 0.0 and bool(torch.isfinite(sg.grad).all()))
    checks.append(("BMSE sigma gradient (bad)", 0.0, 0.0 if g_ok else 1.0))

    print("  loss self-test: CORN vs coral_pytorch's corn_loss and Eq. 5; MSE_RANK vs a loop "
          "and torch MarginRankingLoss; BMSE vs Ren et al.'s Gaussian definition")
    bad = []
    for name, a, b in checks:
        diff = abs(a - b)
        flag = "ok" if diff < 1e-9 else "FAIL"
        if diff >= 1e-9:
            bad.append(name)
        print(f"    {name:<26} {a:>18.12f}  {b:>18.12f}  {diff:.2e}  {flag}")
    assert not bad, f"loss mismatch: {bad}"


def self_test_metrics():
    rng = np.random.default_rng(1)
    gold = rng.integers(0, K, 500)
    pred = np.clip(gold + rng.integers(-4, 5, 500), 0, K - 1)
    m = metrics_from_preds(gold, pred)
    assert abs(m["dist"] - np.abs(gold - pred).mean()) < 1e-12
    assert abs(m["acc19_pm1"] - (np.abs(gold - pred) <= 1).mean()) < 1e-12
    assert abs(m["far_ge3"] - (np.abs(gold - pred) >= 3).mean()) < 1e-12
    h = error_histogram(gold, pred)
    assert abs(sum(h.values()) - 1.0) < 1e-12 and abs(h["0"] - m["acc19"]) < 1e-12
    assert abs(h["3"] + h["4"] + h["5+"] - m["far_ge3"]) < 1e-12 and abs(h["5+"] - m["far_ge5"]) < 1e-12
    perfect = metrics_from_preds(gold, gold)
    assert abs(perfect["qwk"] - 1.0) < 1e-12 and abs(perfect["acc3"] - 1.0) < 1e-12
    assert set(BAREC_7.values()) == set(range(1, 8))
    assert set(BAREC_5.values()) == set(range(1, 6))
    assert set(BAREC_3.values()) == set(range(1, 4))
    print("  metric self-test ok")


# ----------------------------------------------------------------- 7. data (identical to v3)
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


def train_label_stats(train_labels):
    """Everything this file derives from the TRAIN labels, and nothing else:
    CORN's conditional bias init and BMSE's label prior."""
    y = np.asarray(list(train_labels)).astype(int)
    counts = np.bincount(y, minlength=K).astype(np.float64)
    prior = counts / counts.sum()
    cond = np.array([(y > k).sum() / max((y >= k).sum(), 1) for k in range(K - 1)])
    cond = np.clip(cond, 1e-4, 1 - 1e-4)
    return dict(counts=counts, prior=prior,
                log_prior=np.log(np.clip(prior, 1e-12, None)),
                corn_bias=np.log(cond / (1 - cond)).astype(np.float32))


# ----------------------------------------------------------------- 8. model heads
class BalancedRegModel(nn.Module):
    """v3's MSE head -- AutoModelForSequenceClassification(num_labels=1), built by
    the SAME call, so under seed 42 it starts from the same weights as v3 MSE --
    plus the one learnable noise scale of Balanced MSE (no RNG is drawn for it)."""

    def __init__(self, checkpoint, init_sigma):
        super().__init__()
        self.net = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=1, ignore_mismatched_sizes=True)
        self.noise_sigma = nn.Parameter(torch.tensor(float(init_sigma)))
        self.config = self.net.config

    def gradient_checkpointing_enable(self, **kw):
        self.net.gradient_checkpointing_enable(**kw)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kw):
        out = self.net(input_ids=input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids)
        return SequenceClassifierOutput(logits=out.logits)


def build_model_init(head, stats):
    ck = CONFIG["model_name"]
    if head == "reg":
        return lambda: AutoModelForSequenceClassification.from_pretrained(
            ck, num_labels=1, ignore_mismatched_sizes=True)
    if head == "corn":
        def init():
            m = AutoModelForSequenceClassification.from_pretrained(
                ck, num_labels=K - 1, ignore_mismatched_sizes=True)
            with torch.no_grad():
                m.classifier.bias.copy_(torch.as_tensor(stats["corn_bias"], dtype=m.classifier.bias.dtype))
            return m
        return init
    if head == "bmse":
        return lambda: BalancedRegModel(ck, BMSE_SIGMA0)
    raise ValueError(head)


# ----------------------------------------------------------------- 9. trainer and callbacks
class ArmTrainer(Trainer):
    """v3's ArmTrainer. Two changes: the loss also receives the model (BMSE reads
    its noise_sigma; every other loss ignores it), and param_lrs gives named
    top-level parameters their own learning rate (BMSE's noise_sigma)."""

    def __init__(self, *a, loss_fn=None, loss_kwargs=None, param_lrs=None, **kw):
        super().__init__(*a, **kw)
        self.loss_fn = loss_fn
        self.loss_kwargs = loss_kwargs or {}
        self.param_lrs = param_lrs or {}
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels, model=model, **self.loss_kwargs)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self, *a, **kw):
        if self.optimizer is not None or not self.param_lrs:
            return super().create_optimizer(*a, **kw)
        own = {k: [] for k in self.param_lrs}
        rest = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            top = n.split(".")[0]
            (own[top] if top in own else rest).append(p)
        missing = [k for k, v in own.items() if not v]
        if missing:
            raise SystemExit(f"ABORT: the model has no parameter named {missing}")
        cls_, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        kwargs.pop("lr", None)
        groups = [{"params": rest, "lr": self.args.learning_rate, "weight_decay": self.args.weight_decay}]
        groups += [{"params": v, "lr": self.param_lrs[k], "weight_decay": 0.0} for k, v in own.items()]
        self.optimizer = cls_(groups, lr=self.args.learning_rate, **kwargs)
        return self.optimizer


class BestSnapshots(TrainerCallback):
    """FIX 2 (v3). torch.save snapshots of the lowest-dev-loss epoch (BAREC's
    rule, strict '<' exactly like HF's load_best_model_at_end) and of the
    highest-dev-QWK epoch (ablation)."""

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


class SigmaLogger(TrainerCallback):
    """BMSE: the learned noise scale at every evaluation."""

    def __init__(self):
        self.history = []

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        if model is None or not hasattr(model, "noise_sigma"):
            return
        s = abs(float(model.noise_sigma.detach().float().cpu()))
        ep = float(state.epoch or 0)
        self.history.append({"epoch": ep, "noise_sigma": s})
        if metrics is not None:
            metrics["eval_noise_sigma"] = s
        print(f"    [BMSE] epoch {ep:.0f}  noise sigma {s:.4f}")


# ----------------------------------------------------------------- 10. plumbing (identical to v3)
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
             reference_family=REFERENCE_FAMILY, reference_transformers=REFERENCE_TRANSFORMERS,
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


def _valid_result(res):
    p = res.get("provenance") or {}
    return (p.get("family") == SCRIPT_FAMILY and not p.get("smoke", True)
            and (p.get("n_gpu") is not None and p.get("n_gpu") <= 1)
            and p.get("realised_train_batch") == EXPECTED_TRAIN_BATCH
            and "dev" in res and "test" in res)


def restore_previous_results():
    """FIX 3 (v3). Resume across Kaggle commits on THIS account: attach a previous
    version's output as an input. Only single-GPU, batch-64 results of this
    file family and this part's arms are restored."""
    if not KAGGLE_INPUT.exists():
        print("  not running on Kaggle -- nothing to restore")
        return
    mine = {a["name"] for a in ARMS}
    took, refused = [], []
    for f in sorted(KAGGLE_INPUT.glob("**/contrib_v1/results/*.json")):
        dst = RESULTS / f.name
        if f.stem not in mine or dst.exists():
            continue
        try:
            res = json.loads(f.read_text())
        except Exception:
            refused.append((f, "unreadable"))
            continue
        if _valid_result(res):
            dst.write_text(f.read_text())
            took.append(f)
        else:
            refused.append((f, f"not a single-GPU, batch-64 {SCRIPT_FAMILY} result"))
    for f in took:
        print(f"  restored {f.name}  <-  {f.parent}")
    for f, why in refused:
        print(f"  REFUSED  {f.name}  <-  {f.parent}  ({why})")
    if not took and not refused:
        print(f"  no previous {SCRIPT_FAMILY} output under /kaggle/input -- starting fresh")


# ----------------------------------------------------------------- 11. arms
# Part A: CORN.  Part B: MSE_RANK.  Part C: BMSE.  One part per Kaggle account.
ALL_ARMS = [
    dict(part="A", name="CORN", head="corn", loss=loss_corn, group="new", control="CORAL",
         batch_statistic=True,
         target="new: CORN (Shi, Cao & Raschka 2023); ZAI 2025 future work"),
    dict(part="B", name="MSE_RANK", head="reg", loss=loss_mse_rank, group="new", control="MSE",
         batch_statistic=True, kw=dict(margin=RANK_MARGIN, lam=RANK_LAMBDA),
         target="new: MSE + pairwise margin ranking (Trokhymovych et al. 2024; Lee & Vajjala 2022)"),
    dict(part="C", name="BMSE", head="bmse", loss=loss_bmse, group="new", control="MSE",
         param_lrs={"noise_sigma": BMSE_SIGMA_LR},
         target="new: Balanced MSE, BNI (Ren et al. 2022); STBW 2025 future work"),
]
ARMS = [a for a in ALL_ARMS if a["part"] == PART]


# ----------------------------------------------------------------- 12. one arm
def run_arm(arm, tok, enc, collator, stats):
    name = arm["name"]
    decode = {"reg": decode_reg, "bmse": decode_reg, "corn": decode_corn}[arm["head"]]
    out_dir = CKPT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if arm.get("batch_statistic"):
        micro, accum, gckpt = CONFIG["true_batch"], 1, True          # D3
    else:
        micro, accum, gckpt = CONFIG["micro_batch"], CONFIG["grad_accum"], False

    epochs = 1 if SMOKE else CONFIG["epochs"]
    n_train = len(enc["train"])
    loss_kw = dict(arm.get("kw", {}))
    if arm["head"] == "bmse":
        loss_kw["log_prior"] = stats["log_prior"]
    res = {"arm": name, "group": arm["group"], "target": arm.get("target"), "head": arm["head"],
           "control": arm.get("control"), "loss_hyperparameters": dict(arm.get("kw", {})),
           "param_lrs": arm.get("param_lrs"),
           "micro_batch": micro, "grad_accum": accum, "gradient_checkpointing": gckpt,
           "epochs": epochs, "loss_is_batch_statistic": bool(arm.get("batch_statistic"))}
    if arm["head"] == "bmse":
        res["train_label_prior"] = stats["prior"].tolist()
        res["noise_sigma_init"] = BMSE_SIGMA0
    if arm["head"] == "corn":
        res["corn_bias_init"] = stats["corn_bias"].tolist()

    sub_dir = out_dir / "run"
    sub_dir.mkdir(parents=True, exist_ok=True)
    snaps = BestSnapshots(sub_dir)
    cbs = [snaps]
    siglog = SigmaLogger() if arm["head"] == "bmse" else None
    if siglog is not None:
        cbs.append(siglog)
    trainer = ArmTrainer(
        args=make_args(sub_dir, epochs, micro, accum, grad_ckpt=gckpt),
        model_init=build_model_init(arm["head"], stats),
        train_dataset=enc["train"], eval_dataset=enc["dev"],
        data_collator=collator, compute_metrics=make_compute_metrics(decode),
        callbacks=cbs, loss_fn=arm["loss"], loss_kwargs=loss_kw,
        param_lrs=arm.get("param_lrs"), **tok_kwarg(tok))
    realised = assert_realised_batch(trainer)
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
        pred = decode(lg)
        res[split] = metrics_from_preds(gd, pred)
        res[f"{split}_error_hist"] = error_histogram(gd, pred)
        res[f"{split}_gold"] = gd.tolist()
        res[f"{split}_pred"] = pred.tolist()
        logits[split] = (lg, gd)
    res["selection_check"] = selection_check(snaps.loss_metrics, res["dev"])

    if siglog is not None:
        res["noise_sigma_history"] = siglog.history
        res["noise_sigma_selected"] = abs(float(trainer.model.noise_sigma.detach().float().cpu()))

    if arm["head"] == "corn":
        (dl, dg), (tl, tg) = logits["dev"], logits["test"]
        grid = np.arange(0.05, 0.96, 0.01)
        qwks = [metrics_from_preds(dg, decode_corn(dl, t))["qwk"] for t in grid]
        t_star = float(grid[int(np.argmax(qwks))])
        res["ablation_dev_threshold"] = {
            "note": "ZAI-style single threshold tuned on dev. Ablation only; its dev numbers leak.",
            "t": t_star,
            "dev": metrics_from_preds(dg, decode_corn(dl, t_star)),
            "test": metrics_from_preds(tg, decode_corn(tl, t_star)),
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(out_dir, ignore_errors=True)
    return res


# ----------------------------------------------------------------- 13. report
def report():
    rows = [json.loads(f.read_text()) for f in sorted(RESULTS.glob("*.json"))]
    order = [a["name"] for a in ALL_ARMS]
    rows.sort(key=lambda r: order.index(r["arm"]) if r["arm"] in order else 99)

    def line(name, m):
        return (f"{name:<16}{m['acc19']*100:>8.1f}{m['acc19_pm1']*100:>10.1f}"
                f"{m['dist']:>8.2f}{m['qwk']*100:>8.1f}{m['acc7']*100:>8.1f}"
                f"{m['acc5']*100:>7.1f}{m['acc3']*100:>7.1f}"
                f"{m['far_ge3']*100:>9.1f}{m['far_ge5']*100:>9.1f}")

    for split in ("dev", "test"):
        print(f"\n=== PART {PART} -- BAREC {split.upper()} set (AraBERTv2 + D3Tok) ===")
        print(f"  {'arm':<16}{'Acc19':>8}{'+-1Acc':>10}{'Dist':>8}{'QWK':>8}"
              f"{'Acc7':>8}{'Acc5':>7}{'Acc3':>7}{'far>=3*':>9}{'far>=5*':>9}")
        print("  -- NEW ARMS (this file) --")
        for r in rows:
            if split in r:
                print("  " + line(r["arm"], r[split]))
        print(f"  -- REFERENCE: {REFERENCE_FAMILY} arms, same pipeline --")
        for name, ref in V3_REFERENCE.items():
            print("  " + line(f"{name} (v3)", ref[split]))
        print("  * far>=3 / far>=5 are THIS THESIS'S metrics, not BAREC's.")

    print("\n=== new arm minus its matched v3 control (points; single seed) ===")
    for r in rows:
        c = V3_REFERENCE.get(r.get("control"))
        if not c:
            continue
        for split in ("dev", "test"):
            n, o = r[split], c[split]
            print(f"  {r['arm']:<10} vs {r['control']:<6} {split:<5}"
                  f"  QWK {(n['qwk']-o['qwk'])*100:+6.2f}"
                  f"  Acc19 {(n['acc19']-o['acc19'])*100:+6.2f}"
                  f"  +-1Acc {(n['acc19_pm1']-o['acc19_pm1'])*100:+6.2f}"
                  f"  far>=3 {(n['far_ge3']-o['far_ge3'])*100:+6.2f}"
                  f"  far>=5 {(n['far_ge5']-o['far_ge5'])*100:+6.2f}")
    print("  One seed: a difference of a few tenths of a QWK point is inside seed-to-seed noise.")

    print("\n=== error-distance histogram, TEST (share of predictions; THIS THESIS'S analysis) ===")
    print(f"  {'arm':<12}" + "".join(f"{('|d|=' + k):>9}" for k in ("0", "1", "2", "3", "4", "5+")))
    for r in rows:
        h = r.get("test_error_hist")
        if h:
            print(f"  {r['arm']:<12}" + "".join(f"{h[k]*100:>9.1f}" for k in ("0", "1", "2", "3", "4", "5+")))

    print("\n=== run checks (FIX 1: one GPU, batch 64 | FIX 2: reported == selected epoch) ===")
    for r in rows:
        p = r.get("provenance", {})
        sc = r.get("selection_check", {})
        spe = r.get("steps_per_epoch_observed") or [None]
        print(f"  {r['arm']:<12} n_gpu={p.get('n_gpu')}  batch={r.get('realised_train_batch')}  "
              f"steps/epoch={spe[0]} (expected {r.get('steps_per_epoch_expected')}, "
              f"ok={r.get('steps_per_epoch_ok')})  reported==selected: {sc.get('exact')}  "
              f"selected epoch {r.get('selected_epoch')}  transformers {p.get('transformers')}")

    for r in rows:
        if "noise_sigma_selected" in r:
            hist = ", ".join(f"{h['noise_sigma']:.3f}" for h in r.get("noise_sigma_history", []))
            print(f"\n=== {r['arm']} learned noise sigma: {r['noise_sigma_selected']:.4f} at the "
                  f"selected epoch (per epoch: {hist})")
        if "ablation_dev_threshold" in r:
            a = r["ablation_dev_threshold"]
            print(f"    {r['arm']} dev-tuned threshold t={a['t']:.2f}: "
                  f"dev QWK {a['dev']['qwk']*100:.2f}, test QWK {a['test']['qwk']*100:.2f}")
        if "ablation_qwk_selected" in r:
            a = r["ablation_qwk_selected"]
            print(f"    {r['arm']} QWK-selected checkpoint (epoch {a['selected_epoch']}): "
                  f"dev QWK {a['dev']['qwk']*100:.2f}, test QWK {a['test']['qwk']*100:.2f}")


# ----------------------------------------------------------------- 14. main
def main():
    global DATASET_SHA
    print("=" * 78)
    print(f"barec_contrib_v1_part{PART}.py -- arms: {', '.join(a['name'] for a in ARMS)}")
    print(SCRIPT_VERSION)
    print("=" * 78)
    print("\nDECLARED DEFAULTS -- values that appear in NO paper, or that deviate from train.py:")
    for k, v, why in DECLARED_DEFAULTS:
        print(f"  - {k}: {v}\n      why: {why}")
    print(f"\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"visible GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}  "
          f"fp16: {CONFIG['fp16']}")
    if transformers.__version__ != REFERENCE_TRANSFORMERS:
        print(f"  NOTE: transformers {transformers.__version__} here, {REFERENCE_TRANSFORMERS} in the "
              f"v3 reference runs. A library change can move numbers slightly; say so in the thesis.")

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
    stats = train_label_stats(ds["train"]["labels"])
    print(f"  train rows per level: {stats['counts'].astype(int).tolist()}")
    print(f"  CORN bias init (logit P(y > j | y >= j)): "
          f"{np.round(stats['corn_bias'], 2).tolist()}")

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
        print(f"\n--- {arm['name']}  (group={arm['group']}, control={arm['control']}, "
              f"{arm['target']})  [{elapsed/3600:.2f} h elapsed]")
        try:
            res = run_arm(arm, tok, enc, collator, stats)
            f.write_text(json.dumps(res, indent=2))
            arm_minutes.append(res["minutes"])
            print(f"  {arm['name']}: dev QWK {res['dev']['qwk']*100:.2f}  "
                  f"test QWK {res['test']['qwk']*100:.2f}  ({res['minutes']} min)  "
                  f"batch {res['realised_train_batch']}  "
                  f"reported==selected: {res['selection_check']['exact']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            (RESULTS / f"{arm['name']}.FAILED.txt").write_text(repr(e))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

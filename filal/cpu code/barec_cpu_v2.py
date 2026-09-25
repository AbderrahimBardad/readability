# =============================================================================
#  barec_cpu_v2.py
#  FROZEN-BACKBONE companion to barec_exact_v2.py.
#
#  This is a NEW file. It does not modify or overwrite barec_cpu.py.
#  It is barec_cpu.py plus the fixes from the 2026-09-11 audit, and nothing
#  else -- every line that was already verified correct (data loading, split
#  asserts, frozen-feature extraction, hyper-parameters, selection rule,
#  metrics, loss formulas for CE / MSE / OLL / EMD / SOFT / CORAL / Soft-QWK)
#  is carried over unchanged.
#
#  CHANGELOG vs barec_cpu.py (only these three things changed):
#    1. WKL removed entirely (loss function, arm, self-test check). Same
#       reason as barec_exact_v2.py: WKL is in train.py but in NO table of
#       the BAREC paper and is not CORAL or Soft-QWK. Author's instruction:
#       drop it.
#    2. CPU["dropout"] = 0.1 is now declared. It was a hardcoded literal in
#       barec_cpu.py's CPU{} dict that never appeared in DECLARED_DEFAULTS_CPU
#       even though it is used inside Head.__init__ and is active during
#       training (it affects results, unlike extract_bs/chunk which only
#       affect the frozen, eval()-mode feature-extraction pass and cannot
#       change the extracted feature values). It is numerically equal to
#       aubmindlab/bert-base-arabertv2's own hidden_dropout_prob (confirmed
#       against its published config.json: hidden_dropout_prob=0.1), which is
#       exactly what barec_exact_v2.py's CoralModel reads dynamically from the
#       pretrained config -- so the VALUE was already correct, only its
#       declaration was missing. Fixed by adding it to DECLARED_DEFAULTS_CPU.
#    3. Arms are now tagged by group and reported in three separate sections,
#       mirroring barec_exact_v2.py:
#         "core"     -- CE, MSE, OLL15, EMD, SOFT2-5, CORAL. Identical head
#                       training settings (lr, epochs, batch, dropout) for
#                       every arm; differs ONLY in head + loss.
#         "softqwk"  -- Soft-QWK. Two-phase MSE -> SoftQWK warm-up, same as
#                       barec_exact_v2.py, because that is how STBW define the
#                       method. Reported on its own.
#         "ablation" -- CORAL_ZEROINIT (kept: it is the direct CPU-side test
#                       of the GPU travel-budget diagnosis, useful on its own
#                       terms, but explicitly not part of the fair comparison).
#       Output directory is ./barec_cpu_out_v2 (Kaggle: /kaggle/working/v2cpu),
#       a NEW directory, so this script never reads or overwrites a feature
#       cache or a results_cpu/*.json produced by the original barec_cpu.py.
#    4. restore_previous_results(): lets a Kaggle 'Save Version' commit pick
#       up where an earlier, separate commit left off -- both finished arm
#       JSONs AND the frozen-feature cache (the slowest single step here) are
#       copied in from a previous commit's output attached as an input.
#
# -----------------------------------------------------------------------------
#  WHAT THIS IS -- AND WHAT IT IS NOT
#
#  This is NOT a reproduction of BAREC Table 6. Table 6 is a FINE-TUNING result.
#  A frozen AraBERTv2 scores far below it and that is the correct, expected
#  outcome, not a failure. Never merge this table with barec_exact_v2.py's
#  table and never compare a number here to a number in the paper.
#
#  What it IS, and why it earns its place in the thesis:
#
#   1. Goudjil's brief asks for "CPU-friendly training settings". Head training
#      here runs on a laptop CPU in minutes.
#   2. It is a CLEANER isolation of head-and-loss than fine-tuning. When the
#      encoder is fine-tuned, each arm ends up with a DIFFERENT encoder, so
#      the comparison is contaminated. Frozen, every arm sees byte-identical
#      features and the only thing that varies is the head and the loss.
#   3. It is cheap enough for 5 seeds per arm, so every number is reported as
#      mean +- std. The BAREC paper reports no variance at all.
#   4. It answers a question one run cannot: does the loss ranking survive
#      when the encoder cannot move? The script prints both rankings side by
#      side when barec_exact_v2.py's results are present.
#   5. CORAL_ZEROINIT is free here and is a DIRECT test of the travel-budget
#      diagnosis. On GPU, 6 epochs at lr 5e-5 give the cut points a total
#      travel budget of only 5e-5 * 5142 = 0.257 against a required spread
#      near 10, so a zero-initialised CORAL head cannot become usable. Here
#      the budget is 1e-3 * 42850 = 42.9, comfortably more than enough.
#
# -----------------------------------------------------------------------------
#  WHAT IS SHARED WITH barec_exact_v2.py, BYTE FOR BYTE
#    the loss functions, the metric set, the decoders, the BAREC 7/5/3 collapse
#    tables, the split asserts, the D3Tok assert, and the self-tests.
#
#  WHAT IS NECESSARILY DIFFERENT, AND WHY
#    BAREC's optimiser settings (lr 5e-5, 6 epochs) are for fine-tuning 135M
#    parameters. A linear head on frozen features at lr 5e-5 for 6 epochs simply
#    underfits -- it would be a broken experiment, not a faithful one. So the
#    head-training hyper-parameters here are OURS and are listed in
#    DECLARED_DEFAULTS_CPU, IDENTICAL across all arms. Batch size stays at
#    BAREC's 64 so that SoftQWK -- a batch-level statistic -- is computed over
#    the same number of samples as on GPU.
# =============================================================================

import os, sys, json, math, time, shutil, gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, accuracy_score, mean_absolute_error, f1_score

SCRIPT_VERSION = ("barec_cpu_v2 (2026-09-11): WKL removed; dropout declared; "
                   "SOFTQWK and CORAL_ZEROINIT isolated from the fair-comparison "
                   "core; separate output directory from barec_cpu.py")

# ----------------------------------------------------------------- config
CONFIG = dict(
    model_name = "aubmindlab/bert-base-arabertv2",
    dataset    = "CAMeL-Lab/BAREC-Corpus-v1.0",
    input_var  = "D3Tok",
    label_col  = "Readability_Level_19",
    max_length = 512,
    num_labels = 19,
)

CPU = dict(
    lr          = 1e-3,
    epochs      = 50,
    batch       = 64,          # BAREC's batch, so SoftQWK sees 64 samples
    weight_decay= 0.0,         # BAREC / HF default
    warmup      = 0.0,         # BAREC / HF default
    dropout     = 0.1,         # OURS -- now declared below, see CHANGELOG #2
    seeds       = [42, 43, 44, 45, 46],
    extract_bs  = 32,
    chunk       = 2000,        # resumable extraction chunk
)

DECLARED_DEFAULTS_CPU = [
    ("head learning rate", CPU["lr"],
     "OURS. BAREC's 5e-5 is a fine-tuning rate; a linear probe underfits at it. "
     "Identical across all arms."),
    ("head epochs", CPU["epochs"],
     "OURS. Equal budget for every arm, no early stopping, exactly as on GPU."),
    ("head dropout", CPU["dropout"],
     "OURS -- a hardcoded literal, not read from the pretrained config the way "
     "barec_exact_v2.py's CoralModel does it. Numerically equal to "
     "aubmindlab/bert-base-arabertv2's own hidden_dropout_prob (0.1, from its "
     "published config.json), so it matches the fine-tuned head's dropout, but "
     "that match was previously asserted only in a comment, not declared here. "
     "Identical across all arms."),
    ("seeds", CPU["seeds"],
     "OURS. Five seeds so every number carries a standard deviation."),
    ("representations", "pooler_output (primary) and masked mean of last_hidden_state",
     "OURS. pooler_output is the input the fine-tuned head sees, so it preserves the "
     "invariant between the two experiments; mean-pooling is reported alongside "
     "because a frozen NSP pooler is a known-weak representation. BOTH are reported "
     "-- neither is chosen after seeing the results."),
    ("batch size", CPU["batch"],
     "BAREC's 64, kept so SoftQWK matches the GPU run's batch-level statistic."),
]

SMOKE = False

BASE = (Path("/kaggle/working") / "v2cpu") if Path("/kaggle").exists() else Path("./barec_cpu_out_v2")
FEAT = BASE / "features"
RES = BASE / "results_cpu"
GPU_RES = Path("/kaggle/working/v2") / "results" if Path("/kaggle").exists() else Path("./barec_out_v2/results")
for p in (FEAT, RES):
    p.mkdir(parents=True, exist_ok=True)


def restore_previous_results():
    """Same purpose as barec_exact_v2.py's function of the same name: resume
    ACROSS separate Kaggle 'Save Version' commits, since /kaggle/working is
    wiped clean at the start of each new commit. Attach the PREVIOUS commit's
    own output as an input ('File > Add Input' > this notebook > an earlier
    version) before running the next commit.

    Two things are recovered here, both from /kaggle/input/<slug>/v2cpu/...:
      1. results_cpu/*.json -- a finished (arm, representation) combination.
         Only WHOLE, finished results are recovered; a seed that was
         mid-training when the previous commit died is simply retrained.
      2. features/*  -- the frozen AraBERTv2 forward pass over all 69,441
         sentences. This is the single slowest step on CPU (roughly 30-60
         minutes, per the estimate printed in [frozen feature extraction]
         below); recovering it means a second commit skips straight to head
         training instead of redoing the encoder pass for nothing.
    """
    kag_input = Path("/kaggle/input")
    if not kag_input.exists():
        print("  not running on Kaggle -- nothing to restore from /kaggle/input")
        return
    restored_json, restored_feat = [], None
    for src in sorted(kag_input.glob("*/v2cpu/results_cpu")):
        for f in sorted(src.glob("*.json")):
            dst = RES / f.name
            if not dst.exists():
                dst.write_text(f.read_text())
                restored_json.append((f.name, str(src)))
    if not any(FEAT.iterdir()):
        for src in sorted(kag_input.glob("*/v2cpu/features")):
            shutil.copytree(src, FEAT, dirs_exist_ok=True)
            restored_feat = str(src)
            break   # only need the extraction cache from one previous commit
    if restored_json:
        print(f"  restored {len(restored_json)} finished (arm, representation) result(s):")
        for name, src in restored_json:
            print(f"    {name}  <-  {src}")
    else:
        print("  no previous-commit results_cpu found under /kaggle/input")
    print(f"  restored frozen-feature cache from: {restored_feat}" if restored_feat
          else "  no previous-commit feature cache found -- extraction will run from scratch")


K = CONFIG["num_labels"]

# =============================================================================
#  SHARED BLOCK -- copied verbatim from barec_exact_v2.py. Do not edit one
#  without the other; self_test_losses() below is what catches a divergence.
# =============================================================================
BAREC_7 = {1:1,2:1,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:4,11:4,12:5,13:5,14:6,15:6,16:7,17:7,18:7,19:7}
BAREC_5 = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:2,9:2,10:2,11:2,12:3,13:3,14:4,15:4,16:5,17:5,18:5,19:5}
BAREC_3 = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:2,13:2,14:3,15:3,16:3,17:3,18:3,19:3}

_LEVELS = np.arange(K)
_W = (_LEVELS[:, None] - _LEVELS[None, :]) ** 2
_D = np.abs(_LEVELS[:, None] - _LEVELS[None, :]).astype(np.float64)


def _collapse(y0, table):
    """y0 is 0-based; the BAREC tables are 1-based."""
    return np.array([table[int(v) + 1] for v in y0])


def metrics_from_preds(gold0, pred0):
    gold0 = np.asarray(gold0).astype(int)
    pred0 = np.asarray(pred0).astype(int)
    d = np.abs(gold0 - pred0)
    return {
        "qwk":       float(cohen_kappa_score(gold0, pred0, weights="quadratic")),
        "acc19":     float(accuracy_score(gold0, pred0)),
        "acc19_pm1": float((d <= 1).mean()),
        "dist":      float(mean_absolute_error(gold0, pred0)),
        "acc7":      float(accuracy_score(_collapse(gold0, BAREC_7), _collapse(pred0, BAREC_7))),
        "acc5":      float(accuracy_score(_collapse(gold0, BAREC_5), _collapse(pred0, BAREC_5))),
        "acc3":      float(accuracy_score(_collapse(gold0, BAREC_3), _collapse(pred0, BAREC_3))),
        "macro_f1":  float(f1_score(gold0, pred0, average="macro", zero_division=0)),
        "far_ge3":   float((d >= 3).mean()),      # THIS THESIS'S OWN metric
        "far_ge5":   float((d >= 5).mean()),      # THIS THESIS'S OWN metric
    }


def decode_cls(logits):
    return np.argmax(logits, axis=1)


def decode_reg(logits):
    return np.clip(np.rint(np.asarray(logits).squeeze(-1)), 0, K - 1).astype(int)


def decode_coral(logits, t=0.5):
    return (1.0 / (1.0 + np.exp(-np.asarray(logits))) > t).sum(axis=1).astype(int)


def _fp32(x):
    """D2: never compute an objective in fp16. float64 is preserved so the
    self-test can compare against train.py at double precision."""
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


def self_test_losses():
    torch.manual_seed(0)
    n = 37
    logits = torch.randn(n, K, dtype=torch.float64) * 3
    labels = torch.randint(0, K, (n,))
    dm = [[abs(i - j) for i in range(K)] for j in range(K)]
    pr = F.softmax(logits, dim=1)
    tl = [K * [labels[k].item()] for k in range(n)]
    li = n * [[k for k in range(K)]]
    checks = []
    checks.append(("CE", float(np.mean([-math.log(pr[j, labels[j]].item()) for j in range(n)])),
                   float(loss_ce(logits, labels))))
    dist = torch.tensor([[float(dm[tl[j][i]][li[j][i]]) for i in range(K)] for j in range(n)],
                        dtype=torch.float64)
    checks.append(("OLL15", float((-torch.log(1 - pr) * dist ** 1.5).sum(1).mean()),
                   float(loss_oll(logits, labels, alpha=1.5))))
    cdf = torch.tensor([labels[k].item() * [0.] + (K - labels[k].item()) * [1.] for k in range(n)],
                       dtype=torch.float64)
    checks.append(("EMD", float(((torch.cumsum(pr, 1) - cdf) ** 2).sum(1).mean()),
                   float(loss_emd(logits, labels))))
    for beta in (2, 3, 4, 5):
        s = torch.tensor([[math.exp(-beta * dm[tl[j][i]][li[j][i]])
                           / sum(math.exp(-beta * dm[q][li[j][i]]) for q in range(K))
                           for i in range(K)] for j in range(n)], dtype=torch.float64)
        checks.append((f"SOFT{beta}", float((-torch.log(pr) * s).sum(1).mean()),
                       float(loss_soft(logits, labels, beta=float(beta)))))
    proj = torch.randn(n, 1, dtype=torch.float64)
    z = proj + torch.sort(torch.randn(K - 1, dtype=torch.float64), descending=True).values
    t = (labels.unsqueeze(1) > torch.arange(K - 1).unsqueeze(0)).to(torch.float64)
    lit = float(np.mean([sum(math.log1p(math.exp(-abs(z[j, k].item()))) + max(z[j, k].item(), 0.)
                             - z[j, k].item() * t[j, k].item() for k in range(K - 1))
                         for j in range(n)]))
    checks.append(("CORAL", lit, float(loss_coral(z, labels))))
    yh = labels.to(torch.float64) + torch.randn(n, dtype=torch.float64)
    hard = np.clip(np.rint(yh.numpy()), 0, K - 1).astype(int)
    checks.append(("SoftQWK->sklearn",
                   1.0 - cohen_kappa_score(labels.numpy(), hard, weights="quadratic"),
                   float(loss_softqwk(torch.tensor(hard, dtype=torch.float64).unsqueeze(1),
                                      labels, sigma=1e-3))))
    bad = [nm for nm, a, b in checks if abs(a - b) >= 1e-9]
    for nm, a, b in checks:
        print(f"    {nm:<18}{a:>18.12f}{b:>18.12f}{abs(a-b):>12.2e}"
              f"  {'ok' if abs(a-b) < 1e-9 else 'FAIL'}")
    assert not bad, f"loss mismatch vs barec_exact_v2.py: {bad}"


# =============================================================================
#  END SHARED BLOCK
# =============================================================================

# ----------------------------------------------------------------- data
def load_barec():
    from datasets import load_dataset
    ds = load_dataset(CONFIG["dataset"])
    dev_key = "dev" if "dev" in ds else "validation"          # NOT 'validation'
    expect = {"train": 54845, dev_key: 7310, "test": 7286}    # paper Table 3
    for k, v in expect.items():
        assert len(ds[k]) == v, f"split {k} is {len(ds[k])}, paper Table 3 says {v}"
    iv, lc = CONFIG["input_var"], CONFIG["label_col"]
    sample = ds["train"][iv][:2000]
    rate = sum("+" in s for s in sample) / len(sample)
    assert rate > 0.5, f"D3Tok column has '+' in only {rate:.1%} of rows"
    for n_lv, table in ((7, BAREC_7), (5, BAREC_5), (3, BAREC_3)):
        col = f"Readability_Level_{n_lv}"
        if col in ds["train"].column_names:
            assert (np.array(ds["train"][col])
                    == np.array([table[v] for v in ds["train"][lc]])).all(), \
                f"collapse map {n_lv} disagrees with {col}"
    print(f"  splits ok: 54845 / 7310 / 7286   D3Tok '+' in {rate:.1%} of sample")
    out = {}
    for split, key in (("train", "train"), ("dev", dev_key), ("test", "test")):
        out[split] = (list(ds[key][iv]), np.array([int(l) - 1 for l in ds[key][lc]]))
    return out


# ----------------------------------------------------------------- extraction
@torch.no_grad()
def extract(split, texts, tok, model, device):
    """Frozen forward pass. Resumable in chunks; length-sorted to cut padding."""
    done = FEAT / f"{split}.npz"
    if done.exists():
        z = np.load(done)
        print(f"  {split}: cached {z['pooler'].shape}")
        return z["pooler"], z["mean"]

    n = len(texts)
    lens = np.array([len(tok(t, truncation=True, max_length=CONFIG["max_length"])["input_ids"])
                     for t in texts])
    order = np.argsort(lens)                      # short first -> far less padding
    pool = np.zeros((n, model.config.hidden_size), dtype=np.float32)
    mean = np.zeros((n, model.config.hidden_size), dtype=np.float32)

    part_dir = FEAT / f"{split}_parts"
    part_dir.mkdir(exist_ok=True)
    chunks = [order[i:i + CPU["chunk"]] for i in range(0, n, CPU["chunk"])]
    t0 = time.time()
    for ci, idx in enumerate(chunks):
        pf = part_dir / f"{ci}.npz"
        if pf.exists():
            z = np.load(pf)
            pool[idx], mean[idx] = z["p"], z["m"]
            continue
        ps, ms = [], []
        for s in range(0, len(idx), CPU["extract_bs"]):
            bi = idx[s:s + CPU["extract_bs"]]
            enc = tok([texts[i] for i in bi], truncation=True,
                      max_length=CONFIG["max_length"], padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            ps.append(out.pooler_output.float().cpu().numpy())
            m = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
            ms.append(((out.last_hidden_state * m).sum(1) / m.sum(1)).float().cpu().numpy())
        p, mm = np.concatenate(ps), np.concatenate(ms)
        np.savez_compressed(pf, p=p, m=mm)
        pool[idx], mean[idx] = p, mm
        el = time.time() - t0
        print(f"    {split} chunk {ci+1}/{len(chunks)}  {el/60:.1f} min elapsed, "
              f"~{el/(ci+1)*(len(chunks)-ci-1)/60:.1f} min left", flush=True)
    np.savez_compressed(done, pooler=pool, mean=mean)
    shutil.rmtree(part_dir, ignore_errors=True)
    return pool, mean


# ----------------------------------------------------------------- heads
class Head(nn.Module):
    """Exactly the head barec_exact_v2.py trains: dropout -> linear on the
    frozen feature. n_out is 19 (softmax), 1 (regression) or 18 (CORAL)."""

    def __init__(self, dim, n_out, coral=False, cutpoint_init=None):
        super().__init__()
        self.drop = nn.Dropout(CPU["dropout"])
        self.coral = coral
        if coral:
            self.proj = nn.Linear(dim, 1, bias=False)
            b = torch.zeros(K - 1) if cutpoint_init is None else torch.as_tensor(cutpoint_init)
            self.cutpoints = nn.Parameter(b.clone().float())
        else:
            self.proj = nn.Linear(dim, n_out)

    def forward(self, x):
        z = self.proj(self.drop(x))
        return z + self.cutpoints if self.coral else z


def coral_bias_init(y):
    ps = np.clip(np.array([(np.asarray(y) > k).mean() for k in range(K - 1)]), 1e-4, 1 - 1e-4)
    b = np.log(ps / (1 - ps))
    assert np.all(np.diff(b) <= 0)
    return b.astype(np.float32)


# ----------------------------------------------------------------- training
def _eval(model, X, y, loss_fn, loss_kw, decode, batch):
    model.eval()
    losses, outs = [], []
    with torch.no_grad():
        for s in range(0, len(y), batch):
            xb, yb = X[s:s + batch], y[s:s + batch]
            lg = model(xb)
            losses.append(float(loss_fn(lg, yb, **loss_kw)) * len(yb))
            outs.append(lg.numpy())
    lg = np.concatenate(outs)
    return sum(losses) / len(y), lg, metrics_from_preds(y.numpy(), decode(lg))


def train_head(make_model, loss_fn, loss_kw, data, decode, epochs, seed, lr=None):
    """Manual loop. Selection: lowest dev loss over the full budget, evaluated
    every epoch, NO early stopping -- the same rule barec_exact_v2.py uses."""
    lr = lr or CPU["lr"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr = data["train"]
    Xdv, ydv = data["dev"]
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CPU["weight_decay"],
                            betas=(0.9, 0.999), eps=1e-8)
    batch = CPU["batch"]
    spe = math.ceil(len(ytr) / batch)
    total = spe * epochs
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total))
    rng = np.random.default_rng(seed)
    best = (float("inf"), None, -1)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(ytr)))
        for s in range(0, len(ytr), batch):
            bi = perm[s:s + batch]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(Xtr[bi]), ytr[bi], **loss_kw).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # HF default
            opt.step()
            sch.step()
        dl, _, dm = _eval(model, Xdv, ydv, loss_fn, loss_kw, decode, batch)
        hist.append({"epoch": ep + 1, "dev_loss": dl, "dev_qwk": dm["qwk"]})
        if dl < best[0]:
            best = (dl, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    return model, hist, best[2]


# ----------------------------------------------------------------- arms
# Every arm carries a "group" tag, mirroring barec_exact_v2.py:
#   "core"     -- identical head hyper-parameters, differs ONLY in head+loss.
#   "softqwk"  -- necessarily different training procedure (two-phase warm-up,
#                 batch-level loss). Reported on its own.
#   "ablation" -- explicitly not part of the fair comparison.
ARMS = [
    dict(name="CE",       kind="cls",   loss=loss_ce,      group="core"),
    dict(name="MSE",      kind="reg",   loss=loss_mse,     group="core"),
    dict(name="OLL15",    kind="cls",   loss=loss_oll,     group="core", kw=dict(alpha=1.5)),
    dict(name="EMD",      kind="cls",   loss=loss_emd,     group="core"),
    dict(name="SOFT2",    kind="cls",   loss=loss_soft,    group="core", kw=dict(beta=2.0)),
    dict(name="SOFT3",    kind="cls",   loss=loss_soft,    group="core", kw=dict(beta=3.0)),
    dict(name="SOFT4",    kind="cls",   loss=loss_soft,    group="core", kw=dict(beta=4.0)),
    dict(name="SOFT5",    kind="cls",   loss=loss_soft,    group="core", kw=dict(beta=5.0)),
    dict(name="CORAL",    kind="coral", loss=loss_coral,   group="core"),
    dict(name="SOFTQWK",  kind="reg",   loss=loss_softqwk, group="softqwk", kw=dict(sigma=1.0), two_phase=True),
    # free here, and a direct test of the travel-budget diagnosis
    dict(name="CORAL_ZEROINIT", kind="coral", loss=loss_coral, group="ablation", zero_init=True),
]

DECODE = {"cls": decode_cls, "reg": decode_reg, "coral": decode_coral}
NOUT = {"cls": K, "reg": 1, "coral": K - 1}


def run_arm(arm, rep, data, cut_init):
    kind = arm["kind"]
    decode = DECODE[kind]
    dim = data["train"][0].shape[1]
    epochs = 3 if SMOKE else CPU["epochs"]
    per_seed = []
    for seed in (CPU["seeds"][:1] if SMOKE else CPU["seeds"]):
        def make():
            return Head(dim, NOUT[kind], coral=(kind == "coral"),
                        cutpoint_init=None if arm.get("zero_init") else cut_init)

        if arm.get("two_phase"):
            p1 = epochs // 2
            m1, h1, _ = train_head(make, loss_mse, {}, data, decode_reg, p1, seed)
            state = {k: v.clone() for k, v in m1.state_dict().items()}

            def make2(_s=state):
                m = make()
                m.load_state_dict(_s)
                return m
            # fresh optimiser AND fresh scheduler
            model, hist, sel = train_head(make2, arm["loss"], arm.get("kw", {}), data,
                                          decode, epochs - p1, seed)
            hist = [{"phase": 1, **h} for h in h1] + [{"phase": 2, **h} for h in hist]
        else:
            model, hist, sel = train_head(make, arm["loss"], arm.get("kw", {}), data,
                                          decode, epochs, seed)

        row = {"seed": seed, "selected_epoch": sel, "history": hist}
        for split in ("dev", "test"):
            X, y = data[split]
            _, lg, mt = _eval(model, X, y, arm["loss"], arm.get("kw", {}), decode, CPU["batch"])
            row[split] = mt
            if kind == "coral" and split == "dev":
                b = model.cutpoints.detach().numpy()
                row["cutpoint_spread"] = float(b.max() - b.min())
                row["cutpoints_ordered"] = bool(np.all(np.diff(b) <= 1e-6))
        per_seed.append(row)
        print(f"    seed {seed}: dev QWK {row['dev']['qwk']*100:6.2f}  "
              f"test QWK {row['test']['qwk']*100:6.2f}"
              + (f"  spread {row['cutpoint_spread']:.2f}" if "cutpoint_spread" in row else ""))

    agg = {"arm": arm["name"], "group": arm["group"], "representation": rep,
           "seeds": per_seed, "n_seeds": len(per_seed)}
    for split in ("dev", "test"):
        agg[split] = {}
        for k in per_seed[0][split]:
            v = np.array([r[split][k] for r in per_seed])
            agg[split][k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    if "cutpoint_spread" in per_seed[0]:
        v = np.array([r["cutpoint_spread"] for r in per_seed])
        agg["cutpoint_spread"] = {"mean": float(v.mean()),
                                  "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return agg


# ----------------------------------------------------------------- report
def report():
    rows = [json.loads(f.read_text()) for f in sorted(RES.glob("*.json"))]
    order = [a["name"] for a in ARMS]
    groups = [
        ("core", "CORE ARMS -- fair comparison: identical head hyper-parameters, "
                  "differ ONLY in head+loss"),
        ("softqwk", "SOFTQWK -- two-phase warm-up + batch-level loss. NOT directly "
                    "comparable row-for-row with CORE"),
        ("ablation", "ABLATIONS -- explicitly excluded from the fair comparison"),
    ]
    for rep in ("pooler", "mean"):
        sub = [r for r in rows if r["representation"] == rep]
        if not sub:
            continue
        sub.sort(key=lambda r: order.index(r["arm"]) if r["arm"] in order else 99)
        for split in ("dev", "test"):
            print(f"\n=== FROZEN AraBERTv2 [{rep}] -- BAREC {split.upper()} "
                  f"(mean +- std over {sub[0]['n_seeds']} seeds)")
            print("    NOT comparable to BAREC Table 6 -- that is a fine-tuning result.")
            for gkey, glabel in groups:
                grows = [r for r in sub if r.get("group") == gkey]
                if not grows:
                    continue
                print(f"\n    -- {glabel} --")
                print(f"    {'arm':<16}{'QWK':>15}{'Acc19':>15}{'+-1Acc':>15}{'Dist':>14}"
                      f"{'far>=3*':>14}{'far>=5*':>14}")
                for r in grows:
                    m = r[split]
                    def c(k, s=100):
                        return f"{m[k]['mean']*s:7.2f}+-{m[k]['std']*s:<5.2f}"
                    print(f"    {r['arm']:<16}{c('qwk'):>15}{c('acc19'):>15}{c('acc19_pm1'):>15}"
                          f"{c('dist',1):>14}{c('far_ge3'):>14}{c('far_ge5'):>14}")
            print("  * far>=3 / far>=5 are THIS THESIS'S metrics, not BAREC's.")

        # the CORAL travel-budget test
        z = next((r for r in sub if r["arm"] == "CORAL_ZEROINIT"), None)
        c = next((r for r in sub if r["arm"] == "CORAL"), None)
        if z and c:
            steps = CPU["epochs"] * math.ceil(54845 / CPU["batch"])
            print(f"\n=== CORAL travel-budget test [{rep}]")
            print(f"    budget here = lr*steps = {CPU['lr']*steps:.1f}  (on GPU it is 0.26)")
            print(f"    logit-init  spread {c['cutpoint_spread']['mean']:6.2f}  "
                  f"dev QWK {c['dev']['qwk']['mean']*100:.2f}")
            print(f"    zero-init   spread {z['cutpoint_spread']['mean']:6.2f}  "
                  f"dev QWK {z['dev']['qwk']['mean']*100:.2f}")
            print("    If zero-init recovers here but failed on GPU, the diagnosis is a "
                  "measurement, not an argument.")

        # does the loss ranking survive freezing? (core arms only, apples to apples)
        gpu = {}
        if GPU_RES.exists():
            for f in GPU_RES.glob("*.json"):
                try:
                    g = json.loads(f.read_text())
                    if g.get("group") == "core" and "dev" in g:
                        gpu[g["arm"]] = g["dev"]["qwk"]
                except Exception:
                    pass
        core_names = {a["name"] for a in ARMS if a["group"] == "core"}
        shared = [r["arm"] for r in sub if r["arm"] in gpu and r["arm"] in core_names]
        if len(shared) >= 3:
            fro = sorted(shared, key=lambda a: -next(r for r in sub if r["arm"] == a)["dev"]["qwk"]["mean"])
            fin = sorted(shared, key=lambda a: -gpu[a])
            print(f"\n=== loss ranking, dev QWK, best first, CORE arms only [{rep}]")
            print(f"    fine-tuned (GPU): {' > '.join(fin)}")
            print(f"    frozen    (CPU): {' > '.join(fro)}")
            print(f"    {'RANKING SURVIVES FREEZING' if fro == fin else 'RANKING CHANGES -- that is the finding'}")


# ----------------------------------------------------------------- main
def main():
    print("=" * 78)
    print("barec_cpu_v2.py -- frozen AraBERTv2, head and loss only, CPU")
    print(SCRIPT_VERSION)
    print("=" * 78)
    print("\nThis is NOT a reproduction of BAREC Table 6. Table 6 is a fine-tuning")
    print("result; a frozen encoder scores far below it and that is expected.\n")
    print("DECLARED DEFAULTS -- values that appear in NO paper:")
    for k, v, why in DECLARED_DEFAULTS_CPU:
        print(f"  - {k}: {v}\n      why: {why}")

    print("\n[resume check -- across separate Kaggle commits]")
    restore_previous_results()

    print("\n[self-tests]  (identical losses to barec_exact_v2.py)")
    self_test_losses()

    print("\n[data]")
    raw = load_barec()
    if SMOKE:
        raw["train"] = (raw["train"][0][:2000], raw["train"][1][:2000])

    print("\n[frozen feature extraction]")
    from transformers import AutoTokenizer, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}"
          + ("" if device == "cuda" else "   (CPU: expect roughly half an hour to an hour "
                                         "for all 69,441 sentences; it is resumable)"))
    tok = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    enc_model = AutoModel.from_pretrained(CONFIG["model_name"]).to(device).eval()
    for p in enc_model.parameters():
        p.requires_grad_(False)

    feats = {}
    for split in ("train", "dev", "test"):
        pool, mean = extract(split, raw[split][0], tok, enc_model, device)
        feats[split] = {"pooler": pool, "mean": mean}
    del enc_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    cut_init = coral_bias_init(raw["train"][1])
    print(f"  CORAL cut-point init: ordered, spread {cut_init.max()-cut_init.min():.2f}")

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    for rep in ("pooler", "mean"):
        data = {s: (torch.from_numpy(feats[s][rep]).float(),
                    torch.from_numpy(raw[s][1]).long()) for s in ("train", "dev", "test")}
        print(f"\n[arms on {rep} features, dim {data['train'][0].shape[1]}]")
        for arm in ARMS:
            f = RES / f"{rep}__{arm['name']}.json"
            if f.exists():
                print(f"  {rep}/{arm['name']}: done, skipping")
                continue
            print(f"  {rep}/{arm['name']}  (group={arm['group']})")
            t0 = time.time()
            agg = run_arm(arm, rep, data, cut_init)
            agg["minutes"] = round((time.time() - t0) / 60, 1)
            f.write_text(json.dumps(agg, indent=2))
            shutil.make_archive(str(BASE / "results_cpu_backup"), "zip", RES)

    report()
    shutil.make_archive(str(BASE / "results_cpu_backup"), "zip", RES)
    print(f"\nresults in {RES} and {BASE}/results_cpu_backup.zip")


if __name__ == "__main__":
    main()

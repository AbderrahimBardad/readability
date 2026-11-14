# GPU results

One JSON file per method, written by the scripts in `../gpu code/`.
All metric values are fractions between 0 and 1 (multiply by 100 for percentages), except `dist`.

## Main fields

| Field | Meaning |
|---|---|
| `arm` | method name |
| `group`, `target` | group of the method and its source (paper or BAREC table) |
| `head` | prediction head: `cls`, `reg`, `coral`, `corn` or `clm` |
| `micro_batch`, `grad_accum`, `gradient_checkpointing` | how the batch of 64 was formed |
| `history` | development metrics after each of the 6 epochs |
| `selected_epoch` | epoch with the lowest development loss (BAREC rule) |
| `dev`, `test` | metrics of the selected checkpoint |
| `selection_check` | reloaded checkpoint vs training log (must be `exact: true`) |
| `ablation_qwk_selected` | the same metrics for the epoch with the highest development QWK |
| `provenance` | software versions, GPU, corpus revision, batch |
| `minutes` | run time |

Metrics: `qwk`, `acc19` (exact level), `acc19_pm1` (within one level), `dist` (mean distance in levels),
`acc7`, `acc5`, `acc3` (accuracy at 7, 5 and 3 levels), `macro_f1`, `far_ge3` and `far_ge5`
(share of errors of at least 3 and 5 levels).

Extra fields: CORAL files contain the cut-points after every epoch and the tuned-threshold ablation;
SOFTQWK contains the history of its first (MSE) phase; LBL10 contains its class weights.

## Files

- Group 1: `CE`, `MSE`, `OLL15`, `EMD`, `SOFT3` (and `SOFT2`, `SOFT4`, `SOFT5`, run but not selected)
- Group 2: `CORAL`, `SOFTQWK`, and the ablation `CORAL_HEADLR`
- Group 3: `MV`, `CCC`, `CLM`, `OE`, `RANK`, `PLR`, `CORN`
- Group 4: `LISTNET`, `DLDL2`, `TSOFT12`, `LBL10`
- `WKL`: not used (different software environment, see `../gpu code/README.md`)
- `__huggingface_repos__.json`: exact revisions of the corpus and the model

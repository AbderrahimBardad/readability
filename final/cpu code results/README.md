# CPU-track results

Written by `../cpu code/barec_cpu_v2.py`. One file per method and sentence representation:

- `pooler__<METHOD>.json`: linear head on the pooled AraBERTv2 output
- `mean__<METHOD>.json`: linear head on the mean of the last-layer token vectors

Each file has `arm`, `group`, `representation`, `seeds`, `n_seeds`, `dev`, `test` (mean and standard
deviation over the 5 seeds) and `minutes`.

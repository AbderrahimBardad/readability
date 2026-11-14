# CPU-friendly track (frozen encoder)

`barec_cpu_v2.py` answers the "CPU-friendly training" requirement of the research topic.

1. AraBERTv2 is run once over the corpus, without training, and two vectors are stored for every
   sentence: the pooled output (`pooler`) and the mean of the last-layer token vectors (`mean`).
2. Only a linear head is trained on the stored vectors, which takes minutes on a laptop processor.
3. Every method is run with 5 seeds (42 to 46); results are reported as mean and standard deviation.

Methods: CE, MSE, OLL15, EMD, SOFT2 to SOFT5, CORAL, Soft-QWK (two phases), and CORAL with
zero-initialised cut-points (ablation).

Head settings: 50 epochs, learning rate 1e-3, batch 64, dropout 0.1; the rest as in the GPU scripts.

These numbers are much lower than fine-tuning and must not be compared with the BAREC paper.
Results are in `../cpu code results/`.

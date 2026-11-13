# Data provenance and known analysis limits
Synthetic results are generated from i.i.d. categorical states and Bernoulli hidden losses, with paired method-level outcomes. Real-label analyses use the public plotnine/ggplot2 `diamonds` dataset (53,940 rows); splits: train=35,061, development=5,394, evaluation=13,485. The development set fixes state cut points. The held-out evaluation outcomes were inspected at multiple error thresholds before the final threshold was selected. Consequently this is exploratory and not a pristine confirmation dataset.

Theoretical guarantees are about stochastic iid streams, not automatically about a finite fixed test pool sampled without replacement. Repeated order permutations are correlated. The 2 unsafe method-level rows in the seven-method benchmark correspond to **one** underlying world+seed false-certification event in related fixed methods.

Freeze hashes of canonical data in SHA256SUMS.txt. Public sharing still requires institutional/authorship review.

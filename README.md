# IGRM: Safe-Coverage Recovery under Partial Feedback

Research code and frozen results for a *Mathematics* manuscript (submission draft). Not the CAIS source code, and not a deployment-ready assurance component.

## Scope
Finite families of overlapping acceptance masks, Bernoulli conditional losses of a frozen model, unknown categorical context law from an independent unlabeled pilot, and a pathwise label cap. Method `mask_pooled_v19` requires **mask-complete** querying: while A is active, every arrival in A must be labeled. The mathematical guarantee **does not apply** to arbitrary within-mask, state-biased VOI queries.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.:v19_dependency:v19_dependency/scripts
pytest -q test_stronger_baselines.py v19_dependency/scripts/test_*.py
```
The exact recorded environment used numpy 2.3.5, pandas 2.2.3, scipy 1.17.0, scikit-learn 1.8.0, plotnine 0.15.3, matplotlib 3.10.8, and pytest 9.0.2. A separate Mac v0.10 historical execution used different versions; its CSV is preserved only as historical provenance.

## Reproduce analyses
```bash
export PYTHONPATH=.:v19_dependency:v19_dependency/scripts
python run_stronger_benchmark.py
python run_diamonds_holdout.py
python run_matched_fixed_real.py
python analyze_results.py
python plot_audit.py
```
`run_stronger_benchmark.py` reads frozen `prior_benchmark/benchmark_paired.csv` and recomputes two stronger methods, then joins seven methods. For a fresh run of the original five methods see `run_previous_benchmark.py` (see expected output and seed handling in source). The diamonds script obtains `plotnine.data.diamonds` locally and trains the predictor anew; no unrelated proprietary data.

## Primary evidence
- `results/all_paired_7_methods.csv`: 7 synthetic worlds x 200 paired seeds x 7 methods = 9,800 method-level rows, not 9,800 independent streams.
- `results/summary_7_methods.csv`: means and unsafe events.
- `results/diamonds/diamonds_order_sensitivity.csv`: 24 **dependent permutations** of the same 13,485 held-out examples, not 24 independent external experiments.
- `results/diamonds/frozen_heldout_annotations.csv`: derived labels for analysis of that finite pool.
- `results/historical/final_v10_30seeds.csv`: frozen older VOI benchmark, separate from v1.9 validity claims.

## Limits
The principal exact-binomial theorem assumes an iid stationary monitoring epoch; the `diamonds` finite no-replacement order permutations do not satisfy the theorem conditional on the observed pool. The outcome-error threshold used in the diamonds case was examined during exploratory analysis and the case study is **not prospectively untouched**. Fixed pooled CP is often more powerful near the safety boundary; predictable betting may use fewer labels in easy worlds. No claim is made that the method is globally optimal or that Fisher--Rao dominates nongeometric alternatives.


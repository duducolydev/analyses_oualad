# The Ranking Transfers, the Threshold Does Not — code and results

Code that reproduces every figure, table and number of the article
*The Ranking Transfers, the Threshold Does Not: Early At-Risk Prediction in
Unseen Courses* (D. Coly, E. H. B. Touré, E. H. M. Nguer) and of its
supplementary material, on the Open University Learning Analytics Dataset
(OULAD).

The study asks which parts of an early at-risk model survive the move to a
course it has never seen. It predicts withdrawal or failure from the first
10–50% of each presentation, holds out one module at a time (strict
leave-one-module-out: no student on both sides of the split), and separates
ranking from alert volume and calibration.

Code comments and the decision log are written in French.

## Data

OULAD is not included. Download it (CC-BY 4.0) from
<https://analyse.kmi.open.ac.uk/open_dataset> and place the seven CSV files
(`studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv`,
`studentAssessment.csv`, `assessments.csv`, `vle.csv`, `courses.csv`) in
`data/OULAD/` (ignored by git).

Please cite the dataset: Kuzilek, J., Hlosta, M., Zdrahal, Z. (2017). Open
University Learning Analytics dataset. *Scientific Data* 4, 170171.
https://doi.org/10.1038/sdata.2017.171

## Installation

Python 3.12. Exact versions are pinned in `requirements.txt`.

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
# PyTorch (CPU) is only needed for the GRU of the supplementary material:
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

Compiling the article requires a LaTeX distribution with `latexmk` and the
IEEEtran class.

## Reproducing everything

```
python -m pytest tests          # leakage and protocol tests (synthetic data)
python run_all.py               # all results, figures (FR/EN), tables, article and supplement
```

`run_all.py` takes about five hours on a 16-core machine with 16 GB of RAM.
`python run_all.py --liste` lists the steps; `--depuis N` resumes from step N;
`--sans-shap`, `--sans-sequences` and `--sans-graines` reuse the existing SHAP,
sequence-model and multi-seed results. All seeds are fixed and a full rerun
reproduces the numbers of the article exactly.

The article (`paper/main.tex`) contains no hand-typed result: every number is a
macro `\resXxx` defined in `paper/generated/macros.tex`, and every table is a
file of `paper/generated/`, both written by `src/08_paper_assets.py` from
`results/`.

## Repository layout

| Path | Content |
|---|---|
| `src/oulad_features.py` | Early-window features, population still enrolled at the cut-off, cohort normalizations (ranks, z-score, robust) |
| `src/oulad_sequences.py` | Weekly activity sequences and trends |
| `src/transfer_eval.py` | Generic protocols: random split grouped by individual (A), context hold-out with optional exclusion of shared individuals (B), models, Platt recalibration |
| `src/03_early_curve.py` | Predictions of (A) and (B) for five cut-offs |
| `src/04_transfer_analysis.py` | Difficulty, ranking cost, flag-rate drift, context descriptors |
| `src/05_shap_by_module.py`, `06_shap_stability.py` | SHAP of transferred and local models, drift decomposition |
| `src/07_fairness.py`, `18_fairness_fe.py` | Group gaps (pooled and within modules), fixed-effects regression, intersections |
| `src/09_sequence_models.py` | Gradient boosting and GRU |
| `src/10_overlap_check.py`, `13_overlap_control.py`, `19_size_control.py` | Strict vs standard hold-out, training-size control |
| `src/11_seed_variability.py`, `12_robustness.py` | Seeds, bootstrap intervals, calibration, threshold sensitivity |
| `src/14_remedies.py` | Remedies on the same students: Platt, intercept-only, top-q, EM prevalence correction, relative features |
| `src/15_normalisations.py`, `16_cohort_size.py` | Normalizations, logistic regression, calibrated unweighted forest, cohort size |
| `src/17_ap_reliability.py` | Average precision and reliability diagrams |
| `src/fig_*.py` | Figures (`--lang fr` for the thesis in `figures/`, `--lang en` for the article in `figures/en/`) |
| `src/08_paper_assets.py` | Numbers and tables of the article and supplement |
| `tests/` | Invariance test against temporal leakage; disjointness of both protocols |
| `results/` | Logs and small outputs (larger outputs are regenerated) |
| `paper/` | Article, supplementary material and bibliography |
| `DECISIONS.md` | Dated log of every methodological decision and its justification (in French) |

`src/01_build_features.py` and `02_train_evaluate.py` are command-line entry
points kept from the first version; `results/reference_rf_c25.txt` is the output
of that first version, used as the starting point of the methodological audit.

## Citation

See `CITATION.cff`.

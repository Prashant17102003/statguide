# StatGuide

A Streamlit app for statistical analysis of your own data, with a plain-language interpretation at every step.
It is a **statistics tool**, not a machine-learning product: the only ML-related part is the research code that
compares two models fairly (`tests_lib.py`, `simulation.py`, `power.py`, `compare_real.py`).

## What the app does
Upload a CSV, Excel, text or JSON file, then use the sections:

| Section | What you get |
|---|---|
| Data | preview, column types, missing values, summary statistics |
| EDA | missing-value map, histogram + boxplot + Q-Q plot, correlation heatmap, scatter and group plots |
| Test advisor | normality screen, a **per-group normality check**, and a suggested test for your goal |
| PCA | components, loadings, variable importance, biplot, KMO / Bartlett; optional log(1+x) for skewed variables |
| Regression | numeric **and categorical (dummy-coded)** predictors, fitted line, diagnostics, optional **hold-out test** |
| Two-group test | Welch t-test (+ Mann-Whitney check), CI, Cohen's d, rank-biserial |
| Paired / one-sample | paired t / Wilcoxon, one-sample t / Wilcoxon, Friedman test for 3+ related columns |
| ANOVA | Welch ANOVA, Kruskal-Wallis, Holm-adjusted pairwise tests |
| Chi-square | chi-square, Cramer's V, Fisher (2x2), adjusted standardised residuals |
| Report | one report as Excel, PDF or Word |

## Run locally
```
pip install -r requirements.txt              # app only
streamlit run app.py
pip install -r requirements-dev.txt          # optional: run the checks below
pytest -q                                    # or: python tests/test_against_scipy.py
pip install -r requirements-research.txt     # only for simulation.py / power.py / compare_real.py
```
Try it with `data/breast_cancer_sample.csv`, `data/synthetic_mixed_with_nulls.csv` or `data/iris_3class_with_nulls.csv`.
The look and feel lives in `theme.py` and `.streamlit/config.toml` (the app is locked to a light theme so text stays readable).

## How the numbers were checked
`tests/test_against_scipy.py` compares the hand-written statistics with scipy, numpy and scikit-learn
(Welch t, paired and one-sample t, Wilcoxon, Friedman, ANOVA, Kruskal-Wallis, Holm, chi-square, regression, PCA,
Nadeau-Bengio and Bayesian correlated t-tests) and checks the dummy coding and the hold-out split.
Not cross-checked against a second library: Welch ANOVA (only for k = 2), Breusch-Pagan, KMO / Bartlett, Cook's distance.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app |
| `theme.py`, `.streamlit/config.toml` | look and feel, light theme |
| `io_tools.py` | reads CSV, delimited text, Excel and JSON |
| `eda_tools.py`, `advisor_tools.py`, `pca_tools.py`, `regression_tools.py`, `stats_tools.py` | the analysis code |
| `report_tools.py` | builds the report and writes Excel, PDF and Word files |
| `tests_lib.py`, `simulation.py`, `power.py`, `compare_real.py`, `mixed_data.py` | research code: fair comparison of two models |
| `tests/` | numerical checks |
| `data/`, `make_sample_data.py` | sample datasets and the script that regenerates the synthetic ones |

## Limitations (please read)
- Interpretations are rules of thumb. Always check the plots and the assumptions yourself.
- The whole-dataset normality screen is only a rough guide; normality matters per group or for model residuals.
- Regression is ordinary least squares: linear effects, independent errors. No interactions, no regularisation.
- PCA fills missing values with the median unless you drop rows, and works on Pearson correlations.
- Repeated-measures ANOVA and Dunn's post-hoc test are not included (Friedman and Holm-adjusted Wilcoxon are).
- Do not upload personal or confidential data to a public deployment.
- Corrected and Bayesian tests follow Nadeau & Bengio (2003) and Corani & Benavoli (2015); verify the formulas against the papers.
- Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository; check the licence and cite the source.
  `synthetic_mixed_with_nulls.csv` is synthetic; `iris_3class_with_nulls.csv` is Iris with 8% of values removed at random.

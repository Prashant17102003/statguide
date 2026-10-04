"""Test advisor: assess normality for the whole dataset, then suggest parametric or non-parametric tests.

Rules of thumb only. Normality matters for each group (or the model residuals), not for the dataset as a whole:
the whole-dataset summary is a rough screen, and group_normality() gives the check that t-tests / ANOVA actually need.
Treat every result as guidance, not a verdict.
"""
import numpy as np
import pandas as pd
from scipy import stats

GOALS = ["Compare 2 independent groups", "Compare 2 paired / related measurements",
         "Compare 3+ independent groups", "Compare 3+ related / repeated measurements",
         "Relationship between two numeric variables", "Association between two categorical variables",
         "Compare one sample with a known value"]


def _fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _assess(x, alpha):
    """Normality verdict for one sample: Shapiro-Wilk p up to 200 values, skewness/kurtosis rule above that."""
    n = len(x)
    if n < 8 or x.std() == 0:
        return {"n": n, "test": "-", "p-value": "-", "skewness": np.nan, "excess kurtosis": np.nan,
                "rule": "too few values", "verdict": "Not assessed"}
    name, p = ("Shapiro-Wilk", stats.shapiro(x).pvalue) if n <= 5000 else ("D'Agostino K2", stats.normaltest(x).pvalue)
    sk, ku = stats.skew(x, bias=False), stats.kurtosis(x, bias=False)
    if n <= 200:
        ok, rule = p >= alpha, f"p >= {alpha}"
    else:
        ok, rule = (abs(sk) < .5 and abs(ku) < 1), "|skew| < 0.5 and |kurtosis| < 1 (large n)"
    return {"n": n, "test": name, "p-value": _fmt_p(p), "skewness": round(sk, 2), "excess kurtosis": round(ku, 2),
            "rule": rule, "verdict": "Normal-like" if ok else "Not normal"}


def normality_table(df, alpha=0.05):
    rows = []
    for c in df.select_dtypes("number").columns:
        x = df[c].dropna().to_numpy(float)
        if len(np.unique(x)) <= 2:
            continue                                   # binary columns are not tested for normality
        rows.append({"variable": c, **_assess(x, alpha)})
    return pd.DataFrame(rows)


def group_normality(df, value, group, alpha=0.05):
    """Normality check of one numeric variable inside each level of a grouping column (what t-tests / ANOVA actually need)."""
    d = df[[value, group]].dropna(); rows = []
    for lv, g in d.groupby(group):
        rows.append({"group": str(lv), **_assess(g[value].to_numpy(float), alpha)})
    return pd.DataFrame(rows)


def overall(tab, threshold_pct):
    t = tab[tab.verdict != "Not assessed"]
    if t.empty:
        return None, 0.0
    share = 100 * (t.verdict == "Normal-like").mean()
    return share >= threshold_pct, share


_R = {
 GOALS[0]: {"n": dict(test="Welch's t-test (Student's t-test if variances are equal)", alt="Independent-samples t-test",
                      post="-", check="Group sizes, outliers, Levene's test for equal variances",
                      where="Two-group test tab (Welch t-test)"),
            "x": dict(test="Mann-Whitney U test (Wilcoxon rank-sum)", alt="Permutation test; Brunner-Munzel if spreads differ",
                      post="-", check="Similar shape in both groups if you want to compare medians",
                      where="Two-group test tab (Mann-Whitney is shown as the sensitivity check)")},
 GOALS[1]: {"n": dict(test="Paired t-test", alt="-", post="-", check="Differences (after minus before) should look normal; no extreme outliers",
                      where="Paired / one-sample tab (paired t-test)"),
            "x": dict(test="Wilcoxon signed-rank test", alt="Sign test (if differences are very skewed or tied)", post="-",
                      check="Differences should be roughly symmetric", where="Paired / one-sample tab (Wilcoxon signed-rank is shown as the sensitivity check)")},
 GOALS[2]: {"n": dict(test="One-way ANOVA (Welch ANOVA if variances are unequal)", alt="-",
                      post="Tukey HSD (Games-Howell if variances are unequal); or Welch t-tests with Holm correction",
                      check="Residual normality, Levene's test, group sizes", where="ANOVA tab (Welch ANOVA + Holm pairwise)"),
            "x": dict(test="Kruskal-Wallis H test", alt="Median test (less powerful)", post="Dunn's test with Holm or Bonferroni correction",
                      check="Similar shapes if you want to compare medians", where="ANOVA tab (Kruskal-Wallis shown as sensitivity check)")},
 GOALS[3]: {"n": dict(test="Repeated-measures ANOVA", alt="Linear mixed-effects model", post="Paired t-tests with Holm or Bonferroni correction",
                      check="Sphericity (Mauchly); use Greenhouse-Geisser correction if violated",
                      where="Not in the app yet (statsmodels AnovaRM)"),
            "x": dict(test="Friedman test", alt="-", post="Nemenyi test, or Wilcoxon signed-rank tests with Holm correction",
                      check="Complete data for every subject", where="Paired / one-sample tab (Friedman + Holm-adjusted Wilcoxon pairs)")},
 GOALS[4]: {"n": dict(test="Pearson correlation", alt="Simple linear regression", post="-",
                      check="Linear relationship (scatter plot), no extreme outliers, roughly bivariate normal",
                      where="EDA tab, section 3 (Pearson and Spearman); Regression section for the fitted line"),
            "x": dict(test="Spearman rank correlation", alt="Kendall's tau (small n or many ties)", post="-",
                      check="Monotonic relationship (scatter plot)", where="EDA tab, section 3 (Spearman shown)")},
 GOALS[5]: {k: dict(test="Chi-square test of independence (Fisher's exact test if expected counts are below 5)",
                    alt="Cramer's V as effect size; McNemar's test for paired binary data", post="Standardised residuals to find which cells differ",
                    check="Expected count of at least 5 in most cells; independent observations",
                    where="Chi-square tab") for k in ("n", "x")},
 GOALS[6]: {"n": dict(test="One-sample t-test", alt="-", post="-", check="Normality of the values, no extreme outliers",
                      where="Paired / one-sample tab (one-sample t-test)"),
            "x": dict(test="One-sample Wilcoxon signed-rank test (against a median)", alt="Sign test", post="-",
                      check="Roughly symmetric distribution", where="Paired / one-sample tab (Wilcoxon signed-rank is shown as the sensitivity check)")},
}


def recommend(goal, normal):
    r = dict(_R[goal]["n" if normal else "x"])
    r["family"] = "Parametric" if normal else "Non-parametric"
    if goal == GOALS[5]:
        r["family"] = "Not affected by normality (categorical data)"
    return r

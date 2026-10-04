"""Checks the hand-written statistics against scipy / numpy / scikit-learn.

Run with:  pytest -q        (or: python tests/test_against_scipy.py  if pytest is not installed)
"""
import os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stats_tools as S, regression_tools as RG, pca_tools as PC, tests_lib as T, advisor_tools as A

RNG = np.random.default_rng(12345)


def test_welch_matches_scipy():
    x, y = RNG.normal(0, 1, 40), RNG.normal(.5, 2, 55)
    r = S.two_group_test(x, y); t = stats.ttest_ind(x, y, equal_var=False); ci = t.confidence_interval()
    assert np.isclose(r["welch_p"], t.pvalue) and np.isclose(r["mean_diff_CI"][0], ci.low) and np.isclose(r["mean_diff_CI"][1], ci.high)
    assert np.isclose(r["mannwhitney_p"], stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def test_paired_matches_scipy():
    a = RNG.normal(10, 2, 30); b = a + RNG.normal(.4, 1, 30); a[3] = np.nan
    r = S.paired_test(a, b); ok = ~np.isnan(a)
    t = stats.ttest_rel(a[ok], b[ok]); ci = t.confidence_interval()
    assert r["n"] == ok.sum() and np.isclose(r["p_t"], t.pvalue)
    assert np.isclose(r["ci"][0], ci.low) and np.isclose(r["ci"][1], ci.high)
    assert np.isclose(r["p_wilcoxon"], stats.wilcoxon(a[ok] - b[ok]).pvalue)
    assert -1 <= r["rank_biserial"] <= 1


def test_one_sample_matches_scipy():
    x = RNG.normal(5.3, 1.2, 35)
    r = S.one_sample_test(x, 5.0); t = stats.ttest_1samp(x, 5.0); ci = t.confidence_interval()
    assert np.isclose(r["p_t"], t.pvalue) and np.isclose(r["ci"][0], ci.low) and np.isclose(r["ci"][1], ci.high)
    assert np.isclose(r["p_wilcoxon"], stats.wilcoxon(x - 5.0).pvalue)


def test_friedman_matches_scipy():
    base = RNG.normal(size=25)[:, None]
    d = pd.DataFrame(base + np.array([0, .3, .8]) + RNG.normal(0, .5, (25, 3)), columns=["t1", "t2", "t3"])
    r = S.friedman_test(d); chi2, p = stats.friedmanchisquare(d.t1, d.t2, d.t3)
    assert np.isclose(r["chi2"], chi2) and np.isclose(r["p"], p) and np.isclose(r["kendall_w"], chi2 / (25 * 2))
    assert len(r["pairs"]) == 3


def test_anova_kruskal_holm():
    g = {"A": RNG.normal(0, 1, 30), "B": RNG.normal(.3, 2, 40), "C": RNG.normal(.8, 1.5, 35)}
    r = S.multi_group_test(g)
    assert np.isclose(r["p_anova"], stats.f_oneway(*g.values()).pvalue) and np.isclose(r["p_kw"], stats.kruskal(*g.values()).pvalue)
    assert np.allclose(S.holm([.01, .04, .03]), [.03, .06, .06])


def test_welch_anova_reduces_to_welch_t_for_two_groups():
    # for k = 2 the Welch ANOVA has the same p-value as the Welch t-test (F = t^2)
    x, y = RNG.normal(0, 1, 30), RNG.normal(.6, 2.2, 45)
    r = S.multi_group_test({"A": x, "B": y}); t = stats.ttest_ind(x, y, equal_var=False)
    assert np.isclose(r["p_welch"], t.pvalue)


def test_chi_square_and_adjusted_residuals():
    a, b = RNG.choice(list("abc"), 300), RNG.choice(list("xyz"), 300)
    r = S.chi_square_test(a, b); tab = pd.crosstab(a, b); c = stats.chi2_contingency(tab)
    assert np.isclose(r["chi2"], c[0]) and np.isclose(r["p"], c[1])
    assert np.isclose(r["cramers_v"], np.sqrt(c[0] / (300 * 2)))
    # adjusted residuals are never smaller in size than plain Pearson residuals (their denominator is smaller)
    pearson = (tab.values - c[3]) / np.sqrt(c[3])
    assert (np.abs(r["residuals"].to_numpy()) + .01 >= np.abs(pearson)).all()
    assert "fisher_p" in S.chi_square_test(RNG.choice(list("ab"), 80), RNG.choice(list("xy"), 80))
    # for a 2x2 table the adjusted residual squared equals the (uncorrected) chi-square statistic
    a2, b2 = RNG.choice(list("ab"), 200, p=[.4, .6]), RNG.choice(list("xy"), 200); b2 = np.where((a2 == "a") & (RNG.random(200) < .35), "x", b2)
    r2 = S.chi_square_test(a2, b2); nc = stats.chi2_contingency(pd.crosstab(a2, b2), correction=False)[0]
    assert np.isclose(r2["residuals"].iloc[0, 0] ** 2, nc, atol=0.06)


def test_regression_matches_linregress_and_lstsq():
    a = RNG.normal(size=300); b = 2 + 1.5 * a + RNG.normal(size=300)
    f = RG.fit_ols(pd.DataFrame({"a": a, "b": b}), "b", ["a"]); lr = stats.linregress(a, b)
    assert np.isclose(f["beta"][1], lr.slope) and np.isclose(f["coef"].loc["a", "p-value"], lr.pvalue) and np.isclose(f["r2"], lr.rvalue ** 2)
    c = RNG.normal(size=300); df = pd.DataFrame({"a": a, "c": c, "y": 1 + a - c + RNG.normal(size=300)})
    f2 = RG.fit_ols(df, "y", ["a", "c"]); X = np.column_stack([np.ones(300), a, c])
    assert np.allclose(f2["beta"], np.linalg.lstsq(X, df.y, rcond=None)[0])


def test_dummy_coding_recovers_group_effects_and_keeps_missing():
    n = 500; g = RNG.choice(["a", "b", "c"], n).astype(object); g[7] = None
    df = pd.DataFrame({"x": RNG.normal(size=n), "g": g}); df["y"] = 1 + 2 * df.x + np.where(df.g == "b", 1.5, 0) + np.where(df.g == "c", -1, 0) + RNG.normal(0, .5, n)
    d2, new = RG.add_dummies(df, ["g"]); assert new == ["g=b", "g=c"] and d2.loc[7, new].isna().all()
    f = RG.fit_ols(d2, "y", ["x"] + new); assert f["n"] == n - 1
    assert abs(f["coef"].loc["g=b", "estimate"] - 1.5) < .2 and abs(f["coef"].loc["g=c", "estimate"] + 1) < .2


def test_holdout_is_honest():
    n = 400; x = RNG.normal(size=n); df = pd.DataFrame({"x": x, "noise": RNG.normal(size=n), "y": 3 * x + RNG.normal(size=n)})
    h = RG.holdout_eval(df, "y", ["x"]); assert h["n_train"] + h["n_test"] == n and h["r2_test"] > .7
    h0 = RG.holdout_eval(df.assign(y=RNG.normal(size=n)), "y", ["noise"]); assert h0["r2_test"] < .1          # no signal -> no test R2
    assert RG.holdout_eval(df, "y", ["x"], seed=1)["rmse_test"] != RG.holdout_eval(df, "y", ["x"], seed=2)["rmse_test"]


def test_pca_matches_numpy_and_sklearn():
    d = pd.DataFrame(RNG.normal(size=(200, 5)), columns=list("abcde")); d["b"] = d.a * .8 + RNG.normal(size=200) * .3
    r = PC.run_pca(d, list(d.columns)); w = np.linalg.eigvalsh(np.corrcoef(d.to_numpy().T))[::-1]
    assert np.allclose(r["eig"], w)
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        assert np.allclose(r["ratio"], PCA().fit(StandardScaler().fit_transform(d)).explained_variance_ratio_)
    except ImportError:
        pass


def test_pca_log_option_only_touches_skewed_nonnegative_columns():
    d = pd.DataFrame({"a": RNG.normal(size=300), "b": RNG.exponential(size=300) ** 2, "c": RNG.normal(size=300), "d": RNG.normal(size=300)})
    r = PC.run_pca(d, list(d.columns), True, True); assert r["logged"] == ["b"]
    assert PC.run_pca(d, list(d.columns), True, False)["logged"] == []


def test_model_comparison_tests():
    a, b = RNG.normal(.9, .02, 10), RNG.normal(.89, .02, 10)
    t, p = T.naive_ttest(a, b); assert np.isclose(p, stats.ttest_rel(a, b).pvalue)
    tc, pc = T.corrected_ttest(a, b, 90, 10); assert pc > p                                   # correction is more conservative
    pr = T.bayes_correlated_ttest(a, b, 90, 10); assert np.isclose(sum(pr.values()), 1.0)


def test_normality_helpers():
    df = pd.DataFrame({"g": ["u"] * 150 + ["v"] * 150, "x": np.r_[RNG.normal(size=150), RNG.exponential(size=150)]})
    gt = A.group_normality(df, "x", "g"); assert list(gt.group) == ["u", "v"] and gt.verdict.iloc[1] == "Not normal"
    assert A.normality_table(pd.DataFrame({"x": RNG.exponential(size=5000)})).iloc[0]["p-value"] == "<0.0001"


if __name__ == "__main__":                       # tiny runner so the file also works without pytest
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:
                failed += 1; print("FAIL", name, "->", repr(e))
    sys.exit(1 if failed else 0)

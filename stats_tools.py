"""Two-group comparison helper for the app.

Default = Welch t-test (does not assume equal variances). Normality and variance
checks are SHOWN as information, and Mann-Whitney is reported alongside as a
sensitivity check. The user makes the final decision.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats


def two_group_test(x, y, alpha=0.05):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    m1, m2 = x.mean(), y.mean()
    v1, v2 = x.var(ddof=1), y.var(ddof=1)

    # Welch t-test, CI and effect size computed by hand (so you can verify them)
    se = np.sqrt(v1 / n1 + v2 / n2)
    df = se**4 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t = (m1 - m2) / se
    p_welch = 2 * stats.t.sf(abs(t), df)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ci = ((m1 - m2) - tcrit * se, (m1 - m2) + tcrit * se)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    cohen_d = (m1 - m2) / sp

    # Mann-Whitney and rank-biserial effect size
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    rank_biserial = 2 * u.statistic / (n1 * n2) - 1

    # Assumption information (not a gate)
    sh1 = stats.shapiro(x).pvalue if 3 <= n1 <= 5000 else np.nan
    sh2 = stats.shapiro(y).pvalue if 3 <= n2 <= 5000 else np.nan
    lev = stats.levene(x, y).pvalue

    return {
        "n1": n1, "n2": n2, "mean1": m1, "mean2": m2,
        "welch_t": t, "welch_df": df, "welch_p": p_welch,
        "mean_diff_CI": ci, "cohen_d": cohen_d,
        "mannwhitney_p": u.pvalue, "rank_biserial": rank_biserial,
        "shapiro_p_group1": sh1, "shapiro_p_group2": sh2, "levene_p": lev,
    }


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def multi_group_test(groups):
    """groups: dict name -> values (3+ groups). Classic ANOVA, Welch ANOVA, Kruskal-Wallis, Holm pairwise Welch."""
    g = {k: np.asarray(v, float) for k, v in groups.items()}
    g = {k: v[~np.isnan(v)] for k, v in g.items()}
    g = {k: v for k, v in g.items() if len(v) >= 2}
    names = list(g); k = len(names); xs = [g[n] for n in names]
    n = np.array([len(x) for x in xs]); m = np.array([x.mean() for x in xs])
    v = np.array([x.var(ddof=1) for x in xs]); allx = np.concatenate(xs); gm = allx.mean()
    eta2 = (n * (m - gm) ** 2).sum() / ((allx - gm) ** 2).sum()
    F, p_f = stats.f_oneway(*xs)
    if (v > 0).all():
        w = n / v; W = w.sum(); mw = (w * m).sum() / W; tmp = ((1 - w / W) ** 2 / (n - 1)).sum()
        Fw = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
        df2 = (k ** 2 - 1) / (3 * tmp); p_w = stats.f.sf(Fw, k - 1, df2)
    else:
        Fw, df2, p_w = np.nan, np.nan, np.nan
    H, p_kw = stats.kruskal(*xs)
    pairs = [(names[i], names[j], m[i] - m[j], stats.ttest_ind(xs[i], xs[j], equal_var=False).pvalue)
             for i, j in combinations(range(k), 2)]
    adj = holm([q[3] for q in pairs])
    return {
        "k": k, "eta2": eta2, "p_anova": p_f, "p_welch": p_w, "p_kw": p_kw,
        "levene_p": stats.levene(*xs).pvalue,
        "groups": pd.DataFrame({"n": n, "mean": m.round(3), "sd": np.sqrt(v).round(3)}, index=names),
        "tests": pd.DataFrame({"Test": ["Classic one-way ANOVA", "Welch ANOVA (default)", "Kruskal-Wallis (sensitivity)"],
                               "Statistic": [f"F = {F:.3f}", f"F = {Fw:.3f}, df2 = {df2:.1f}", f"H = {H:.3f}"],
                               "p-value": [fmt_p(p_f), fmt_p(p_w), fmt_p(p_kw)]}),
        "pairs": pd.DataFrame([{"A": a, "B": b, "mean diff (A-B)": round(d, 3), "p (raw)": round(p, 4), "p (Holm)": round(q, 4)}
                               for (a, b, d, p), q in zip(pairs, adj)]),
    }


def interpret_anova(r):
    p = r["p_welch"] if not np.isnan(r["p_welch"]) else r["p_anova"]
    size = "small" if r["eta2"] < .06 else "medium" if r["eta2"] < .14 else "large"
    o = [f"Welch ANOVA p = {fmt_p(p)}: " + ("evidence that at least one group mean differs." if p < .05
         else "no evidence that the group means differ.") + f" Effect size eta-squared = {r['eta2']:.3f} ({size} by the usual rule of thumb)."]
    if r["levene_p"] < .05:
        o.append(f"Levene p = {fmt_p(r['levene_p'])}: variances look unequal, so prefer Welch ANOVA over the classic one.")
    if (r["p_kw"] < .05) != (p < .05):
        o.append("Kruskal-Wallis and Welch ANOVA disagree: check the distributions (EDA tab) before concluding.")
    sig = r["pairs"][r["pairs"]["p (Holm)"] < .05]
    o.append("Pairs that differ after Holm adjustment: " + (", ".join(f"{a} vs {b}" for a, b in zip(sig.A, sig.B)) if len(sig) else "none") + ".")
    return o


def chi_square_test(a, b):
    tab = pd.crosstab(a, b)
    chi2, p, dof, exp = stats.chi2_contingency(tab)          # Yates correction is applied automatically for 2x2
    chi2_nc = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum(); v = np.sqrt(chi2_nc / (n * (min(tab.shape) - 1)))
    rs, cs = tab.values.sum(1, keepdims=True), tab.values.sum(0, keepdims=True)      # adjusted standardised residuals (Haberman)
    adj = (tab.values - exp) / np.sqrt(exp * (1 - rs / n) * (1 - cs / n))
    res = pd.DataFrame(adj, index=tab.index, columns=tab.columns).round(2)
    out = {"table": tab, "chi2": chi2, "p": p, "dof": dof, "n": n, "cramers_v": v,
           "min_expected": exp.min(), "pct_low": 100 * (exp < 5).mean(), "residuals": res}
    if tab.shape == (2, 2):
        out["odds_ratio"], out["fisher_p"] = stats.fisher_exact(tab.values)
    return out


def interpret_chi(r):
    v = r["cramers_v"]; size = "negligible" if v < .1 else "weak" if v < .3 else "moderate" if v < .5 else "strong"
    o = [f"Chi-square = {r['chi2']:.3f}, df = {r['dof']}, p = {fmt_p(r['p'])}: " +
         ("evidence of an association." if r["p"] < .05 else "no evidence of an association.") +
         f" Cramer's V = {v:.3f} ({size}, rule of thumb)."]
    if r["min_expected"] < 5:
        o.append(f"Warning: smallest expected count is {r['min_expected']:.2f} ({r['pct_low']:.0f}% of cells below 5). "
                 "The chi-square approximation may be unreliable; merge sparse categories or use Fisher's exact test.")
    if "fisher_p" in r:
        o.append(f"2x2 table: Fisher's exact p = {fmt_p(r['fisher_p'])}, odds ratio = {r['odds_ratio']:.3f} "
                 "(chi-square p uses Yates continuity correction).")
    rs = r["residuals"].stack(); big = rs[rs.abs() > 2]
    o.append("Cells far from independence (|adjusted standardised residual| > 2): " +
             (", ".join(f"{i[0]} / {i[1]} ({x:+.1f})" for i, x in big.items()) if len(big) else "none") + ".")
    return o


# ----------------------------------------------------------------------------------------------
# Paired, one-sample and repeated-measures (Friedman) tests
# ----------------------------------------------------------------------------------------------
def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def _signed_rank_effect(d):
    """Matched-pairs rank-biserial correlation from the non-zero differences (positive = first larger)."""
    nz = d[d != 0]
    if len(nz) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(nz)); wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return (wp - wm) / (wp + wm)


def _wilcoxon_p(d):
    return stats.wilcoxon(d).pvalue if (d != 0).any() else np.nan


def paired_test(x, y, alpha=0.05):
    """Paired t-test on the differences x - y (+ Wilcoxon signed-rank as sensitivity check). Rows with a missing value are dropped."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y)); x, y = x[keep], y[keep]; d = x - y; n = len(d)
    if n < 3:
        raise ValueError("Need at least 3 complete pairs.")
    sd = d.std(ddof=1)
    if sd == 0:
        raise ValueError("All differences are identical, so the test is undefined.")
    se = sd / np.sqrt(n); t = d.mean() / se; df = n - 1; tc = stats.t.ppf(1 - alpha / 2, df)
    return {"n": n, "mean1": x.mean(), "mean2": y.mean(), "mean_diff": d.mean(), "sd_diff": sd, "t": t, "df": df,
            "p_t": 2 * stats.t.sf(abs(t), df), "ci": (d.mean() - tc * se, d.mean() + tc * se), "cohen_dz": d.mean() / sd,
            "p_wilcoxon": _wilcoxon_p(d), "rank_biserial": _signed_rank_effect(d), "n_zero": int((d == 0).sum()),
            "shapiro_p_diff": stats.shapiro(d).pvalue if n <= 5000 else np.nan}


def one_sample_test(x, mu0, alpha=0.05):
    """One-sample t-test against mu0 (+ Wilcoxon signed-rank against mu0 as sensitivity check)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 values.")
    sd = x.std(ddof=1)
    if sd == 0:
        raise ValueError("All values are identical, so the test is undefined.")
    se = sd / np.sqrt(n); t = (x.mean() - mu0) / se; df = n - 1; tc = stats.t.ppf(1 - alpha / 2, df); d = x - mu0
    return {"n": n, "mean": x.mean(), "median": float(np.median(x)), "sd": sd, "mu0": mu0, "t": t, "df": df,
            "p_t": 2 * stats.t.sf(abs(t), df), "ci": (x.mean() - tc * se, x.mean() + tc * se), "cohen_d": (x.mean() - mu0) / sd,
            "p_wilcoxon": _wilcoxon_p(d), "rank_biserial": _signed_rank_effect(d), "n_zero": int((d == 0).sum()),
            "shapiro_p": stats.shapiro(x).pvalue if n <= 5000 else np.nan}


def friedman_test(data):
    """Friedman test for k >= 3 related measurements (DataFrame, one column per condition, one row per subject)."""
    d = data.apply(pd.to_numeric, errors="coerce").dropna(); n, k = d.shape
    if k < 3:
        raise ValueError("Choose at least 3 columns.")
    if n < 5:
        raise ValueError("Need at least 5 complete rows (subjects).")
    chi2, p = stats.friedmanchisquare(*[d[c].to_numpy() for c in d.columns])
    names = list(d.columns); rows = []
    for i, j in combinations(range(k), 2):
        diff = d.iloc[:, i].to_numpy() - d.iloc[:, j].to_numpy()
        rows.append((names[i], names[j], float(np.median(diff)), _wilcoxon_p(diff) if (diff != 0).any() else 1.0))
    adj = holm([r[3] for r in rows])
    return {"n": n, "k": k, "chi2": chi2, "p": p, "kendall_w": chi2 / (n * (k - 1)),
            "summary": pd.DataFrame({"median": d.median().round(3), "mean": d.mean().round(3), "mean rank": d.rank(axis=1).mean().round(2)}),
            "pairs": pd.DataFrame([{"A": a, "B": b, "median diff (A-B)": round(m, 3), "p (raw)": round(q, 4), "p (Holm)": round(h, 4)}
                                   for (a, b, m, q), h in zip(rows, adj)])}


def _eff_label(v, cuts=(.1, .3, .5)):
    v = abs(v); return "negligible" if v < cuts[0] else "small" if v < cuts[1] else "medium" if v < cuts[2] else "large"


def interpret_paired(r, a="first", b="second"):
    p = r["p_t"]; o = [f"Paired t-test p = {fmt_p(p)}: " + (f"evidence that the mean of {a} differs from {b}" if p < .05 else f"no evidence that the means of {a} and {b} differ") +
                       f". Mean difference ({a} - {b}) = {r['mean_diff']:.4g}, 95% CI {r['ci'][0]:.4g} to {r['ci'][1]:.4g}."]
    dz = abs(r["cohen_dz"]); o.append(f"Effect size Cohen's dz = {r['cohen_dz']:.3f} ({'small' if dz < .5 else 'medium' if dz < .8 else 'large'} by the usual rule of thumb, 0.2 / 0.5 / 0.8).")
    pw = r["p_wilcoxon"]
    if not np.isnan(pw):
        o.append(f"Wilcoxon signed-rank p = {fmt_p(pw)}, matched-pairs rank-biserial = {r['rank_biserial']:.3f}." + (" The two tests disagree: look at the differences before concluding." if (pw < .05) != (p < .05) else ""))
    if r["n_zero"]:
        o.append(f"{r['n_zero']} pair(s) have zero difference; Wilcoxon ignores them.")
    sp = r["shapiro_p_diff"]
    if not np.isnan(sp):
        o.append(f"Shapiro-Wilk on the differences: p = {fmt_p(sp)}" + (" (evidence against normality; prefer Wilcoxon, and check a histogram of the differences)." if sp < .05 else " (no evidence against normality)."))
    return o


def interpret_one_sample(r):
    p = r["p_t"]; o = [f"One-sample t-test p = {fmt_p(p)}: " + (f"evidence that the mean differs from {r['mu0']:g}" if p < .05 else f"no evidence that the mean differs from {r['mu0']:g}") +
                       f". Mean = {r['mean']:.4g}, 95% CI {r['ci'][0]:.4g} to {r['ci'][1]:.4g}."]
    d = abs(r["cohen_d"]); o.append(f"Effect size Cohen's d = {r['cohen_d']:.3f} ({'small' if d < .5 else 'medium' if d < .8 else 'large'} by the usual rule of thumb).")
    pw = r["p_wilcoxon"]
    if not np.isnan(pw):
        o.append(f"Wilcoxon signed-rank p = {fmt_p(pw)} (tests the median; assumes a roughly symmetric distribution)." + (" The two tests disagree: check the distribution." if (pw < .05) != (p < .05) else ""))
    sp = r["shapiro_p"]
    if not np.isnan(sp):
        o.append(f"Shapiro-Wilk p = {fmt_p(sp)}" + (" (evidence against normality; check the Q-Q plot, and prefer Wilcoxon if the data are skewed and n is small)." if sp < .05 else " (no evidence against normality)."))
    return o


def interpret_friedman(r):
    p = r["p"]; w = r["kendall_w"]
    o = [f"Friedman chi-square = {r['chi2']:.3f}, df = {r['k'] - 1}, p = {fmt_p(p)}: " + ("evidence that at least one condition differs." if p < .05 else "no evidence that the conditions differ.") +
         f" Kendall's W = {w:.3f} ({_eff_label(w, (.1, .3, .5))} agreement/effect, rule of thumb)."]
    sig = r["pairs"][r["pairs"]["p (Holm)"] < .05]
    o.append("Pairs that differ after Holm adjustment (Wilcoxon signed-rank): " + (", ".join(f"{a} vs {b}" for a, b in zip(sig.A, sig.B)) if len(sig) else "none") + ".")
    if r["n"] < 10:
        o.append("Fewer than 10 subjects: the chi-square approximation is rough.")
    return o

"""EDA helpers: figures + plain-language interpretation (rules of thumb, not proofs)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def split_columns(df):
    num = [c for c in df.select_dtypes("number").columns if df[c].nunique(dropna=True) > 2]
    cat = [c for c in df.columns if c not in num and df[c].nunique(dropna=True) <= 20]
    return num, cat


def summarize(s):
    s = pd.to_numeric(s, errors="coerce")
    x = s.dropna().to_numpy(float)
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    out = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()
    return dict(n=len(x), mean=x.mean(), median=float(np.median(x)), sd=x.std(ddof=1),
                skew=stats.skew(x, bias=False), kurt=stats.kurtosis(x, bias=False), n_out=int(out),
                shapiro_p=stats.shapiro(x).pvalue if 3 <= len(x) <= 5000 else np.nan)


def interpret_numeric(name, d):
    sk, k, p = d["skew"], d["kurt"], d["shapiro_p"]
    o = []
    if abs(sk) < 0.5:
        o.append(f"**Shape:** roughly symmetric (skewness {sk:.2f}).")
    else:
        side = "right (long upper tail)" if sk > 0 else "left (long lower tail)"
        o.append(f"**Shape:** {'moderately' if abs(sk) < 1 else 'strongly'} skewed to the {side} "
                 f"(skewness {sk:.2f}); mean {d['mean']:.3g} vs median {d['median']:.3g}.")
    o.append("**Tails:** " + (f"heavier than normal (excess kurtosis {k:.2f}), so extreme values are more likely." if k > 1
             else f"lighter than normal, flatter (excess kurtosis {k:.2f})." if k < -1
             else f"close to normal (excess kurtosis {k:.2f})."))
    o.append(f"**Outliers (1.5xIQR rule):** {d['n_out']} value(s), {100 * d['n_out'] / d['n']:.1f}%. " +
             ("Check whether they are errors or genuine extreme cases before removing them." if d["n_out"] else "None flagged."))
    if np.isnan(p):
        o.append("**Shapiro-Wilk:** not computed (needs 3 to 5000 values).")
    elif p < 0.05:
        o.append(f"**Shapiro-Wilk:** p = {p:.4f}, evidence against normality. With large n even tiny deviations are flagged, so judge mainly by the Q-Q plot.")
    else:
        o.append(f"**Shapiro-Wilk:** p = {p:.4f}, no evidence against normality (this does not prove normality).")
    if abs(sk) >= 1 or (not np.isnan(p) and p < 0.05 and abs(sk) >= 0.5):
        o.append("**Suggestion:** consider a log or square-root transformation (positive values only) or rank-based tests such as Mann-Whitney.")
    else:
        o.append("**Suggestion:** a mean-based method such as the Welch t-test looks reasonable for this variable.")
    return o


def dist_figures(s, name):
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    f1, a = plt.subplots(figsize=(4, 3.2))
    a.hist(x, bins="auto", density=True, alpha=.6, edgecolor="white")
    if x.std() > 0:
        g = np.linspace(x.min(), x.max(), 200); a.plot(g, stats.gaussian_kde(x)(g))
    a.axvline(x.mean(), ls="--", color="red", label="mean"); a.axvline(np.median(x), ls=":", color="green", label="median")
    a.set_title("Histogram + density"); a.set_xlabel(name); a.legend(fontsize=8)
    f2, a = plt.subplots(figsize=(4, 3.2)); a.boxplot(x, showmeans=True); a.set_title("Boxplot"); a.set_ylabel(name)
    f3, a = plt.subplots(figsize=(4, 3.2)); stats.probplot(x, dist="norm", plot=a); a.set_title("Normal Q-Q plot")
    for f in (f1, f2, f3):
        f.tight_layout()
    return f1, f2, f3


def missing_fig(df):
    m = df.isna().mean().mul(100); m = m[m > 0].sort_values()
    if m.empty:
        return None
    f, a = plt.subplots(figsize=(5, max(2, .3 * len(m) + 1))); a.barh(m.index, m.values)
    a.set_xlabel("% missing"); f.tight_layout()
    return f


def strength(r):
    a = abs(r)
    return "very strong" if a >= .9 else "strong" if a >= .7 else "moderate" if a >= .4 else "weak" if a >= .2 else "negligible"


def corr_heatmap(df, cols, method):
    c = df[cols].corr(method=method); n = len(cols); sz = min(10, max(4, .45 * n))
    f, a = plt.subplots(figsize=(sz, sz * .85)); im = a.imshow(c, vmin=-1, vmax=1, cmap="coolwarm")
    a.set_xticks(range(n)); a.set_xticklabels(cols, rotation=90, fontsize=7)
    a.set_yticks(range(n)); a.set_yticklabels(cols, fontsize=7)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                a.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    f.colorbar(im, ax=a, shrink=.8); a.set_title(f"{method.title()} correlation"); f.tight_layout()
    return f, c


def top_pairs(c, k=5):
    m = c.where(np.triu(np.ones(c.shape, bool), 1)).stack()
    m = m.reindex(m.abs().sort_values(ascending=False).index).head(k)
    return pd.DataFrame({"pair": [f"{a} vs {b}" for a, b in m.index], "r": m.round(3).values,
                         "strength": [strength(v) for v in m.values]})


def interpret_pairs(tp):
    o = [f"Strongest relationship: **{tp.pair[0]}** (r = {tp.r[0]}, {tp.strength[0]})."]
    if (tp.r.abs() >= .9).any():
        o.append("Pairs with |r| >= 0.9 carry almost the same information. In regression this causes multicollinearity "
                 "(check VIF); consider keeping only one variable from each such pair.")
    return o


def scatter_fig(df, x, y):
    d = df[[x, y]].dropna()
    f, a = plt.subplots(figsize=(5, 3.8)); a.scatter(d[x], d[y], s=14, alpha=.6)
    if len(d) > 2 and d[x].std() > 0:
        g = np.linspace(d[x].min(), d[x].max(), 50); a.plot(g, np.polyval(np.polyfit(d[x], d[y], 1), g), color="red")
    a.set_xlabel(x); a.set_ylabel(y); f.tight_layout()
    return f, stats.pearsonr(d[x], d[y]), stats.spearmanr(d[x], d[y])


def interpret_scatter(x, y, pr, sr):
    r, p, rs = pr[0], pr[1], sr[0]
    o = [f"Pearson r = {r:.3f} (p = {p:.4g}): a **{strength(r)} {'positive' if r > 0 else 'negative'}** linear relationship; "
         f"Spearman rho = {rs:.3f}."]
    if abs(r - rs) > 0.1:
        o.append("Pearson and Spearman differ noticeably: the relationship may be non-linear or driven by outliers. Look at the plot.")
    o.append("Correlation does not prove causation, and a strong correlation can be produced by a few extreme points.")
    return o


def group_box(df, num, cat):
    d = df[[num, cat]].dropna(); levels = sorted(d[cat].unique(), key=str)
    f, a = plt.subplots(figsize=(5, 3.8)); a.boxplot([d.loc[d[cat] == l, num] for l in levels], showmeans=True)
    a.set_xticks(range(1, len(levels) + 1)); a.set_xticklabels([str(l) for l in levels]); a.set_ylabel(num); a.set_xlabel(cat)
    f.tight_layout()
    return f, d.groupby(cat)[num].agg(["count", "mean", "median", "std"]).round(3)


def cat_bar(s, name):
    vc = s.value_counts(); sh = vc / vc.sum()
    f, a = plt.subplots(figsize=(5, 3.5)); a.bar([str(i) for i in vc.index], vc.values)
    a.set_title(f"Counts: {name}"); plt.setp(a.get_xticklabels(), rotation=45, ha="right"); f.tight_layout()
    msg = f"Most common: **{vc.index[0]}** ({sh.iloc[0]:.0%}). " + (
        f"The smallest class is only {sh.min():.0%}: imbalanced. If this is your target, accuracy alone can mislead (also check AUC/F1)."
        if sh.min() < .10 else "Classes are reasonably balanced.")
    return f, msg


def profile_columns(df):
    n = len(df); rows = []
    for c in df.columns:
        s = df[c]; nu = s.nunique(dropna=True); miss = int(s.isna().sum())
        if nu <= 1:
            kind = "constant/empty (excluded)"
        elif nu == 2:
            kind = "binary"
        elif pd.api.types.is_integer_dtype(s) and nu >= 0.95 * n:
            kind = "possible ID (excluded)"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            kind = "numeric"
        elif nu > max(50, 0.5 * n):
            kind = "text/ID (excluded)"
        else:
            kind = "categorical"
        rows.append({"column": c, "type": kind, "unique": nu, "missing": miss, "missing %": round(100 * miss / n, 1)})
    return pd.DataFrame(rows)

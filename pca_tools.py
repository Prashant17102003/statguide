"""PCA on the correlation matrix: which variables matter most for the main sources of variation.

Unsupervised: 'importance' here means contribution to the retained components,
NOT predictive power for a target variable.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def run_pca(df, cols, impute=True, log_skewed=False):
    """PCA on the correlation matrix. log_skewed=True applies log(1+x) to non-negative columns with |skewness| > 1 first."""
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()) if impute else X.dropna()
    X = X.loc[:, X.std(ddof=1) > 0]
    logged = []
    if log_skewed:
        for c in X.columns:
            if (X[c] >= 0).all() and abs(stats.skew(X[c], bias=False)) > 1:
                X[c] = np.log1p(X[c]); logged.append(c)
        X = X.loc[:, X.std(ddof=1) > 0]
    n, p = X.shape
    if p < 3 or n < p + 2:
        raise ValueError("Need at least 3 variable columns with variation and more rows than variables.")
    Z = (X - X.mean()) / X.std(ddof=1)
    R = np.corrcoef(Z.to_numpy().T)
    w, V = np.linalg.eigh(R); o = np.argsort(w)[::-1]; w = np.clip(w[o], 0, None); V = V[:, o]
    for k in range(p):                                  # sign convention: largest |loading| positive
        if V[np.abs(V[:, k]).argmax(), k] < 0:
            V[:, k] *= -1
    pcs = [f"PC{i + 1}" for i in range(p)]
    return dict(cols=list(X.columns), n=n, p=p, imputed=bool(impute), logged=logged, eig=w, ratio=w / w.sum(), cum=np.cumsum(w / w.sum()), V=V, R=R,
                scores=pd.DataFrame(Z.to_numpy() @ V, columns=pcs, index=X.index))


def suitability(res):
    R, n, p = res["R"], res["n"], res["p"]
    sign, logdet = np.linalg.slogdet(R)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet if sign > 0 else np.inf
    dof = p * (p - 1) / 2
    inv = np.linalg.pinv(R); d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    part = -inv / d; np.fill_diagonal(part, 0)
    r2 = R.copy(); np.fill_diagonal(r2, 0); r2, q2 = r2 ** 2, part ** 2
    return dict(chi2=chi2, df=dof, p=stats.chi2.sf(chi2, dof), kmo=r2.sum() / (r2.sum() + q2.sum()))


def choose_m(res, mode, value):
    if mode.startswith("Cumulative"):
        return int(np.searchsorted(res["cum"], value / 100 - 1e-12)) + 1
    if mode.startswith("Kaiser"):
        return max(1, int((res["eig"] > 1).sum()))
    return int(min(max(value, 1), res["p"]))


def loadings(res, m):
    return pd.DataFrame(res["V"][:, :m] * np.sqrt(res["eig"][:m]), index=res["cols"], columns=[f"PC{i + 1}" for i in range(m)])


def importance(res, m):
    L = loadings(res, m); w = res["eig"][:m]
    imp = (res["V"][:, :m] ** 2 * w).sum(1) / w.sum() * 100
    j = L.abs().to_numpy().argmax(1)
    return pd.DataFrame({"importance %": imp.round(2), "communality": (L ** 2).sum(1).round(3), "main PC": L.columns[j],
                         "loading on main PC": L.to_numpy()[np.arange(len(j)), j].round(3)}).sort_values("importance %", ascending=False)


def scree_fig(res, m):
    p = res["p"]; f, a = plt.subplots(figsize=(5, 3.4)); x = np.arange(1, p + 1)
    a.bar(x, res["eig"], alpha=.7); a.axhline(1, color="red", ls="--", lw=1, label="eigenvalue = 1")
    a.axvline(m + .5, color="green", ls=":", label=f"retained = {m}"); a.set_xlabel("Component"); a.set_ylabel("Eigenvalue")
    b = a.twinx(); b.plot(x, res["cum"] * 100, "o-", color="black", ms=3); b.set_ylabel("Cumulative % variance"); b.set_ylim(0, 105)
    a.legend(fontsize=8, loc="center right"); a.set_title("Scree plot"); f.tight_layout()
    return f


def loadings_fig(L):
    p, m = L.shape; f, a = plt.subplots(figsize=(min(9, 2 + m * .9), max(3, p * .28 + 1)))
    im = a.imshow(L.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    a.set_xticks(range(m)); a.set_xticklabels(L.columns); a.set_yticks(range(p)); a.set_yticklabels(L.index, fontsize=7)
    if p * m <= 120:
        for i in range(p):
            for j in range(m):
                a.text(j, i, f"{L.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    f.colorbar(im, ax=a, shrink=.8); a.set_title("Loadings (correlation of variable with component)"); f.tight_layout()
    return f


def importance_fig(imp, top=20):
    t = imp.head(top)[::-1]; p = len(imp)
    f, a = plt.subplots(figsize=(5, max(2.5, .28 * len(t) + 1))); a.barh(t.index, t["importance %"])
    a.axvline(100 / p, color="red", ls="--", label=f"equal share = {100 / p:.1f}%"); a.set_xlabel("Importance %")
    a.legend(fontsize=8); a.set_title("Variable importance (retained components)"); f.tight_layout()
    return f


def biplot(res, groups=None, top=10):
    S, L = res["scores"], loadings(res, 2); f, a = plt.subplots(figsize=(5.5, 4.4))
    if groups is None:
        a.scatter(S["PC1"], S["PC2"], s=10, alpha=.5)
    else:
        g = groups.loc[S.index].fillna("(missing)").astype(str)
        for lv in sorted(g.unique()):
            k = (g == lv).to_numpy(); a.scatter(S["PC1"][k], S["PC2"][k], s=10, alpha=.55, label=lv)
        a.legend(fontsize=7)
    sc = .8 * np.abs(S[["PC1", "PC2"]].to_numpy()).max() / np.abs(L.to_numpy()).max()
    for v in (L ** 2).sum(1).sort_values(ascending=False).head(top).index:
        a.arrow(0, 0, L.loc[v, "PC1"] * sc, L.loc[v, "PC2"] * sc, color="red", alpha=.7, head_width=.08)
        a.text(L.loc[v, "PC1"] * sc * 1.08, L.loc[v, "PC2"] * sc * 1.08, v, color="red", fontsize=7)
    a.set_xlabel(f"PC1 ({res['ratio'][0] * 100:.1f}%)"); a.set_ylabel(f"PC2 ({res['ratio'][1] * 100:.1f}%)")
    a.set_title("Biplot: scores and top variable arrows"); f.tight_layout()
    return f


def interpret_pca(res, m, imp, suit):
    L = loadings(res, m); p = res["p"]; o = []
    kaiser = int((res["eig"] > 1).sum()); m80 = int(np.searchsorted(res["cum"], .8 - 1e-12)) + 1
    pre = "missing values were filled with the column median" if res.get("imputed") else "rows with missing values were dropped"
    pre += (f"; log(1+x) was applied to skewed variables: {', '.join(res['logged'])}" if res.get("logged") else "; no transformation was applied, so strongly skewed variables can distort the components")
    o.append(f"**Preprocessing:** {pre}.")
    o.append(f"**Components:** the {m} retained component(s) explain **{res['cum'][m - 1] * 100:.1f}%** of the total variance "
             f"(Kaiser rule keeps {kaiser}; {m80} needed for 80%).")
    for k in range(min(m, 2)):
        c = L.iloc[:, k]; top = c.abs().sort_values(ascending=False).head(3).index
        txt = ", ".join(f"{v} ({c[v]:+.2f})" for v in top)
        same = (c > 0).all() or (c < 0).all()
        o.append(f"**PC{k + 1} ({res['ratio'][k] * 100:.1f}%)** is driven mainly by {txt}." +
                 (" All variables load in the same direction, so it acts as a general 'overall level / size' component." if same and k == 0 else ""))
    o.append("**Most important variables:** " + ", ".join(f"{v} ({x:.1f}%)" for v, x in imp["importance %"].head(5).items()) + ".")
    low = imp[imp["importance %"] < .5 * 100 / p].index.tolist()
    if low:
        o.append("**Low contribution** to the retained components (below half the equal share): " + ", ".join(low[:8]) +
                 (" and more" if len(low) > 8 else "") + ". They vary in directions the retained components do not capture; this does not mean they are useless for predicting a target.")
    k_ = suit["kmo"]; lab = "marvelous" if k_ >= .9 else "meritorious" if k_ >= .8 else "middling" if k_ >= .7 else "mediocre" if k_ >= .6 else "poor"
    o.append(f"**Suitability:** KMO = {k_:.2f} ({lab}, rule of thumb: 0.6 or more is adequate); Bartlett's test p = {suit['p']:.4g} " +
             ("(correlations differ from zero, PCA is meaningful)." if suit["p"] < .05 else "(no clear correlation structure; PCA may not help)."))
    pairs = int(((np.abs(res["R"]) >= .9).sum() - p) / 2)
    if pairs:
        o.append(f"**Redundancy:** {pairs} variable pair(s) have |r| >= 0.9, so those variables carry nearly the same information.")
    return o

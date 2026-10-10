"""Ordinary least squares regression (1 or more numeric predictors) with a fitted line, diagnostics and interpretation."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def pf(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4g}"


def add_dummies(df, cats):
    """Dummy-code categorical predictors (first level alphabetically = reference). Missing categories stay missing (row dropped later)."""
    out, new = df.copy(), []
    for c in cats:
        lv = sorted(df[c].dropna().unique(), key=str)
        for v in lv[1:]:
            name = f"{c}={v}"
            out[name] = (df[c].astype(object) == v).astype(float).where(df[c].notna())
            new.append(name)
    return out, new


def holdout_eval(df, y, xs, test_frac=0.25, seed=42):
    """Fit on a random training part, then report error on the unseen test part (honest check of predictive accuracy)."""
    d = df[[y] + xs].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    n = len(d); n_test = int(round(n * test_frac))
    if n_test < 5 or n - n_test < len(xs) + 5:
        raise ValueError("Too few rows for a hold-out split with these settings.")
    idx = np.random.default_rng(seed).permutation(n); te, tr = d.iloc[idx[:n_test]], d.iloc[idx[n_test:]]
    r = fit_ols(tr, y, xs); b = r["beta"]
    pred = np.column_stack([np.ones(len(te)), te[xs].to_numpy(float)]) @ b; err = te[y].to_numpy(float) - pred
    sst = ((te[y] - tr[y].mean()) ** 2).sum()                     # baseline = predicting the training mean
    return {"n_train": len(tr), "n_test": len(te), "rmse_train": float(np.sqrt(np.mean(r["resid"] ** 2))),
            "rmse_test": float(np.sqrt(np.mean(err ** 2))), "mae_test": float(np.mean(np.abs(err))),
            "r2_train": r["r2"], "r2_test": float(1 - (err ** 2).sum() / sst), "pred": pred, "actual": te[y].to_numpy(float)}


def fit_ols(df, y, xs):
    d = df[[y] + xs].apply(pd.to_numeric, errors="coerce").dropna()
    n, k = len(d), len(xs)
    if n < k + 3:
        raise ValueError("Too few complete rows for this many predictors.")
    X = np.column_stack([np.ones(n), d[xs].to_numpy(float)]); Y = d[y].to_numpy(float)
    if np.linalg.matrix_rank(X) < k + 1:
        raise ValueError("Predictors are perfectly collinear (or constant). Remove one of them.")
    XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ X.T @ Y; fit = X @ beta; e = Y - fit
    dfr = n - k - 1; s2 = e @ e / dfr; se = np.sqrt(s2 * np.diag(XtXi)); t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dfr); tc = stats.t.ppf(.975, dfr)
    r2 = 1 - e @ e / ((Y - Y.mean()) ** 2).sum(); F = (r2 / k) / ((1 - r2) / dfr)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X); cook = e ** 2 / ((k + 1) * s2) * h / (1 - h) ** 2
    # Breusch-Pagan (Koenker): regress e^2 on X, LM = n * R2_aux
    e2 = e ** 2; b2 = np.linalg.lstsq(X, e2, rcond=None)[0]
    r2a = 1 - ((e2 - X @ b2) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum(); lm = n * r2a
    vif = {}
    if k >= 2:
        for j, c in enumerate(xs):
            o = np.delete(np.arange(1, k + 1), j); Xo = X[:, np.r_[0, o]]
            xj = X[:, j + 1]; rj = 1 - ((xj - Xo @ np.linalg.lstsq(Xo, xj, rcond=None)[0]) ** 2).sum() / ((xj - xj.mean()) ** 2).sum()
            vif[c] = 1 / (1 - rj) if rj < 1 else np.inf
    coef = pd.DataFrame({"estimate": beta, "std error": se, "t": t, "p-value": p, "CI low": beta - tc * se, "CI high": beta + tc * se},
                        index=["intercept"] + xs)
    return dict(data=d, y=y, xs=xs, n=n, k=k, dfr=dfr, s2=s2, XtXi=XtXi, beta=beta, coef=coef, fit=fit, resid=e,
                r2=r2, adj_r2=1 - (1 - r2) * (n - 1) / dfr, F=F, pF=stats.f.sf(F, k, dfr), cook=cook,
                stud=e / np.sqrt(s2 * (1 - h)), shapiro_p=stats.shapiro(e).pvalue if 3 <= n <= 5000 else np.nan,
                bp_lm=lm, bp_p=stats.chi2.sf(lm, k), dw=((np.diff(e) ** 2).sum() / (e ** 2).sum()), vif=vif)


def line_fig(r):
    d, x, y = r["data"], r["xs"][0], r["y"]; n, s2, b = r["n"], r["s2"], r["beta"]
    g = np.linspace(d[x].min(), d[x].max(), 100); xb = d[x].mean(); sxx = ((d[x] - xb) ** 2).sum()
    tc = stats.t.ppf(.975, r["dfr"]); yh = b[0] + b[1] * g
    ci = tc * np.sqrt(s2 * (1 / n + (g - xb) ** 2 / sxx)); pi = tc * np.sqrt(s2 * (1 + 1 / n + (g - xb) ** 2 / sxx))
    f, a = plt.subplots(figsize=(6, 4.2)); a.scatter(d[x], d[y], s=14, alpha=.55)
    a.plot(g, yh, color="red", label=f"y = {b[0]:.3g} + {b[1]:.3g} x"); a.fill_between(g, yh - ci, yh + ci, color="red", alpha=.25, label="95% CI of the line")
    a.plot(g, yh - pi, "k--", lw=.8, label="95% prediction band"); a.plot(g, yh + pi, "k--", lw=.8)
    a.set_xlabel(x); a.set_ylabel(y); a.legend(fontsize=8); a.set_title("Regression line"); f.tight_layout()
    return f


def avp_fig(r):
    f, a = plt.subplots(figsize=(6, 4.2)); y = r["data"][r["y"]]; a.scatter(r["fit"], y, s=14, alpha=.55)
    lo, hi = min(y.min(), r["fit"].min()), max(y.max(), r["fit"].max()); a.plot([lo, hi], [lo, hi], color="red")
    a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title("Actual vs predicted (red = perfect fit)"); f.tight_layout()
    return f


def diag_figs(r):
    f1, a = plt.subplots(figsize=(4, 3.2)); a.scatter(r["fit"], r["resid"], s=10, alpha=.55); a.axhline(0, color="red")
    a.set_xlabel("Fitted"); a.set_ylabel("Residual"); a.set_title("Residuals vs fitted")
    f2, a = plt.subplots(figsize=(4, 3.2)); stats.probplot(r["resid"], dist="norm", plot=a); a.set_title("Residual Q-Q plot")
    f3, a = plt.subplots(figsize=(4, 3.2)); a.vlines(np.arange(r["n"]), 0, r["cook"], lw=.6, color="#4f46e5"); a.axhline(4 / r["n"], color="red", ls="--", label="4/n")
    a.set_xlim(0, r["n"]); a.set_xlabel("Observation"); a.set_ylabel("Cook's distance"); a.legend(fontsize=8); a.set_title("Influential points")
    for f in (f1, f2, f3):
        f.tight_layout()
    return f1, f2, f3


def _pp(p):
    return ("= " + pf(p)) if p >= 1e-4 else pf(p)


def interpret(r, show_dw=False):
    c, o, xs = r["coef"], [], r["xs"]; dm = r.get("dummies") or []
    if r["k"] == 1 and xs[0] in dm:
        b = c.loc[xs[0]]
        o.append(f"**Group difference:** for {xs[0]} (1) compared with the reference level (0), the mean of {r['y']} differs by **{b['estimate']:+.4g}** "
                 f"(95% CI {b['CI low']:.4g} to {b['CI high']:.4g}, p {_pp(b['p-value'])}).")
    elif r["k"] == 1:
        b = c.loc[xs[0]]
        o.append(f"**Line:** {r['y']} = {c.loc['intercept', 'estimate']:.4g} + ({b['estimate']:.4g}) x {xs[0]}. Each 1-unit increase in {xs[0]} goes with a "
                 f"**{b['estimate']:+.4g}** change in {r['y']} (95% CI {b['CI low']:.4g} to {b['CI high']:.4g}, p {_pp(b['p-value'])}).")
    else:
        sig = [v for v in xs if c.loc[v, "p-value"] < .05]
        o.append(f"**Coefficients:** each is the effect of that predictor holding the others fixed. Significant at 5%: {', '.join(sig) if sig else 'none'}.")
    if dm and not (r["k"] == 1 and xs[0] in dm):
        o.append("**Dummy variables (name=level):** each coefficient is the difference from the reference level of that categorical variable (its first level alphabetically), holding the other predictors fixed.")
    o.append(f"**Fit:** R-squared = {r['r2']:.3f} (adjusted {r['adj_r2']:.3f}), so the model explains about {r['r2'] * 100:.0f}% of the variation in {r['y']}. "
             f"Overall F-test p {_pp(r['pF'])}.")
    sp, bp = r["shapiro_p"], r["bp_p"]
    o.append("**Normal residuals:** " + ("not computed." if np.isnan(sp) else f"Shapiro-Wilk p = {sp:.4f} ({'evidence against normality; check the residual Q-Q plot' if sp < .05 else 'no evidence against normality'}). With large n tiny deviations are flagged, so judge mainly by the Q-Q plot."))
    o.append(f"**Constant variance:** Breusch-Pagan p = {bp:.4f} ({'variance changes with the fitted values (heteroscedasticity)' if bp < .05 else 'no evidence of changing variance'}).")
    if show_dw:
        o.append(f"**Independence:** Durbin-Watson = {r['dw']:.2f} (about 2 means no autocorrelation; valid only because you said the rows are in time order).")
    if r["vif"]:
        bad = [f"{v} ({x:.1f})" for v, x in r["vif"].items() if x > 5]
        o.append("**Multicollinearity (VIF):** " + (f"high for {', '.join(bad)}; coefficients are unstable." if bad else "all VIF are 5 or below."))
    ninf, nout = int((r["cook"] > 4 / r["n"]).sum()), int((np.abs(r["stud"]) > 3).sum())
    o.append(f"**Unusual points:** {ninf} observation(s) with Cook's distance above 4/n and {nout} with |studentised residual| above 3. Check them before trusting the fit.")
    if (not np.isnan(sp) and sp < .05) or bp < .05:
        o.append("**Suggestion:** assumptions look violated; consider a log or square-root transform of the response, robust standard errors, or a rank-based measure such as Spearman correlation.")
    return o


def interpret_holdout(h):
    o = [f"**Hold-out check:** the model was fitted on {h['n_train']} rows and tested on {h['n_test']} unseen rows. Test RMSE = {h['rmse_test']:.4g} "
         f"(training RMSE {h['rmse_train']:.4g}), test MAE = {h['mae_test']:.4g}, test R-squared = {h['r2_test']:.3f} (training {h['r2_train']:.3f})."]
    if h["r2_test"] < h["r2_train"] - .1:
        o.append("Test R-squared is clearly lower than training R-squared: the model may be overfitting, so trust the test figures.")
    if h["r2_test"] <= 0.02:
        o.append("Test R-squared is close to zero or negative: these predictors predict unseen rows no better than simply using the average.")
    return o

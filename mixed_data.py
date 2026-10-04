"""Handle any column type (numeric, binary, categorical) and missing values.

Imputation and encoding happen INSIDE each cross-validation fold (via Pipeline),
so nothing from the test fold leaks into preprocessing.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from eda_tools import profile_columns
from tests_lib import naive_ttest, corrected_ttest, bayes_correlated_ttest


def compare_mixed(df, target, k=10, r=5, rope=0.01, impute=True, seed=42):
    d = df.dropna(subset=[target]).copy()
    if d[target].nunique() < 2:
        raise ValueError("Target needs at least 2 classes after removing rows with a missing target.")
    y = pd.factorize(d[target].astype(str))[0]
    prof = profile_columns(d.drop(columns=[target]))
    excluded = prof[prof.type.str.contains("excluded")].column.tolist()
    feats = [c for c in prof.column if c not in excluded]
    if not feats:
        raise ValueError("No usable feature columns.")
    X = d[feats].copy()
    for c in X.columns:                       # bool and object columns -> categorical with real NaN
        if pd.api.types.is_bool_dtype(X[c]) or not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    num = [c for c in feats if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    cat = [c for c in feats if c not in num]
    n_missing = int(X.isna().sum().sum())
    if not impute:
        keep = ~X.isna().any(axis=1)
        X, y = X[keep], y[keep.to_numpy()]
        if len(y) < 20 or np.bincount(y).min() < 2:
            raise ValueError("Too few complete rows. Use imputation instead.")
    k_eff = min(k, int(np.bincount(y).min()))
    if k_eff < 2:
        raise ValueError("Smallest class has fewer than 2 rows; cannot cross-validate.")

    def pre():
        return ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    lr = Pipeline([("pre", pre()), ("m", LogisticRegression(max_iter=2000))])
    rf = Pipeline([("pre", pre()), ("m", RandomForestClassifier(n_estimators=100, random_state=0))])

    cv = RepeatedStratifiedKFold(n_splits=k_eff, n_repeats=r, random_state=seed)
    a, b = [], []
    for tr, te in cv.split(X, y):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        a.append(lr.fit(Xtr, y[tr]).score(Xte, y[te]))
        b.append(rf.fit(Xtr, y[tr]).score(Xte, y[te]))
    n_tr, n_te = len(tr), len(te)
    bay = bayes_correlated_ttest(a, b, n_tr, n_te, rope)
    return {"LR_acc": np.mean(a), "RF_acc": np.mean(b), "n": len(y), "classes": int(len(np.unique(y))),
            "k_used": k_eff, "n_num": len(num), "n_cat": len(cat), "n_missing_cells": n_missing,
            "excluded": excluded, "p_naive": naive_ttest(a, b)[1], "p_corrected": corrected_ttest(a, b, n_tr, n_te)[1],
            "P(LR better)": bay["A_better"], "P(equal)": bay["equal"], "P(RF better)": bay["B_better"]}

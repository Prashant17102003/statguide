"""Compare Logistic Regression vs Random Forest on REAL datasets with 3 tests.

Built-in datasets (no download): breast_cancer, wine (class 0 vs rest).
Your own CSV:  from compare_real import compare_csv
               compare_csv("heart_failure.csv", target="DEATH_EVENT")
Run:  python compare_real.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from tests_lib import naive_ttest, corrected_ttest, bayes_correlated_ttest

K, R, ROPE = 10, 5, 0.01      # 10-fold CV repeated 5 times -> 50 scores per model


def compare(X, y, name, k=K, r=R, rope=ROPE, seed=42):
    X = np.asarray(X, float)
    X = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
    y = pd.factorize(np.asarray(y))[0]
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=r, random_state=seed)
    a, b = [], []
    for tr, te in cv.split(X, y):
        a.append(LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).score(X[te], y[te]))
        b.append(RandomForestClassifier(n_estimators=100, random_state=0)
                 .fit(X[tr], y[tr]).score(X[te], y[te]))
    n_tr, n_te = len(tr), len(te)
    bay = bayes_correlated_ttest(a, b, n_tr, n_te, rope)
    return {"dataset": name, "n": len(y), "LR_acc": np.mean(a), "RF_acc": np.mean(b),
            "p_naive": naive_ttest(a, b)[1], "p_corrected": corrected_ttest(a, b, n_tr, n_te)[1],
            "P(LR better)": bay["A_better"], "P(equal)": bay["equal"], "P(RF better)": bay["B_better"]}


def compare_csv(path, target, **kw):
    df = pd.read_csv(path).dropna()
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True).astype(float)
    return compare(X, df[target], path, **kw)


if __name__ == "__main__":
    rows = []
    bc = load_breast_cancer()
    rows.append(compare(bc.data, bc.target, "breast_cancer"))
    w = load_wine()
    rows.append(compare(w.data, (w.target == 0).astype(int), "wine (class0 vs rest)"))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv("real_data_results.csv", index=False)

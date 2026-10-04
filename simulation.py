"""Type I error simulation (null hypothesis is TRUE by construction).

Model A uses feature block 1, Model B uses feature block 2. Both blocks carry
exactly the same amount of signal about y, so the two models have EQUAL expected
accuracy, but their per-sample results differ because they use different data.
(Two models that differ only by a random seed would NOT show the problem.)

A good test should reject about 5% of the time at alpha = 0.05.

Run:  python simulation.py            (quick: 100 replicates per n)
      python simulation.py 1000       (final: 1000 replicates per n)
"""
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from tests_lib import naive_ttest, corrected_ttest, bayes_correlated_ttest

ALPHA = 0.05
K, R = 10, 1         # 10-fold CV, repeated R times (try 10x10 too: more repeats -> naive test gets worse)
P, DELTA = 5, 0.4    # features per block, signal strength per feature


def make_data(n, rng):
    y = rng.integers(0, 2, n)
    X1 = rng.normal(size=(n, P)) + DELTA * y[:, None]
    X2 = rng.normal(size=(n, P)) + DELTA * y[:, None]
    return X1, X2, y


def one_replicate(n, seed):
    rng = np.random.default_rng(seed)
    X1, X2, y = make_data(n, rng)
    cv = RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=seed)
    a, b = [], []
    for tr, te in cv.split(X1, y):
        a.append(LogisticRegression().fit(X1[tr], y[tr]).score(X1[te], y[te]))
        b.append(LogisticRegression().fit(X2[tr], y[tr]).score(X2[te], y[te]))
    n_train, n_test = len(tr), len(te)
    p_naive = naive_ttest(a, b)[1]
    p_corr = corrected_ttest(a, b, n_train, n_test)[1]
    bayes = bayes_correlated_ttest(a, b, n_train, n_test, rope=0.01)
    # Bayesian decision: declare a "difference" if P(A better) or P(B better) > 0.95
    bayes_decides = max(bayes["A_better"], bayes["B_better"]) > 0.95
    return p_naive < ALPHA, p_corr < ALPHA, bayes_decides


def run(n, reps):
    res = Parallel(n_jobs=-1)(delayed(one_replicate)(n, s) for s in range(reps))
    res = np.array(res, dtype=float)
    rates = res.mean(axis=0)
    se = np.sqrt(rates * (1 - rates) / reps)
    return rates, se


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rows = []
    for n in [50, 100, 300, 1000]:
        rates, se = run(n, reps)
        rows.append({"n": n, "naive": rates[0], "corrected": rates[1],
                     "bayes_95%": rates[2], "MC_SE": se[0]})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(f"\nType I error rates (target = {ALPHA}), {reps} replicates per n:")
    print(df.round(3).to_string(index=False))
    df.to_csv("simulation_results.csv", index=False)

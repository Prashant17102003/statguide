"""Power simulation: a REAL difference exists (Model A's block is stronger than Model B's).

Reuses settings (P, DELTA, K, R) from simulation.py.
Run:  python power.py          (quick: 50 replicates per setting)
      python power.py 200
"""
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

import simulation as sim
from tests_lib import naive_ttest, corrected_ttest, bayes_correlated_ttest


def true_accuracy(delta):
    """Best possible accuracy for P independent features with mean shift delta."""
    return norm.cdf(delta * np.sqrt(sim.P) / 2)


def one_replicate(n, seed, delta2):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X1 = rng.normal(size=(n, sim.P)) + sim.DELTA * y[:, None]   # model A: strong block
    X2 = rng.normal(size=(n, sim.P)) + delta2 * y[:, None]      # model B: weaker block
    cv = RepeatedStratifiedKFold(n_splits=sim.K, n_repeats=sim.R, random_state=seed)
    a, b = [], []
    for tr, te in cv.split(X1, y):
        a.append(LogisticRegression().fit(X1[tr], y[tr]).score(X1[te], y[te]))
        b.append(LogisticRegression().fit(X2[tr], y[tr]).score(X2[te], y[te]))
    n_train, n_test = len(tr), len(te)
    p_naive = naive_ttest(a, b)[1]
    p_corr = corrected_ttest(a, b, n_train, n_test)[1]
    bayes = bayes_correlated_ttest(a, b, n_train, n_test, rope=0.01)
    return (p_naive < sim.ALPHA, p_corr < sim.ALPHA,
            max(bayes["A_better"], bayes["B_better"]) > 0.95)


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rows = []
    for delta2 in [0.4, 0.3, 0.2]:          # 0.4 = no difference (Type I error row)
        gap = true_accuracy(sim.DELTA) - true_accuracy(delta2)
        for n in [100, 300, 1000]:
            res = Parallel(n_jobs=-1)(delayed(one_replicate)(n, s, delta2) for s in range(reps))
            rate = np.array(res, dtype=float).mean(axis=0)
            rows.append({"true_gap": round(gap, 3), "n": n, "naive": rate[0],
                         "corrected": rate[1], "bayes_95%": rate[2]})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(f"\nRejection rates ({reps} replicates). true_gap=0 -> Type I error, else -> power")
    print(df.round(3).to_string(index=False))
    df.to_csv("power_results.csv", index=False)

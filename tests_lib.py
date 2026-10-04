"""Tests for comparing two models from cross-validation scores.

a, b      : arrays of CV scores (same folds, same order) for model A and model B
n_train   : size of each training set
n_test    : size of each test set
"""
import numpy as np
from scipy import stats


def naive_ttest(a, b):
    """Plain paired t-test. Ignores overlap of training sets (too optimistic)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    s = d.std(ddof=1)
    if s == 0:
        return 0.0, 1.0
    t = d.mean() / (s / np.sqrt(n))
    return t, 2 * stats.t.sf(abs(t), n - 1)


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau-Bengio corrected t-test: variance factor 1/n + n_test/n_train."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / n + n_test / n_train) * v)
    return t, 2 * stats.t.sf(abs(t), n - 1)


def bayes_correlated_ttest(a, b, n_train, n_test, rope=0.01):
    """Bayesian correlated t-test (Corani & Benavoli, 2015) with a ROPE.

    Returns probabilities: A better, practically equal (inside ROPE), B better.
    rope = smallest difference in the metric that you consider meaningful.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    scale = np.sqrt((1 / n + n_test / n_train) * d.var(ddof=1))
    if scale == 0:
        mean = d.mean()
        return {"A_better": float(mean > rope), "equal": float(abs(mean) <= rope),
                "B_better": float(mean < -rope)}
    dist = stats.t(df=n - 1, loc=d.mean(), scale=scale)
    p_b = dist.cdf(-rope)
    p_a = dist.sf(rope)
    return {"A_better": p_a, "equal": 1 - p_a - p_b, "B_better": p_b}

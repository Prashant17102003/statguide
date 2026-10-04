"""Creates SYNTHETIC sample datasets (not real data) for testing every column type and missing values."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

rng = np.random.default_rng(7); n = 600
age = rng.integers(18, 75, n).astype(float); income = np.round(rng.lognormal(10.3, .5, n)); tenure = rng.integers(1, 72, n).astype(float)
gender = rng.choice(["Male", "Female"], n); partner = rng.choice(["Yes", "No"], n, p=[.45, .55])
plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[.45, .35, .2]); region = rng.choice(["North", "South", "East", "West", "Central"], n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2])
monthly = np.round(20 + 25 * (plan == "Standard") + 55 * (plan == "Premium") + rng.normal(0, 6, n), 2)
logit = -.6 - .03 * tenure + .015 * (monthly - 50) + .9 * (contract == "Month-to-month") - .3 * (partner == "Yes")
churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
df = pd.DataFrame({"customer_id": [f"C{i:04d}" for i in range(n)], "age": age, "income": income, "tenure_months": tenure,
                   "gender": gender, "has_partner": partner, "plan": plan, "region": region, "contract": contract,
                   "monthly_charges": monthly, "churn": churn})
for col, frac in [("age", .05), ("income", .08), ("tenure_months", .03), ("gender", .03), ("plan", .04), ("region", .06)]:
    df.loc[rng.random(n) < frac, col] = np.nan
df.loc[rng.choice(n, 3, replace=False), "churn"] = np.nan          # a few missing targets
df.to_csv("data/synthetic_mixed_with_nulls.csv", index=False)

iris = load_iris(as_frame=True); ir = iris.data.copy(); ir["species"] = iris.target_names[iris.target]
for c in iris.data.columns:
    ir.loc[rng.random(len(ir)) < .08, c] = np.nan
ir.to_csv("data/iris_3class_with_nulls.csv", index=False)
print(df.shape, ir.shape)

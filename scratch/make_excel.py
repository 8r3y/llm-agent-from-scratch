from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

df = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "tenure_months": rng.integers(1, 60, n),
    "monthly_spend": rng.normal(50, 15, n).round(2),
    "support_tickets": rng.poisson(1.5, n),
    "plan": rng.choice(["basic", "standard", "premium"], n),
})

logit = (
    -1.0
    - 0.06 * df["tenure_months"]
    + 0.6 * df["support_tickets"]
    + 0.02 * (df["monthly_spend"] - 50)
    + df["plan"].map({"basic": 0.8, "standard": 0.0, "premium": -0.8})
)
df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

out = Path("data")
out.mkdir(exist_ok=True)
df.to_excel(out / "churn.xlsx", index=False)
print(df.head())
print("churn rate:", df["churned"].mean())
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, default="results/disagreement_scores.csv")
parser.add_argument("--output_csv", type=str, default="results/hard_reasoning_benchmark.csv")
args = parser.parse_args()

input_csv = args.input_csv
output_csv = args.output_csv

df = pd.read_csv(
    input_csv,
    dtype={
        "prompt": str,
        "disagreement_score": float,
        "used_tokens_pct": float,
    },
    low_memory=False,
)

required_cols = {"prompt", "disagreement_score", "used_tokens_pct"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(missing)

df["hardness_score"] = df["disagreement_score"] * (1.0 + df["used_tokens_pct"] / 100.0)

df = df.sort_values("hardness_score").reset_index(drop=True)
n = len(df)

df["hard_bucket"] = "easy"
df.loc[int(0.50 * n):, "hard_bucket"] = "medium"
df.loc[int(0.80 * n):, "hard_bucket"] = "hard"
df.loc[int(0.95 * n):, "hard_bucket"] = "very_hard"

os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
df.to_csv(output_csv, index=False)

print(output_csv)
print(df["hard_bucket"].value_counts())
print(
    df.sort_values("hardness_score", ascending=False)
      .head(10)[["hardness_score", "used_tokens_pct", "prompt"]]
)

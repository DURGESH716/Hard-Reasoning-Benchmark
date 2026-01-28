import os
import csv
import pandas as pd

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_csv(path, dtype=None):
    return pd.read_csv(path, dtype=dtype, low_memory=False)

def write_csv(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def txt_disagreement_to_csv(txt_path, csv_path):
    ensure_dir(os.path.dirname(csv_path))

    rows = []
    prompt = None
    score = None
    tokens = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Prompt:"):
                prompt = line.replace("Prompt:", "").strip()
            elif line.startswith("Disagreement score:"):
                score = float(line.split(":")[1].strip())
            elif line.startswith("Used tokens:"):
                tokens = float(line.split(":")[1].replace("%", "").strip())
                rows.append([prompt, score, tokens])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "disagreement_score", "used_tokens_pct"])
        writer.writerows(rows)

def load_benchmark(csv_path):
    df = read_csv(csv_path)
    return (
        df["prompt"].tolist(),
        df["hard_bucket"].tolist(),
        df["hardness_score"].tolist(),
        df["used_tokens_pct"].tolist(),
    )

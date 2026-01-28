import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--txt_file", type=str, default="results/disagreement_scores.txt")
parser.add_argument("--csv_file", type=str, default=None)
args = parser.parse_args()

txt_file = args.txt_file
csv_file = args.csv_file or os.path.splitext(txt_file)[0] + ".csv"

rows = []
prompt = None
score = None
tokens = None

with open(txt_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("Prompt:"):
            prompt = line.replace("Prompt:", "").strip()
        elif line.startswith("Disagreement score:"):
            score = float(line.split(":", 1)[1].strip())
        elif line.startswith("Used tokens:"):
            tokens = float(line.split(":", 1)[1].replace("%", "").strip())
            rows.append([prompt, score, tokens])

os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)

with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["prompt", "disagreement_score", "used_tokens_pct"])
    writer.writerows(rows)

print(csv_file)

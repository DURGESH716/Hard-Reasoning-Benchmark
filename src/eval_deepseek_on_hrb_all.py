import time
import torch
import pandas as pd
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-math-7b-base")
parser.add_argument("--benchmark_csv", type=str, default="results/hard_reasoning_benchmark.csv")
parser.add_argument("--output_csv", type=str, default="results/deepseek_hrb_results.csv")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

df = pd.read_csv(args.benchmark_csv)

prompts = df["prompt"].tolist()
buckets = df["hard_bucket"].tolist()
hardness_scores = df["hardness_score"].tolist()
used_tokens = df["used_tokens_pct"].tolist()

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

results = []

for i, prompt in enumerate(tqdm(prompts)):
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    end = time.time()

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results.append({
        "prompt": prompt,
        "hard_bucket": buckets[i],
        "hardness_score": hardness_scores[i],
        "used_tokens_pct": used_tokens[i],
        "response": decoded[len(prompt):].strip(),
        "latency_sec": round(end - start, 3),
        "input_tokens": inputs["input_ids"].shape[1],
        "output_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
    })

os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
pd.DataFrame(results).to_csv(args.output_csv, index=False)

print(args.output_csv)

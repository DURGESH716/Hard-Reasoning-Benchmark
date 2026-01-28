import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--deepseek_model", type=str, default="deepseek-ai/deepseek-math-7b-base")
parser.add_argument("--qwen_model", type=str, default="Qwen/Qwen3-8B-Base")
parser.add_argument("--dataset_root", type=str, default="data")
parser.add_argument("--output_dir", type=str, default="results")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_len", type=int, default=1024)
parser.add_argument("--disagree_percentile", type=int, default=80)
parser.add_argument("--checkpoint_every", type=int, default=1000)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

DATASETS = {
    "gsm8k": os.path.join(args.dataset_root, "gsm8k"),
    "math": os.path.join(args.dataset_root, "math"),
    "arc-challenge": os.path.join(args.dataset_root, "arc-challenge"),
    "arc-easy": os.path.join(args.dataset_root, "arc-easy"),
}

tokenizer = AutoTokenizer.from_pretrained(args.deepseek_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=DTYPE,
        device_map="auto"
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

deepseek = load_model(args.deepseek_model)
qwen = load_model(args.qwen_model)

def load_gsm8k(path):
    all_data = []
    for split in ["train", "test"]:
        file = os.path.join(path, f"{split}.jsonl")
        if not os.path.exists(file):
            continue
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                all_data.append({"question": obj.get("question", "")})
    return HFDataset.from_list(all_data)

def load_math_txt(path):
    all_data = []
    if not os.path.exists(path):
        return None
    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_data.append({"problem": line})
    return HFDataset.from_list(all_data)

def load_parquet_dataset(path):
    if not os.path.exists(path):
        return None
    datasets = []
    for fname in os.listdir(path):
        if fname.endswith(".parquet"):
            ds = load_dataset("parquet", data_files=os.path.join(path, fname))["train"]
            datasets.append(ds)
    if datasets:
        return concatenate_datasets(datasets)
    return None

all_datasets = []

gsm = load_gsm8k(DATASETS["gsm8k"])
if gsm:
    all_datasets.append(gsm)

math = load_math_txt(DATASETS["math"])
if math:
    all_datasets.append(math)

arc_c = load_parquet_dataset(DATASETS["arc-challenge"])
if arc_c:
    all_datasets.append(arc_c)

arc_e = load_parquet_dataset(DATASETS["arc-easy"])
if arc_e:
    all_datasets.append(arc_e)

if not all_datasets:
    raise RuntimeError("No datasets found")

dataset = concatenate_datasets(all_datasets)

def collate(batch):
    texts = []
    for x in batch:
        if "question" in x:
            texts.append(str(x["question"]))
        elif "problem" in x:
            texts.append(str(x["problem"]))
        else:
            texts.append(str(x))
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_len
    )
    return {k: v.to(DEVICE) for k, v in tokens.items()}

loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

@torch.no_grad()
def get_logits(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask).logits

@torch.no_grad()
def disagreement_mask(logits_a, logits_b, top_k=50):
    topk_a = torch.topk(logits_a, k=top_k, dim=-1)
    topk_b = torch.topk(logits_b, k=top_k, dim=-1)
    p = F.softmax(topk_a.values, dim=-1)
    q = F.softmax(topk_b.values, dim=-1)
    log_p = torch.log(p + 1e-12)
    kl = F.kl_div(log_p, q, reduction="none").sum(-1)
    kl = kl.float()
    threshold = torch.quantile(kl, args.disagree_percentile / 100.0)
    mask = kl > threshold
    return mask, kl.mean().item()

os.makedirs(args.output_dir, exist_ok=True)
results = []

for idx, batch in enumerate(tqdm(loader)):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    logits_ds = get_logits(deepseek, input_ids, attention_mask)
    logits_qw = get_logits(qwen, input_ids, attention_mask)

    mask, avg_kl = disagreement_mask(logits_ds, logits_qw)
    prompt_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

    results.append({
        "prompt": prompt_text,
        "disagreement_score": avg_kl,
        "used_tokens_pct": mask.float().mean().item() * 100
    })

    if (idx + 1) % args.checkpoint_every == 0:
        torch.save(
            results,
            os.path.join(args.output_dir, f"disagreement_scores_checkpoint_{idx+1}.pt")
        )

torch.save(results, os.path.join(args.output_dir, "disagreement_scores.pt"))

with open(os.path.join(args.output_dir, "disagreement_scores.txt"), "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"{r['prompt']}\n{r['disagreement_score']:.6f},{r['used_tokens_pct']:.2f}\n\n")

print(len(results))

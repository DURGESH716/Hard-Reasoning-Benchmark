#!/usr/bin/env bash
set -e

echo "🔹 Evaluating Qwen3"
python src/eval_qwen3_on_hrb_all.py

echo "🔹 Evaluating DeepSeek"
python src/eval_deepseek_on_hrb_all.py

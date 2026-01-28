#!/usr/bin/env bash
set -e

echo "🔹 Running disagreement computation"
python src/frame.py
python src/txt_csv.py

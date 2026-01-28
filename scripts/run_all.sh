#!/usr/bin/env bash
set -e

bash scripts/run_disagreement.sh
bash scripts/build_benchmark.sh
bash scripts/eval_all_models.sh

echo "✅ Full pipeline completed"

#!/usr/bin/env bash
# Usage:
#   bash judge_gpt5p1.sh [CUDA_VISIBLE_DEVICES]
#
# Examples:
#   bash judge_gpt5p1.sh 0
#   bash judge_gpt5p1.sh 0,1
#   CUDA_VISIBLE_DEVICES=2 bash judge_gpt5p1.sh
#
# First run: automatically creates $ROOT_DIR/.venv_judge and installs
#   requirements_judge.txt. Torch may be CPU-only — reinstall with the
#   right CUDA wheel if needed:
#     .venv_judge/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# CUDA_VISIBLE_DEVICES: first positional arg overrides env var
if [[ -n "${1:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$1"
fi

# --- Python / venv setup ---
JUDGE_VENV_PATH="${JUDGE_VENV_PATH:-$ROOT_DIR/.venv_judge}"
if [[ ! -x "$JUDGE_VENV_PATH/bin/python3" ]]; then
    echo "No judge venv found. Creating at $JUDGE_VENV_PATH ..."
    python3 -m venv "$JUDGE_VENV_PATH"
    "$JUDGE_VENV_PATH/bin/pip" install --upgrade pip --quiet
    "$JUDGE_VENV_PATH/bin/pip" install -r "$ROOT_DIR/requirements_judge.txt"
    echo "Venv ready."
fi
PYTHON_BIN="${PYTHON_BIN:-$JUDGE_VENV_PATH/bin/python3}"

# --- Paths ---
JUDGE_MODEL_DIR="${JUDGE_MODEL_DIR:-$ROOT_DIR/model/Qwen3.5-27B}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-64}"
STRUCTRAG_ENABLE_THINKING="${STRUCTRAG_ENABLE_THINKING:-0}"

INPUT_FILE="$SCRIPT_DIR/loong_gpt5p1_99_results.jsonl"
LLM_JUDGE_OUTPUT="$SCRIPT_DIR/lambo_v2_llm_judge.json"
STRUCTURED_EVAL_OUTPUT="$SCRIPT_DIR/structured_eval.json"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "python_bin=$PYTHON_BIN"
echo "judge_model_dir=$JUDGE_MODEL_DIR"
echo "input_file=$INPUT_FILE"

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: input file not found: $INPUT_FILE"
    exit 1
fi

if [[ ! -d "$JUDGE_MODEL_DIR" ]]; then
    echo "ERROR: judge model not found: $JUDGE_MODEL_DIR"
    echo "Set JUDGE_MODEL_DIR=/path/to/Qwen3.5-27B and retry."
    exit 1
fi

"$PYTHON_BIN" - <<PY
import json
import sys
from pathlib import Path

lambo_root = Path(r"$ROOT_DIR/vendor/LAMBO")
input_file = Path(r"$INPUT_FILE")
llm_judge_output = Path(r"$LLM_JUDGE_OUTPUT")
structured_eval_output = Path(r"$STRUCTURED_EVAL_OUTPUT")
judge_model_dir = r"$JUDGE_MODEL_DIR"
enable_thinking = r"$STRUCTRAG_ENABLE_THINKING".strip().lower() in {"1", "true", "yes", "on"}

if str(lambo_root) not in sys.path:
    sys.path.insert(0, str(lambo_root))

# also make structrag_local_backend importable
root = Path(r"$ROOT_DIR")
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lambo_v2.backend import QwenLocalClient
from lambo_v2.eval.llm_judge import run_llm_judge
from lambo_v2.eval.structured_eval import evaluate_predictions

rows = [
    json.loads(line)
    for line in input_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(f"loaded_rows={len(rows)}", flush=True)

# structured eval (EM-style)
print("Running structured eval...", flush=True)
structured_summary = evaluate_predictions(rows)
structured_eval_output.write_text(
    json.dumps(structured_summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"structured_exact_match_rate={structured_summary['exact_match_rate']:.4f}", flush=True)
print(f"structured_eval_output={structured_eval_output}", flush=True)

# LLM judge
print("Loading Qwen judge model...", flush=True)
client = QwenLocalClient(
    model_dir=judge_model_dir,
    max_output_tokens=int(r"$JUDGE_MAX_NEW_TOKENS"),
    max_input_tokens=32768,
    compute_dtype="bfloat16",
    enable_thinking=enable_thinking,
)
print("Running LLM judge...", flush=True)
judge_out = run_llm_judge(llm=client, prediction_rows=rows)
llm_judge_output.write_text(
    json.dumps(judge_out, ensure_ascii=False, indent=2), encoding="utf-8"
)
summary = judge_out.get("summary", {})
print(f"llm_judge_output={llm_judge_output}", flush=True)
print(f"llm_judge_total={summary.get('total')}", flush=True)
print(f"llm_judge_avg_score={summary.get('avg_score'):.2f}", flush=True)
print(f"llm_judge_scoring_success_rate={summary.get('scoring_success_rate'):.4f}", flush=True)
print(f"llm_judge_perfect_rate={summary.get('perfect_rate'):.4f}", flush=True)
PY

echo "Done."

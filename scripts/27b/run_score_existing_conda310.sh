#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-structrag310}"
CONDA_PYTHON_VERSION="${CONDA_PYTHON_VERSION:-3.10.14}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
DOWNLOAD_MODEL_IF_MISSING="${DOWNLOAD_MODEL_IF_MISSING:-1}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/model/Qwen3.5-27B}"
START_SERVER="${START_SERVER:-1}"

model_ready() {
    local model_dir="$1"
    [[ -d "$model_dir" ]] || return 1
    [[ -f "$model_dir/config.json" ]] || return 1
    if [[ ! -f "$model_dir/tokenizer.json" && ! -f "$model_dir/tokenizer.model" && ! -f "$model_dir/tokenizer_config.json" ]]; then
        return 1
    fi
    if [[ ! -f "$model_dir/model.safetensors.index.json" ]] && ! compgen -G "$model_dir/*.safetensors" >/dev/null; then
        return 1
    fi
    return 0
}

usage() {
    cat <<EOF
Usage:
  bash scripts/27b/run_score_existing_conda310.sh
  CONDA_ENV_NAME=structrag310 bash scripts/27b/run_score_existing_conda310.sh
  MODEL_DIR=/path/to/Qwen3.5-27B bash scripts/27b/run_score_existing_conda310.sh
  START_SERVER=1 bash scripts/27b/run_score_existing_conda310.sh

Behavior:
  1. Creates or reuses a conda env
  2. Uses Python $CONDA_PYTHON_VERSION
  3. Installs requirements.txt
  4. Downloads Qwen3.5-27B if needed
  5. Re-runs judge/scoring from the bundled completed inference outputs

Defaults:
  CONDA_ENV_NAME=$CONDA_ENV_NAME
  CONDA_PYTHON_VERSION=$CONDA_PYTHON_VERSION
  INSTALL_REQUIREMENTS=$INSTALL_REQUIREMENTS
  DOWNLOAD_MODEL_IF_MISSING=$DOWNLOAD_MODEL_IF_MISSING
  MODEL_DIR=$MODEL_DIR
  START_SERVER=$START_SERVER
EOF
}

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found."
    echo "Install Miniconda/Anaconda first, then rerun this script."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fx "$CONDA_ENV_NAME" >/dev/null 2>&1; then
    echo "Creating conda env: $CONDA_ENV_NAME (python=$CONDA_PYTHON_VERSION)"
    conda create -y -n "$CONDA_ENV_NAME" "python=$CONDA_PYTHON_VERSION"
else
    echo "Using existing conda env: $CONDA_ENV_NAME"
fi

conda activate "$CONDA_ENV_NAME"

PYTHON_BIN="$(command -v python)"
SERVER_PYTHON_BIN="$PYTHON_BIN"

echo "python_bin=$PYTHON_BIN"
python --version

ACTIVE_PYTHON_VERSION="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"
if [[ "$ACTIVE_PYTHON_VERSION" != "$CONDA_PYTHON_VERSION" ]]; then
    echo "Activated conda env has Python $ACTIVE_PYTHON_VERSION, expected $CONDA_PYTHON_VERSION"
    echo "Use a different CONDA_ENV_NAME or remove the old env and rerun."
    exit 1
fi

if [[ "$INSTALL_REQUIREMENTS" == "1" ]]; then
    python -m pip install -U pip
    python -m pip install -r "$ROOT_DIR/requirements.txt"
else
    echo "INSTALL_REQUIREMENTS=0, skipping pip install -r requirements.txt"
fi

if [[ "$DOWNLOAD_MODEL_IF_MISSING" == "1" ]]; then
    if ! model_ready "$MODEL_DIR"; then
        echo "Model missing or incomplete, downloading first: $MODEL_DIR"
        PYTHON_BIN="$PYTHON_BIN" MODEL_DIR="$MODEL_DIR" bash "$ROOT_DIR/scripts/27b/download_model.sh"
    else
        echo "Model already present: $MODEL_DIR"
    fi
else
    echo "DOWNLOAD_MODEL_IF_MISSING=0, skipping model download check"
fi

echo ""
echo "Starting judge/scoring from existing inference outputs..."
PYTHON_BIN="$PYTHON_BIN" \
SERVER_PYTHON_BIN="$SERVER_PYTHON_BIN" \
MODEL_DIR="$MODEL_DIR" \
START_SERVER="$START_SERVER" \
bash "$ROOT_DIR/scripts/27b/run_score_existing.sh" --latest

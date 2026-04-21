#!/usr/bin/env bash

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/leadawon/structrag_inference.git}"
CLONE_DIR="${CLONE_DIR:-/workspace/structrag_inference}"

if [[ -d "$CLONE_DIR/.git" ]]; then
    echo "Using existing clone: $CLONE_DIR"
    git -C "$CLONE_DIR" pull --ff-only
else
    echo "Cloning repo to: $CLONE_DIR"
    git clone "$REPO_URL" "$CLONE_DIR"
fi

exec bash "$CLONE_DIR/scripts/27b/run_score_existing_conda310.sh"

# StructRAG 27B Judge Bundle

This folder is a portable 27B-only bundle for re-running judge/scoring on the completed StructRAG `exper99` outputs.

It includes:

- the completed 27B inference outputs we already generated
- the bundled `exper99` 99-sample dataset subset
- the local 27B judge helper scripts
- the minimal LAMBO v2 files needed for subset validation and judge/scoring
- a judge-focused `requirements.txt`

It does not include:

- the Qwen3.5-27B model weights
- the full 1.1GB Loong source dataset

The bundle is intentionally centered on the finished `exper99` run plus judge-only reruns.

## Quick Start

If you already cloned this repo and just want the shortest path:

```bash
bash bootstrap_clone_conda_judge.sh
```

That script updates the repo if needed, creates or reuses a conda env with Python 3.10.14, downloads the model if needed, and runs judge/scoring from the bundled outputs.

If you prefer to do it step by step:

```bash
eval "$(conda shell.bash hook)"
conda create -y -n structrag310 python=3.10.14
conda activate structrag310
pip install -U pip
pip install -r requirements.txt
```

Download the 27B model if it is not present:

```bash
bash scripts/27b/download_model.sh
```

Run judge/scoring only:

```bash
bash scripts/27b/run_score_existing.sh --latest
```

No vLLM server is used. Judge/scoring runs with local `transformers` inference directly from `model/Qwen3.5-27B`.

## One-Command Conda Setup

If you want one repo-local command after clone:

```bash
bash scripts/27b/run_score_existing_conda310.sh
```

Preview only:

```bash
DRY_RUN=1 bash scripts/27b/run_score_existing.sh --latest
```

## Main Scripts

- `bash scripts/27b/download_model.sh`
  - downloads `Qwen/Qwen3.5-27B` into `model/Qwen3.5-27B`
- `bash scripts/27b/run_score_existing.sh --latest`
  - reuses finished inference outputs and runs judge/scoring only
- `bash scripts/27b/run_score_existing_conda310.sh`
  - creates/reuses a conda env and then runs judge/scoring only
- `bash bootstrap_clone_conda_judge.sh`
  - clones or updates the repo and then runs the conda-based judge workflow

## Included Result Folder

The current completed 27B inference outputs are in:

```text
eval_results/qwen/loong_exper99_ts-20260420T102150Z_act-exper99_api-qwen3-5-27b_tok-qwen3-5-27b_gen-qwen2_eval-qwen-local-judge_guide-lm-format-enforcer_url-127-0-0-1-1225_lbl-qwen35-think-off
```

and the corresponding merged generate file is in:

```text
loong/Loong/output/qwen_ts-20260420T102150Z_act-exper99_api-qwen3-5-27b_tok-qwen3-5-27b_gen-qwen2_eval-qwen-local-judge_guide-lm-format-enforcer_url-127-0-0-1-1225_lbl-qwen35-think-off
```

## Notes

- `think` is configured off for the bundled 27B workflow.
- judge/scoring now uses local `transformers` inference directly and does not require `vllm`.
- `run_score_existing.sh` is patched to tolerate the old absolute paths stored in the copied `run_manifest.json` and resolve them to this local bundle.
- The bundled Loong data file is the `exper99` subset only, not the full original dataset.

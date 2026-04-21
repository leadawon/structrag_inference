# StructRAG 27B Inference Bundle

This folder is a portable 27B-only bundle for the StructRAG `exper99` run.

It includes:

- the StructRAG code paths used for 27B inference
- the bundled `exper99` 99-sample dataset subset
- the local 27B helper scripts
- the minimal LAMBO v2 files needed for subset validation and judge/scoring
- the completed 27B inference outputs we already generated
- `requirements.txt` from the `structrag` virtual environment

It does not include:

- the Qwen3.5-27B model weights
- the full 1.1GB Loong source dataset

The bundle is intentionally centered on the `exper99` workflow.

## Quick Start

From this directory:

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

Download the 27B model:

```bash
bash scripts/27b/download_model.sh
```

Run the full 27B `exper99` inference:

```bash
bash scripts/27b/run_inference_exper99.sh
```

That script uses the bundled 99-sample subset, runs StructRAG inference, and if GPU is still available for judge at the end, continues into scoring.

## Re-run Judge Later From Finished Outputs

If inference is already finished and you only want to run judge/scoring later:

```bash
START_SERVER=1 bash scripts/27b/run_score_existing.sh --latest
```

Preview only:

```bash
DRY_RUN=1 bash scripts/27b/run_score_existing.sh --latest
```

If you prefer `conda` with Python 3.10.14 and want one command that sets up the env and runs judge/scoring:

```bash
bash scripts/27b/run_score_existing_conda310.sh
```

## Main Scripts

- `bash scripts/27b/download_model.sh`
  - downloads `Qwen/Qwen3.5-27B` into `model/Qwen3.5-27B`
- `bash scripts/27b/run_server.sh --detach`
  - starts the local 27B vLLM server only
- `bash scripts/27b/run_inference_exper99.sh`
  - main entrypoint for the 99-sample 27B run
- `START_SERVER=1 bash scripts/27b/run_score_existing.sh --latest`
  - reuses finished inference outputs and runs judge/scoring only

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
- `run_score_existing.sh` is patched to tolerate the old absolute paths stored in the copied `run_manifest.json` and resolve them to this local bundle.
- The bundled Loong data file is the `exper99` subset only, not the full original dataset.

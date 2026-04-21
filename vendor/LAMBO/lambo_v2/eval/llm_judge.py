"""Loong-style LLM judge using the local transformers backend."""

from __future__ import annotations

import json
import os
import re
from statistics import mean
import time
from typing import Any, Dict, List, Optional

from ..backend import QwenLocalClient


JUDGE_PROMPT = """You are grading an assistant answer against a gold answer.

Criteria:
- Accuracy and hallucinations
- Completeness

Score from 1 to 100.
Use 100 only if the assistant answer is fully correct.

Return exactly one line in this format and nothing else:
Rating: [[score]]

[Question]
{question}

[Gold Answer]
{gold}

[Assistant Answer]
{prediction}
"""


def _extract_score(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"\[\[([0-9]*\.?[0-9]+)\]\]", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\[([0-9]*\.?[0-9]+)\]", text)
    if m:
        return float(m.group(1))
    for pattern in (
        r"(?i)\b(?:rating|score|overall score)\s*[:=]\s*([0-9]{1,3}(?:\.[0-9]+)?)\b",
        r"(?i)\b([0-9]{1,3}(?:\.[0-9]+)?)\s*/\s*100\b",
        r"(?i)\b(?:rating|score|overall score)\D{0,12}\b([0-9]{1,3}(?:\.[0-9]+)?)\b",
    ):
        m = re.search(pattern, text)
        if m:
            value = float(m.group(1))
            if 0 <= value <= 100:
                return value
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _repair_score(
    *,
    llm: QwenLocalClient,
    raw_evaluation: str,
) -> tuple[Optional[float], str]:
    repair_prompt = f"""The following evaluation may not follow the required score format.
Extract the final score as a number from 1 to 100.
If the score is implied but not explicitly bracketed, infer the intended numeric score from the evaluation text.
Return exactly one line in this format and nothing else:
Rating: [[score]]

[Evaluation]
{raw_evaluation}
"""
    repaired = llm.generate_text(
        system_prompt="You normalize evaluator outputs into a single numeric score.",
        user_prompt=repair_prompt,
        max_output_tokens=32,
    )
    return _extract_score(repaired), repaired


def run_llm_judge(
    *,
    llm: QwenLocalClient,
    prediction_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    verdicts: List[Dict[str, Any]] = []
    scores: List[float] = []
    total = len(prediction_rows)
    progress_every = max(1, int(os.environ.get("LLM_JUDGE_PROGRESS_EVERY", "1")))
    started_at = time.time()
    print(f"llm_judge_total={total}", flush=True)

    for index, row in enumerate(prediction_rows, start=1):
        sample_started_at = time.time()
        question = str(row.get("question", ""))
        instruction = str(row.get("instruction", ""))
        gold = _stringify(row.get("answer", ""))
        prediction = _stringify(row.get("generate_response", ""))
        full_question = question if not instruction else f"{question}\n\n[Instruction]\n{instruction}"
        prompt = JUDGE_PROMPT.format(
            question=full_question,
            gold=gold,
            prediction=prediction,
        )
        raw = llm.generate_text(
            system_prompt=(
                "You are a strict, impartial evaluator. "
                "Do not explain your reasoning. "
                "Do not restate the task. "
                "Output exactly one line: Rating: [[score]]."
            ),
            user_prompt=prompt,
            max_output_tokens=64,
            metadata={"module": "llm_judge", "sample_id": row.get("sample_id", "")},
        )
        score = _extract_score(raw)
        repair_raw = None
        if score is None:
            score, repair_raw = _repair_score(llm=llm, raw_evaluation=raw)
        if score is not None:
            scores.append(score)
        verdicts.append(
            {
                "sample_id": row.get("sample_id"),
                "id": row.get("id"),
                "type": row.get("type"),
                "level": row.get("level"),
                "score": score,
                "gold": gold,
                "prediction": prediction,
                "raw": raw,
                "repair_raw": repair_raw,
            }
        )

        if index % progress_every == 0 or index == total:
            sample_elapsed = time.time() - sample_started_at
            total_elapsed = time.time() - started_at
            if score is None:
                preview = re.sub(r"\s+", " ", raw).strip()[:200]
                print(
                    "llm_judge_parse_warning="
                    f"{index}/{total} "
                    f"sample_id={row.get('id', '')} "
                    f"raw_preview={preview}",
                    flush=True,
                )
            print(
                "llm_judge_progress="
                f"{index}/{total} "
                f"sample_id={row.get('id', '')} "
                f"score={score} "
                f"sample_sec={sample_elapsed:.1f} "
                f"elapsed_min={total_elapsed / 60:.1f}",
                flush=True,
            )

    effective = len(scores)
    perfect = sum(1 for s in scores if s == 100)
    summary = {
        "total": total,
        "scoring_success_rate": (effective / total) if total else 0.0,
        "avg_score": mean(scores) if scores else 0.0,
        "perfect_count": perfect,
        "perfect_rate": (perfect / effective) if effective else 0.0,
    }
    return {"summary": summary, "verdicts": verdicts}

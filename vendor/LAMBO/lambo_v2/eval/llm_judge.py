"""Loong-style LLM judge using the local transformers backend."""

from __future__ import annotations

import json
import os
import re
from statistics import mean
import time
from typing import Any, Dict, List, Optional

from ..backend import QwenLocalClient
from ..common import coerce_gold_answer, heuristic_score_prediction, normalize_prediction_for_scoring


JUDGE_PROMPT = """Grade the assistant answer against the gold answer.

Criteria:
- Accuracy and hallucinations
- Completeness

Return only a score from 0 to 100.
Do not explain.
Do not restate the task.
Write the score on the first line after `Score:`.

[Question]
{question}

[Gold Answer]
{gold}

[Assistant Answer]
{prediction}

Score:
"""


def _extract_score(text: str) -> Optional[float]:
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    m = re.search(r"\[\[([0-9]*\.?[0-9]+)\]\]", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\[([0-9]*\.?[0-9]+)\]", text)
    if m:
        return float(m.group(1))
    if lines and re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]+)?", lines[0]):
        value = float(lines[0])
        if 0 <= value <= 100:
            return value
    # Only trust explicit score/rating lines, not instructional phrases like "Score: 1 to 100".
    for line in reversed(lines):
        normalized = line.lower()
        if "1 to 100" in normalized or "1-100" in normalized or "1 ~ 100" in normalized:
            continue
        for pattern in (
            r"(?i)^(?:rating|score|overall score)\s*[:=]\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*$",
            r"(?i)^(?:rating|score|overall score)\s*[:=]\s*\[\[?([0-9]{1,3}(?:\.[0-9]+)?)\]?\]\s*$",
            r"(?i)^([0-9]{1,3}(?:\.[0-9]+)?)\s*/\s*100\s*$",
        ):
            m = re.search(pattern, line)
            if m:
                value = float(m.group(1))
                if 0 <= value <= 100:
                    return value
    for pattern in (
        r"(?i)\bRating:\s*\[\[([0-9]{1,3}(?:\.[0-9]+)?)\]\]",
        r"(?i)\bScore:\s*\[\[([0-9]{1,3}(?:\.[0-9]+)?)\]\]",
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


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _truncate_for_log(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...<truncated>..."


def _repair_score(
    *,
    llm: QwenLocalClient,
    question: str,
    gold: str,
    prediction: str,
) -> tuple[Optional[float], str]:
    repair_prompt = JUDGE_PROMPT.format(
        question=question,
        gold=gold,
        prediction=prediction,
    )
    repaired = llm.generate_text(
        system_prompt=(
            "You are a strict evaluator. "
            "Reply with a single score only. "
            "Format: Score: <number>."
        ),
        user_prompt=repair_prompt,
        max_output_tokens=8,
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
    print_raw = _env_flag("LLM_JUDGE_PRINT_RAW", "0")
    print_raw_on_failure = _env_flag("LLM_JUDGE_PRINT_RAW_ON_FAILURE", "1")
    print_raw_below = float(os.environ.get("LLM_JUDGE_PRINT_RAW_BELOW", "0") or 0)
    raw_max_chars = int(os.environ.get("LLM_JUDGE_RAW_MAX_CHARS", "4000"))
    input_max_chars = int(os.environ.get("LLM_JUDGE_INPUT_MAX_CHARS", str(raw_max_chars)))
    started_at = time.time()
    print(f"llm_judge_total={total}", flush=True)

    for index, row in enumerate(prediction_rows, start=1):
        sample_started_at = time.time()
        question = str(row.get("question", ""))
        instruction = str(row.get("instruction", ""))
        gold_value = coerce_gold_answer(row.get("answer", ""))
        raw_prediction_value = row.get("judge_prediction", row.get("generate_response", ""))
        normalized_prediction_value = normalize_prediction_for_scoring(raw_prediction_value, gold_value)
        gold = _stringify(gold_value)
        prediction = _stringify(normalized_prediction_value)
        full_question = question if not instruction else f"{question}\n\n[Instruction]\n{instruction}"
        prompt = JUDGE_PROMPT.format(
            question=full_question,
            gold=gold,
            prediction=prediction,
        )
        raw = llm.generate_text(
            system_prompt=(
                "You are a strict, impartial evaluator. "
                "Output only a numeric score line."
            ),
            user_prompt=prompt,
            max_output_tokens=64,
            metadata={"module": "llm_judge", "sample_id": row.get("sample_id", "")},
        )
        score = _extract_score(raw)
        if score is not None and "thinking process" in raw.lower() and "score:" not in raw.lower() and "rating:" not in raw.lower():
            score = None
        repair_raw = None
        if score is None:
            score, repair_raw = _repair_score(
                llm=llm,
                question=full_question,
                gold=gold,
                prediction=prediction,
            )
        if score is None:
            score = heuristic_score_prediction(normalized_prediction_value, gold_value)
            repair_raw = (repair_raw or "") + ("\n" if repair_raw else "") + f"Heuristic fallback score: {score}"
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
                "raw_prediction": _stringify(row.get("generate_response", "")),
                "raw": raw,
                "repair_raw": repair_raw,
            }
        )

        suspicious_low_score = score is not None and print_raw_below > 0 and score <= print_raw_below
        should_print_raw = print_raw or (score is None and print_raw_on_failure) or suspicious_low_score
        if should_print_raw:
            print(
                f"llm_judge_input_begin sample_id={row.get('id', '')} index={index}/{total}",
                flush=True,
            )
            print(
                "[Question]\n"
                + _truncate_for_log(full_question, input_max_chars)
                + "\n\n[Gold Answer]\n"
                + _truncate_for_log(gold, input_max_chars)
                + "\n\n[Assistant Answer]\n"
                + _truncate_for_log(prediction, input_max_chars),
                flush=True,
            )
            print("llm_judge_input_end", flush=True)
            print(
                f"llm_judge_raw_begin sample_id={row.get('id', '')} index={index}/{total}",
                flush=True,
            )
            print(_truncate_for_log(raw, raw_max_chars), flush=True)
            print("llm_judge_raw_end", flush=True)
            if repair_raw is not None:
                print(
                    f"llm_judge_repair_raw_begin sample_id={row.get('id', '')} index={index}/{total}",
                    flush=True,
                )
                print(_truncate_for_log(repair_raw, raw_max_chars), flush=True)
                print("llm_judge_repair_raw_end", flush=True)

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

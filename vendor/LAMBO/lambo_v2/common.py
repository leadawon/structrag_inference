from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TITLE_START = "<标题起始符>"
TITLE_END = "<标题终止符>"
DOC_END = "<doc终止符>"


def safe_filename(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name or "")).strip("._")
    return safe or "unknown"


def compact_text(text: str, limit: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_docs_bundle(raw_docs: str) -> List[Dict[str, str]]:
    parts = str(raw_docs or "").split(TITLE_START)
    docs: List[Dict[str, str]] = []
    for index, part in enumerate(parts[1:], start=1):
        if TITLE_END not in part:
            continue
        title, content = part.split(TITLE_END, 1)
        normalized_title = normalize_ws(title)
        normalized_content = content.replace(DOC_END, "").strip()
        if not normalized_title:
            continue
        docs.append(
            {
                "doc_id": f"DOC{index}",
                "doc_title": normalized_title,
                "content": normalized_content,
            }
        )
    return docs


def split_sentences(text: str) -> List[str]:
    raw_parts = re.split(r"(?<=[。！？；;.!?])\s+", str(text or "").strip())
    return [part.strip() for part in raw_parts if part and part.strip()]


def split_long_paragraph(text: str, target_chars: int = 1200) -> List[str]:
    compact = str(text or "").strip()
    if not compact:
        return []
    if len(compact) <= target_chars:
        return [compact]

    parts = split_sentences(compact)
    if len(parts) <= 1:
        return [compact[idx : idx + target_chars].strip() for idx in range(0, len(compact), target_chars)]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for part in parts:
        part_len = len(part)
        if current and current_len + part_len > target_chars:
            chunks.append(" ".join(current).strip())
            current = [part]
            current_len = part_len
        else:
            current.append(part)
            current_len += part_len + 1
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def tokenize_query(text: str) -> List[str]:
    raw = re.findall(
        r"[A-Za-z][A-Za-z0-9_.-]{1,}|[\u4e00-\u9fff]{2,}|[0-9]+(?:\.[0-9]+)?%?",
        str(text or ""),
    )
    tokens: List[str] = []
    seen = set()
    for token in raw:
        normalized = token.strip().casefold()
        if len(normalized) <= 1:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
    return tokens


def quoted_terms(text: str) -> List[str]:
    hits = re.findall(r'["""\u2018\u2019\'\'"]([^"""\u2018\u2019\'\'\"]{1,120})["""\u2018\u2019\'\'"]', str(text or ""))
    return [normalize_ws(item) for item in hits if normalize_ws(item)]


def current_query_from_record(question: str, instruction: str) -> str:
    question = normalize_ws(question)
    instruction = normalize_ws(instruction)
    if question:
        return question
    sentences = split_sentences(instruction)
    return sentences[0] if sentences else instruction


def instruction_from_record(instruction: str) -> str:
    return normalize_ws(instruction) or "(none)"


def json_dumps_pretty(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _strip_code_fences(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json_payload(text: str) -> Optional[Any]:
    stripped = _strip_code_fences(text)
    if not stripped:
        return None
    for candidate in (stripped,):
        try:
            return json.loads(candidate)
        except Exception:
            pass

    candidates: List[str] = []
    stack: List[str] = []
    start_idx: Optional[int] = None
    for idx, char in enumerate(stripped):
        if char in "{[":
            if start_idx is None:
                start_idx = idx
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                continue
            stack.pop()
            if start_idx is not None and not stack:
                candidates.append(stripped[start_idx : idx + 1])
                start_idx = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def extract_literal_payload(text: str) -> Optional[Any]:
    payload = extract_json_payload(text)
    if payload is not None:
        return payload

    stripped = _strip_code_fences(text)
    if not stripped:
        return None

    candidates: List[str] = [stripped]
    stack: List[str] = []
    start_idx: Optional[int] = None
    for idx, char in enumerate(stripped):
        if char in "{[":
            if start_idx is None:
                start_idx = idx
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                continue
            stack.pop()
            if start_idx is not None and not stack:
                candidates.append(stripped[start_idx : idx + 1])
                start_idx = None

    for candidate in reversed(candidates):
        try:
            return ast.literal_eval(candidate)
        except Exception:
            continue
    return None


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(str(text or ""))
    if not matches:
        return None
    return matches[-1].strip()


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _label_sort_key(label: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", label)
    if match:
        return int(match.group(1)), label
    return 10**9, label


def _normalize_scalar(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    compact = re.sub(r"\s+", "", text)
    numeric_like = bool(re.fullmatch(r"[()\-+0-9,.$¥€£%元美元股]+", compact))
    if not numeric_like:
        return re.sub(r"\s+", " ", text)
    parenthesized = bool(re.search(r"\(\s*[$¥€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:元|%|美元|股)?\s*\)", text))
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return re.sub(r"\s+", " ", text)
    number = match.group(0).replace(",", "")
    if parenthesized and not number.startswith("-"):
        number = f"-{number}"
    if "$" in text:
        return f"${number}" if not number.startswith("-") else f"-${number[1:]}"
    if "元" in text:
        return f"{number}元"
    if "%" in text:
        return f"{number}%"
    return number


def _normalize_mapping_value(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\#+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("《") and text.endswith("》"):
        title_inner = re.sub(r"\s+", "", text[1:-1])
        text = f"《{title_inner}》"
    return text


def normalize_sequence_values(value: Any) -> List[str]:
    raw_values = value if isinstance(value, list) else [value]
    values: List[str] = []
    seen = set()
    for item in raw_values:
        text = _normalize_mapping_value(item)
        if not text:
            continue
        folded = text.casefold()
        if folded in {"none", "null"} or folded in seen:
            continue
        seen.add(folded)
        values.append(text)
    return values


def normalize_mapping_values(value: Any) -> List[str]:
    return sorted(set(normalize_sequence_values(value)), key=_label_sort_key)


def coerce_gold_answer(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float)):
        return value
    text = str(value or "").strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def extract_loong_score(text: str) -> Optional[float]:
    match = re.search(r"\[\[([0-9]*\.?[0-9]+)\]\]", str(text or ""))
    if match:
        return float(match.group(1))
    match = re.search(r"\[([0-9]*\.?[0-9]+)\]", str(text or ""))
    if match:
        return float(match.group(1))
    return None


def flatten_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        flattened.append(item)
    return flattened


def _extract_numeric_candidate(text: str) -> Optional[str]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return None

    best: Optional[Tuple[int, int, str]] = None
    pattern = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
    for match in pattern.finditer(raw_text):
        token = match.group(0)
        compact = token.replace(",", "")
        if not compact:
            continue
        score = 0
        if "," in token:
            score += 4
        if "." in token:
            score += 2
        if len(re.sub(r"[^0-9]", "", token)) >= 6:
            score += 2
        context = raw_text[max(0, match.start() - 40) : min(len(raw_text), match.end() + 40)].lower()
        if any(keyword in context for keyword in (" is ", " was ", " 为", "是", " amount", "余额", "损失", "收益", "资产", "价值", "期末", "数值")):
            score += 4
        if any(keyword in context for keyword in ("year", "date", "月", "日", "2024", "2023", "2022")):
            score -= 2
        candidate = (score, -match.start(), token)
        if best is None or candidate > best:
            best = candidate

    return best[2] if best is not None else None


def normalize_prediction_for_scoring(prediction: Any, gold_answer: Any) -> Any:
    gold = coerce_gold_answer(gold_answer)
    if prediction is None:
        return prediction

    if isinstance(prediction, (dict, list, int, float)):
        return prediction

    text = str(prediction).strip()
    if not text:
        return text

    if isinstance(gold, dict):
        payload = extract_literal_payload(text)
        if isinstance(payload, dict):
            return payload
        return text

    if isinstance(gold, list):
        payload = extract_literal_payload(text)
        if isinstance(payload, list):
            return payload
        return text

    normalized_gold = _normalize_scalar(gold)
    numeric_like_gold = bool(re.fullmatch(r"[()\-+0-9,.$¥€£%元美元股]+", normalized_gold))
    if numeric_like_gold:
        numeric_candidate = _extract_numeric_candidate(text)
        if numeric_candidate is not None:
            return numeric_candidate

    payload = extract_literal_payload(text)
    if payload is not None and not isinstance(payload, (dict, list)):
        return payload
    return text


def heuristic_score_prediction(prediction: Any, gold_answer: Any) -> float:
    gold = coerce_gold_answer(gold_answer)
    pred = normalize_prediction_for_scoring(prediction, gold)

    if isinstance(gold, dict) and isinstance(pred, dict):
        normalized_prediction = {key: normalize_mapping_values(values) for key, values in pred.items()}
        normalized_gold = {key: normalize_mapping_values(values) for key, values in gold.items()}
        gold_pairs = {(key, value) for key, values in normalized_gold.items() for value in values}
        pred_pairs = {(key, value) for key, values in normalized_prediction.items() for value in values}
        if normalized_prediction == normalized_gold:
            return 100.0
        true_positive = len(gold_pairs & pred_pairs)
        precision = true_positive / len(pred_pairs) if pred_pairs else 0.0
        recall = true_positive / len(gold_pairs) if gold_pairs else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        extra_count = len(pred_pairs - gold_pairs)
        missing_count = len(gold_pairs - pred_pairs)
        score = 100.0 * f1 - 10.0 * extra_count - 6.0 * missing_count
        if true_positive == 0 and pred_pairs:
            score = max(score, 10.0)
        return max(0.0, min(100.0, round(score, 1)))

    if isinstance(gold, list) and isinstance(pred, list):
        normalized_prediction = normalize_sequence_values(pred)
        normalized_gold = normalize_sequence_values(gold)
        if normalized_prediction == normalized_gold:
            return 100.0
        pred_set = set(normalized_prediction)
        gold_set = set(normalized_gold)
        true_positive = len(pred_set & gold_set)
        precision = true_positive / len(pred_set) if pred_set else 0.0
        recall = true_positive / len(gold_set) if gold_set else 0.0
        set_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        position_hits = sum(
            1
            for idx, item in enumerate(normalized_gold)
            if idx < len(normalized_prediction) and normalized_prediction[idx] == item
        )
        position_acc = position_hits / len(normalized_gold) if normalized_gold else 0.0
        extra_count = max(0, len(normalized_prediction) - len(normalized_gold))
        score = 70.0 * set_f1 + 30.0 * position_acc - 15.0 * extra_count
        if true_positive == 0 and normalized_prediction:
            score = max(score, 10.0)
        return max(0.0, min(100.0, round(score, 1)))

    if _normalize_scalar(pred) == _normalize_scalar(gold):
        return 100.0
    if str(pred).strip():
        return 10.0
    return 0.0

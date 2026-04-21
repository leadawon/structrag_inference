import re, json
import numpy as np


def extract_number(text):
    if not text:
        return None
    match = re.search(r'\[\[([0-9]*\.?[0-9]+)\]\]', text)
    if match:
        return float(match.group(1))
    match = re.search(r'\[([0-9]*\.?[0-9]+)\]', text)
    if match:
        return float(match.group(1))
    for pattern in (
        r'(?i)\b(?:rating|score|overall score)\s*[:=]\s*([0-9]{1,3}(?:\.[0-9]+)?)\b',
        r'(?i)\b([0-9]{1,3}(?:\.[0-9]+)?)\s*/\s*100\b',
        r'(?i)\b(?:rating|score|overall score)\D{0,12}\b([0-9]{1,3}(?:\.[0-9]+)?)\b',
    ):
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if 0 <= value <= 100:
                return value
    return None


def failure_prompts(args, tag):
    eval_lines = open(args.old_evaluate_output_path).readlines()
    gen_lines = open(args.old_output_path).readlines()
    scores = []
    effective_samples = []
    no_effective_samples = []
    for line in eval_lines:
        line = json.loads(line.strip())
        if not extract_number(line[tag]) or line['generate_response'] == "":
            no_effective_samples.append(line['id'])
    for line in gen_lines:
        line = json.loads(line.strip())
        if line['id'] in no_effective_samples:
            effective_samples.append(
                {'id': line['id'], 'prompt': line['prompt'], 'question': line['question'], 'answer': line['answer']})
    return effective_samples


def cal_metric(args, tag, level=None, set=None):
    lines = open(args.evaluate_output_path).readlines()
    scores = []
    effective_samples = []
    no_effective_samples = []
    filtered_samples = []
    for line in lines:
        line = json.loads(line.strip())

        _level = line.get("level", None)
        _set = line.get("set", None)
        if level and _level and _level != level:
            continue
        if set and _set and _set != set:
            continue
        filtered_samples.append(line)

        if extract_number(line[tag]) is not None:
            scores.append(extract_number(line[tag]))
            effective_samples.append(line)
        else:
            no_effective_samples.append(line['id'])

    num_full_marks = sum(1 for x in scores if x == 100)
    filtered_count = len(filtered_samples)
    effective_count = len(effective_samples)
    if filtered_count == 0 or effective_count == 0:
        print(f"level: {level}, set: {set}, scoring_success_rate:0.00, avg_score:0.00, perfect_rate_calculation:0/0, perfect_rate:0.00")
        return None

    metric = (
        effective_count / filtered_count,
        float(np.mean(scores)),
        f"{num_full_marks}/{effective_count}",
        num_full_marks / effective_count,
    )

    print(f"level: {level}, set: {set}, scoring_success_rate: {metric[0]:.2f} , avg_score: {metric[1]:.2f} , perfect_rate_calculation: {metric[2]} , perfect_rate: {metric[3]:.2f}")
    return metric

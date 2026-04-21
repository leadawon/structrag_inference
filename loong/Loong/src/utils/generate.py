import json
from tqdm import tqdm
import multiprocessing
import numpy as np
from functools import partial
from decimal import Decimal
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from structrag_local_backend import LocalChatModel

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            try:
                return str(obj, encoding='utf-8')
            except:
                return str(obj, encoding='gbk')
        elif isinstance(obj, Decimal):
            return float(obj)
        # print(obj, type(obj))
        return json.JSONEncoder.default(self, obj)


def _parse_bool(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    value = str(raw_value).strip().lower()
    if value in {"", "auto", "none", "null"}:
        return None
    return value in {"1", "true", "yes", "on"}


def _get_enable_thinking(config):
    run_args = config.get("run_args", {})
    if "enable_thinking" in run_args:
        return _parse_bool(run_args.get("enable_thinking"))
    return _parse_bool(os.environ.get("STRUCTRAG_ENABLE_THINKING", "0"))


def get_api_results(prompt_input, config):
    prompt = prompt_input['prompt']

    if config['type'] == 'local_transformers':
        model_dir = config['args']['model_dir']
        trust_remote_code = bool(config['args'].get('trust_remote_code', False))
        compute_dtype = config['args'].get('compute_dtype', 'bfloat16')
        max_input_tokens = int(config['args'].get('max_input_tokens', 32768))
        max_new_tokens = int(config['run_args'].get('max_new_tokens', 400))
        temperature = float(config['run_args'].get('temperature', 0.0))
        enable_thinking = _get_enable_thinking(config)

        llm = LocalChatModel(
            model_dir=model_dir,
            compute_dtype=compute_dtype,
            enable_thinking=enable_thinking,
            max_input_tokens=max_input_tokens,
            trust_remote_code=trust_remote_code,
        )
        return llm.generate_text(
            system_prompt=config['args'].get('system_prompt', ''),
            user_prompt=prompt,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
        )

    raise ValueError(f"type of {config['type']} is not valid for this judge-only bundle")

def fetch_api_result(prompt_input, config, max_retries=5):
    """Attempt to get a valid result from the API, with a maximum number of retries."""
    for _ in range(max_retries):
        result = get_api_results(prompt_input, config)
        if result: 
            return result
        # Sleep briefly to not hammer the API in case of errors or rate limits
        time.sleep(5) # Uncomment if needed
    return None


def api(prompt, output_path, config, tag):
    response_content = fetch_api_result(prompt, config)
    result = prompt.copy()
    result[tag] = response_content or ""
    with open(output_path, 'a', encoding='utf-8') as fw:
        fw.write(json.dumps(result, ensure_ascii=False) + '\n')


def generate(prompts, config, output_path, process_num, tag):
    if config['type'] == 'local_transformers':
        for prompt in tqdm(prompts, total=len(prompts)):
            api(prompt, output_path=output_path, config=config, tag=tag)
        return

    func = partial(api, output_path=output_path, config=config, tag=tag)
    with multiprocessing.Pool(processes=process_num) as pool:
        for _ in tqdm(pool.imap(func, prompts), total=len(prompts)):
            pass

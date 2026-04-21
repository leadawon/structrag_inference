import json
from tqdm import tqdm
import multiprocessing
import requests
import numpy as np
from functools import partial
from decimal import Decimal
import numpy as np
import time
import os
import sys
from pathlib import Path
from openai import OpenAI
from anthropic import Anthropic
# import google.generativeai as genai

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


def _normalize_auth_header(api_key):
    if not api_key or api_key == "EMPTY":
        return None
    if api_key.lower().startswith("bearer "):
        return api_key
    return f"Bearer {api_key}"


def _extract_chat_content(result):
    if isinstance(result, dict):
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        if "data" in result and isinstance(result["data"], dict):
            return _extract_chat_content(result["data"])
        if "response" in result and isinstance(result["response"], dict):
            return _extract_chat_content(result["response"])
    return None


def _call_custom_chat_api(prompt, config):
    api_url = config["args"]["api_url"].rstrip("/")
    if not api_url.endswith("/api/ask"):
        api_url = f"{api_url}/api/ask"

    headers = {"Content-Type": "application/json"}
    auth_header = _normalize_auth_header(config["args"].get("api_key", ""))
    if auth_header:
        headers["Authorization"] = auth_header

    raw_info = {
        "model": config["args"]["api_name"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config["run_args"]["temperature"],
    }
    enable_thinking = _get_enable_thinking(config)
    if enable_thinking is not None:
        raw_info["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    callback = requests.post(api_url, json=raw_info, headers=headers, timeout=(600, 600))
    callback.raise_for_status()
    result = callback.json()
    content = _extract_chat_content(result)
    if content is None:
        raise ValueError(f"Unexpected API response format from {api_url}: {result}")
    return content


def _call_openai_compatible_api(prompt, config):
    api_key = config["args"].get("api_key", "EMPTY")
    api_url = config["args"].get("api_url", "")
    client = OpenAI(
        api_key=api_key if api_key != "EMPTY" else "EMPTY",
        base_url=api_url if api_url else None,
    )
    create_kwargs = {
        "messages": [{"role": "user", "content": prompt}],
        "model": config["args"]["api_name"],
        "temperature": config["run_args"]["temperature"],
    }
    enable_thinking = _get_enable_thinking(config)
    if enable_thinking is not None:
        create_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    response = client.chat.completions.create(**create_kwargs)
    message = response.choices[0].message
    content = message.content or ""
    if not content:
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        return reasoning
    return content


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

    if config['type'] == 'openai' or config['type'] == 'vllm':
        try:
            api_url = config['args'].get('api_url', '')
            if '/api/ask' in api_url or ('/v1' not in api_url and 'api.openai.com' not in api_url):
                return _call_custom_chat_api(prompt, config)
            return _call_openai_compatible_api(prompt, config)
        except Exception as e:
            print(e)
            return []
        
    elif config['type'] == 'gemini':
        genai.configure(api_key=config['args']['api_key'])

        model = genai.GenerativeModel(name=config['args']['api_name'])
        try:
            response = model.generate_content(prompt,
                        generation_config=genai.types.GenerationConfig(
                        temperature=config['run_args']['temperature']))
            return response.text
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'claude':
        client = Anthropic(api_key=config['args']['api_key'])
        try:
            message = client.messages.create(
                messages=[{"role": "user", "content": prompt,}],
                model=config['args']['api_name'],
            )
            return message.content
        except Exception as e:
            print(e)
            return []
    
    elif config['type'] == 'http':
        headers = {"Content-Type": "application/json",
                "Authorization": config['args']['api_key']}
        raw_info = {
            "model": config['args']['api_name'],
            "messages": [{"role": "user", "content": prompt}],
            "n": 1}
        raw_info.update(config['run_args'])
        enable_thinking = _get_enable_thinking(config)
        if enable_thinking is not None:
            raw_info["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        try:
            callback = requests.post(config['args']['api_url'], data=json.dumps(raw_info, cls=MyEncoder), headers=headers,
                                    timeout=(600, 600))
            result = callback.json()
            # todo: customize the result
            return result['data']['response']['choices'][0]['message']['content']
        except Exception as e:
            print(e)
            return []
        
    else:
        raise f"type of {config['type']} is not valid"

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

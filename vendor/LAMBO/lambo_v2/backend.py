from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

from .common import extract_json_payload

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BUNDLE_ROOT = Path(__file__).resolve().parents[3]
if str(BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(BUNDLE_ROOT))

from structrag_local_backend import LocalChatModel


class QwenLocalClient:
    def __init__(
        self,
        model_dir: Path | str,
        max_output_tokens: int = 2048,
        max_input_tokens: int = 120000,
        per_gpu_max_memory_gib: int = 20,
        load_in_4bit: Optional[bool] = None,
        compute_dtype: str = "bfloat16",
        enable_thinking: Optional[bool] = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model = LocalChatModel(
            model_dir=model_dir,
            compute_dtype=compute_dtype,
            enable_thinking=enable_thinking,
            max_input_tokens=max_input_tokens,
            trust_remote_code=trust_remote_code,
        )
        self.config = {
            "model_dir": str(model_dir),
            "max_output_tokens": max_output_tokens,
            "max_input_tokens": max_input_tokens,
            "compute_dtype": compute_dtype,
            "enable_thinking": enable_thinking,
        }

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        return self.model.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens or self.config["max_output_tokens"],
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, str]:
        raw_text = self.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            metadata=metadata,
        )
        payload = extract_json_payload(raw_text)
        if payload is None:
            repair_prompt = (
                user_prompt
                + "\n\nYour previous answer was not valid JSON. Re-answer with strict JSON only. "
                + "Do not include markdown fences or commentary."
            )
            raw_text = self.generate_text(
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                metadata=metadata,
            )
            payload = extract_json_payload(raw_text)
        return payload, raw_text


_client_cache: dict[str, Any] = {}


def get_default_client(backend: str = "local") -> QwenLocalClient:
    if backend in _client_cache:
        return _client_cache[backend]

    client = QwenLocalClient(
        model_dir=os.getenv("STRUCTRAG_LOCAL_MODEL_DIR", "/workspace/structrag_inference/model/Qwen3.5-27B")
    )
    _client_cache[backend] = client
    return client

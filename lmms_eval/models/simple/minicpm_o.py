"""
MiniCPM-o 4.5 API Model Adapter for lmms-eval

This module provides integration with MiniCPM-o 4.5 model via its custom FastAPI server.
The server exposes multiple endpoints:
  - /v1/chat/completions (JSON with base64 media - unified multimodal)
  - /v1/image/chat (multipart form - image only)
  - /v1/video/chat (multipart form - video only)
  - /v1/audio/chat (multipart form - audio only)

We use the unified JSON endpoint (/v1/chat/completions) for evaluation since it
supports sending image + video + audio simultaneously via base64 encoding.

Usage:
    python -m lmms_eval \
        --model minicpm_o \
        --model_args api_url=$MINICPM_O_API_URL \
        --tasks task_v2_minicpm_o \
        --batch_size 1
"""

import base64
import json
import os
import time
from typing import Dict, List, Tuple, Union

import requests as http_requests
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.env import get_env_value

NUM_SECONDS_TO_SLEEP = 5


@register_model("minicpm_o")
class MiniCPMo(lmms):
    """
    MiniCPM-o 4.5 model adapter using custom FastAPI HTTP endpoint.

    Uses the /v1/chat/completions JSON endpoint which accepts:
    - messages: list of {role, content}
    - images: list of base64-encoded images
    - video_base64: base64-encoded video
    - audio_base64: base64-encoded audio
    - max_new_tokens, temperature, do_sample

    Returns JSON: {"id": str, "text": str, "usage": {"inference_time_s": float}}
    """

    is_simple: bool = False  # Use ConfigurableMessagesTask to receive doc_to_messages

    def __init__(
        self,
        api_url: str = None,
        timeout: int = 300,
        max_retries: int = 5,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        resolved_api_url = api_url or get_env_value(
            ["MINICPM_O_API_URL", "MINICPM_O_BASE_URL"],
            required=True,
            description="MiniCPM-o API URL",
        )
        self.api_url = resolved_api_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.continual_mode = continual_mode

        eval_logger.info(f"Initialized MiniCPMo with endpoint: {self.api_url}")

        # Response caching setup
        if self.continual_mode:
            if response_persistent_folder is None:
                response_persistent_folder = "./logs/minicpm_o_persistent_folder"
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder,
                "minicpm_o_response.json",
            )

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        # Distributed setup
        accelerator = Accelerator()
        self.accelerator = accelerator
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------
    def _parse_messages(self, messages: List[Dict]) -> Tuple[str, str, List]:
        """
        Parse messages format to extract system prompt, user text, and visuals.

        Returns:
            Tuple of (system_text, combined_user_text, all_visuals)
            where all_visuals is a list of (type, path) tuples:
              type is one of: "image", "video", "audio"
        """
        system_texts = []
        all_texts = []
        all_visuals = []

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", [])

            if isinstance(content, str):
                if role == "system":
                    system_texts.append(content)
                else:
                    all_texts.append(f"[{role.capitalize()}]: {content}")
                continue

            if not isinstance(content, list):
                continue

            message_texts = []
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type", "")

                if item_type == "text":
                    text = item.get("text", "")
                    if text:
                        message_texts.append(text)

                elif item_type == "image":
                    url = item.get("url", "")
                    if url and isinstance(url, str):
                        all_visuals.append(("image", url))

                elif item_type == "video":
                    url = item.get("url", "")
                    if url and isinstance(url, str):
                        all_visuals.append(("video", url))

                elif item_type == "audio":
                    url = item.get("url", "")
                    if url:
                        if isinstance(url, str):
                            all_visuals.append(("audio", url))
                        elif isinstance(url, dict) and "path" in url:
                            all_visuals.append(("audio", url["path"]))

            if message_texts:
                if role == "system":
                    system_texts.append(" ".join(message_texts))
                elif role == "user":
                    all_texts.append(f"[User]: {' '.join(message_texts)}")
                elif role == "assistant":
                    all_texts.append(f"[Assistant]: {' '.join(message_texts)}")
                else:
                    all_texts.append(" ".join(message_texts))

        system_text = "\n\n".join(system_texts) if system_texts else ""
        combined_text = "\n\n".join(all_texts)
        return system_text, combined_text, all_visuals

    # ------------------------------------------------------------------
    # File encoding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_file_base64(file_path: str) -> str:
        """Encode a file to base64 string."""
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _is_image(path: str) -> bool:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        return ext in {"jpg", "jpeg", "png", "gif", "bmp", "webp"}

    @staticmethod
    def _is_video(path: str) -> bool:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        return ext in {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"}

    @staticmethod
    def _is_audio(path: str) -> bool:
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        return ext in {"wav", "mp3", "m4a", "flac", "ogg"}

    # ------------------------------------------------------------------
    # Build JSON request payload
    # ------------------------------------------------------------------
    def _build_payload(
        self,
        question: str,
        visuals: List[Tuple[str, str]],
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict:
        """
        Build JSON payload for the /v1/chat/completions endpoint.

        The MiniCPM-o server supports:
        - messages: [{"role": "user", "content": "..."}]
        - images: [base64_str, ...]  (list of base64 encoded images)
        - video_base64: base64_str   (single video)
        - audio_base64: base64_str   (single audio)
        """
        payload = {
            "messages": [{"role": "user", "content": question}],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }

        images_b64 = []
        video_b64 = None
        audio_b64 = None

        for vtype, vpath in visuals:
            if not os.path.exists(vpath):
                eval_logger.warning(f"File not found: {vpath}")
                continue

            if vtype == "image":
                images_b64.append(self._encode_file_base64(vpath))
            elif vtype == "video":
                # Only use the last video (server supports one)
                video_b64 = self._encode_file_base64(vpath)
            elif vtype == "audio":
                # Only use the last audio
                audio_b64 = self._encode_file_base64(vpath)

        if images_b64:
            payload["images"] = images_b64
        if video_b64:
            payload["video_base64"] = video_b64
        if audio_b64:
            payload["audio_base64"] = audio_b64

        return payload

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    def generate_until(self, requests) -> List[str]:
        """Generate responses for a list of requests."""
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="MiniCPM-o Responding",
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for request in requests:
            args = request.args

            # Parse request arguments
            if len(args) >= 6 and callable(args[1]):
                # Messages mode
                ctx, doc_to_messages_func, gen_kwargs, doc_id, task, split = args
                messages = doc_to_messages_func(self.task_dict[task][split][doc_id])
                system_prompt, user_text, visuals = self._parse_messages(messages)
            else:
                eval_logger.error(f"Unexpected args format: {args}")
                res.append("")
                pbar.update(1)
                continue

            # Check cache
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            # Combine system prompt and user text as the question
            question_parts = []
            if system_prompt:
                question_parts.append(
                    f"[System Instructions]\n{system_prompt}\n[End System Instructions]"
                )
            if user_text:
                question_parts.append(user_text)
            question = "\n\n".join(question_parts)

            max_new_tokens = gen_kwargs.get("max_new_tokens", 4096)
            temperature = gen_kwargs.get("temperature", 0)

            # Build JSON payload
            payload = self._build_payload(question, visuals, max_new_tokens, temperature)

            # Retry logic
            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    resp = http_requests.post(
                        f"{self.api_url}/v1/chat/completions",
                        json=payload,
                        timeout=self.timeout,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    # Support both custom server {"text": "..."} and
                    # OpenAI-compatible {"choices": [{"message": {"content": "..."}}]}
                    response_text = result.get("text", "")
                    if not response_text and "choices" in result:
                        choices = result["choices"]
                        if choices and len(choices) > 0:
                            response_text = choices[0].get("message", {}).get("content", "")

                    if response_text:
                        eval_logger.debug(
                            f"MiniCPM-o response (attempt {attempt+1}): {response_text[:200]}..."
                        )
                        break
                    else:
                        eval_logger.warning(
                            f"Attempt {attempt+1}/{self.max_retries}: empty response, retrying..."
                        )
                        if attempt < self.max_retries - 1:
                            time.sleep(NUM_SECONDS_TO_SLEEP)

                except Exception as e:
                    eval_logger.info(f"Attempt {attempt+1}/{self.max_retries} failed: {str(e)}")
                    if attempt < self.max_retries - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:
                        eval_logger.error(
                            f"All {self.max_retries} attempts failed. Last error: {str(e)}"
                        )
                        response_text = ""

            res.append(response_text)
            pbar.update(1)

            # Cache response
            if self.continual_mode:
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation not supported for MiniCPM-o")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("MiniCPM-o API does not support loglikelihood")

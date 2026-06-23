"""AI client utilities for dataset analysis, dynamic model fetching, and TUI recommendation flows."""
from __future__ import annotations

import io
import os
import re
import json
import time
import base64
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from src.utils.cli import (
    console,
    clear_screen,
    get_user_choice,
    get_parameter_value_input,
    make_panel,
    print_stylized_header,
    render_summary_panel,
    expected_error_panel,
    warning_panel,
    TUIState,
    ParameterDefinition,
    NAV_BACK,
    NAV_LIST,
)
from src.config.settings import load_settings
from src.datasets.core import summarize_dataset

# Fallback models in case fetching fails
FALLBACK_MODELS = {
    "Gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
    ],
}

# Gemini models that retain free-tier request quota. Pro models have *no* free
# tier (the API rejects them with "limit: 0"), so we transparently fall back to
# these when a free-tier-only key selects a Pro model.
FREE_TIER_GEMINI_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]


def is_free_tier_gemini_model(name: str) -> bool:
    """Heuristic: a Gemini model is free-tier eligible iff it is a Flash variant.

    Flash and Flash-Lite models have free-tier request quota; Pro models do not
    (the API returns "limit: 0" for them on a free key). Used both for runtime
    fallback and to label/sort models in the settings picker.
    """
    return "flash" in str(name).lower()


SUPPORTED_TRANSFORMS = [
    "HorizontalFlip", "VerticalFlip", "D4", "RandomRotate90", "Transpose", "Rotate",
    "Affine", "RandomScale", "RandomCrop", "Pad", "Perspective", "ElasticTransform",
    "GridDistortion", "OpticalDistortion", "RandomBrightnessContrast", "ColorJitter",
    "Solarize", "Posterize", "Equalize", "ToGray", "HueSaturationValue", "RGBShift",
    "GaussNoise", "ISONoise", "MultiplicativeNoise", "GaussianBlur", "MedianBlur",
    "MotionBlur", "AdvancedBlur", "CLAHE", "Sharpen", "UnsharpMask", "RandomFog",
    "RandomRain", "RandomShadow", "RandomSunFlare", "ImageCompression", "Downscale",
    "CoarseDropout", "ShotNoise"
]


# HTTP status codes worth retrying (rate limits and transient server errors).
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class GeminiFreeTierUnavailable(ValueError):
    """Raised when a Gemini model has no free-tier quota (the API returns
    "limit: 0" for it). Carries the rejected model name so callers can retry
    with a free-tier-eligible model."""

    def __init__(self, model_name: str, message: str = "") -> None:
        self.model_name = model_name
        super().__init__(message or f"Model '{model_name}' has no free-tier quota.")


def _serialize_body(body: Any) -> str:
    """Render a response body as searchable text, regardless of shape."""
    if isinstance(body, (dict, list)):
        try:
            return json.dumps(body)
        except (TypeError, ValueError):
            return str(body)
    return str(body)


def _is_free_tier_unavailable(body: Any) -> bool:
    """True when a Gemini 429 means the model has no free tier (limit: 0), as
    opposed to a transient per-minute rate limit (limit > 0)."""
    text = _serialize_body(body)
    if "free_tier" not in text:
        return False
    # The violation text reads e.g. "limit: 0"; structured details use "limit": 0.
    return bool(re.search(r'limit["\s:]*\s*0\b', text))


def _parse_retry_delay(body: Any) -> str | None:
    """Extract a human-readable retry-after hint from a Gemini error body."""
    text = _serialize_body(body)
    match = (
        re.search(r"retry in ([\d.]+)s", text)
        or re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', text)
    )
    if not match:
        return None
    try:
        return f"{round(float(match.group(1)))}s"
    except (ValueError, TypeError):
        return None


def _extract_api_error(res: Any, default: str = "Unknown API error") -> str:
    """Pull a human-readable error message out of a provider/transport response.

    Handles both API error envelopes ({"error": {"message": ...}}) and the flat
    transport-error shape this module produces ({"error": "<text>"}), plus bare
    strings, so callers never trip over `.get` on a non-dict value.
    """
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        err = res.get("error", res)
        if isinstance(err, dict):
            msg = err.get("message")
            if msg:
                return str(msg)
            return json.dumps(err)[:500]
        if err:
            return str(err)
    return f"{default}: {str(res)[:300]}"


def _do_single_request(url: str, method: str, headers: dict, data: Any, timeout: int) -> tuple[int, Any]:
    """Perform one HTTP request, preferring `requests` and falling back to urllib."""
    # Try using requests first if available
    try:
        import requests
    except ImportError:
        requests = None

    if requests is not None:
        try:
            if method == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
            else:
                response = requests.get(url, headers=headers, timeout=timeout)
            try:
                payload = response.json()
            except ValueError:
                payload = {"error": (response.text or "Non-JSON response")[:1000]}
            return response.status_code, payload
        except Exception as exc:
            # Network/transport failure — fall through to urllib so a single
            # broken backend doesn't take out the whole request.
            return 0, {"error": f"{type(exc).__name__}: {exc}"}

    import urllib.request
    import urllib.error

    req = urllib.request.Request(url, method=method)
    for k, v in headers.items():
        req.add_header(k, v)

    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode("utf-8")
        req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, data=req_data, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8")
            try:
                return status, json.loads(body)
            except json.JSONDecodeError:
                return status, {"error": (body or "Non-JSON response")[:1000]}
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
            return e.code, json.loads(error_body)
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def make_http_request(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    data: Any = None,
    timeout: int = 25,
    max_retries: int = 2,
    backoff: float = 1.5,
) -> tuple[int, Any]:
    """Perform an HTTP request with bounded retry on transient failures.

    Returns (status_code, parsed_body). status_code is 0 when the request never
    reached the server (DNS/connection/timeout). Retries cover transport errors
    and retryable HTTP statuses (429/5xx); a small backoff avoids hammering the
    API on rate limits.
    """
    headers = headers or {}
    last_status, last_body = 0, {"error": "Request was not attempted"}

    for attempt in range(max_retries + 1):
        status, body = _do_single_request(url, method, headers, data, timeout)
        last_status, last_body = status, body
        if status != 0 and status not in _RETRYABLE_STATUS:
            return status, body
        # A free-tier "limit: 0" rejection can never succeed on retry — fail fast
        # instead of burning two more doomed calls plus backoff sleeps.
        if status == 429 and _is_free_tier_unavailable(body):
            return status, body
        if attempt < max_retries:
            time.sleep(backoff * (attempt + 1))

    return last_status, last_body


def _slugify(name: str, fallback: str = "ai_profile", max_length: int = 64) -> str:
    """Reduce an LLM-supplied name to a safe, filesystem-friendly slug.

    Guards against path traversal and invalid filenames: lowercases, replaces any
    run of non-alphanumeric characters with a single underscore, strips leading/
    trailing underscores, and bounds the length. Returns `fallback` if nothing
    usable remains.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    slug = slug[:max_length].strip("_")
    return slug or fallback


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Robustly interpret an LLM-supplied truthy/falsy value as a bool.

    Unlike `bool(x)`, the string "false"/"0"/"no" correctly maps to False.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "on"}:
            return True
        if text in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


def fetch_multimodal_models(provider: str, api_key: str) -> list[str]:
    """Dynamically fetch the latest multimodal models from the provider API."""
    if not api_key:
        return []
    
    if provider.lower() == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        status, res = make_http_request(url, "GET")
        if status == 200 and isinstance(res, dict) and "models" in res:
            models = []
            for m in res["models"]:
                name = m.get("name", "")
                if "models/" in name:
                    name = name.split("models/")[-1]

                # Check for modern multimodal Gemini models that support text/content generation
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods:
                    if "gemini" in name.lower() and not any(x in name.lower() for x in ["embedding", "vision-preview", "aqa"]):
                        models.append(name)
            return sorted(set(models))
        raise ValueError(_extract_api_error(res, "Failed to fetch models from Gemini API"))

    elif provider.lower() == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        status, res = make_http_request(url, "GET", headers=headers)
        if status == 200 and isinstance(res, dict) and "data" in res:
            # Include known multimodal-capable chat families; exclude non-chat
            # endpoints (audio/realtime/embeddings/image/etc.) that can't take
            # an image-in / JSON-out request.
            include = ("gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-4-vision", "chatgpt-4o", "o1", "o3", "o4")
            exclude = ("audio", "realtime", "transcribe", "tts", "embedding", "moderation", "image", "search", "dall-e")
            models = []
            for m in res["data"]:
                model_id = str(m.get("id", ""))
                lower = model_id.lower()
                if any(tok in lower for tok in include) and not any(tok in lower for tok in exclude):
                    models.append(model_id)
            return sorted(set(models))
        raise ValueError(_extract_api_error(res, "Failed to fetch models from OpenAI API"))

    return []


def get_dataset_ai_summary(dataset_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize dataset metadata and prepare 1 random image for multimodal API consumption."""
    path = Path(dataset_path)
    summary = summarize_dataset(path)

    # Collect all image files in a single walk, matching by case-insensitive
    # suffix. This avoids duplicate hits on case-insensitive filesystems and
    # covers every extension the augmentation engine supports (incl. tif/tiff).
    valid_exts = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    all_images = [
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    samples_base64 = []
    image_stats = []
    
    if all_images:
        num_samples = min(1, len(all_images))
        selected_samples = random.sample(all_images, num_samples)
        
        for img_path in selected_samples:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    bands = img.getbands()
                    channels = len(bands)
                    mode = img.mode
                    
                    image_stats.append({
                        "filename": img_path.name,
                        "original_size": f"{width}x{height}",
                        "channels": channels,
                        "color_mode": mode
                    })
                    
                    # Resize to maximum 512px on any dimension to respect payload limits
                    if img.width > 512 or img.height > 512:
                        img.thumbnail((512, 512))
                    
                    buffer = io.BytesIO()
                    img.convert("RGB").save(buffer, format="JPEG", quality=75)
                    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    
                    samples_base64.append({
                        "mime_type": "image/jpeg",
                        "base64_data": b64_data,
                        "filename": img_path.name
                    })
            except Exception:
                continue
                
    dataset_details = {
        "name": summary.name,
        "format": summary.format,
        "task": summary.task,
        "num_classes": len(summary.classes),
        "classes": summary.classes,
        "total_images": summary.image_count,
        "total_annotations": summary.annotation_count,
        "splits": {
            k: {
                "images": v.image_count,
                "annotations": v.annotation_count
            } for k, v in summary.splits.items()
        },
        "sample_image_properties": image_stats
    }
    
    return dataset_details, samples_base64


def query_llm_multimodal(
    provider: str,
    api_key: str,
    model_name: str,
    text_prompt: str,
    samples_base64: list[dict[str, Any]],
    system_instruction: str = ""
) -> str:
    """Call the provider multimodal API and retrieve the text response."""
    if provider.lower() == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

        parts: list[dict[str, Any]] = [{"text": text_prompt}]
        for sample in samples_base64:
            parts.append({
                "inlineData": {
                    "mimeType": sample["mime_type"],
                    "data": sample["base64_data"]
                }
            })

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        status, res = make_http_request(url, "POST", data=payload, timeout=90)
        if status != 200:
            if status == 429 and _is_free_tier_unavailable(res):
                raise GeminiFreeTierUnavailable(model_name)
            message = f"Gemini API Error (status {status}): {_extract_api_error(res)}"
            if status == 429:
                retry_after = _parse_retry_delay(res)
                hint = f" Retry in ~{retry_after}." if retry_after else ""
                message = f"Gemini rate limit hit for '{model_name}'.{hint}"
            raise ValueError(message)
        if not isinstance(res, dict):
            raise ValueError(f"Gemini API returned a non-JSON response: {str(res)[:300]}")

        # Surface prompt-level blocks (e.g. safety filters) with a clear message.
        block_reason = (res.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            raise ValueError(f"Gemini blocked the prompt (reason: {block_reason}).")

        candidates = res.get("candidates") or []
        if not candidates:
            raise ValueError(f"Gemini returned no candidates: {str(res)[:300]}")

        candidate = candidates[0]
        text_chunks = [
            part["text"]
            for part in (candidate.get("content") or {}).get("parts", [])
            if isinstance(part, dict) and "text" in part
        ]
        if text_chunks:
            return "".join(text_chunks)

        finish_reason = candidate.get("finishReason", "UNKNOWN")
        raise ValueError(
            f"Gemini returned an empty response (finishReason: {finish_reason}). "
            "Try a different model or simplify the request."
        )

    elif provider.lower() == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        content_parts = [{"type": "text", "text": text_prompt}]
        
        for sample in samples_base64:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{sample['mime_type']};base64,{sample['base64_data']}"
                }
            })
            
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        messages.append({"role": "user", "content": content_parts})
        
        payload = {
            "model": model_name,
            "response_format": {"type": "json_object"},
            "messages": messages
        }
        
        status, res = make_http_request(url, "POST", headers=headers, data=payload, timeout=90)
        if status != 200:
            raise ValueError(f"OpenAI API Error (status {status}): {_extract_api_error(res)}")
        if not isinstance(res, dict):
            raise ValueError(f"OpenAI API returned a non-JSON response: {str(res)[:300]}")
        try:
            content = res["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = None
        if content:
            return content
        raise ValueError(f"OpenAI API returned an empty/unexpected response: {str(res)[:300]}")

    raise ValueError(f"Unsupported provider: {provider}")


def query_llm_with_free_tier_fallback(
    provider: str,
    api_key: str,
    model_name: str,
    text_prompt: str,
    samples_base64: list[dict[str, Any]],
    system_instruction: str = "",
) -> tuple[str, str]:
    """Call the LLM, transparently retrying with a free-tier model if the chosen
    Gemini model has no free-tier quota.

    Returns (response_text, model_used). When `model_used` differs from
    `model_name`, the caller can surface a notice that it auto-switched.
    """
    try:
        return (
            query_llm_multimodal(
                provider, api_key, model_name, text_prompt, samples_base64, system_instruction
            ),
            model_name,
        )
    except GeminiFreeTierUnavailable:
        if provider.lower() != "gemini":
            raise

    # The selected Gemini model has no free tier — try free-tier Flash models.
    for fallback in FREE_TIER_GEMINI_FALLBACKS:
        if fallback == model_name:
            continue
        try:
            return (
                query_llm_multimodal(
                    provider, api_key, fallback, text_prompt, samples_base64, system_instruction
                ),
                fallback,
            )
        except GeminiFreeTierUnavailable:
            continue

    raise ValueError(
        f"'{model_name}' has no Gemini free-tier quota, and no free-tier fallback "
        f"model was usable. Switch to a free-tier model (e.g. gemini-2.5-flash) in "
        f"Settings → AI Recommendations, or enable billing on your Google API key."
    )


def verify_ai_setup() -> tuple[bool, str, str, str]:
    """Helper to verify settings and prompt user to configure if keys are missing."""
    # Load .env so the environment-variable key fallback works the same way the
    # Roboflow integration does (keys are commonly stored in the project .env).
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(".env"))
    except ImportError:
        pass

    settings = load_settings()
    ai_config = settings.get("ai", {})
    provider = ai_config.get("provider", "Gemini")
    model_name = ai_config.get("selected_model", "gemini-2.5-flash")

    api_key = ""
    if provider.lower() == "gemini":
        api_key = (ai_config.get("gemini_api_key") or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    elif provider.lower() == "openai":
        api_key = (ai_config.get("openai_api_key") or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        console.print(warning_panel(
            f"AI key for [bold cyan]{provider}[/bold cyan] is not configured.\n\n"
            f"Please go to [bold]Main Menu -> Settings -> AI Recommendations[/bold] to configure your API key "
            f"or set the `GEMINI_API_KEY` / `OPENAI_API_KEY` environment variable.",
            title="AI Config Required"
        ))
        input("\nPress Enter to return...")
        return False, "", "", ""
        
    return True, provider, api_key, model_name


def _try_parse_json_object(candidate: str) -> dict | None:
    """Parse `candidate` and return it only if it decodes to a JSON object."""
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_json_block(text: str) -> dict:
    """Extract and parse the first valid JSON object {...} from text.

    Handles LLM markdown decorations, leading/trailing conversational text, and
    whitespace. Only JSON objects are accepted (callers index by key), so a bare
    array or scalar is treated as "not found".
    """
    cleaned = text.strip()

    # 1. Direct parse.
    result = _try_parse_json_object(cleaned)
    if result is not None:
        return result

    # 2. Inside markdown code fences (```json ... ``` or ``` ... ```).
    for block in re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL):
        block = block.strip()
        result = _try_parse_json_object(block)
        if result is not None:
            return result
        first, last = block.find("{"), block.rfind("}")
        if first != -1 and last > first:
            result = _try_parse_json_object(block[first:last + 1])
            if result is not None:
                return result

    # 3. Outermost braces anywhere in the text.
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first != -1 and last > first:
        result = _try_parse_json_object(cleaned[first:last + 1])
        if result is not None:
            return result

    raise ValueError(
        "Failed to locate or parse a valid JSON object in the AI response.\n"
        f"Raw response preview:\n{cleaned[:300]}..."
    )


# Numeric clamp ranges for AI-suggested Ultralytics training params. Values that
# fall outside these bounds would either be rejected by Ultralytics or silently
# produce a broken run, so we clamp rather than trust the model.
_TRAINING_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    # Optimizer
    "lr0": (1e-9, 1.0),
    "lrf": (1e-9, 1.0),
    "momentum": (0.0, 1.0),
    "weight_decay": (0.0, 1.0),
    "warmup_epochs": (0.0, 100.0),
    "warmup_momentum": (0.0, 1.0),
    # Augmentation (probabilities / normalized magnitudes)
    "hsv_h": (0.0, 1.0),
    "hsv_s": (0.0, 1.0),
    "hsv_v": (0.0, 1.0),
    "translate": (0.0, 1.0),
    "scale": (0.0, 1.0),
    "perspective": (0.0, 0.001),
    "flipud": (0.0, 1.0),
    "fliplr": (0.0, 1.0),
    "mosaic": (0.0, 1.0),
    "mixup": (0.0, 1.0),
    "copy_paste": (0.0, 1.0),
    "erasing": (0.0, 1.0),
    # Geometric angles
    "degrees": (-180.0, 180.0),
    "shear": (-180.0, 180.0),
    # Loss weights
    "box": (0.0, 100.0),
    "cls": (0.0, 100.0),
    "dfl": (0.0, 100.0),
    "label_smoothing": (0.0, 0.9),
    # Dataset fraction
    "fraction": (0.01, 1.0),
}

_TRAINING_BOOL_KEYS = {"cos_lr", "rect", "pretrained"}
_VALID_OPTIMIZERS = {"auto", "SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSProp", "MuSGD"}
_VALID_AUTO_AUGMENT = {"randaugment", "autoaugment", "augmix"}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sanitize_training_params(params: dict) -> dict:
    """Sanitize, cast, and clamp AI-suggested training parameters.

    Guarantees strict type safety and valid ranges matching YOLOmatic and
    Ultralytics expectations, so a hallucinated value (e.g. fliplr=2.0,
    cos_lr="false", batch=0) can never produce an invalid or broken config.
    """
    if not isinstance(params, dict):
        return {}

    sanitized: dict[str, Any] = {}
    float_keys = set(_TRAINING_FLOAT_RANGES)
    int_keys = {"epochs", "patience", "imgsz", "close_mosaic"}

    for key, raw in params.items():
        if key in _TRAINING_BOOL_KEYS:
            sanitized[key] = _coerce_bool(raw)
        elif key in float_keys:
            try:
                low, high = _TRAINING_FLOAT_RANGES[key]
                sanitized[key] = _clamp(float(raw), low, high)
            except (ValueError, TypeError):
                continue
        elif key in int_keys:
            try:
                sanitized[key] = int(float(raw))
            except (ValueError, TypeError):
                continue
        elif key == "batch":
            try:
                batch = int(float(raw))
            except (ValueError, TypeError):
                continue
            # -1 means Ultralytics auto-batch; otherwise must be a positive count.
            sanitized[key] = batch if batch == -1 else max(1, batch)
        elif key == "optimizer":
            opt = str(raw).strip()
            if opt in _VALID_OPTIMIZERS:
                sanitized[key] = opt
            else:
                match = next((vo for vo in _VALID_OPTIMIZERS if vo.lower() == opt.lower()), None)
                sanitized[key] = match or "auto"
        elif key == "auto_augment":
            val = str(raw).strip().lower()
            if val in _VALID_AUTO_AUGMENT:
                sanitized[key] = val
            elif val in {"", "none", "null", "false"}:
                sanitized[key] = ""
            # else: drop unknown policy rather than passing a bad value through
        # Unknown keys are intentionally dropped.

    # Post-cast validation for keys with structural constraints.
    if "imgsz" in sanitized:
        val = sanitized["imgsz"]
        sanitized["imgsz"] = max(32, min(2048, ((val + 16) // 32) * 32))

    if "epochs" in sanitized:
        sanitized["epochs"] = max(1, min(10000, sanitized["epochs"]))

    if "patience" in sanitized:
        sanitized["patience"] = max(0, sanitized["patience"])

    if "close_mosaic" in sanitized:
        sanitized["close_mosaic"] = max(0, sanitized["close_mosaic"])

    return sanitized


def _transform_param_defs(name: str) -> dict[str, Any]:
    """Return {param_name: ParameterDefinition} for a transform, excluding the
    shared probability `p`. Empty for transforms that take no extra params."""
    try:
        from src.augmentation.transforms import get_params_for_transform
    except ImportError:  # pragma: no cover - defensive
        return {}
    return {d.name: d for d in get_params_for_transform(name) if d.name != "p"}


def _coerce_param_value(definition: Any, value: Any) -> Any:
    """Cast and clamp a single transform parameter to its declared type/range.

    Returns None when the value cannot be coerced (caller drops it).
    """
    if definition.allowed_values:
        candidate = str(value).strip()
        chosen = candidate if candidate in definition.allowed_values else str(definition.default)
        if definition.value_type == "int":
            try:
                return int(float(chosen))
            except (ValueError, TypeError):
                return int(float(str(definition.default)))
        return chosen

    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return None if definition.value_type in {"int", "float"} else value

    if definition.min_value is not None:
        numeric = max(float(definition.min_value), numeric)
    if definition.max_value is not None:
        numeric = min(float(definition.max_value), numeric)
    return int(numeric) if definition.value_type == "int" else float(numeric)


def _order_range_pairs(defs: dict[str, Any], result: dict[str, Any]) -> None:
    """Ensure low/high (and lower/upper) sibling params are correctly ordered so
    the augmentation engine reassembles valid (low, high) tuples."""
    for lo_suffix, hi_suffix in (("_low", "_high"), ("_lower", "_upper")):
        for key in defs:
            if not key.endswith(lo_suffix):
                continue
            hi_key = key[: -len(lo_suffix)] + hi_suffix
            if hi_key in result and key in result:
                lo_v, hi_v = result[key], result[hi_key]
                if isinstance(lo_v, (int, float)) and isinstance(hi_v, (int, float)) and lo_v > hi_v:
                    result[key], result[hi_key] = hi_v, lo_v


def build_transform_param_catalog() -> dict[str, list[str]]:
    """Human-readable per-transform tunable parameter hints for the AI prompt.

    Only includes transforms that accept extra parameters; the values describe
    the exact keys/types/ranges the augmentation engine understands.
    """
    catalog: dict[str, list[str]] = {}
    for name in SUPPORTED_TRANSFORMS:
        defs = _transform_param_defs(name)
        if not defs:
            continue
        entries = []
        for d in defs.values():
            if d.allowed_values:
                rng = f" (one of {list(d.allowed_values)})"
            elif d.min_value is not None or d.max_value is not None:
                lo = d.min_value if d.min_value is not None else "-inf"
                hi = d.max_value if d.max_value is not None else "+inf"
                rng = f" [{lo}..{hi}]"
            else:
                rng = ""
            entries.append(f"{d.name}:{d.value_type}{rng}")
        catalog[name] = entries
    return catalog


def _format_transform_param_catalog() -> str:
    """Render build_transform_param_catalog() as compact prompt text."""
    catalog = build_transform_param_catalog()
    lines = []
    for name, entries in catalog.items():
        lines.append(f"  - {name}: {', '.join(entries)}")
    return "\n".join(lines)


def sanitize_augmentation_transforms(transforms: list) -> list:
    """Sanitize AI-suggested Albumentations transforms into engine-compatible entries.

    Validates each transform name against the supported catalog, coerces and
    clamps every parameter to the schema the augmentation engine expects
    (including completing low/high pairs from defaults), and drops unknown
    parameter keys. This guarantees the resulting profile actually instantiates
    instead of being silently skipped at augmentation time.
    """
    if not isinstance(transforms, list):
        return []

    supported_set = set(SUPPORTED_TRANSFORMS)
    sanitized_list = []

    for tx in transforms:
        if not isinstance(tx, dict):
            continue

        name = tx.get("name")
        if not name or name not in supported_set:
            continue

        enabled = _coerce_bool(tx.get("enabled", True), default=True)
        try:
            p = _clamp(float(tx.get("p", 0.5)), 0.0, 1.0)
        except (ValueError, TypeError):
            p = 0.5

        sanitized_tx: dict[str, Any] = {"name": name, "enabled": enabled, "p": round(p, 4)}

        defs = _transform_param_defs(name)
        # Seed every known parameter with its default so the engine always gets a
        # complete, valid kwarg set (matches the built-in profile convention).
        for pname, definition in defs.items():
            sanitized_tx[pname] = _coerce_param_value(definition, definition.default)

        # Override defaults with any valid AI-supplied values; drop unknown keys.
        for key, value in tx.items():
            if key in ("name", "enabled", "p") or key not in defs:
                continue
            coerced = _coerce_param_value(defs[key], value)
            if coerced is not None:
                sanitized_tx[key] = coerced

        _order_range_pairs(defs, sanitized_tx)
        sanitized_list.append(sanitized_tx)

    return sanitized_list


def run_ai_recommendation_flow(model_choice: str, dataset_choice: str) -> dict | None:
    """TUI flow that runs AI dataset analysis and suggests a custom training config YAML."""
    ok, provider, api_key, model_name = verify_ai_setup()
    if not ok:
        return None
        
    clear_screen()
    print_stylized_header("AI Dataset Analysis & Training Config recommendation")
    
    # Prompt user for project description and preferences
    desc_param = ParameterDefinition(
        name="project_description",
        category="AI",
        default="Standard object detection project",
        value_type="str",
        description="Describe your dataset and what you are trying to detect",
        help_text="Provide context like: 'Aerial drone images of solar panels' or 'Thermal night vision camera detecting pedestrians'."
    )
    project_desc = get_parameter_value_input(desc_param)
    if project_desc in (None, NAV_BACK, NAV_LIST):
        return None
        
    pref_param = ParameterDefinition(
        name="ai_prompt_preferences",
        category="AI",
        default="prioritize overall mAP50-95",
        value_type="str",
        description="AI Preferences / Custom prompts",
        help_text="Tell the AI any preferences like: 'keep model lightweight', 'prioritize recall', 'noisy labels', 'high imbalance'."
    )
    user_prompt = get_parameter_value_input(pref_param)
    if user_prompt in (None, NAV_BACK, NAV_LIST):
        return None

    # Load dataset details and sample images
    dataset_path = Path(dataset_choice)
    console.print("\n[bold green]Gathering dataset information and sampling images...[/bold green]")
    try:
        dataset_details, samples_base64 = get_dataset_ai_summary(dataset_path)
    except Exception as e:
        console.print(expected_error_panel(f"Failed to scan dataset: {e}"))
        input("\nPress Enter to return...")
        return None

    console.print(f"Scanned [yellow]{dataset_details['total_images']}[/yellow] images across [yellow]{len(dataset_details['splits'])}[/yellow] splits.")
    console.print(f"Sampled [yellow]{len(samples_base64)}[/yellow] images for multimodal analysis.")
    
    # Run the LLM recommendation
    console.print(f"\n[bold green]Consulting AI Expert ({provider} : {model_name})...[/bold green]")
    
    system_instruction = (
        "You are an Elite Principal Computer Vision Architect and Deep Learning Engineer specializing in YOLO architectures, "
        "dataset quality audits, and extreme hyperparameter optimization.\n"
        "Your task is to analyze the provided dataset statistics, visual sample properties, and the user's specific project goals. "
        "Then, critically formulate and recommend the absolute best training hyperparameter configuration tailored exactly to their dataset and model.\n\n"
        "CRITICAL PROFESSIONAL AUDIT GUIDELINES:\n"
        "1. Act like a rigorous, critical senior Computer Vision specialist. Do not blindly suggest default values (e.g., standard AdamW or SGD). "
        "Audit the dataset sizes, class imbalance, input channels, visual lighting/contrast/aspect-ratio, and determine the optimal setup.\n"
        "2. Guard against common training failures:\n"
        "   - Overfitting: If the dataset is small (<1000 images), suggest smaller models or higher weight decay (e.g. 0.001 - 0.01), higher dropout/erasing, and fewer epochs or lower learning rates.\n"
        "   - Small/Tiny Objects: If the average object size is very small, suggest a larger imgsz (e.g., 960 or 1280), lower downscale, and higher dfl/box loss weights to optimize precision.\n"
        "   - Class Imbalance: For highly imbalanced classes, adjust the cls (classification loss weight) upwards, and suggest careful augmentations.\n"
        "   - Color Semantics: If the color of the target objects is critical (e.g., medical pathology, traffic lights), keep 'hsv_h' extremely low or 0.0 so color meanings are preserved.\n"
        "   - Rotational Invariance: Only suggest rotation/flips if it makes sense in the domain (e.g., standard cars do not flip vertically, but cell biology or satellite imagery does).\n"
        "3. You must output your recommendation strictly as a compliant JSON object matching the requested schema."
    )
    
    prompt = (
        f"You are training a '{model_choice}' model.\n"
        f"The user describes their project as: '{project_desc}'\n"
        f"User special instructions & constraints: '{user_prompt}'\n\n"
        f"Dataset structural details:\n"
        f"{json.dumps(dataset_details, indent=2)}\n\n"
        f"=== COMPREHENSIVE TRAINING PARAMETER CATALOG ===\n"
        f"You must select and suggest values for the following keys to construct the optimal training configuration:\n\n"
        f"Core Parameters:\n"
        f"  - 'epochs': integer (1 to 1000, e.g. 100-300 standard)\n"
        f"  - 'patience': integer (0 to 500, early stopping threshold)\n"
        f"  - 'batch': integer (-1 for auto-batching, or 8, 16, 32, 64 standard)\n"
        f"  - 'imgsz': integer (32 to 2048, standard 640. Must be a multiple of 32)\n\n"
        f"Optimizer Parameters:\n"
        f"  - 'optimizer': string ('auto', 'SGD', 'Adam', 'AdamW', 'Adamax', 'NAdam', 'RAdam', 'RMSProp', 'MuSGD')\n"
        f"  - 'lr0': float (initial learning rate, e.g. 0.01 standard)\n"
        f"  - 'lrf': float (final learning rate factor, e.g. 0.01 standard)\n"
        f"  - 'momentum': float (e.g. 0.937 standard)\n"
        f"  - 'weight_decay': float (L2 penalty, e.g. 0.0005 standard)\n"
        f"  - 'warmup_epochs': float (warmup phase, e.g. 3.0 standard)\n"
        f"  - 'warmup_momentum': float (warmup starting momentum, e.g. 0.8 standard)\n"
        f"  - 'cos_lr': boolean (use cosine learning rate scheduler)\n\n"
        f"Augmentation Parameters (Ultralytics built-in):\n"
        f"  - 'hsv_h': float (0.0 to 1.0, color hue shift)\n"
        f"  - 'hsv_s': float (0.0 to 1.0, color saturation shift)\n"
        f"  - 'hsv_v': float (0.0 to 1.0, color brightness shift)\n"
        f"  - 'degrees': float (-180.0 to 180.0, random rotation)\n"
        f"  - 'translate': float (0.0 to 1.0, random translation)\n"
        f"  - 'scale': float (0.0 to 1.0, random scaling zoom factor)\n"
        f"  - 'shear': float (-180.0 to 180.0, random slant)\n"
        f"  - 'perspective': float (0.0 to 0.001, 3D perspective distortion — Ultralytics keeps this tiny)\n"
        f"  - 'flipud': float (0.0 to 1.0, vertical flip probability)\n"
        f"  - 'fliplr': float (0.0 to 1.0, horizontal flip probability)\n"
        f"  - 'mosaic': float (0.0 to 1.0, mosaic augmentation probability)\n"
        f"  - 'mixup': float (0.0 to 1.0, mixup probability)\n"
        f"  - 'copy_paste': float (0.0 to 1.0, copy-paste probability)\n"
        f"  - 'auto_augment': string ('', 'randaugment', 'autoaugment', 'augmix')\n"
        f"  - 'erasing': float (0.0 to 1.0, random erasing probability)\n"
        f"  - 'close_mosaic': integer (0 to 100, epochs to disable mosaic at the end)\n\n"
        f"Loss Weight Parameters:\n"
        f"  - 'box': float (bounding box loss weight, e.g. 7.5 standard)\n"
        f"  - 'cls': float (class classification loss weight, e.g. 0.5 standard)\n"
        f"  - 'dfl': float (distribution focal loss weight, e.g. 1.5 standard)\n"
        f"  - 'label_smoothing': float (0.0 to 0.9, label smoothing value)\n\n"
        f"Advanced/Hardware Parameters:\n"
        f"  - 'rect': boolean (aspect ratio training)\n"
        f"  - 'fraction': float (0.01 to 1.0, dataset fraction to train on)\n"
        f"  - 'pretrained': boolean (start from pretrained COCO weights)\n\n"
        f"=== EXPECTED OUTPUT FORMAT ===\n"
        f"You must return a single JSON object containing exactly three fields:\n"
        f"1. 'suggested_name': a short, lowercase_with_underscores string identifying the project profile (e.g. 'aerial_solar_panels').\n"
        f"2. 'rationale': a highly detailed, professional, multi-line paragraph explaining your critical CV design decisions, "
        f"analyzing the image sample details (channels, lighting, color characteristics, classes) and justifying your parameter deviations from standard defaults.\n"
        f"3. 'training': a dictionary of custom hyperparameter keys chosen from the list above. Please specify values that are strictly within their permitted types and ranges.\n"
    )
    
    try:
        raw_response, model_used = query_llm_with_free_tier_fallback(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            text_prompt=prompt,
            samples_base64=samples_base64,
            system_instruction=system_instruction
        )
        if model_used != model_name:
            console.print(warning_panel(
                f"'{model_name}' has no Gemini free tier, so the request was "
                f"automatically sent using [bold cyan]{model_used}[/bold cyan] instead.\n"
                f"Tip: select a free-tier model in Settings → AI Recommendations to silence this.",
                title="Switched to a free-tier model",
            ))

        # Use our robust block extractor
        recommendation = extract_json_block(raw_response)

        suggested_name = _slugify(
            recommendation.get("suggested_name", ""),
            fallback=_slugify(f"ai_{model_choice}", fallback="ai_config"),
        )
        rationale = recommendation.get("rationale", "Optimized by AI based on dataset analysis.")
        training_overrides = recommendation.get("training", {})
        
        # Sanitize and strictly type-check hyperparameter suggestions
        training_overrides = sanitize_training_params(training_overrides)
        
        # We will create a config using generators and apply the AI overrides
        clear_screen()
        print_stylized_header("AI Recommendation Success")
        
        console.print(make_panel(
            f"[bold green]AI Suggested Profile Name:[/bold green] {suggested_name}\n\n"
            f"[bold green]Rationale:[/bold green]\n{rationale}",
            title="AI Analysis Rationale",
            state=TUIState.SUCCESS
        ))
        
        # Render a summary table of the suggested parameters
        console.print("\n[bold]Suggested Parameters Overrides:[/bold]")
        param_table = {}
        for k, v in training_overrides.items():
            param_table[k] = str(v)
        render_summary_panel("AI Training Parameters", param_table)
        
        choice = get_user_choice(
            ["Apply AI Recommendation and Save Config", "Discard and Return"],
            title="Confirm AI Configuration",
            text="Do you want to write these hyperparameters to a new YAML config file?"
        )
        
        if choice == "Apply AI Recommendation and Save Config":
            return {
                "name": suggested_name,
                "training_overrides": training_overrides,
                "rationale": rationale
            }
            
    except Exception as e:
        console.print(expected_error_panel(f"Failed to fetch AI recommendation: {e}"))
        console.print(traceback.format_exc(), style="dim")
        input("\nPress Enter to return...")
        
    return None


def run_ai_augmentation_flow(dataset_choice: str) -> str | None:
    """TUI flow that runs AI dataset analysis and suggests a custom Albumentations augmentation profile."""
    ok, provider, api_key, model_name = verify_ai_setup()
    if not ok:
        return None
        
    clear_screen()
    print_stylized_header("AI Dataset Analysis & Augmentation Suggestion")
    
    # Prompt user for project description and preferences
    desc_param = ParameterDefinition(
        name="project_description",
        category="AI",
        default="Standard object detection project",
        value_type="str",
        description="Describe your dataset and what you are trying to detect",
        help_text="Provide context like: 'Aerial drone images of solar panels' or 'Thermal night vision camera detecting pedestrians'."
    )
    project_desc = get_parameter_value_input(desc_param)
    if project_desc in (None, NAV_BACK, NAV_LIST):
        return None
        
    pref_param = ParameterDefinition(
        name="ai_prompt_preferences",
        category="AI",
        default="prioritize geometric robustness",
        value_type="str",
        description="AI Preferences / Custom prompts",
        help_text="Tell the AI any preferences like: 'keep variations minor', 'heavy atmospheric distortions needed', 'rotation invariance'."
    )
    user_prompt = get_parameter_value_input(pref_param)
    if user_prompt in (None, NAV_BACK, NAV_LIST):
        return None

    # Load dataset details and sample images
    dataset_path = Path(dataset_choice)
    console.print("\n[bold green]Gathering dataset information and sampling images...[/bold green]")
    try:
        dataset_details, samples_base64 = get_dataset_ai_summary(dataset_path)
    except Exception as e:
        console.print(expected_error_panel(f"Failed to scan dataset: {e}"))
        input("\nPress Enter to return...")
        return None

    console.print(f"Scanned [yellow]{dataset_details['total_images']}[/yellow] images across [yellow]{len(dataset_details['splits'])}[/yellow] splits.")
    console.print(f"Sampled [yellow]{len(samples_base64)}[/yellow] images for multimodal analysis.")
    
    # Run the LLM recommendation
    console.print(f"\n[bold green]Consulting AI Expert ({provider} : {model_name})...[/bold green]")
    
    system_instruction = (
        "You are an Elite Principal Computer Vision Architect specializing in dataset augmentation pipelines using the Albumentations library.\n"
        "Your task is to analyze the provided dataset statistics, visual sample properties, and the user's specific project goals. "
        "Then, critically formulate and recommend a highly optimized, safe, and effective Albumentations pipeline.\n\n"
        "CRITICAL PROFESSIONAL AUDIT GUIDELINES:\n"
        "1. Act like a rigorous, critical senior Computer Vision specialist. Do not suggest transforms that destroy semantic context or introduce harmful artifacts.\n"
        "2. Guard against standard augmentation bugs:\n"
        "   - If color plays a deterministic role in detection (e.g. traffic lights, colored wires, red/green blood cells), DO NOT suggest transforms like ToGray, HueSaturationValue with high ranges, Solarize, or heavy ColorJitter.\n"
        "   - If orientation is critical (e.g., reading text, identifying upright objects, pedestrian feet on ground), DO NOT suggest vertical flips (VerticalFlip) or D4/RandomRotate90 which could make objects upside down.\n"
        "   - If the dataset already has significant camera blur or low resolution, DO NOT suggest strong GaussianBlur, MedianBlur, or MotionBlur, as it will render objects completely unidentifiable. Recommend Sharpen or UnsharpMask instead.\n"
        "   - If objects are small or dense, suggest crop transforms carefully to avoid cropping out the entire object.\n"
        "3. You must output your recommendation strictly as a compliant JSON object matching the requested schema."
    )
    
    prompt = (
        f"You are building an augmentation profile for Albumentations.\n"
        f"The user describes their project as: '{project_desc}'\n"
        f"User special instructions & preferences: '{user_prompt}'\n\n"
        f"Dataset structural details:\n"
        f"{json.dumps(dataset_details, indent=2)}\n\n"
        f"=== SUPPORTED TRANSFORMS REFERENCE ===\n"
        f"Only select transforms from the following supported catalog:\n"
        f"{json.dumps(SUPPORTED_TRANSFORMS)}\n\n"
        f"=== TRANSFORM PARAMETER REFERENCE ===\n"
        f"Some transforms accept extra tuning parameters. Range parameters are split\n"
        f"into explicit low/high (or lower/upper) keys — always provide BOTH sides of\n"
        f"a range. Use ONLY the exact parameter names listed below; any other names are\n"
        f"ignored. Any parameter you omit falls back to a sensible default. Transforms\n"
        f"not listed here take only 'name', 'enabled', and 'p'.\n"
        f"{_format_transform_param_catalog()}\n\n"
        f"=== EXPECTED OUTPUT FORMAT ===\n"
        f"You must return a single JSON object containing exactly the following fields:\n"
        f"1. 'profile_name': a short, lowercase_with_underscores string identifying the augmentation profile (e.g. 'aerial_solar_panels_aug').\n"
        f"2. 'description': a detailed explanation of your augmentation strategy, explaining why you selected or avoided specific transforms based on the image characteristics and project goals.\n"
        f"3. 'multiplier': integer (1 to 10 copies per source image to generate)\n"
        f"4. 'include_originals': boolean (whether to include original images in the final set)\n"
        f"5. 'transforms': a list of objects representing Albumentations transforms. "
        f"Each object in the list must have exactly these keys:\n"
        f"   - 'name': string (must be exactly one of the supported transforms from the catalog above)\n"
        f"   - 'enabled': true\n"
        f"   - 'p': float (probability of applying this transform, from 0.0 to 1.0)\n"
        f"   - optionally any transform-specific parameters using the exact names from the Parameter Reference above.\n"
    )
    
    try:
        raw_response, model_used = query_llm_with_free_tier_fallback(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            text_prompt=prompt,
            samples_base64=samples_base64,
            system_instruction=system_instruction
        )
        if model_used != model_name:
            console.print(warning_panel(
                f"'{model_name}' has no Gemini free tier, so the request was "
                f"automatically sent using [bold cyan]{model_used}[/bold cyan] instead.\n"
                f"Tip: select a free-tier model in Settings → AI Recommendations to silence this.",
                title="Switched to a free-tier model",
            ))

        # Use our robust block extractor
        recommendation = extract_json_block(raw_response)

        profile_name = _slugify(recommendation.get("profile_name", ""), fallback="ai_profile")
        description = recommendation.get("description", "Optimized by AI based on dataset analysis.")
        try:
            multiplier = max(1, min(10, int(float(recommendation.get("multiplier", 3)))))
        except (ValueError, TypeError):
            multiplier = 3
        include_originals = _coerce_bool(recommendation.get("include_originals", True), default=True)
        suggested_transforms = recommendation.get("transforms", [])
        
        # Sanitize and type-check suggested transforms
        suggested_transforms = sanitize_augmentation_transforms(suggested_transforms)

        if not any(tx.get("enabled") for tx in suggested_transforms):
            console.print(warning_panel(
                "The AI did not return any valid, enabled transforms for this dataset.\n"
                "Nothing would be augmented, so no profile was saved. Try rephrasing "
                "your preferences or pick a built-in profile instead.",
                title="No Usable Transforms",
            ))
            input("\nPress Enter to return...")
            return None

        clear_screen()
        print_stylized_header("AI Recommendation Success")
        
        console.print(make_panel(
            f"[bold green]AI Suggested Profile Name:[/bold green] {profile_name}\n\n"
            f"[bold green]Description:[/bold green]\n{description}\n\n"
            f"Multiplier: [yellow]{multiplier}[/yellow]  |  Include Originals: [yellow]{include_originals}[/yellow]",
            title="AI Augmentation Strategy",
            state=TUIState.SUCCESS
        ))
        
        # Show suggested transforms
        console.print("\n[bold]Suggested Augmentations:[/bold]")
        tx_dict = {}
        for tx in suggested_transforms:
            if tx.get("enabled"):
                params = [f"p={tx.get('p')}"]
                for k, v in tx.items():
                    if k not in ["name", "enabled", "p"]:
                        params.append(f"{k}={v}")
                tx_dict[tx.get("name")] = ", ".join(params)
                
        render_summary_panel("Augmentation Profile Transforms", tx_dict)
        
        choice = get_user_choice(
            ["Save Augmentation Profile", "Discard and Return"],
            title="Confirm Augmentation Profile",
            text="Do you want to save this augmentation profile to configs/augmentation_profiles/?"
        )
        
        if choice == "Save Augmentation Profile":
            # Build and save profile
            from src.augmentation.profiles import AugmentationProfile, save_profile, PROFILES_DIR
            
            profile = AugmentationProfile(
                name=profile_name,
                description=description,
                multiplier=multiplier,
                seed=42,
                include_originals=include_originals,
                transforms=suggested_transforms,
                created_at=datetime.now().isoformat(timespec="seconds"),
                modified_at=datetime.now().isoformat(timespec="seconds")
            )
            
            save_profile(profile, PROFILES_DIR)
            console.print(f"\n✅ Augmentation profile saved successfully: [cyan]{profile_name}[/cyan]", style="bold green")
            input("\nPress Enter to return...")
            return profile_name
            
    except Exception as e:
        console.print(expected_error_panel(f"Failed to fetch AI recommendation: {e}"))
        console.print(traceback.format_exc(), style="dim")
        input("\nPress Enter to return...")
        
    return None

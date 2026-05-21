"""AI client utilities for dataset analysis, dynamic model fetching, and TUI recommendation flows."""
from __future__ import annotations

import io
import os
import json
import base64
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
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
)
from src.config.settings import load_settings, save_settings
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


def make_http_request(url: str, method: str = "GET", headers: dict = None, data: Any = None, timeout: int = 25) -> tuple[int, Any]:
    """Helper to perform HTTP requests using standard urllib, with fallback to requests if installed."""
    headers = headers or {}
    
    # Try using requests first if available
    try:
        import requests
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
            return response.status_code, response.json()
        except Exception:
            pass
    except ImportError:
        pass

    # Urllib fallback
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
            return status, json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
            return e.code, json.loads(error_body)
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 500, {"error": str(e)}


def fetch_multimodal_models(provider: str, api_key: str) -> list[str]:
    """Dynamically fetch the latest multimodal models from the provider API."""
    if not api_key:
        return []
    
    if provider.lower() == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        status, res = make_http_request(url, "GET")
        if status == 200 and "models" in res:
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
            return sorted(list(set(models)))
        else:
            raise ValueError(res.get("error", {}).get("message", "Failed to fetch models from Gemini API"))
            
    elif provider.lower() == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        status, res = make_http_request(url, "GET", headers=headers)
        if status == 200 and "data" in res:
            models = []
            for m in res["data"]:
                model_id = m.get("id", "")
                # Prioritize GPT-4o family models
                if "gpt-4o" in model_id.lower() or "gpt-4-vision" in model_id.lower():
                    models.append(model_id)
            return sorted(list(set(models)))
        else:
            raise ValueError(res.get("error", {}).get("message", "Failed to fetch models from OpenAI API"))
            
    return []


def get_dataset_ai_summary(dataset_path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize dataset metadata and prepare 1-2 random images for multimodal API consumption."""
    path = Path(dataset_path)
    summary = summarize_dataset(path)
    
    # Collect all image files
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
    all_images = []
    for ext in valid_exts:
        all_images.extend(list(path.rglob(f"*{ext}")))
    
    all_images = list(set([p for p in all_images if p.is_file()]))
    
    samples_base64 = []
    image_stats = []
    
    if all_images:
        num_samples = min(2, len(all_images))
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
                    
                    # Resize to maximum 1024px on any dimension to respect payload limits
                    if img.width > 1024 or img.height > 1024:
                        img.thumbnail((1024, 1024))
                    
                    buffer = io.BytesIO()
                    img.convert("RGB").save(buffer, format="JPEG", quality=85)
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
        
        parts = []
        if system_instruction:
            parts.append({"text": f"System Guidelines:\n{system_instruction}\n\n"})
            
        parts.append({"text": text_prompt})
        
        for sample in samples_base64:
            parts.append({
                "inlineData": {
                    "mimeType": sample["mime_type"],
                    "data": sample["base64_data"]
                }
            })
            
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        status, res = make_http_request(url, "POST", data=payload)
        if status == 200:
            try:
                candidates = res.get("candidates", [])
                if candidates:
                    return candidates[0]["content"]["parts"][0]["text"]
            except Exception:
                pass
            raise ValueError(f"Gemini API returned unexpected response structure: {res}")
        else:
            err_msg = res.get("error", {}).get("message", str(res))
            raise ValueError(f"Gemini API Error (status {status}): {err_msg}")
            
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
        
        status, res = make_http_request(url, "POST", headers=headers, data=payload)
        if status == 200:
            try:
                return res["choices"][0]["message"]["content"]
            except Exception:
                pass
            raise ValueError(f"OpenAI API returned unexpected response structure: {res}")
        else:
            err_msg = res.get("error", {}).get("message", str(res))
            raise ValueError(f"OpenAI API Error (status {status}): {err_msg}")
            
    raise ValueError(f"Unsupported provider: {provider}")


def verify_ai_setup() -> tuple[bool, str, str, str]:
    """Helper to verify settings and prompt user to configure if keys are missing."""
    settings = load_settings()
    ai_config = settings.get("ai", {})
    provider = ai_config.get("provider", "Gemini")
    model_name = ai_config.get("selected_model", "gemini-2.5-flash")
    
    api_key = ""
    if provider.lower() == "gemini":
        api_key = ai_config.get("gemini_api_key", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    elif provider.lower() == "openai":
        api_key = ai_config.get("openai_api_key", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        
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


def extract_json_block(text: str) -> dict:
    """Extract and parse the first valid JSON block {...} from text.
    Handles potential LLM markdown decorations, trailing/leading 
    conversational text, and whitespace issues.
    """
    import re
    cleaned = text.strip()
    
    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
        
    # Look inside markdown code blocks
    # This matches any ```json ... ``` or ``` ... ``` block
    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    for b in blocks:
        b_clean = b.strip()
        try:
            return json.loads(b_clean)
        except json.JSONDecodeError:
            # Maybe the block itself has conversational junk, try finding outer braces within it
            first = b_clean.find("{")
            last = b_clean.rfind("}")
            if first != -1 and last != -1 and last > first:
                try:
                    return json.loads(b_clean[first:last+1])
                except json.JSONDecodeError:
                    pass

    # Look for outer-most braces in the entire text
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
            
    raise ValueError(
        f"Failed to locate or parse a valid JSON block in the AI response.\n"
        f"Raw response preview:\n{cleaned[:300]}..."
    )


def sanitize_training_params(params: dict) -> dict:
    """Sanitize and cast the AI suggested parameters to ensure strict type safety 
    matching YOLOmatic and Ultralytics expectations.
    """
    sanitized = {}
    
    # Define mapping of keys to casting functions
    casts = {
        # Core
        "epochs": int,
        "patience": int,
        "batch": int,
        "imgsz": int,
        # Optimizer
        "optimizer": str,
        "lr0": float,
        "lrf": float,
        "momentum": float,
        "weight_decay": float,
        "warmup_epochs": float,
        "warmup_momentum": float,
        "cos_lr": lambda x: str(x).lower() == "true" or bool(x),
        # Augmentation
        "hsv_h": float,
        "hsv_s": float,
        "hsv_v": float,
        "degrees": float,
        "translate": float,
        "scale": float,
        "shear": float,
        "perspective": float,
        "flipud": float,
        "fliplr": float,
        "mosaic": float,
        "mixup": float,
        "copy_paste": float,
        "auto_augment": str,
        "erasing": float,
        "close_mosaic": int,
        # Loss
        "box": float,
        "cls": float,
        "dfl": float,
        "label_smoothing": float,
        # Advanced
        "rect": lambda x: str(x).lower() == "true" or bool(x),
        "pretrained": lambda x: str(x).lower() == "true" or bool(x),
        "fraction": float,
    }
    
    for k, v in params.items():
        if k in casts:
            try:
                # Handle special casing
                if casts[k] == int:
                    sanitized[k] = int(float(v))
                elif casts[k] == float:
                    sanitized[k] = float(v)
                elif casts[k] == str:
                    sanitized[k] = str(v).strip()
                else:
                    sanitized[k] = casts[k](v)
            except (ValueError, TypeError):
                pass
                
    # Extra validation for specific keys
    if "imgsz" in sanitized:
        val = sanitized["imgsz"]
        # Round to nearest multiple of 32
        sanitized["imgsz"] = max(32, min(2048, ((val + 16) // 32) * 32))
        
    if "epochs" in sanitized:
        sanitized["epochs"] = max(1, min(10000, sanitized["epochs"]))
        
    if "optimizer" in sanitized:
        valid_opts = {"auto", "SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "RMSProp", "MuSGD"}
        opt = sanitized["optimizer"]
        if opt not in valid_opts:
            matched = False
            for vo in valid_opts:
                if vo.lower() == opt.lower():
                    sanitized["optimizer"] = vo
                    matched = True
                    break
            if not matched:
                sanitized["optimizer"] = "auto"
                
    return sanitized


def sanitize_augmentation_transforms(transforms: list) -> list:
    """Sanitize and cast the AI suggested Albumentations transforms to ensure strict type safety."""
    sanitized_list = []
    supported_set = set(SUPPORTED_TRANSFORMS)
    
    if not isinstance(transforms, list):
        return []
        
    for tx in transforms:
        if not isinstance(tx, dict):
            continue
            
        name = tx.get("name")
        if not name or name not in supported_set:
            continue
            
        enabled = str(tx.get("enabled", "true")).lower() == "true"
        try:
            p = float(tx.get("p", 0.5))
        except (ValueError, TypeError):
            p = 0.5
            
        sanitized_tx = {
            "name": name,
            "enabled": enabled,
            "p": max(0.0, min(1.0, p))
        }
        
        # Keep and sanitise other parameters
        for k, v in tx.items():
            if k in ["name", "enabled", "p"]:
                continue
            if isinstance(v, (int, float, bool)):
                sanitized_tx[k] = v
            else:
                try:
                    if "." in str(v):
                        sanitized_tx[k] = float(v)
                    else:
                        sanitized_tx[k] = int(v)
                except (ValueError, TypeError):
                    val_str = str(v).strip()
                    if val_str.lower() == "true":
                        sanitized_tx[k] = True
                    elif val_str.lower() == "false":
                        sanitized_tx[k] = False
                    else:
                        sanitized_tx[k] = val_str
                        
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
    if project_desc in (None, NAV_BACK):
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
    if user_prompt in (None, NAV_BACK):
        return None

    # Load dataset details and sample images
    dataset_path = Path(dataset_choice)
    console.print(f"\n[bold green]Gathering dataset information and sampling images...[/bold green]")
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
        f"  - 'perspective': float (0.0 to 1.0, 3D perspective distortion)\n"
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
        raw_response = query_llm_multimodal(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            text_prompt=prompt,
            samples_base64=samples_base64,
            system_instruction=system_instruction
        )
        
        # Use our robust block extractor
        recommendation = extract_json_block(raw_response)
        
        suggested_name = recommendation.get("suggested_name", f"ai_{model_choice.lower()}").strip().lower().replace(" ", "_")
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
    if project_desc in (None, NAV_BACK):
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
    if user_prompt in (None, NAV_BACK):
        return None

    # Load dataset details and sample images
    dataset_path = Path(dataset_choice)
    console.print(f"\n[bold green]Gathering dataset information and sampling images...[/bold green]")
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
        f"   - optionally any transform-specific parameters (e.g. 'brightness_limit': 0.2, 'degrees': 15.0) to fine-tune it.\n"
    )
    
    try:
        raw_response = query_llm_multimodal(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            text_prompt=prompt,
            samples_base64=samples_base64,
            system_instruction=system_instruction
        )
        
        # Use our robust block extractor
        recommendation = extract_json_block(raw_response)
        
        profile_name = recommendation.get("profile_name", "ai_profile").strip().lower().replace(" ", "_")
        description = recommendation.get("description", "Optimized by AI based on dataset analysis.")
        multiplier = int(recommendation.get("multiplier", 3))
        include_originals = bool(recommendation.get("include_originals", True))
        suggested_transforms = recommendation.get("transforms", [])
        
        # Sanitize and type-check suggested transforms
        suggested_transforms = sanitize_augmentation_transforms(suggested_transforms)
        
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

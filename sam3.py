import os

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "facebook/sam3"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError(
        "HF_TOKEN is not set. Export it first, e.g.:\n"
        "export HF_TOKEN=hf_xxxxxxxxxxxxx"
    )

# Load model and processor
model = Sam3Model.from_pretrained(model_id, token=hf_token).to(device)
processor = Sam3Processor.from_pretrained(model_id, token=hf_token)

# Load image
image = Image.open(
    "/datasets/QGIS Ready.v1i.yolo26/train/images/IO_ORI_RGB_14N_618_5568_r04608_c14336_jpg.rf.13c4badf16b30a3416b68a422ef45584.jpg"
).convert("RGB")

text_prompt = "high vegetations like trees and bushes"

# Prepare inputs
inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Extract binary masks
results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.4,
    mask_threshold=0.5,
    target_sizes=inputs["original_sizes"].tolist(),
)[0]

# Save masks
if "masks" in results:
    for i, mask_tensor in enumerate(results["masks"]):
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)
        mask_pil.save(f"mask_{i}.png")

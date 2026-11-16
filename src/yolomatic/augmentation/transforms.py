"""
Albumentations transform catalog with ParameterDefinition metadata.

Each transform's specific numeric parameters are listed here so the TUI can
display them in get_user_multi_select.  The `p` (probability) parameter is
shared and prepended by get_params_for_transform().
"""
from __future__ import annotations

from typing import Any

try:
    from src.utils.tui import ParameterDefinition
except ImportError:
    from utils.tui import ParameterDefinition  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Transform groups — order controls display in the TUI menu
# ---------------------------------------------------------------------------

TRANSFORM_GROUPS: dict[str, list[str]] = {
    "Geometric":    ["HorizontalFlip", "VerticalFlip", "D4", "RandomRotate90",
                     "Transpose", "Rotate"],
    "Affine":       ["Affine", "RandomScale"],
    "Crops":        ["RandomCrop", "Pad"],
    "Distortion":   ["Perspective", "ElasticTransform", "GridDistortion",
                     "OpticalDistortion"],
    "Color":        ["RandomBrightnessContrast", "ColorJitter", "Solarize",
                     "Posterize", "Equalize", "ToGray"],
    "HSV / RGB":    ["HueSaturationValue", "RGBShift"],
    "Noise":        ["GaussNoise", "ISONoise", "MultiplicativeNoise"],
    "Blur":         ["GaussianBlur", "MedianBlur", "MotionBlur", "AdvancedBlur"],
    "Enhancement":  ["CLAHE", "Sharpen", "UnsharpMask"],
    "Weather":      ["RandomFog", "RandomRain", "RandomShadow", "RandomSunFlare"],
    "Compression":  ["ImageCompression", "Downscale"],
    "Dropout":      ["CoarseDropout"],
    "Misc":         ["ShotNoise"],
}

TRANSFORM_GUIDANCE: dict[str, dict[str, str]] = {
    "HorizontalFlip": {
        "summary": "Mirror images left-to-right.",
        "use": "Good when object meaning does not depend on left/right orientation.",
        "caution": "Avoid for asymmetric labels, text, road-side rules, or medical laterality.",
    },
    "VerticalFlip": {
        "summary": "Mirror images top-to-bottom.",
        "use": "Useful for aerial, microscopy, and other views where gravity is not meaningful.",
        "caution": "Avoid for normal ground-level scenes where upside-down images are unrealistic.",
    },
    "D4": {
        "summary": "Randomly apply one of the eight square symmetries: rotations and flips.",
        "use": "Best for aerial, satellite, microscopy, tiles, and rotation-invariant object layouts.",
        "caution": "Can create unrealistic training samples when upright orientation matters.",
    },
    "RandomRotate90": {
        "summary": "Randomly rotate by 0, 90, 180, or 270 degrees.",
        "use": "Use for domains where right-angle orientation is arbitrary.",
        "caution": "Even with p=1.0, one sampled outcome is the identity rotation.",
    },
    "Transpose": {
        "summary": "Swap image rows and columns, equivalent to reflecting across the diagonal.",
        "use": "Useful for square tiles when diagonal symmetry is plausible.",
        "caution": "Changes orientation strongly; avoid when image axes have semantic meaning.",
    },
    "Rotate": {
        "summary": "Rotate by a sampled angle while updating masks, boxes, and keypoints.",
        "use": "Small ranges improve robustness to camera tilt and annotation orientation variance.",
        "caution": "Large rotations can add border fill and clip objects unless that matches reality.",
    },
    "Affine": {
        "summary": "Apply scale, shear, translation, and rotation-style geometric changes.",
        "use": "Good for viewpoint and camera placement variation.",
        "caution": "Aggressive values can distort object shape and reduce label quality.",
    },
    "RandomScale": {
        "summary": "Randomly zoom in or out while preserving the image canvas workflow.",
        "use": "Helps models handle object scale variation.",
        "caution": "Too much scaling can remove small objects or create unrealistic context.",
    },
    "RandomCrop": {
        "summary": "Crop a fixed-size region from the image and matching annotations.",
        "use": "Useful when training on large images or forcing local context.",
        "caution": "Can remove objects; use carefully with sparse detection datasets.",
    },
    "Pad": {
        "summary": "Add pixels around image borders.",
        "use": "Helpful before crops or geometric transforms that need extra canvas.",
        "caution": "Large constant padding can teach the model artificial borders.",
    },
    "Perspective": {
        "summary": "Move image corners to simulate perspective/viewpoint change.",
        "use": "Good for camera angle variation and planar scenes.",
        "caution": "High scale values can bend boxes/masks into unrealistic supervision.",
    },
    "ElasticTransform": {
        "summary": "Apply smooth local deformation to the image and spatial targets.",
        "use": "Useful for organic shapes, vegetation, biomedical, and soft deformable objects.",
        "caution": "Often wrong for rigid objects, text, manufactured parts, and precise geometry.",
    },
    "GridDistortion": {
        "summary": "Warp the image on a coarse grid.",
        "use": "Can model lens or surface deformation.",
        "caution": "Use low probability; strong grid warps can damage annotation geometry.",
    },
    "OpticalDistortion": {
        "summary": "Simulate camera lens distortion.",
        "use": "Useful when deployment cameras have barrel or pincushion distortion.",
        "caution": "Avoid if source and target cameras are already calibrated similarly.",
    },
    "RandomBrightnessContrast": {
        "summary": "Randomly adjust image brightness and contrast.",
        "use": "A strong baseline for lighting variation in detection and segmentation.",
        "caution": "Very wide ranges can hide objects or shift imagery away from deployment data.",
    },
    "ColorJitter": {
        "summary": "Randomly vary brightness, contrast, saturation, and hue.",
        "use": "Good for natural color variation and camera/color balance differences.",
        "caution": "Hue/saturation changes can hurt tasks where color is the class signal.",
    },
    "Solarize": {
        "summary": "Invert pixels above a sampled threshold.",
        "use": "Regularizes against unusual tone response and high-contrast corruption.",
        "caution": "Usually aggressive; keep probability low for detection datasets.",
    },
    "Posterize": {
        "summary": "Reduce channel bit depth.",
        "use": "Models low-quality capture, compression, or limited color depth.",
        "caution": "Low bit counts can destroy subtle visual cues.",
    },
    "Equalize": {
        "summary": "Equalize image histogram to redistribute contrast.",
        "use": "Useful for variable exposure and contrast conditions.",
        "caution": "Can exaggerate noise and create unnatural textures.",
    },
    "ToGray": {
        "summary": "Convert image to grayscale.",
        "use": "Helps when color should not be required for recognition.",
        "caution": "Avoid when class identity depends on color.",
    },
    "HueSaturationValue": {
        "summary": "Shift hue, saturation, and value channels independently.",
        "use": "Good for camera white balance, vegetation tone, and lighting variation.",
        "caution": "Large hue shifts can change class semantics for color-coded targets.",
    },
    "RGBShift": {
        "summary": "Shift red, green, and blue channels independently.",
        "use": "Models sensor/channel bias and white balance differences.",
        "caution": "Strong shifts can make images unrealistic.",
    },
    "GaussNoise": {
        "summary": "Add Gaussian noise sampled from a configured standard deviation range.",
        "use": "Improves robustness to sensor noise, low light, and compression artifacts.",
        "caution": "Too much noise hides small objects and fine masks.",
    },
    "ISONoise": {
        "summary": "Add camera-like luminance noise and color shift.",
        "use": "Good for low-light or high-ISO camera imagery.",
        "caution": "Less appropriate for synthetic, scanned, or aerial data without sensor noise.",
    },
    "MultiplicativeNoise": {
        "summary": "Multiply pixel values by sampled factors.",
        "use": "Models uneven gain, exposure, or illumination changes.",
        "caution": "Large multipliers can wash out or black out visual evidence.",
    },
    "GaussianBlur": {
        "summary": "Blur with a Gaussian kernel.",
        "use": "Models mild defocus, resizing blur, and motion-independent softness.",
        "caution": "Large kernels can erase small objects.",
    },
    "MedianBlur": {
        "summary": "Apply median filtering.",
        "use": "Models denoising and salt-and-pepper robustness.",
        "caution": "Can remove thin structures and fine boundaries.",
    },
    "MotionBlur": {
        "summary": "Blur along a sampled line direction.",
        "use": "Good for moving cameras, moving objects, drones, and vehicle footage.",
        "caution": "Keep moderate for static imagery; heavy blur harms small labels.",
    },
    "AdvancedBlur": {
        "summary": "Apply a richer randomized blur kernel with anisotropy and sigma variation.",
        "use": "Useful when blur shape varies across cameras or acquisition conditions.",
        "caution": "More aggressive than GaussianBlur; start with low probability.",
    },
    "CLAHE": {
        "summary": "Apply contrast-limited adaptive histogram equalization.",
        "use": "Helps local contrast in low-contrast or unevenly lit imagery.",
        "caution": "High clip limits can amplify noise and halos.",
    },
    "Sharpen": {
        "summary": "Blend sharpened detail back into the image.",
        "use": "Models camera sharpening and helps robustness to crisp imagery.",
        "caution": "Can amplify noise and create edge artifacts.",
    },
    "UnsharpMask": {
        "summary": "Sharpen using a blurred image subtraction mask.",
        "use": "Useful for camera post-processing variation.",
        "caution": "Large blur or alpha values can create halos around object boundaries.",
    },
    "RandomFog": {
        "summary": "Overlay fog-like brightness and haze.",
        "use": "Useful for outdoor deployment under fog, haze, or atmospheric scattering.",
        "caution": "Usually unrealistic for indoor, close-range, or controlled imagery.",
    },
    "RandomRain": {
        "summary": "Add rain streaks with sampled slant and size.",
        "use": "Good for outdoor camera datasets exposed to weather.",
        "caution": "Avoid unless rain is plausible at deployment time.",
    },
    "RandomShadow": {
        "summary": "Add polygonal shadow regions.",
        "use": "Helps outdoor models handle clouds, trees, buildings, and object shadows.",
        "caution": "Strong shadows can hide labels; keep probabilities moderate.",
    },
    "RandomSunFlare": {
        "summary": "Add lens flare artifacts.",
        "use": "Useful for cameras facing sun or bright point light sources.",
        "caution": "Very domain-specific; use low probability.",
    },
    "ImageCompression": {
        "summary": "Re-encode images with JPEG or WebP quality loss.",
        "use": "Improves robustness to camera, upload, and web compression.",
        "caution": "Low quality values damage small objects and thin masks.",
    },
    "Downscale": {
        "summary": "Downscale then upscale to simulate resolution loss.",
        "use": "Good when deployment images may be lower quality than training data.",
        "caution": "Too much downscaling removes small targets.",
    },
    "CoarseDropout": {
        "summary": "Erase random rectangular regions.",
        "use": "Regularizes against occlusion and missing visual evidence.",
        "caution": "Large holes can erase complete objects; use small sizes for detection.",
    },
    "ShotNoise": {
        "summary": "Add signal-dependent shot noise.",
        "use": "Models photon noise in low-light or sensor-limited imaging.",
        "caution": "Use only when this noise resembles deployment data.",
    },
}


def get_transform_guidance(name: str) -> dict[str, str]:
    """Return user-facing guidance for a transform."""
    return TRANSFORM_GUIDANCE.get(
        name,
        {
            "summary": f"Configure the Albumentations {name} transform.",
            "use": "Enable when this visual variation is realistic for deployment data.",
            "caution": "Start with a low probability and inspect samples before training.",
        },
    )


# Probability param shared by every transform
_P_PARAM = ParameterDefinition(
    name="p",
    category="probability",
    default=0.5,
    value_type="float",
    description="Probability",
    help_text=(
        "Albumentations checks this probability independently each time the transform is called.\n"
        "0.0 = never considered, 0.5 = about half of calls, 1.0 = always considered.\n"
        "Some transforms can still sample an identity/no-change result even when p=1.0."
    ),
    min_value=0.0,
    max_value=1.0,
    affects="Controls how often this transform contributes augmented samples.",
)

# ---------------------------------------------------------------------------
# Per-transform parameter definitions
# Range params are stored as low/high pairs so the profile YAML can hold
# individual floats instead of Python tuples.
# ---------------------------------------------------------------------------

_DEFS: dict[str, list[ParameterDefinition]] = {

    # --- Geometric (no extra params) ---
    "HorizontalFlip": [],
    "VerticalFlip":   [],
    "D4":             [],
    "RandomRotate90": [],
    "Transpose":      [],

    "Rotate": [
        ParameterDefinition("limit_low",    "rotation", -15,  "int",   "Min rotation (°)",
                            "Lower bound of rotation range in degrees.", -180, 0),
        ParameterDefinition("limit_high",   "rotation",  15,  "int",   "Max rotation (°)",
                            "Upper bound of rotation range in degrees.",    0, 180),
        ParameterDefinition("border_mode",  "rotation",   4,  "int",
                            "Border fill (cv2)",
                            "0=black fill, 1=replicate edge, 4=reflect (recommended for seamless tiling).",
                            allowed_values=["0", "1", "2", "4"]),
    ],

    # --- Affine ---
    "Affine": [
        ParameterDefinition("shear_low",    "affine", -10.0, "float", "Min shear (°)",
                            "Lower shear angle bound.", -45.0, 0.0),
        ParameterDefinition("shear_high",   "affine",  10.0, "float", "Max shear (°)",
                            "Upper shear angle bound.", 0.0, 45.0),
        ParameterDefinition("scale_low",    "affine",  0.9,  "float", "Min scale",
                            "Lower scale factor.", 0.5, 1.0),
        ParameterDefinition("scale_high",   "affine",  1.1,  "float", "Max scale",
                            "Upper scale factor.", 1.0, 2.0),
    ],
    "RandomScale": [
        ParameterDefinition("scale_limit_low",  "scale", -0.1, "float", "Min scale offset",
                            "Lower bound of scale change. -0.1 means 0.9×.", -0.5, 0.0),
        ParameterDefinition("scale_limit_high", "scale",  0.1, "float", "Max scale offset",
                            "Upper bound of scale change. 0.1 means 1.1×.",   0.0, 0.5),
    ],

    # --- Crops ---
    "RandomCrop": [
        ParameterDefinition("height", "crop", 512, "int", "Crop height (px)",
                            "Height of the crop window in pixels.", 32, 4096),
        ParameterDefinition("width",  "crop", 512, "int", "Crop width (px)",
                            "Width of the crop window in pixels.",  32, 4096),
    ],
    "Pad": [
        ParameterDefinition("padding", "pad", 32, "int", "Padding (px)",
                            "Number of pixels to pad on all sides.", 0, 512),
    ],

    # --- Distortion ---
    "Perspective": [
        ParameterDefinition("scale_low",  "distortion", 0.05, "float", "Min perspective",
                            "Minimum corner displacement as fraction of image size.", 0.0, 0.5),
        ParameterDefinition("scale_high", "distortion", 0.1,  "float", "Max perspective",
                            "Maximum corner displacement as fraction of image size.", 0.0, 0.5),
    ],
    "ElasticTransform": [
        ParameterDefinition("alpha", "distortion", 1.0,  "float", "Alpha (strength)",
                            "Intensity of elastic distortion. Higher = more deformation.", 0.1, 10.0),
        ParameterDefinition("sigma", "distortion", 50.0, "float", "Sigma (smoothness)",
                            "Smoothness of distortion field. Higher = smoother deformation.", 5.0, 200.0),
    ],
    "GridDistortion": [
        ParameterDefinition("num_steps",    "distortion",  5,   "int",   "Grid steps",
                            "Number of grid subdivisions.", 2, 20),
        ParameterDefinition("distort_low",  "distortion", -0.3, "float", "Min distortion",
                            "Minimum distortion coefficient.", -1.0, 0.0),
        ParameterDefinition("distort_high", "distortion",  0.3, "float", "Max distortion",
                            "Maximum distortion coefficient.",  0.0, 1.0),
    ],
    "OpticalDistortion": [
        ParameterDefinition("distort_low",  "distortion", -0.05, "float", "Min distortion",
                            "Minimum optical distortion coefficient.", -1.0, 0.0),
        ParameterDefinition("distort_high", "distortion",  0.05, "float", "Max distortion",
                            "Maximum optical distortion coefficient.",  0.0, 1.0),
    ],

    # --- Color ---
    "RandomBrightnessContrast": [
        ParameterDefinition("brightness_limit_low",  "color", -0.2, "float", "Min brightness",
                            "Minimum brightness shift (−0.2 = 20% darker).", -0.5, 0.0),
        ParameterDefinition("brightness_limit_high", "color",  0.2, "float", "Max brightness",
                            "Maximum brightness shift ( 0.2 = 20% brighter).",  0.0, 0.5),
        ParameterDefinition("contrast_limit_low",    "color", -0.2, "float", "Min contrast",
                            "Minimum contrast shift.", -0.5, 0.0),
        ParameterDefinition("contrast_limit_high",   "color",  0.2, "float", "Max contrast",
                            "Maximum contrast shift.",  0.0, 0.5),
    ],
    "ColorJitter": [
        ParameterDefinition("brightness", "color", 0.2, "float", "Brightness jitter",
                            "Random brightness change range.", 0.0, 1.0),
        ParameterDefinition("contrast",   "color", 0.2, "float", "Contrast jitter",
                            "Random contrast change range.",   0.0, 1.0),
        ParameterDefinition("saturation", "color", 0.2, "float", "Saturation jitter",
                            "Random saturation change range.", 0.0, 1.0),
        ParameterDefinition("hue",        "color", 0.0, "float", "Hue jitter",
                            "Random hue change range.",        0.0, 0.5),
    ],
    "Solarize": [
        ParameterDefinition("threshold_low",  "color", 0.0, "float", "Min threshold (0–1)",
                            "Minimum pixel threshold (normalized 0–1). Values above this are inverted.", 0.0, 1.0),
        ParameterDefinition("threshold_high", "color", 0.5, "float", "Max threshold (0–1)",
                            "Maximum pixel threshold (normalized 0–1).", 0.0, 1.0),
    ],
    "Posterize": [
        ParameterDefinition("num_bits", "color", 4, "int", "Color depth bits",
                            "Reduce each channel to this many bits (1–8). Lower = more posterize.", 1, 8),
    ],
    "Equalize":  [],
    "ToGray":    [],

    # --- HSV / RGB ---
    "HueSaturationValue": [
        ParameterDefinition("hue_shift_limit",        "hsv",  20, "int", "Hue shift limit",
                            "Maximum hue shift in degrees.", 0, 180),
        ParameterDefinition("sat_shift_limit",        "hsv",  30, "int", "Saturation shift",
                            "Maximum saturation shift.",     0, 255),
        ParameterDefinition("val_shift_limit",        "hsv",  20, "int", "Value shift",
                            "Maximum value (brightness) shift.", 0, 255),
    ],
    "RGBShift": [
        ParameterDefinition("r_shift_limit", "rgb", 20, "int", "Red shift",
                            "Maximum random shift for the R channel.",   0, 255),
        ParameterDefinition("g_shift_limit", "rgb", 20, "int", "Green shift",
                            "Maximum random shift for the G channel.",   0, 255),
        ParameterDefinition("b_shift_limit", "rgb", 20, "int", "Blue shift",
                            "Maximum random shift for the B channel.",   0, 255),
    ],

    # --- Noise ---
    "GaussNoise": [
        ParameterDefinition("std_range_low",  "noise", 0.01, "float", "Min std (0–1)",
                            "Lower bound of Gaussian noise std dev, normalized to [0,1]. "
                            "0.01 ≈ subtle noise, 0.05 ≈ strong.", 0.0, 1.0),
        ParameterDefinition("std_range_high", "noise", 0.05, "float", "Max std (0–1)",
                            "Upper bound of Gaussian noise std dev, normalized to [0,1].", 0.0, 1.0),
    ],
    "ISONoise": [
        ParameterDefinition("color_shift_low",  "noise", 0.01, "float", "Min color shift",
                            "Lower bound of ISO color channel shift (0–1).",  0.0, 1.0),
        ParameterDefinition("color_shift_high", "noise", 0.05, "float", "Max color shift",
                            "Upper bound of ISO color channel shift (0–1).",  0.0, 1.0),
        ParameterDefinition("intensity_low",    "noise", 0.10, "float", "Min intensity",
                            "Lower bound of luminance noise intensity (0–1).", 0.0, 1.0),
        ParameterDefinition("intensity_high",   "noise", 0.50, "float", "Max intensity",
                            "Upper bound of luminance noise intensity (0–1).", 0.0, 1.0),
    ],
    "MultiplicativeNoise": [
        ParameterDefinition("multiplier_low",  "noise", 0.9, "float", "Min multiplier",
                            "Lower bound of pixel multiplier.", 0.1, 1.0),
        ParameterDefinition("multiplier_high", "noise", 1.1, "float", "Max multiplier",
                            "Upper bound of pixel multiplier.", 1.0, 3.0),
    ],

    # --- Blur ---
    "GaussianBlur": [
        ParameterDefinition("blur_limit_low",  "blur", 3, "int", "Min kernel size",
                            "Minimum blur kernel size (must be odd).", 3, 31),
        ParameterDefinition("blur_limit_high", "blur", 7, "int", "Max kernel size",
                            "Maximum blur kernel size (must be odd).", 3, 31),
    ],
    "MedianBlur": [
        ParameterDefinition("blur_limit_low",  "blur", 3, "int", "Min kernel size",
                            "Minimum median blur kernel size.", 3, 31),
        ParameterDefinition("blur_limit_high", "blur", 7, "int", "Max kernel size",
                            "Maximum median blur kernel size.", 3, 31),
    ],
    "MotionBlur": [
        ParameterDefinition("blur_limit_low",  "blur", 3, "int", "Min kernel size",
                            "Minimum motion blur kernel size.", 3, 31),
        ParameterDefinition("blur_limit_high", "blur", 7, "int", "Max kernel size",
                            "Maximum motion blur kernel size.", 3, 31),
    ],
    "AdvancedBlur": [
        ParameterDefinition("blur_limit_low",    "blur", 3,   "int",   "Min kernel size",
                            "Minimum blur kernel size.", 3, 31),
        ParameterDefinition("blur_limit_high",   "blur", 7,   "int",   "Max kernel size",
                            "Maximum blur kernel size.", 3, 31),
        ParameterDefinition("sigma_x_limit_low", "blur", 0.2, "float", "Min σX",
                            "Minimum X-axis Gaussian sigma.", 0.0, 5.0),
        ParameterDefinition("sigma_x_limit_high","blur", 1.0, "float", "Max σX",
                            "Maximum X-axis Gaussian sigma.", 0.0, 5.0),
        ParameterDefinition("sigma_y_limit_low", "blur", 0.2, "float", "Min σY",
                            "Minimum Y-axis Gaussian sigma.", 0.0, 5.0),
        ParameterDefinition("sigma_y_limit_high","blur", 1.0, "float", "Max σY",
                            "Maximum Y-axis Gaussian sigma.", 0.0, 5.0),
    ],

    # --- Enhancement ---
    "CLAHE": [
        ParameterDefinition("clip_limit_low",  "enhancement", 1.0, "float", "Min clip limit",
                            "Lower bound of CLAHE clip limit. Higher = more contrast enhancement.", 1.0, 10.0),
        ParameterDefinition("clip_limit_high", "enhancement", 4.0, "float", "Max clip limit",
                            "Upper bound of CLAHE clip limit.", 1.0, 10.0),
    ],
    "Sharpen": [
        ParameterDefinition("alpha_low",    "enhancement", 0.2, "float", "Min alpha (strength)",
                            "Minimum sharpening blend strength (0=original, 1=full sharpen).", 0.0, 1.0),
        ParameterDefinition("alpha_high",   "enhancement", 0.5, "float", "Max alpha (strength)",
                            "Maximum sharpening blend strength.", 0.0, 1.0),
        ParameterDefinition("lightness_low",  "enhancement", 0.5, "float", "Min lightness",
                            "Minimum lightness factor of the sharpened image.", 0.0, 2.0),
        ParameterDefinition("lightness_high", "enhancement", 1.0, "float", "Max lightness",
                            "Maximum lightness factor of the sharpened image.", 0.0, 2.0),
    ],
    "UnsharpMask": [
        ParameterDefinition("blur_limit_low",  "enhancement", 3,   "int",   "Min kernel size",
                            "Minimum blur kernel size for unsharp mask.", 3, 31),
        ParameterDefinition("blur_limit_high", "enhancement", 7,   "int",   "Max kernel size",
                            "Maximum blur kernel size for unsharp mask.", 3, 31),
        ParameterDefinition("alpha_low",  "enhancement", 0.5, "float", "Min alpha (strength)",
                            "Minimum unsharp mask blend strength.", 0.0, 1.0),
        ParameterDefinition("alpha_high", "enhancement", 1.0, "float", "Max alpha (strength)",
                            "Maximum unsharp mask blend strength.", 0.0, 1.0),
    ],

    # --- Weather ---
    "RandomFog": [
        ParameterDefinition("fog_coef_lower", "weather", 0.05, "float", "Min fog coefficient",
                            "Minimum fog density (0=clear, 1=white).", 0.0, 1.0),
        ParameterDefinition("fog_coef_upper", "weather", 0.20, "float", "Max fog coefficient",
                            "Maximum fog density.", 0.0, 1.0),
    ],
    "RandomRain": [
        ParameterDefinition("slant_low",   "weather", -10, "int", "Min slant (°)",
                            "Minimum rain slant angle in degrees.", -20, 0),
        ParameterDefinition("slant_high",  "weather",  10, "int", "Max slant (°)",
                            "Maximum rain slant angle in degrees.",   0, 20),
        ParameterDefinition("drop_length", "weather",  20, "int", "Drop length (px)",
                            "Length of each raindrop streak.", 1, 100),
        ParameterDefinition("drop_width",  "weather",   1, "int", "Drop width (px)",
                            "Width of each raindrop streak.",  1, 5),
    ],
    "RandomShadow": [
        ParameterDefinition("num_shadows_lower", "weather", 1, "int", "Min shadows",
                            "Minimum number of shadow regions per image.", 1, 10),
        ParameterDefinition("num_shadows_upper", "weather", 2, "int", "Max shadows",
                            "Maximum number of shadow regions per image.", 1, 10),
        ParameterDefinition("shadow_dimension",  "weather", 5, "int", "Shadow polygon vertices",
                            "Number of vertices in each shadow polygon.", 3, 10),
    ],
    "RandomSunFlare": [
        ParameterDefinition("src_radius", "weather", 100, "int", "Flare source radius (px)",
                            "Radius of the sun flare source circle in pixels.", 10, 400),
        ParameterDefinition("num_flare_circles_low",  "weather", 6,  "int", "Min flare circles",
                            "Minimum number of secondary flare circles.", 1, 20),
        ParameterDefinition("num_flare_circles_high", "weather", 10, "int", "Max flare circles",
                            "Maximum number of secondary flare circles.", 1, 20),
    ],

    # --- Compression ---
    "ImageCompression": [
        ParameterDefinition("quality_low",  "compression",  70, "int", "Min JPEG quality",
                            "Minimum JPEG quality (1=worst, 100=best).", 1, 100),
        ParameterDefinition("quality_high", "compression", 100, "int", "Max JPEG quality",
                            "Maximum JPEG quality.", 1, 100),
    ],
    "Downscale": [
        ParameterDefinition("scale_range_low",  "compression", 0.25, "float", "Min scale",
                            "Minimum scale factor for downscaling (then upscaling back).", 0.05, 1.0),
        ParameterDefinition("scale_range_high", "compression", 0.75, "float", "Max scale",
                            "Maximum scale factor for downscaling.", 0.05, 1.0),
    ],

    # --- Dropout ---
    "CoarseDropout": [
        ParameterDefinition("num_holes_low",    "dropout",  1,  "int", "Min holes",
                            "Minimum number of rectangular regions to erase.", 1, 50),
        ParameterDefinition("num_holes_high",   "dropout",  8,  "int", "Max holes",
                            "Maximum number of rectangular regions to erase.", 1, 50),
        ParameterDefinition("hole_height_low",  "dropout",  8,  "int", "Min hole height (px)",
                            "Minimum height of each erased region in pixels.", 1, 256),
        ParameterDefinition("hole_height_high", "dropout", 32,  "int", "Max hole height (px)",
                            "Maximum height of each erased region in pixels.", 1, 256),
        ParameterDefinition("hole_width_low",   "dropout",  8,  "int", "Min hole width (px)",
                            "Minimum width of each erased region in pixels.",  1, 256),
        ParameterDefinition("hole_width_high",  "dropout", 32,  "int", "Max hole width (px)",
                            "Maximum width of each erased region in pixels.",  1, 256),
    ],

    # --- Misc ---
    "ShotNoise": [
        ParameterDefinition("scale_range_low",  "misc", 0.01, "float", "Min scale",
                            "Lower bound of the shot noise scale factor.", 0.0, 1.0),
        ParameterDefinition("scale_range_high", "misc", 0.10, "float", "Max scale",
                            "Upper bound of the shot noise scale factor.", 0.0, 1.0),
    ],
}


def get_params_for_transform(name: str) -> list[ParameterDefinition]:
    """Return [p_param] + transform-specific ParameterDefinitions."""
    specific = _DEFS.get(name, [])
    if specific:
        return [_P_PARAM] + specific
    return [_P_PARAM]


def build_albu_kwargs(transform_cfg: dict) -> dict[str, Any]:
    """
    Convert a profile transforms[] entry to albumentations constructor kwargs.

    Range params stored as (low, high) pairs are reassembled into tuples.
    Unknown keys (name, enabled) are stripped automatically.
    """
    skip = {"name", "enabled"}
    # Params that must be reassembled into tuples
    range_pairs = {
        # Geometric
        "limit":            ("limit_low",              "limit_high"),
        # Color
        "brightness_limit": ("brightness_limit_low",   "brightness_limit_high"),
        "contrast_limit":   ("contrast_limit_low",     "contrast_limit_high"),
        # Enhancement
        "clip_limit":       ("clip_limit_low",         "clip_limit_high"),
        # alpha as a range (Sharpen, UnsharpMask) — ElasticTransform uses scalar alpha directly
        "alpha":            ("alpha_low",              "alpha_high"),
        "lightness":        ("lightness_low",          "lightness_high"),
        # Noise
        "std_range":        ("std_range_low",          "std_range_high"),
        "multiplier":       ("multiplier_low",         "multiplier_high"),
        # Blur / scale
        "blur_limit":       ("blur_limit_low",         "blur_limit_high"),
        "scale":            ("scale_low",              "scale_high"),
        "scale_limit":      ("scale_limit_low",        "scale_limit_high"),
        "scale_range":      ("scale_range_low",        "scale_range_high"),
        # Distortion
        "distort_limit":    ("distort_low",            "distort_high"),
        # Compression
        "quality_range":    ("quality_low",            "quality_high"),
        # Color
        "threshold_range":  ("threshold_low",          "threshold_high"),
        # Noise
        "color_shift":      ("color_shift_low",        "color_shift_high"),
        "intensity":        ("intensity_low",           "intensity_high"),
        # Blur
        "sigma_x_limit":    ("sigma_x_limit_low",      "sigma_x_limit_high"),
        "sigma_y_limit":    ("sigma_y_limit_low",      "sigma_y_limit_high"),
        # Weather
        "slant_range":      ("slant_low",              "slant_high"),
        "num_shadows_limit": ("num_shadows_lower",     "num_shadows_upper"),
        "fog_coef_range":   ("fog_coef_lower",         "fog_coef_upper"),
        "num_flare_circles_range": ("num_flare_circles_low", "num_flare_circles_high"),
        # Dropout
        "num_holes_range":  ("num_holes_low",          "num_holes_high"),
        "hole_height_range": ("hole_height_low",       "hole_height_high"),
        "hole_width_range": ("hole_width_low",         "hole_width_high"),
    }

    kwargs: dict[str, Any] = {}
    consumed: set[str] = set()

    for out_key, pair in range_pairs.items():
        if pair is None:
            continue
        lo_key, hi_key = pair
        if lo_key in transform_cfg and hi_key in transform_cfg:
            lo = transform_cfg[lo_key]
            hi = transform_cfg[hi_key]
            kwargs[out_key] = (lo, hi)
            consumed.update({lo_key, hi_key})

    # Special case: shear stored as shear_low/shear_high → shear=(-x, x) or dict
    if "shear_low" in transform_cfg and "shear_high" in transform_cfg:
        kwargs["shear"] = (transform_cfg["shear_low"], transform_cfg["shear_high"])
        consumed.update({"shear_low", "shear_high"})

    # Special case: Affine scale stored as scale_low/scale_high
    if "scale_low" in transform_cfg and "scale_high" in transform_cfg:
        # Only if not already consumed (Perspective uses the same key names)
        if "scale" not in kwargs:
            kwargs["scale"] = (transform_cfg["scale_low"], transform_cfg["scale_high"])
            consumed.update({"scale_low", "scale_high"})

    # All remaining keys pass through verbatim (p, alpha, sigma, max_holes, etc.)
    for key, val in transform_cfg.items():
        if key not in skip and key not in consumed:
            kwargs[key] = val

    return kwargs

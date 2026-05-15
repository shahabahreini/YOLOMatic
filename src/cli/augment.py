"""Albumentations dataset augmentation wizard."""
from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.live import Live
from rich.panel import Panel

from src.augmentation.engine import AugmentationStats, SplitConfig, run_augmentation
from src.augmentation.profiles import (
    BUILT_IN_PROFILES,
    PROFILES_DIR,
    AugmentationProfile,
    clone_profile,
    delete_profile,
    ensure_builtin_profiles,
    list_profiles,
    load_profile,
    save_profile,
)
from src.augmentation.transforms import TRANSFORM_GROUPS, get_params_for_transform
from src.utils.cli import (
    NAV_BACK,
    ParameterDefinition,
    clear_screen,
    console,
    get_parameter_value_input,
    get_user_choice,
    get_user_multi_select,
    print_stylized_header,
    render_summary_panel,
)

try:
    from src.datasets.core import summarize_dataset
    from src.utils.project import list_dataset_directories
except ImportError:
    from datasets.core import summarize_dataset  # type: ignore[no-redef]
    from utils.project import list_dataset_directories  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_profile_names() -> list[str]:
    """Builtin names first (deduped), then user-saved names."""
    builtin = list(BUILT_IN_PROFILES.keys())
    saved = list_profiles(PROFILES_DIR)
    return list(dict.fromkeys(builtin + saved))


def _profile_description(name: str) -> str:
    try:
        p = load_profile(name, PROFILES_DIR)
        is_builtin = name in BUILT_IN_PROFILES and name not in list_profiles(PROFILES_DIR)
        tag = " [dim](built-in)[/dim]" if is_builtin else ""
        active = sum(1 for t in p.transforms if t.get("enabled", False))
        return (
            f"[bold cyan]{p.name}[/bold cyan]{tag}\n\n"
            f"{p.description}\n\n"
            f"Multiplier: [yellow]{p.multiplier}[/yellow]  |  "
            f"Active transforms: [yellow]{active}[/yellow]  |  "
            f"Include originals: [yellow]{p.include_originals}[/yellow]\n"
            f"[dim]Modified: {p.modified_at[:10]}[/dim]"
        )
    except Exception as exc:
        return f"[red]Error loading '{name}': {exc}[/red]"


# ---------------------------------------------------------------------------
# Profile selector helper
# ---------------------------------------------------------------------------

def _select_profile_name(title: str, *, breadcrumb_action: str) -> str | None:
    names = _all_profile_names()
    if not names:
        console.print(Panel(
            "[bold yellow]No profiles found.[/bold yellow]\n\nCreate a profile first.",
            border_style="yellow", padding=(1, 2),
        ))
        input("\nPress Enter to continue...")
        return None
    descriptions = {n: _profile_description(n) for n in names}
    choice = get_user_choice(
        names,
        allow_back=True,
        title=title,
        text="Select a profile:",
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Augment Dataset", "Profiles", breadcrumb_action],
    )
    return None if choice in (NAV_BACK, "Back") else choice


# ---------------------------------------------------------------------------
# Transform group editor  (Level 2 of the two-level editor)
# ---------------------------------------------------------------------------

def _edit_transform_group(
    group_name: str,
    members: list[str],
    transform_by_name: dict[str, dict],
) -> None:
    """
    Two-level group editor.
    Level 1: get_user_choice showing all transforms with enabled status.
    Level 2: per-transform get_user_multi_select for enable, p, and all specific params.
    """
    while True:
        clear_screen()
        print_stylized_header(f"{group_name} Transforms")

        descriptions: dict[str, str] = {}
        for t_name in members:
            cfg = transform_by_name.get(t_name, {})
            enabled = cfg.get("enabled", False)
            p = cfg.get("p", 0.5)
            specific = get_params_for_transform(t_name)[1:]  # skip p
            status = "[green]✓ Enabled[/green]" if enabled else "[dim]○ Disabled[/dim]"
            param_lines = []
            for pd in specific:
                val = cfg.get(pd.name, pd.default)
                param_lines.append(f"  [dim]{pd.description}:[/dim] [yellow]{val}[/yellow]")
            param_block = "\n".join(param_lines) if param_lines else "  [dim]No additional parameters[/dim]"
            descriptions[t_name] = (
                f"[bold cyan]{t_name}[/bold cyan]  {status}\n\n"
                f"p (probability): [yellow]{p}[/yellow]\n\n"
                f"{param_block}"
            )

        active_count = sum(
            1 for t in members if transform_by_name.get(t, {}).get("enabled", False)
        )
        choice = get_user_choice(
            members,
            allow_back=True,
            title=f"{group_name} — Select Transform",
            text=(
                f"Active: [bold yellow]{active_count}/{len(members)}[/bold yellow]  |  "
                "Select a transform to configure its parameters."
            ),
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Augment Dataset", "Profiles", group_name],
        )

        if choice in (NAV_BACK, "Back"):
            return
        if choice in members:
            _edit_single_transform(choice, transform_by_name, group_name)


def _edit_single_transform(
    t_name: str,
    transform_by_name: dict[str, dict],
    group_name: str,
) -> None:
    """Edit enable toggle, probability, and all specific params for one transform."""
    cfg = transform_by_name.get(t_name, {"name": t_name, "enabled": False, "p": 0.5})
    specific_params = get_params_for_transform(t_name)[1:]  # skip shared p param

    parameters: list[ParameterDefinition] = [
        ParameterDefinition(
            name="enabled",
            category=group_name,
            default=False,
            value_type="bool",
            description=f"Enable {t_name}",
            help_text=(
                f"Toggle [bold]{t_name}[/bold] on or off.\n\n"
                "[Space] toggles  [Enter/→] opens value editor"
            ),
        ),
        ParameterDefinition(
            name="p",
            category=group_name,
            default=0.5,
            value_type="float",
            description="Probability",
            help_text="Probability of applying this transform to each image. 0.0 = never, 1.0 = always.",
            min_value=0.0,
            max_value=1.0,
        ),
        *specific_params,
    ]

    pre_selected: set[str] = {"enabled"} if cfg.get("enabled", False) else set()
    pre_values: dict[str, Any] = {
        "enabled": cfg.get("enabled", False),
        "p": cfg.get("p", 0.5),
        **{pd.name: cfg.get(pd.name, pd.default) for pd in specific_params},
    }

    result = get_user_multi_select(
        parameters=parameters,
        title=f"Configure: {t_name}",
        instruction="[Space] Toggle Enable  [Enter/→] Edit Value  [F] Finish & Return",
        pre_selected=pre_selected,
        pre_values=pre_values,
    )
    if result is None:
        return

    selected_names, updated_values = result

    new_cfg: dict[str, Any] = {"name": t_name}
    new_cfg["enabled"] = "enabled" in selected_names
    new_cfg["p"] = float(updated_values.get("p", 0.5))
    for pd in specific_params:
        if pd.name in updated_values:
            new_cfg[pd.name] = updated_values[pd.name]
    transform_by_name[t_name] = new_cfg


# ---------------------------------------------------------------------------
# Transform editor  (Level 1: group menu)
# ---------------------------------------------------------------------------

def _edit_profile_transforms(profile: AugmentationProfile) -> AugmentationProfile | None:
    """
    Two-level transform editor.
    Level 1: group selection menu.
    Level 2: per-group get_user_multi_select.
    Returns modified profile, or None if user discards.
    """
    # Fast lookup: transform_name → current config dict
    transform_by_name: dict[str, dict] = {t["name"]: dict(t) for t in profile.transforms}
    # Ensure every known transform has an entry
    for group_members in TRANSFORM_GROUPS.values():
        for t_name in group_members:
            if t_name not in transform_by_name:
                transform_by_name[t_name] = {"name": t_name, "enabled": False, "p": 0.5}

    while True:
        clear_screen()
        print_stylized_header(f"Edit Transforms: {profile.name}")

        enabled_count = sum(1 for cfg in transform_by_name.values() if cfg.get("enabled", False))

        group_descriptions: dict[str, str] = {}
        for gname, members in TRANSFORM_GROUPS.items():
            active = [m for m in members if transform_by_name.get(m, {}).get("enabled", False)]
            lines = "\n".join(
                f"  {'[green]✓[/green]' if m in active else '[dim]○[/dim]'} {m}"
                for m in members
            )
            group_descriptions[gname] = (
                f"[bold cyan]{gname}[/bold cyan]\n\n"
                f"Total: {len(members)}  |  Active: [yellow]{len(active)}[/yellow]\n\n"
                f"{lines}"
            )

        group_descriptions.update({
            "Set Multiplier":    f"Currently [yellow]{profile.multiplier}[/yellow] augmented copies per source image.",
            "Set Seed":          f"Currently [yellow]{profile.seed}[/yellow] — controls shuffle for split redistribution.",
            "Include Originals": f"Currently [yellow]{profile.include_originals}[/yellow] — include source images in output.",
            "Save & Back":       "Save all changes and return to the profile manager.",
            "Discard & Back":    "Discard all changes and return without saving.",
        })

        choice = get_user_choice(
            [
                "[Transform Groups]",
                *list(TRANSFORM_GROUPS.keys()),
                "[Profile Settings]",
                "Set Multiplier",
                "Set Seed",
                "Include Originals",
                "[Actions]",
                "Save & Back",
                "Discard & Back",
            ],
            title=f"Transforms — {profile.name}",
            text=(
                f"Active transforms: [bold yellow]{enabled_count}[/bold yellow]  |  "
                f"Multiplier: [bold yellow]{profile.multiplier}[/bold yellow]  |  "
                f"Seed: [bold yellow]{profile.seed}[/bold yellow]\n\n"
                "Select a group to configure its transforms, or adjust profile settings."
            ),
            descriptions=group_descriptions,
            breadcrumbs=["YOLOmatic", "Augment Dataset", "Profiles", profile.name],
        )

        if choice == "Save & Back":
            profile.transforms = list(transform_by_name.values())
            profile.modified_at = datetime.now().isoformat(timespec="seconds")
            return profile

        elif choice == "Discard & Back":
            return None

        elif choice == "Set Multiplier":
            param = ParameterDefinition(
                name="multiplier", category="profile",
                default=profile.multiplier, value_type="int",
                description="Augmented copies per source image (1–20)",
                help_text=(
                    "Each source image produces this many augmented variants.\n"
                    "multiplier=3 triples your dataset size (plus originals if enabled).\n\n"
                    "[bold yellow]Tip:[/bold yellow] 3–5 is a good starting point for most datasets."
                ),
                min_value=1, max_value=20,
            )
            raw = get_parameter_value_input(param, profile.multiplier)
            if raw not in (None, NAV_BACK):
                profile.multiplier = int(raw)

        elif choice == "Set Seed":
            param = ParameterDefinition(
                name="seed", category="profile",
                default=profile.seed, value_type="int",
                description="Random seed (0–9999)",
                help_text=(
                    "Controls the shuffle order when redistributing images to train/val/test splits.\n"
                    "Using the same seed produces identical splits every run."
                ),
                min_value=0, max_value=9999,
            )
            raw = get_parameter_value_input(param, profile.seed)
            if raw not in (None, NAV_BACK):
                profile.seed = int(raw)

        elif choice == "Include Originals":
            param = ParameterDefinition(
                name="include_originals", category="profile",
                default=profile.include_originals, value_type="bool",
                description="Include original (non-augmented) images in the output?",
                help_text=(
                    "[bold]True (recommended):[/bold] output contains both the original images "
                    "and all augmented copies. Models benefit from seeing clean examples.\n\n"
                    "[bold]False:[/bold] output contains only augmented copies. "
                    "Use when the source dataset is very large and you want to cap disk usage."
                ),
            )
            raw = get_parameter_value_input(param, profile.include_originals)
            if raw not in (None, NAV_BACK):
                profile.include_originals = bool(raw)

        elif choice in TRANSFORM_GROUPS:
            _edit_transform_group(choice, TRANSFORM_GROUPS[choice], transform_by_name)


# ---------------------------------------------------------------------------
# Profile CRUD flows
# ---------------------------------------------------------------------------

def _create_profile_flow() -> None:
    clear_screen()
    print_stylized_header("Create Augmentation Profile")

    # Name
    name_param = ParameterDefinition(
        name="profile_name", category="profile", default="my_profile",
        value_type="str",
        description="Profile name (no spaces — use underscores)",
        help_text="Used as the YAML filename under configs/augmentation_profiles/.\nExample: vegetation_aerial, general_det, my_custom_profile",
    )
    raw_name = get_parameter_value_input(name_param, "my_profile")
    if raw_name in (None, NAV_BACK):
        return
    name = str(raw_name).strip().replace(" ", "_")

    # Description
    desc_param = ParameterDefinition(
        name="description", category="profile", default="Custom augmentation profile",
        value_type="str",
        description="Short description of this profile's purpose",
        help_text="Shown in the profile selector.\nExample: 'Optimized for aerial vegetation detection'",
    )
    raw_desc = get_parameter_value_input(desc_param, "Custom augmentation profile")
    if raw_desc in (None, NAV_BACK):
        return

    # Multiplier
    mult_param = ParameterDefinition(
        name="multiplier", category="profile", default=3,
        value_type="int",
        description="Augmented copies per source image (1–20)",
        help_text=(
            "3 means each source image produces 3 augmented variants.\n"
            "multiplier=1 applies one random transform per image.\n\n"
            "[bold yellow]Tip:[/bold yellow] 3–5 is a good starting point."
        ),
        min_value=1, max_value=20,
    )
    raw_mult = get_parameter_value_input(mult_param, 3)
    if raw_mult in (None, NAV_BACK):
        return

    # Include originals
    orig_param = ParameterDefinition(
        name="include_originals", category="profile", default=True,
        value_type="bool",
        description="Include original (non-augmented) images in the output?",
        help_text="[bold]True (recommended):[/bold] model sees both clean and augmented images.",
    )
    raw_orig = get_parameter_value_input(orig_param, True)
    if raw_orig in (None, NAV_BACK):
        return

    # Build initial profile with all transforms disabled
    now = datetime.now().isoformat(timespec="seconds")
    profile = AugmentationProfile(
        name=name,
        description=str(raw_desc),
        multiplier=int(raw_mult),
        seed=42,
        include_originals=bool(raw_orig),
        transforms=[
            {"name": t_name, "enabled": False, "p": 0.5}
            for group_members in TRANSFORM_GROUPS.values()
            for t_name in group_members
        ],
        created_at=now,
        modified_at=now,
    )

    # Configure transforms
    profile = _edit_profile_transforms(profile)
    if profile is None:
        return

    save_profile(profile, PROFILES_DIR)
    console.print(f"\n[bold green]Profile '{name}' saved to configs/augmentation_profiles/{name}.yaml[/bold green]")
    input("\nPress Enter to continue...")


def _edit_profile_flow(name: str) -> None:
    try:
        profile = load_profile(name, PROFILES_DIR)
    except Exception as exc:
        console.print(Panel(
            f"[bold red]Cannot load profile '{name}':[/bold red] {exc}",
            border_style="red", padding=(1, 2),
        ))
        input("\nPress Enter to continue...")
        return

    profile = _edit_profile_transforms(profile)
    if profile is None:
        return
    save_profile(profile, PROFILES_DIR)
    console.print(f"\n[bold green]Profile '{name}' updated.[/bold green]")
    input("\nPress Enter to continue...")


def _clone_profile_flow() -> None:
    source_name = _select_profile_name(
        "Clone Profile — Select Source",
        breadcrumb_action="Clone",
    )
    if source_name is None:
        return

    clear_screen()
    print_stylized_header("Clone Profile — New Name")

    name_param = ParameterDefinition(
        name="new_name", category="profile",
        default=f"{source_name}_copy",
        value_type="str",
        description="Name for the cloned profile",
        help_text=(
            "The source profile is not modified.\n"
            "Enter a unique name (no spaces)."
        ),
    )
    raw = get_parameter_value_input(name_param, f"{source_name}_copy")
    if raw in (None, NAV_BACK):
        return
    new_name = str(raw).strip().replace(" ", "_")

    try:
        new_profile = clone_profile(source_name, new_name, PROFILES_DIR)
        save_profile(new_profile, PROFILES_DIR)
        console.print(
            f"\n[bold green]Profile '{new_name}' cloned from '{source_name}'.[/bold green]"
        )
    except Exception as exc:
        console.print(Panel(
            f"[bold red]Clone failed:[/bold red] {exc}",
            border_style="red", padding=(1, 2),
        ))
    input("\nPress Enter to continue...")


def _delete_profile_flow() -> None:
    source_name = _select_profile_name(
        "Delete Profile — Select",
        breadcrumb_action="Delete",
    )
    if source_name is None:
        return

    # Prevent deleting built-in that hasn't been saved to disk
    if source_name in BUILT_IN_PROFILES and source_name not in list_profiles(PROFILES_DIR):
        console.print(Panel(
            f"[bold yellow]'{source_name}' is a built-in profile and has not been saved to disk.\n"
            "There is nothing to delete.",
            border_style="yellow", padding=(1, 2),
        ))
        input("\nPress Enter to continue...")
        return

    clear_screen()
    print_stylized_header("Delete Profile — Confirm")
    console.print(Panel(
        f"[bold yellow]Delete '[cyan]{source_name}[/cyan]'?[/bold yellow]\n\n"
        "This action cannot be undone. The YAML file will be permanently removed.",
        border_style="yellow", padding=(1, 2),
    ))
    confirm = get_user_choice(
        ["Yes, Delete", "No, Keep"],
        title="Confirm Deletion",
        text=f"Delete profile '{source_name}'?",
        descriptions={
            "Yes, Delete": "[red]Permanently remove this profile from configs/augmentation_profiles/.[/red]",
            "No, Keep":    "[green]Cancel — keep the profile.[/green]",
        },
        breadcrumbs=["YOLOmatic", "Augment Dataset", "Profiles", "Delete"],
    )
    if confirm == "Yes, Delete":
        delete_profile(source_name, PROFILES_DIR)
        console.print(f"\n[bold green]Profile '{source_name}' deleted.[/bold green]")
    input("\nPress Enter to continue...")


# ---------------------------------------------------------------------------
# Profile manager
# ---------------------------------------------------------------------------

def _profile_manager_menu() -> None:
    while True:
        clear_screen()
        print_stylized_header("Augmentation Profile Manager")

        all_names = _all_profile_names()
        saved_on_disk = set(list_profiles(PROFILES_DIR))
        descriptions: dict[str, str] = {n: _profile_description(n) for n in all_names}
        descriptions.update({
            "Create New Profile": (
                "[bold cyan]Create New Profile[/bold cyan]\n\n"
                "Start a fresh augmentation profile from scratch.\n\n"
                "You will set a name, description, multiplier, and then configure "
                "which Albumentations transforms to enable and tune."
            ),
            "Edit Profile": (
                "[bold cyan]Edit Profile[/bold cyan]\n\n"
                "Select a saved or built-in profile and modify its transforms and parameters.\n\n"
                "Built-in profiles are read-only until you first edit and save them."
            ),
            "Clone Profile": (
                "[bold cyan]Clone Profile[/bold cyan]\n\n"
                "Copy any profile (including built-ins) under a new name. "
                "The clone is fully independent of the original."
            ),
            "Delete Profile": (
                "[bold cyan]Delete Profile[/bold cyan]\n\n"
                "Permanently remove a saved profile YAML from disk. "
                "Built-in profiles that have not been saved to disk cannot be deleted."
            ),
        })

        menu_items = [
            "[Saved Profiles]",
            *all_names,
            "[Actions]",
            "Create New Profile",
            "Edit Profile",
            "Clone Profile",
            "Delete Profile",
            "Back",
        ]

        choice = get_user_choice(
            menu_items,
            title="Profile Manager",
            text=(
                "Profiles are stored in [cyan]configs/augmentation_profiles/[/cyan].\n"
                "Built-in profiles are pre-loaded and appear at the top. "
                "Click a profile to edit it directly."
            ),
            descriptions=descriptions,
            breadcrumbs=["YOLOmatic", "Augment Dataset", "Profiles"],
        )

        if choice in (NAV_BACK, "Back"):
            return
        elif choice == "Create New Profile":
            _create_profile_flow()
        elif choice == "Edit Profile":
            name = _select_profile_name("Edit Profile", breadcrumb_action="Edit")
            if name:
                _edit_profile_flow(name)
        elif choice == "Clone Profile":
            _clone_profile_flow()
        elif choice == "Delete Profile":
            _delete_profile_flow()
        elif choice in all_names:
            _edit_profile_flow(choice)


# ---------------------------------------------------------------------------
# Run augmentation sub-flows
# ---------------------------------------------------------------------------

def _select_dataset() -> Path | None:
    clear_screen()
    print_stylized_header("Select Source Dataset")

    try:
        datasets = list_dataset_directories()
    except Exception:
        datasets = []

    if not datasets:
        console.print(Panel(
            "[bold yellow]No datasets found in ./datasets/[/bold yellow]\n\n"
            "Place a YOLO or COCO dataset (with a data.yaml) in the datasets/ folder first.",
            border_style="yellow", padding=(1, 2),
        ))
        input("\nPress Enter to continue...")
        return None

    descriptions: dict[str, str] = {}
    path_by_name: dict[str, Path] = {}
    for ds in datasets:
        ds_path = Path(ds["path"])
        name = ds["name"]
        path_by_name[name] = ds_path
        try:
            summary = summarize_dataset(str(ds_path))
            class_preview = ", ".join(summary.classes[:6])
            if len(summary.classes) > 6:
                class_preview += "…"
            descriptions[name] = (
                f"[bold cyan]{name}[/bold cyan]\n\n"
                f"Format: [yellow]{summary.format}[/yellow]  |  "
                f"Images: [yellow]{summary.image_count}[/yellow]  |  "
                f"Classes: [yellow]{len(summary.classes)}[/yellow]  |  "
                f"Size: [yellow]{ds.get('size', '?')}[/yellow]\n\n"
                f"[dim]{class_preview}[/dim]"
            )
        except Exception:
            descriptions[name] = f"[dim]{ds_path}[/dim]"

    options = list(path_by_name.keys())
    choice = get_user_choice(
        options,
        allow_back=True,
        title="Select Source Dataset",
        text=(
            "Choose the dataset to augment.\n"
            "All images across all splits will be pooled, augmented, then redistributed."
        ),
        descriptions=descriptions,
        breadcrumbs=["YOLOmatic", "Augment Dataset", "Dataset"],
    )
    if choice in (NAV_BACK, "Back"):
        return None
    return path_by_name.get(choice)


def _select_output_format() -> str | None:
    clear_screen()
    print_stylized_header("Select Output Format")
    choice = get_user_choice(
        ["YOLO Detection", "YOLO Segmentation", "COCO"],
        allow_back=True,
        title="Output Format",
        text="Select the annotation format for the augmented output dataset:",
        descriptions={
            "YOLO Detection": (
                "[bold cyan]YOLO Detection Format[/bold cyan]\n\n"
                "Output structure:\n"
                "  train/images/  train/labels/  data.yaml\n\n"
                "Label format: [dim]class_id cx cy w h[/dim] (normalized)\n\n"
                "Compatible with all Ultralytics YOLO variants."
            ),
            "YOLO Segmentation": (
                "[bold cyan]YOLO Segmentation Format[/bold cyan]\n\n"
                "Output structure:\n"
                "  train/images/  train/labels/  data.yaml\n\n"
                "Label format: [dim]class_id x1 y1 x2 y2 … xn yn[/dim] (normalized polygon)\n\n"
                "Compatible with yolo26, yolov11-seg, yolov8-seg, and other -seg variants.\n\n"
                "[dim]Note: requires a segmentation-format source dataset.[/dim]"
            ),
            "COCO": (
                "[bold cyan]COCO JSON Format[/bold cyan]\n\n"
                "Output structure:\n"
                "  annotations/instances_train.json\n"
                "  train/images/\n\n"
                "Standard COCO JSON with bboxes and segmentation polygons.\n\n"
                "Compatible with Detectron2, RF-DETR, MMDetection, and most frameworks."
            ),
        },
        breadcrumbs=["YOLOmatic", "Augment Dataset", "Format"],
    )
    return None if choice in (NAV_BACK, "Back") else choice


def _configure_split_ratios() -> SplitConfig | None:
    clear_screen()
    print_stylized_header("Configure Split Ratios")
    choice = get_user_choice(
        [
            "Standard (70 / 20 / 10)",
            "Large Train (80 / 15 / 5)",
            "Half-Half (50 / 30 / 20)",
            "Custom",
            "Back",
        ],
        title="Train / Val / Test Split",
        text=(
            "All images from the source dataset are pooled and randomly redistributed.\n"
            "Choose how to divide the augmented output:"
        ),
        descriptions={
            "Standard (70 / 20 / 10)": (
                "70% train — 20% validation — 10% test\n\n"
                "Recommended for most datasets. Provides sufficient validation data "
                "without sacrificing too much training data."
            ),
            "Large Train (80 / 15 / 5)": (
                "80% train — 15% validation — 5% test\n\n"
                "Best for large datasets (>10k images) where you have enough data "
                "to spare smaller validation and test sets."
            ),
            "Half-Half (50 / 30 / 20)": (
                "50% train — 30% validation — 20% test\n\n"
                "Useful when validation and test accuracy matter as much as training. "
                "Increases confidence in generalization metrics."
            ),
            "Custom": (
                "Enter three custom ratios.\n\n"
                "The values must sum to 1.0.\n"
                "Example: 0.75 train / 0.15 val / 0.10 test."
            ),
        },
        breadcrumbs=["YOLOmatic", "Augment Dataset", "Splits"],
    )
    if choice in (NAV_BACK, "Back"):
        return None
    if choice == "Standard (70 / 20 / 10)":
        return SplitConfig(0.70, 0.20, 0.10)
    if choice == "Large Train (80 / 15 / 5)":
        return SplitConfig(0.80, 0.15, 0.05)
    if choice == "Half-Half (50 / 30 / 20)":
        return SplitConfig(0.50, 0.30, 0.20)

    # Custom
    clear_screen()
    print_stylized_header("Custom Split Ratios")
    params = [
        ParameterDefinition("train_ratio", "split", 0.70, "float",
                            "Train ratio", "Fraction for training. e.g. 0.70",
                            min_value=0.0, max_value=1.0),
        ParameterDefinition("val_ratio",   "split", 0.20, "float",
                            "Validation ratio", "Fraction for validation. e.g. 0.20",
                            min_value=0.0, max_value=1.0),
        ParameterDefinition("test_ratio",  "split", 0.10, "float",
                            "Test ratio", "Fraction for test. e.g. 0.10",
                            min_value=0.0, max_value=1.0),
    ]
    result = get_user_multi_select(
        parameters=params,
        title="Custom Split Ratios",
        instruction="[Enter/→] Edit Value  [F] Finish  — values must sum to 1.0",
        pre_selected={"train_ratio", "val_ratio", "test_ratio"},
        pre_values={"train_ratio": 0.70, "val_ratio": 0.20, "test_ratio": 0.10},
    )
    if result is None:
        return None

    _, vals = result
    train = float(vals.get("train_ratio", 0.70))
    val   = float(vals.get("val_ratio",   0.20))
    test  = float(vals.get("test_ratio",  0.10))
    total = train + val + test
    if abs(total - 1.0) > 0.02:
        console.print(Panel(
            f"[bold red]Ratios must sum to 1.0 (got {total:.3f}). Defaulting to 70 / 20 / 10.[/bold red]",
            border_style="red", padding=(1, 2),
        ))
        input("\nPress Enter to continue...")
        return SplitConfig(0.70, 0.20, 0.10)
    return SplitConfig(train, val, test)


# ---------------------------------------------------------------------------
# Progress runner
# ---------------------------------------------------------------------------

def _run_with_progress(
    dataset_path: Path,
    output_name: str,
    profile: AugmentationProfile,
    split_config: SplitConfig,
    output_format: str,
) -> None:
    log_lines: list[str] = []
    stats_holder: list[AugmentationStats] = []
    error_holder: list[Exception] = []
    progress_state: dict = {"done": 0, "total": 0, "current": ""}
    done_event = threading.Event()

    def _worker() -> None:
        try:
            def callback(done: int, total: int, msg: str) -> None:
                progress_state.update({"done": done, "total": total, "current": msg})
                if msg:
                    log_lines.append(f"  [{done}/{total}] {msg}")

            stats = run_augmentation(
                source_dataset_path=dataset_path,
                output_name=output_name,
                profile=profile,
                split_config=split_config,
                output_format=output_format,
                progress_callback=callback,
            )
            stats_holder.append(stats)
        except Exception as exc:
            error_holder.append(exc)
        finally:
            done_event.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    with Live(refresh_per_second=4) as live:
        while not done_event.is_set():
            done = progress_state["done"]
            total = progress_state["total"]
            pct = f"{done / total * 100:.0f}%" if total > 0 else "..."
            lines = log_lines[-18:]
            live.update(Panel(
                "\n".join(lines) or "[dim]Initializing...[/dim]",
                title=f"[cyan]Augmenting [{pct}][/cyan]",
                border_style="cyan",
            ))
            time.sleep(0.25)
        # Final frame
        lines = log_lines[-18:]
        live.update(Panel(
            "\n".join(lines) or "[dim]Done.[/dim]",
            title="[green]Augmentation Complete[/green]",
            border_style="green",
        ))

    thread.join()

    if error_holder:
        console.print(Panel(
            f"[bold red]Augmentation failed:[/bold red] {error_holder[0]}\n\n"
            "[dim]Check that albumentations is installed and the dataset path is valid.[/dim]",
            border_style="red", padding=(1, 2),
        ))
        return

    if stats_holder:
        s = stats_holder[0]
        render_summary_panel("Augmentation Results", {
            "Source Dataset":       s.source_dataset,
            "Output Path":          s.output_path,
            "Format Detected":      s.annotation_format,
            "Output Format":        s.output_format,
            "Profile":              s.profile_name,
            "Source Images":        str(s.total_source_images),
            "Total Output Images":  str(s.total_output_images),
            "Train":                str(s.split_counts.get("train", 0)),
            "Validation":           str(s.split_counts.get("valid", 0)),
            "Test":                 str(s.split_counts.get("test", 0)),
            "Skipped":              str(s.images_skipped),
            "Annotations Removed":  str(s.annotations_discarded),
            "Time":                 f"{s.elapsed_seconds:.1f}s",
        })


# ---------------------------------------------------------------------------
# Main run augmentation flow
# ---------------------------------------------------------------------------

def _run_augmentation_flow() -> None:
    # Step 1: Dataset
    dataset_path = _select_dataset()
    if dataset_path is None:
        return

    # Step 2: Profile
    profile_name = _select_profile_name(
        "Select Augmentation Profile",
        breadcrumb_action="Select",
    )
    if profile_name is None:
        return
    try:
        profile = load_profile(profile_name, PROFILES_DIR)
    except Exception as exc:
        console.print(Panel(
            f"[bold red]Failed to load profile:[/bold red] {exc}",
            border_style="red", padding=(1, 2),
        ))
        input("\nPress Enter to continue...")
        return

    # Step 3: Output format
    output_format = _select_output_format()
    if output_format is None:
        return

    # Step 4: Split ratios
    split_config = _configure_split_ratios()
    if split_config is None:
        return

    # Step 5: Output name
    default_name = dataset_path.name + "_augmented"
    out_param = ParameterDefinition(
        name="output_name", category="output",
        default=default_name, value_type="str",
        description="Output dataset folder name (placed alongside source in datasets/)",
        help_text=(
            f"Output will be written to: [cyan]datasets/{default_name}/[/cyan]\n\n"
            "Change this to avoid overwriting existing datasets or to give the "
            "augmented dataset a descriptive name."
        ),
    )
    raw_name = get_parameter_value_input(out_param, default_name)
    if raw_name in (None, NAV_BACK):
        return
    output_name = str(raw_name).strip()

    # Estimate output size
    try:
        summary = summarize_dataset(str(dataset_path))
        source_count: int | str = summary.image_count
    except Exception:
        source_count = "unknown"

    if isinstance(source_count, int):
        aug_count = source_count * profile.multiplier
        orig_count = source_count if profile.include_originals else 0
        estimated: str = f"~{aug_count + orig_count:,}"
    else:
        estimated = "unknown"

    active_transforms = sum(1 for t in profile.transforms if t.get("enabled", False))

    # Step 6: Confirm
    clear_screen()
    print_stylized_header("Augment Dataset — Confirm")
    render_summary_panel("Augmentation Plan", {
        "Source Dataset":         dataset_path.name,
        "Source Images":          str(source_count),
        "Profile":                profile.name,
        "Active Transforms":      str(active_transforms),
        "Multiplier":             f"×{profile.multiplier}",
        "Include Originals":      str(profile.include_originals),
        "Output Format":          output_format,
        "Split (train/val/test)": (
            f"{split_config.train_ratio:.0%} / "
            f"{split_config.val_ratio:.0%} / "
            f"{split_config.test_ratio:.0%}"
        ),
        "Estimated Output":       estimated,
        "Output Location":        f"datasets/{output_name}/",
    })

    confirm = get_user_choice(
        ["Start Augmentation", "Back"],
        title="Confirm",
        text=(
            "Review the plan above.\n"
            "This operation may take several minutes for large datasets.\n"
            "The original dataset is not modified."
        ),
        descriptions={
            "Start Augmentation": (
                "[green]Begin augmentation.[/green]\n\n"
                "The original dataset is never modified. "
                f"Output will be written to datasets/{output_name}/."
            ),
            "Back": "[dim]Return and change settings.[/dim]",
        },
        breadcrumbs=["YOLOmatic", "Augment Dataset", "Confirm"],
    )
    if confirm in (NAV_BACK, "Back"):
        return

    # Step 7: Run
    clear_screen()
    print_stylized_header("Running Augmentation...")
    _run_with_progress(dataset_path, output_name, profile, split_config, output_format)
    input("\nPress Enter to return to main menu...")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Seed built-in profiles on first run
    ensure_builtin_profiles(PROFILES_DIR)

    while True:
        clear_screen()
        print_stylized_header("Dataset Augmentation")

        choice = get_user_choice(
            [
                "[Profile Management]",
                "Manage Augmentation Profiles",
                "[Run Augmentation]",
                "Augment Dataset",
                "Back",
            ],
            title="Dataset Augmentation",
            text=(
                "Expand and diversify YOLO or COCO datasets using Albumentations transforms.\n"
                "Profiles store your transform configuration and can be reused across datasets."
            ),
            descriptions={
                "Manage Augmentation Profiles": (
                    "[bold cyan]Profile Manager[/bold cyan]\n\n"
                    "Create, edit, clone, and delete named augmentation profiles.\n\n"
                    "Each profile stores:\n"
                    "  • A curated set of Albumentations transforms with parameters\n"
                    "  • Multiplier — how many augmented copies per source image\n"
                    "  • Whether to include original images alongside augmented ones\n\n"
                    "Three built-in profiles are pre-loaded:\n"
                    "  • [bold]vegetation_aerial_optimal[/bold] — tuned for QGIS NIR aerial imagery\n"
                    "  • [bold]general_detection[/bold] — conservative baseline for any dataset\n"
                    "  • [bold]minimal[/bold] — D4 symmetry only, zero annotation risk"
                ),
                "Augment Dataset": (
                    "[bold cyan]Run Augmentation[/bold cyan]\n\n"
                    "Select a source dataset and an augmentation profile, then:\n\n"
                    "  1. Choose output format: YOLO Detection, YOLO Segmentation, or COCO\n"
                    "  2. Set train / val / test split ratios\n"
                    "  3. Confirm and run\n\n"
                    "Supports [bold]YOLO bbox[/bold] and [bold]YOLO seg[/bold] formats "
                    "(auto-detected from source).\n"
                    "Output is a new independent dataset folder — originals are never modified."
                ),
            },
            breadcrumbs=["YOLOmatic", "Augment Dataset"],
        )

        if choice in (NAV_BACK, "Back"):
            return
        elif choice == "Manage Augmentation Profiles":
            _profile_manager_menu()
        elif choice == "Augment Dataset":
            _run_augmentation_flow()

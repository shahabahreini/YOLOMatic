"""Albumentations dataset augmentation wizard."""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.augmentation.engine import (
    AugmentationStats,
    SplitConfig,
    collect_all_images,
    run_augmentation,
)
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
from src.augmentation.transforms import (
    TRANSFORM_GROUPS,
    get_params_for_transform,
    get_transform_guidance,
)
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
    from src.utils.project import list_dataset_directories
except ImportError:
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
    if name == "AI Recommendation":
        return (
            "[bold green]AI Recommendation[/bold green]\n\n"
            "Let OpenAI or Gemini inspect your dataset images and context to "
            "recommend and save a custom Albumentations augmentation pipeline."
        )
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
    if breadcrumb_action == "Select":
        names = ["AI Recommendation"] + names
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
# Unified profile editor helpers
# ---------------------------------------------------------------------------

def _copy_profile(profile: AugmentationProfile) -> AugmentationProfile:
    """Return a detached profile copy so editing built-ins is non-mutating until save."""
    return AugmentationProfile(
        name=profile.name,
        description=profile.description,
        multiplier=profile.multiplier,
        seed=profile.seed,
        include_originals=profile.include_originals,
        transforms=[dict(t) for t in profile.transforms],
        created_at=profile.created_at,
        modified_at=profile.modified_at,
        schema_version=profile.schema_version,
    )


def _ensure_transform_entries(profile: AugmentationProfile) -> dict[str, dict]:
    """Build a complete transform config map from the profile and catalog."""
    transform_by_name: dict[str, dict] = {t["name"]: dict(t) for t in profile.transforms}
    for group_members in TRANSFORM_GROUPS.values():
        for t_name in group_members:
            transform_by_name.setdefault(
                t_name,
                {"name": t_name, "enabled": False, "p": 0.5},
            )
    return transform_by_name


def _transform_param_key(t_name: str, param_name: str) -> str:
    return f"{t_name}.{param_name}"


def _profile_setting_definitions(profile: AugmentationProfile) -> list[ParameterDefinition]:
    return [
        ParameterDefinition(
            name="multiplier",
            category="Profile Settings",
            default=profile.multiplier,
            value_type="int",
            description="Augmented copies per source image",
            help_text=(
                "Each source image produces this many randomized augmented variants. "
                "Start with 2-5 for most datasets; higher values increase disk usage and can "
                "over-represent synthetic variation."
            ),
            min_value=1,
            max_value=20,
            affects="Controls output dataset size before originals are optionally added.",
            config_section="augmentation.profile",
        ),
        ParameterDefinition(
            name="seed",
            category="Profile Settings",
            default=profile.seed,
            value_type="int",
            description="Random seed for reproducible redistribution",
            help_text=(
                "Using the same seed keeps train/val/test redistribution stable across runs. "
                "Change it when you intentionally want a different split shuffle."
            ),
            min_value=0,
            max_value=9999,
            affects="Controls deterministic split shuffling, not the visual strength of transforms.",
            config_section="augmentation.profile",
        ),
        ParameterDefinition(
            name="include_originals",
            category="Profile Settings",
            default=profile.include_originals,
            value_type="bool",
            description="Include clean source images in the output",
            help_text=(
                "True is recommended: the model sees both clean images and augmented variants. "
                "Set False only when the source dataset is already large or disk space is tight."
            ),
            affects="Controls whether output contains original images alongside augmented copies.",
            config_section="augmentation.profile",
        ),
    ]


def _build_profile_editor_state(
    profile: AugmentationProfile,
    transform_by_name: dict[str, dict],
) -> tuple[list[ParameterDefinition], set[str], dict[str, Any], dict[str, tuple[str, str]]]:
    """
    Build the flat parameter editor state.

    field_map maps editor parameter names to (transform_name, transform_field). Profile
    settings are not included because their editor names map directly to profile fields.
    """
    parameters = _profile_setting_definitions(profile)
    selected: set[str] = {"multiplier", "seed", "include_originals"}
    values: dict[str, Any] = {
        "multiplier": profile.multiplier,
        "seed": profile.seed,
        "include_originals": profile.include_originals,
    }
    field_map: dict[str, tuple[str, str]] = {}

    for group_name, members in TRANSFORM_GROUPS.items():
        for t_name in members:
            cfg = transform_by_name[t_name]
            guidance = get_transform_guidance(t_name)
            enabled_key = _transform_param_key(t_name, "enabled")
            p_key = _transform_param_key(t_name, "p")
            is_enabled = bool(cfg.get("enabled", False))

            parameters.append(
                ParameterDefinition(
                    name=enabled_key,
                    category=group_name,
                    default=False,
                    value_type="bool",
                    description=f"{t_name}: enable transform",
                    help_text=(
                        f"{guidance['summary']}\n\n"
                        f"When to use: {guidance['use']}\n\n"
                        f"Watch out: {guidance['caution']}"
                    ),
                    affects=guidance["summary"],
                    config_section=f"augmentation.{t_name}",
                )
            )
            field_map[enabled_key] = (t_name, "enabled")
            values[enabled_key] = is_enabled
            if is_enabled:
                selected.add(enabled_key)

            p_def = get_params_for_transform(t_name)[0]
            parameters.append(
                replace(
                    p_def,
                    name=p_key,
                    category=group_name,
                    description=f"{t_name}: probability",
                    help_text=(
                        f"{p_def.help_text}\n\n"
                        f"{guidance['summary']}\n\n"
                        "Practical starting points: 0.1-0.2 for aggressive or weather effects, "
                        "0.3-0.5 for common geometric/color variation, 1.0 only for safe symmetry."
                    ),
                    affects=f"Controls how often {t_name} is considered during augmentation.",
                    config_section=f"augmentation.{t_name}",
                )
            )
            field_map[p_key] = (t_name, "p")
            values[p_key] = cfg.get("p", p_def.default)
            if is_enabled:
                selected.add(p_key)

            for param in get_params_for_transform(t_name)[1:]:
                key = _transform_param_key(t_name, param.name)
                parameters.append(
                    replace(
                        param,
                        name=key,
                        category=group_name,
                        description=f"{t_name}: {param.description}",
                        help_text=(
                            f"{param.help_text}\n\n"
                            f"Transform effect: {guidance['summary']}\n"
                            f"Use when: {guidance['use']}"
                        ),
                        affects=param.affects or guidance["summary"],
                        config_section=f"augmentation.{t_name}",
                    )
                )
                field_map[key] = (t_name, param.name)
                values[key] = cfg.get(param.name, param.default)
                if is_enabled:
                    selected.add(key)

    return parameters, selected, values, field_map


def _apply_profile_editor_result(
    profile: AugmentationProfile,
    transform_by_name: dict[str, dict],
    selected_names: set[str],
    updated_values: dict[str, Any],
    field_map: dict[str, tuple[str, str]],
) -> AugmentationProfile:
    profile.multiplier = int(updated_values.get("multiplier", profile.multiplier))
    profile.seed = int(updated_values.get("seed", profile.seed))
    profile.include_originals = bool(
        updated_values.get("include_originals", profile.include_originals)
    )

    for editor_name, (t_name, field_name) in field_map.items():
        cfg = transform_by_name.setdefault(
            t_name,
            {"name": t_name, "enabled": False, "p": 0.5},
        )
        if field_name == "enabled":
            cfg["enabled"] = editor_name in selected_names
        else:
            cfg[field_name] = updated_values.get(editor_name, cfg.get(field_name))

    profile.transforms = [
        transform_by_name[t_name]
        for members in TRANSFORM_GROUPS.values()
        for t_name in members
    ]
    profile.modified_at = datetime.now().isoformat(timespec="seconds")
    return profile


# ---------------------------------------------------------------------------
# Transform editor
# ---------------------------------------------------------------------------

def _edit_profile_transforms(profile: AugmentationProfile) -> AugmentationProfile | None:
    """
    Flat profile editor modeled after the fully customized training config flow.

    The editor keeps profile settings, transform enable toggles, probabilities,
    and transform-specific parameters in one navigable list grouped by transform
    family. This avoids the old group -> transform -> parameter page stack.

    Returns modified profile, or None if user discards.
    """
    working_profile = _copy_profile(profile)
    transform_by_name = _ensure_transform_entries(working_profile)
    parameters, selected, values, field_map = _build_profile_editor_state(
        working_profile,
        transform_by_name,
    )

    result = get_user_multi_select(
        parameters=parameters,
        title=f"Augmentation Profile — {working_profile.name}",
        instruction=(
            "Use Space on a *.enabled row to enable/disable transforms. "
            "Press Enter/Right to edit profile values, probabilities, and parameters. "
            "Press F when done."
        ),
        pre_selected=selected,
        pre_values=values,
    )
    if result is None:
        return None

    selected_names, updated_values = result
    updated_profile = _apply_profile_editor_result(
        working_profile,
        transform_by_name,
        selected_names,
        updated_values,
        field_map,
    )

    active_count = sum(1 for t in updated_profile.transforms if t.get("enabled", False))
    choice = get_user_choice(
        ["Save & Back", "Review Again", "Discard & Back"],
        title=f"Save Profile — {updated_profile.name}",
        text=(
            f"Active transforms: [bold yellow]{active_count}[/bold yellow]\n"
            f"Multiplier: [bold yellow]{updated_profile.multiplier}[/bold yellow]\n"
            f"Seed: [bold yellow]{updated_profile.seed}[/bold yellow]\n"
            f"Include originals: [bold yellow]{updated_profile.include_originals}[/bold yellow]"
        ),
        descriptions={
            "Save & Back": "Persist this profile and return to the profile manager.",
            "Review Again": "Return to the unified editor with the current changes loaded.",
            "Discard & Back": "Discard these edits and return without saving.",
        },
        breadcrumbs=["YOLOmatic", "Augment Dataset", "Profiles", updated_profile.name],
    )
    if choice == "Review Again":
        return _edit_profile_transforms(updated_profile)
    if choice == "Save & Back":
        return updated_profile
    return None


# ---------------------------------------------------------------------------
# Profile CRUD flows
# ---------------------------------------------------------------------------

def _create_profile_flow() -> None:
    clear_screen()
    print_stylized_header("Create Augmentation Profile")

    basics = [
        ParameterDefinition(
            name="profile_name",
            category="Profile Basics",
            default="my_profile",
            value_type="str",
            description="Profile name used as the YAML filename",
            help_text=(
                "Use a short unique name. Spaces are converted to underscores.\n"
                "Examples: vegetation_aerial, general_det, my_custom_profile"
            ),
            config_section="augmentation.profile",
        ),
        ParameterDefinition(
            name="description",
            category="Profile Basics",
            default="Custom augmentation profile",
            value_type="str",
            description="Short purpose shown in the profile selector",
            help_text=(
                "Describe the deployment condition or dataset this profile is designed for. "
                "Example: Optimized for aerial vegetation detection."
            ),
            config_section="augmentation.profile",
        ),
        ParameterDefinition(
            name="multiplier",
            category="Profile Basics",
            default=3,
            value_type="int",
            description="Augmented copies per source image",
            help_text=(
                "3 means each source image produces 3 randomized augmented variants. "
                "Start with 2-5 for most datasets."
            ),
            min_value=1,
            max_value=20,
            affects="Controls output dataset size before originals are optionally added.",
            config_section="augmentation.profile",
        ),
        ParameterDefinition(
            name="seed",
            category="Profile Basics",
            default=42,
            value_type="int",
            description="Random seed for reproducible redistribution",
            help_text="Use the same seed to keep split redistribution repeatable.",
            min_value=0,
            max_value=9999,
            config_section="augmentation.profile",
        ),
        ParameterDefinition(
            name="include_originals",
            category="Profile Basics",
            default=True,
            value_type="bool",
            description="Include clean source images in the output",
            help_text="True is recommended so the model sees clean images and augmented variants.",
            affects="Controls whether originals are copied into the augmented output dataset.",
            config_section="augmentation.profile",
        ),
    ]
    result = get_user_multi_select(
        parameters=basics,
        title="Create Augmentation Profile",
        instruction="Set the profile basics, then press F to configure transforms.",
        pre_selected={p.name for p in basics},
        pre_values={
            "profile_name": "my_profile",
            "description": "Custom augmentation profile",
            "multiplier": 3,
            "seed": 42,
            "include_originals": True,
        },
    )
    if result is None:
        return
    _, values = result
    name = str(values.get("profile_name", "my_profile")).strip().replace(" ", "_")
    raw_desc = values.get("description", "Custom augmentation profile")

    # Build initial profile with all transforms disabled
    now = datetime.now().isoformat(timespec="seconds")
    profile = AugmentationProfile(
        name=name,
        description=str(raw_desc),
        multiplier=int(values.get("multiplier", 3)),
        seed=int(values.get("seed", 42)),
        include_originals=bool(values.get("include_originals", True)),
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
        datasets = list_dataset_directories(include_size=False)
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
        descriptions[name] = _quick_dataset_description(name, ds_path)

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


def _quick_dataset_description(name: str, ds_path: Path) -> str:
    """Build a selector description without recursively scanning large datasets."""
    metadata: dict[str, Any] = {}
    yaml_name = ""
    for candidate in ("data.yaml", "dataset.yaml"):
        yaml_path = ds_path / candidate
        if not yaml_path.exists():
            continue
        try:
            loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            metadata = loaded if isinstance(loaded, dict) else {}
            yaml_name = candidate
        except Exception:
            metadata = {}
        break

    if metadata:
        raw_names = metadata.get("names", [])
        if isinstance(raw_names, dict):
            def _sort_key(value: Any) -> tuple[int, Any]:
                text = str(value)
                return (0, int(text)) if text.isdigit() else (1, text)

            classes = [str(raw_names[key]) for key in sorted(raw_names, key=_sort_key)]
        elif isinstance(raw_names, list):
            classes = [str(item) for item in raw_names]
        else:
            classes = []
        class_preview = ", ".join(classes[:6])
        if len(classes) > 6:
            class_preview += "…"
        if not class_preview:
            class_preview = "No classes listed"
        task = str(metadata.get("task", "unknown"))
        return (
            f"[bold cyan]{name}[/bold cyan]\n\n"
            f"Format: [yellow]yolo[/yellow]  |  "
            f"Task: [yellow]{task}[/yellow]  |  "
            f"Classes: [yellow]{len(classes)}[/yellow]\n"
            f"Config: [dim]{yaml_name}[/dim]\n\n"
            f"[dim]{class_preview}[/dim]"
        )

    if (ds_path / "annotations").exists():
        return (
            f"[bold cyan]{name}[/bold cyan]\n\n"
            "Format: [yellow]coco[/yellow]\n"
            f"[dim]{ds_path}[/dim]"
        )
    return f"[bold cyan]{name}[/bold cyan]\n\n[dim]{ds_path}[/dim]"


def _select_output_format(is_pose: bool = False) -> str | None:
    clear_screen()
    print_stylized_header("Select Output Format")
    options = ["YOLO Detection", "YOLO Segmentation", "COCO"]
    descriptions = {
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
    }
    if is_pose:
        options.insert(2, "YOLO Pose")
        descriptions["YOLO Pose"] = (
            "[bold cyan]YOLO Pose Format[/bold cyan]\n\n"
            "Output structure:\n"
            "  train/images/  train/labels/  data.yaml\n\n"
            "Label format: [dim]class_id cx cy w h kpt_x kpt_y [v] …[/dim] (normalized)\n\n"
            "Preserves kpt_shape. Flip transforms are skipped for keypoint safety.\n\n"
            "[dim]Only available because the source dataset has keypoints.[/dim]"
        )
    choice = get_user_choice(
        options,
        allow_back=True,
        title="Output Format",
        text="Select the annotation format for the augmented output dataset:",
        descriptions=descriptions,
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
    stats_holder: list[AugmentationStats] = []
    error_holder: list[Exception] = []
    progress_state: dict = {"done": 0, "total": 0, "current": ""}
    done_event = threading.Event()

    def _worker() -> None:
        try:
            def callback(done: int, total: int, msg: str) -> None:
                progress_state.update({"done": done, "total": total, "current": msg})

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

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]Augmenting[/cyan]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[current]}[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # total=None → indeterminate spinner until the first callback arrives.
        task = progress.add_task("aug", total=None, current="starting…")
        while not done_event.is_set():
            total = progress_state["total"]
            if total > 0:
                progress.update(
                    task,
                    total=total,
                    completed=progress_state["done"],
                    current=progress_state["current"][:48],
                )
            time.sleep(0.1)
        # Final frame: snap the bar to 100% if any work was tracked.
        total = progress_state["total"]
        if total > 0:
            progress.update(task, total=total, completed=total, current="done")

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
        
    if profile_name == "AI Recommendation":
        from src.utils.ai_client import run_ai_augmentation_flow
        profile_name = run_ai_augmentation_flow(str(dataset_path))
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
    try:
        from src.augmentation.engine import detect_annotation_format
        is_pose_source = detect_annotation_format(dataset_path) == "yolo_pose"
    except Exception:
        is_pose_source = False
    output_format = _select_output_format(is_pose=is_pose_source)
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
        source_count: int | str = len(collect_all_images(dataset_path))
    except Exception:
        source_count = "unknown"

    # Nothing to augment — bail out before the confirm/run steps.
    if source_count == 0:
        console.print(Panel(
            "[bold yellow]No images found in this dataset — nothing to augment.[/bold yellow]\n\n"
            "Check that the dataset contains an images/ folder with .jpg/.png files.",
            border_style="yellow", padding=(1, 2),
        ))
        input("\nPress Enter to continue...")
        return

    est_splits = "unknown"
    if isinstance(source_count, int):
        aug_count = source_count * profile.multiplier
        orig_count = source_count if profile.include_originals else 0
        est_total = aug_count + orig_count
        estimated: str = f"~{est_total:,}"
        # Mirror the engine's split math (remainder → train) for a per-split estimate.
        est_test = int(est_total * split_config.test_ratio)
        est_val = int(est_total * split_config.val_ratio)
        est_train = est_total - est_val - est_test
        est_splits = f"{est_train:,} / {est_val:,} / {est_test:,}"
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
        "Profile Transforms":     f"{active_transforms} (fixed per profile, applies to every image)",
        "Multiplier":             f"×{profile.multiplier}",
        "Include Originals":      str(profile.include_originals),
        "Output Format":          output_format,
        "Split (train/val/test)": (
            f"{split_config.train_ratio:.0%} / "
            f"{split_config.val_ratio:.0%} / "
            f"{split_config.test_ratio:.0%}"
        ),
        "Estimated Output":       estimated,
        "Est. Train / Val / Test": est_splits,
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

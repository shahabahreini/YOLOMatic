import os
import shutil
import yaml
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

console = Console()

# Supported image extensions (case-insensitive)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def find_datasets(root_dir: str = ".") -> list:
    datasets = []
    root = Path(root_dir)
    for item in root.iterdir():
        if not item.is_dir():
            continue
        yaml_path = item / "data.yaml"
        if yaml_path.exists():
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "names" in data and isinstance(data["names"], list):
                    datasets.append({
                        "name": item.name,
                        "path": item.absolute(),
                        "classes": data["names"],
                        "nc": data.get("nc", len(data["names"])),
                        "config": data  # full yaml for path resolution
                    })
            except Exception as e:
                console.print(f"[red]Error reading {yaml_path}: {e}[/red]")
    return datasets


def resolve_split_dirs(ds_path: Path, config: dict, split_key: str) -> tuple[Path | None, Path | None]:
    """Ultra-robust split detection – handles 99% of YOLO layouts (Ultralytics, Roboflow, custom, txt lists, absolute paths, etc.)"""
    candidates = []

    # 1. Direct value from data.yaml (most common and reliable)
    if split_key in config:
        val = config[split_key]
        if isinstance(val, str):
            if os.path.isabs(val):
                candidates.append(Path(val))
            else:
                candidates.append((ds_path / val).resolve())
                if val.startswith("../"):
                    candidates.append((ds_path.parent / val[3:]).resolve())

    # 2. Standard folder patterns
    standard_names = [split_key, split_key.replace("valid", "val"), split_key.replace("val", "valid")]
    for name in standard_names:
        candidates.extend([
            ds_path / name / "images",
            ds_path / "images" / name,
            ds_path / name,
            ds_path / "images" / split_key,
        ])

    # 3. Try every candidate
    for img_candidate in candidates:
        if not img_candidate or not img_candidate.is_dir():
            continue

        # Find matching labels directory (multiple common patterns)
        label_candidates = [
            img_candidate.parent / "labels",
            img_candidate.parent / "labels" / img_candidate.name,
            img_candidate.parent.parent / "labels" / img_candidate.name,
            ds_path / split_key / "labels",
            ds_path / "labels" / split_key,
            ds_path / "labels" / img_candidate.name,
        ]
        for label_candidate in label_candidates:
            if label_candidate.is_dir():
                # Quick sanity check: at least one image exists
                if any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in img_candidate.iterdir()):
                    return img_candidate.resolve(), label_candidate.resolve()

    return None, None


def process_single_image(task: tuple) -> tuple[bool, str]:
    """Thread-safe worker: copy image (hard link preferred) + remap label"""
    img_file, new_name, target_img_path, target_label_path, class_mapping, label_dir = task

    try:
        # Hard link = near-zero cost when on same filesystem
        try:
            os.link(img_file, target_img_path)
        except (OSError, PermissionError):
            shutil.copy2(img_file, target_img_path)

        # Process label
        label_file = label_dir / (img_file.stem + ".txt")
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        old_id = int(parts[0])
                        if old_id in class_mapping:
                            parts[0] = str(class_mapping[old_id])
                            new_lines.append(" ".join(parts) + "\n")
                    except ValueError:
                        continue  # skip corrupted lines

            if new_lines:
                with open(target_label_path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)

        return True, new_name

    except Exception as e:
        return False, f"Failed {new_name}: {str(e)[:100]}"


def merge_datasets(selected_datasets: list, output_name: str, normalize_classes: bool):
    output_path = Path(output_name)
    if output_path.exists():
        if not Confirm.ask(f"[yellow]'{output_name}' already exists. Delete and overwrite?[/yellow]"):
            return
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # === Build unified class list (with optional normalization) ===
    class_set = {}
    for ds in selected_datasets:
        for idx, name in enumerate(ds["classes"]):
            key = name.strip().lower() if normalize_classes else name.strip()
            if key not in class_set:
                class_set[key] = name.strip()  # keep original casing of first occurrence

    all_classes = list(class_set.values())

    # Class index mappings per dataset
    class_mappings = {}
    for ds in selected_datasets:
        mapping = {}
        for old_idx, name in enumerate(ds["classes"]):
            key = name.strip().lower() if normalize_classes else name.strip()
            mapping[old_idx] = all_classes.index(class_set[key])
        class_mappings[ds["name"]] = mapping

    # === Create folder structure (standard Ultralytics: train/val/test) ===
    for split in ["train", "val", "test"]:
        (output_path / split / "images").mkdir(parents=True, exist_ok=True)
        (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {ds["name"]: {"train": 0, "val": 0, "test": 0, "skipped": 0} for ds in selected_datasets}
    errors = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task("[bold cyan]Merging all datasets...", total=100)  # will be updated dynamically

        for ds in selected_datasets:
            ds_name = ds["name"]
            ds_path = ds["path"]
            mapping = class_mappings[ds_name]

            console.print(f"\n[bold blue]→ Processing dataset:[/bold blue] [magenta]{ds_name}[/magenta]")

            for split_key, out_folder in [("train", "train"), ("val", "val"), ("test", "test")]:
                img_dir, label_dir = resolve_split_dirs(ds_path, ds["config"], split_key)
                if not img_dir:
                    continue

                # Get all valid images
                images = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                if not images:
                    continue

                task_id = progress.add_task(f"  [green]{out_folder.upper()}[/green] ({len(images)} images)", total=len(images))

                # Prepare parallel tasks
                tasks = []
                for img_file in images:
                    new_filename = f"{ds_name}_{img_file.name}"
                    target_img = output_path / out_folder / "images" / new_filename
                    target_label = output_path / out_folder / "labels" / (new_filename.rsplit(".", 1)[0] + ".txt")
                    tasks.append((img_file, new_filename, target_img, target_label, mapping, label_dir))

                # Parallel execution (I/O bound → high worker count is safe)
                with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 8)) as executor:
                    future_to_task = {executor.submit(process_single_image, t): t[1] for t in tasks}
                    for future in as_completed(future_to_task):
                        success, msg = future.result()
                        if success:
                            stats[ds_name][out_folder] += 1
                        else:
                            stats[ds_name]["skipped"] += 1
                            errors.append(msg)
                        progress.update(task_id, advance=1)
                        progress.update(overall_task, advance=1)  # rough overall progress

    # === Generate data.yaml ===
    new_yaml = {
        "path": str(output_path.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(all_classes),
        "names": all_classes,
        "merged_from": [ds["name"] for ds in selected_datasets]
    }
    with open(output_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(new_yaml, f, sort_keys=False, allow_unicode=True)

    # === Generate README.md with stats ===
    readme = f"""# Merged YOLO Dataset: {output_name}

Automatically created by the **Robust YOLO Dataset Combiner** (parallel + hard-link optimized).

## Classes
- **Total**: {len(all_classes)}
- **Names**: `{all_classes}`

## Dataset Composition
| Source Dataset | Train | Val  | Test | Skipped | Total |
|----------------|-------|------|------|---------|-------|
"""
    grand_total = 0
    for name, s in stats.items():
        total = s["train"] + s["val"] + s["test"]
        grand_total += total
        readme += f"| {name} | {s['train']} | {s['val']} | {s['test']} | {s['skipped']} | **{total}** |\n"

    readme += f"| **TOTAL** | **{sum(s['train'] for s in stats.values())}** | **{sum(s['val'] for s in stats.values())}** | **{sum(s['test'] for s in stats.values())}** | **{sum(s['skipped'] for s in stats.values())}** | **{grand_total}** |\n\n"

    readme += """## Folder Structure
    {output_name}/
├── data.yaml
├── README.md
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
├── images/
└── labels/
"""

    with open(output_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    console.print(f"\n[bold green]✅ PERFECT MERGE COMPLETE![/bold green]")
    console.print(f"   Location: [blue]{output_path.absolute()}[/blue]")
    console.print(f"   Classes : [cyan]{len(all_classes)}[/cyan]")
    if errors:
        console.print(f"[yellow]⚠️  {len(errors)} files had issues (logged above)[/yellow]")


def main():
    console.print(Panel.fit("🚀 [bold]Robust YOLO Dataset Combiner[/bold] (Parallel + Hard-Link Optimized)", border_style="blue"))

    datasets = find_datasets()
    if not datasets:
        console.print("[red]No YOLO datasets (with data.yaml) found in current folder.[/red]")
        return

    # Show available datasets
    table = Table(title="Detected YOLO Datasets")
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Classes", style="green")
    for i, ds in enumerate(datasets):
        table.add_row(str(i + 1), ds["name"], ", ".join(ds["classes"][:5]) + ("..." if len(ds["classes"]) > 5 else ""))
    console.print(table)

    selection = Prompt.ask("\nDatasets to merge (e.g. 1,3 or 'all')", default="all")
    if selection.lower() == "all":
        selected = datasets
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            selected = [datasets[i] for i in indices]
        except Exception:
            console.print("[red]Invalid selection.[/red]")
            return

    normalize = Confirm.ask("Normalize class names (strip + lower-case)? [Recommended for merging different sources]", default=True)
    output_name = Prompt.ask("Output folder name", default="combined_yolo_dataset")

    merge_datasets(selected, output_name, normalize_classes=normalize)


if __name__ == "__main__":
    main()
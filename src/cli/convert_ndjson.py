import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.utils.cli import (
    NAV_BACK,
    clear_screen,
    console,
    get_parameter_value_input,
    get_user_choice,
    ParameterDefinition,
    print_stylized_header,
)


def _download_image(url: str, output_path: Path) -> bool:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return True
    except Exception:
        return False


def convert_ndjson_to_format(ndjson_path: Path, output_format: str, output_dir: Path) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if output_format.startswith("YOLO"):
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
    
    lines = ndjson_path.read_text("utf-8").splitlines()
    data_rows = [json.loads(line) for line in lines if line.strip()]
    
    classes: dict[str, int] = {}
    def get_class_id(name: str) -> int:
        if name not in classes:
            classes[name] = len(classes)
        return classes[name]
    
    coco_images = []
    coco_annotations = []
    annotation_id = 1
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=False,
    ) as progress:
        task_id = progress.add_task(f"Processing {len(data_rows)} rows...", total=len(data_rows))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {}
            for row in data_rows:
                # Basic Labelbox NDJSON structure parsing
                data_row = row.get("data_row", {})
                row_data = data_row.get("row_data")
                if not row_data:
                    continue
                    
                filename = data_row.get("global_key") or Path(urlparse(row_data).path).name
                if not filename:
                    filename = f"{data_row.get('id', 'unknown')}.jpg"
                
                image_path = images_dir / filename
                futures[executor.submit(_download_image, row_data, image_path)] = (row, filename, image_path)
            
            for future in as_completed(futures):
                row, filename, image_path = futures[future]
                if not future.result():
                    progress.advance(task_id)
                    continue
                
                try:
                    from PIL import Image
                    with Image.open(image_path) as img:
                        img_w, img_h = img.size
                except Exception:
                    progress.advance(task_id)
                    continue
                    
                labels = row.get("projects", {}).values()
                all_objects = []
                for project in labels:
                    for label in project.get("labels", []):
                        all_objects.extend(label.get("annotations", {}).get("objects", []))
                        
                if output_format.startswith("YOLO"):
                    yolo_lines = []
                    for obj in all_objects:
                        name = obj.get("name")
                        class_id = get_class_id(name)
                        if "polygon" in obj:
                            points = obj["polygon"]
                            flat_points = []
                            for p in points:
                                flat_points.extend([p["x"] / img_w, p["y"] / img_h])
                            yolo_lines.append(f"{class_id} " + " ".join(f"{x:.6f}" for x in flat_points))
                        elif "bounding_box" in obj:
                            bbox = obj["bounding_box"]
                            top, left = bbox["top"], bbox["left"]
                            h, w = bbox["height"], bbox["width"]
                            xc = (left + w / 2) / img_w
                            yc = (top + h / 2) / img_h
                            nw = w / img_w
                            nh = h / img_h
                            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    
                    if yolo_lines:
                        (labels_dir / f"{Path(filename).stem}.txt").write_text("\n".join(yolo_lines))
                
                elif output_format == "COCO":
                    image_id = len(coco_images) + 1
                    coco_images.append({
                        "id": image_id,
                        "file_name": filename,
                        "width": img_w,
                        "height": img_h
                    })
                    for obj in all_objects:
                        name = obj.get("name")
                        class_id = get_class_id(name)
                        if "polygon" in obj:
                            points = obj["polygon"]
                            flat_points = []
                            for p in points:
                                flat_points.extend([p["x"], p["y"]])
                            # calculate bbox from polygon
                            xs = [p["x"] for p in points]
                            ys = [p["y"] for p in points]
                            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                            coco_annotations.append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id,
                                "segmentation": [flat_points],
                                "bbox": bbox,
                                "area": bbox[2] * bbox[3],
                                "iscrowd": 0
                            })
                            annotation_id += 1
                        elif "bounding_box" in obj:
                            bbox_data = obj["bounding_box"]
                            top, left = bbox_data["top"], bbox_data["left"]
                            h, w = bbox_data["height"], bbox_data["width"]
                            coco_annotations.append({
                                "id": annotation_id,
                                "image_id": image_id,
                                "category_id": class_id,
                                "segmentation": [],
                                "bbox": [left, top, w, h],
                                "area": w * h,
                                "iscrowd": 0
                            })
                            annotation_id += 1

                progress.advance(task_id)

    if output_format.startswith("YOLO"):
        yaml_content = f"path: {output_dir.absolute()}\nimages: images\nlabels: labels\n\nnames:\n"
        # sort classes by id
        sorted_classes = sorted(classes.keys(), key=lambda k: classes[k])
        for idx, name in enumerate(sorted_classes):
            yaml_content += f"  {idx}: {name}\n"
        (output_dir / "data.yaml").write_text(yaml_content)
    elif output_format == "COCO":
        coco_categories = [{"id": cid, "name": name} for name, cid in classes.items()]
        coco_data = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories
        }
        (output_dir / "annotations.json").write_text(json.dumps(coco_data, indent=2))

    console.print("[bold green]Conversion completed![/bold green]")
    console.print(f"Output saved to: [cyan]{output_dir.absolute()}[/cyan]")


def main() -> None:
    while True:
        clear_screen()
        print_stylized_header("Convert Dataset Format")
        
        # Get NDJSON path
        ndjson_path_str = get_parameter_value_input(
            ParameterDefinition("ndjson", "Path to Labelbox .ndjson file", str, ""),
            "",
            title="Select Input File"
        )
        if not ndjson_path_str:
            return
            
        ndjson_path = Path(ndjson_path_str)
        if not ndjson_path.exists() or not ndjson_path.is_file():
            console.print(f"[bold red]File not found:[/bold red] {ndjson_path_str}")
            input("Press Enter to try again...")
            continue
            
        format_choice = get_user_choice(
            ["YOLO", "COCO", NAV_BACK],
            title="Select Output Format",
            text="Choose the desired format for the dataset:"
        )
        if format_choice == NAV_BACK:
            return
            
        output_dir_str = get_parameter_value_input(
            ParameterDefinition("output_dir", "Output directory path", str, "converted_dataset"),
            "converted_dataset",
            title="Select Output Directory"
        )
        if not output_dir_str:
            return
            
        convert_ndjson_to_format(ndjson_path, format_choice, Path(output_dir_str))
        input("\nPress Enter to return to main menu...")
        return

if __name__ == "__main__":
    main()

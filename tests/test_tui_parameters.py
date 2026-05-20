import unittest

from src.utils.tui import (
    MenuRenderer,
    MultiSelectRenderer,
    ParameterDefinition,
    convert_and_validate_parameter_value,
    expected_error_panel,
    format_path,
    render_hints,
    parse_parameter_value,
    resolve_finish_option,
    shorten_middle,
)


class TUIParameterValidationTest(unittest.TestCase):
    def test_bool_values_parse_from_common_tui_strings(self) -> None:
        param = ParameterDefinition(
            name="amp",
            category="advanced",
            default=True,
            value_type="bool",
            description="Speed boost",
            help_text="Enable mixed precision.",
        )

        self.assertIs(parse_parameter_value(param, "False"), False)
        self.assertIs(parse_parameter_value(param, "true"), True)
        self.assertIs(parse_parameter_value(param, "0"), False)
        self.assertIs(parse_parameter_value(param, "yes"), True)

    def test_allowed_values_are_validated_after_conversion(self) -> None:
        param = ParameterDefinition(
            name="optimizer",
            category="optimizer",
            default="auto",
            value_type="str",
            description="Learning algorithm",
            help_text="Select the optimizer.",
            allowed_values=["auto", "SGD", "AdamW"],
        )

        is_valid, value, error = convert_and_validate_parameter_value(
            param,
            "AdamW",
        )

        self.assertTrue(is_valid)
        self.assertEqual(value, "AdamW")
        self.assertIsNone(error)

        is_valid, value, error = convert_and_validate_parameter_value(
            param,
            "DefinitelyNotAnOptimizer",
        )

        self.assertFalse(is_valid)
        self.assertEqual(value, "DefinitelyNotAnOptimizer")
        self.assertIn("Choose one of", str(error))

    def test_numeric_range_validation_reports_clear_errors(self) -> None:
        param = ParameterDefinition(
            name="imgsz",
            category="core",
            default=640,
            value_type="int",
            description="Input resolution",
            help_text="Set image size.",
            min_value=32,
            max_value=2048,
        )

        is_valid, value, error = convert_and_validate_parameter_value(param, "16")

        self.assertFalse(is_valid)
        self.assertEqual(value, 16)
        self.assertEqual(error, "Value must be >= 32.")

    def test_long_menu_labels_are_shortened_in_the_middle(self) -> None:
        label = "very_long_config_name_with_dataset_and_timestamp_20260505_152200.yaml"

        shortened = shorten_middle(label, max_chars=32)

        self.assertLessEqual(len(shortened), 32)
        self.assertTrue(shortened.startswith("very_long"))
        self.assertTrue(shortened.endswith(".yaml"))
        self.assertIn("...", shortened)

    def test_format_path_preserves_important_suffix(self) -> None:
        path = "/tmp/some/deeply/nested/project/runs/detect/train/weights/best.pt"

        formatted = format_path(path, max_chars=34)

        self.assertLessEqual(len(formatted), 34)
        self.assertTrue(formatted.endswith("best.pt"))
        self.assertIn("...", formatted)

    def test_standard_footer_hints_render_expected_menu_keys(self) -> None:
        hints = render_hints("menu")

        self.assertIn("Move", hints.plain)
        self.assertIn("Select", hints.plain)
        self.assertIn("Back", hints.plain)
        self.assertIn("Quit", hints.plain)

    def test_finish_footer_hints_render_continue_key(self) -> None:
        hints = render_hints("menu_finish")

        self.assertIn("Continue", hints.plain)
        self.assertIn("F", hints.plain)

    def test_finish_shortcut_resolves_only_explicit_forward_action(self) -> None:
        self.assertEqual(
            resolve_finish_option(
                ["model_a.pt", "Confirm Selection", "Back"],
                {"Confirm Selection"},
            ),
            "Confirm Selection",
        )
        self.assertEqual(resolve_finish_option(["Start Benchmark", "Cancel"]), "Start Benchmark")
        self.assertIsNone(resolve_finish_option(["Train", "Predict", "Benchmark"]))

    def test_menu_renderer_includes_breadcrumbs_status_description_and_tip(self) -> None:
        from rich.console import Console

        console = Console(record=True, width=80, height=24, color_system=None)
        renderer = MenuRenderer(
            options=["Train", "Predict"],
            current_selection=0,
            title="YOLOmatic",
            instruction="Choose a workflow.",
            descriptions={"Train": "Create or run a training configuration."},
            breadcrumbs=["YOLOmatic", "Main Menu"],
            tip="Recommended: start with Train when configuring a new dataset.",
            status_fields={
                "Dataset": "/very/long/path/to/datasets/project.v1i.yolov11/data.yaml",
            },
        )

        console.print(renderer)
        output = console.export_text()

        self.assertIn("YOLOmatic", output)
        self.assertIn("Main Menu", output)
        self.assertIn("Current Choice", output)
        self.assertIn("Create or run a training configuration.", output)
        self.assertIn("Recommended: start with Train", output)
        self.assertIn("data.yaml", output)

    def test_menu_renderer_windows_long_option_lists(self) -> None:
        from rich.console import Console

        options = [f"Option {i:02d}" for i in range(40)]
        console = Console(record=True, width=80, height=20, color_system=None)
        renderer = MenuRenderer(
            options=options,
            current_selection=30,
            title="Long Menu",
            instruction="Choose an item.",
        )

        console.print(renderer)
        output = console.export_text()

        self.assertIn("Option 30", output)
        self.assertIn("more above", output)
        self.assertIn("more below", output)
        self.assertNotIn("Option 00", output)

    def test_menu_renderer_does_not_leak_markup_in_checked_labels(self) -> None:
        from rich.console import Console
        from unittest.mock import patch

        with patch("src.utils.tui.TUI_TERM") as mock_term:
            mock_term.width = 80
            mock_term.height = 24
            console = Console(record=True, width=80, height=20, color_system=None)
            renderer = MenuRenderer(
                options=["✓ runs/detect/train/weights/best.pt", "  Confirm Selection"],
                current_selection=1,
                title="Weights",
                instruction="Choose weights.",
            )

            console.print(renderer)
            output = console.export_text()

            self.assertIn("✓", output)
            self.assertIn("runs/det", output)
            self.assertIn("est.pt", output)
            self.assertNotIn("[/bold green]", output)
            self.assertNotIn("[bold green]", output)

    def test_menu_renderer_shows_finish_shortcut_when_available(self) -> None:
        from rich.console import Console

        console = Console(record=True, width=90, height=22, color_system=None)
        renderer = MenuRenderer(
            options=["weight_a.pt", "Confirm Selection", "Back"],
            current_selection=0,
            title="Weights",
            instruction="Choose weights.",
            finish_option="Confirm Selection",
        )

        console.print(renderer)
        output = console.export_text()

        self.assertIn("F Continue", output)
        self.assertIn("Confirm Selection", output)

    def test_multiselect_renderer_shortens_long_parameter_names(self) -> None:
        from rich.console import Console

        long_name = "RandomBrightnessContrast.brightness_limit_low"
        param = ParameterDefinition(
            name=long_name,
            category="Color",
            default=0.1,
            value_type="float",
            description="Brightness lower bound",
            help_text="Controls the lower brightness range.",
        )
        console = Console(record=True, width=80, height=24, color_system=None)
        renderer = MultiSelectRenderer(
            parameters=[param],
            selected={long_name},
            values={long_name: 0.1},
            current_index=0,
            title="Augmentation Profile",
            instruction="Edit values.",
        )

        console.print(renderer)
        output = console.export_text()

        self.assertIn("RandomBrightness", output)
        self.assertIn("...", output)
        self.assertIn("htness_limit_low", output)

    def test_parameter_editor_render_shows_validation_and_allowed_values(self) -> None:
        from rich.console import Console

        param = ParameterDefinition(
            name="optimizer",
            category="optimizer",
            default="auto",
            value_type="str",
            description="Learning algorithm",
            help_text="Choose the optimizer for training.",
            allowed_values=["auto", "SGD", "AdamW"],
        )
        console = Console(record=True, width=80, height=24, color_system=None)
        renderer = MultiSelectRenderer(
            parameters=[param],
            selected={"optimizer"},
            values={"optimizer": "AdamW"},
            current_index=0,
            title="Custom Parameters",
            instruction="Edit selected parameters.",
            focus="input",
            input_buffer="bad",
            validation_error="Choose one of: 'auto', 'SGD', 'AdamW'.",
        )

        console.print(renderer)
        output = console.export_text()

        self.assertIn("Custom Parameters", output)
        self.assertIn("optimizer", output)
        self.assertIn("Allowed values", output)
        self.assertIn("Validation", output)
        self.assertIn("AdamW", output)

    def test_expected_error_panel_has_next_step_without_traceback(self) -> None:
        from rich.console import Console

        console = Console(record=True, width=80, color_system=None)
        console.print(
            expected_error_panel(
                "Path not found: missing/images",
                next_step="Choose an existing image file or folder.",
            )
        )
        output = console.export_text()

        self.assertIn("Path not found", output)
        self.assertIn("Choose an existing image file or folder.", output)
        self.assertNotIn("Traceback", output)

    def test_clone_helpers_collect_only_known_tunable_sections(self) -> None:
        from pathlib import Path

        from src.cli.run import (
            clone_config_filename,
            collect_known_config_sections,
            extract_regular_yolo_model_choice,
        )

        source_config = {
            "settings": {
                "model_type": "yolo11n-seg",
                "dataset": "old-dataset",
            },
            "training": {
                "epochs": 100,
                "optimizer": "AdamW",
                "data": "stale/path/data.yaml",
            },
            "export": {
                "format": "onnx",
                "simplify": True,
            },
            "prediction": {
                "save_txt": True,
            },
        }

        self.assertEqual(
            extract_regular_yolo_model_choice(source_config),
            "yolo11n-seg",
        )
        self.assertEqual(
            collect_known_config_sections(source_config),
            {
                "training": {"epochs": 100, "optimizer": "AdamW"},
                "export": {"format": "onnx", "simplify": True},
                "prediction": {"save_txt": True},
            },
        )
        self.assertTrue(
            clone_config_filename(
                Path("configs/source_with_a_very_long_name.yaml"),
                "QGIS Vegetation.v10i.yolo26",
            ).startswith("clone_source_with_a_very_long_name"),
        )

    def test_detectron2_summary_paths_use_dataset_splits_schema(self) -> None:
        from src.cli.run import dataset_path_rows_for_config

        rows = dataset_path_rows_for_config(
            "Mask R-CNN R50-FPN 3x",
            {
                "dataset": {
                    "splits": {
                        "train": {
                            "images_path": "datasets/cache/train/images",
                            "annotations_path": "datasets/cache/train_annotations.coco.json",
                        },
                        "val": {
                            "images_path": "datasets/cache/val/images",
                            "annotations_path": "datasets/cache/valid_annotations.coco.json",
                        },
                        "test": {
                            "images_path": "datasets/cache/test/images",
                            "annotations_path": "datasets/cache/test_annotations.coco.json",
                        },
                    }
                }
            },
        )

        flattened = [value for row in rows for value in row]
        self.assertIn("Train Images", flattened)
        self.assertIn("datasets/cache/train/images", flattened)
        self.assertIn("datasets/cache/train_annotations.coco.json", flattened)
        self.assertNotIn("N/A", flattened)

    def test_rfdetr_seg_update_config_extracts_dataset_before_mismatch_check(self) -> None:
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from src.cli import run

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "datasets" / "roboflow-yolo-seg"
            (dataset / "train" / "images").mkdir(parents=True)
            (dataset / "train" / "labels").mkdir(parents=True)
            (dataset / "valid" / "images").mkdir(parents=True)
            (dataset / "test" / "images").mkdir(parents=True)
            (dataset / "data.yaml").write_text(
                "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: [item]\n",
                encoding="utf-8",
            )
            (dataset / "train" / "labels" / "polygon.txt").write_text(
                "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",
                encoding="utf-8",
            )

            with patch.object(run, "display_configuration_summary"), patch.object(
                run,
                "display_paths_info",
            ), patch.object(run.console, "print"):
                self.assertTrue(run.update_config("RF-DETR-Seg-Small", str(dataset)))

            config_files = sorted((Path("configs")).glob("RF-DETR-Seg-Small_roboflow-yolo-seg_*.yaml"))
            for config_file in config_files:
                config_file.unlink()

    def test_fully_customized_catalog_covers_ultralytics_tunable_args(self) -> None:
        from src.cli.run import YOLO_TRAINING_PARAMETERS

        generated_or_read_only = {
            "data",
            "mode",
            "model",
            "save_dir",
            "task",
        }
        expected_args = {
            "agnostic_nms",
            "amp",
            "angle",
            "augment",
            "auto_augment",
            "batch",
            "bgr",
            "box",
            "cache",
            "cfg",
            "classes",
            "close_mosaic",
            "cls",
            "cls_pw",
            "compile",
            "conf",
            "copy_paste",
            "copy_paste_mode",
            "cos_lr",
            "cutmix",
            "degrees",
            "deterministic",
            "device",
            "dfl",
            "dnn",
            "dropout",
            "dynamic",
            "embed",
            "end2end",
            "epochs",
            "erasing",
            "exist_ok",
            "fliplr",
            "flipud",
            "format",
            "fraction",
            "freeze",
            "half",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "imgsz",
            "int8",
            "iou",
            "keras",
            "kobj",
            "line_width",
            "lr0",
            "lrf",
            "mask_ratio",
            "max_det",
            "mixup",
            "momentum",
            "mosaic",
            "multi_scale",
            "name",
            "nbs",
            "nms",
            "opset",
            "optimize",
            "optimizer",
            "overlap_mask",
            "patience",
            "perspective",
            "plots",
            "pose",
            "pretrained",
            "profile",
            "project",
            "rect",
            "resume",
            "retina_masks",
            "rle",
            "save",
            "save_conf",
            "save_crop",
            "save_frames",
            "save_json",
            "save_period",
            "save_txt",
            "scale",
            "seed",
            "shear",
            "show",
            "show_boxes",
            "show_conf",
            "show_labels",
            "simplify",
            "single_cls",
            "source",
            "split",
            "stream_buffer",
            "tracker",
            "translate",
            "val",
            "verbose",
            "vid_stride",
            "visualize",
            "warmup_bias_lr",
            "warmup_epochs",
            "warmup_momentum",
            "weight_decay",
            "workers",
            "workspace",
        }
        expected_args -= generated_or_read_only

        catalog_names = [parameter.name for parameter in YOLO_TRAINING_PARAMETERS]
        self.assertEqual(len(catalog_names), len(set(catalog_names)))
        self.assertTrue(expected_args.issubset(set(catalog_names)))

        for parameter in YOLO_TRAINING_PARAMETERS:
            self.assertTrue(
                parameter.allowed_values
                or parameter.min_value is not None
                or parameter.max_value is not None
                or parameter.value_type in {
                    "bool",
                    "bool_or_str",
                    "int_list",
                    "optional_bool",
                    "optional_float",
                    "optional_int",
                    "optional_str",
                },
                f"{parameter.name} needs allowed values, a range, or an optional/bool type",
            )


if __name__ == "__main__":
    unittest.main()

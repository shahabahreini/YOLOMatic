import unittest

from src.utils.tui import (
    ParameterDefinition,
    convert_and_validate_parameter_value,
    parse_parameter_value,
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

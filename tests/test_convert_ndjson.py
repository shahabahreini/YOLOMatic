import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil
import tempfile

from src.cli import convert_ndjson
from src.cli.convert_ndjson import convert_ndjson_to_format, _discover_ndjson_files


class TestConvertNDJSON(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.ndjson_file = self.tmp_dir / "labels.ndjson"
        self.row = {
            "data_row": {
                "global_key": "test.jpg",
                "row_data": "https://example.com/test.jpg"
            },
            "projects": {
                "proj1": {
                    "labels": [
                        {
                            "annotations": {
                                "objects": [
                                    {
                                        "name": "cat",
                                        "bounding_box": {"top": 10, "left": 20, "height": 30, "width": 40}
                                    },
                                    {
                                        "name": "dog",
                                        "polygon": [{"x": 10, "y": 10}, {"x": 20, "y": 10}, {"x": 20, "y": 20}]
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        }
        self.ndjson_file.write_text(json.dumps(self.row))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_convert_ndjson_to_coco(self):
        output_dir = self.tmp_dir / "output_coco"
        
        with patch("requests.get") as mock_get, \
             patch("PIL.Image.open") as mock_image_open:
            
            # Mock requests.get
            mock_response = MagicMock()
            mock_response.content = b"fake image content"
            mock_get.return_value = mock_response
            
            # Mock PIL.Image.open
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_img.__enter__.return_value = mock_img
            mock_image_open.return_value = mock_img
            
            convert_ndjson_to_format(self.ndjson_file, "COCO", output_dir)
            
            self.assertTrue((output_dir / "annotations.json").exists())
            self.assertTrue((output_dir / "conversion_manifest.json").exists())
            self.assertTrue((output_dir / "images" / "test.jpg").exists())
            
            with open(output_dir / "annotations.json") as f:
                data = json.load(f)
                self.assertEqual(len(data["images"]), 1)
                self.assertEqual(len(data["annotations"]), 2)
                self.assertEqual(data["categories"][0]["id"], 1)
                self.assertEqual(data["categories"][0]["name"], "cat")
                self.assertEqual(data["categories"][1]["id"], 2)
                self.assertEqual(data["categories"][1]["name"], "dog")

    def test_convert_ndjson_to_yolo(self):
        output_dir = self.tmp_dir / "output_yolo"
        
        with patch("requests.get") as mock_get, \
             patch("PIL.Image.open") as mock_image_open:
            
            # Mock requests.get
            mock_response = MagicMock()
            mock_response.content = b"fake image content"
            mock_get.return_value = mock_response
            
            # Mock PIL.Image.open
            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_img.__enter__.return_value = mock_img
            mock_image_open.return_value = mock_img
            
            stats = convert_ndjson_to_format(self.ndjson_file, "YOLO Detection", output_dir)
            
            self.assertTrue((output_dir / "data.yaml").exists())
            self.assertTrue((output_dir / "images" / "test.jpg").exists())
            self.assertTrue((output_dir / "labels" / "test.txt").exists())
            self.assertEqual(stats.converted_images, 1)
            self.assertEqual(stats.total_annotations, 2)
            
            labels_text = (output_dir / "labels" / "test.txt").read_text()
            lines = labels_text.strip().split("\n")
            self.assertEqual(len(lines), 2)
            
            # Line 0: cat (bbox)
            self.assertTrue(lines[0].startswith("0 0.400000 0.250000 0.400000 0.300000"))
            
            # Line 1: dog polygon converted to a tight detection box
            self.assertTrue(lines[1].startswith("1 0.150000 0.150000 0.100000 0.100000"))

    def test_convert_ndjson_to_yolo_segmentation(self):
        output_dir = self.tmp_dir / "output_yolo_seg"

        with patch("requests.get") as mock_get, \
             patch("PIL.Image.open") as mock_image_open:
            mock_response = MagicMock()
            mock_response.content = b"fake image content"
            mock_get.return_value = mock_response

            mock_img = MagicMock()
            mock_img.size = (100, 100)
            mock_img.__enter__.return_value = mock_img
            mock_image_open.return_value = mock_img

            convert_ndjson_to_format(self.ndjson_file, "YOLO Segmentation", output_dir)

            labels_text = (output_dir / "labels" / "test.txt").read_text()
            lines = labels_text.strip().split("\n")
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith("0 0.200000 0.100000 0.600000 0.100000"))
            self.assertTrue(lines[1].startswith("1 0.100000 0.100000 0.200000 0.100000 0.200000 0.200000"))

    def test_discover_ndjson_files_for_conversion(self):
        root_export = self.tmp_dir / "labels.ndjson"
        nested_export = self.tmp_dir / "datasets" / "project" / "export.ndjson"
        hidden_export = self.tmp_dir / "datasets" / ".hidden.ndjson"
        nested_export.parent.mkdir(parents=True, exist_ok=True)
        nested_export.write_text("{}", encoding="utf-8")
        hidden_export.write_text("{}", encoding="utf-8")

        self.assertEqual(_discover_ndjson_files(self.tmp_dir), [nested_export, root_export])

    def test_interactive_main_guided_flow_uses_current_prompt_api(self):
        output_dir = self.tmp_dir / "interactive_output"
        prompt_values = iter([str(self.ndjson_file), str(output_dir), 4])
        choices = iter(["Enter Manual Path", "YOLO Detection", "Start Conversion"])

        def prompt_side_effect(*args, **kwargs):
            self.assertNotIn("title", kwargs)
            return next(prompt_values)

        with patch("src.cli.convert_ndjson.clear_screen"), \
             patch("src.cli.convert_ndjson.print_stylized_header"), \
             patch("src.cli.convert_ndjson.get_parameter_value_input", side_effect=prompt_side_effect), \
             patch("src.cli.convert_ndjson.get_user_choice", side_effect=lambda *args, **kwargs: next(choices)), \
             patch("src.cli.convert_ndjson.convert_ndjson_to_format") as mock_convert, \
             patch("builtins.input", return_value=""):
            convert_ndjson.main()

        self.assertEqual(mock_convert.call_count, 1)
        _source, _format, _output = mock_convert.call_args.args
        self.assertEqual((_source, _format, _output), (self.ndjson_file, "YOLO Detection", output_dir))
        self.assertEqual(mock_convert.call_args.kwargs["max_workers"], 4)

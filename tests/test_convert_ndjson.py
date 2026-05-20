import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import shutil
import tempfile

from src.cli.convert_ndjson import convert_ndjson_to_format


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
            self.assertTrue((output_dir / "images" / "test.jpg").exists())
            
            with open(output_dir / "annotations.json") as f:
                data = json.load(f)
                self.assertEqual(len(data["images"]), 1)
                self.assertEqual(len(data["annotations"]), 2)
                self.assertEqual(data["categories"][0]["name"], "cat")
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
            
            convert_ndjson_to_format(self.ndjson_file, "YOLO", output_dir)
            
            self.assertTrue((output_dir / "data.yaml").exists())
            self.assertTrue((output_dir / "images" / "test.jpg").exists())
            self.assertTrue((output_dir / "labels" / "test.txt").exists())
            
            labels_text = (output_dir / "labels" / "test.txt").read_text()
            lines = labels_text.strip().split("\n")
            self.assertEqual(len(lines), 2)
            
            # Line 0: cat (bbox)
            self.assertTrue(lines[0].startswith("0 0.400000 0.250000 0.400000 0.300000"))
            
            # Line 1: dog (polygon)
            self.assertTrue(lines[1].startswith("1 0.100000 0.100000 0.200000 0.100000 0.200000 0.200000"))

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.convert_ndjson import convert_ndjson_to_format


@pytest.fixture
def mock_labelbox_ndjson(tmp_path: Path) -> Path:
    ndjson_file = tmp_path / "labels.ndjson"
    row = {
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
    ndjson_file.write_text(json.dumps(row))
    return ndjson_file


def test_convert_ndjson_to_coco(tmp_path: Path, mock_labelbox_ndjson: Path):
    output_dir = tmp_path / "output_coco"
    
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
        
        convert_ndjson_to_format(mock_labelbox_ndjson, "COCO", output_dir)
        
        assert (output_dir / "annotations.json").exists()
        assert (output_dir / "images" / "test.jpg").exists()
        
        with open(output_dir / "annotations.json") as f:
            data = json.load(f)
            assert len(data["images"]) == 1
            assert len(data["annotations"]) == 2
            assert data["categories"][0]["name"] == "cat"
            assert data["categories"][1]["name"] == "dog"


def test_convert_ndjson_to_yolo(tmp_path: Path, mock_labelbox_ndjson: Path):
    output_dir = tmp_path / "output_yolo"
    
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
        
        convert_ndjson_to_format(mock_labelbox_ndjson, "YOLO", output_dir)
        
        assert (output_dir / "data.yaml").exists()
        assert (output_dir / "images" / "test.jpg").exists()
        assert (output_dir / "labels" / "test.txt").exists()
        
        labels_text = (output_dir / "labels" / "test.txt").read_text()
        lines = labels_text.strip().split("\n")
        assert len(lines) == 2
        
        # Line 0: cat (bbox)
        # top:10, left:20, h:30, w:40 -> xc=(20+20)/100=0.4, yc=(10+15)/100=0.25, w=0.4, h=0.3
        assert lines[0].startswith("0 0.400000 0.250000 0.400000 0.300000")
        
        # Line 1: dog (polygon)
        assert lines[1].startswith("1 0.100000 0.100000 0.200000 0.100000 0.200000 0.200000")

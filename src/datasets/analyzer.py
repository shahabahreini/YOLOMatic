import os
import logging
from pathlib import Path
from typing import Optional

import yaml

from src.datasets.core import read_yaml_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data_yaml = None
        self.readme_dataset = None
        self.readme_roboflow = None
        self.dataset_info = {}

    def clean_path(self, path: str) -> str:
        """Clean path string by removing spaces and normalizing separators."""
        return str(Path(path.replace(" ", "_")))

    def get_relative_path(self, path: str, base_path: str) -> str:
        """Get relative path that works with the training script."""
        try:
            # Convert to absolute paths first
            abs_path = Path(path).absolute()
            base_path = Path(base_path).absolute()

            # Get relative path
            rel_path = str(Path(os.path.relpath(abs_path, base_path)))

            # Ensure forward slashes
            return rel_path.replace("\\", "/")
        except Exception as e:
            logger.error(f"Error calculating relative path: {e}")
            return path

    def extract_dataset_info(self):
        """Extract dataset information from data.yaml and README files."""
        # Find and read data.yaml
        data_yaml_path = self.dataset_path / "data.yaml"
        if data_yaml_path.exists():
            try:
                self.data_yaml = read_yaml_file(data_yaml_path)
            except Exception as e:
                logger.error(f"Error reading YAML file {data_yaml_path}: {e}")
                self.data_yaml = None
            if self.data_yaml:
                self.dataset_info.update(
                    {
                        "classes": self.data_yaml.get("names", []),
                        "num_classes": self.data_yaml.get("nc", 0),
                        "train_path": self.data_yaml.get("train", ""),
                        "valid_path": self.data_yaml.get("val", ""),
                        "test_path": self.data_yaml.get("test", ""),
                        "dataset_name": self.data_yaml.get(
                            "project", self.dataset_path.name
                        ),
                    }
                )

        # Read README files for additional metadata
        readme_dataset = self.dataset_path / "README.dataset.txt"
        readme_roboflow = self.dataset_path / "README.roboflow.txt"

        if readme_dataset.exists():
            self.readme_dataset = self.read_text_file(str(readme_dataset))
        if readme_roboflow.exists():
            self.readme_roboflow = self.read_text_file(str(readme_roboflow))

    def read_text_file(self, file_path: str) -> Optional[str]:
        """Read text file content."""
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return None

    def analyze(self) -> dict:
        """Extract dataset metadata without generating a training config."""
        logger.info(f"Analyzing dataset in {self.dataset_path}")
        self.extract_dataset_info()

        if not self.dataset_info:
            logger.error("No dataset information could be extracted")
            return {}

        logger.info("Dataset analysis complete:")
        logger.info(f"Classes: {self.dataset_info.get('classes', [])}")
        logger.info(f"Number of classes: {self.dataset_info.get('num_classes', 0)}")
        return self.dataset_info


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a YOLO-format dataset and print extracted metadata"
    )
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    args = parser.parse_args()

    analyzer = DatasetAnalyzer(args.dataset_path)
    print(yaml.safe_dump(analyzer.analyze(), sort_keys=False))


if __name__ == "__main__":
    main()

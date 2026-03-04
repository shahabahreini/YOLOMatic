import os
import yaml
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.data_yaml = None
        self.readme_dataset = None
        self.readme_roboflow = None
        self.dataset_info = {}

    def read_yaml(self, file_path: str) -> Optional[Dict]:
        """Read and parse YAML file."""
        try:
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading YAML file {file_path}: {e}")
            return None

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
            self.data_yaml = self.read_yaml(str(data_yaml_path))
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

    def generate_nas_config(self, output_path: str):
        """Generate YOLO NAS configuration based on dataset analysis."""
        # Read base configuration template
        with open("config_NAS.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Clean dataset name and path
        dataset_name = self.clean_path(
            self.dataset_info.get("dataset_name", self.dataset_path.name)
        )
        base_dir = self.clean_path(str(self.dataset_path))

        # Update configuration
        config["dataset"].update(
            {
                "name": dataset_name,
                "base_dir": base_dir,
                "classes": self.dataset_info["classes"],
                "structure": {
                    "train": {"images": "train/images", "labels": "train/labels"},
                    "valid": {"images": "valid/images", "labels": "valid/labels"},
                    "test": {"images": "test/images", "labels": "test/labels"},
                },
            }
        )

        # Ensure all required sections are present
        if "loss" not in config["model"]:
            config["model"]["loss"] = {
                "type": "PPYoloELoss",
                "use_static_assigner": False,
                "reg_max": 16,
            }

        # Adjust batch size based on number of classes
        num_classes = len(self.dataset_info["classes"])
        if num_classes > 0:
            # Adjust batch size based on number of classes
            recommended_batch_size = min(35, max(8, 32 // num_classes * 8))
            config["training"]["batch_size"] = recommended_batch_size

        # Save configuration with proper formatting
        with open(output_path, "w") as f:
            yaml.dump(
                config, f, default_flow_style=False, sort_keys=False, allow_unicode=True
            )

        logger.info(f"Generated NAS configuration saved to {output_path}")
        return config

    def read_text_file(self, file_path: str) -> Optional[str]:
        """Read text file content."""
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return None

    def analyze_and_generate_config(self, output_path: str):
        """Main method to analyze dataset and generate configuration."""
        logger.info(f"Analyzing dataset in {self.dataset_path}")
        self.extract_dataset_info()

        if not self.dataset_info:
            logger.error("No dataset information could be extracted")
            return False

        logger.info("Dataset analysis complete:")
        logger.info(f"Classes: {self.dataset_info.get('classes', [])}")
        logger.info(f"Number of classes: {self.dataset_info.get('num_classes', 0)}")

        config = self.generate_nas_config(output_path)
        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze dataset and generate YOLO NAS configuration"
    )
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument(
        "--output",
        "-o",
        default="generated_config.yaml",
        help="Output path for generated configuration",
    )
    args = parser.parse_args()

    analyzer = DatasetAnalyzer(args.dataset_path)
    analyzer.analyze_and_generate_config(args.output)


if __name__ == "__main__":
    main()

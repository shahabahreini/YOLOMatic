import unittest
from unittest.mock import MagicMock, patch

import src.cli.run as run
from src.datasets.core import DatasetSummary


class DatasetListCachingTest(unittest.TestCase):
    def setUp(self):
        # Reset the cache before each test
        run._CACHED_DATASETS = None
        run._CACHED_DATASET_DESCRIPTIONS = None

    def tearDown(self):
        run._CACHED_DATASETS = None
        run._CACHED_DATASET_DESCRIPTIONS = None

    @patch("src.cli.run.os.path.exists", return_value=True)
    @patch("src.cli.run.list_dataset_directories")
    @patch("src.cli.run.summarize_dataset")
    @patch("src.cli.run.get_user_choice")
    def test_list_datasets_caches_and_uses_cache(
        self, mock_get_user_choice, mock_summarize_dataset, mock_list_dataset_dirs, mock_exists
    ):
        # Setup mocks
        mock_list_dataset_dirs.return_value = [
            {"name": "test_ds", "path": "datasets/test_ds", "size": "10MB"}
        ]
        
        dummy_summary = MagicMock(spec=DatasetSummary)
        dummy_summary.classes = ["cat"]
        dummy_summary.splits = {}
        dummy_summary.errors = []
        dummy_summary.warnings = []
        dummy_summary.format = "yolo"
        dummy_summary.task = "detect"
        dummy_summary.image_count = 10
        dummy_summary.annotation_count = 50
        dummy_summary.compatibility = {"yolo": "compatible", "detectron2": "compatible"}
        mock_summarize_dataset.return_value = dummy_summary

        mock_get_user_choice.return_value = "test_ds"

        # 1. First call: Should scan directories
        choice = run.list_datasets()
        self.assertEqual(choice, "datasets/test_ds")
        self.assertEqual(mock_list_dataset_dirs.call_count, 1)
        self.assertEqual(mock_summarize_dataset.call_count, 1)
        self.assertIsNotNone(run._CACHED_DATASETS)
        self.assertIsNotNone(run._CACHED_DATASET_DESCRIPTIONS)

        # 2. Second call: Should return cached result without scanning again
        mock_get_user_choice.reset_mock()
        mock_get_user_choice.return_value = "test_ds"

        choice2 = run.list_datasets()
        self.assertEqual(choice2, "datasets/test_ds")
        # Call count remains 1 because it loaded from cache
        self.assertEqual(mock_list_dataset_dirs.call_count, 1)
        self.assertEqual(mock_summarize_dataset.call_count, 1)

    @patch("src.cli.run.os.path.exists", return_value=True)
    @patch("src.cli.run.list_dataset_directories")
    @patch("src.cli.run.summarize_dataset")
    @patch("src.cli.run.get_user_choice")
    def test_list_datasets_refresh_invalidates_cache(
        self, mock_get_user_choice, mock_summarize_dataset, mock_list_dataset_dirs, mock_exists
    ):
        mock_list_dataset_dirs.return_value = [
            {"name": "test_ds", "path": "datasets/test_ds", "size": "10MB"}
        ]
        
        dummy_summary = MagicMock(spec=DatasetSummary)
        dummy_summary.classes = ["cat"]
        dummy_summary.splits = {}
        dummy_summary.errors = []
        dummy_summary.warnings = []
        dummy_summary.format = "yolo"
        dummy_summary.task = "detect"
        dummy_summary.image_count = 10
        dummy_summary.annotation_count = 50
        dummy_summary.compatibility = {"yolo": "compatible", "detectron2": "compatible"}
        mock_summarize_dataset.return_value = dummy_summary

        # We simulate the user selecting "__refresh__" first, then selecting the dataset
        mock_get_user_choice.side_effect = ["__refresh__", "test_ds"]

        # Call list_datasets: it should loop, clear cache, and run a second scan
        choice = run.list_datasets()
        self.assertEqual(choice, "datasets/test_ds")
        # Should be called twice due to the refresh
        self.assertEqual(mock_list_dataset_dirs.call_count, 2)
        self.assertEqual(mock_summarize_dataset.call_count, 2)

    @patch("src.cli.run.clear_screen")
    @patch("builtins.input", return_value="")
    def test_safe_subcommand_invalidates_cache(self, mock_input, mock_clear_screen):
        # Set cache variables
        run._CACHED_DATASETS = [{"name": "dummy"}]
        run._CACHED_DATASET_DESCRIPTIONS = {"dummy": "desc"}

        # Run safe subcommand helper
        dummy_target = MagicMock()
        run._safe_subcommand("Dummy Label", dummy_target, prog="dummy-prog")

        self.assertTrue(dummy_target.called)
        # Should be invalidated in the finally block
        self.assertIsNone(run._CACHED_DATASETS)
        self.assertIsNone(run._CACHED_DATASET_DESCRIPTIONS)


if __name__ == "__main__":
    unittest.main()

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from classes.tracker import Tracker


class TestTracker(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory and file for testing
        self.test_dir = TemporaryDirectory()
        self.output_path = Path(self.test_dir.name) / "metrics.json"
        self.tracker = Tracker(self.output_path)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_track_and_export_data(self):
        # Track some sample metrics
        self.tracker.track("loss", 0.25, epoch=1)
        self.tracker.track("accuracy", 0.9, epoch=1)
        self.tracker.track("loss", 0.22, epoch=2)

        # Export data to JSON
        self.tracker.export_data()

        # Read and verify JSON content
        with self.output_path.open("r") as f:
            data = json.load(f)

        expected = {"epoch": [1, 2], "loss": [0.25, 0.22], "accuracy": [0.9]}

        self.assertEqual(data, expected)

    def test_empty_export(self):
        # Export without tracking anything
        self.tracker.export_data()

        with self.output_path.open("r") as f:
            data = json.load(f)

        self.assertEqual(data, {})  # Should be an empty dict


if __name__ == "__main__":
    unittest.main()

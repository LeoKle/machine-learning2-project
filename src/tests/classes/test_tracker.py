from math import isnan
import math
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
        self.tracker.track("loss", 0.25, epoch=1)
        self.tracker.track("accuracy", 0.9, epoch=1)
        self.tracker.track("loss", 0.22, epoch=2)

        self.tracker.export_data()

        with self.output_path.open("r") as f:
            data = json.load(f)

        self.assertEqual(data["epoch"], [1, 2])
        self.assertEqual(data["loss"][0], 0.25)
        self.assertEqual(data["loss"][1], 0.22)
        self.assertEqual(data["accuracy"][0], 0.9)
        self.assertTrue(data["accuracy"][1] is None or math.isnan(data["accuracy"][1]))

    def test_empty_export(self):
        # Export without tracking anything
        self.tracker.export_data()

        with self.output_path.open("r") as f:
            data = json.load(f)

        self.assertEqual(data, {})  # Should be an empty dict

    def test_partial_metrics_tracking(self):

        self.tracker.track("loss", 0.5, 1)

        # Epoch 2: track accuracy only
        self.tracker.track("accuracy", 0.8, 2)

        # Epoch 3: track both
        self.tracker.track("loss", 0.4, 3)
        self.tracker.track("accuracy", 0.85, 3)

        metrics = self.tracker.get_metrics()

        # All metric lists should have length 3 (one per epoch)
        self.assertEqual(len(metrics["epoch"]), 3)
        self.assertEqual(len(metrics["loss"]), 3)
        self.assertEqual(len(metrics["accuracy"]), 3)

        # Check epoch numbers are correct
        self.assertEqual(metrics["epoch"], [1, 2, 3])

        # Check values
        # Epoch 1: loss = 0.5, accuracy missing -> nan
        self.assertEqual(metrics["loss"][0], 0.5)
        self.assertTrue(isnan(metrics["accuracy"][0]))

        # Epoch 2: accuracy = 0.8, loss missing -> nan
        self.assertEqual(metrics["accuracy"][1], 0.8)
        self.assertTrue(isnan(metrics["loss"][1]))

        # Epoch 3: both tracked
        self.assertEqual(metrics["loss"][2], 0.4)
        self.assertEqual(metrics["accuracy"][2], 0.85)


if __name__ == "__main__":
    unittest.main()

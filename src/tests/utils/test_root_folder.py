import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from utils.root_folder import find_project_root


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory with a 'environment.yml' file
        self.test_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.test_dir.name) / "project"
        self.project_dir.mkdir(parents=True)
        self.environment_file = self.project_dir / "environment.yml"
        self.environment_file.touch()

        # Create subdirectories to simulate subdirectory structure
        self.subdir_1 = self.project_dir / "subdir_1"
        self.subdir_2 = self.subdir_1 / "subdir_2"
        self.subdir_2.mkdir(parents=True)

    def tearDown(self):
        # Cleanup the temporary directory after the test
        self.test_dir.cleanup()

    def test_find_project_root_found(self):
        # Patch resolve to simulate calling the function from subdir_2
        with patch("pathlib.Path.resolve") as mock_resolve:
            mock_resolve.side_effect = [
                self.subdir_2,  # First call: inside subdir_2
                self.subdir_1,  # Second call: move up to subdir_1
                self.project_dir,  # Third call: found project_dir (where 'environment.yml' is)
            ]

            result = find_project_root("environment.yml")
            self.assertEqual(str(result), str(self.project_dir))


if __name__ == "__main__":
    unittest.main()

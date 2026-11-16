import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from main import resolve_output_paths


class MainOutputPathTests(unittest.TestCase):
    def test_default_event_log_uses_a_new_path(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            existing_log = root / "violations_detected.csv"
            existing_log.write_text("keep this\n", encoding="utf-8")

            output_path, log_path = resolve_output_paths(None, root)

            self.assertIsNone(output_path)
            self.assertNotEqual(log_path, existing_log)
            self.assertTrue(log_path.name.startswith("violations_detected_"))
            self.assertEqual(existing_log.read_text(encoding="utf-8"), "keep this\n")

            log_path.touch()
            _, next_log_path = resolve_output_paths(None, root)
            self.assertNotEqual(next_log_path, log_path)

    def test_explicit_video_or_log_collision_is_rejected(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            existing_video = root / "result.mp4"
            existing_video.touch()
            with self.assertRaisesRegex(FileExistsError, "video output"):
                resolve_output_paths(existing_video, root)

            output_path = root / "new-result.mp4"
            output_path.with_suffix(".csv").touch()
            with self.assertRaisesRegex(FileExistsError, "event log"):
                resolve_output_paths(output_path, root)

    def test_video_and_log_cannot_target_same_path(self):
        with TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "result.csv"
            with self.assertRaisesRegex(ValueError, "different paths"):
                resolve_output_paths(output_path, temporary_dir)


if __name__ == "__main__":
    unittest.main()

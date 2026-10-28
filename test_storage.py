import os
import tempfile
import unittest
from pathlib import Path


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["CHEM_LAB_DB_PATH"] = str(Path(self.tmp.name) / "test.db")
        import storage
        storage.DEFAULT_DB = Path(os.environ["CHEM_LAB_DB_PATH"])
        storage.init_db()
        self.storage = storage

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("CHEM_LAB_DB_PATH", None)

    def test_registration_is_hashed_and_stable(self):
        a = self.storage.make_student_key("Alice", "SRM001")
        b = self.storage.make_student_key("Different Name", "srm001")
        self.assertEqual(a, b)
        self.assertNotIn("SRM001", a)

    def test_activity_and_quiz_are_persisted(self):
        key = self.storage.make_student_key("Alice", "SRM001")
        self.storage.save_activity(key, "Alice", "2026-10-28T12:00:00+05:30", "Experiment", "test")
        attempt = {"timestamp": "2026-10-28T12:01:00+05:30", "label": "Test", "correct": 4, "total": 5, "score_percent": 80}
        self.storage.save_quiz_attempt(key, "Alice", attempt, [])
        self.assertEqual(self.storage.get_progress_summary(key)["activity_count"], 1)
        self.assertEqual(self.storage.get_progress_summary(key)["best_score"], 80)
        self.assertEqual(len(self.storage.get_activities(key)), 1)
        self.assertEqual(len(self.storage.get_quiz_attempts(key)), 1)


if __name__ == "__main__":
    unittest.main()

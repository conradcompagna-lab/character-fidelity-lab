import tempfile
import unittest
from pathlib import Path

from character_lab.chunking import chunk_file


class ChunkingTests(unittest.TestCase):
    def test_source_id_extracted(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "lore.md"
            path.write_text("# Title\n\nSource ID: test_source\n\nSome lore text.", encoding="utf-8")
            chunks = chunk_file(path)
            self.assertEqual(chunks[0].source_id, "test_source")
            self.assertIn("Some lore", chunks[0].text)


if __name__ == "__main__":
    unittest.main()

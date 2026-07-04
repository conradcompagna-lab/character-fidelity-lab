import tempfile
import unittest
from pathlib import Path

from character_lab.chunking import chunk_lore_dir
from character_lab.vector_store import VectorStore


class RetrievalTests(unittest.TestCase):
    def test_retrieves_betrayal_lore(self):
        with tempfile.TemporaryDirectory() as td:
            lore = Path(td) / "lore"
            lore.mkdir()
            (lore / "maelor.md").write_text("# Maelor\n\nSource ID: maelor_bio\n\nMaelor did not betray the Sunken Temple.", encoding="utf-8")
            (lore / "prism.md").write_text("# Prism\n\nSource ID: prism\n\nThe prism is a ring.", encoding="utf-8")
            chunks = chunk_lore_dir(lore)
            store = VectorStore.build(chunks)
            results = store.search("Did Maelor betray the temple?", top_k=1)
            self.assertEqual(results[0].chunk.source_id, "maelor_bio")


if __name__ == "__main__":
    unittest.main()

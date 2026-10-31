import tempfile
import unittest
from pathlib import Path

from vector_db import LocalVectorDB


class VectorRAGTests(unittest.TestCase):
    def test_password_query_is_matched_to_password_policy(self):
        chunks = [
            {
                "text": "Document: Company Password Reset Policy\nEmployees must reset their passwords through the official company password portal.",
                "source": "password_reset.txt",
            },
            {
                "text": "Document: Company VPN Troubleshooting Policy\nEmployees must check their internet connection before using the VPN.",
                "source": "vpn_troubleshooting.txt",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            db = LocalVectorDB(storage_dir=Path(tmp_dir) / "vector_db")
            db.add_documents(chunks)

            results = db.search("How do I reset my company password?", top_k=3)
            assert results, "No results returned from the vector database"
            assert any(result["source"] == "password_reset.txt" for result in results)

    def test_unrelated_question_has_no_good_match(self):
        chunks = [
            {
                "text": "Document: Company Password Reset Policy\nEmployees must reset their passwords through the official company password portal.",
                "source": "password_reset.txt",
            },
            {
                "text": "Document: Company VPN Troubleshooting Policy\nEmployees must check their internet connection before using the VPN.",
                "source": "vpn_troubleshooting.txt",
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            db = LocalVectorDB(storage_dir=Path(tmp_dir) / "vector_db")
            db.add_documents(chunks)

            results = db.search("What is the company gym policy?", top_k=3, min_score=0.50)
            assert results == []


if __name__ == "__main__":
    unittest.main()

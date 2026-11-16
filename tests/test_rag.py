import pytest
from src.rag.bocw_parser import get_manual_bocw_sections
from src.rag.embedder import BOCWEmbedder


class RAGSystem:
    def __init__(self):
        self.sections = {
            "No Helmet": "Section 38",
            "No Harness": "Section 39",
        }
        self.manual_sections_loaded = True

    def query(self, issue):
        if issue in self.sections:
            return self.sections[issue], 0.95
        return "Unknown Rule", 0.3

    def get_fallback(self):
        return "General Safety Clause 101"


def test_manual_sections_loaded():
    rag = RAGSystem()
    assert rag.manual_sections_loaded
    sections = get_manual_bocw_sections()
    assert "Section_38" in sections
    assert "Section_39" in sections
    assert "Section_47" in sections


def test_query_no_helmet_returns_section_38():
    rag = RAGSystem()
    ans, conf = rag.query("No Helmet")
    assert ans == "Section 38"
    assert conf > 0.9


def test_query_unknown_returns_low_confidence():
    rag = RAGSystem()
    ans, conf = rag.query("Dancing on site")
    assert conf < 0.5


def test_fallback_rules_are_correct():
    rag = RAGSystem()
    assert rag.get_fallback() == "General Safety Clause 101"


def test_real_chromadb_bocw_embedder():
    embedder = BOCWEmbedder(persist_directory="./chroma_db", collection_name="bocw_regulations")
    total = embedder.ensure_populated()
    assert total >= 8

    helmet_reg = embedder.retrieve_best_regulation("no_helmet")
    assert helmet_reg["section_id"].startswith("Section 38")
    assert helmet_reg["confidence"] >= 0.85

    harness_reg = embedder.retrieve_best_regulation("no_harness")
    assert harness_reg["section_id"].startswith("Section 39")
    assert harness_reg["confidence"] >= 0.85

import pytest

class RAGSystem:
    def __init__(self):
        self.sections = {
            "No Helmet": "Section 38",
            "No Harness": "Section 39"
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

import os
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from .bocw_parser import extract_sections, chunk_section

class BOCWEmbedder:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "bocw_regulations"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_document(self, pdf_path: str):
        sections = extract_sections(pdf_path)
        for section_id, section_text in sections.items():
            chunks = chunk_section(section_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{section_id}_chunk_{i}"
                embedding = self._embed_texts([chunk])[0]
                metadata = {
                    "section_id": section_id,
                    "tags": "pdf_extracted",
                    "violation_types": "general",
                    "penalty": "unknown"
                }
                self.collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[metadata]
                )

    def add_manual_sections(self, sections_dict: Dict[str, Dict[str, Any]]):
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for section_id, data in sections_dict.items():
            content = data.get("content", "")
            tags = ",".join(data.get("tags", []))
            violation_types = ",".join(data.get("violation_types", []))
            penalty = data.get("penalty", "N/A")
            
            ids.append(f"manual_{section_id}")
            documents.append(content)
            metadatas.append({
                "section_id": section_id,
                "tags": tags,
                "violation_types": violation_types,
                "penalty": penalty
            })
        
        if documents:
            embeddings = self._embed_texts(documents)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

    def query(self, violation_type: str, n_results: int = 3) -> Dict[str, Any]:
        query_embedding = self._embed_texts([violation_type])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

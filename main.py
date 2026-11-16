import argparse
from pathlib import Path
import uvicorn
from backend.api import app


def initialize_database():
    print("Initializing Database...")
    from src.database.models import init_db

    init_db("sqlite:///rakshasite.db")


def initialize_chromadb():
    print("Initializing ChromaDB Embeddings & Risk Model...")
    from src.rag.embedder import BOCWEmbedder

    embedder = BOCWEmbedder()
    count = embedder.ensure_populated()
    print(f"ChromaDB initialized with {count} BOCW statutory chunks.")

    if not Path("models/risk_xgboost.json").exists():
        from src.agents.risk_agent import train_and_evaluate_risk_model

        train_and_evaluate_risk_model()


def print_banner():
    banner = """
    ======================================
           RakshaSite FastAPI Backend
    ======================================
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="RakshaSite Backend Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print_banner()
    initialize_database()
    initialize_chromadb()

    print(f"Starting server at {args.host}:{args.port}")
    uvicorn.run("backend.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

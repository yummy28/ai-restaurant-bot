from qdrant_client import QdrantClient, models

vector_db = QdrantClient("http://localhost:6333", timeout=600)
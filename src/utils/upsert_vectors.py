from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

from qdrant_client import QdrantClient, models
from vector_search.dense_embeddings import DenseEmbedder
from vector_search.sparse_embeddings import SparseEmbedder
from agent.openai_manager import OpenAIManager


client = QdrantClient("http://localhost:6333", timeout=600)


def create_collection():    
    client.create_collection(
        "restaurant",
        vectors_config={
            "text-embedding-3-small": models.VectorParams(
                size=1536,
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )

sparse_embedder = SparseEmbedder()
dense_embedder = DenseEmbedder()
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

def remove_technical_strings(text):
    return text.replace("title:", "").replace("segment:", "")


def upsert_doc(doc, num):
    keywords = OpenAIManager.extract_keywords(doc["description"])
    keywords_str = " ".join(keywords)

    sparse_vector = sparse_embedder.get_embedding_for_passage(keywords_str)
    sparse_query_vector = models.SparseVector(
        indices=sparse_vector.indices,
        values=sparse_vector.values
    )

    vectors = {
        "text-embedding-3-small": dense_embedder.get_openai_embedding(remove_technical_strings(doc["description"])),
        "colbertv2.0": list(late_interaction_embedding_model.query_embed(remove_technical_strings(doc["description"])))[0],
        "bm25": sparse_query_vector,
    }

    operation_info = client.upsert(
        collection_name="restaurant",
        points=[
            models.PointStruct(
                id=num,
                vector=vectors,
                payload=doc
            )
        ]
    )
    print(operation_info)


import json

with open('../docs.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

for num, item in enumerate(data):
    upsert_doc(item, num)
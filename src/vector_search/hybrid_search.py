from fastembed import LateInteractionTextEmbedding
from qdrant_client import models

from database.qdrant_core import vector_db
from .sparse_embeddings import SparseEmbedder
from .dense_embeddings import DenseEmbedder


sparse_embedder = SparseEmbedder()
dense_embedder = DenseEmbedder()
late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

class HybridSearchManager:

    @staticmethod
    def generate_vectors(text):
        dense_vector = dense_embedder.get_openai_embedding(text=text)
        sparse_vector = sparse_embedder.get_embedding_for_query(text=text)
        late_query_vector  = list(late_interaction_embedding_model.query_embed(text))[0]
        return dense_vector, sparse_vector, late_query_vector 

    @staticmethod
    def query(text, namespace="restaurant"):
        dense_vector, sparse_vector, late_query_vector = HybridSearchManager.generate_vectors(text)
        sparse_query_vector = models.SparseVector(
            indices=sparse_vector.indices,
            values=sparse_vector.values
        )

        dense_results = vector_db.query_points(
            namespace,
            query=dense_vector,
            using="text-embedding-3-small",
            with_payload=True,
            limit=3,
        )

        sparse_results = vector_db.query_points(
            namespace,
            query=sparse_query_vector,
            using="bm25",
            with_payload=True,
            limit=3,
        )

        exclude_ids = {point.id for point in dense_results.points} | {point.id for point in sparse_results.points}


        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using="text-embedding-3-small",
                limit=20,
            ),
            models.Prefetch(
                query=sparse_query_vector,
                using="bm25",
                limit=20,
            ),
        ]
        
        colbert_results = vector_db.query_points(
            namespace,
            prefetch=prefetch,
            query=late_query_vector,
            using="colbertv2.0",
            with_payload=True,
            limit=4,
            query_filter=models.Filter(
                must_not=[
                    models.HasIdCondition(has_id=list(exclude_ids))
                ]
            )
        )

        combined_results = dense_results.points + sparse_results.points + colbert_results.points

        return combined_results
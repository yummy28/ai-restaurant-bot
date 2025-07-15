from config import openai_client, OPENAI_EMBEDDING_MODEL


class DenseEmbedder:

    @staticmethod
    def get_openai_embedding(text):
        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
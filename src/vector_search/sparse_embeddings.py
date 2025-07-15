import os
import nltk
import stanza
from nltk.corpus import stopwords
from fastembed import SparseTextEmbedding
from config import NLTK_DATA_DIRECTORY, STANZA_RESOURCES_DIRECTORY

class SparseEmbedder:
    def __init__(self):
        self.nltk_dir = NLTK_DATA_DIRECTORY
        self.stanza_dir = STANZA_RESOURCES_DIRECTORY
        self._setup()

        self.nlp = stanza.Pipeline(
            lang='uk',
            model_dir=self.stanza_dir,
            processors='tokenize,lemma',
            use_gpu=False
        )

        self.stop_words = set(stopwords.words('ukrainian'))
        self.embedding_model = SparseTextEmbedding("Qdrant/bm25")

    def _setup(self):
        os.makedirs(self.nltk_dir, exist_ok=True)
        os.makedirs(self.stanza_dir, exist_ok=True)

        if self.nltk_dir not in nltk.data.path:
            nltk.data.path.insert(0, self.nltk_dir)

        for resource in ['punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, download_dir=self.nltk_dir)

        uk_model_dir = os.path.join(self.stanza_dir, 'uk')
        if not os.path.exists(uk_model_dir) or not os.listdir(uk_model_dir):
            stanza.download('uk', model_dir=self.stanza_dir)

    def preprocess(self, text):
        doc = self.nlp(text)
        lemmas = [word.lemma.lower() for sentence in doc.sentences for word in sentence.words]
        clean_tokens = [lemma for lemma in lemmas if lemma.isalpha() and lemma not in self.stop_words]
        return " ".join(clean_tokens)

    def get_embedding_for_query(self, text):
        preprocessed_text = self.preprocess(text)
        return list(self.embedding_model.query_embed([preprocessed_text]))[0]

    def get_embedding_for_passage(self, text):
        preprocessed_text = self.preprocess(text)
        return list(self.embedding_model.passage_embed([preprocessed_text]))[0]


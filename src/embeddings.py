from typing import List
from src.config import EMBEDDING_MODEL_NAME

class EmbeddingService:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                print(f"[EmbeddingService] Loading fast embedding model: {self.model_name}...")
                self._model = TextEmbedding(model_name=self.model_name)
            except ImportError:
                raise ImportError(
                    "fastembed is required. Install it using: pip install fastembed"
                )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts and returns a list of float vectors (384 dims)."""
        self._load_model()
        # FastEmbed returns a generator of numpy arrays
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string for vector similarity search."""
        return self.embed_texts([query])[0]

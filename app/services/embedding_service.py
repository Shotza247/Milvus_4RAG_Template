import hashlib
import math
from typing import List
from app.core.config import settings
from app.core.logging import logger
from app.services.interfaces import BaseEmbeddingService


class EmbeddingService(BaseEmbeddingService):
    """
    Embedding service supporting Sentence-Transformers, FastEmbed, or deterministic CPU fallback.
    """

    def __init__(self):
        self._provider = settings.EMBEDDING_PROVIDER.lower()
        self._model_name = settings.EMBEDDING_MODEL_NAME
        self._dimension = settings.EMBEDDING_DIMENSION
        self._model = None
        self._init_model()

    def _init_model(self):
        if self._provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self._model_name}...")
                self._model = SentenceTransformer(self._model_name, device=settings.EMBEDDING_DEVICE)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded {self._model_name} (dim={self._dimension}).")
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer load failed ({e}). Falling back to deterministic hashing embedding."
                )
                self._model = None
        elif self._provider == "fastembed":
            try:
                from fastembed import TextEmbedding
                logger.info(f"Loading FastEmbed model: {self._model_name}...")
                self._model = TextEmbedding(model_name=self._model_name)
            except Exception as e:
                logger.warning(f"FastEmbed load failed ({e}). Falling back.")
                self._model = None
        else:
            logger.info("Using lightweight deterministic embedding backend.")
            self._model = None

    @property
    def dimension(self) -> int:
        return self._dimension

    def _deterministic_embed(self, text: str) -> List[float]:
        """
        Deterministic pseudo-vector generated from text hash for tests or zero-dependency environments.
        Normalized to unit length for cosine similarity.
        """
        vec = []
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        for i in range(self._dimension):
            # Deterministic linear congruential generator step
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            val = ((seed / 0x7FFFFFFF) * 2.0) - 1.0
            vec.append(val)

        # Normalize vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        if not text:
            return [0.0] * self._dimension

        if self._model is not None:
            try:
                if self._provider == "sentence-transformers":
                    emb = self._model.encode(text, convert_to_numpy=True).tolist()
                    return emb
                elif self._provider == "fastembed":
                    return list(list(self._model.embed([text]))[0])
            except Exception as e:
                logger.error(f"Error encoding with model: {e}")

        return self._deterministic_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of strings."""
        if not texts:
            return []

        if self._model is not None:
            try:
                if self._provider == "sentence-transformers":
                    embs = self._model.encode(
                        texts,
                        batch_size=settings.EMBEDDING_BATCH_SIZE,
                        convert_to_numpy=True,
                    ).tolist()
                    return embs
                elif self._provider == "fastembed":
                    return [list(e) for e in self._model.embed(texts)]
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")

        return [self._deterministic_embed(t) for t in texts]


embedding_service = EmbeddingService()

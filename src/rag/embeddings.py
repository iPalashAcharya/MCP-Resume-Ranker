"""
rag/embeddings.py — Qwen2.5-0.5B embedding wrapper.
Uses mean-pooling over the last hidden state for dense embeddings.
Singleton pattern with lazy loading so the model only loads once.
"""
from __future__ import annotations

import asyncio
import hashlib
from functools import lru_cache
from threading import Lock
from typing import List, Optional

import numpy as np

from src.config import get_logger, settings

logger = get_logger(__name__)

_model_lock = Lock()


class QwenEmbeddings:
    """
    Dense text embedder using Qwen2.5-0.5B-Instruct.

    Why Qwen for embeddings:
    - Multilingual support out of the box
    - Instruction-tuned variant follows embed-style prompts
    - Tiny footprint (0.5B params, ~1 GB RAM on CPU)

    For production at scale consider:
    - BGE-M3 (FlagEmbedding) — SOTA multilingual embedder
    - text-embedding-3-small (OpenAI) — excellent quality, API-based
    - Qwen3-Embedding (when available)
    """

    _instance: Optional["QwenEmbeddings"] = None
    _tokenizer = None
    _model = None
    _loaded = False

    def __new__(cls) -> "QwenEmbeddings":
        if cls._instance is None:
            with _model_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Lazy-load the model (called on first use or at startup)."""
        if self._loaded:
            return
        with _model_lock:
            if self._loaded:
                return
            logger.info("embeddings.loading", model=settings.embedding.model_name)

            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.embedding.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModel.from_pretrained(
                settings.embedding.model_name,
                trust_remote_code=True,
            )
            self._model.eval()

            device = settings.embedding.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            self._model.to(device)
            self._loaded = True

            logger.info(
                "embeddings.loaded",
                model=settings.embedding.model_name,
                device=device,
                dimension=settings.embedding.dimension,
            )

    @staticmethod
    def _mean_pool(token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        """Attention-mask-aware mean pooling."""
        import torch
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(
            mask_expanded.sum(1), min=1e-9
        )

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.load()

        # Add embedding instruction prefix (improves retrieval quality)
        prefixed = [f"Represent this text for retrieval: {t}" for t in texts]

        encoded = self._tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=settings.embedding.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**encoded)

        embeddings = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, processing in batches."""
        if not texts:
            return []

        batch_size = settings.embedding.batch_size
        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug("embeddings.batch", start=i, size=len(batch))
            batch_embs = self._embed_batch(batch)
            all_embeddings.append(batch_embs)

        result = np.vstack(all_embeddings)
        return result.tolist()

    def embed_single(self, text: str) -> List[float]:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Async wrapper: run embedding in a thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)

    async def aembed_single(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_single, text)

    @staticmethod
    def cache_key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()


# Module-level singleton
embedder = QwenEmbeddings()
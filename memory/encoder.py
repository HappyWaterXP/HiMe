import os
from typing import List, Optional

from openai import OpenAI

from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """
    Abstract interface for embedding encoders.

    Implementations are responsible for:
        - Reading API keys / config from environment variables (or other config).
        - Calling the external embedding service.
        - Returning list[float] embeddings of fixed dimension.
    """

    def __init__(self, embedding_dim: int):
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @abstractmethod
    def encode_obj(self, obj_name: str) -> List[float]:
        """
        Encode an object name into an embedding.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """
        Encode arbitrary text into an embedding.
        """
        raise NotImplementedError


class ZeroEncoder(BaseEncoder):
    """
    Debug encoder: reads API key from environment variables,
    but always returns all-zero vectors.

    This keeps the interface & error paths realistic, while making
    the behavior deterministic and cheap during development.

    Environment variables (example):
        - MY_EMBEDDING_API_KEY
        - or any name you prefer
    """

    def __init__(
        self,
        embedding_dim: int = 8,
        api_key_env: str = "MY_EMBEDDING_API_KEY",
    ):
        super().__init__(embedding_dim)
        # Read API key from environment, just to validate / reserve the path
        self.api_key_env = api_key_env
        self.api_key: Optional[str] = os.getenv(api_key_env)

        # For debug you can choose not to enforce API key existence.
        # if self.api_key is None:
        #     raise RuntimeError(f"{api_key_env} is not set")

    def _zero_vec(self) -> List[float]:
        return [0.0] * self._embedding_dim

    def encode_obj(self, obj_name: str) -> List[float]:
        _ = self.api_key  # avoid unused warning
        _ = obj_name
        return self._zero_vec()

    def encode_text(self, text: str) -> List[float]:
        _ = self.api_key
        _ = text
        return self._zero_vec()


class OpenAIEmbeddingEncoder(BaseEncoder):
    """
    Real encoder implementation using OpenAI's text-embedding-3-large model.

    Behavior:
        - Reads API key from environment (OPENAI_API_KEY by default).
        - Uses OpenAI client to generate embeddings.
        - Returns embeddings as Python List[float].
        - Enforces a fixed embedding_dim; optionally validates server dim.

    Environment variables:
        - OPENAI_API_KEY (default), or a custom env name via api_key_env.

    Notes:
        - text-embedding-3-large has 3072 dimensions. By default we set
          embedding_dim=3072 to match it. If you pass a different dimension,
          you must also set validate_dim=False or handle truncation/padding.
    """

    def __init__(
        self,
        embedding_dim: int = 3072,
        model: str = "text-embedding-3-large",
        api_key_env: str = "OPENAI_API_KEY",
        validate_dim: bool = True,
    ):
        super().__init__(embedding_dim)

        self.model = model
        self.api_key_env = api_key_env
        self.api_key: Optional[str] = os.getenv(api_key_env)

        if self.api_key is None:
            raise RuntimeError(
                f"{api_key_env} is not set; cannot initialize OpenAIEmbeddingEncoder."
            )

        # OpenAI Python SDK client
        self._client = OpenAI(api_key=self.api_key)

        # If you want to enforce that server embedding size == self._embedding_dim,
        # you can keep validate_dim=True. Otherwise, we will truncate or pad.
        self._validate_dim = validate_dim
        # Cached server dimension after first call
        self._server_dim: Optional[int] = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _postprocess_embedding(self, emb: List[float]) -> List[float]:
        """
        Adjust the raw embedding to match self._embedding_dim if needed.
        - If validate_dim is True and dim mismatch occurs, raise.
        - Otherwise, truncate or zero-pad to required length.
        """
        dim = len(emb)
        if self._server_dim is None:
            self._server_dim = dim

        if self._validate_dim:
            if dim != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: server={dim}, "
                    f"expected={self._embedding_dim}"
                )
            return emb

        # If not validating strictly, adapt:
        if dim == self._embedding_dim:
            return emb
        elif dim > self._embedding_dim:
            # Truncate
            return emb[: self._embedding_dim]
        else:
            # Zero-pad to the right
            padded = emb + [0.0] * (self._embedding_dim - dim)
            return padded

    def _encode(self, text: str) -> List[float]:
        """
        Low-level call to the OpenAI embeddings API with error handling.

        Returns zero vector on failure to maintain pipeline robustness.
        """
        if not text:
            # Empty text -> deterministic zero vector instead of failing
            return [0.0] * self._embedding_dim

        try:
            resp = self._client.embeddings.create(
                model=self.model,
                input=text,
            )
        except Exception as e:
            # Log error for debugging
            print(f"❌ [Encoder] Embedding API error: {e}")
            print(f"   Text (first 100 chars): {text[:100]}...")
            return [0.0] * self._embedding_dim

        if not resp.data or not hasattr(resp.data[0], "embedding"):
            # Unexpected response shape -> fallback
            print(f"⚠️  [Encoder] Unexpected API response format")
            return [0.0] * self._embedding_dim

        raw_emb = list(resp.data[0].embedding)
        return self._postprocess_embedding(raw_emb)

    # -------------------------------------------------------------------------
    # Public BaseEncoder API
    # -------------------------------------------------------------------------

    def encode_obj(self, obj_name: str) -> List[float]:
        """
        Encode an object name into an embedding using the OpenAI API.
        """
        return self._encode(obj_name)

    def encode_text(self, text: str) -> List[float]:
        """
        Encode arbitrary text into an embedding using the OpenAI API.
        """
        return self._encode(text)
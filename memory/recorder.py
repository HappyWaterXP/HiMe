from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Any, Tuple, Set
import numpy as np

from src.memory.encoder import BaseEncoder, ZeroEncoder


DataType = Literal["text", "image"]


@dataclass
class MemoryRecord:
    id: int
    obj_name: str
    data: Dict[str, Any]          # {"type": "text"|"image", "value": ...}
    obj_embedding: List[float]
    text_embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Memory:
    """
    In-memory key-value memory store with pluggable encoder.

    Public API:
        - create(obj_name, data_type, data_value, text=None)
        - read(query_obj_name=None, query_text=None, obj_threshold=..., top_k=...)
        - update(rec_id, obj_name=None, data_type=None, data_value=None, text=None)
        - delete(rec_id)
        - get, all, snapshot

    The encoder:
        - Must implement BaseEncoder.
        - Is responsible for generating embeddings from text.
        - For now, you can use ZeroEncoder to always return all-zero vectors.
    """

    def __init__(self, encoder: BaseEncoder):
        self._store: Dict[int, MemoryRecord] = {}
        self._next_id: int = 1
        self._encoder = encoder

    # -------------------------------------------------------------------------
    # Internal: encoding wrappers
    # -------------------------------------------------------------------------

    def _encode_obj(self, obj_name: str) -> List[float]:
        return self._encoder.encode_obj(obj_name)

    def _encode_text(self, text: str) -> List[float]:
        return self._encoder.encode_text(text)

    # -------------------------------------------------------------------------
    # Utility: cosine similarity
    # -------------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        if va.shape != vb.shape or va.size == 0:
            return 0.0
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    # -------------------------------------------------------------------------
    # CRUD: Create
    # -------------------------------------------------------------------------

    def create(
        self,
        obj_name: str,
        data_type: DataType,
        data_value: Any,
        text: Optional[str] = None,
    ) -> MemoryRecord:
        rec_id = self._next_id
        self._next_id += 1

        obj_emb = self._encode_obj(obj_name)
        txt_emb = self._encode_text(text) if text is not None else None

        record = MemoryRecord(
            id=rec_id,
            obj_name=obj_name,
            data={"type": data_type, "value": data_value},
            obj_embedding=obj_emb,
            text_embedding=txt_emb,
        )
        self._store[rec_id] = record
        return record

    # -------------------------------------------------------------------------
    # CRUD: Read
    # -------------------------------------------------------------------------

    def read(
        self,
        query_obj_name: Optional[str] = None,
        query_text: Optional[str] = None,
        obj_threshold: float = 0.4,
        top_k: int = 5,
    ) -> Tuple[List[MemoryRecord], Dict[int, float]]:
        matched_records: List[MemoryRecord] = []
        sim_scores: Dict[int, float] = {}

        # 1) object-level
        if query_obj_name is not None:
            query_obj_emb = self._encode_obj(query_obj_name)

            best_id: Optional[int] = None
            best_sim: float = -1.0

            for rec_id, rec in self._store.items():
                if not rec.obj_embedding:
                    continue
                sim = self._cosine_sim(query_obj_emb, rec.obj_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_id = rec_id

            if best_id is not None and best_sim >= obj_threshold:
                winner_name = self._store[best_id].obj_name
                for rec_id, rec in self._store.items():
                    if rec.obj_name == winner_name:
                        matched_records.append(rec)
                        sim_scores[rec_id] = best_sim
                matched_records.sort(key=lambda r: r.id)
                return matched_records, sim_scores

        # 2) text-level fallback
        if query_text is not None:
            query_text_emb = self._encode_text(query_text)

            sims: List[Tuple[int, float]] = []
            for rec_id, rec in self._store.items():
                if rec.text_embedding:
                    sim = self._cosine_sim(query_text_emb, rec.text_embedding)
                    sims.append((rec_id, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            top = sims[:top_k]

            for rec_id, sim in top:
                rec = self._store[rec_id]
                matched_records.append(rec)
                sim_scores[rec_id] = sim

            return matched_records, sim_scores

        return [], {}
    
    def query(
        self,
        content: str,
        obj_threshold: float = 0.75,
        top_k: int = 5,
    ) -> Tuple[List[MemoryRecord], Dict[int, float]]:
        """
        Unified query entry:
        - First try to match by treating `content` as obj_name.
        - If no object match exceeds threshold, fall back to text match
          using the same `content` string.
        """
        # 1) 尝试按 obj_name 匹配
        obj_matches, obj_scores = self.read(
            query_obj_name=content,
            query_text=None,
            obj_threshold=obj_threshold,
            top_k=top_k,
        )
        if obj_matches:
            # read 里只有在 best_sim>=threshold 时才会返回非空
            return obj_matches, obj_scores

        # 2) 如果对象匹配失败，再按 text 匹配
        text_matches, text_scores = self.read(
            query_obj_name=None,
            query_text=content,
            obj_threshold=0,
            top_k=top_k,
        )
        return text_matches, text_scores
    # -------------------------------------------------------------------------
    # CRUD: Update
    # -------------------------------------------------------------------------

    def update(
        self,
        rec_id: int,
        obj_name: Optional[str] = None,
        data_type: Optional[DataType] = None,
        data_value: Optional[Any] = None,
        text: Optional[str] = None,
    ) -> Optional[MemoryRecord]:
        rec = self._store.get(rec_id)
        if rec is None:
            return None

        if obj_name is not None:
            rec.obj_name = obj_name
            rec.obj_embedding = self._encode_obj(obj_name)

        if data_type is not None or data_value is not None:
            old_type = rec.data.get("type")
            old_value = rec.data.get("value")
            rec.data = {
                "type": data_type if data_type is not None else old_type,
                "value": data_value if data_value is not None else old_value,
            }

        if text is not None:
            rec.text_embedding = self._encode_text(text)

        self._store[rec_id] = rec
        return rec

    # -------------------------------------------------------------------------
    # CRUD: Delete
    # -------------------------------------------------------------------------

    def delete(self, rec_id: int) -> bool:
        if rec_id in self._store:
            del self._store[rec_id]
            return True
        return False

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def get(self, rec_id: int) -> Optional[MemoryRecord]:
        return self._store.get(rec_id)

    def all(self) -> List[MemoryRecord]:
        return list(self._store.values())

    def snapshot(self) -> List[Dict[str, Any]]:
        return [rec.to_dict() for rec in self._store.values()]

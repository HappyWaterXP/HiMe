from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Any, Tuple, Set
import numpy as np

from src.memory.encoder import BaseEncoder


DataType = Literal["text", "image"]


@dataclass
class MultiTagMemoryRecord:
    """
    多标签记录类（优化版）
    
    字段说明：
    - id: 记录唯一标识
    - tags: 标签列表
    - data: {"type": "text"|"image", "value": ...}
    - text_embedding: 文本描述的 embedding（每条记录单独存储）
    - image_path: 图片路径（可选）
    """
    id: int
    tags: List[str]
    data: Dict[str, Any]
    text_embedding: Optional[List[float]] = None
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiTagMemory:
    """
    多标签 Memory（完整优化版）
    
    ✅ 核心特性：
    1. Tag embeddings 全局单独存储
    2. Tag → Record IDs 倒排索引（快速查找）
    3. Tag 引用计数自动清理
    4. Text embeddings 每条记录单独存储
    5. 完整的 tag 生命周期管理
    
    ✅ Tag 维护场景：
    - create(): 新 tag → 创建 embedding + 索引
    - update(): 
      - 添加 tag → 创建 embedding + 索引
      - 移除 tag → 减少引用，可能删除 embedding + 索引
    - delete(): 减少所有 tag 引用，可能删除 embedding + 索引
    """

    def __init__(self, encoder: BaseEncoder):
        # 记录存储
        self._store: Dict[int, MultiTagMemoryRecord] = {}
        self._next_id: int = 1
        self._encoder = encoder
        
        # ✅ Tag 管理（全局）
        self._tag_embeddings_cache: Dict[str, List[float]] = {}  # tag -> embedding
        self._tag_ref_count: Dict[str, int] = {}  # tag -> 引用次数
        
        # ✅ 倒排索引
        self._tag_to_record_ids: Dict[str, Set[int]] = {}  # tag -> {record_id1, record_id2, ...}

    def _encode_obj(self, obj_name: str) -> List[float]:
        """编码 tag，使用缓存避免重复计算"""
        if obj_name in self._tag_embeddings_cache:
            return self._tag_embeddings_cache[obj_name]
        
        embedding = self._encoder.encode_obj(obj_name)
        self._tag_embeddings_cache[obj_name] = embedding
        return embedding

    def _encode_text(self, text: str) -> List[float]:
        return self._encoder.encode_text(text)

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
    # Tag 引用计数和索引管理（核心）
    # -------------------------------------------------------------------------

    def _add_tag(self, tag: str, record_id: int) -> None:
        """
        添加 tag（处理新 tag 和已有 tag）
        
        ✅ 完整逻辑：
        1. 如果是新 tag：
           - 创建引用计数 = 1
           - 计算并缓存 embedding
           - 创建倒排索引条目
        2. 如果是已有 tag：
           - 引用计数 +1
           - 倒排索引添加 record_id
        
        Args:
            tag: 标签名
            record_id: 记录 ID
        """
        # 1. 处理引用计数
        if tag in self._tag_ref_count:
            # 已有 tag，引用计数 +1
            self._tag_ref_count[tag] += 1
        else:
            # ✅ 新 tag，初始化引用计数
            self._tag_ref_count[tag] = 1
            
            # ✅ 计算并缓存 embedding（只在第一次创建时）
            if tag not in self._tag_embeddings_cache:
                print(f"[Memory] Creating new tag '{tag}' with embedding")
                self._tag_embeddings_cache[tag] = self._encoder.encode_obj(tag)
        
        # 2. 更新倒排索引
        if tag not in self._tag_to_record_ids:
            # ✅ 新 tag，创建索引条目
            self._tag_to_record_ids[tag] = set()
        
        self._tag_to_record_ids[tag].add(record_id)

    def _remove_tag(self, tag: str, record_id: int) -> None:
        """
        移除 tag（处理引用计数和清理）
        
        ✅ 完整逻辑：
        1. 从倒排索引移除 record_id
        2. 引用计数 -1
        3. 如果引用计数 = 0：
           - 删除 embedding
           - 删除倒排索引条目
           - 删除引用计数
        
        Args:
            tag: 标签名
            record_id: 记录 ID
        """
        # 0. 检查 tag 是否存在
        if tag not in self._tag_ref_count:
            print(f"[Memory] Warning: Trying to remove non-existent tag '{tag}'")
            return
        
        # 1. 从倒排索引移除
        if tag in self._tag_to_record_ids:
            self._tag_to_record_ids[tag].discard(record_id)
            
            # 如果该 tag 的索引为空，删除索引条目
            if not self._tag_to_record_ids[tag]:
                del self._tag_to_record_ids[tag]
        
        # 2. 减少引用计数
        self._tag_ref_count[tag] -= 1
        
        # 3. ✅ 引用计数为 0 时，完全清理该 tag
        if self._tag_ref_count[tag] <= 0:
            # 删除引用计数
            del self._tag_ref_count[tag]
            
            # 删除 embedding 缓存
            if tag in self._tag_embeddings_cache:
                del self._tag_embeddings_cache[tag]
            
            print(f"[Memory] Tag '{tag}' completely removed (ref_count=0)")

    def _increment_tag_refs(self, tags: List[str], record_id: int) -> None:
        """
        批量添加 tags
        
        Args:
            tags: 标签列表
            record_id: 记录 ID
        """
        for tag in tags:
            self._add_tag(tag, record_id)

    def _decrement_tag_refs(self, tags: List[str], record_id: int) -> None:
        """
        批量移除 tags
        
        Args:
            tags: 标签列表
            record_id: 记录 ID
        """
        for tag in tags:
            self._remove_tag(tag, record_id)

    # -------------------------------------------------------------------------
    # CRUD: Create
    # -------------------------------------------------------------------------

    def create(
        self,
        tags: List[str],
        data_type: DataType,
        data_value: Any,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> MultiTagMemoryRecord:
        """
        创建记录
        
        ✅ Tag 维护：
        - 新 tag → 创建 embedding + 索引 + 引用计数
        - 已有 tag → 引用计数 +1，添加到索引
        
        Args:
            tags: 标签列表，如 ["apple", "fruit", "red"]
            data_type: "text" 或 "image"
            data_value: 数据内容
            text: 文本描述（用于检索）
            image_path: 图片路径（仅当 data_type="image" 时使用）
        """
        rec_id = self._next_id
        self._next_id += 1

        # ✅ 添加 tags（自动处理新 tag 的创建）
        self._increment_tag_refs(tags, rec_id)

        # 生成文本 embedding（每条记录单独存储）
        txt_emb = self._encode_text(text) if text is not None else None

        record = MultiTagMemoryRecord(
            id=rec_id,
            tags=tags,
            data={"type": data_type, "value": data_value},
            text_embedding=txt_emb,
            image_path=image_path,
        )
        self._store[rec_id] = record
        
        print(f"[Memory] Created record {rec_id} with tags: {tags}")
        return record

    # -------------------------------------------------------------------------
    # CRUD: Read
    # -------------------------------------------------------------------------

    def read(
        self,
        query_tags: Optional[List[str]] = None,
        query_text: Optional[str] = None,
        tag_threshold: float = 0.4,
        top_k: int = 5,
    ) -> Tuple[List[MultiTagMemoryRecord], Dict[int, float]]:
        """
        检索记录（优化版）
        
        ✅ 优化逻辑：
        1. Tag 匹配：找到最相似的 tag，通过倒排索引快速获取所有相关记录
        2. Text 匹配：只有 tag 匹配失败时才遍历所有记录
        
        Returns:
            (matched_records, sim_scores)
        """
        matched_records: List[MultiTagMemoryRecord] = []
        sim_scores: Dict[int, float] = {}

        # 1) Tag-level matching
        if query_tags is not None:
            query_tag_embs = {tag: self._encode_obj(tag) for tag in query_tags}
            
            # ✅ 第一步：找到与 query 最相似的 tag（全局搜索）
            best_tag = None
            best_tag_sim = -1.0
            
            for q_tag, q_emb in query_tag_embs.items():
                # 遍历所有已知的 tags（全局）
                # import pdb; pdb.set_trace()
                for stored_tag, stored_emb in self._tag_embeddings_cache.items():
                    sim = self._cosine_sim(q_emb, stored_emb)
                    if sim > best_tag_sim:
                        best_tag_sim = sim
                        best_tag = stored_tag
            
            # ✅ 第二步：如果找到相似度 >= threshold 的 tag
            if best_tag is not None and best_tag_sim >= tag_threshold:
                # ✅ 通过倒排索引快速获取所有包含该 tag 的记录
                if best_tag in self._tag_to_record_ids:
                    record_ids = self._tag_to_record_ids[best_tag]
                    
                    print(f"[Memory] Tag match: '{best_tag}' (sim={best_tag_sim:.3f}) -> {len(record_ids)} records")
                    
                    for rec_id in record_ids:
                        rec = self._store.get(rec_id)
                        if rec:
                            matched_records.append(rec)
                            sim_scores[rec_id] = best_tag_sim
                    
                    matched_records.sort(key=lambda r: r.id)
                    return matched_records, sim_scores

        # 2) Text-level fallback
        if query_text is not None:
            query_text_emb = self._encode_text(query_text)
            sims: List[Tuple[int, float]] = []

            print(f"[Memory] Tag match failed, using text search")
            
            # ✅ 只有 tag 匹配失败时才遍历所有记录
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
        tag_threshold: float = 0.3,
        top_k: int = 5,
    ) -> Tuple[List[MultiTagMemoryRecord], Dict[int, float]]:
        """
        统一查询接口：先尝试 tag 匹配，失败后作为 text 匹配
        
        ✅ 这是给 PlannerAgent 调用的主要接口
        """
        tag_matches, tag_scores = self.read(
            query_tags=[content],
            query_text=None,
            tag_threshold=tag_threshold,
            top_k=top_k,
        )
        if tag_matches:
            return tag_matches, tag_scores

        text_matches, text_scores = self.read(
            query_tags=None,
            query_text=content,
            tag_threshold=0,
            top_k=top_k,
        )
        return text_matches, text_scores

    # -------------------------------------------------------------------------
    # CRUD: Update
    # -------------------------------------------------------------------------

    def update(
        self,
        rec_id: int,
        tags: Optional[List[str]] = None,
        data_type: Optional[DataType] = None,
        data_value: Optional[Any] = None,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Optional[MultiTagMemoryRecord]:
        """
        更新记录
        
        ✅ Tag 维护（完整逻辑）：
        1. 计算 removed_tags 和 added_tags
        2. removed_tags:
           - 引用计数 -1
           - 从倒排索引移除
           - 如果计数 = 0，删除 embedding
        3. added_tags:
           - 如果是新 tag，创建 embedding + 索引
           - 引用计数 +1
           - 添加到倒排索引
        
        Args:
            rec_id: 记录 ID
            tags: 新的标签列表（可选）
            data_type: 新的数据类型（可选）
            data_value: 新的数据值（可选）
            text: 新的文本描述（可选）
            image_path: 新的图片路径（可选）
        """
        rec = self._store.get(rec_id)
        if rec is None:
            print(f"[Memory] Warning: Record {rec_id} not found")
            return None

        # ✅ 处理 tags 变更
        if tags is not None:
            old_tags = set(rec.tags)
            new_tags = set(tags)
            
            # 计算差异
            removed_tags = list(old_tags - new_tags)
            added_tags = list(new_tags - old_tags)
            
            # ✅ 移除旧 tags（可能导致 tag 被完全删除）
            if removed_tags:
                print(f"[Memory] Removing tags from record {rec_id}: {removed_tags}")
                self._decrement_tag_refs(removed_tags, rec_id)
            
            # ✅ 添加新 tags（可能创建新 tag）
            if added_tags:
                print(f"[Memory] Adding tags to record {rec_id}: {added_tags}")
                self._increment_tag_refs(added_tags, rec_id)
            
            # 更新记录
            rec.tags = tags

        # 更新数据
        if data_type is not None or data_value is not None:
            old_type = rec.data.get("type")
            old_value = rec.data.get("value")
            rec.data = {
                "type": data_type if data_type is not None else old_type,
                "value": data_value if data_value is not None else old_value,
            }

        # 更新文本 embedding
        if text is not None:
            rec.text_embedding = self._encode_text(text)

        # 更新图片路径
        if image_path is not None:
            rec.image_path = image_path

        self._store[rec_id] = rec
        return rec

    # -------------------------------------------------------------------------
    # CRUD: Delete
    # -------------------------------------------------------------------------

    def delete(self, rec_id: int) -> bool:
        """
        删除记录
        
        ✅ Tag 维护：
        - 减少所有 tags 的引用计数
        - 从倒排索引移除
        - 如果任何 tag 的引用计数 = 0，删除 embedding
        
        Args:
            rec_id: 记录 ID
        """
        if rec_id not in self._store:
            print(f"[Memory] Warning: Record {rec_id} not found")
            return False
        
        rec = self._store[rec_id]
        
        print(f"[Memory] Deleting record {rec_id} with tags: {rec.tags}")
        
        # ✅ 减少 tag 引用计数和更新索引（可能导致 tag 被完全删除）
        self._decrement_tag_refs(rec.tags, rec_id)
        
        del self._store[rec_id]
        return True

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def get(self, rec_id: int) -> Optional[MultiTagMemoryRecord]:
        """获取单条记录"""
        return self._store.get(rec_id)

    def all(self) -> List[MultiTagMemoryRecord]:
        """获取所有记录"""
        return list(self._store.values())

    def snapshot(self) -> List[Dict[str, Any]]:
        """获取所有记录的字典形式"""
        return [rec.to_dict() for rec in self._store.values()]

    def get_tag_stats(self) -> Dict[str, int]:
        """
        返回所有 tag 及其引用计数
        
        用于调试和监控
        """
        return dict(self._tag_ref_count)

    def get_all_tags(self) -> List[str]:
        """返回当前所有活跃的 tag"""
        return list(self._tag_ref_count.keys())
    
    def get_records_by_tag(self, tag: str) -> List[MultiTagMemoryRecord]:
        """
        根据 tag 获取所有相关记录
        
        Args:
            tag: 标签名
            
        Returns:
            包含该 tag 的所有记录
        """
        if tag not in self._tag_to_record_ids:
            return []
        
        records = []
        for rec_id in self._tag_to_record_ids[tag]:
            rec = self._store.get(rec_id)
            if rec:
                records.append(rec)
        
        return records
    
    # ✅ 新增：调试接口
    def debug_info(self) -> Dict[str, Any]:
        """
        返回内部状态（用于调试）
        
        Returns:
            包含所有内部数据结构的字典
        """
        return {
            "total_records": len(self._store),
            "total_tags": len(self._tag_embeddings_cache),
            "tag_ref_counts": dict(self._tag_ref_count),
            "tag_index_sizes": {
                tag: len(rec_ids) 
                for tag, rec_ids in self._tag_to_record_ids.items()
            },
        }

    def all_light(self) -> List[Dict[str, Any]]:
        """
        返回所有记录的轻量视图（不包含 text_embedding）
        用于调试、打印或对外展示，避免泄露 embedding 数据
        """
        light_records: List[Dict[str, Any]] = []
        for rec in self._store.values():
            light_records.append({
                "id": rec.id,
                "tags": list(rec.tags),
                "data": {
                    "type": rec.data.get("type"),
                    "value": rec.data.get("value"),
                },
                "image_path": rec.image_path,
                # 不返回 text_embedding
            })
        return light_records

    def snapshot_light(self) -> List[Dict[str, Any]]:
        """
        轻量快照，不包含 text_embedding
        与 all_light 相同，但命名区分为快照语义，便于保存到 JSON
        """
        return self.all_light()
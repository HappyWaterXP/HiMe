from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional, Any, Tuple, Set
import numpy as np

from .encoder import BaseEncoder


DataType = Literal["text", "image"]


@dataclass
class MultiTagMemoryRecord:
    """
    多标签记录类（优化版）
    
    字段说明：
    - id: 记录唯一标识
    - tags: 标签列表
    - data_type: text/image
    - text: 如果 data_type 是 image, 则为图片的 caption
    - text_embedding: 文本描述的 embedding（每条记录单独存储）
    - image_path: 图片路径（可选）
    """
    id: int
    tags: List[str]
    data_type: DataType = Literal["text", "image"]
    text: str
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
        text: srt,
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
            data_type=data_type,
            text=text,
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
        1. Tag 匹配：找到最相似的 tag, 通过倒排索引快速获取所有相关记录
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
        if query_tags is None and query_text is not None:
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
            top_k=0,
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
        text: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> Optional[MultiTagMemoryRecord]:
        """
        更新记录：
        - rec_id 必填
        - 其余字段：只要参数不为 None 就覆盖；为 None 则保持不变
        """
        rec = self._store.get(rec_id)
        if rec is None:
            print(f"[Memory] Warning: Record {rec_id} not found")
            return None

        # 1) tags：只要传了（非 None）就覆盖，并正确维护引用计数
        if tags is not None:
            old_tags = set(rec.tags or [])
            new_tags = set(tags)

            removed_tags = list(old_tags - new_tags)
            added_tags = list(new_tags - old_tags)

            if added_tags:
                print(f"[Memory] Adding tags to record {rec_id}: {added_tags}")
                self._increment_tag_refs(added_tags, rec_id)

            if removed_tags:
                print(f"[Memory] Removing tags from record {rec_id}: {removed_tags}")
                self._decrement_tag_refs(removed_tags, rec_id)

            rec.tags = tags  # 允许 [] 覆盖（清空）

        # 2) text：传了就覆盖 value 并重算 embedding；没传就都不动
        if text is not None:
            rec.text = text
            rec.text_embedding = self._encode_text(text)

        # 3) data_type：传了就覆盖
        if data_type is not None:
            rec.data_type = data_type

        # 4) image_path：传了就覆盖（允许传 "" 来清空，若你需要）
        if image_path is not None:
            rec.image_path = image_path

        # 一般不需要再写回 _store（rec 是引用），但写了也无妨
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
        用于调试、打印或对外展示
        """
        light_records: List[Dict[str, Any]] = []
        for rec in self._store.values():
            light_records.append({
                "id": rec.id,
                "tags": list(rec.tags),
                "data_type": rec.data_type,
                "text": rec.text,
                "image_path": rec.image_path,
                # 不返回 text_embedding
            })
        return light_records
    
    def prune_to_max_records(self, n: int) -> List[int]:
        """
        将 memory 记录数量裁剪到最多 n 条。
        若当前记录数 > n，则按 id 从小到大删除多余记录（删除最旧的）。

        Args:
            n: 允许保留的最大记录数（n >= 0）

        Returns:
            deleted_ids: 实际被删除的 record ids（按删除顺序）
        """
        if n < 0:
            raise ValueError("n must be >= 0")

        total = len(self._store)
        if total <= n:
            return []

        # 需要删除的数量
        need_delete = total - n

        # 按 id 从小到大选出要删的记录
        ids_sorted = sorted(self._store.keys())
        to_delete = ids_sorted[:need_delete]

        deleted_ids: List[int] = []
        for rid in to_delete:
            ok = self.delete(rid)  # 复用既有 delete()，保证 tag 生命周期一致
            if ok:
                deleted_ids.append(rid)

        return deleted_ids

    # -------------------------------------------------------------------------
    # Persistence: Save & Resume
    # -------------------------------------------------------------------------

    def save_to_json(self, filepath: str) -> None:
        """
        保存 memory 状态到 JSON 文件（不包含 embeddings）

        保存内容：
        - 所有记录的 id, tags, data, image_path
        - next_id（用于恢复 ID 生成器）

        不保存：
        - text_embedding（会在 resume 时重新生成）
        - tag_embeddings_cache（会在 resume 时重新生成）
        - tag_ref_count（会在 resume 时重新计算）
        - tag_to_record_ids（会在 resume 时重新建立）

        Args:
            filepath: JSON 文件路径
        """
        import json
        from pathlib import Path

        # 准备数据
        data = {
            "version": "1.0",
            "next_id": self._next_id,
            "records": []
        }

        # 保存所有记录（不含 embeddings）
        for rec in self._store.values():
            data["records"].append({
                "id": rec.id,
                "tags": list(rec.tags),
                "data_type": rec.data_type,
                "text": rec.text,
                "image_path": str(rec.image_path) if rec.image_path else None,
            })

        # 写入文件
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[Memory] Saved {len(data['records'])} records to {filepath}")

    @classmethod
    def resume_from_json(cls, filepath: str, encoder: BaseEncoder) -> "MultiTagMemory":
        """
        从 JSON 文件恢复 memory 状态

        支持两种格式：
        1. 标准格式（save_to_json 保存的）:
           {
             "version": "1.0",
             "next_id": 4,
             "records": [...]
           }

        2. 轻量格式（all_light/snapshot_light 保存的）:
           [
             {"id": 1, "tags": [...], ...},
             {"id": 2, "tags": [...], ...},
             ...
           ]

        完整重建：
        1. 恢复所有记录
        2. 重新生成 text_embedding（每条记录）
        3. 重新生成 tag_embeddings_cache（所有唯一 tags）
        4. 重建 tag_ref_count（统计每个 tag 的引用次数）
        5. 重建 tag_to_record_ids 倒排索引
        6. 恢复 next_id（如果有）

        Args:
            filepath: JSON 文件路径
            encoder: Encoder 实例（用于生成 embeddings）

        Returns:
            恢复的 MultiTagMemory 实例
        """
        import json
        from pathlib import Path

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Memory file not found: {filepath}")

        # 读取 JSON
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 创建新实例
        memory = cls(encoder=encoder)

        # ✅ 兼容两种格式
        if isinstance(data, list):
            # 轻量格式：直接是记录数组
            records_data = data
            # 从记录中推断 next_id（最大 id + 1）
            if records_data:
                max_id = max(rec.get("id", 0) for rec in records_data)
                memory._next_id = max_id + 1
            print(f"[Memory] Resuming from {filepath}: {len(records_data)} records (lightweight format)")
        elif isinstance(data, dict):
            # 标准格式：带 version 和 next_id
            records_data = data.get("records", [])
            memory._next_id = data.get("next_id", 1)
            print(f"[Memory] Resuming from {filepath}: {len(records_data)} records (standard format)")
        else:
            raise ValueError(f"Invalid JSON format: expected list or dict, got {type(data)}")

        # Step 1: 收集所有唯一的 tags
        all_tags: set[str] = set()
        for rec_data in records_data:
            tags = rec_data.get("tags", [])
            all_tags.update(tags)

        # Step 2: 预先生成所有 tag embeddings（避免重复计算）
        print(f"[Memory] Generating embeddings for {len(all_tags)} unique tags...")
        for tag in all_tags:
            memory._tag_embeddings_cache[tag] = encoder.encode_obj(tag)

        # Step 3: 恢复每条记录
        for rec_data in records_data:
            rec_id = rec_data["id"]
            tags = rec_data.get("tags", [])
            data_type = rec_data.get("data_type", "text")
            text = rec_data.get("text", "")
            image_path = rec_data.get("image_path")

            # 生成 text_embedding
            text_embedding = None
            if data_value is not None:
                text_embedding = encoder.encode_text(str(text))

            # 创建记录对象
            record = MultiTagMemoryRecord(
                id=rec_id,
                tags=tags,
                data_type=data_type,
                text=text,
                text_embedding=text_embedding,
                image_path=image_path,
            )

            # 添加到 store
            memory._store[rec_id] = record

            # 更新 tag 引用计数和倒排索引
            for tag in tags:
                # 引用计数
                if tag not in memory._tag_ref_count:
                    memory._tag_ref_count[tag] = 0
                memory._tag_ref_count[tag] += 1

                # 倒排索引
                if tag not in memory._tag_to_record_ids:
                    memory._tag_to_record_ids[tag] = set()
                memory._tag_to_record_ids[tag].add(rec_id)

        print(f"[Memory] Resume complete: {len(memory._store)} records, {len(memory._tag_embeddings_cache)} tags")

        return memory
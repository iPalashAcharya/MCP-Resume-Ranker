"""
rag/vector_store.py — Milvus vector store manager.
Handles collection creation, upsert, similarity search, and deletion.
Supports both resume embeddings and JD embeddings in separate collections.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from src.config import get_logger, settings
from src.rag.chunker import chunk_text_from_raw

logger = get_logger(__name__)

# ─── Schema helpers ──────────────────────────────────────────────────────────

# If you add fields (e.g. project_skills), existing Milvus collections must be dropped/recreated
# or migrated so the schema matches; otherwise insert/query will fail.
RESUME_SCHEMA_FIELDS = [
    ("resume_id",        "VARCHAR",  64,   True),   # primary key
    ("s3_key",           "VARCHAR",  512,  False),
    ("s3_bucket",        "VARCHAR",  128,  False),
    ("candidate_name",   "VARCHAR",  256,  False),
    ("skills",           "VARCHAR",  1024, False),
    ("project_skills",   "VARCHAR",  1024, False),
    ("experience_years", "FLOAT",    None, False),
    ("education",        "VARCHAR",  512,  False),
    ("email",            "VARCHAR",  256,  False),
    ("word_count",       "INT64",    None, False),
    ("chunk_index",      "INT64",    None, False),
    ("section",          "VARCHAR",  64,   False),
    ("skill_signals",    "VARCHAR",  1024, False),
    ("chunk_text",       "VARCHAR",  2048, False),
    ("embedding",        "FLOAT_VECTOR", settings.embedding.dimension, False),
]

JD_SCHEMA_FIELDS = [
    ("jd_id",               "VARCHAR",  64,   True),
    ("title",               "VARCHAR",  256,  False),
    ("company",             "VARCHAR",  256,  False),
    ("required_skills",     "VARCHAR",  1024, False),
    ("preferred_skills",    "VARCHAR",  512,  False),
    ("min_experience_years","FLOAT",    None, False),
    ("location",            "VARCHAR",  256,  False),
    ("employment_type",     "VARCHAR",  128,  False),
    ("s3_key",              "VARCHAR",  512,  False),
    ("embedding",           "FLOAT_VECTOR", settings.embedding.dimension, False),
]


def _build_schema(fields_def: list, description: str):
    from pymilvus import CollectionSchema, FieldSchema, DataType

    dt_map = {
        "VARCHAR": DataType.VARCHAR,
        "FLOAT": DataType.FLOAT,
        "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
        "INT64": DataType.INT64,
    }

    fields = []
    for item in fields_def:
        name, dtype_str, dim_or_len, is_pk = item
        dtype = dt_map[dtype_str]
        kwargs: dict[str, Any] = {"name": name, "dtype": dtype, "is_primary": is_pk, "auto_id": False}

        if dtype == DataType.VARCHAR:
            kwargs["max_length"] = dim_or_len
        elif dtype == DataType.FLOAT_VECTOR:
            kwargs["dim"] = dim_or_len

        fields.append(FieldSchema(**kwargs))

    return CollectionSchema(fields=fields, description=description)


# ─── Chunker ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Backward-compatible chunking helper.
    Prefer importing `chunk_text_from_raw` from `src.rag.chunker` directly.
    """
    return chunk_text_from_raw(text=text, chunk_size=chunk_size, overlap=overlap)


# ─── MilvusVectorStore ────────────────────────────────────────────────────────

class MilvusVectorStore:
    """
    Manages two Milvus collections:
     - resume_embeddings   (chunked resume text)
     - jd_embeddings       (whole JD text)
    """

    def __init__(self):
        self._connected = False
        self._resume_collection = None
        self._jd_collection = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        from pymilvus import connections, db
        from pymilvus.exceptions import MilvusException

        if self._connected:
            return

        # Connect to the built-in database first so we can list/create logical DBs if needed.
        connections.connect(
            alias="default",
            host=settings.milvus.host,
            port=settings.milvus.port,
            user=settings.milvus.user,
            password=settings.milvus.password,
            db_name="default",
            timeout=30,
        )

        target_db = settings.milvus.db_name or "default"
        if target_db != "default":
            try:
                existing = set(db.list_database())
            except Exception as exc:
                logger.warning("milvus.list_database_failed", error=str(exc))
                existing = set()
            if target_db not in existing:
                try:
                    db.create_database(target_db)
                    logger.info("milvus.database_created", name=target_db)
                except MilvusException:
                    try:
                        present = target_db in db.list_database()
                    except Exception:
                        connections.disconnect("default")
                        raise
                    if not present:
                        connections.disconnect("default")
                        raise
                    logger.info("milvus.database_exists", name=target_db)

        db.using_database(target_db)

        self._connected = True
        logger.info(
            "milvus.connected",
            host=settings.milvus.host,
            port=settings.milvus.port,
            db=target_db,
        )

        self._ensure_collections()

    def _ensure_collections(self) -> None:
        from pymilvus import Collection, utility

        # Resume collection
        if not utility.has_collection(settings.milvus.collection_name):
            schema = _build_schema(RESUME_SCHEMA_FIELDS, "Resume embeddings for RAG ranking")
            col = Collection(name=settings.milvus.collection_name, schema=schema, consistency_level="Strong")
            col.create_index(
                field_name="embedding",
                index_params={
                    "index_type": settings.milvus.index_type,
                    "metric_type": settings.milvus.metric_type,
                    "params": {"nlist": settings.milvus.nlist},
                },
            )
            logger.info("milvus.collection_created", name=settings.milvus.collection_name)
        self._resume_collection = Collection(settings.milvus.collection_name)
        self._resume_collection.load()

        # JD collection
        if not utility.has_collection(settings.milvus.jd_collection_name):
            schema = _build_schema(JD_SCHEMA_FIELDS, "Job Description embeddings")
            col = Collection(name=settings.milvus.jd_collection_name, schema=schema, consistency_level="Strong")
            col.create_index(
                field_name="embedding",
                index_params={
                    "index_type": settings.milvus.index_type,
                    "metric_type": settings.milvus.metric_type,
                    "params": {"nlist": settings.milvus.nlist},
                },
            )
            logger.info("milvus.jd_collection_created", name=settings.milvus.jd_collection_name)
        self._jd_collection = Collection(settings.milvus.jd_collection_name)
        self._jd_collection.load()

    # ── Resume CRUD ───────────────────────────────────────────────────────────

    def upsert_resume(
        self,
        resume_id: str,
        metadata: dict,
        embeddings: List[List[float]],
        chunks: List[str],
        chunk_sections: Optional[List[str]] = None,
        chunk_skill_signals: Optional[List[str]] = None,
    ) -> int:
        """Insert or overwrite all chunks for a resume. Returns inserted count."""
        self.connect()

        # Delete existing chunks for this resume_id
        self._resume_collection.delete(expr=f'resume_id == "{resume_id}"')

        n = len(chunks)
        secs = chunk_sections if chunk_sections is not None else [""] * n
        sigs = chunk_skill_signals if chunk_skill_signals is not None else [""] * n
        if len(secs) != n or len(sigs) != n:
            raise ValueError(
                f"chunk_sections and chunk_skill_signals must match chunks length ({n})",
            )

        rows = []
        for idx, (emb, chunk) in enumerate(zip(embeddings, chunks)):
            chunk_id = f"{resume_id}__{idx}"
            sec = (secs[idx] or "")[:60]
            sig = (sigs[idx] or "")[:1000]
            rows.append({
                "resume_id": chunk_id,   # unique PK per chunk
                "s3_key": metadata.get("s3_key", ""),
                "s3_bucket": metadata.get("s3_bucket", ""),
                "candidate_name": metadata.get("candidate_name", ""),
                "skills": metadata.get("skills", ""),
                "project_skills": metadata.get("project_skills", ""),
                "experience_years": float(metadata.get("experience_years", 0.0)),
                "education": metadata.get("education", ""),
                "email": metadata.get("email", ""),
                "word_count": int(metadata.get("word_count", 0)),
                "chunk_index": idx,
                "section": sec,
                "skill_signals": sig,
                "chunk_text": chunk[:2000],
                "embedding": emb,
            })

        if rows:
            self._resume_collection.insert(rows)
            self._resume_collection.flush()

        logger.info("milvus.upsert_resume", resume_id=resume_id, chunks=len(rows))
        return len(rows)

    def delete_resume(self, resume_id: str) -> None:
        self.connect()
        self._resume_collection.delete(expr=f'resume_id like "{resume_id}__%"')
        self._resume_collection.flush()
        logger.info("milvus.delete_resume", resume_id=resume_id)

    # ── JD CRUD ───────────────────────────────────────────────────────────────

    def upsert_jd(
        self,
        jd_id: str,
        metadata: dict,
        embedding: List[float],
    ) -> None:
        self.connect()
        self._jd_collection.delete(expr=f'jd_id == "{jd_id}"')
        row = {
            "jd_id": jd_id,
            "title": metadata.get("title", ""),
            "company": metadata.get("company", ""),
            "required_skills": metadata.get("required_skills", ""),
            "preferred_skills": metadata.get("preferred_skills", ""),
            "min_experience_years": float(metadata.get("min_experience_years", 0.0)),
            "location": metadata.get("location", ""),
            "employment_type": metadata.get("employment_type", ""),
            "s3_key": metadata.get("s3_key", ""),
            "embedding": embedding,
        }
        self._jd_collection.insert([row])
        self._jd_collection.flush()
        logger.info("milvus.upsert_jd", jd_id=jd_id)

    # ── Search ────────────────────────────────────────────────────────────────

    def search_resumes(
        self,
        query_embedding: List[float],
        top_k: int = None,
        filters: Optional[str] = None,
    ) -> List[Dict]:
        """
        Vector similarity search over resume chunks.
        Returns deduplicated candidates with best chunk score.
        """
        self.connect()
        top_k = top_k or settings.rag.top_k

        search_params = {
            "metric_type": settings.milvus.metric_type,
            "params": {"nprobe": 16},
        }
        output_fields = [
            "s3_key", "s3_bucket", "candidate_name", "skills", "project_skills",
            "experience_years", "education", "email", "chunk_index", "section",
            "skill_signals", "chunk_text",
        ]

        results = self._resume_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k * 3,   # fetch more to deduplicate
            expr=filters,
            output_fields=output_fields,
        )

        # Deduplicate by candidate (keep best score per s3_key)
        seen: Dict[str, dict] = {}
        for hit in results[0]:
            key = hit.entity.get("s3_key")
            score = float(hit.score)
            if score < settings.rag.score_threshold:
                continue
            if key not in seen or score > seen[key]["score"]:
                ps_raw = hit.entity.get("project_skills") or ""
                seen[key] = {
                    "s3_key": key,
                    "s3_bucket": hit.entity.get("s3_bucket"),
                    "candidate_name": hit.entity.get("candidate_name"),
                    "skills": hit.entity.get("skills", "").split(","),
                    "project_skills": [x.strip() for x in ps_raw.split(",") if x.strip()],
                    "experience_years": hit.entity.get("experience_years"),
                    "education": hit.entity.get("education", "").split(" | "),
                    "email": hit.entity.get("email"),
                    "best_chunk": hit.entity.get("chunk_text"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "score": score,
                }

        ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def _escape_milvus_str(self, s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def query_chunks_by_s3_keys(self, s3_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Load all indexed chunks for the given S3 keys and merge chunk_text in chunk_index order.
        Returns map: s3_key -> {merged_text, chunk_rows, skills, project_skills, ...}.
        Each chunk_rows[i] has chunk_index, section, skill_signals, chunk_text.
        Keys with no rows in Milvus are omitted.
        """
        if not s3_keys:
            return {}
        self.connect()
        quoted = ",".join(f'"{self._escape_milvus_str(k)}"' for k in s3_keys) # quoted = '"resumes/john_doe.pdf","resumes/jane_smith.pdf"'
        expr = f"s3_key in [{quoted}]" # filter (like WHERE IN in SQL) s3_key in ["resumes/john_doe.pdf","resumes/jane_smith.pdf"]
        output_fields = [
            "s3_key",
            "s3_bucket",
            "candidate_name",
            "skills",
            "project_skills",
            "experience_years",
            "education",
            "email",
            "chunk_index",
            "section",
            "skill_signals",
            "chunk_text",
        ]
        rows = self._resume_collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=16384,
        )
        by_key: Dict[str, List[Tuple[int, str]]] = {}
        meta_by_key: Dict[str, Dict[str, Any]] = {}
        # Per-chunk rows for ranking boosts (section + skill_signals + chunk_text).
        rows_by_key: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            sk = row.get("s3_key")
            if not sk:
                continue
            idx = int(row.get("chunk_index") or 0)
            text = row.get("chunk_text") or ""
            by_key.setdefault(sk, []).append((idx, text))
            rows_by_key.setdefault(sk, []).append(
                {
                    "chunk_index": idx,
                    "section": (row.get("section") or "").strip(),
                    "skill_signals": (row.get("skill_signals") or "").strip(),
                    "chunk_text": text,
                }
            )
            if sk not in meta_by_key:
                ps_row = row.get("project_skills") or ""
                meta_by_key[sk] = {
                    "s3_bucket": row.get("s3_bucket"),
                    "candidate_name": row.get("candidate_name") or "Unknown",
                    "skills": row.get("skills", ""),
                    "project_skills": [x.strip() for x in str(ps_row).split(",") if x.strip()],
                    "experience_years": row.get("experience_years"),
                    "education": row.get("education", ""),
                    "email": row.get("email"),
                }
        out: Dict[str, Dict[str, Any]] = {}
        for sk, parts in by_key.items(): # for each resume key, sort the chunks by chunk_index and merge the chunks into a single string
            parts.sort(key=lambda x: x[0]) # sort the chunks by chunk_index
            merged = "\n\n".join(p[1] for p in parts if p[1]) # merge the chunks into a single string
            m = meta_by_key.get(sk, {})
            detail = sorted(rows_by_key.get(sk, []), key=lambda x: x["chunk_index"])
            out[sk] = {
                **m,
                "s3_key": sk,
                "merged_text": merged,
                "chunk_rows": detail,
            }
        return out

    def get_resume_by_id(self, resume_id: str) -> Optional[dict]:
        self.connect()
        results = self._resume_collection.query(
            expr=f'resume_id like "{resume_id}__%"',
            output_fields=[
                "s3_key", "s3_bucket", "candidate_name", "skills", "project_skills", "experience_years",
            ],
            limit=1,
        )
        return results[0] if results else None

    def count_resumes(self) -> int:
        self.connect()
        return self._resume_collection.num_entities

    def health_check(self) -> dict:
        try:
            self.connect()
            return {
                "status": "healthy",
                "host": settings.milvus.host,
                "port": settings.milvus.port,
                "resume_count": self._resume_collection.num_entities,
                "jd_count": self._jd_collection.num_entities,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # ── Async wrappers ────────────────────────────────────────────────────────

    async def asearch_resumes(self, query_embedding: List[float], top_k: int = None) -> List[Dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_resumes, query_embedding, top_k)

    async def aquery_chunks_by_s3_keys(self, s3_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query_chunks_by_s3_keys, s3_keys)

    async def aupsert_resume(
        self,
        resume_id,
        metadata,
        embeddings,
        chunks,
        chunk_sections=None,
        chunk_skill_signals=None,
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.upsert_resume(
                resume_id,
                metadata,
                embeddings,
                chunks,
                chunk_sections,
                chunk_skill_signals,
            ),
        )

    async def aupsert_jd(self, jd_id, metadata, embedding):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.upsert_jd, jd_id, metadata, embedding)


# Module-level singleton
vector_store = MilvusVectorStore()
import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np
from ..config import settings, STORAGE_DIR
from ..models.schemas import VideoMetadata, SpannerStats

logger = logging.getLogger(__name__)

MOCK_DB_FILE = STORAGE_DIR / "spanner_mock_db.json"

class SpannerService:
    def __init__(self):
        self._client = None
        self._instance = None
        self._database = None
        self._initialized = False
        self._local_db: Dict[str, Dict[str, Any]] = {}
        self._load_local_db()

    def _load_local_db(self):
        """Loads local simulated Spanner database from JSON file if present."""
        if MOCK_DB_FILE.exists():
            try:
                with open(MOCK_DB_FILE, "r") as f:
                    self._local_db = json.load(f)
                logger.info(f"Loaded {len(self._local_db)} video records from local Spanner cache.")
            except Exception as e:
                logger.error(f"Failed to load local db file ({e}), starting empty.")
                self._local_db = {}
        else:
            self._local_db = {}

    def _save_local_db(self):
        """Persists local simulated Spanner database to disk."""
        try:
            with open(MOCK_DB_FILE, "w") as f:
                json.dump(self._local_db, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save local db: {e}")

    def _init_client(self):
        if self._initialized:
            return
        if settings.USE_MOCK_GCP:
            logger.info("Running in Mock GCP mode. Spanner will use high-fidelity in-memory vector engine.")
            self._initialized = True
            return
        try:
            import google.auth
            from google.cloud import spanner
            credentials, _ = google.auth.default(quota_project_id=settings.GCP_PROJECT_ID)
            self._client = spanner.Client(project=settings.GCP_PROJECT_ID, credentials=credentials)
            self._instance = self._client.instance(settings.SPANNER_INSTANCE_ID)
            self._database = self._instance.database(settings.SPANNER_DATABASE_ID)
            logger.info(f"Connected to Cloud Spanner: {settings.SPANNER_INSTANCE_ID}/{settings.SPANNER_DATABASE_ID}")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to connect to Cloud Spanner ({e}). Falling back to local vector engine.")
            self._initialized = True

    def insert_or_update_video(self, video_data: Dict[str, Any]) -> bool:
        """
        Inserts or updates a video record and its 1408-dim vector embedding in Cloud Spanner.
        """
        self._init_client()
        now_iso = datetime.now(timezone.utc).isoformat()
        video_data["updated_at"] = now_iso
        if "created_at" not in video_data or not video_data["created_at"]:
            video_data["created_at"] = now_iso

        # Maintain local mirror cache
        self._local_db[video_data["video_id"]] = video_data
        self._save_local_db()

        if self._database and not settings.USE_MOCK_GCP:
            try:
                from google.cloud import spanner
                emb = video_data.get("embedding")
                if emb is not None:
                    emb = [float(x) for x in emb]

                with self._database.batch() as batch:
                    batch.insert_or_update(
                        table=settings.SPANNER_TABLE_NAME,
                        columns=[
                            "video_id", "title", "description", "tags",
                            "gcs_uri", "gcs_bucket", "gcs_object_name",
                            "content_type", "duration_seconds", "file_size_bytes",
                            "embedding", "embedding_model", "status",
                            "error_message", "created_at", "updated_at"
                        ],
                        values=[[
                            video_data["video_id"],
                            video_data["title"],
                            video_data.get("description", ""),
                            video_data.get("tags", []),
                            video_data["gcs_uri"],
                            video_data.get("gcs_bucket", settings.GCS_BUCKET_NAME),
                            video_data.get("gcs_object_name", f"videos/{video_data['video_id']}.mp4"),
                            video_data.get("content_type", "video/mp4"),
                            float(video_data.get("duration_seconds", 0.0)),
                            int(video_data.get("file_size_bytes", 0)),
                            emb,
                            video_data.get("embedding_model", settings.EMBEDDING_MODEL_NAME),
                            video_data.get("status", "INDEXED"),
                            video_data.get("error_message"),
                            spanner.COMMIT_TIMESTAMP,
                            spanner.COMMIT_TIMESTAMP
                        ]]
                    )
                logger.info(f"Persisted video {video_data['video_id']} with vector embedding to Cloud Spanner.")
                return True
            except Exception as e:
                logger.error(f"Cloud Spanner insert failed: {e}. Saved in local store.")
                return False
        return True

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single video by its ID."""
        self._init_client()
        if self._database and not settings.USE_MOCK_GCP:
            try:
                from google.cloud import spanner
                query = f"""
                    SELECT video_id, title, description, tags, gcs_uri, gcs_bucket, gcs_object_name,
                           content_type, duration_seconds, file_size_bytes, embedding, embedding_model, status,
                           error_message, created_at, updated_at
                    FROM {settings.SPANNER_TABLE_NAME}
                    WHERE video_id = @video_id
                """
                with self._database.snapshot() as snapshot:
                    results = snapshot.execute_sql(
                        query,
                        params={"video_id": video_id},
                        param_types={"video_id": spanner.param_types.STRING}
                    )
                    for row in results:
                        return {
                            "video_id": row[0],
                            "title": row[1],
                            "description": row[2],
                            "tags": row[3] or [],
                            "gcs_uri": row[4],
                            "gcs_bucket": row[5],
                            "gcs_object_name": row[6],
                            "content_type": row[7],
                            "duration_seconds": row[8],
                            "file_size_bytes": row[9],
                            "embedding": row[10],
                            "embedding_model": row[11],
                            "status": row[12],
                            "error_message": row[13],
                            "created_at": str(row[14]) if row[14] else None,
                            "updated_at": str(row[15]) if row[15] else None,
                            "video_url": self._local_db.get(video_id, {}).get("video_url")
                        }
            except Exception as e:
                logger.warning(f"Spanner get_video error ({e}), reading from local cache.")

        return self._local_db.get(video_id)

    def list_videos(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Lists all videos in Cloud Spanner."""
        self._init_client()
        if self._database and not settings.USE_MOCK_GCP:
            try:
                from google.cloud import spanner
                where_clause = "WHERE status = @status" if status else ""
                query = f"""
                    SELECT video_id, title, description, tags, gcs_uri, gcs_bucket, gcs_object_name,
                           content_type, duration_seconds, file_size_bytes, embedding, embedding_model, status,
                           error_message, created_at, updated_at
                    FROM {settings.SPANNER_TABLE_NAME}
                    {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT @limit
                """
                params = {"limit": limit}
                param_types = {"limit": spanner.param_types.INT64}
                if status:
                    params["status"] = status
                    param_types["status"] = spanner.param_types.STRING

                with self._database.snapshot() as snapshot:
                    results = snapshot.execute_sql(query, params=params, param_types=param_types)
                    spanner_videos = []
                    for row in results:
                        vid_id = row[0]
                        spanner_videos.append({
                            "video_id": vid_id,
                            "title": row[1],
                            "description": row[2],
                            "tags": row[3] or [],
                            "gcs_uri": row[4],
                            "gcs_bucket": row[5],
                            "gcs_object_name": row[6],
                            "content_type": row[7],
                            "duration_seconds": row[8],
                            "file_size_bytes": row[9],
                            "embedding": row[10],
                            "embedding_model": row[11],
                            "status": row[12],
                            "error_message": row[13],
                            "created_at": str(row[14]) if row[14] else None,
                            "updated_at": str(row[15]) if row[15] else None,
                            "video_url": self._local_db.get(vid_id, {}).get("video_url") or f"/api/storage/{row[6]}"
                        })
                    if spanner_videos:
                        return spanner_videos
            except Exception as e:
                logger.warning(f"Cloud Spanner list_videos error ({e}), reading from local cache.")

        videos = list(self._local_db.values())
        if status:
            videos = [v for v in videos if v.get("status") == status]
        videos.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return videos[:limit]

    def delete_video(self, video_id: str) -> bool:
        """Deletes video from Spanner and local cache."""
        self._init_client()
        if video_id in self._local_db:
            del self._local_db[video_id]
            self._save_local_db()

        if self._database and not settings.USE_MOCK_GCP:
            try:
                from google.cloud import spanner
                with self._database.batch() as batch:
                    batch.delete(
                        table=settings.SPANNER_TABLE_NAME,
                        keyset=spanner.KeySet(keys=[[video_id]])
                    )
                return True
            except Exception as e:
                logger.error(f"Cloud Spanner delete failed: {e}")
                return False
        return True

    @staticmethod
    def calibrate_similarity(sim: float) -> float:
        """
        Calibrated Multimodal Similarity Scoring for 1408-D Vertex AI Multimodal Embeddings:
        - In 1408-D multimodal vector space, cross-modality (text-to-video) dot products range:
          * Background/Unrelated: <= 0.12 (score: 5% - 40%)
          * Moderate relevance: 0.12 - 0.16 (score: 40% - 70%)
          * Strong relevance: 0.16 - 0.20 (score: 70% - 88%)
          * Top/Direct relevance: 0.20+ (score: 88% - 98%)
        """
        if sim <= 0.10:
            return max(0.05, round((max(0.0, sim) / 0.10) * 0.25, 4))
        elif sim <= 0.13:
            return round(0.25 + ((sim - 0.10) / 0.03) * 0.25, 4)
        elif sim <= 0.16:
            return round(0.50 + ((sim - 0.13) / 0.03) * 0.25, 4)
        elif sim <= 0.20:
            return round(0.75 + ((sim - 0.16) / 0.04) * 0.14, 4)
        else:
            return min(0.98, round(0.89 + min(0.09, ((sim - 0.20) / 0.06) * 0.09), 4))

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        min_similarity: float = 0.0,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Vector Similarity Search using Spanner's COSINE_DISTANCE.
        In Live Mode: Queries Cloud Spanner with COSINE_DISTANCE(embedding, @query_embedding).
        In Mock Mode: Computes exact cosine similarity with NumPy.
        """
        self._init_client()

        if self._database and not settings.USE_MOCK_GCP:
            try:
                from google.cloud.spanner_v1 import param_types
                # Query Cloud Spanner with exact cosine distance
                query = f"""
                    SELECT 
                        video_id, title, description, tags, gcs_uri, gcs_bucket, gcs_object_name,
                        content_type, duration_seconds, file_size_bytes, embedding_model, status,
                        created_at, updated_at,
                        COSINE_DISTANCE(embedding, @query_embedding) AS distance,
                        (1.0 - COSINE_DISTANCE(embedding, @query_embedding)) AS similarity_score
                    FROM {settings.SPANNER_TABLE_NAME}
                    WHERE status = 'INDEXED'
                    ORDER BY distance ASC
                    LIMIT @top_k
                """
                with self._database.snapshot() as snapshot:
                    results = snapshot.execute_sql(
                        query,
                        params={
                            "query_embedding": query_embedding,
                            "top_k": top_k
                        },
                        param_types={
                            "query_embedding": param_types.Array(param_types.FLOAT32),
                            "top_k": param_types.INT64
                        }
                    )
                    ranked = []
                    for row in results:
                        sim = float(row[15]) if row[15] is not None else 0.0
                        dist = float(row[14]) if row[14] is not None else 1.0
                        vid_id = row[0]
                        
                        norm_sim = self.calibrate_similarity(sim)
                        
                        if norm_sim >= min_similarity:
                            ranked.append({
                                "video_id": vid_id,
                                "title": row[1],
                                "description": row[2],
                                "tags": row[3] or [],
                                "gcs_uri": row[4],
                                "gcs_bucket": row[5],
                                "gcs_object_name": row[6],
                                "content_type": row[7],
                                "duration_seconds": row[8],
                                "file_size_bytes": row[9],
                                "embedding_model": row[10],
                                "status": row[11],
                                "created_at": str(row[12]),
                                "updated_at": str(row[13]),
                                "distance": round(dist, 4),
                                "similarity_score": round(norm_sim, 4),
                                "video_url": self._local_db.get(vid_id, {}).get("video_url") or f"/api/storage/{row[6]}"
                            })
                    if ranked:
                        return ranked
            except Exception as e:
                logger.warning(f"Cloud Spanner vector search error ({e}), using local vector engine.")

        # High-Fidelity Local Vector Search (NumPy)
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        results = []
        for vid, item in self._local_db.items():
            if item.get("status") != "INDEXED":
                continue
            
            # Tag filter
            if tags:
                item_tags = [t.lower() for t in item.get("tags", [])]
                if not any(t.lower() in item_tags for t in tags):
                    continue

            emb = item.get("embedding")
            if not emb:
                continue

            v_vec = np.array(emb, dtype=np.float32)
            v_norm = np.linalg.norm(v_vec)
            if v_norm > 0:
                v_vec = v_vec / v_norm
            
            # Cosine similarity
            cosine_sim = float(np.dot(q_vec, v_vec))
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            cosine_dist = 1.0 - cosine_sim
            normalized_sim = self.calibrate_similarity(cosine_sim)

            if normalized_sim >= min_similarity:
                res_item = dict(item)
                res_item["similarity_score"] = round(normalized_sim, 4)
                res_item["distance"] = round(cosine_dist, 4)
                if "embedding" in res_item:
                    res_item["embedding_preview"] = res_item["embedding"][:8]
                    del res_item["embedding"]
                results.append(res_item)

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> SpannerStats:
        """Calculates database statistics."""
        self._init_client()
        if self._database and not settings.USE_MOCK_GCP:
            try:
                query = f"""
                    SELECT 
                        COUNT(*) AS total,
                        COUNTIF(status = 'INDEXED') AS indexed,
                        COUNTIF(status = 'PROCESSING') AS processing,
                        COUNTIF(status = 'FAILED') AS failed,
                        COALESCE(SUM(file_size_bytes), 0) AS total_storage
                    FROM {settings.SPANNER_TABLE_NAME}
                """
                with self._database.snapshot() as snapshot:
                    results = snapshot.execute_sql(query)
                    for row in results:
                        return SpannerStats(
                            total_videos=row[0] or 0,
                            indexed_videos=row[1] or 0,
                            processing_videos=row[2] or 0,
                            failed_videos=row[3] or 0,
                            vector_dimension=settings.EMBEDDING_DIMENSION,
                            total_storage_bytes=row[4] or 0,
                            is_mock_mode=False,
                            spanner_instance=settings.SPANNER_INSTANCE_ID,
                            spanner_database=settings.SPANNER_DATABASE_ID
                        )
            except Exception as e:
                logger.warning(f"Spanner get_stats error: {e}")

        videos = list(self._local_db.values())
        total = len(videos)
        indexed = sum(1 for v in videos if v.get("status") == "INDEXED")
        processing = sum(1 for v in videos if v.get("status") == "PROCESSING")
        failed = sum(1 for v in videos if v.get("status") == "FAILED")
        storage_bytes = sum(v.get("file_size_bytes", 0) for v in videos)

        return SpannerStats(
            total_videos=total,
            indexed_videos=indexed,
            processing_videos=processing,
            failed_videos=failed,
            vector_dimension=settings.EMBEDDING_DIMENSION,
            total_storage_bytes=storage_bytes,
            is_mock_mode=settings.USE_MOCK_GCP,
            spanner_instance=settings.SPANNER_INSTANCE_ID,
            spanner_database=settings.SPANNER_DATABASE_ID
        )

spanner_service = SpannerService()

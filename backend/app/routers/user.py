import time
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from ..config import settings
from ..models.schemas import VideoSearchRequest, VideoSearchResponse, VideoMetadata
from ..services.embedding_service import embedding_service
from ..services.spanner_service import spanner_service
from ..services.gcs_service import gcs_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/user", tags=["User Facet"])

@router.post("/search", response_model=VideoSearchResponse)
async def search_videos(req: VideoSearchRequest):
    """
    User Endpoint: Semantic Video Search.
    1. Generates text query embedding via Vertex AI Multimodal Embedding model.
    2. Performs vector similarity search in Cloud Spanner using COSINE_DISTANCE.
    3. Ranks videos by cosine similarity score and produces fresh GCS playback URLs.
    """
    start_time = time.perf_counter()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    # 1. Generate text embedding vector (1408-dimensional)
    t_emb_start = time.perf_counter()
    logger.info(f"Generating query embedding for text: '{query}'")
    query_vector = embedding_service.generate_text_embedding(query)
    embedding_time_ms = round((time.perf_counter() - t_emb_start) * 1000, 2)

    # 2. Vector search in Cloud Spanner
    t_spanner_start = time.perf_counter()
    raw_results = spanner_service.vector_search(
        query_embedding=query_vector,
        top_k=req.top_k,
        min_similarity=req.min_similarity,
        tags=req.tags
    )
    spanner_time_ms = round((time.perf_counter() - t_spanner_start) * 1000, 2)

    # 3. Format results and generate direct GCS playback streaming URLs
    results: List[VideoMetadata] = []
    for idx, r in enumerate(raw_results):
        bucket_name = r.get("gcs_bucket", settings.GCS_BUCKET_NAME)
        object_name = r.get("gcs_object_name", f"videos/{r['video_id']}.mp4")
        
        # Direct Google Cloud Storage Streaming URL
        import re
        direct_gcs_object = re.sub(r'_\d{4}\.mp4$', '.mp4', object_name)
        gcs_direct_url = f"https://storage.googleapis.com/{bucket_name}/{direct_gcs_object}"
        playback_url = gcs_direct_url

        emb_preview = r.get("embedding_preview")
        if not emb_preview and "embedding" in r and isinstance(r["embedding"], list):
            emb_preview = r["embedding"][:8]

        results.append(VideoMetadata(
            video_id=r["video_id"],
            title=r["title"],
            description=r.get("description", ""),
            tags=r.get("tags", []),
            gcs_uri=r["gcs_uri"],
            gcs_bucket=r.get("gcs_bucket", settings.GCS_BUCKET_NAME),
            gcs_object_name=r.get("gcs_object_name", ""),
            content_type=r.get("content_type", "video/mp4"),
            duration_seconds=r.get("duration_seconds", 0.0),
            file_size_bytes=r.get("file_size_bytes", 0),
            embedding_model=r.get("embedding_model", settings.EMBEDDING_MODEL_NAME),
            embedding_preview=emb_preview,
            embedding_dimension=settings.EMBEDDING_DIMENSION,
            status=r.get("status", "INDEXED"),
            video_url=playback_url,
            similarity_score=r.get("similarity_score"),
            distance=r.get("distance"),
            created_at=r.get("created_at"),
            updated_at=r.get("updated_at")
        ))

    # 1. Parameterized SQL Query (Used by application client libraries)
    formatted_sql = f"""SELECT 
    video_id,
    title,
    description,
    tags,
    gcs_uri,
    duration_seconds,
    COSINE_DISTANCE(embedding, @query_embedding) AS distance,
    (1.0 - COSINE_DISTANCE(embedding, @query_embedding)) AS similarity_score
FROM {settings.SPANNER_TABLE_NAME}
WHERE status = 'INDEXED'
ORDER BY distance ASC
LIMIT @top_k;"""

    # 2. Standalone Literal SQL Query (Ready to copy-paste directly into Spanner Studio)
    array_literal = ", ".join(f"{x:.6f}" for x in query_vector)
    standalone_sql = f"""SELECT 
    video_id,
    title,
    description,
    tags,
    gcs_uri,
    duration_seconds,
    COSINE_DISTANCE(embedding, ARRAY<FLOAT32>[{array_literal}]) AS distance,
    (1.0 - COSINE_DISTANCE(embedding, ARRAY<FLOAT32>[{array_literal}])) AS similarity_score
FROM {settings.SPANNER_TABLE_NAME}
WHERE status = 'INDEXED'
ORDER BY distance ASC
LIMIT {req.top_k};"""

    query_params_info = {
        "@query_embedding": f"ARRAY<FLOAT32>({len(query_vector)} elements)",
        "@top_k": req.top_k,
        "database": f"{settings.SPANNER_INSTANCE_ID}/{settings.SPANNER_DATABASE_ID}",
        "instance": settings.SPANNER_INSTANCE_ID,
        "distance_function": "COSINE_DISTANCE",
        "dimension": len(query_vector),
        "spanner_latency_ms": spanner_time_ms,
        "embedding_latency_ms": embedding_time_ms
    }

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    logger.info(f"Search '{query}' returned {len(results)} matches in total={elapsed_ms}ms (Spanner DB={spanner_time_ms}ms, Vertex AI={embedding_time_ms}ms)")

    return VideoSearchResponse(
        query=query,
        total_found=len(results),
        execution_time_ms=elapsed_ms,
        spanner_latency_ms=spanner_time_ms,
        embedding_latency_ms=embedding_time_ms,
        spanner_sql_query=formatted_sql,
        spanner_standalone_sql=standalone_sql,
        query_vector_preview=query_vector[:8],
        full_query_embedding=query_vector,
        spanner_params=query_params_info,
        results=results
    )

@router.get("/videos/{video_id}", response_model=VideoMetadata)
def get_video_details(video_id: str):
    """User Endpoint: Retrieve single video metadata and stream URL."""
    video = spanner_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    playback_url = video.get("video_url")
    if not playback_url or playback_url.startswith("gs://"):
        playback_url = gcs_service.generate_signed_url(
            bucket_name=video.get("gcs_bucket", settings.GCS_BUCKET_NAME),
            object_name=video.get("gcs_object_name", "")
        )

    emb = video.get("embedding")
    preview = emb[:8] if (emb and isinstance(emb, list)) else None

    return VideoMetadata(
        video_id=video["video_id"],
        title=video["title"],
        description=video.get("description", ""),
        tags=video.get("tags", []),
        gcs_uri=video["gcs_uri"],
        gcs_bucket=video.get("gcs_bucket", settings.GCS_BUCKET_NAME),
        gcs_object_name=video.get("gcs_object_name", ""),
        content_type=video.get("content_type", "video/mp4"),
        duration_seconds=video.get("duration_seconds", 0.0),
        file_size_bytes=video.get("file_size_bytes", 0),
        embedding_model=video.get("embedding_model", settings.EMBEDDING_MODEL_NAME),
        embedding_preview=preview,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        status=video.get("status", "INDEXED"),
        video_url=playback_url,
        created_at=video.get("created_at"),
        updated_at=video.get("updated_at")
    )

@router.get("/tags", response_model=List[str])
def get_all_tags():
    """Returns unique list of tags across all indexed videos."""
    videos = spanner_service.list_videos(status="INDEXED")
    tags_set = set()
    for v in videos:
        for t in v.get("tags", []):
            if t:
                tags_set.add(t.strip())
    return sorted(list(tags_set))

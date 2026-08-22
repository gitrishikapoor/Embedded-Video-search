import uuid
import logging
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
from ..config import settings
from ..models.schemas import (
    VideoMetadata, VideoUploadResponse, AppConfig, SpannerStats, SeedSampleRequest
)
from ..services.gcs_service import gcs_service
from ..services.embedding_service import embedding_service
from ..services.spanner_service import spanner_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Admin Facet"])

async def process_video_ingestion(
    video_id: str,
    file_bytes: bytes,
    filename: str,
    title: str,
    description: str,
    tags: List[str],
    content_type: str,
    duration_seconds: float,
    custom_video_url: Optional[str] = None
):
    """
    Background Task Pipeline:
    1. Upload video file to Google Cloud Storage (GCS)
    2. Call Vertex AI to generate 1408-dim Multimodal Vector Embedding
    3. Persist metadata + vector embedding to Cloud Spanner database
    """
    try:
        logger.info(f"Starting ingestion pipeline for video_id={video_id}, title='{title}'")
        
        # 1. Upload to GCS and save local copy for fast keyframe processing
        local_vid_path = settings.STORAGE_DIR / "videos" / filename if hasattr(settings, "STORAGE_DIR") else Path(f"storage/videos/{filename}")
        local_vid_path.parent.mkdir(parents=True, exist_ok=True)
        if file_bytes and len(file_bytes) > 0:
            local_vid_path.write_bytes(file_bytes)
            gcs_uri, gcs_bucket, gcs_object, video_url = await gcs_service.upload_file(
                file_bytes, filename, content_type
            )
            file_size = len(file_bytes)
        else:
            # External or sample video
            gcs_uri = f"gs://{settings.GCS_BUCKET_NAME}/videos/{filename}"
            gcs_bucket = settings.GCS_BUCKET_NAME
            gcs_object = f"videos/{filename}"
            video_url = custom_video_url or f"/api/storage/videos/{filename}"
            file_size = 1024 * 1024 * 5  # Estimated 5MB

        # 2. Generate Multimodal Vector Embedding via Keyframe Visual Signature + 50:50 Fusion
        metadata_context = f"{title}. {description}. Tags: {', '.join(tags)}"
        vector_embedding = embedding_service.generate_video_embedding(
            gcs_uri=gcs_uri,
            metadata_context=metadata_context,
            local_video_path=local_vid_path
        )

        # 3. Store into Cloud Spanner
        record = {
            "video_id": video_id,
            "title": title,
            "description": description,
            "tags": tags,
            "gcs_uri": gcs_uri,
            "gcs_bucket": gcs_bucket,
            "gcs_object_name": gcs_object,
            "content_type": content_type,
            "duration_seconds": duration_seconds,
            "file_size_bytes": file_size,
            "embedding": vector_embedding,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "status": "INDEXED",
            "video_url": video_url,
            "error_message": None
        }
        spanner_service.insert_or_update_video(record)
        logger.info(f"Successfully indexed video_id={video_id} in Cloud Spanner.")
    except Exception as e:
        logger.error(f"Ingestion pipeline failed for video_id={video_id}: {e}")
        # Update Spanner record with FAILED status
        failed_record = {
            "video_id": video_id,
            "title": title,
            "description": description,
            "tags": tags,
            "gcs_uri": f"gs://{settings.GCS_BUCKET_NAME}/videos/{filename}",
            "gcs_bucket": settings.GCS_BUCKET_NAME,
            "gcs_object_name": f"videos/{filename}",
            "content_type": content_type,
            "status": "FAILED",
            "error_message": str(e)
        }
        spanner_service.insert_or_update_video(failed_record)

@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    description: Optional[str] = Form(""),
    tags: Optional[str] = Form(""),
    duration_seconds: Optional[float] = Form(10.0)
):
    """
    Admin Endpoint: Uploads a video file, generates GCS destination, and triggers the
    asynchronous Vertex AI Embedding + Cloud Spanner insertion background task.
    """
    video_id = str(uuid.uuid4())
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    filename = f"{video_id}_{file.filename}"
    
    file_bytes = await file.read()
    gcs_uri = f"gs://{settings.GCS_BUCKET_NAME}/videos/{filename}"

    # Initial placeholder record in Spanner
    initial_record = {
        "video_id": video_id,
        "title": title,
        "description": description,
        "tags": tag_list,
        "gcs_uri": gcs_uri,
        "gcs_bucket": settings.GCS_BUCKET_NAME,
        "gcs_object_name": f"videos/{filename}",
        "content_type": file.content_type or "video/mp4",
        "duration_seconds": duration_seconds,
        "file_size_bytes": len(file_bytes),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "status": "PROCESSING",
        "video_url": f"/api/storage/videos/{filename}"
    }
    spanner_service.insert_or_update_video(initial_record)

    # Launch background ingestion pipeline
    background_tasks.add_task(
        process_video_ingestion,
        video_id=video_id,
        file_bytes=file_bytes,
        filename=filename,
        title=title,
        description=description,
        tags=tag_list,
        content_type=file.content_type or "video/mp4",
        duration_seconds=duration_seconds
    )

    return VideoUploadResponse(
        video_id=video_id,
        title=title,
        status="PROCESSING",
        gcs_uri=gcs_uri,
        message="Video uploaded successfully. Vertex AI embedding generation and Spanner indexing started in background."
    )

@router.get("/videos", response_model=List[VideoMetadata])
def list_admin_videos(status: Optional[str] = None, limit: int = 100):
    """Admin Endpoint: List all videos stored in Cloud Spanner with their status and vector metadata."""
    records = spanner_service.list_videos(status=status, limit=limit)
    results = []
    for r in records:
        emb = r.get("embedding")
        preview = emb[:8] if (emb and isinstance(emb, list)) else None
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
            embedding_preview=preview,
            embedding_dimension=len(emb) if emb else settings.EMBEDDING_DIMENSION,
            status=r.get("status", "INDEXED"),
            error_message=r.get("error_message"),
            video_url=r.get("video_url"),
            created_at=r.get("created_at"),
            updated_at=r.get("updated_at")
        ))
    return results

@router.delete("/videos/{video_id}")
def delete_video(video_id: str):
    """Admin Endpoint: Delete video from Cloud Spanner and GCS."""
    video = spanner_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    gcs_service.delete_file(
        video.get("gcs_bucket", settings.GCS_BUCKET_NAME),
        video.get("gcs_object_name", f"videos/{video_id}.mp4")
    )
    spanner_service.delete_video(video_id)
    return {"status": "success", "message": f"Video {video_id} deleted successfully"}

@router.post("/reindex/{video_id}")
def reindex_video(video_id: str, background_tasks: BackgroundTasks):
    """Admin Endpoint: Recompute Vertex AI embeddings and update Cloud Spanner."""
    video = spanner_service.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    background_tasks.add_task(
        process_video_ingestion,
        video_id=video_id,
        file_bytes=b"",
        filename=video.get("gcs_object_name", "").replace("videos/", ""),
        title=video["title"],
        description=video.get("description", ""),
        tags=video.get("tags", []),
        content_type=video.get("content_type", "video/mp4"),
        duration_seconds=video.get("duration_seconds", 0.0),
        custom_video_url=video.get("video_url")
    )
    return {"status": "success", "message": f"Re-indexing scheduled for video {video_id}"}

@router.get("/stats", response_model=SpannerStats)
def get_spanner_statistics():
    """Admin Endpoint: Returns Cloud Spanner database statistics and vector search metrics."""
    return spanner_service.get_stats()

@router.get("/config", response_model=AppConfig)
def get_app_config():
    """Returns current GCP & environment settings."""
    return AppConfig(
        gcp_project_id=settings.GCP_PROJECT_ID,
        gcp_region=settings.GCP_REGION,
        gcs_bucket_name=settings.GCS_BUCKET_NAME,
        spanner_instance_id=settings.SPANNER_INSTANCE_ID,
        spanner_database_id=settings.SPANNER_DATABASE_ID,
        spanner_table_name=settings.SPANNER_TABLE_NAME,
        embedding_model_name=settings.EMBEDDING_MODEL_NAME,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        use_mock_gcp=settings.USE_MOCK_GCP
    )

@router.post("/config")
def update_app_config(config: AppConfig):
    """Updates runtime configuration."""
    settings.GCP_PROJECT_ID = config.gcp_project_id
    settings.GCP_REGION = config.gcp_region
    settings.GCS_BUCKET_NAME = config.gcs_bucket_name
    settings.SPANNER_INSTANCE_ID = config.spanner_instance_id
    settings.SPANNER_DATABASE_ID = config.spanner_database_id
    settings.USE_MOCK_GCP = config.use_mock_gcp
    return {"status": "success", "message": "Configuration updated"}

@router.post("/seed-samples")
async def seed_sample_videos(background_tasks: BackgroundTasks, req: Optional[SeedSampleRequest] = None):
    """
    Populates sample curated videos across various domains (Nature, Sports, Pets, Cooking, Astronomy, Tech)
    with direct playable video sources and vector embeddings for instant demonstration.
    """
    samples = [
        {
            "title": "Golden Retriever Puppy Playing with Ball in Green Grass",
            "description": "An adorable happy golden retriever puppy fetching a red rubber ball in a sunny park lawn, wagging its tail and running towards the camera.",
            "tags": ["dog", "puppy", "golden retriever", "pet", "animals", "cute"],
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
            "duration": 15.0,
            "filename": "sample_dog_puppy.mp4"
        },
        {
            "title": "Sunset Over Ocean Waves and Golden Sandy Beach",
            "description": "Cinematic aerial view of gentle ocean waves crashing against a golden sandy coastline during a vibrant orange and purple sunset.",
            "tags": ["ocean", "sea", "waves", "beach", "sunset", "nature", "water"],
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
            "duration": 30.0,
            "filename": "sample_ocean_sunset.mp4"
        },
        {
            "title": "High Speed Red Sports Car Drifting on Racetrack",
            "description": "High performance sports car drifting around hairpin curves on a wet asphalt racing circuit with smoking tires and high speed motion.",
            "tags": ["car", "sports car", "racing", "drifting", "speed", "vehicle"],
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
            "duration": 20.0,
            "filename": "sample_sports_car.mp4"
        },
        {
            "title": "Master Chef Preparing Homemade Italian Pasta Recipe",
            "description": "Professional culinary chef kneading fresh dough, rolling thin fettuccine pasta, and tossing in a simmering garlic tomato basil sauce in a gourmet kitchen.",
            "tags": ["cooking", "recipe", "food", "pasta", "chef", "kitchen", "delicious"],
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
            "duration": 25.0,
            "filename": "sample_cooking_pasta.mp4"
        },
        {
            "title": "Deep Space Nebula and Galaxy Stars Exploration",
            "description": "Spectacular high resolution astronomical journey through glowing cosmic nebulae, swirling spiral galaxies, and distant shining stars in deep space.",
            "tags": ["space", "galaxy", "stars", "universe", "astronomy", "nebula", "cosmic"],
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyBlazes.mp4",
            "duration": 18.0,
            "filename": "sample_deep_space.mp4"
        },
        {
            "title": "Rock Band Live Music Concert with Electric Guitar Solo",
            "description": "Energetic rock band performing on stage under laser stadium lights, featuring a passionate lead electric guitar solo and roaring crowd.",
            "tags": ["music", "concert", "guitar", "playing", "band", "stage", "performance"],
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4",
            "duration": 22.0,
            "filename": "sample_music_concert.mp4"
        },
        {
            "title": "Software Engineer Coding AI Applications on Multi-Monitor Setup",
            "description": "Software developer typing clean Python code, debugging cloud microservices, and monitoring neural network training dashboards on triple monitors.",
            "tags": ["coding", "programming", "developer", "software", "cloud", "ai", "technology"],
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
            "duration": 28.0,
            "filename": "sample_coding_developer.mp4"
        },
        {
            "title": "High Intensity Athlete Gym Workout and Fitness Training",
            "description": "Athlete performing intense barbell deadlifts, kettlebell swings, and sprint intervals in a modern training facility gym.",
            "tags": ["fitness", "gym", "workout", "running", "sports", "athlete"],
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
            "duration": 19.0,
            "filename": "sample_fitness_gym.mp4"
        }
    ]

    count = min(req.count if req else 8, len(samples))
    seeded_ids = []
    
    for item in samples[:count]:
        vid = str(uuid.uuid4())
        seeded_ids.append(vid)
        background_tasks.add_task(
            process_video_ingestion,
            video_id=vid,
            file_bytes=b"",
            filename=item["filename"],
            title=item["title"],
            description=item["description"],
            tags=item["tags"],
            content_type="video/mp4",
            duration_seconds=item["duration"],
            custom_video_url=item["url"]
        )

    return {
        "status": "success",
        "message": f"Seeded {count} sample videos into ingestion pipeline.",
        "video_ids": seeded_ids
    }

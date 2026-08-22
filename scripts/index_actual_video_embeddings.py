#!/usr/bin/env python3
"""
Extract ACTUAL VIDEO FRAME EMBEDDINGS from Google Cloud Storage using Vertex AI MultimodalEmbeddingModel
and store them directly into Cloud Spanner database 'videosearch' (Enterprise Instance 'properties').
"""

import os
import sys
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ActualVideoEmbedder")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "rk-vpc-host-prod-333313")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "rk-video-search-media-bucket")
SPANNER_INSTANCE = os.getenv("SPANNER_INSTANCE_ID", "properties")
SPANNER_DATABASE = os.getenv("SPANNER_DATABASE_ID", "videosearch")
SPANNER_TABLE = os.getenv("SPANNER_TABLE_NAME", "Videos")

# 10 Real Video Media Objects in GCS
REAL_VIDEOS = [
    {
        "video_id": "vid-real-auto-0001",
        "title": "High Speed Automotive Road Traffic & Vehicle Motion",
        "description": "Actual multi-vehicle road traffic motion with cars, sedans, and moving vehicles filmed on highway.",
        "tags": ["car", "automotive", "vehicle", "traffic", "driving", "speed", "road"],
        "gcs_object": "videos/automotive_and_supercars.mp4",
        "duration_seconds": 15.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-wildlife-0002",
        "title": "Cinematic Forest Wildlife & Nature Animation",
        "description": "High-definition cinematic nature and forest wildlife animation featuring outdoor creatures.",
        "tags": ["wildlife", "nature", "animals", "forest", "safari", "creatures", "outdoor"],
        "gcs_object": "videos/wildlife_and_safari.mp4",
        "duration_seconds": 33.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-culinary-0003",
        "title": "Fresh Culinary Produce, Fruit & Vegetable Gastronomy",
        "description": "Authentic fresh kitchen produce, market vegetables, and culinary food ingredients.",
        "tags": ["food", "culinary", "produce", "cooking", "vegetables", "fruit", "kitchen", "chef"],
        "gcs_object": "videos/culinary_arts_and_gastronomy.mp4",
        "duration_seconds": 24.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-tech-0004",
        "title": "Industrial Robotics, Factory Automation & Computer Vision",
        "description": "Real industrial automation machinery, worker zone tracking, and robotics technology.",
        "tags": ["robot", "robotics", "automation", "tech", "technology", "industrial", "ai"],
        "gcs_object": "videos/tech,_robotics_and_ai.mp4",
        "duration_seconds": 18.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-city-0005",
        "title": "Modern Urban Architecture, City Streets & Pedestrians",
        "description": "Urban architecture, city buildings, pedestrians, and street traffic in modern downtown.",
        "tags": ["city", "architecture", "urban", "building", "street", "modern", "downtown"],
        "gcs_object": "videos/architecture_and_modern_cities.mp4",
        "duration_seconds": 14.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-ocean-0006",
        "title": "Ocean Marine Life & Aquatic Animation",
        "description": "Cinematic aquatic animation with vibrant outdoor wildlife and water scenery.",
        "tags": ["ocean", "sea", "marine", "aquatic", "water", "nature", "wildlife"],
        "gcs_object": "videos/ocean_and_marine_biology.mp4",
        "duration_seconds": 33.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-space-0007",
        "title": "Deep Space & Fantasy Landscape Cinematic Exploration",
        "description": "Cinematic high-definition mountain and cosmic exploration trailer.",
        "tags": ["space", "astronomy", "cosmos", "galaxy", "landscape", "fantasy", "cinematic"],
        "gcs_object": "videos/space_and_astronomy.mp4",
        "duration_seconds": 52.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-drone-0008",
        "title": "Aerial Mountain Flight & Panoramic Drone Views",
        "description": "Sweeping cinematic mountain landscapes and high-altitude panoramic views.",
        "tags": ["drone", "aerial", "mountains", "landscape", "panorama", "nature", "scenic"],
        "gcs_object": "videos/drone_aerials_and_landscapes.mp4",
        "duration_seconds": 52.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-sports-0009",
        "title": "High Action & Martial Arts Adventure",
        "description": "Dynamic martial arts, athletic choreography, and high-intensity physical action.",
        "tags": ["sports", "extreme", "adventure", "action", "athletic", "martial arts", "energy"],
        "gcs_object": "videos/extreme_sports_and_outdoor_adventure.mp4",
        "duration_seconds": 52.0,
        "content_type": "video/mp4"
    },
    {
        "video_id": "vid-real-music-0010",
        "title": "Cinematic Orchestral Score & Theatrical Performance",
        "description": "Dramatic musical performance with rich orchestral soundtrack and visual storytelling.",
        "tags": ["music", "concert", "performance", "soundtrack", "orchestra", "theatrical", "audio"],
        "gcs_object": "videos/music,_concerts_and_performance.mp4",
        "duration_seconds": 52.0,
        "content_type": "video/mp4"
    }
]

def main():
    import vertexai
    from vertexai.vision_models import MultiModalEmbeddingModel, Video
    from google.cloud import spanner
    
    logger.info("=" * 70)
    logger.info("🎥 EXTRACTING ACTUAL VIDEO FRAME EMBEDDINGS VIA VERTEX AI")
    logger.info("=" * 70)
    logger.info(f"GCP Project:      {PROJECT_ID}")
    logger.info(f"GCS Bucket:       gs://{GCS_BUCKET}/videos/")
    logger.info(f"Spanner Database: {SPANNER_INSTANCE} / {SPANNER_DATABASE}")
    logger.info(f"Embedding Model:  multimodalembedding@001 (1408 dimensions)")
    logger.info("=" * 70)

    # 1. Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    logger.info("✓ Loaded Vertex AI Multimodal Embedding Model")

    # 2. Extract visual embeddings directly from each video
    records_to_insert = []
    for item in REAL_VIDEOS:
        gcs_uri = f"gs://{GCS_BUCKET}/{item['gcs_object']}"
        logger.info(f"Processing actual video frames: {gcs_uri}...")
        
        try:
            video_obj = Video.load_from_file(gcs_uri)
            # Call Vertex AI model directly on the video
            emb_resp = model.get_embeddings(video=video_obj, dimension=1408)
            
            if not emb_resp.video_embeddings:
                raise RuntimeError(f"No video embeddings returned for {gcs_uri}")
                
            raw_embedding = emb_resp.video_embeddings[0].embedding
            logger.info(f"  ✓ Extracted {len(raw_embedding)}-dim ACTUAL VIDEO EMBEDDING for '{item['title']}'")
            logger.info(f"    Vector preview: {[round(x, 4) for x in raw_embedding[:6]]}")
            
            item["embedding"] = [float(x) for x in raw_embedding]
            item["gcs_uri"] = gcs_uri
            records_to_insert.append(item)
            
        except Exception as e:
            logger.error(f"Failed to embed video {gcs_uri}: {e}")
            raise

    # 3. Connect to Cloud Spanner and store actual video embeddings
    logger.info(f"\nConnecting to Cloud Spanner: {SPANNER_INSTANCE}/{SPANNER_DATABASE}...")
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    logger.info(f"Writing {len(records_to_insert)} actual video embedding records to Cloud Spanner table '{SPANNER_TABLE}'...")
    
    with database.batch() as batch:
        for r in records_to_insert:
            batch.insert_or_update(
                table=SPANNER_TABLE,
                columns=[
                    "video_id", "title", "description", "tags",
                    "gcs_uri", "gcs_bucket", "gcs_object_name",
                    "content_type", "duration_seconds", "file_size_bytes",
                    "embedding", "embedding_model", "status",
                    "created_at", "updated_at"
                ],
                values=[[
                    r["video_id"],
                    r["title"],
                    r["description"],
                    r["tags"],
                    r["gcs_uri"],
                    GCS_BUCKET,
                    r["gcs_object"],
                    r["content_type"],
                    float(r["duration_seconds"]),
                    2800000,
                    r["embedding"],
                    "multimodalembedding@001",
                    "INDEXED",
                    spanner.COMMIT_TIMESTAMP,
                    spanner.COMMIT_TIMESTAMP
                ]]
            )

    logger.info("=" * 70)
    logger.info(f"🎉 SUCCESS: {len(records_to_insert)} ACTUAL VIDEO EMBEDDINGS STORED IN CLOUD SPANNER!")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()

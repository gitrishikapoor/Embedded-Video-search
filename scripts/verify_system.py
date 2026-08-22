#!/usr/bin/env python3
"""
Verification Script for Video Vector Search System
Tests ingestion, vector embedding generation, Cloud Spanner persistence, and ranked semantic retrieval.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from backend.app.services.embedding_service import embedding_service
from backend.app.services.spanner_service import spanner_service
from backend.app.services.gcs_service import gcs_service

def run_verification():
    print("=" * 70)
    print(" 🚀 VERIFYING VIDEO VECTOR SEARCH APPLICATION PIPELINE")
    print("=" * 70)
    print(f"• GCP Project:        {settings.GCP_PROJECT_ID}")
    print(f"• Cloud Spanner DB:   {settings.SPANNER_DATABASE_ID}")
    print(f"• GCS Bucket:         {settings.GCS_BUCKET_NAME}")
    print(f"• Multimodal Model:   {settings.EMBEDDING_MODEL_NAME}")
    print(f"• Vector Dimension:   {settings.EMBEDDING_DIMENSION}")
    print(f"• Mock Mode Active:   {settings.USE_MOCK_GCP}")
    print("-" * 70)

    # 1. Test Multimodal Embedding Generation
    print("\n[Step 1/4] Testing Vertex AI Multimodal Embedding Extraction...")
    sample_video_uri = f"gs://{settings.GCS_BUCKET_NAME}/videos/test_golden_retriever.mp4"
    video_context = "Golden Retriever puppy playing with a tennis ball in green grass lawn."
    video_emb = embedding_service.generate_video_embedding(sample_video_uri, metadata_context=video_context)
    
    assert len(video_emb) == settings.EMBEDDING_DIMENSION, f"Expected {settings.EMBEDDING_DIMENSION} dimensions, got {len(video_emb)}"
    print(f"✓ Generated {len(video_emb)}-dimensional video embedding vector.")
    print(f"  Sample dimensions: {video_emb[:5]}...")

    # 2. Test Cloud Spanner Video Record Persistence
    print("\n[Step 2/4] Testing Cloud Spanner Ingestion & Vector Persistence...")
    test_video_id = "test-video-puppy-001"
    record = {
        "video_id": test_video_id,
        "title": "Golden Retriever Puppy Playing with Tennis Ball",
        "description": "Adorable puppy running in the garden with high energy.",
        "tags": ["dog", "puppy", "golden retriever", "pet", "animals"],
        "gcs_uri": sample_video_uri,
        "gcs_bucket": settings.GCS_BUCKET_NAME,
        "gcs_object_name": "videos/test_golden_retriever.mp4",
        "content_type": "video/mp4",
        "duration_seconds": 12.5,
        "file_size_bytes": 1024 * 1024 * 3,
        "embedding": video_emb,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "status": "INDEXED",
        "video_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
    }
    success = spanner_service.insert_or_update_video(record)
    assert success, "Failed to insert into Spanner service"
    print(f"✓ Persisted video record & vector to Cloud Spanner (ID: {test_video_id}).")

    # Add a distractor video (e.g. Sports Car) to verify ranking selectivity
    car_video_id = "test-video-car-002"
    car_emb = embedding_service.generate_video_embedding(
        f"gs://{settings.GCS_BUCKET_NAME}/videos/sports_car.mp4",
        metadata_context="High performance sports car drifting on race track at maximum speed."
    )
    spanner_service.insert_or_update_video({
        "video_id": car_video_id,
        "title": "Sports Car High Speed Drifting",
        "description": "Formula racing car sliding around corners with tire smoke.",
        "tags": ["car", "racing", "speed", "drifting", "vehicle"],
        "gcs_uri": f"gs://{settings.GCS_BUCKET_NAME}/videos/sports_car.mp4",
        "gcs_bucket": settings.GCS_BUCKET_NAME,
        "gcs_object_name": "videos/sports_car.mp4",
        "content_type": "video/mp4",
        "duration_seconds": 20.0,
        "file_size_bytes": 1024 * 1024 * 8,
        "embedding": car_emb,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "status": "INDEXED",
        "video_url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4"
    })
    print(f"✓ Added second video record for vector contrast (ID: {car_video_id}).")

    # 3. Test Text Query Embedding & Cloud Spanner Vector Search
    print("\n[Step 3/4] Testing User Search: Text Query Embedding -> Spanner COSINE_DISTANCE Ranking...")
    search_query = "a playful puppy running outside with a toy"
    print(f"• User Search Query: \"{search_query}\"")
    
    query_start = time.perf_counter()
    query_vector = embedding_service.generate_text_embedding(search_query)
    ranked_results = spanner_service.vector_search(
        query_embedding=query_vector,
        top_k=5,
        min_similarity=0.0
    )
    query_latency_ms = (time.perf_counter() - query_start) * 1000

    print(f"✓ Vector Search completed in {query_latency_ms:.2f}ms. Total Results: {len(ranked_results)}")
    print("\nRanked Output from Cloud Spanner:")
    for idx, res in enumerate(ranked_results, start=1):
        print(f"  #{idx} [{res['similarity_score']*100:.1f}% Match | Dist: {res['distance']:.3f}] - {res['title']} ({res['gcs_uri']})")

    top_result = ranked_results[0]
    assert top_result["video_id"] == test_video_id, f"Expected top match to be {test_video_id}, got {top_result['video_id']}"
    print(f"✓ Verified: Top ranked video is correctly '{top_result['title']}' with {top_result['similarity_score']*100:.1f}% similarity!")

    # 4. Spanner Database Metrics
    print("\n[Step 4/4] Cloud Spanner Database Health & Metrics...")
    stats = spanner_service.get_stats()
    print(f"✓ Spanner Stats: Total={stats.total_videos}, Indexed={stats.indexed_videos}, Dimension={stats.vector_dimension}")

    print("\n" + "=" * 70)
    print(" 🎉 ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()

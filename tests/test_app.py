import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "spanner_instance" in data
    assert data["dimension"] == 1408

def test_seed_sample_videos():
    response = client.post("/api/admin/seed-samples", json={"count": 4})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["video_ids"]) == 4

def test_list_admin_videos():
    response = client.get("/api/admin/videos")
    assert response.status_code == 200
    videos = response.json()
    assert isinstance(videos, list)
    assert len(videos) >= 4

def test_vector_search_puppy():
    response = client.post("/api/user/search", json={
        "query": "golden retriever puppy playing with ball in grass",
        "top_k": 5,
        "min_similarity": 0.0
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] > 0
    assert "spanner_sql_query" in data
    assert "COSINE_DISTANCE" in data["spanner_sql_query"]
    results = data["results"]
    assert len(results) > 0
    top_video = results[0]
    assert "dog" in top_video["tags"] or "puppy" in top_video["tags"] or "Golden" in top_video["title"]
    assert top_video["similarity_score"] > 0.5
    assert top_video["distance"] is not None

def test_vector_search_ocean():
    response = client.post("/api/user/search", json={
        "query": "ocean sunset gentle waves on sandy beach",
        "top_k": 5,
        "min_similarity": 0.0
    })
    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] > 0
    assert "spanner_sql_query" in data
    top_video = data["results"][0]
    assert "ocean" in top_video["tags"] or "sunset" in top_video["tags"] or "Sunset" in top_video["title"]

def test_admin_stats():
    response = client.get("/api/admin/stats")
    assert response.status_code == 200
    stats = response.json()
    assert stats["total_videos"] >= 4
    assert stats["vector_dimension"] == 1408

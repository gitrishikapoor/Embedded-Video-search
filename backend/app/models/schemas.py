from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class VideoMetadata(BaseModel):
    video_id: str
    title: str
    description: Optional[str] = ""
    tags: List[str] = Field(default_factory=list)
    gcs_uri: str
    gcs_bucket: str
    gcs_object_name: str
    content_type: str = "video/mp4"
    duration_seconds: Optional[float] = 0.0
    file_size_bytes: Optional[int] = 0
    embedding_model: str = "multimodalembedding@001"
    embedding_preview: Optional[List[float]] = None
    embedding_dimension: Optional[int] = 1408
    status: str = "INDEXED"  # PROCESSING, INDEXED, FAILED
    error_message: Optional[str] = None
    video_url: Optional[str] = None
    similarity_score: Optional[float] = None
    distance: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class VideoUploadResponse(BaseModel):
    video_id: str
    title: str
    status: str
    gcs_uri: str
    message: str

class VideoSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=4, ge=1, le=50)
    min_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: Optional[List[str]] = None

class VideoSearchResponse(BaseModel):
    query: str
    total_found: int
    execution_time_ms: float
    spanner_latency_ms: Optional[float] = 0.0
    embedding_latency_ms: Optional[float] = 0.0
    spanner_sql_query: Optional[str] = None
    spanner_standalone_sql: Optional[str] = None
    query_vector_preview: Optional[List[float]] = None
    full_query_embedding: Optional[List[float]] = None
    spanner_params: Optional[dict] = None
    results: List[VideoMetadata]

class AppConfig(BaseModel):
    gcp_project_id: str
    gcp_region: str
    gcs_bucket_name: str
    spanner_instance_id: str
    spanner_database_id: str
    spanner_table_name: str
    embedding_model_name: str
    embedding_dimension: int
    use_mock_gcp: bool

class SpannerStats(BaseModel):
    total_videos: int
    indexed_videos: int
    processing_videos: int
    failed_videos: int
    vector_dimension: int
    total_storage_bytes: int
    is_mock_mode: bool
    spanner_instance: str
    spanner_database: str

class SeedSampleRequest(BaseModel):
    count: Optional[int] = 8

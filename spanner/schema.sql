-- ==============================================================================
-- Cloud Spanner DDL Schema for Video Vector Search System
-- ==============================================================================

CREATE TABLE Videos (
    video_id STRING(64) NOT NULL,
    title STRING(256) NOT NULL,
    description STRING(MAX),
    tags ARRAY<STRING(64)>,
    gcs_uri STRING(1024) NOT NULL,
    gcs_bucket STRING(256) NOT NULL,
    gcs_object_name STRING(512) NOT NULL,
    content_type STRING(64) NOT NULL,
    duration_seconds FLOAT64,
    file_size_bytes INT64,
    embedding ARRAY<FLOAT32>(vector_length => 1408),
    embedding_model STRING(64) NOT NULL,
    status STRING(32) NOT NULL,
    error_message STRING(MAX),
    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
    updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
) PRIMARY KEY (video_id);

CREATE INDEX Idx_Videos_Status_CreatedAt ON Videos(status, created_at DESC);

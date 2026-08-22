# Video Vector Search Application (Admin & User Facets)

An intelligent, full-stack video retrieval platform built on **Google Cloud Storage (GCS)**, **Vertex AI Multimodal Embeddings**, and **Google Cloud Spanner Vector Search**.

---

## 🌟 Application Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             ADMIN FACET                                     │
│                                                                             │
│  [Video File] ──► Upload to GCS (gs://bucket/videos/...)                    │
│                        │                                                    │
│                        ▼                                                    │
│           Vertex AI Multimodal Embedding Model                              │
│              (multimodalembedding@001)                                      │
│                        │                                                    │
│                        ▼                                                    │
│           Extract 1408-dim Vector Float Array                               │
│                        │                                                    │
│                        ▼                                                    │
│           Persist into Cloud Spanner Table `Videos`                         │
│           (video_id, title, gcs_uri, ARRAY<FLOAT32>(1408), status)          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                             USER FACET                                      │
│                                                                             │
│  [Text Query: "golden retriever playing with ball"]                         │
│                        │                                                    │
│                        ▼                                                    │
│           Vertex AI Multimodal Embedding Model                              │
│              (multimodalembedding@001)                                      │
│                        │                                                    │
│                        ▼                                                    │
│           Generate 1408-dim Text Vector Embedding                           │
│                        │                                                    │
│                        ▼                                                    │
│           Cloud Spanner Vector Search Query:                                │
│           SELECT video_id, title, gcs_uri,                                  │
│                  COSINE_DISTANCE(embedding, @query_vector) AS distance,     │
│                  (1.0 - COSINE_DISTANCE(...)) AS similarity_score           │
│           FROM Videos WHERE status = 'INDEXED'                              │
│           ORDER BY distance ASC LIMIT 10;                                   │
│                        │                                                    │
│                        ▼                                                    │
│           Generate GCS Streaming Signed URLs & Rank Videos                  │
│                        │                                                    │
│                        ▼                                                    │
│           Render Interactive Ranked Video Feed & HTML5 Player               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

### 1. Admin Facet (Video Ingestion & Cataloging)
- **Drag & Drop Video Uploader**: Supports `.mp4`, `.webm`, and `.mov` video files.
- **Automated Ingestion Pipeline**:
  - Asynchronously uploads raw video to designated GCS Bucket.
  - Automatically extracts 1408-dimensional multimodal embeddings using Vertex AI.
  - Inserts video record, metadata, tags, and vector embedding into Cloud Spanner database.
- **Spanner Inventory Inspector**: View all indexed videos, inspect vector embeddings (first 8 dimensions preview), trigger on-demand re-indexing, or delete videos.
- **One-Click Demo Seeder**: Populates curated sample videos across categories (Nature, Sports, Pets, Cooking, Astronomy, Music, Tech) for instant exploration.

### 2. User Facet (Semantic Natural Language Search)
- **Natural Language Search Bar**: Search for any scene or action without requiring exact keyword matches (e.g. `"golden retriever puppy"`, `"car drifting at high speed"`, `"chef rolling fresh pasta"`).
- **Sub-Second Vector Search**: Queries Cloud Spanner utilizing high-speed vector distance calculations.
- **Relevance Ranking & Visual Badges**:
  - Displays match rank (`#1 Top Match`, `#2`, etc.).
  - Color-coded Cosine Similarity percentage badge (`95% Match`).
  - Cosine distance metric breakdown.
- **Direct Video Streaming**: Plays video clips inline in an embedded modal player using GCS signed URLs.
- **Interactive Threshold & Result Filters**: Adjust minimum similarity cutoff slider and top-k result limit.

---

## 🗄️ Cloud Spanner Vector Schema (DDL)

```sql
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
    status STRING(32) NOT NULL, -- 'PROCESSING', 'INDEXED', 'FAILED'
    error_message STRING(MAX),
    created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
    updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
) PRIMARY KEY (video_id);

CREATE INDEX Idx_Videos_Status_CreatedAt ON Videos(status, created_at DESC);
```

## 🛠️ Getting Started & Deployment

### 1. Clone the Repository
```bash
git clone https://github.com/gitrishikapoor/Embedded-Video-search.git
cd Embedded-Video-search
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory with the following variables:
```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GCS_BUCKET_NAME=your-gcs-bucket-name
SPANNER_INSTANCE_ID=your-spanner-instance
SPANNER_DATABASE_ID=your-spanner-database
USE_MOCK_GCP=false
```

### 2. Provision Infrastructure with Terraform
```bash
cd terraform
terraform init
terraform apply -var="project_id=your-gcp-project-id"
```

### 3. Initialize Cloud Spanner Database & Schema
```bash
python3 spanner/spanner_setup.py
```

---

## 📡 REST API Reference

### User Search Endpoints
- `POST /api/user/search`: Execute semantic video search.
  ```json
  {
    "query": "golden retriever puppy playing with ball",
    "top_k": 8,
    "min_similarity": 0.2
  }
  ```
- `GET /api/user/videos/{video_id}`: Retrieve single video metadata and stream URL.
- `GET /api/user/tags`: List all available video tags.

### Admin Ingestion Endpoints
- `POST /api/admin/upload`: Upload video file + metadata (`multipart/form-data`).
- `GET /api/admin/videos`: List all stored video records and vector previews.
- `DELETE /api/admin/videos/{video_id}`: Delete video from GCS and Spanner.
- `POST /api/admin/reindex/{video_id}`: Re-trigger Vertex AI embedding generation.
- `GET /api/admin/stats`: Get Cloud Spanner database and vector storage metrics.
- `POST /api/admin/seed-samples`: Seed sample demo videos.

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.20.0"
    }
  }
}

variable "project_id" {
  type        = string
  description = "The Google Cloud project ID"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = "The GCP region for resources"
}

variable "bucket_name" {
  type        = string
  default     = "video-vector-search-media-bucket"
  description = "Google Cloud Storage bucket for video uploads"
}

variable "spanner_instance_id" {
  type        = string
  default     = "video-search-instance"
  description = "Cloud Spanner instance ID"
}

variable "spanner_database_id" {
  type        = string
  default     = "video-search-db"
  description = "Cloud Spanner database ID"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required GCP APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "spanner.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "run.googleapis.com",
    "artifactregistry.googleapis.com"
  ])
  service            = each.key
  disable_on_destroy = false
}

# 2. Google Cloud Storage Bucket for Video Files
resource "google_storage_bucket" "video_bucket" {
  name          = var.bucket_name
  location      = var.region
  force_destroy = false
  uniform_bucket_level_access = true

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  depends_on = [google_project_service.apis]
}

# 3. Cloud Spanner Instance
resource "google_spanner_instance" "spanner_instance" {
  name         = var.spanner_instance_id
  config       = "regional-${var.region}"
  display_name = "Video Vector Search Spanner Instance"
  num_nodes    = 1

  depends_on = [google_project_service.apis]
}

# 4. Cloud Spanner Database with Vector DDL Schema
resource "google_spanner_database" "spanner_db" {
  instance = google_spanner_instance.spanner_instance.name
  name     = var.spanner_database_id

  ddl = [
    <<-EOT
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
        embedding ARRAY<FLOAT32>(1408),
        embedding_model STRING(64) NOT NULL,
        status STRING(32) NOT NULL,
        error_message STRING(MAX),
        created_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true),
        updated_at TIMESTAMP NOT NULL OPTIONS (allow_commit_timestamp=true)
    ) PRIMARY KEY (video_id)
    EOT
    ,
    "CREATE INDEX Idx_Videos_Status_CreatedAt ON Videos(status, created_at DESC)"
  ]

  deletion_protection = false
}

# 5. Service Account for Video Search Application
resource "google_service_account" "app_sa" {
  account_id   = "video-search-app-sa"
  display_name = "Video Vector Search App Service Account"
}

# IAM Permissions: Spanner Database User
resource "google_spanner_database_iam_member" "spanner_user" {
  instance = google_spanner_instance.spanner_instance.name
  database = google_spanner_database.spanner_db.name
  role     = "roles/spanner.databaseUser"
  member   = "serviceAccount:${google_service_account.app_sa.email}"
}

# IAM Permissions: Cloud Storage Object Admin
resource "google_storage_bucket_iam_member" "storage_admin" {
  bucket = google_storage_bucket.video_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.app_sa.email}"
}

# IAM Permissions: Vertex AI User
resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.app_sa.email}"
}

output "gcs_bucket_name" {
  value = google_storage_bucket.video_bucket.name
}

output "spanner_instance" {
  value = google_spanner_instance.spanner_instance.name
}

output "spanner_database" {
  value = google_spanner_database.spanner_db.name
}

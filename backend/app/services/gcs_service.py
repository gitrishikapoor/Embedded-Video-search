import os
import logging
import datetime
from pathlib import Path
from typing import Tuple, Optional
from ..config import settings, STORAGE_DIR

logger = logging.getLogger(__name__)

VIDEOS_DIR = STORAGE_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

class GCSService:
    def __init__(self):
        self._client = None
        self._bucket = None
        self._initialized = False

    def _init_client(self):
        if self._initialized:
            return
        if settings.USE_MOCK_GCP:
            logger.info("Running in Mock GCP mode. GCS operations will use local filesystem storage.")
            self._initialized = True
            return
        try:
            from google.cloud import storage
            self._client = storage.Client(project=settings.GCP_PROJECT_ID)
            self._bucket = self._client.bucket(settings.GCS_BUCKET_NAME)
            logger.info(f"Connected to GCS bucket: {settings.GCS_BUCKET_NAME}")
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to connect to Google Cloud Storage ({e}). Falling back to local storage.")
            self._initialized = True

    async def upload_file(
        self, file_content: bytes, filename: str, content_type: str = "video/mp4"
    ) -> Tuple[str, str, str, str]:
        """
        Uploads a video to GCS (or local simulated storage).
        Returns (gcs_uri, gcs_bucket, gcs_object_name, public_or_local_url).
        """
        self._init_client()
        object_name = f"videos/{filename}"
        
        # Always write to local storage as well for fast local preview
        local_path = VIDEOS_DIR / filename
        with open(local_path, "wb") as f:
            f.write(file_content)

        if self._client and not settings.USE_MOCK_GCP:
            try:
                blob = self._bucket.blob(object_name)
                blob.upload_from_string(file_content, content_type=content_type)
                gcs_uri = f"gs://{settings.GCS_BUCKET_NAME}/{object_name}"
                signed_url = self.generate_signed_url(settings.GCS_BUCKET_NAME, object_name)
                logger.info(f"Uploaded {filename} to {gcs_uri}")
                return gcs_uri, settings.GCS_BUCKET_NAME, object_name, signed_url
            except Exception as e:
                logger.error(f"GCS upload failed ({e}), using local fallback.")

        # Local fallback representation
        gcs_uri = f"gs://{settings.GCS_BUCKET_NAME}/{object_name}"
        local_url = f"/api/storage/videos/{filename}"
        return gcs_uri, settings.GCS_BUCKET_NAME, object_name, local_url

    def generate_signed_url(
        self, bucket_name: str, object_name: str, expiration_minutes: int = 120
    ) -> str:
        """
        Generates a v4 signed URL for streaming or downloading the GCS video.
        """
        self._init_client()
        if self._client and not settings.USE_MOCK_GCP:
            try:
                bucket = self._client.bucket(bucket_name)
                blob = bucket.blob(object_name)
                url = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(minutes=expiration_minutes),
                    method="GET"
                )
                return url
            except Exception as e:
                logger.warning(f"Could not generate signed URL ({e}).")

        filename = os.path.basename(object_name)
        return f"/api/storage/videos/{filename}"

    def delete_file(self, bucket_name: str, object_name: str) -> bool:
        """Deletes a file from GCS and local cache."""
        self._init_client()
        filename = os.path.basename(object_name)
        local_path = VIDEOS_DIR / filename
        if local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass

        if self._client and not settings.USE_MOCK_GCP:
            try:
                bucket = self._client.bucket(bucket_name)
                blob = bucket.blob(object_name)
                blob.delete()
                return True
            except Exception as e:
                logger.error(f"Failed to delete GCS object ({e})")
                return False
        return True

gcs_service = GCSService()

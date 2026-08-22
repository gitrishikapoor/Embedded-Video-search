import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    load_dotenv()

STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Video Vector Search AI"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = True
    
    # GCP Infrastructure Settings
    GCP_PROJECT_ID: str = os.getenv("GOOGLE_CLOUD_PROJECT", "rk-vpc-host-prod-333313")
    GCP_REGION: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-west1")
    
    # Cloud Storage Settings
    GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "rk-video-search-media-bucket")
    
    # Cloud Spanner Settings (Enterprise Edition: Property Graph Database)
    SPANNER_INSTANCE_ID: str = os.getenv("SPANNER_INSTANCE_ID", "properties")
    SPANNER_DATABASE_ID: str = os.getenv("SPANNER_DATABASE_ID", "videosearch")
    SPANNER_TABLE_NAME: str = "Videos"
    
    # Vertex AI Multimodal Embeddings Settings
    EMBEDDING_MODEL_NAME: str = "multimodalembedding@001"
    EMBEDDING_DIMENSION: int = 1408  # 1408 dimensions
    
    # Execution Mode: Live Cloud Spanner by default
    USE_MOCK_GCP: bool = os.getenv("USE_MOCK_GCP", "false").lower() in ("true", "1", "yes")
    
    # Local Storage Directory
    LOCAL_STORAGE_PATH: str = str(STORAGE_DIR)

    model_config = SettingsConfigDict(env_file=str(ENV_FILE), extra="allow")

settings = Settings()

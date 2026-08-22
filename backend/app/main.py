import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from .config import settings, STORAGE_DIR
from .routers import admin, user
from .services.spanner_service import spanner_service

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR = STORAGE_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multimodal Video Vector Search Engine using Google Cloud Storage, Vertex AI, and Cloud Spanner Vector Search"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(admin.router, prefix=settings.API_PREFIX)
app.include_router(user.router, prefix=settings.API_PREFIX)

@app.get("/api/storage/videos/{filename:path}")
async def get_storage_video(filename: str):
    """Serves video files directly from local storage or redirects to GCS."""
    from pathlib import Path
    from fastapi.responses import FileResponse, RedirectResponse

    # 1. Direct match in storage/videos/
    file_path = VIDEOS_DIR / filename
    if file_path.is_file():
        return FileResponse(path=str(file_path), media_type="video/mp4", filename=Path(filename).name)

    # 2. Check in procedural_100 or distinct_100 subfolders
    simple_name = Path(filename).name
    for subdir in ["procedural_100", "distinct_100"]:
        sub_path = VIDEOS_DIR / subdir / simple_name
        if sub_path.is_file():
            return FileResponse(path=str(sub_path), media_type="video/mp4", filename=simple_name)

    # 3. Direct GCS fallback redirect
    gcs_url = f"https://storage.googleapis.com/{settings.GCS_BUCKET_NAME}/videos/distinct_100/{simple_name}"
    return RedirectResponse(url=gcs_url)

@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "mode": "Mock GCP Mode" if settings.USE_MOCK_GCP else "Live Google Cloud Mode",
        "spanner_instance": settings.SPANNER_INSTANCE_ID,
        "spanner_database": settings.SPANNER_DATABASE_ID,
        "gcs_bucket": settings.GCS_BUCKET_NAME,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "dimension": settings.EMBEDDING_DIMENSION
    }

# Mount static web frontend assets
if (STATIC_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "assets")), name="assets") if (STATIC_DIR / "assets").exists() else None

@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({
        "message": f"Welcome to {settings.APP_NAME}. Please visit /docs for API documentation."
    })

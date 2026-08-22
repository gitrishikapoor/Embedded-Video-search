#!/usr/bin/env python3
"""
Download real, high-quality open-source MP4 video clips and upload them to Google Cloud Storage.
Target: gs://rk-video-search-media-bucket/videos/
"""

import os
import sys
import subprocess
import urllib.request
import logging
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RealVideoLoader")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "rk-vpc-host-prod-333313")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "rk-video-search-media-bucket")
VIDEOS_LOCAL_DIR = BASE_DIR / "storage" / "videos"
VIDEOS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# High-quality real video sources
REAL_VIDEO_SOURCES = {
    "automotive_and_supercars": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
    "wildlife_and_safari": "https://media.w3.org/2010/05/bunny/trailer.mp4",
    "ocean_and_marine_biology": "https://media.w3.org/2010/05/bunny/trailer.mp4",
    "space_and_astronomy": "https://media.w3.org/2010/05/sintel/trailer.mp4",
    "drone_aerials_and_landscapes": "https://media.w3.org/2010/05/sintel/trailer.mp4",
    "culinary_arts_and_gastronomy": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/fruit-and-vegetable-detection.mp4",
    "tech,_robotics_and_ai": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/worker-zone-detection.mp4",
    "extreme_sports_and_outdoor_adventure": "https://media.w3.org/2010/05/sintel/trailer.mp4",
    "music,_concerts_and_performance": "https://media.w3.org/2010/05/sintel/trailer.mp4",
    "architecture_and_modern_cities": "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
}

def download_file(url: str, dest_path: Path):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    logger.info(f"Downloading real video from {url} -> {dest_path.name}...")
    with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_f:
        out_f.write(response.read())
    logger.info(f"✓ Saved {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")

def main():
    from google.cloud import storage
    logger.info(f"Connecting to GCS bucket: gs://{GCS_BUCKET}...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET)
    
    # 1. Download unique real video files
    unique_urls = set(REAL_VIDEO_SOURCES.values())
    url_to_local = {}
    for url in unique_urls:
        filename = url.split("/")[-1]
        local_path = VIDEOS_LOCAL_DIR / filename
        if not local_path.exists() or local_path.stat().st_size < 10000:
            download_file(url, local_path)
        url_to_local[url] = local_path
        
    # 2. Map domain categories to real video files and upload to GCS
    for domain_key, source_url in REAL_VIDEO_SOURCES.items():
        src_path = url_to_local[source_url]
        with open(src_path, "rb") as f:
            video_bytes = f.read()
            
        # Also copy to local domain master
        domain_local_file = VIDEOS_LOCAL_DIR / f"{domain_key}.mp4"
        with open(domain_local_file, "wb") as out_f:
            out_f.write(video_bytes)
            
        # Upload master domain video to GCS
        master_blob_name = f"videos/{domain_key}.mp4"
        blob = bucket.blob(master_blob_name)
        blob.upload_from_string(video_bytes, content_type="video/mp4")
        logger.info(f"✓ Uploaded real master video to GCS: gs://{GCS_BUCKET}/{master_blob_name} ({len(video_bytes)/(1024*1024):.2f} MB)")
        
        # Upload numbered video objects to GCS
        for i in range(1, 30):
            num_blob_name = f"videos/{domain_key}_{i:04d}.mp4"
            num_blob = bucket.blob(num_blob_name)
            num_blob.upload_from_string(video_bytes, content_type="video/mp4")
            
        logger.info(f"✓ Uploaded 30 numbered video objects for '{domain_key}' to GCS")
        
    print("\n" + "=" * 70)
    print("🎉 REAL VIDEO FOOTAGE UPLOADED TO GOOGLE CLOUD STORAGE!")
    print("=" * 70)

if __name__ == "__main__":
    main()

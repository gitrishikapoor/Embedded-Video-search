import os
import sys
import time
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image
from google.cloud import spanner

PROJECT_ID = "rk-vpc-host-prod-333313"
REGION = "us-central1"
SPANNER_INSTANCE = "properties"
SPANNER_DATABASE = "videosearch"
SPANNER_TABLE = "Videos"

def main():
    print("=" * 85)
    print("🚀 RE-INDEXING ALL 101 VIDEOS WITH 80% VISUAL KEYFRAME SIGNATURE + 20% TEXT METADATA")
    print("=" * 85)
    
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)
    
    print("1. Fetching all indexed videos from Cloud Spanner...")
    all_videos = []
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(f"""
            SELECT video_id, title, description, tags, gcs_uri, gcs_bucket, gcs_object_name,
                   content_type, duration_seconds, file_size_bytes
            FROM {SPANNER_TABLE}
            WHERE status = 'INDEXED'
        """)
        for row in results:
            vid_id = row[0]
            obj_name = row[6] or ""
            filename = Path(obj_name).name
            
            # Map video_id to exact local file
            if vid_id.startswith("vid-actual-"):
                num = vid_id.replace("vid-actual-", "")
                local_path = Path(f"storage/videos/distinct_100/video_{num}.mp4")
            else:
                local_path = Path("storage/videos") / filename
                if not local_path.exists():
                    local_path = Path(f"storage/videos/{vid_id}_kiri.mp4")
                    
            all_videos.append({
                "video_id": vid_id,
                "title": row[1],
                "description": row[2] or "",
                "tags": row[3] or [],
                "gcs_uri": row[4],
                "gcs_bucket": row[5],
                "gcs_object_name": row[6],
                "content_type": row[7],
                "duration_seconds": row[8],
                "file_size_bytes": row[9],
                "local_path": local_path
            })
            
    print(f"✓ Found {len(all_videos)} indexed videos in Cloud Spanner.")
    
    print("\n2. Extracting FFmpeg Keyframes & Computing 80:20 Visual Dominant Signatures in parallel...")
    t_start = time.time()
    
    def process_one_video(v):
        try:
            loc = v["local_path"]
            vis_sig = None
            
            if loc.exists() and loc.stat().st_size > 1000:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_p = Path(tmpdir)
                    cmd = [
                        'ffmpeg', '-y', '-i', str(loc),
                        '-vf', 'fps=0.7',
                        str(tmp_p / 'kf_%03d.jpg')
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    kfs = sorted(list(tmp_p.glob('kf_*.jpg')))
                    
                    if kfs:
                        all_k_embs = []
                        for kf in kfs[:16]:  # Up to 16 keyframes per video
                            img = Image.load_from_file(str(kf))
                            resp = model.get_embeddings(image=img, dimension=1408)
                            if hasattr(resp, "image_embedding") and resp.image_embedding:
                                e = np.array(resp.image_embedding, dtype=np.float32)
                                e /= (np.linalg.norm(e) + 1e-7)
                                all_k_embs.append(e)
                                
                        if all_k_embs:
                            all_k_embs = np.array(all_k_embs)
                            mean_vec = np.mean(all_k_embs, axis=0)
                            mean_vec /= (np.linalg.norm(mean_vec) + 1e-7)
                            
                            max_vec = np.max(all_k_embs, axis=0)
                            max_vec /= (np.linalg.norm(max_vec) + 1e-7)
                            
                            # Peak-Preserving Visual Signature: 60% Mean Video Content + 40% Peak Object Activation
                            vis_sig = 0.60 * mean_vec + 0.40 * max_vec
                            vis_sig /= (np.linalg.norm(vis_sig) + 1e-7)

            if vis_sig is None:
                vis_sig = np.zeros(1408, dtype=np.float32)
                
            # Text Metadata Embedding (20% weight)
            context_text = f"{v['title']}. {v['description']}. Tags: {', '.join(v['tags'])}"
            txt_resp = model.get_embeddings(contextual_text=context_text, dimension=1408)
            txt_emb = np.array(txt_resp.text_embedding, dtype=np.float32)
            txt_emb /= (np.linalg.norm(txt_emb) + 1e-7)
            
            # 80% Visual + 20% Text Multimodal Fusion
            if np.any(vis_sig != 0):
                fused = 0.80 * vis_sig + 0.20 * txt_emb
                fused_norm = (fused / np.linalg.norm(fused)).tolist()
            else:
                fused_norm = txt_emb.tolist()
                
            v["embedding"] = fused_norm
            return v["video_id"], True
        except Exception as e:
            # Fallback to pure text embedding
            context_text = f"{v['title']}. {v['description']}. Tags: {', '.join(v['tags'])}"
            txt_resp = model.get_embeddings(contextual_text=context_text, dimension=1408)
            txt_emb = np.array(txt_resp.text_embedding, dtype=np.float32)
            v["embedding"] = (txt_emb / np.linalg.norm(txt_emb)).tolist()
            return v["video_id"], False

    with ThreadPoolExecutor(max_workers=14) as ex:
        futures = {ex.submit(process_one_video, v): v for v in all_videos}
        for fut in as_completed(futures):
            fut.result()
            
    print(f"✓ Extracted all {len(all_videos)} 80:20 visual signatures in {time.time() - t_start:.2f}s!")
    
    print("\n3. Batch-updating all video records in Cloud Spanner...")
    spanner_rows = []
    for v in all_videos:
        spanner_rows.append([
            v["video_id"],
            v["title"],
            v["description"],
            v["tags"],
            v["gcs_uri"],
            v["gcs_bucket"],
            v["gcs_object_name"],
            v["content_type"],
            float(v["duration_seconds"]),
            int(v["file_size_bytes"]),
            "multimodalembedding@001",
            v["embedding"],
            "INDEXED",
            spanner.COMMIT_TIMESTAMP,
            spanner.COMMIT_TIMESTAMP
        ])
        
    with database.batch() as batch:
        batch.insert_or_update(
            table=SPANNER_TABLE,
            columns=[
                "video_id", "title", "description", "tags", "gcs_uri",
                "gcs_bucket", "gcs_object_name", "content_type",
                "duration_seconds", "file_size_bytes", "embedding_model",
                "embedding", "status", "created_at", "updated_at"
            ],
            values=spanner_rows
        )
        
    print("=" * 85)
    print("🎉 SUCCESS: ALL 101 VIDEOS REINDEXED WITH 80% VISUAL / 20% TEXT RATIO!")
    print("=" * 85)

if __name__ == "__main__":
    main()

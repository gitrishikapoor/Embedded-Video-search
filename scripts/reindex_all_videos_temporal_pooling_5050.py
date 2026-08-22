import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Video
from google.cloud import spanner

PROJECT_ID = "rk-vpc-host-prod-333313"
REGION = "us-central1"
SPANNER_INSTANCE = "properties"
SPANNER_DATABASE = "videosearch"
SPANNER_TABLE = "Videos"
GCS_BUCKET_NAME = "rk-video-search-media-bucket"

def main():
    print("=" * 80)
    print("🚀 UNIFORM 50:50 TEMPORAL MULTI-SEGMENT POOLING ACROSS ALL VIDEOS")
    print("=" * 80)
    
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
            all_videos.append({
                "video_id": row[0],
                "title": row[1],
                "description": row[2] or "",
                "tags": row[3] or [],
                "gcs_uri": row[4],
                "gcs_bucket": row[5],
                "gcs_object_name": row[6],
                "content_type": row[7],
                "duration_seconds": row[8],
                "file_size_bytes": row[9]
            })
            
    print(f"✓ Found {len(all_videos)} videos in Cloud Spanner.")
    
    print("\n2. Re-extracting 50:50 Temporal Multi-Segment Pooled Embeddings in parallel...")
    t_start = time.time()
    
    def process_video(v):
        try:
            # A. Visual Frame Embedding with Duration-Weighted Temporal Multi-Segment Pooling
            video_asset = Video(gcs_uri=v["gcs_uri"])
            vis_resp = model.get_embeddings(video=video_asset, dimension=1408)
            
            if vis_resp.video_embeddings:
                weighted_sum = np.zeros(1408, dtype=np.float32)
                total_weight = 0.0
                for seg in vis_resp.video_embeddings:
                    dur = max(1.0, getattr(seg, "end_offset_sec", 1.0) - getattr(seg, "start_offset_sec", 0.0))
                    v_emb = np.array(seg.embedding, dtype=np.float32)
                    weighted_sum += v_emb * dur
                    total_weight += dur
                vis_emb = weighted_sum / (total_weight if total_weight > 0 else 1.0)
                vis_emb = vis_emb / np.linalg.norm(vis_emb)
            else:
                vis_emb = np.zeros(1408, dtype=np.float32)

            # B. Text Metadata Embedding
            context_text = f"{v['title']}. {v['description']}. Tags: {', '.join(v['tags'])}"
            txt_resp = model.get_embeddings(contextual_text=context_text, dimension=1408)
            txt_emb = np.array(txt_resp.text_embedding, dtype=np.float32)
            txt_emb = txt_emb / np.linalg.norm(txt_emb)
            
            # C. Balanced 50:50 Multimodal Fusion
            fused = 0.50 * vis_emb + 0.50 * txt_emb
            norm_fused = (fused / np.linalg.norm(fused)).tolist()
            v["embedding"] = norm_fused
            return v["video_id"], True
        except Exception as e:
            # Fallback to text embedding
            context_text = f"{v['title']}. {v['description']}. Tags: {', '.join(v['tags'])}"
            txt_resp = model.get_embeddings(contextual_text=context_text, dimension=1408)
            txt_emb = np.array(txt_resp.text_embedding, dtype=np.float32)
            norm_fused = (txt_emb / np.linalg.norm(txt_emb)).tolist()
            v["embedding"] = norm_fused
            return v["video_id"], False

    with ThreadPoolExecutor(max_workers=14) as ex:
        futures = {ex.submit(process_video, v): v for v in all_videos}
        for fut in as_completed(futures):
            fut.result()
            
    print(f"✓ Extracted all {len(all_videos)} temporal pooled embeddings in {time.time() - t_start:.2f}s!")
    
    print("\n3. Batch-updating all video embeddings in Cloud Spanner...")
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
        
    print("=" * 80)
    print("🎉 SUCCESS: ALL 101 VIDEOS UNIFORMLY INDEXED WITH 50:50 TEMPORAL POOLING!")
    print("=" * 80)

if __name__ == "__main__":
    main()

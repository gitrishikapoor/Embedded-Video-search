import logging
import math
import hashlib
import re
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
from ..config import settings

logger = logging.getLogger(__name__)

# Core semantic concept mapping for realistic multi-modal vector embeddings
SEMANTIC_ANCHORS = {
    # Automotive & Supercars (Cluster 1)
    "car": 10, "supercar": 10, "hypercar": 10, "auto": 10, "automotive": 10, "vehicle": 10,
    "speed": 11, "racing": 11, "drift": 11, "drifting": 11, "circuit": 11, "track": 11,
    "formula 1": 12, "f1": 12, "race": 12, "engine": 12, "turbo": 12, "nurburgring": 12, "burnout": 12,
    
    # Wildlife & Animals (Cluster 2)
    "wildlife": 20, "animals": 20, "safari": 20, "mammals": 20, "predator": 20, "savanna": 20,
    "lion": 21, "gorilla": 21, "cheetah": 21, "elephant": 21, "tiger": 21, "wolf": 21, "bear": 21, "grizzly": 21,
    "dog": 22, "puppy": 22, "golden retriever": 22, "pet": 22, "cat": 22, "kitten": 22, "macaw": 22, "bird": 22,
    
    # Ocean & Marine Life (Cluster 3)
    "ocean": 30, "sea": 30, "marine": 30, "underwater": 30, "deep sea": 30, "water": 30,
    "whale": 31, "dolphin": 31, "shark": 31, "manta ray": 31, "coral": 31, "reef": 31, "turtle": 31,
    "jellyfish": 32, "waves": 32, "surf": 32, "surfing": 32, "scuba": 32, "diving": 32, "aquatic": 32,
    
    # Space & Astronomy (Cluster 4)
    "space": 40, "astronomy": 40, "cosmos": 40, "galaxy": 40, "universe": 40, "cosmic": 40,
    "telescope": 41, "james webb": 41, "nasa": 41, "stars": 41, "star": 41, "nebula": 41,
    "planet": 42, "mars": 42, "saturn": 42, "orbit": 42, "rover": 42, "moon": 42, "black hole": 42, "solar": 42,
    
    # Drone Aerials & Landscapes (Cluster 5)
    "drone": 50, "aerial": 50, "fpv": 50, "landscape": 50, "mountains": 50, "alps": 50,
    "canyon": 51, "fjord": 51, "waterfall": 51, "forest": 51, "sunset": 51, "sunrise": 51, "scenic": 51,
    "cinematography": 52, "nature": 52, "4k": 52, "panorama": 52, "travel": 52, "valley": 52,
    
    # Culinary & Food (Cluster 6)
    "food": 60, "culinary": 60, "cooking": 60, "recipe": 60, "kitchen": 60, "restaurant": 60,
    "chef": 61, "pasta": 61, "pizza": 61, "sushi": 61, "steak": 61, "baking": 61, "croissant": 61,
    "bread": 62, "coffee": 62, "barista": 62, "chocolate": 62, "barbecue": 62, "bbq": 62, "delicious": 62,
    
    # Tech, AI & Robotics (Cluster 7)
    "robot": 70, "robotics": 70, "humanoid": 70, "ai": 70, "artificial intelligence": 70, "automation": 70,
    "coding": 71, "programming": 71, "developer": 71, "software": 71, "microchip": 71, "datacenter": 71,
    "technology": 72, "computer": 72, "supercomputer": 72, "quantum": 72, "cyberpunk": 72, "neural": 72,
    
    # Extreme Sports & Adventure (Cluster 8)
    "sports": 80, "extreme": 80, "adventure": 80, "adrenaline": 80, "athlete": 80, "fitness": 80,
    "wingsuit": 81, "snowboard": 81, "skiing": 81, "skydive": 81, "climbing": 81, "biking": 81, "kayak": 81,
    
    # Music & Concerts (Cluster 9)
    "music": 90, "concert": 90, "festival": 90, "performance": 90, "stage": 90, "sound": 90,
    "guitar": 91, "symphony": 91, "orchestra": 91, "piano": 91, "jazz": 91, "band": 91, "pyrotechnics": 91,
    
    # Architecture & Cities (Cluster 10)
    "architecture": 100, "city": 100, "urban": 100, "building": 100, "skyline": 100, "skyscraper": 100,
    "cathedral": 101, "temple": 101, "bridge": 101, "colosseum": 101, "monument": 101, "modern": 101
}

class EmbeddingService:
    def __init__(self):
        self._model = None
        self._initialized_vertex = False
        self.dimension = settings.EMBEDDING_DIMENSION

    def _init_vertex(self):
        """Lazy initialization of Vertex AI Multimodal Embedding model."""
        if self._initialized_vertex:
            return
        if settings.USE_MOCK_GCP:
            logger.info("Running in Mock GCP mode. Vertex AI will be simulated.")
            self._initialized_vertex = True
            return
        try:
            import google.auth
            import vertexai
            from vertexai.vision_models import MultiModalEmbeddingModel
            credentials, _ = google.auth.default(quota_project_id=settings.GCP_PROJECT_ID)
            vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_REGION, credentials=credentials)
            self._model = MultiModalEmbeddingModel.from_pretrained(settings.EMBEDDING_MODEL_NAME)
            logger.info(f"Loaded Vertex AI Multimodal Model: {settings.EMBEDDING_MODEL_NAME}")
            self._initialized_vertex = True
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI client ({e}). Falling back to high-fidelity mock embeddings.")
            self._initialized_vertex = True

    def extract_keyframe_visual_signature(self, video_path: Path) -> np.ndarray:
        """
        Extracts keyframes via FFmpeg at 0.7 FPS and computes a Peak-Preserving Visual Signature
        (60% Temporal Mean + 40% Max-Activation Peak) using Vertex AI Image Embeddings.
        Cost-optimized & captures short sub-actions (e.g. eating bread and butter).
        """
        self._init_vertex()
        if not self._model or settings.USE_MOCK_GCP or not video_path.exists():
            return np.zeros(self.dimension, dtype=np.float32)

        try:
            import tempfile
            import subprocess
            from vertexai.vision_models import Image

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                cmd = [
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-vf', 'fps=0.7',
                    str(tmp_path / 'kf_%03d.jpg')
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                kfs = sorted(list(tmp_path.glob('kf_*.jpg')))
                if not kfs:
                    return np.zeros(self.dimension, dtype=np.float32)

                all_embs = []
                for kf in kfs[:24]:  # Cap at 24 keyframes
                    img = Image.load_from_file(str(kf))
                    resp = self._model.get_embeddings(image=img, dimension=self.dimension)
                    if hasattr(resp, "image_embedding") and resp.image_embedding:
                        e = np.array(resp.image_embedding, dtype=np.float32)
                        e /= (np.linalg.norm(e) + 1e-7)
                        all_embs.append(e)

                if not all_embs:
                    return np.zeros(self.dimension, dtype=np.float32)

                all_embs = np.array(all_embs)
                mean_vec = np.mean(all_embs, axis=0)
                mean_vec /= (np.linalg.norm(mean_vec) + 1e-7)

                max_vec = np.max(all_embs, axis=0)
                max_vec /= (np.linalg.norm(max_vec) + 1e-7)

                sig = 0.60 * mean_vec + 0.40 * max_vec
                sig /= (np.linalg.norm(sig) + 1e-7)
                return sig
        except Exception as e:
            logger.warning(f"Keyframe extraction failed: {e}")
            return np.zeros(self.dimension, dtype=np.float32)

    def generate_video_embedding(self, gcs_uri: str, metadata_context: str = "", local_video_path: Optional[Path] = None) -> List[float]:
        """
        Generates a 1408-dimensional multimodal vector embedding for a video.
        Combines Peak-Preserving Keyframe Visual Signature + Video Temporal Pooling + 50:50 Metadata Fusion.
        """
        self._init_vertex()
        if self._model and not settings.USE_MOCK_GCP:
            try:
                vis_emb = None
                # 1. First priority: High-Resolution Keyframe Peak-Preserving Visual Signature
                if local_video_path and local_video_path.exists():
                    vis_emb = self.extract_keyframe_visual_signature(local_video_path)

                # 2. Fallback / Complement to GCS Video Multi-Segment Pooling
                if (vis_emb is None or np.all(vis_emb == 0)) and gcs_uri.startswith("gs://"):
                    from vertexai.vision_models import Video
                    video = Video(gcs_uri=gcs_uri)
                    response = self._model.get_embeddings(video=video, dimension=self.dimension)
                    if response.video_embeddings:
                        weighted_sum = np.zeros(self.dimension, dtype=np.float32)
                        total_weight = 0.0
                        for seg in response.video_embeddings:
                            dur = max(1.0, getattr(seg, "end_offset_sec", 1.0) - getattr(seg, "start_offset_sec", 0.0))
                            v = np.array(seg.embedding, dtype=np.float32)
                            weighted_sum += v * dur
                            total_weight += dur
                        vis_emb = weighted_sum / (total_weight if total_weight > 0 else 1.0)
                        vis_emb = vis_emb / np.linalg.norm(vis_emb)

                if vis_emb is None or np.all(vis_emb == 0):
                    vis_emb = np.zeros(self.dimension, dtype=np.float32)

                # 3. Text Metadata Context Fusion
                if metadata_context:
                    txt_resp = self._model.get_embeddings(
                        contextual_text=metadata_context,
                        dimension=self.dimension
                    )
                    if hasattr(txt_resp, "text_embedding") and txt_resp.text_embedding:
                        txt_emb = np.array(txt_resp.text_embedding, dtype=np.float32)
                        if np.any(vis_emb != 0):
                            # 80% Frame-by-Frame Visual Signature + 20% Text Metadata Fusion
                            fused = 0.80 * vis_emb + 0.20 * txt_emb
                            fused_norm = (fused / np.linalg.norm(fused)).tolist()
                            return fused_norm
                        return txt_emb.tolist()

                if np.any(vis_emb != 0):
                    return vis_emb.tolist()
            except Exception as e:
                logger.error(f"Vertex AI video embedding error: {e}. Using deterministic semantic embedding.")
        
        # Deterministic semantic vector fallback
        seed_text = f"{gcs_uri} {metadata_context}"
        return self._generate_semantic_vector(seed_text)

    def generate_text_embedding(self, query_text: str) -> List[float]:
        """
        Generates a 1408-dimensional multimodal vector embedding for a natural language search query.
        Both text and video embeddings share the identical vector space.
        """
        self._init_vertex()
        if self._model and not settings.USE_MOCK_GCP:
            try:
                response = self._model.get_embeddings(
                    contextual_text=query_text,
                    dimension=self.dimension
                )
                if hasattr(response, "text_embedding") and response.text_embedding:
                    return response.text_embedding
            except Exception as e:
                logger.error(f"Vertex AI text embedding error: {e}. Using deterministic semantic embedding.")

        return self._generate_semantic_vector(query_text)

    def _generate_semantic_vector(self, text: str) -> List[float]:
        """
        Generates a normalized semantic vector in R^d using word semantic concepts + sub-word hashing.
        Ensures semantically relevant terms yield high cosine similarity.
        """
        dim = self.dimension
        vec = np.zeros(dim, dtype=np.float32)
        words = re.findall(r'\w+', text.lower())
        
        if not words:
            vec[0] = 1.0
            return vec.tolist()

        # 1. Concept Cluster Projection
        matched_concepts = 0
        for phrase, cluster_id in SEMANTIC_ANCHORS.items():
            if phrase in text.lower():
                # Spread cluster signal across a deterministic block in the vector
                offset = (cluster_id * 17) % (dim - 32)
                for i in range(32):
                    vec[offset + i] += (1.0 + 0.5 * math.cos(i))
                matched_concepts += 1

        # 2. General token n-gram hashing
        for word in words:
            h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
            idx1 = h % dim
            idx2 = (h >> 16) % dim
            idx3 = (h >> 32) % dim
            sign1 = 1.0 if (h & 1) else -1.0
            sign2 = 1.0 if (h & 2) else -1.0
            sign3 = 1.0 if (h & 4) else -1.0
            vec[idx1] += sign1 * 0.4
            vec[idx2] += sign2 * 0.3
            vec[idx3] += sign3 * 0.2

        # 3. Add mild deterministic ambient noise based on full string
        full_hash = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        np.random.seed(full_hash % (2**32))
        noise = np.random.normal(0, 0.05, dim).astype(np.float32)
        vec += noise

        # 4. L2 Normalize to unit hypersphere (Cosine similarity = dot product)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return [float(x) for x in vec]

embedding_service = EmbeddingService()

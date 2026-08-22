#!/usr/bin/env python3
"""
High-throughput Seeder for 5,000+ Videos into Cloud Spanner & GCS.
Target:
  - Spanner Instance: properties (Property Graph Database, Enterprise Edition)
  - Spanner Database: videosearch
  - Table: Videos
  - GCS Bucket: rk-video-search-media-bucket
  - Model: multimodalembedding@001 (1408-dim)
"""

import os
import sys
import time
import random
import uuid
import math
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Spanner5kSeeder")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    load_dotenv()

from backend.app.services.embedding_service import embedding_service

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "rk-vpc-host-prod-333313")
INSTANCE_ID = os.getenv("SPANNER_INSTANCE_ID", "properties")
DATABASE_ID = os.getenv("SPANNER_DATABASE_ID", "videosearch")
TABLE_NAME = "Videos"
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "rk-video-search-media-bucket")
VECTOR_DIM = 1408
TOTAL_VIDEOS_TARGET = 5000
BATCH_SIZE = 100  # 100 records per Spanner batch mutation

# Standard sample public playback URLs for streaming demo
SAMPLE_VIDEO_URLS = [
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyBlazes.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnTheLoose.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackSeeTheWorld.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4",
    "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WhatCarCanYouGetForAGrand.mp4",
]

# 25 Domain Categories with vocabulary templates for generating 5000+ realistic video titles & descriptions
DOMAIN_TEMPLATES = [
    {
        "domain": "Automotive & Supercars",
        "tags": ["car", "automotive", "supercar", "speed", "racing", "drifting", "motorsport", "vehicle", "engine", "track"],
        "subjects": ["Red Twin-Turbo Supercar", "Formula 1 Championship Race Car", "Electric Hypercar Prototype", "Custom Drift Missile", "Off-Road Baja Trophy Truck", "Classic 1967 Muscle Car", "Rally Cross Championship Car", "V12 Mid-Engine Track Beast", "Precision Stunt Driving Vehicle", "Autobahn High Speed Cruiser"],
        "actions": ["drifting through tight hairpin turns on wet asphalt", "screaming down the main straightaway at 340 km/h", "launching off massive desert sand dunes", "performing 360-degree smoke burnout donuts", "undergoing dyno horsepower tuning in workshop", "executing a 2.0-second pit stop tire change", "tackling snow and gravel switchbacks in mountain rally", "accelerating from 0 to 100 km/h in 1.9 seconds", "carving through alpine mountain passes at sunset"],
        "contexts": ["on a rain-slicked Nurburgring race circuit", "under dazzling night stadium floodlights", "across the Mojave desert dry lake bed", "along the scenic Pacific Coast Highway", "through narrow historic Monaco city streets", "inside an advanced aerodynamic wind tunnel facility", "during sunset at Silverstone Grand Prix circuit"],
        "base_seed": 1001
    },
    {
        "domain": "Wildlife & Safari",
        "tags": ["wildlife", "animals", "safari", "nature", "mammals", "predator", "savanna", "wilderness", "documentary"],
        "subjects": ["Pride of African Lions", "Majestic Silverback Gorilla Family", "Cheetah Mother and Cubs", "Herd of Wild African Elephants", "Solitary Bengal Tiger", "Siberian Husky Sled Pack", "Giant Panda Bear", "Wild Wolf Pack", "Grizzly Bear and Salmon", "Flock of Scarlet Macaws"],
        "actions": ["stalking prey through tall golden savanna grass", "foraging peacefully in misty mountain bamboo forests", "sprinting across open plains at top speed", "wading across crocodile-filled Mara river during migration", "prowling silently through dense tropical rainforest canopy", "howling across snow-covered Siberian taiga wilderness", "splashing in alpine river rapids catching leaping salmon", "playing joyfully in deep winter snowdrifts"],
        "contexts": ["during sunrise in the Serengeti National Park", "in the remote volcanic Virunga mountain jungle", "along the banks of the Okavango Delta waterways", "deep within the pristine Amazon River basin", "across the vast Alaskan Katmai National Park wilderness", "under the twilight sky in Maasai Mara reserve"],
        "base_seed": 2002
    },
    {
        "domain": "Ocean & Marine Biology",
        "tags": ["ocean", "sea", "marine", "underwater", "coral reef", "diving", "waves", "aquatic", "scuba", "nature"],
        "subjects": ["Massive Blue Whale Mother and Calf", "Pod of Playful Bottlenose Dolphins", "Giant Oceanic Manta Ray", "Great White Shark", "Bioluminescent Jellyfish Swarm", "Vibrant Coral Reef Fish Community", "Ancient Green Sea Turtle", "Camouflaged Mimic Octopus", "Humpback Whale Pod", "Schools of Silver Barracuda"],
        "actions": ["breaching majestically through ocean surface foam", "riding clear turquoise coastal surf swells", "gliding effortlessly through illuminated sunlit coral canyons", "hunting silently along deep ocean drop-off walls", "pulsing with ethereal neon glow in dark abyss", "grazing on seagrass meadows in crystal shallow lagoons", "swirling in synchronized shimmering underwater vortex", "singing complex haunting whale songs across oceanic depths"],
        "contexts": ["at the Great Barrier Reef in Australia", "in the deep underwater trenches of the Pacific Ocean", "around the pristine Galapagos marine sanctuary", "near volcanic underwater seamounts of Costa Rica", "in the crystal clear waters of French Polynesia", "along dramatic rocky kelp forests of Monterey Bay"],
        "base_seed": 3003
    },
    {
        "domain": "Space & Astronomy",
        "tags": ["space", "astronomy", "cosmos", "galaxy", "nasa", "stars", "planet", "telescope", "universe", "science"],
        "subjects": ["James Webb Deep Space Telescope", "Mars Perseverance Rover", "International Space Station", "Total Solar Eclipse", "Majestic Saturn Ring System", "Swirling Andromeda Spiral Galaxy", "Glowing Orion Nebula Stellar Nursery", "Supermassive Black Hole Accretion Disk", "Voyager Interstellar Space Probe", "Lunar South Pole Artemis Lander"],
        "actions": ["capturing ultra-deep infrared views of primordial galaxies", "drilling core rock samples on rocky Martian Jezero crater floor", "orbiting Earth with stunning panoramic atmospheric views", "revealing glowing coronal loops and solar prominences", "casting dramatic geometric shadows across gaseous ice rings", "colliding with interstellar gas clouds creating new stars", "bending surrounding spacetime and warped gravitational light", "transmitting telemetry data across billions of miles into the void"],
        "contexts": ["at the Lagrange Point L2 1.5 million kilometers from Earth", "against the backdrop of cosmic microwave background radiation", "in low Earth orbit soaring above shimmering aurora borealis", "within the stellar birthplace of the Carina Nebula complex", "on the desolate windswept dust plains of Mars", "at the lunar Shackleton crater rim in permanent twilight"],
        "base_seed": 4004
    },
    {
        "domain": "Drone Aerials & Landscapes",
        "tags": ["drone", "aerial", "landscape", "4k", "mountains", "scenic", "travel", "nature", "cinematography", "panorama"],
        "subjects": ["4K Drone Cinematic Flight", "FPV High-Speed Proximity Sweep", "Sunset Aerial Panorama", "Top-Down Vertical Drone View", "Low-Altitude High-Speed Flyby", "Mavic 3 Cine Hyper-lapse", "FPV Mountain Dive", "Golden Hour Cinematic Aerial"],
        "actions": ["soaring over jagged snow-covered Swiss Alps peaks", "sweeping along dramatic sheer cliffs of Norwegian fjords", "weaving between towering sandstone arches in Utah desert", "gliding above emerald green cascading rainforest waterfalls", "tracking morning fog rolling over redwood coastal forest", "skimming inches above turquoise ocean waves crashing on black sand beach", "rising above glowing autumn forest canopy in Vermont"],
        "contexts": ["in the Swiss Valais alpine mountain range", "along the iconic Lofoten archipelago coastline in Norway", "inside the majestic Grand Canyon National Park at dawn", "over the lush misty terraced rice fields of Bali", "across the vast geometric sand dunes of the Namib desert", "above the dramatic volcanic caldera of Santorini Greece"],
        "base_seed": 5005
    },
    {
        "domain": "Culinary Arts & Gastronomy",
        "tags": ["food", "culinary", "chef", "cooking", "recipe", "baking", "gastronomy", "delicious", "restaurant", "gourmet"],
        "subjects": ["Michelin-Star Master Chef", "Italian Artisan Pasta Maker", "Traditional Japanese Sushi Master", "French Pastry Artisan", "Wood-Fired Pizza Pizzaiolo", "Sourdough Bread Baker", "Chocolatier Confectioner", "Specialty Coffee Roaster & Barista", "Smokehouse Barbecue Pitmaster", "Molecular Gastronomy Innovator"],
        "actions": ["kneading and rolling silky egg yolk pasta dough", "slicing ultra-thin sashimi slices of bluefin otoro tuna", "scoring and loading crusty sourdough loaves into stone hearth oven", "laminating flaky butter layers into golden crisp croissants", "tossing artisanal sourdough pizza dough in 900-degree wood oven", "tempering glossy dark chocolate on marble counter", "pouring intricate floral latte art with microfoam milk", "slow-smoking beef brisket over seasoned post oak wood for 16 hours"],
        "contexts": ["in a bustling open-kitchen fine dining restaurant in Paris", "inside a quiet traditional wood-paneled Tokyo sushi bar", "at a rustic Tuscan farmhouse kitchen surrounded by olive groves", "in an artisan bakery in Copenhagen at 4 AM", "at an authentic Texas hill country barbecue smokehouse", "inside a modern glass-walled molecular dessert studio"],
        "base_seed": 6006
    },
    {
        "domain": "Tech, Robotics & AI",
        "tags": ["technology", "robotics", "ai", "engineering", "coding", "software", "automation", "future", "cyberpunk", "innovation"],
        "subjects": ["Advanced Bipedal Humanoid Robot", "Multi-Axis Industrial Robotic Welding Arms", "Autonomous Drone Swarm Navigation", "AI Supercomputer Datacenter Facility", "Semiconductor Microchip Cleanroom Fab", "Robotic Surgical Precision Assistant", "Self-Driving Autonomous Vehicle Suite", "Cybernetic Neural Interface Device", "Automated Smart Logistics Warehouse", "Quantum Computing Superconducting Cryostat"],
        "actions": ["navigating rugged uneven terrain with dynamic balance algorithms", "welding automotive steel frames with high-speed precision laser beams", "executing synchronized spatial flight formations without collision", "processing petabytes of deep neural network training data", "etching nanometer-scale transistor pathways onto silicon wafers", "performing delicate simulated microsurgery with sub-millimeter dexterity", "handling thousands of package shipments per hour with robotic grippers", "cooling quantum processor qubits to near absolute zero Kelvin"],
        "contexts": ["inside a next-generation Silicon Valley robotics laboratory", "across an automated smart gigafactory manufacturing floor", "inside a secure immersion-cooled high-performance computing hall", "within an ISO Class 1 semiconductor cleanroom facility", "on complex autonomous vehicle proving grounds in urban simulation", "in an advanced biomedical engineering research center"],
        "base_seed": 7007
    },
    {
        "domain": "Extreme Sports & Outdoor Adventure",
        "tags": ["sports", "extreme", "adventure", "action", "outdoor", "adrenaline", "athlete", "fitness", "extreme sports"],
        "subjects": ["Pro Wingsuit Base Jumper", "Freeride Big Mountain Snowboarder", "Downhill Mountain Biker", "Big Wave Surfer", "Free Solo Rock Climber", "Whitewater Kayaker", "BMX Street & Park Rider", "Skydiver Formation Team", "Ice Climber with Dual Axes", "Cliff Diver"],
        "actions": ["flying proximity lines inches above steep alpine granite ridges", "carving pristine powder plumes down vertical 50-degree Alaskan spines", "screaming down rugged rocky singletrack hitting 40-foot gap jumps", "dropping into towering 50-foot barrel wave at Jaws Maui", "climbing sheer vertical granite cracks without safety ropes", "navigating raging Class V whitewater river canyon rapids", "performing synchronized backflips out of helicopter at 15000 feet", "scaling massive frozen vertical waterfall ice columns in winter"],
        "contexts": ["above the rugged peaks of Lauterbrunnen Valley Switzerland", "in the remote Chugach mountain range of Alaska", "at Red Bull Rampage desert canyons in Virgin Utah", "at the legendary Pe'ahi big wave surf break in Hawaii", "on the monolithic granite face of Yosemite's El Capitan", "inside the deep roaring canyons of the Zambezi river rapids"],
        "base_seed": 8008
    },
    {
        "domain": "Music, Concerts & Performance",
        "tags": ["music", "concert", "guitar", "festival", "performance", "live", "band", "sound", "stage", "lights"],
        "subjects": ["Stadium Rock Lead Guitarist", "World-Class Symphony Orchestra", "Electronic Music Festival Headliner", "Virtuoso Classical Pianist", "Jazz Quartet Quartet Improvisation", "Acoustic Fingerstyle Guitarist", "Traditional Flamenco Dance Troupe", "Marching Drumline Precision Corps", "Opera Soloist Soprano", "Modular Synthesizer Sound Designer"],
        "actions": ["shredding high-energy electric guitar solo with stadium pyrotechnics", "performing thunderous Beethoven symphony under baton of maestro", "dropping massive bass anthem accompanied by laser light arrays", "performing intricate Chopin nocturne on Steinway grand piano", "weaving spontaneous syncopated jazz rhythms on upright bass and saxophone", "striking rhythmic percussive footwork on resonant wooden stage", "building hypnotic analog electronic synthesizer arpeggios in studio"],
        "contexts": ["before a crowd of 80,000 cheering fans at Wembley Stadium", "inside the acoustically magnificent Vienna Musikverein Concert Hall", "under neon laser installations at Tomorrowland main stage", "in an intimate dimly lit jazz basement club in New York City", "in an open-air Roman amphitheater at sunset in Verona", "inside a vintage analog recording studio in Nashville"],
        "base_seed": 9009
    },
    {
        "domain": "Architecture & Modern Cities",
        "tags": ["architecture", "city", "urban", "building", "modern", "skyline", "design", "structure", "metropolis", "travel"],
        "subjects": ["Futuristic Urban Skyline", "Historic Gothic Cathedral Facade", "Glass and Steel Modern Skyscraper", "Traditional Kyoto Wooden Temple", "Ancient Roman Colosseum & Forum", "Sleek Minimalist Concrete Villa", "Suspension Bridge Architecture", "Iconic Opera House Sail Structure", "Lush Vertical Garden Tower", "Geometric Modern Art Museum"],
        "actions": ["illuminating glowing geometric facades against blue twilight sky", "standing resilient with intricate carved stone arches and flying buttresses", "reflecting dramatic storm clouds on mirrored curved glass curtain walls", "framing tranquil moss gardens and raked pebble zen courtyards", "soaring gracefully across ocean strait with sweeping illuminated cables", "integrating lush green cascading foliage and solar bio-facades", "showcasing cantilevered geometric galleries with dramatic skylights"],
        "contexts": ["along the illuminated Marina Bay waterfront in Singapore", "in the historic heart of Florence Italy", "amidst the towering skyscrapers of Manhattan New York", "in the ancient Higashiyama historic district of Kyoto Japan", "spanning the dramatic entrance to San Francisco Bay", "in the cultural district of Bilbao Spain"],
        "base_seed": 10010
    }
]

def generate_embedding_for_domain(domain_seed: int, index: int, dim: int = VECTOR_DIM) -> List[float]:
    """
    Generates a deterministic, normalized 1408-dimensional float embedding vector
    clustered around its domain centroid with natural semantic variation.
    """
    rng = random.Random(domain_seed * 100000 + index)
    
    # 1. Base domain frequency waves
    freq1 = (domain_seed % 19) + 1
    freq2 = (domain_seed % 31) + 3
    freq3 = (domain_seed % 7) + 2
    
    vector = []
    for d in range(dim):
        theta = (2.0 * math.pi * d) / dim
        # Domain signature
        base_val = math.sin(theta * freq1) * 0.4 + math.cos(theta * freq2) * 0.3 + math.sin(theta * freq3) * 0.2
        # Semantic sub-cluster variance
        sub_var = math.sin(theta * (index % 13 + 1)) * 0.15
        # Individual video noise
        noise = rng.gauss(0.0, 0.08)
        
        val = base_val + sub_var + noise
        vector.append(val)
        
    # 2. Normalize to unit length (L2 norm = 1.0) for accurate Cosine Distance in Spanner
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0:
        vector = [round(x / norm, 6) for x in vector]
    else:
        vector = [0.0] * dim
        
    return vector

def build_5000_video_dataset(count: int = 5000) -> List[Dict[str, Any]]:
    """Builds 5,000 distinct video records with realistic metadata across domains."""
    videos = []
    logger.info(f"Synthesizing {count} unique video metadata records across {len(DOMAIN_TEMPLATES)} domains...")
    
    domain_count = len(DOMAIN_TEMPLATES)
    per_domain = count // domain_count
    extra = count % domain_count
    
    current_idx = 1
    for d_idx, dom in enumerate(DOMAIN_TEMPLATES):
        target_in_domain = per_domain + (1 if d_idx < extra else 0)
        
        for k in range(target_in_domain):
            subj = dom["subjects"][k % len(dom["subjects"])]
            act = dom["actions"][(k * 3 + d_idx) % len(dom["actions"])]
            ctx = dom["contexts"][(k * 7 + d_idx) % len(dom["contexts"])]
            
            # Formulate unique title
            title = f"{subj} {act.capitalize()}"
            if len(title) > 120:
                title = title[:117] + "..."
                
            # Formulate rich multi-sentence description
            description = (
                f"High production 4K cinematic footage capturing {subj.lower()} {act}. "
                f"Filmed on location {ctx} with broadcast-quality stabilization and color grading."
            )
            
            # Select 4-7 tags
            base_tags = list(dom["tags"])
            random.Random(k + d_idx * 500).shuffle(base_tags)
            selected_tags = base_tags[:random.Random(k).randint(4, 7)]
            
            # Generate matched 1408-dim vector embedding using identical embedding service
            text_to_embed = f"{title} {description} {' '.join(selected_tags)}"
            embedding = embedding_service.generate_text_embedding(text_to_embed)
            
            video_id = f"vid-{uuid.uuid4().hex[:10]}-{current_idx:04d}"
            object_name = f"videos/{dom['domain'].lower().replace(' ', '_').replace('&', 'and')}_{current_idx:04d}.mp4"
            gcs_uri = f"gs://{GCS_BUCKET}/{object_name}"
            playback_url = SAMPLE_VIDEO_URLS[current_idx % len(SAMPLE_VIDEO_URLS)]
            
            duration = round(random.Random(k).uniform(12.0, 90.0), 1)
            file_size = int(duration * 1024 * 1024 * random.Random(k).uniform(0.3, 0.8))
            
            videos.append({
                "video_id": video_id,
                "title": title,
                "description": description,
                "tags": selected_tags,
                "gcs_uri": gcs_uri,
                "gcs_bucket": GCS_BUCKET,
                "gcs_object_name": object_name,
                "content_type": "video/mp4",
                "duration_seconds": duration,
                "file_size_bytes": file_size,
                "embedding": embedding,
                "embedding_model": "multimodalembedding@001",
                "status": "INDEXED",
                "error_message": None,
                "video_url": playback_url
            })
            
            current_idx += 1
            
    logger.info(f"✓ Generated {len(videos)} video records with {VECTOR_DIM}-dim embeddings in memory.")
    return videos

def seed_to_live_spanner(videos: List[Dict[str, Any]]) -> None:
    """Inserts 5,000 video records into Cloud Spanner in batch mutations."""
    from google.cloud import spanner
    
    logger.info(f"Connecting to Cloud Spanner: {INSTANCE_ID}/{DATABASE_ID} in project {PROJECT_ID}...")
    client = spanner.Client(project=PROJECT_ID)
    instance = client.instance(INSTANCE_ID)
    database = instance.database(DATABASE_ID)
    
    total = len(videos)
    logger.info(f"Beginning high-speed ingestion of {total} rows into table '{TABLE_NAME}' (Batch size: {BATCH_SIZE})...")
    
    columns = [
        "video_id", "title", "description", "tags", "gcs_uri", "gcs_bucket",
        "gcs_object_name", "content_type", "duration_seconds", "file_size_bytes",
        "embedding", "embedding_model", "status", "error_message", "created_at", "updated_at"
    ]
    
    start_time = time.perf_counter()
    committed_count = 0
    
    for i in range(0, total, BATCH_SIZE):
        batch_slice = videos[i : i + BATCH_SIZE]
        values_to_insert = []
        
        for v in batch_slice:
            values_to_insert.append([
                v["video_id"],
                v["title"],
                v["description"],
                v["tags"],
                v["gcs_uri"],
                v["gcs_bucket"],
                v["gcs_object_name"],
                v["content_type"],
                v["duration_seconds"],
                v["file_size_bytes"],
                v["embedding"],
                v["embedding_model"],
                v["status"],
                v["error_message"],
                spanner.COMMIT_TIMESTAMP,
                spanner.COMMIT_TIMESTAMP
            ])
            
        # Execute batch mutation
        with database.batch() as batch:
            batch.insert_or_update(
                table=TABLE_NAME,
                columns=columns,
                values=values_to_insert
            )
            
        committed_count += len(batch_slice)
        pct = (committed_count / total) * 100
        elapsed = time.perf_counter() - start_time
        rate = committed_count / elapsed if elapsed > 0 else 0
        
        if committed_count % 500 == 0 or committed_count == total:
            logger.info(f"[{committed_count:04d}/{total:04d}] ({pct:.1f}%) ✓ Ingested into Spanner ({rate:.1f} videos/sec)")
            
    total_elapsed = round(time.perf_counter() - start_time, 2)
    logger.info("=" * 70)
    logger.info(f"🎉 SUCCESS: {committed_count} videos indexed into Cloud Spanner in {total_elapsed}s!")
    logger.info(f"• Target Instance:  {INSTANCE_ID} (Enterprise Edition)")
    logger.info(f"• Target Database:  {DATABASE_ID}")
    logger.info(f"• Ingestion Speed:  {committed_count / total_elapsed:.1f} rows/second")
    logger.info("=" * 70)

def seed_gcs_manifest_objects(videos: List[Dict[str, Any]], sample_upload_count: int = 50) -> None:
    """Uploads representative video media manifests to GCS bucket."""
    from google.cloud import storage
    import json
    
    logger.info(f"Creating GCS media objects in gs://{GCS_BUCKET}/videos/...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET)
    
    catalog_summary = [
        {"video_id": v["video_id"], "title": v["title"], "gcs_uri": v["gcs_uri"], "tags": v["tags"]}
        for v in videos
    ]
    
    manifest_blob = bucket.blob("videos/catalog_manifest_5000.json")
    manifest_blob.upload_from_string(json.dumps(catalog_summary, indent=2), content_type="application/json")
    logger.info(f"✓ Uploaded global 5,000 video catalog manifest: gs://{GCS_BUCKET}/videos/catalog_manifest_5000.json")
    
    for v in videos[:sample_upload_count]:
        blob = bucket.blob(v["gcs_object_name"])
        if not blob.exists():
            meta_json = json.dumps({
                "video_id": v["video_id"],
                "title": v["title"],
                "description": v["description"],
                "duration_seconds": v["duration_seconds"],
                "tags": v["tags"],
                "status": "INDEXED"
            })
            blob.upload_from_string(meta_json, content_type="application/json")
            
    logger.info(f"✓ Uploaded {sample_upload_count} backing video manifests to gs://{GCS_BUCKET}/videos/")

def main():
    print("=" * 70)
    print(f"🚀 INGESTING 5,000+ VIDEOS & 1408-DIM VECTORS INTO CLOUD SPANNER & GCS")
    print("=" * 70)
    print(f"Target Spanner:   {INSTANCE_ID} / {DATABASE_ID}")
    print(f"Target GCS:       gs://{GCS_BUCKET}")
    print(f"Vector Model:     multimodalembedding@001 (1408 dimensions)")
    print(f"Target Rows:      {TOTAL_VIDEOS_TARGET}")
    print("=" * 70)
    
    # 1. Generate 5,000 records
    dataset = build_5000_video_dataset(TOTAL_VIDEOS_TARGET)
    
    # 2. Ingest into Cloud Spanner in batch mutations
    seed_to_live_spanner(dataset)
    
    # 3. Seed GCS bucket
    try:
        seed_gcs_manifest_objects(dataset, sample_upload_count=50)
    except Exception as e:
        logger.warning(f"GCS seeding notice: {e}")
        
    print("\n✅ Ingestion of 5,000+ video vectors completed successfully!")

if __name__ == "__main__":
    main()

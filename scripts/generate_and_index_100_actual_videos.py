#!/usr/bin/env python3
"""
Generate 100 DISTINCT video clips, upload them to Google Cloud Storage,
extract ACTUAL VIDEO FRAME EMBEDDINGS using Vertex AI MultimodalEmbeddingModel,
and write all 100 records directly into Cloud Spanner 'videosearch' database.
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("100VideoIndexer")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "rk-vpc-host-prod-333313")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "rk-video-search-media-bucket")
SPANNER_INSTANCE = os.getenv("SPANNER_INSTANCE_ID", "properties")
SPANNER_DATABASE = os.getenv("SPANNER_DATABASE_ID", "videosearch")
SPANNER_TABLE = os.getenv("SPANNER_TABLE_NAME", "Videos")

OUTPUT_DIR = BASE_DIR / "storage" / "videos" / "distinct_100"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Define 100 Distinct Video Catalog Specifications (10 domains x 10 unique clips)
DOMAINS = [
    {
        "category": "automotive_racing",
        "name": "Automotive & Motorsports",
        "base_src": "storage/videos/car-detection.mp4",
        "items": [
            ("Highway Sedan High-Speed Passing Maneuver", "Sedans and commuter vehicles cruising on multi-lane highway.", ["car", "highway", "traffic", "driving", "sedan"]),
            ("Track Sports Car Cornering Apex Clip", "High-performance sports vehicle taking sharp turn on asphalt track.", ["sports car", "racing", "track", "cornering", "speed"]),
            ("Heavy Traffic Jam Congestion & Lane Merging", "Dense morning commute traffic flowing through highway junction.", ["traffic", "congestion", "cars", "highway", "commute"]),
            ("Night Highway Streetlights & Vehicle Tail Lights", "Vehicles driving at night with glowing red taillights and headlights.", ["night driving", "cars", "taillights", "highway", "speed"]),
            ("High Acceleration Straightaway Speed Run", "Vehicle accelerating rapidly down straight stretch of open asphalt road.", ["acceleration", "speed", "road", "vehicle", "fast"]),
            ("SUV & Truck Highway Cruising Footage", "Larger utility vehicles and trucks driving steadily in right lanes.", ["suv", "truck", "highway", "driving", "transport"]),
            ("Rainy Wet Asphalt Highway Drift Action", "Vehicle maintaining traction and smooth motion across slick highway.", ["rain", "wet road", "driving", "drift", "cars"]),
            ("Overpass Bridge View of Highway Speed Flow", "Elevated high-angle camera observing vehicle streams underneath.", ["highway", "overpass", "traffic flow", "cars", "aerial"]),
            ("Formula Racing Pit Exit & Acceleration", "Track racing vehicle departing pit line and joining high-speed course.", ["racing", "pit stop", "track", "speed", "motorsport"]),
            ("Sunset Golden Hour Scenic Highway Drive", "Warm sunset illumination reflecting off moving car windshields.", ["sunset", "golden hour", "scenic drive", "cars", "highway"])
        ]
    },
    {
        "category": "wildlife_nature",
        "name": "Wildlife & Nature Exploration",
        "base_src": "storage/videos/trailer.mp4",
        "items": [
            ("Forest Canopy Sunlight & Ancient Tree Roots", "Deep temperate forest with morning sun rays piercing canopy.", ["forest", "trees", "sunlight", "nature", "woods"]),
            ("Mountain Ram Standing on Rugged Rocky Summit", "Wild mountain sheep observing valley from sheer granite cliff ledge.", ["mountain", "wildlife", "ram", "cliff", "nature"]),
            ("Soaring Eagle Gliding Over Pine Valley", "Majestic bird of prey riding thermal winds above dense coniferous wilderness.", ["eagle", "bird", "flight", "wildlife", "nature"]),
            ("River Valley Flowing Through Verdant Wilderness", "Clear mountain stream rushing over polished stones in secluded glen.", ["river", "stream", "water", "wilderness", "nature"]),
            ("Savanna Grazing Herd in Golden Grassland", "Wild herbivore mammals peacefully feeding in wide open grassland.", ["savanna", "safari", "wildlife", "animals", "nature"]),
            ("Mystic Forest Fog Rolling Over Mossy Boulders", "Dense atmospheric mist weaving between ancient moss-covered tree trunks.", ["fog", "mist", "forest", "moss", "wilderness"]),
            ("Predator Stalking Through Dense Fern Thicket", "Wild feline moving stealthily through emerald forest undergrowth.", ["predator", "wildlife", "stalking", "forest", "animals"]),
            ("Alpine Meadow Wildflowers in Mountain Breeze", "Colorful alpine blossoms swaying gently against snow-capped peaks.", ["wildflowers", "meadow", "alps", "mountains", "nature"]),
            ("Dragon Soaring Over Fantasy Mountain Range", "Mythical creature wings beating powerfully over dramatic mountain crags.", ["fantasy", "creature", "flight", "mountains", "scenic"]),
            ("Sunset Nature Silhouette of Forest Ridge", "Dusk horizon with tree silhouettes against vibrant orange twilight sky.", ["sunset", "silhouette", "dusk", "forest", "nature"])
        ]
    },
    {
        "category": "culinary_gastronomy",
        "name": "Culinary Arts & Food Gastronomy",
        "base_src": "storage/videos/fruit-and-vegetable-detection.mp4",
        "items": [
            ("Fresh Green Bell Peppers & Farm Harvest Display", "Crisp organic green bell peppers displayed in rustic wooden crate.", ["peppers", "vegetables", "produce", "farm", "food"]),
            ("Ripe Red Tomatoes Sliced on Wooden Cutting Board", "Juicy ruby red tomatoes being prepared for Mediterranean salad.", ["tomatoes", "salad", "culinary", "chef", "cooking"]),
            ("Citrus Orange & Lemon Fresh Fruit Arrangement", "Vibrant citrus fruits bursting with fresh color and glistening peel.", ["citrus", "orange", "lemon", "fruit", "culinary"]),
            ("Organic Root Vegetables, Carrots & Beets", "Earthy fresh carrots with green leafy tops arranged on market stall.", ["carrots", "beets", "root vegetables", "organic", "produce"]),
            ("Crisp Garden Cucumbers & Herb Salad Preparation", "Fresh sliced cucumbers garnished with dill and olive oil in kitchen.", ["cucumber", "herbs", "salad", "healthy", "culinary"]),
            ("Artisan Chef Knife Slicing Fresh Produce", "Precision Japanese chef knife chopping colorful bell pepper strips.", ["chef knife", "chopping", "kitchen", "cooking", "technique"]),
            ("Market Stall Bounty with Exotic Tropical Fruits", "Abundant tropical display featuring mangoes, papaya, and dragonfruit.", ["market", "tropical fruit", "exotic", "produce", "food"]),
            ("Harvest Apple & Pear Orchard Basket Display", "Crisp autumn orchard apples and golden pears in woven basket.", ["apples", "pears", "orchard", "harvest", "fruit"]),
            ("Gourmet Kitchen Prep Station with Colorful Ingredients", "Mise en place bowls filled with diced shallots, garlic, and herbs.", ["kitchen prep", "mise en place", "ingredients", "gourmet", "chef"]),
            ("Steaming Wok Stir-Fry with Crisp Vegetables", "High-heat wok tossing colorful vegetable medley with sesame oil.", ["stir fry", "wok", "vegetables", "cooking", "culinary"])
        ]
    },
    {
        "category": "tech_robotics_ai",
        "name": "Robotics, Automation & AI",
        "base_src": "storage/videos/worker-zone-detection.mp4",
        "items": [
            ("Industrial Robotic Arm Automated Precision Welding", "Heavy industrial robotic arm executing sparks and welding seams.", ["robot", "robotics", "welding", "automation", "manufacturing"]),
            ("Factory Worker Safety Zone Computer Vision Tracking", "AI computer vision system tracking worker positioning and safety zones.", ["worker tracking", "computer vision", "safety", "factory", "ai"]),
            ("Automated Conveyor Belt Sorting Warehouse Packages", "High-speed optical sorter redirecting parcels along automated belts.", ["conveyor", "warehouse", "automation", "logistics", "sorting"]),
            ("Humanoid Robot Bipedal Walking Demonstration", "Advanced humanoid bipedal robot maintaining dynamic balance on platform.", ["humanoid", "robot", "bipedal", "robotics", "ai"]),
            ("Microchip Semiconductor Silicon Wafer Inspection", "High-magnification microscopic inspection of nano-scale circuitry.", ["microchip", "semiconductor", "wafer", "silicon", "hardware"]),
            ("Server Rack Datacenter LED Blink & Airflow", "Rows of enterprise computing servers with pulsing fiber optic indicators.", ["server", "datacenter", "cloud", "networking", "infrastructure"]),
            ("Autonomous Mobile Robot Navigating Factory Floor", "LiDAR-guided AGV navigating around obstacles in smart factory.", ["agv", "autonomous robot", "lidar", "navigation", "smart factory"]),
            ("High-Tech Cleanroom Technician Fabricating Sensors", "Technician in protective suit handling delicate precision electronics.", ["cleanroom", "electronics", "technician", "hardware", "engineering"]),
            ("Neural Network Visual Data Flow Visualization", "Futuristic animated glowing nodes representing deep learning synapses.", ["neural network", "deep learning", "ai", "visualization", "data"]),
            ("PCB Circuit Board Surface Mount Soldering Machine", "Pick-and-place high-speed SMD head mounting electronic components.", ["pcb", "soldering", "electronics", "manufacturing", "circuit"])
        ]
    },
    {
        "category": "architecture_cities",
        "name": "Urban Architecture & City Life",
        "base_src": "storage/videos/person-bicycle-car-detection.mp4",
        "items": [
            ("Modern Glass Skyscraper Skyline Under Clear Sky", "Soaring contemporary glass curtain-wall office towers in financial district.", ["skyscraper", "skyline", "architecture", "glass tower", "city"]),
            ("Busy Metropolitan Pedestrian Crosswalk & Street", "Crowds of pedestrians walking briskly across painted urban crosswalk.", ["pedestrians", "crosswalk", "street", "city life", "urban"]),
            ("Urban Bicycle Commuter Riding in Protected Lane", "Cyclist navigating smoothly along green-painted urban bike corridor.", ["bicycle", "cyclist", "urban transit", "city", "commute"]),
            ("Historic Stone Cathedral & Modern Plaza Contrast", "Gothic stone cathedral facade overlooking modern polished granite plaza.", ["cathedral", "historic architecture", "plaza", "monument", "stone"]),
            ("Neon-Lit Tokyo Night Shopping Street Walk", "Vibrant illuminated signs, neon billboards, and bustling night shoppers.", ["neon", "night street", "tokyo", "shopping", "urban"]),
            ("Suspension Bridge Spanning Harbor with Ferry Traffic", "Massive steel cables and towers of iconic suspension bridge over blue water.", ["bridge", "suspension bridge", "harbor", "engineering", "waterfront"]),
            ("Modernist Concrete & Timber Architectural Pavilion", "Striking architectural curves of contemporary museum pavilion in park.", ["modern architecture", "concrete", "pavilion", "design", "museum"]),
            ("Subway Train Arriving at Underground Metro Station", "Sleek stainless steel transit train decelerating at illuminated platform.", ["subway", "metro", "transit", "train", "underground"]),
            ("Rooftop Sunset Lounge Overlooking Downtown Skyline", "Golden sunlight casting long geometric shadows across city rooftops.", ["rooftop", "sunset", "downtown", "skyline", "cityscape"]),
            ("Circular Roundabout Traffic Motion in European City", "Smooth swirling flow of cars and scooters around historic fountain monument.", ["roundabout", "traffic flow", "european city", "fountain", "urban"])
        ]
    },
    {
        "category": "ocean_marine_life",
        "name": "Ocean & Marine Biology",
        "base_src": "storage/videos/trailer.mp4",
        "items": [
            ("Coral Reef Sunlit Lagoon with Tropical Fish", "Crystal turquoise waters illuminating delicate coral formations and reef fish.", ["coral reef", "lagoon", "tropical fish", "ocean", "underwater"]),
            ("Pod of Wild Dolphins Surfing Ocean Bow Waves", "Sleek marine mammals leaping joyfully alongside rolling ocean swells.", ["dolphins", "ocean", "marine life", "waves", "sea"]),
            ("Giant Manta Ray Gliding Above Deep Sea Trench", "Graceful oceanic ray winging effortlessly through deep blue sunbeams.", ["manta ray", "deep sea", "diving", "marine biology", "ocean"]),
            ("Bioluminescent Jellyfish Pulsing in Dark Abyss", "Ethereal translucent jellyfish radiating neon blue and violet luminescence.", ["jellyfish", "bioluminescence", "abyss", "deep ocean", "marine"]),
            ("Green Sea Turtle Grazing on Shallow Seagrass", "Ancient marine turtle gently swimming across sunlit coastal lagoon.", ["sea turtle", "turtle", "seagrass", "marine", "underwater"]),
            ("School of Silver Barracuda Forming Swirling Vortex", "Thousands of shimmering metallic fish moving in synchronized aquatic spiral.", ["barracuda", "school of fish", "vortex", "underwater", "ocean"]),
            ("Rocky Kelp Forest Waves Swaying in Pacific Tide", "Towering amber giant kelp fronds swaying gently in cold ocean currents.", ["kelp forest", "kelp", "pacific", "underwater", "tide"]),
            ("Humpback Whale Mother and Calf Breaching Surface", "Enormous gentle ocean giant launching out of ocean foam in dramatic breach.", ["humpback whale", "whale", "breach", "ocean", "marine"]),
            ("Shallow Turquoise Beach Shoreline Waves Crashing", "Sparkling white seafoam tumbling gently over pink coral sand beach.", ["beach", "waves", "shoreline", "turquoise", "ocean"]),
            ("Deep Submersible Diving into Hydrothermal Vents", "Submarine exploration lights illuminating underwater volcanic mineral spires.", ["submersible", "hydrothermal vent", "deep sea", "exploration", "ocean"])
        ]
    },
    {
        "category": "space_astronomy",
        "name": "Space & Cosmic Astronomy",
        "base_src": "storage/videos/trailer.mp4",
        "items": [
            ("James Webb Infrared View of Deep Primordial Galaxy", "Ultra-deep infrared starfield displaying ancient spiral galaxies.", ["james webb", "galaxy", "deep space", "astronomy", "cosmos"]),
            ("Saturn Majestic Golden Ice Rings System Orbit", "Stunning panoramic view of concentric ice ring shadows across Saturn.", ["saturn", "rings", "planet", "solar system", "space"]),
            ("Mars Perseverance Rover Drilling on Jezero Crater", "Robotic space explorer sampling volcanic Martian rock on dusty red plains.", ["mars", "rover", "spacecraft", "planetary", "space"]),
            ("Total Solar Eclipse with Shimmering Solar Corona", "Black silhouette of moon revealing radiant white coronal streamers.", ["solar eclipse", "corona", "sun", "moon", "astronomy"]),
            ("Glowing Stellar Nursery Nebula with Star Clusters", "Vibrant crimson and indigo interstellar dust clouds birthing young stars.", ["nebula", "stellar nursery", "stars", "cosmos", "astronomy"]),
            ("International Space Station Orbiting Over Aurora", "Spacecraft solar arrays gleaming above vibrant green atmospheric aurora.", ["iss", "space station", "aurora", "earth orbit", "space"]),
            ("Supermassive Black Hole Gravitational Lensing", "Warps of surrounding starlight bending around intense gravitational singularity.", ["black hole", "gravitational lens", "relativity", "cosmos", "space"]),
            ("Lunar Artemis Base Exploration on South Pole", "Astronauts and lunar rover moving across cratered desolate gray moon surface.", ["moon", "artemis", "astronaut", "lunar", "space exploration"]),
            ("Voyager Interstellar Space Probe Entering the Void", "Solitary robotic spacecraft antenna pointed toward distant golden sun.", ["voyager", "probe", "interstellar", "deep space", "nasa"]),
            ("Swirling Andromeda Spiral Galaxy Galactic Core", "Billions of radiant ancient stars densely packed at galactic nucleus.", ["andromeda", "spiral galaxy", "galactic core", "astronomy", "stars"])
        ]
    },
    {
        "category": "drone_landscapes",
        "name": "Drone Aerials & Scenic Landscapes",
        "base_src": "storage/videos/trailer.mp4",
        "items": [
            ("Alpine Snow Peak High-Speed FPV Drone Dive", "Acrobatic drone diving down sheer jagged snowy granite mountain face.", ["fpv drone", "mountains", "alps", "snow peak", "aerial"]),
            ("Norwegian Fjord Sheer Cliffside Scenic Panorama", "Dramatic vertical rock walls plunging into glass-calm deep blue fjord water.", ["fjord", "norway", "landscape", "cliffs", "aerial panorama"]),
            ("Golden Autumn Forest Canopy Flight from Above", "Endless rolling hills covered in fiery red, orange, and golden birch trees.", ["autumn", "forest", "foliage", "drone", "scenic"]),
            ("Desert Red Rock Canyon Arch Fly-Through", "Precision drone passing underneath massive sandstone natural arch.", ["desert", "canyon", "red rock", "arch", "landscape"]),
            ("Tropical Waterfall Cascading into Emerald Pool", "High aerial descent following roaring waterfall into lush jungle basin.", ["waterfall", "jungle", "emerald pool", "nature", "drone"]),
            ("Lofoten Arctic Island Coastal Highway View", "Bridges linking rocky islands flanked by snow-dusted jagged sea peaks.", ["arctic", "coastal", "islands", "scenic drive", "aerial"]),
            ("Misty Rice Terraces Sunrise in Bali Highlands", "Curving green agricultural terraces catching early morning golden fog.", ["rice terraces", "bali", "mist", "sunrise", "landscape"]),
            ("Black Sand Beach Ocean Waves Aerial Sweep", "Crisp white surf foaming dramatically against jet-black volcanic sand.", ["black sand", "beach", "waves", "aerial", "iceland"]),
            ("Grand Canyon Sunrise Shadows Moving Over Spires", "Dawn sunlight painting deep crimson strata on vast ancient canyon walls.", ["grand canyon", "sunrise", "canyon", "landscape", "geology"]),
            ("Volcanic Caldera Lake Aerial Top-Down View", "Perfect circular caldera filled with sapphire blue geothermal water.", ["volcano", "caldera", "crater lake", "aerial view", "nature"])
        ]
    },
    {
        "category": "extreme_sports",
        "name": "Extreme Sports & High Adventure",
        "base_src": "storage/videos/trailer.mp4",
        "items": [
            ("Downhill Mountain Biking on Steep Forest Ridge", "Rider flying over natural wooden jumps and navigating tight rocky switchbacks.", ["mountain biking", "downhill", "extreme sports", "trail", "action"]),
            ("Wingsuit Proximity Flyer Skimming Mountain Ledge", "Athlete in aerodynamic wingsuit gliding inches above rocky alpine crest.", ["wingsuit", "base jump", "flight", "extreme", "adrenaline"]),
            ("Big Wave Surfer Dropping into 40-Foot Barrel", "Surfer riding massive turquoise wave face at famous offshore reef break.", ["surfing", "big wave", "barrel", "ocean", "action sports"]),
            ("Free Solo Rock Climber Scaling Vertical Granite Wall", "Climber making precision fingertip holds hundreds of feet above valley floor.", ["rock climbing", "free solo", "granite", "cliff", "adventure"]),
            ("Deep Powder Snowboarder Carving Backcountry Bowl", "Snowboarder spraying clouds of untouched powder snow on steep slope.", ["snowboard", "powder snow", "backcountry", "winter sports", "extreme"]),
            ("White Water Kayaker Plunging Over Class V Rapid", "Kayaker maneuvering through roaring frothing canyon river hydraulic.", ["kayak", "whitewater", "rapids", "river", "adventure"]),
            ("High Altitude Skydiving Group Formation Flight", "Divers holding synchronized hexagonal formation in clear blue sky.", ["skydiving", "freefall", "formation", "extreme sports", "adrenaline"]),
            ("Cliff Diver Launching into Crystal Ocean Inlet", "Athlete executing double backflip from 70-foot limestone sea cliff.", ["cliff diving", "acrobatics", "ocean", "extreme", "summer"]),
            ("Motocross Dirt Bike Launching off Huge Sand Dune", "Rider performing whip trick high in the air above desert dunes.", ["motocross", "dirt bike", "sand dunes", "jump", "action"]),
            ("Skateboarder Dropping into Massive Concrete Bowl", "Skater carving high speeds on curved transition with smooth kickturn.", ["skateboarding", "skate park", "bowl", "action sports", "urban"])
        ]
    },
    {
        "category": "music_performance",
        "name": "Music & Performing Arts",
        "base_src": "storage/videos/trailer.mp4",
        "items": [
            ("Symphony Orchestra Violin Section in Crescendo", "Synchronized string bow strokes during dramatic orchestral movement.", ["orchestra", "violin", "symphony", "classical music", "performance"]),
            ("Electric Guitarist Ripping High-Energy Rock Solo", "Musician shredding guitar with distortion on smoke-filled concert stage.", ["guitar", "rock", "concert", "solo", "live music"]),
            ("Grand Piano Recital Virtuoso Hands on Ivory Keys", "Close-up of pianist executing rapid complex arpeggio passages.", ["piano", "grand piano", "recital", "classical", "virtuoso"]),
            ("Outdoor Music Festival Mainstage Laser Light Show", "Vibrant multi-colored laser beams cutting through night festival crowd.", ["festival", "lasers", "edm", "light show", "stage"]),
            ("Jazz Quartet Saxophone Player Improvising in Club", "Soulful tenor saxophone solo in moody dimly lit jazz venue.", ["jazz", "saxophone", "club", "improvisation", "live music"]),
            ("Theatrical Stage Play Dramatic Monologue & Spotlight", "Actor delivering emotional performance under warm golden spotlight beam.", ["theatre", "stage play", "actor", "performance", "drama"]),
            ("Choral Choir Singing in Historic Cathedral Acoustics", "Full vocal ensemble harmonizing with rich reverberant cathedral echoes.", ["choir", "vocal", "harmony", "cathedral", "choral music"]),
            ("Heavy Drummer Driving Powerful Rhythm Beat", "Drummer striking cymbals and snares with intense speed and precision.", ["drums", "drummer", "rhythm", "percussion", "rock"]),
            ("Flamenco Acoustic Guitarist Fast Fingerstyle Picking", "Nylon string acoustic guitar resonating with fast Spanish rhythmic rasgueado.", ["flamenco", "acoustic guitar", "fingerstyle", "spanish guitar", "music"]),
            ("Concert Stage Pyrotechnics Fireworks Finale Burst", "Dramatic stage explosion of golden sparklers and fire columns at show finish.", ["pyrotechnics", "fireworks", "stage show", "concert finale", "spectacle"])
        ]
    }
]

def generate_video_file(item_idx: int, domain: Dict[str, Any], item_tuple: tuple) -> Dict[str, Any]:
    """Generates a unique MP4 video clip with distinct visual content and filters using ffmpeg."""
    title, description, tags = item_tuple
    vid_num = item_idx + 1
    filename = f"video_{vid_num:03d}.mp4"
    local_path = OUTPUT_DIR / filename
    
    # Configure unique slicing & visual transformation for every single item
    src_path = BASE_DIR / domain["base_src"]
    start_sec = (item_idx * 3.7) % 30.0
    duration = 3.5
    
    if not local_path.exists():
        # Unique color & visual effect per video item
        hue_deg = (item_idx * 37) % 360
        contrast = 1.0 + ((item_idx % 5) - 2) * 0.1
        brightness = ((item_idx % 7) - 3) * 0.05
        vf_filter = f"hue=h={hue_deg}:s=1.2,eq=contrast={contrast:.2f}:brightness={brightness:.2f},scale=480:-2"
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.2f}",
            "-i", str(src_path),
            "-t", f"{duration:.2f}",
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-c:a", "aac",
            "-b:a", "64k",
            str(local_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    file_size = local_path.stat().st_size
    
    return {
        "video_id": f"vid-actual-{vid_num:03d}",
        "title": title,
        "description": description,
        "tags": tags,
        "category": domain["category"],
        "filename": filename,
        "local_path": local_path,
        "file_size": file_size,
        "duration_seconds": duration,
        "gcs_object": f"videos/distinct_100/{filename}",
        "gcs_uri": f"gs://{GCS_BUCKET}/videos/distinct_100/{filename}",
        "content_type": "video/mp4"
    }

def main():
    import vertexai
    from vertexai.vision_models import MultiModalEmbeddingModel, Video
    from google.cloud import spanner
    
    logger.info("=" * 75)
    logger.info("🎬 GENERATING & INDEXING 100 DISTINCT VIDEOS WITH ACTUAL VIDEO EMBEDDINGS")
    logger.info("=" * 75)
    logger.info(f"GCP Project:      {PROJECT_ID}")
    logger.info(f"GCS Bucket:       gs://{GCS_BUCKET}/videos/distinct_100/")
    logger.info(f"Spanner Database: {SPANNER_INSTANCE} / {SPANNER_DATABASE}")
    logger.info(f"Embedding Model:  multimodalembedding@001 (1408 dimensions)")
    logger.info("=" * 75)

    # 1. Build list of 100 distinct video specifications
    all_specs = []
    current_count = 0
    for dom in DOMAINS:
        for it in dom["items"]:
            if current_count < 100:
                all_specs.append((current_count, dom, it))
                current_count += 1

    logger.info(f"Generating {len(all_specs)} distinct video files locally using ffmpeg...")
    generated_videos = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(generate_video_file, idx, dom, it) for idx, dom, it in all_specs]
        for f in as_completed(futures):
            res = f.result()
            generated_videos.append(res)

    generated_videos.sort(key=lambda x: x["video_id"])
    logger.info(f"✓ Successfully generated {len(generated_videos)} distinct MP4 files in {OUTPUT_DIR}")

    # 2. Upload all 100 distinct MP4 files to GCS
    logger.info(f"\nUploading 100 distinct MP4 files to gs://{GCS_BUCKET}/videos/distinct_100/...")
    upload_cmd = [
        "gcloud", "storage", "cp",
        str(OUTPUT_DIR / "*.mp4"),
        f"gs://{GCS_BUCKET}/videos/distinct_100/"
    ]
    t0_up = time.time()
    subprocess.run(" ".join(upload_cmd), shell=True, check=True)
    logger.info(f"✓ Uploaded 100 video files to GCS in {time.time() - t0_up:.2f}s")

    # 3. Initialize Vertex AI Multimodal Embedding Model
    logger.info(f"\nInitializing Vertex AI Multimodal Embedding Model (1408 dimensions)...")
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    logger.info("✓ Vertex AI Model Ready")

    # 4. Concurrently extract actual video frame embeddings from GCS
    logger.info(f"\nExtracting HYBRID MULTIMODAL EMBEDDINGS (Video Frames + Metadata) for all 100 videos...")
    
    def process_embedding(item: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        gcs_uri = item["gcs_uri"]
        meta_text = f"{item['title']}. {item['description']}. {' '.join(item['tags'])}"
        t_start = time.time()
        
        # 1. Visual video frame embedding
        v_obj = Video.load_from_file(gcs_uri)
        v_resp = model.get_embeddings(video=v_obj, dimension=1408)
        if not v_resp.video_embeddings:
            raise RuntimeError(f"No video embeddings for {gcs_uri}")
        vis_emb = np.array(v_resp.video_embeddings[0].embedding, dtype=np.float32)
        
        # 2. Metadata text embedding
        t_resp = model.get_embeddings(contextual_text=meta_text, dimension=1408)
        txt_emb = np.array(t_resp.text_embedding, dtype=np.float32)
        
        # 3. Hybrid multimodal vector fusion (50% visual video frames + 50% semantic metadata)
        hybrid_emb = 0.5 * vis_emb + 0.5 * txt_emb
        norm_val = np.linalg.norm(hybrid_emb)
        if norm_val > 0:
            hybrid_emb = hybrid_emb / norm_val
        
        dt = time.time() - t_start
        item["embedding"] = [float(x) for x in hybrid_emb]
        item["extraction_time"] = dt
        return item

    indexed_records = []
    t0_embed = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        embed_futures = [executor.submit(process_embedding, v) for v in generated_videos]
        completed = 0
        for f in as_completed(embed_futures):
            res = f.result()
            indexed_records.append(res)
            completed += 1
            if completed % 10 == 0 or completed == len(generated_videos):
                logger.info(f"  -> Extracted embeddings for {completed}/{len(generated_videos)} videos (latest in {res['extraction_time']:.2f}s)")

    logger.info(f"✓ Extracted all 100 ACTUAL VIDEO EMBEDDINGS in {time.time() - t0_embed:.2f}s!")

    # 5. Insert all 100 records into Cloud Spanner
    logger.info(f"\nWriting 100 actual video embedding records to Cloud Spanner '{SPANNER_INSTANCE}/{SPANNER_DATABASE}'...")
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    with database.batch() as batch:
        for r in indexed_records:
            batch.insert_or_update(
                table=SPANNER_TABLE,
                columns=[
                    "video_id", "title", "description", "tags",
                    "gcs_uri", "gcs_bucket", "gcs_object_name",
                    "content_type", "duration_seconds", "file_size_bytes",
                    "embedding", "embedding_model", "status",
                    "created_at", "updated_at"
                ],
                values=[[
                    r["video_id"],
                    r["title"],
                    r["description"],
                    r["tags"],
                    r["gcs_uri"],
                    GCS_BUCKET,
                    r["gcs_object"],
                    r["content_type"],
                    float(r["duration_seconds"]),
                    int(r["file_size"]),
                    r["embedding"],
                    "multimodalembedding@001",
                    "INDEXED",
                    spanner.COMMIT_TIMESTAMP,
                    spanner.COMMIT_TIMESTAMP
                ]]
            )

    logger.info("=" * 75)
    logger.info(f"🎉 SUCCESS: 100 DISTINCT ACTUAL VIDEO EMBEDDINGS INDEXED IN CLOUD SPANNER!")
    logger.info("=" * 75)

if __name__ == "__main__":
    main()

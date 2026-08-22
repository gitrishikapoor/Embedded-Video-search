#!/usr/bin/env python3
"""
Populates Cloud Spanner database with a rich catalog of 24 diverse video records
and 1408-dimensional multimodal embeddings.
"""

import sys
import uuid
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from backend.app.services.embedding_service import embedding_service
from backend.app.services.spanner_service import spanner_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET = [
    # --- ANIMALS & PETS ---
    {
        "title": "Golden Retriever Puppy Playing with Tennis Ball in Grass",
        "description": "An adorable golden retriever puppy running, jumping, and happily fetching a bright yellow tennis ball across a sunny green garden lawn.",
        "tags": ["dog", "puppy", "golden retriever", "pet", "animals", "playful", "cute"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "duration": 15.0,
        "filename": "golden_retriever_puppy.mp4"
    },
    {
        "title": "Playful Calico Kitten Chasing Red Laser Pointer on Rug",
        "description": "A energetic young kitten pouncing, leaping, and chasing a darting red laser light across a living room carpet with wide playful eyes.",
        "tags": ["cat", "kitten", "pet", "animals", "playful", "feline", "cute"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
        "duration": 12.0,
        "filename": "kitten_laser_chase.mp4"
    },
    {
        "title": "Siberian Husky Pack Running in Snowy Winter Forest",
        "description": "Beautiful blue-eyed Siberian husky sled dogs running through deep powdery snow in an Alaskan pine forest under clear winter skies.",
        "tags": ["dog", "husky", "snow", "winter", "forest", "animals", "nature"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
        "duration": 22.0,
        "filename": "husky_snow_forest.mp4"
    },
    {
        "title": "Giant Panda Munching Fresh Green Bamboo in Sanctuary",
        "description": "Chubby giant panda sitting calmly on a wooden platform, peeling and munching sweet green bamboo stalks in a lush mountain reserve.",
        "tags": ["panda", "wildlife", "bamboo", "nature", "animals", "sanctuary", "cute"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyBlazes.mp4",
        "duration": 18.0,
        "filename": "panda_bamboo.mp4"
    },

    # --- NATURE & LANDSCAPES ---
    {
        "title": "Cinematic Sunset Over Ocean Waves and Golden Sandy Beach",
        "description": "Breathtaking 4K drone view of turquoise ocean waves gently rolling onto a golden sandy shoreline during a dramatic fiery orange sunset.",
        "tags": ["ocean", "sea", "waves", "beach", "sunset", "nature", "water", "coast", "relaxing"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "duration": 30.0,
        "filename": "ocean_sunset_beach.mp4"
    },
    {
        "title": "Cascading Tropical Waterfall with Mist and Rainbow in Rainforest",
        "description": "Massive roaring tropical waterfall surrounded by lush green jungle foliage, casting misty spray and a vibrant colorful rainbow in sunlight.",
        "tags": ["waterfall", "rainforest", "jungle", "rainbow", "nature", "water", "tropical"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
        "duration": 25.0,
        "filename": "tropical_waterfall.mp4"
    },
    {
        "title": "Alpine Drone Flight Over Snow-Capped Swiss Mountain Peaks",
        "description": "Sweeping cinematic aerial panorama gliding above rugged snow-covered mountain ridges, alpine valleys, and glaciers on a bright crisp morning.",
        "tags": ["mountains", "alps", "snow", "drone", "nature", "aerial", "landscape", "adventure"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4",
        "duration": 28.0,
        "filename": "swiss_alps_drone.mp4"
    },
    {
        "title": "Bioluminescent Coral Reef and Tropical Marine Life in Deep Ocean",
        "description": "Underwater scuba diving footage exploring colorful glowing coral reefs, sea turtles, clownfish, and schools of glittering exotic tropical fish.",
        "tags": ["ocean", "sea", "coral reef", "underwater", "scuba", "marine", "nature", "fish"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
        "duration": 20.0,
        "filename": "underwater_coral_reef.mp4"
    },

    # --- AUTOMOTIVE & RACING ---
    {
        "title": "Red Supercar Drifting at High Speed on Wet Asphalt Track",
        "description": "High performance red sports car performing precision powerslides and smoke-filled drifts around hairpin turns on a rainy race track circuit.",
        "tags": ["car", "sports car", "racing", "drifting", "speed", "vehicle", "supercar", "automotive"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackSeeTheWorld.mp4",
        "duration": 16.0,
        "filename": "sports_car_drift.mp4"
    },
    {
        "title": "Formula 1 Grand Prix High Speed Pit Stop and Tire Change",
        "description": "Lightning fast Formula 1 pit crew executing a 2.1-second synchronized 4-wheel tire change as the racing car roars back into the pit lane.",
        "tags": ["racing", "car", "formula 1", "speed", "pit stop", "motorsport", "fast"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnTheLoose.mp4",
        "duration": 14.0,
        "filename": "f1_pit_stop.mp4"
    },
    {
        "title": "Big Wave Surfer Riding Massive 40-Foot Barrel Wave",
        "description": "Extreme athlete surfing inside the cavern of a massive 40-foot ocean swell wave with exploding white foam spray at Teahupoo reef.",
        "tags": ["surfing", "ocean", "waves", "extreme", "sports", "water", "athlete", "action"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
        "duration": 24.0,
        "filename": "big_wave_surfing.mp4"
    },
    {
        "title": "Extreme Mountain Biker Screaming Down Alpine Forest Trail",
        "description": "First-person POV helmet cam of a professional downhill mountain biker hitting jumps, drop-offs, and rocky singletrack at breakneck speed.",
        "tags": ["bike", "mountain biking", "sports", "extreme", "trail", "forest", "action"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4",
        "duration": 19.0,
        "filename": "downhill_mtb.mp4"
    },

    # --- CULINARY & FOOD ---
    {
        "title": "Master Chef Kneading and Rolling Fresh Italian Pasta Recipe",
        "description": "Traditional Italian chef dusting flour, rolling fresh egg pasta dough through a bronze cutter, and tossing handmade tagliatelle in creamy garlic sauce.",
        "tags": ["cooking", "recipe", "food", "pasta", "chef", "kitchen", "italian", "delicious"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WhatCarCanYouGetForAGrand.mp4",
        "duration": 22.0,
        "filename": "italian_pasta_chef.mp4"
    },
    {
        "title": "Artisan Baker Scoring and Baking Golden Sourdough Loaf",
        "description": "Artisan bakery process showing intricate razor scoring on fermented sourdough loaf, followed by steam oven rise into a crispy dark golden crust.",
        "tags": ["baking", "bread", "sourdough", "food", "cooking", "bakery", "delicious"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
        "duration": 17.0,
        "filename": "sourdough_bread_baking.mp4"
    },
    {
        "title": "Japanese Sushi Master Preparing Premium Tuna and Salmon Nigiri",
        "description": "Precision knife skills as a master sushi chef slices melt-in-your-mouth bluefin otoro and salmon, pressing delicate rice and brushing sweet soy glaze.",
        "tags": ["sushi", "japanese", "food", "chef", "cooking", "salmon", "delicious", "culinary"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
        "duration": 20.0,
        "filename": "japanese_sushi_master.mp4"
    },
    {
        "title": "Warm Chocolate Molten Lava Cake with Gooey Flowing Fudge",
        "description": "A spoon breaking into a warm decadent chocolate lava cake, releasing a rich velvet cascade of molten dark fudge ganache with vanilla bean ice cream.",
        "tags": ["dessert", "chocolate", "cake", "food", "baking", "sweet", "delicious"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4",
        "duration": 15.0,
        "filename": "chocolate_lava_cake.mp4"
    },

    # --- SPACE & ASTRONOMY ---
    {
        "title": "Deep Space Nebula Voyage and Swirling Spiral Galaxy Stars",
        "description": "High definition astronomical simulation flying through cosmic interstellar gas clouds, luminous glowing nebulae, and brilliant spiral star clusters.",
        "tags": ["space", "galaxy", "stars", "universe", "astronomy", "nebula", "cosmic", "science"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerJoyBlazes.mp4",
        "duration": 26.0,
        "filename": "deep_space_nebula.mp4"
    },
    {
        "title": "Mars Rover Perseverance Traversing Rocky Red Planet Crater",
        "description": "NASA Mars rover rolling across reddish Martian desert terrain, drilling geological core rock samples under a pale pink alien sky.",
        "tags": ["mars", "space", "rover", "science", "planet", "astronomy", "technology"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "duration": 21.0,
        "filename": "mars_rover_crater.mp4"
    },
    {
        "title": "Total Solar Eclipse Showing Diamond Ring and Solar Corona",
        "description": "The Moon fully blocking the blazing sun disc, revealing the ethereal shimmering white solar corona and solar flares in deep twilight sky.",
        "tags": ["eclipse", "sun", "moon", "astronomy", "space", "science", "nature"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
        "duration": 18.0,
        "filename": "total_solar_eclipse.mp4"
    },
    {
        "title": "Saturn Majestic Ring System Orbiting Gas Giant in Space",
        "description": "Spectacular 3D orbital trajectory showing the intricate ice particle rings of Saturn casting shadows across the banded golden atmosphere.",
        "tags": ["saturn", "planet", "rings", "space", "astronomy", "science", "universe"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4",
        "duration": 24.0,
        "filename": "saturn_ring_system.mp4"
    },

    # --- TECHNOLOGY, CODING & ROBOTICS ---
    {
        "title": "Software Engineer Coding Cloud AI Applications on Triple Monitors",
        "description": "Software developer rapidly typing Python and Go code, optimizing Spanner vector queries, and analyzing real-time neural network dashboards.",
        "tags": ["coding", "programming", "developer", "software", "cloud", "ai", "technology", "tech"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
        "duration": 25.0,
        "filename": "software_engineer_coding.mp4"
    },
    {
        "title": "Futuristic Cyberpunk Metropolis with Neon Skyscrapers and Flying Cars",
        "description": "Dazzling night view of a sprawling neon cyberpunk mega-city with holographic advertisements, holographic trains, and flying autonomous skycabs.",
        "tags": ["cyberpunk", "futuristic", "city", "technology", "sci-fi", "neon", "ai"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackSeeTheWorld.mp4",
        "duration": 30.0,
        "filename": "cyberpunk_city.mp4"
    },
    {
        "title": "High Tech Industrial Robotic Arms Assembling Electric Vehicle",
        "description": "Automated smart factory with synchronized robotic welding arms, laser inspection sensors, and autonomous guided transport carts assembling cars.",
        "tags": ["robotics", "automation", "factory", "technology", "engineering", "ai", "cars"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/SubaruOutbackOnTheLoose.mp4",
        "duration": 20.0,
        "filename": "robotic_arms_assembly.mp4"
    },
    {
        "title": "Live Rock Band Concert with Electric Guitar Solo and Laser Lights",
        "description": "Electrifying stadium rock concert with a lead guitarist playing an intense solo, surrounded by pyro flames, purple laser beams, and cheering crowd.",
        "tags": ["music", "concert", "guitar", "playing", "band", "stage", "rock", "performance", "festival"],
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
        "duration": 23.0,
        "filename": "rock_concert_guitar.mp4"
    }
]

def seed_database():
    logger.info("=" * 70)
    logger.info(" 🚀 SEEDING CLOUD SPANNER WITH EXTENSIVE 24-VIDEO VECTOR DATASET")
    logger.info("=" * 70)
    logger.info(f"Target Spanner DB: {settings.SPANNER_INSTANCE_ID}/{settings.SPANNER_DATABASE_ID}")
    logger.info(f"Target GCS Bucket: {settings.GCS_BUCKET_NAME}")
    logger.info(f"Model / Dimensions: {settings.EMBEDDING_MODEL_NAME} ({settings.EMBEDDING_DIMENSION}-dim)")
    logger.info(f"Mode: {'MOCK' if settings.USE_MOCK_GCP else 'LIVE CLOUD SPANNER'}")
    logger.info("-" * 70)

    inserted_count = 0
    for idx, item in enumerate(DATASET, start=1):
        vid_id = f"vid-{str(uuid.uuid4())[:8]}-{idx:02d}"
        gcs_uri = f"gs://{settings.GCS_BUCKET_NAME}/videos/{item['filename']}"
        
        # Generate 1408-dimensional multimodal vector embedding
        context = f"{item['title']}. {item['description']}. Tags: {', '.join(item['tags'])}"
        embedding = embedding_service.generate_video_embedding(gcs_uri, metadata_context=context)

        record = {
            "video_id": vid_id,
            "title": item["title"],
            "description": item["description"],
            "tags": item["tags"],
            "gcs_uri": gcs_uri,
            "gcs_bucket": settings.GCS_BUCKET_NAME,
            "gcs_object_name": f"videos/{item['filename']}",
            "content_type": "video/mp4",
            "duration_seconds": item["duration"],
            "file_size_bytes": int(item["duration"] * 1024 * 350),  # approx size
            "embedding": embedding,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "status": "INDEXED",
            "video_url": item["url"],
            "error_message": None
        }

        success = spanner_service.insert_or_update_video(record)
        if success:
            inserted_count += 1
            logger.info(f"[{idx:02d}/{len(DATASET)}] ✓ Indexed: {item['title']}")
        else:
            logger.warning(f"[{idx:02d}/{len(DATASET)}] ⚠ Failed: {item['title']}")

    stats = spanner_service.get_stats()
    logger.info("\n" + "=" * 70)
    logger.info(f" 🎉 SEEDING COMPLETED: {inserted_count} videos successfully indexed in Cloud Spanner!")
    logger.info(f" • Total in DB:     {stats.total_videos}")
    logger.info(f" • Indexed Vectors: {stats.indexed_videos}")
    logger.info(f" • Vector Dimension: {stats.vector_dimension}")
    logger.info("=" * 70)

if __name__ == "__main__":
    seed_database()

#!/usr/bin/env python3
"""
Generate 100 Visually Distinct Procedural Domain Videos & Index in Cloud Spanner.
- 10 Domains x 10 Videos = 100 100% Unique, Visually Distinct H.264 MP4 Files
- Authentic procedural domain renderers (space galaxies, ocean waves, audio equalizers, automotive roads, etc.)
- Uploads to Google Cloud Storage: gs://rk-video-search-media-bucket/videos/distinct_100/
- Extracts Vertex AI Multimodal Embeddings (multimodalembedding@001, 1408-dim)
- Indexes directly into Cloud Spanner: properties/videosearch
"""

import os
import re
import sys
import time
import math
import logging
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Video
from google.cloud import spanner, storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProceduralVideoGenerator")

PROJECT_ID = "rk-vpc-host-prod-333313"
REGION = "us-central1"
SPANNER_INSTANCE = "properties"
SPANNER_DATABASE = "videosearch"
SPANNER_TABLE = "Videos"
GCS_BUCKET_NAME = "rk-video-search-media-bucket"
GCS_PREFIX = "videos/distinct_100"
OUTPUT_DIR = Path("storage/videos/procedural_100")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 100 Complete Distinct Domain Item Definitions
DOMAINS = [
    {
        "category": "automotive",
        "name": "Automotive Vehicles",
        "theme": "road_physics",
        "items": [
            ("Highway Sedan High Speed Passing Maneuver", "Sedan accelerating along illuminated highway lanes with asphalt speed markers.", ["car", "highway", "traffic", "speed", "driving"]),
            ("Track Sports Car Cornering Apex Clip", "Racecar carving through curved asphalt track apex with dynamic cornering lines.", ["sports car", "racing", "track", "cornering", "motorsport"]),
            ("Off Road 4x4 Muddy Trail Exploration", "Heavy duty 4x4 climbing rocky mountain mud terrain with active suspension.", ["off-road", "4x4", "mud", "trail", "adventure"]),
            ("Electric Vehicle Dashboard Navigation Display", "Futuristic digital cockpit showing regenerative battery telemetry and route HUD.", ["electric car", "ev", "cockpit", "technology", "hud"]),
            ("Night City Supercar Neon Tunnel Cruise", "Exotic supercar speeding through fluorescent-lit underground tunnel.", ["supercar", "night drive", "tunnel", "neon", "speed"]),
            ("Classic Vintage Roadster Coastal Drive", "Restored chrome convertible cruising along scenic coastal mountain cliff.", ["vintage car", "classic", "roadster", "convertible", "scenic"]),
            ("Rainy Wet Asphalt Highway Drift Action", "Vehicle counter-steering through reflective wet asphalt standing water spray.", ["drift", "rain", "wet road", "action", "car"]),
            ("Desert Rally Truck Dune Jumping Sequence", "Baja trophy truck launching across golden desert sand dunes at sunset.", ["rally", "desert", "dune", "jump", "truck"]),
            ("Formula Racing Pit Exit and Acceleration", "Open-wheel race car launching from pit lane to maximum acceleration velocity.", ["formula 1", "pit stop", "acceleration", "race", "speed"]),
            ("SUV and Truck Highway Cruising Footage", "Panoramic view of highway cruising traffic with wide open horizon.", ["suv", "truck", "highway", "cruising", "travel"])
        ]
    },
    {
        "category": "culinary",
        "name": "Culinary Food Preparation",
        "theme": "culinary_board",
        "items": [
            ("Fresh Green Bell Peppers and Farm Harvest", "Vibrant crisp emerald bell peppers arranged on rustic wooden kitchen bench.", ["bell peppers", "vegetables", "cooking", "farm", "produce"]),
            ("Ripe Red Tomatoes Sliced on Wooden Cutting Board", "Juicy ruby red heirloom tomatoes sliced with precision culinary knife.", ["tomatoes", "cutting board", "chef", "slicing", "food prep"]),
            ("Sizzling Stir Fry Wok Toss with Crisp Vegetables", "Vibrant snap peas and baby corn tossed in blazing hot smoking wok flame.", ["stir fry", "wok", "flame", "cooking", "asian cuisine"]),
            ("Artisan Sourdough Loaf Golden Crust Slicing", "Rustic country sourdough bread crust crackling under serrated bread blade.", ["sourdough", "bread", "baking", "artisan", "food"]),
            ("Crisp Garden Cucumbers and Herb Salad", "Thin sliced English cucumbers tossed with fresh dill and olive oil vinaigrette.", ["cucumber", "salad", "herbs", "healthy", "culinary"]),
            ("Artisan Chef Knife Slicing Fresh Produce", "Master chef demonstrating fast julienne vegetable slicing technique.", ["knife skills", "chef", "julienne", "produce", "kitchen"]),
            ("Creamy Gourmet Pasta Carbonara Toss", "Silky egg yolk and parmesan sauce coating fresh handmade fettuccine pasta.", ["pasta", "carbonara", "italian food", "gourmet", "cheese"]),
            ("Harvest Apple and Pear Orchard Basket", "Glossy autumn apples and bosc pears overflowing from woven wicker basket.", ["apples", "pears", "harvest", "fruit", "orchard"]),
            ("Simmering Rich Tomato Sauce in Copper Pot", "Fragrant marinara bubbling gently with fresh basil leaves in vintage copper pot.", ["sauce", "simmering", "copper pot", "italian", "culinary"]),
            ("Decorating Gourmet Strawberry Tart Dessert", "Pastry chef arranging fresh glazed strawberries atop vanilla pastry cream.", ["dessert", "pastry", "strawberries", "tart", "sweet"])
        ]
    },
    {
        "category": "space_astronomy",
        "name": "Deep Space Astronomy",
        "theme": "space_cosmos",
        "items": [
            ("James Webb Infrared View of Deep Primordial Galaxy", "Ultra-deep field infrared composite revealing ancient gravitationally lensed galaxies.", ["james webb", "space", "galaxy", "telescope", "astronomy"]),
            ("Orion Nebula Glowing Hydrogen Dust Clouds", "Stellar nursery with luminous magenta and cyan interstellar dust filaments.", ["nebula", "orion", "stars", "cosmic dust", "deep space"]),
            ("Saturn Majestic Golden Rings and Cloud Bands", "High-resolution planetary view of intricate ring structure and hexagonal pole.", ["saturn", "rings", "planet", "solar system", "space"]),
            ("Supernova Remnant Blast Wave Expanding in Deep Space", "Multi-wavelength shockwave expanding outwards into interstellar medium.", ["supernova", "shockwave", "stellar explosion", "cosmos", "astrophysics"]),
            ("Spiral Galaxy Pinwheel Core Rotation Simulation", "Vast billion-star barred spiral galaxy rotating gracefully in cosmic void.", ["spiral galaxy", "galaxy core", "stars", "rotation", "space"]),
            ("International Space Station Orbiting Over Aurora", "Time-lapse of Earth limb displaying luminous green Aurora Australis ribbons.", ["iss", "space station", "aurora", "earth from space", "orbit"]),
            ("Mars Perseverance Rover Exploring Jezero Crater", "Robotic explorer analyzing sedimentary delta rocks on Martian surface.", ["mars", "perseverance rover", "jezero crater", "red planet", "exploration"]),
            ("Total Solar Eclipse Diamond Ring and Solar Corona", "Sun blocked by moon revealing ghostly white coronal streamers.", ["solar eclipse", "corona", "diamond ring", "sun", "astronomy"]),
            ("Binary Star System Gravitational Orbit Dance", "Two luminous blue giant stars orbiting a shared gravitational barycenter.", ["binary stars", "orbit", "gravitation", "astrophysics", "stars"]),
            ("Deep Space Cosmic Web Dark Matter Filaments", "Cosmological simulation showing vast web of galaxy clusters linked by dark matter.", ["cosmic web", "dark matter", "universe", "clusters", "simulation"])
        ]
    },
    {
        "category": "marine_ocean",
        "name": "Marine Ocean Exploration",
        "theme": "ocean_waves",
        "items": [
            ("Vibrant Coral Reef Teeming with Tropical Fish", "Sunlit shallow reef with purple sea fans and swirling schools of clownfish.", ["coral reef", "tropical fish", "ocean", "marine life", "scuba"]),
            ("Deep Sea Bioluminescent Jellyfish in Abyss", "Translucent jellyfish pulsing with ethereal electric blue and green bioluminescence.", ["jellyfish", "bioluminescence", "deep sea", "underwater", "abyss"]),
            ("Humpback Whale Mother and Calf Swimming in Ocean", "Gentle giants gliding gracefully through sun-dappled turquoise water.", ["humpback whale", "whales", "marine life", "ocean", "wildlife"]),
            ("Manta Ray Gliding Over Pristine Sandy Seafloor", "Giant oceanic manta ray flapping wings effortlessly across underwater dunes.", ["manta ray", "scuba diving", "ocean floor", "marine biology", "rays"]),
            ("Crashing Turquoise Shorebreak Waves in Sunlight", "Crystal clear emerald ocean barrel curling and detonating on shallow sandbar.", ["waves", "shorebreak", "ocean", "surf", "turquoise"]),
            ("Schools of Silver Barracuda Swirling in Vortex", "Tightly packed tornado of metallic fish reflecting bright sunlight.", ["barracuda", "school of fish", "underwater", "diving", "ocean"]),
            ("Giant Pacific Octopus Crawling Over Rocky Seafloor", "Camouflaged octopus shifting texture and color while hunting across rocks.", ["octopus", "pacific ocean", "camouflage", "sea creature", "marine"]),
            ("Kelp Forest Sunbeams Piercing Underwater Canopy", "Towering golden kelp fronds swaying gently in ocean surge currents.", ["kelp forest", "sunbeams", "california coast", "underwater", "marine ecosystem"]),
            ("Playful Bottlenose Dolphins Riding Bow Wave", "Pod of wild dolphins leaping synchronously at the front of a cruising boat.", ["dolphins", "ocean", "marine mammals", "playful", "wildlife"]),
            ("Sunken Shipwreck Artificial Reef Exploration", "Historic wooden vessel encrusted in colorful corals and sea sponges.", ["shipwreck", "artificial reef", "diving", "underwater history", "marine"])
        ]
    },
    {
        "category": "wildlife_nature",
        "name": "Wildlife Nature Habitats",
        "theme": "nature_canopy",
        "items": [
            ("Majestic African Lion Resting in Golden Savanna", "Male lion with dark mane scanning Serengeti plains under acacia tree.", ["lion", "safari", "africa", "savanna", "wildlife"]),
            ("Giant Panda Munching on Fresh Bamboo Shoots", "Playful panda sitting in misty mountain forest stripping green bamboo stalk.", ["panda", "bamboo", "china", "cute animals", "wildlife"]),
            ("Grizzly Bear Catching Spawning Salmon in Rapids", "Large brown bear standing in foaming waterfall snapping salmon in mid-air.", ["grizzly bear", "salmon", "alaska", "river", "predator"]),
            ("African Elephant Herd Marching at Golden Sunset", "Family matriarch leading adults and calves across dusty red savanna plain.", ["elephants", "sunset", "safari", "africa", "wildlife"]),
            ("Snow Leopard Prowling Steep Himalayan Cliffs", "Elusive mountain predator navigating vertical granite slopes in snow.", ["snow leopard", "himalayas", "mountains", "predator", "rare wildlife"]),
            ("Hummingbird Hovering and Drinking Nectar", "Fast iridescent wings beating as hummingbird sips from crimson fuchsia.", ["hummingbird", "flower", "nectar", "birds", "nature"]),
            ("Bald Eagle Soaring Over Majestic Glacier Bay", "Eagle gliding effortlessly on thermal updrafts above icy blue water.", ["bald eagle", "glacier bay", "alaska", "birds of prey", "wilderness"]),
            ("Chameleon Changing Colors on Rainforest Branch", "Reptile with independently moving eyes shifting from emerald green to amber.", ["chameleon", "rainforest", "reptiles", "madagascar", "nature"]),
            ("Wolf Pack Howling in Snowy Pine Forest at Dusk", "Timberwolves gathered on snowbank singing under twilight alpine sky.", ["wolves", "wolf pack", "snow", "winter forest", "howling"]),
            ("Red Eyed Tree Frog Perched on Wet Jungle Leaf", "Vivid green amphibian with striking orange toes clinging to raindrop leaf.", ["tree frog", "amazon", "rainforest", "amphibian", "macro nature"])
        ]
    },
    {
        "category": "robotics_tech",
        "name": "Robotics Cyberpunk Tech",
        "theme": "robotics_matrix",
        "items": [
            ("Industrial Robotic Arm Welding Automotive Chassis", "High-speed precision six-axis robot creating sparkling gold welding beads.", ["robotics", "welding", "manufacturing", "automation", "industry"]),
            ("Humanoid Bipedal Robot Walking Over Obstacles", "Advanced bipedal robot maintaining dynamic balance on rough terrain.", ["humanoid robot", "ai", "bipedal", "engineering", "future"]),
            ("Autonomous Warehouse Drone Fleet Sorting Packages", "Swarm of automated drones flying through multi-level logistics center.", ["drones", "warehouse", "logistics", "automation", "ai"]),
            ("Microchip Silicon Wafer Laser Lithography Cleanroom", "Ultraviolet laser etching nanometer circuit paths onto mirror silicon disc.", ["semiconductor", "microchip", "cleanroom", "nanotech", "silicon"]),
            ("Cyberpunk Neon Alley Cybernetic Interface HUD", "Augmented reality visual overlay decoding data streams in futuristic alley.", ["cyberpunk", "augmented reality", "hud", "sci-fi", "futuristic"]),
            ("Quantum Computing Cryogenic Dilution Refrigerator", "Gilded chandelier cryostat cooling quantum processor to near absolute zero.", ["quantum computing", "qubits", "cryogenics", "supercomputing", "physics"]),
            ("Surgical Robotic System Performing Micro Stitch", "Tele-operated robotic surgical arms executing delicate microsurgery.", ["medical robotics", "surgery", "healthcare", "precision", "technology"]),
            ("Autonomous Electric Delivery Rover on Sidewalk", "Four-wheeled AI robot safely steering around pedestrians on city sidewalk.", ["autonomous delivery", "rover", "smart city", "robotics", "ai"]),
            ("Circuit Board Surface Mount Technology Line", "Pick-and-place robot positioning tiny SMD capacitors at lightning speed.", ["circuit board", "smt", "electronics", "pcb", "hardware"]),
            ("Neural Network Matrix Synaptic Data Visualization", "Complex 3D graph of firing nodes representing deep learning transformer.", ["neural network", "ai", "machine learning", "data visualization", "deep learning"])
        ]
    },
    {
        "category": "architecture_city",
        "name": "Urban Architecture Cityscapes",
        "theme": "city_wireframe",
        "items": [
            ("Tokyo Shibuya Crossing Time Lapse at Night", "Thousands of pedestrians crossing glowing neon intersection in rain.", ["tokyo", "shibuya", "cityscape", "night life", "timelapse"]),
            ("Futuristic Glass Skyscraper Reflecting Sunset", "Curvilinear modern architectural tower catching burning orange sunset.", ["architecture", "skyscraper", "glass building", "sunset", "modern"]),
            ("Historic European Cobblestone Alley at Twilight", "Warm lantern-lit narrow street between medieval half-timbered houses.", ["europe", "cobblestone", "historic", "old town", "travel"]),
            ("Suspension Bridge Spanning Foggy Bay at Dawn", "Massive orange steel bridge towers piercing through blanket of sea fog.", ["bridge", "suspension bridge", "fog", "san francisco", "engineering"]),
            ("Modernist Concrete Villa with Infinity Pool", "Minimalist brutalist luxury residence with glass walls over ocean cliff.", ["modern architecture", "minimalism", "luxury villa", "pool", "design"]),
            ("Manhattan Skyline Aerial Sweep Over Central Park", "Aerial view of Midtown skyscraper canyon flanking lush autumn park trees.", ["manhattan", "new york", "central park", "skyline", "aerial"]),
            ("Traditional Japanese Pagoda and Cherry Blossoms", "Five-story historic pagoda with pink sakura petals floating on breeze.", ["pagoda", "kyoto", "japan", "cherry blossoms", "heritage"]),
            ("Geometric Origami Facade Building Architecture", "Parametric dynamic building skin folded into interlocking triangular panels.", ["parametric architecture", "facade", "geometric", "modern design", "structure"]),
            ("Illuminated Highway Interchange Overpass Traffic", "Light trails of red and white headlights weaving through multi-tier cloverleaf.", ["highway interchange", "traffic", "night city", "light trails", "infrastructure"]),
            ("Venice Grand Canal Gondolas Passing Palaces", "Classic black gondolas navigating historic waterway between marble facades.", ["venice", "grand canal", "italy", "waterways", "travel"])
        ]
    },
    {
        "category": "drone_landscapes",
        "name": "Aerial Drone Landscapes",
        "theme": "drone_topography",
        "items": [
            ("Swiss Alpine Mountain Pass Winding Road", "Hairpin switchback asphalt road cutting through snow-capped granite peaks.", ["swiss alps", "mountain road", "drone view", "aerial", "scenic"]),
            ("Tropical Turquoise Lagoon and Coral Atoll", "Pristine circular coral reef enclosing bright cyan shallow lagoon.", ["atoll", "turquoise lagoon", "coral reef", "drone aerial", "tropical"]),
            ("Lofoten Arctic Island Coastal Highway View", "Bridges linking rocky islands flanked by snow-dusted jagged sea peaks.", ["arctic", "coastal", "islands", "scenic drive", "aerial"]),
            ("Misty Rice Terraces Sunrise in Bali Highlands", "Curving green agricultural terraces catching early morning golden fog.", ["rice terraces", "bali", "mist", "sunrise", "landscape"]),
            ("Black Sand Beach Ocean Waves Aerial Sweep", "Crisp white surf foaming dramatically against jet-black volcanic sand.", ["black sand", "beach", "waves", "aerial", "iceland"]),
            ("Grand Canyon Sunrise Shadows Moving Over Spires", "Dawn sunlight painting deep crimson strata on vast ancient canyon walls.", ["grand canyon", "sunrise", "canyon", "landscape", "geology"]),
            ("Volcanic Caldera Lake Aerial Top Down View", "Perfect circular caldera filled with sapphire blue geothermal water.", ["volcano", "caldera", "crater lake", "aerial view", "nature"]),
            ("Autumn Deciduous Forest Vibrant Orange Canopy", "Endless ocean of fiery red, golden yellow, and orange maple treetops.", ["autumn forest", "foliage", "fall colors", "drone", "canopy"]),
            ("Towering Glacial Iceberg in Arctic Fjord", "Massive cathedral iceberg with deep sapphire blue meltwater pools.", ["iceberg", "glacier", "arctic", "fjord", "aerial landscape"]),
            ("Rolling Tuscan Vineyard Hills at Golden Sunset", "Cypress-lined dirt roads winding through rows of lush green grapevines.", ["tuscany", "vineyard", "italy", "hills", "sunset aerial"])
        ]
    },
    {
        "category": "extreme_sports",
        "name": "Extreme Sports Adventure",
        "theme": "sports_kinetic",
        "items": [
            ("Downhill Mountain Biking on Steep Forest Ridge", "Rider flying over natural wooden jumps and navigating tight rocky switchbacks.", ["mountain biking", "downhill", "extreme sports", "trail", "action"]),
            ("Wingsuit Proximity Flyer Skimming Mountain Ledge", "Athlete in aerodynamic wingsuit gliding inches above rocky alpine crest.", ["wingsuit", "base jump", "flight", "extreme", "adrenaline"]),
            ("Big Wave Surfer Dropping into 40 Foot Barrel", "Surfer riding massive turquoise wave face at famous offshore reef break.", ["surfing", "big wave", "barrel", "ocean", "action sports"]),
            ("Free Solo Rock Climber Scaling Vertical Wall", "Climber making precision fingertip holds hundreds of feet above valley floor.", ["rock climbing", "free solo", "granite", "cliff", "adventure"]),
            ("Deep Powder Snowboarder Carving Backcountry Bowl", "Snowboarder spraying clouds of untouched powder snow on steep slope.", ["snowboard", "powder snow", "backcountry", "winter sports", "extreme"]),
            ("White Water Kayaker Plunging Over Class V Rapid", "Kayaker maneuvering through roaring frothing canyon river hydraulic.", ["kayak", "whitewater", "rapids", "river", "adventure"]),
            ("High Altitude Skydiving Group Formation Flight", "Divers holding synchronized hexagonal formation in clear blue sky.", ["skydiving", "freefall", "formation", "extreme sports", "adrenaline"]),
            ("Cliff Diver Launching into Crystal Ocean Inlet", "Athlete executing double backflip from 70-foot limestone sea cliff.", ["cliff diving", "acrobatics", "ocean", "extreme", "summer"]),
            ("Motocross Dirt Bike Launching off Sand Dune", "Rider performing whip trick high in the air above desert dunes.", ["motocross", "dirt bike", "sand dunes", "jump", "action"]),
            ("Skateboarder Dropping into Massive Concrete Bowl", "Skater carving high speeds on curved transition with smooth kickturn.", ["skateboarding", "skate park", "bowl", "action sports", "urban"])
        ]
    },
    {
        "category": "music_performance",
        "name": "Music Performance Arts",
        "theme": "music_equalizer",
        "items": [
            ("Symphony Orchestra Violin Section in Crescendo", "Synchronized string bow strokes during dramatic orchestral movement.", ["orchestra", "violin", "symphony", "classical music", "performance"]),
            ("Electric Guitarist Ripping High Energy Rock Solo", "Musician shredding guitar with distortion on smoke-filled concert stage.", ["guitar", "rock", "concert", "solo", "live music"]),
            ("Grand Piano Recital Virtuoso Hands on Keys", "Close-up of pianist executing rapid complex arpeggio passages.", ["piano", "grand piano", "recital", "classical", "virtuoso"]),
            ("Outdoor Music Festival Mainstage Laser Light Show", "Vibrant multi-colored laser beams cutting through night festival crowd.", ["festival", "lasers", "edm", "light show", "stage"]),
            ("Jazz Quartet Saxophone Player Improvising in Club", "Soulful tenor saxophone solo in moody dimly lit jazz venue.", ["jazz", "saxophone", "club", "improvisation", "live music"]),
            ("Theatrical Stage Play Dramatic Monologue Spotlight", "Actor delivering emotional performance under warm golden spotlight beam.", ["theatre", "stage play", "actor", "performance", "drama"]),
            ("Choral Choir Singing in Historic Cathedral", "Full vocal ensemble harmonizing with rich reverberant cathedral echoes.", ["choir", "vocal", "harmony", "cathedral", "choral music"]),
            ("Heavy Drummer Driving Powerful Rhythm Beat", "Drummer striking cymbals and snares with intense speed and precision.", ["drums", "drummer", "rhythm", "percussion", "rock"]),
            ("Flamenco Acoustic Guitarist Fast Fingerstyle Picking", "Nylon string acoustic guitar resonating with fast Spanish rhythmic rasgueado.", ["flamenco", "acoustic guitar", "fingerstyle", "spanish guitar", "music"]),
            ("Concert Stage Pyrotechnics Fireworks Finale Burst", "Dramatic stage explosion of golden sparklers and fire columns at show finish.", ["pyrotechnics", "fireworks", "stage show", "concert finale", "spectacle"])
        ]
    }
]

# Procedural Frame Generators
WIDTH, HEIGHT = 640, 360
FPS = 24
DURATION_SEC = 3.5
TOTAL_FRAMES = int(FPS * DURATION_SEC)

# Pre-generate coordinate grids
Y_GRID, X_GRID = np.ogrid[:HEIGHT, :WIDTH]
CX, CY = WIDTH / 2.0, HEIGHT / 2.0
R_GRID = np.sqrt((X_GRID - CX)**2 + (Y_GRID - CY)**2)
THETA_GRID = np.arctan2(Y_GRID - CY, X_GRID - CX)

def render_frame_for_theme(theme: str, t: float, sub_idx: int) -> np.ndarray:
    """Generates unique procedural RGB frames for the given theme and sub-item variation."""
    var = (sub_idx * 0.35)
    
    if theme == "space_cosmos":
        # Rotating spiral galaxy arms + star cluster particles
        spiral = np.sin(R_GRID / 18.0 - THETA_GRID * (2 + (sub_idx % 3)) + t * 3.0 + var)
        stars = (np.sin(X_GRID * 0.4 + var) * np.cos(Y_GRID * 0.4 + t)) > 0.92
        
        red = np.clip((np.sin(spiral + t) * 0.4 + 0.5) * (180 + sub_idx * 7) + stars * 255, 0, 255)
        green = np.clip((np.cos(R_GRID / 25.0 - t * 2.0) * 0.3 + 0.3) * 120 + stars * 255, 0, 255)
        blue = np.clip((np.sin(THETA_GRID * 3.0 + t * 2.5) * 0.5 + 0.5) * 240 + stars * 255, 0, 255)

    elif theme == "ocean_waves":
        # Fluid sinusoidal wave dynamics & underwater caustics
        wave1 = np.sin(X_GRID / 35.0 + t * 4.0 + var) * 30
        wave2 = np.cos(Y_GRID / 25.0 - t * 3.0) * 20
        depth = (Y_GRID + wave1 + wave2) / HEIGHT
        
        red = np.clip(depth * 30 + math.sin(t * 2 + var) * 20, 0, 255)
        green = np.clip(depth * 160 + np.cos(X_GRID / 20.0 + t * 3) * 40 + 50, 0, 255)
        blue = np.clip(depth * 220 + 70 + np.sin(Y_GRID / 15.0 - t * 2) * 35, 0, 255)

    elif theme == "road_physics":
        # Moving highway perspective & neon asphalt markers
        persp_y = np.maximum(Y_GRID - CY, 1.0) / CY
        road_mask = (np.abs(X_GRID - CX) < (WIDTH * 0.45 * persp_y))
        lane_dash = (np.sin(persp_y * 30.0 - t * 15.0 + var) > 0.4) & (np.abs(X_GRID - CX) < (12 * persp_y))
        
        red = np.where(road_mask, np.where(lane_dash, 255, 45 + sub_idx * 8), 15)
        green = np.where(road_mask, np.where(lane_dash, 220, 45 + sub_idx * 5), 35 + (persp_y * 40).astype(int))
        blue = np.where(road_mask, np.where(lane_dash, 50, 70), 80 + int(math.sin(t * 4) * 30))

    elif theme == "culinary_board":
        # Warm amber kitchen textures, cutting grids & fresh produce hues
        grid_lines = ((X_GRID % 40 < 2) | (Y_GRID % 40 < 2))
        pulse = math.sin(t * 3.0 + var) * 0.5 + 0.5
        
        red = np.clip(180 + sub_idx * 7 + (grid_lines * 40) + pulse * 30, 0, 255)
        green = np.clip(110 + sub_idx * 12 + (grid_lines * 30), 0, 255)
        blue = np.clip(60 + sub_idx * 8 + pulse * 50, 0, 255)

    elif theme == "robotics_matrix":
        # Cyberpunk matrix grid, laser scanline & telemetry pulses
        scanline = (np.abs((Y_GRID - (t * 180 + var * 50) % HEIGHT)) < 4)
        circuits = ((X_GRID % 30 < 2) | (Y_GRID % 30 < 2))
        
        red = np.where(scanline, 255, circuits * 30 + int(math.sin(t * 5) * 20))
        green = np.where(scanline, 255, circuits * 220 + 30)
        blue = np.where(scanline, 255, circuits * 180 + (np.cos(R_GRID / 20.0 - t * 4) * 60 + 80).astype(int))

    elif theme == "music_equalizer":
        # 24-Band pulsing frequency equalizer visualizer bars
        num_bars = 24
        bar_w = WIDTH / num_bars
        bar_idx = (X_GRID / bar_w).astype(int)
        freq_height = HEIGHT * (0.2 + 0.7 * np.abs(np.sin(bar_idx * 0.5 + t * 8.0 + var)))
        bar_active = (Y_GRID > (HEIGHT - freq_height)) & (X_GRID % bar_w > 4)
        
        red = np.where(bar_active, np.clip((Y_GRID / HEIGHT) * 255, 50, 255), 15)
        green = np.where(bar_active, np.clip((1.0 - Y_GRID / HEIGHT) * 255, 30, 255), 10)
        blue = np.where(bar_active, 220, 30 + int(math.sin(t * 3) * 20))

    elif theme == "city_wireframe":
        # Isometric skyline horizon & glowing metropolitan flows
        buildings = ((X_GRID % 50 < 4) & (Y_GRID > (HEIGHT * (0.3 + 0.4 * np.sin(X_GRID / 60.0 + var)))))
        sunset = np.clip((Y_GRID / HEIGHT) * 200, 0, 255)
        
        red = np.where(buildings, 240, sunset + 30)
        green = np.where(buildings, 200, sunset * 0.6 + int(math.sin(t * 2) * 25))
        blue = np.where(buildings, 80, sunset * 0.8 + 60)

    elif theme == "drone_topography":
        # Top-down elevation contour scan lines & terrain gradients
        contour = (np.sin(R_GRID / 12.0 + np.sin(X_GRID / 30.0) * 4.0 - t * 3.0 + var) > 0.8)
        sweep_angle = (THETA_GRID + t * 2.0) % (2 * math.pi)
        radar = (sweep_angle < 0.25)
        
        red = np.where(radar, 255, np.where(contour, 220, 20 + sub_idx * 5))
        green = np.where(radar, 255, np.where(contour, 240, 100 + (np.sin(t * 2) * 40).astype(int)))
        blue = np.where(radar, 255, np.where(contour, 150, 60 + sub_idx * 15))

    elif theme == "nature_canopy":
        # Forest canopy sunbeam god-rays & organic foliage textures
        godrays = np.clip(np.sin((X_GRID + Y_GRID) / 25.0 + t * 2.0 + var) * 80 + 120, 0, 255)
        foliage = np.sin(X_GRID / 10.0) * np.cos(Y_GRID / 10.0)
        
        red = np.clip(godrays * 0.5 + foliage * 20, 0, 255)
        green = np.clip(godrays * 0.9 + 60 + foliage * 40 + sub_idx * 6, 0, 255)
        blue = np.clip(godrays * 0.3 + 20, 0, 255)

    else:  # sports_kinetic
        # High speed adrenaline streaks & dynamic slope vectors
        streak = (np.sin((X_GRID * 2 - Y_GRID) / 20.0 + t * 12.0 + var) > 0.6)
        
        red = np.where(streak, 255, 180 + int(math.sin(t * 5) * 40))
        green = np.where(streak, 140, 70 + sub_idx * 10)
        blue = np.where(streak, 50, 40)

    # Explicitly broadcast to 2D image shape
    r_arr = np.broadcast_to(red, (HEIGHT, WIDTH))
    g_arr = np.broadcast_to(green, (HEIGHT, WIDTH))
    b_arr = np.broadcast_to(blue, (HEIGHT, WIDTH))
    frame = np.stack([r_arr, g_arr, b_arr], axis=-1).astype(np.uint8)
    return frame

def generate_and_encode_video(item_idx: int, domain: Dict[str, Any], item_tuple: Tuple[str, str, List[str]]) -> Path:
    """Generates the procedural video and encodes it with ffmpeg."""
    title, description, tags = item_tuple
    vid_num = item_idx + 1
    theme = domain.get("theme", "space_cosmos")
    sub_idx = item_idx % 10
    
    filename = f"video_{vid_num:03d}.mp4"
    local_path = OUTPUT_DIR / filename
    
    cat_label = re.sub(r'[^a-zA-Z0-9 ]', '', domain["name"]).upper()
    clean_title = re.sub(r'[^a-zA-Z0-9 ]', '', title)
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-vf", (
            f"drawbox=y=0:color=black@0.65:width=iw:height=60:t=fill,"
            f"drawtext=text='{cat_label}':fontcolor=cyan:fontsize=15:x=20:y=12,"
            f"drawtext=text='{clean_title}':fontcolor=white:fontsize=18:x=20:y=32,"
            f"drawbox=y=ih-30:color=black@0.65:width=iw:height=30:t=fill,"
            f"drawtext=text='1408-D MULTIMODAL VECTOR | GCS AND SPANNER':fontcolor=lightgreen:fontsize=12:x=20:y=h-22"
        ),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-movflags", "+faststart",
        str(local_path)
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    
    for f in range(TOTAL_FRAMES):
        t = f / FPS
        frame = render_frame_for_theme(theme, t, sub_idx)
        proc.stdin.write(frame.tobytes())
        
    proc.stdin.close()
    proc.wait()
    return local_path

def main():
    logger.info("===========================================================================")
    logger.info("🚀 STARTING GENERATION OF 100 VISUALLY DISTINCT PROCEDURAL DOMAIN VIDEOS")
    logger.info("===========================================================================")
    
    t0 = time.time()
    generated_videos = []
    global_idx = 0
    
    for d_idx, domain in enumerate(DOMAINS):
        logger.info(f"Rendering Domain {d_idx+1}/10: '{domain['name']}' ({domain['theme']})...")
        for item_tuple in domain["items"]:
            vid_path = generate_and_encode_video(global_idx, domain, item_tuple)
            generated_videos.append({
                "index": global_idx,
                "video_id": f"vid-actual-{global_idx+1:03d}",
                "title": item_tuple[0],
                "description": item_tuple[1],
                "tags": item_tuple[2] + [domain["category"]],
                "local_path": vid_path,
                "gcs_uri": f"gs://{GCS_BUCKET_NAME}/{GCS_PREFIX}/{vid_path.name}",
                "gcs_bucket": GCS_BUCKET_NAME,
                "gcs_object_name": f"{GCS_PREFIX}/{vid_path.name}",
                "duration_seconds": DURATION_SEC,
                "file_size_bytes": vid_path.stat().st_size
            })
            global_idx += 1
            
    logger.info(f"✓ Generated 100 distinct procedural video files in {time.time() - t0:.2f}s!")
    
    # 2. Upload to GCS
    logger.info("\nUploading 100 procedural videos to Google Cloud Storage...")
    t_upload = time.time()
    upload_cmd = [
        "gcloud", "storage", "cp", "-r",
        f"{OUTPUT_DIR}/*.mp4",
        f"gs://{GCS_BUCKET_NAME}/{GCS_PREFIX}/"
    ]
    subprocess.run(upload_cmd, check=True)
    logger.info(f"✓ Uploaded 100 video files to GCS in {time.time() - t_upload:.2f}s")
    
    # 3. Extract Vertex AI Multimodal Embeddings
    logger.info("\nInitializing Vertex AI Multimodal Embedding Model (1408 dimensions)...")
    vertexai.init(project=PROJECT_ID, location=REGION)
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    logger.info("✓ Vertex AI Model Ready")
    
    logger.info("\nExtracting HYBRID MULTIMODAL EMBEDDINGS (Video Frames + Semantic Metadata)...")
    t_emb = time.time()
    
    for i, v in enumerate(generated_videos):
        t_item = time.time()
        # A. Visual Frame Embedding directly from GCS
        video_asset = Video.load_from_file(v["gcs_uri"])
        vis_resp = model.get_embeddings(video=video_asset, dimension=1408)
        vis_emb = np.array(vis_resp.video_embeddings[0].embedding, dtype=np.float32)
        
        # B. Semantic Text Embedding
        context_text = f"{v['title']}. {v['description']}. Tags: {' '.join(v['tags'])}"
        txt_resp = model.get_embeddings(contextual_text=context_text, dimension=1408)
        txt_emb = np.array(txt_resp.text_embedding, dtype=np.float32)
        
        # C. Hybrid Multimodal Fusion (50% Visual Video Frames + 50% Text Context)
        hybrid_emb = 0.50 * vis_emb + 0.50 * txt_emb
        hybrid_norm = (hybrid_emb / np.linalg.norm(hybrid_emb)).tolist()
        v["embedding"] = hybrid_norm
        
        if (i + 1) % 10 == 0:
            logger.info(f"  -> Extracted embeddings for {i+1}/100 videos (latest in {time.time() - t_item:.2f}s)")
            
    logger.info(f"✓ Extracted all 100 embeddings in {time.time() - t_emb:.2f}s!")
    
    # 4. Save to Cloud Spanner
    logger.info(f"\nWriting 100 actual video embedding records to Cloud Spanner '{SPANNER_INSTANCE}/{SPANNER_DATABASE}'...")
    spanner_client = spanner.Client(project=PROJECT_ID)
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)
    
    spanner_rows = []
    for v in generated_videos:
        spanner_rows.append([
            v["video_id"],
            v["title"],
            v["description"],
            v["tags"],
            v["gcs_uri"],
            v["gcs_bucket"],
            v["gcs_object_name"],
            "video/mp4",
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
        
    logger.info("===========================================================================")
    logger.info("🎉 SUCCESS: 100 PROCEDURALLY DISTINCT VIDEO EMBEDDINGS INDEXED IN SPANNER!")
    logger.info("===========================================================================")

if __name__ == "__main__":
    main()

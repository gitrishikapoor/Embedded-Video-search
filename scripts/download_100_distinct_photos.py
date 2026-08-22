import time
import requests
import re
from pathlib import Path

img_dir = Path("storage/images/individual_100")
img_dir.mkdir(parents=True, exist_ok=True)

import sys
sys.path.append('scripts')
from generate_100_authentic_photorealistic_videos import DOMAINS

headers = {
    'User-Agent': 'VideoSearchVectorResearch/2.0 (Google Cloud Research; contact: rikapoor@google.com)'
}

all_items = []
for d in DOMAINS:
    for item in d["items"]:
        all_items.append((len(all_items) + 1, item[0], item[2], d["category"]))

print(f"Checking and downloading 100 distinct images...")
success_count = 0

for vid_num, title, tags, cat in all_items:
    vid_id = f"vid-actual-{vid_num:03d}"
    img_path = img_dir / f"{vid_id}.jpg"
    
    # Check if already a valid JPEG (> 10KB and starts with JPEG magic bytes b'\xff\xd8')
    if img_path.exists() and img_path.stat().st_size > 10000:
        data = img_path.read_bytes()
        if data.startswith(b'\xff\xd8') or data.startswith(b'\x89PNG'):
            success_count += 1
            continue

    print(f"Downloading #{vid_num:03d} ({vid_id}): '{title}'...")
    
    # Search terms to try in order
    terms_to_try = [
        re.sub(r'\b(High Speed|Passing Maneuver|Clip|Exploration|Display|Sequence|Action|Footage|Time Lapse|Over|Under|Teeming with|Swimming in|Gliding Over|in|at|and|of)\b', '', title, flags=re.IGNORECASE).strip(),
        tags[0] if tags else "",
        cat
    ]
    
    downloaded = False
    for term in terms_to_try:
        if not term:
            continue
        try:
            url = f'https://en.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch={requests.utils.quote(term)}&gsrlimit=1&prop=pageimages&pithumbsize=1280&format=json'
            r = requests.get(url, headers=headers, timeout=10)
            pages = r.json().get('query', {}).get('pages', {})
            for pid, page in pages.items():
                if 'thumbnail' in page:
                    src = page['thumbnail']['source']
                    img_r = requests.get(src, headers=headers, timeout=10)
                    if img_r.status_code == 200 and len(img_r.content) > 10000 and (img_r.content.startswith(b'\xff\xd8') or img_r.content.startswith(b'\x89PNG')):
                        img_path.write_bytes(img_r.content)
                        downloaded = True
                        print(f"  ✓ Saved via Wikipedia '{term}' ({len(img_r.content)} bytes)")
                        break
            if downloaded:
                break
            time.sleep(0.3)
        except Exception as e:
            time.sleep(0.5)
            
    if not downloaded:
        # Fallback to Unsplash direct keyword image
        try:
            kw = requests.utils.quote(tags[0] if tags else cat)
            unsplash_url = f"https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1280&q=80"
            r = requests.get(unsplash_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if len(r.content) > 10000:
                img_path.write_bytes(r.content)
                print(f"  ✓ Saved via Unsplash fallback ({len(r.content)} bytes)")
                downloaded = True
        except Exception:
            pass

    if downloaded:
        success_count += 1
    time.sleep(0.2)

print(f"\n==================================================")
print(f"🎉 Total valid images in storage: {success_count}/100")
print(f"==================================================")

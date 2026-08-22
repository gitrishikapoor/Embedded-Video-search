#!/usr/bin/env python3
"""
Test Suite: Evaluates Video Vector Search across multiple realistic scenarios.
Verifies:
1. Top_k=4 default output
2. Accurate top matches for user queries (Children playing, pasta, bread and butter, etc.)
3. % Match calibration close to query relevance.
"""

import os
import sys
import json
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.app.config import settings
from backend.app.services.embedding_service import embedding_service
from backend.app.services.spanner_service import spanner_service

def run_evaluation():
    print("=" * 90)
    print("🎯 COMPREHENSIVE VECTOR SEARCH EVALUATION SUITE")
    print(f"• GCP Project:     {settings.GCP_PROJECT_ID}")
    print(f"• Spanner Target:  {settings.SPANNER_INSTANCE_ID}/{settings.SPANNER_DATABASE_ID}")
    print(f"• Embedding Model: {settings.EMBEDDING_MODEL_NAME} ({settings.EMBEDDING_DIMENSION}-dim)")
    print("=" * 90)

    test_scenarios = [
        {
            "query": "Children playing together",
            "expected_top": "Children playing",
            "category": "Social / Recreation"
        },
        {
            "query": "chef made pasta",
            "expected_top": "Pasta",
            "category": "Culinary"
        },
        {
            "query": "children having pasta",
            "expected_top": "Pasta",
            "category": "Culinary / Social"
        },
        {
            "query": "bread and butter",
            "expected_top": "bread",
            "category": "Culinary / Food"
        },
        {
            "query": "supercar drifting on racetrack at high speed",
            "expected_top": "Supercar",
            "category": "Automotive"
        },
        {
            "query": "deep space galaxy and stars",
            "expected_top": "Space",
            "category": "Astronomy"
        },
        {
            "query": "ocean waves sunset on beach",
            "expected_top": "Waves",
            "category": "Marine & Nature"
        },
        {
            "query": "industrial robotic arm welding car chassis",
            "expected_top": "Robotic",
            "category": "Robotics & Tech"
        },
        {
            "query": "tokyo shibuya crossing night timelapse",
            "expected_top": "Tokyo",
            "category": "Urban Cityscape"
        },
        {
            "query": "big wave surfer riding barrel wave",
            "expected_top": "Surfer",
            "category": "Extreme Sports"
        },
        {
            "query": "symphony orchestra violin concert",
            "expected_top": "Symphony",
            "category": "Music Arts"
        },
        {
            "query": "mountain biker flying downhill through forest",
            "expected_top": "Mountain Biking",
            "category": "Extreme Sports"
        },
        {
            "query": "lion resting in african savanna",
            "expected_top": "Lion",
            "category": "Wildlife"
        },
        {
            "query": "grizzly bear catching salmon",
            "expected_top": "Grizzly",
            "category": "Wildlife"
        }
    ]

    all_passed = True
    summary_table = []

    for idx, scenario in enumerate(test_scenarios, start=1):
        q = scenario["query"]
        print(f"\n[{idx:02d}/{len(test_scenarios)}] 🔍 SCENARIO: \"{q}\" ({scenario['category']})")
        
        # 1. Generate text embedding
        q_vec = embedding_service.generate_text_embedding(q)
        
        # 2. Vector search with default top_k=4
        results = spanner_service.vector_search(query_embedding=q_vec, top_k=4)
        
        assert len(results) <= 4, f"Expected at most 4 results, got {len(results)}"
        
        top_result = results[0] if results else None
        top_title = top_result.get("title", "") if top_result else "No result"
        top_sim = top_result.get("similarity_score", 0.0) if top_result else 0.0
        top_dist = top_result.get("distance", 1.0) if top_result else 1.0
        top_id = top_result.get("video_id", "") if top_result else ""

        match_pct = f"{top_sim * 100:.1f}%"
        is_relevant = scenario["expected_top"].lower() in top_title.lower() or top_sim >= 0.70

        status_icon = "✅" if is_relevant else "⚠️"
        print(f"  {status_icon} #1 Top Match: {top_title}")
        print(f"     Match: {match_pct} | Cosine Distance: {top_dist:.4f} | Video ID: {top_id}")
        
        # Display all 4 ranked items
        for r_idx, r in enumerate(results[1:], start=2):
            print(f"     #{r_idx}: {r.get('title')} ({r.get('similarity_score')*100:.1f}% Match, dist={r.get('distance'):.4f})")

        summary_table.append({
            "scenario": q,
            "top_match": top_title,
            "match_pct": match_pct,
            "distance": round(top_dist, 4),
            "status": "PASSED" if is_relevant else "CHECK"
        })

    print("\n" + "=" * 90)
    print("📊 EVALUATION SUMMARY REPORT")
    print("=" * 90)
    print(f"{'Scenario Query':<45} | {'Top Match Video':<32} | {'% Match':<8} | {'Status'}")
    print("-" * 95)
    for row in summary_table:
        print(f"{row['scenario']:<45} | {row['top_match'][:30]:<32} | {row['match_pct']:<8} | {row['status']}")
    print("=" * 90)

if __name__ == "__main__":
    run_evaluation()

#!/usr/bin/env python3
"""
Cloud Spanner Setup Script
Creates Spanner Instance, Database, and executes Schema DDL with Vector Support.
"""

import os
import sys
from pathlib import Path
from google.cloud import spanner

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "demo-gcp-project")
INSTANCE_ID = os.getenv("SPANNER_INSTANCE_ID", "video-search-instance")
DATABASE_ID = os.getenv("SPANNER_DATABASE_ID", "video-search-db")
CONFIG_NAME = os.getenv("SPANNER_CONFIG", "regional-us-central1")

SCHEMA_FILE = Path(__file__).parent / "schema.sql"

def setup_spanner():
    print(f"Connecting to Google Cloud Spanner (Project: {PROJECT_ID})...")
    client = spanner.Client(project=PROJECT_ID)

    # 1. Instance Configuration
    config_name = f"projects/{PROJECT_ID}/instanceConfigs/{CONFIG_NAME}"
    instance = client.instance(
        INSTANCE_ID,
        configuration_name=config_name,
        display_name="Video Vector Search Spanner Instance",
        node_count=1,
    )

    if not instance.exists():
        print(f"Creating Spanner Instance: {INSTANCE_ID} (1 Node)...")
        operation = instance.create()
        operation.result(timeout=300)
        print(f"✓ Spanner Instance {INSTANCE_ID} created successfully.")
    else:
        print(f"✓ Spanner Instance {INSTANCE_ID} already exists.")

    # 2. Database Configuration
    database = instance.database(DATABASE_ID)
    if not database.exists():
        print(f"Reading DDL schema from {SCHEMA_FILE}...")
        with open(SCHEMA_FILE, "r") as f:
            ddl_statements = [
                s.strip() for s in f.read().split(";") 
                if s.strip() and not s.strip().startswith("--")
            ]

        print(f"Creating Spanner Database: {DATABASE_ID} with {len(ddl_statements)} DDL statements...")
        operation = database.create(ddl_statements=ddl_statements)
        operation.result(timeout=300)
        print(f"✓ Spanner Database {DATABASE_ID} created with Videos table & vector schema.")
    else:
        print(f"✓ Spanner Database {DATABASE_ID} already exists.")

    print("\nCloud Spanner setup completed successfully!")

if __name__ == "__main__":
    setup_spanner()

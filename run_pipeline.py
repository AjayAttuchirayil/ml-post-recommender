# run_pipeline.py

import os
import subprocess

print("🚀 Running full ML pipeline...\n")

# Step 1: Simulate Data
print("📦 Step 1: Simulating user-post data...")
subprocess.run(["python", "utils/data_loader.py"])

# Step 2: Feature Engineering
print("\n🧠 Step 2: Creating features...")
subprocess.run(["python", "utils/feature_utils.py"])

# Step 3: Train Model
print("\n🤖 Step 3: Training model...")
subprocess.run(["python", "utils/train_model.py"])

print("\n✅ Pipeline complete.")

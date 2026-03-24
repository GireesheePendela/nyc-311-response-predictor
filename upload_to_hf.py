import os
from importlib import import_module

api = import_module("huggingface_hub").HfApi()
token = os.getenv("HF_TOKEN")
repo = os.getenv("HF_SPACE_REPO", "Gireeshee/nyc-311-response-predictor")

if not token:
    raise SystemExit("Missing HF_TOKEN environment variable")

files = [
    "best_model.pkl",
    "feature_cols.pkl",
    "app.py",
    "step7_dashboard.html",
    "requirements.txt",
    "Dockerfile",
    "README.md",
]

for file_name in files:
    print(f"Uploading {file_name}...")
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=repo,
        repo_type="space",
        token=token,
    )
    print(f"✓ {file_name}")

print("\nAll done!")

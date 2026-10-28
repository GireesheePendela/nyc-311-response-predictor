import os
from importlib import import_module

hf = import_module("huggingface_hub")
api = hf.HfApi()
# HF_TOKEN if set, otherwise the token saved by `hf auth login`
token = os.getenv("HF_TOKEN") or hf.get_token()
repo = os.getenv("HF_SPACE_REPO", "Gireeshee/nyc-311-response-predictor")

if not token:
    raise SystemExit(
        "No Hugging Face token found. Run this once, paste a Write token when asked:\n"
        "  .\\.venv\\Scripts\\hf.exe auth login"
    )

files = [
    "best_model.pkl",
    "feature_cols.pkl",
    "target_encodings.pkl",
    "model_metrics.json",
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

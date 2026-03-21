---
title: NYC 311 Response Predictor
emoji: 🗽
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🗽 NYC 311 Response Intelligence App

Predict NYC 311 complaint resolution time with a trained model, and explore historical response behavior through an interactive dashboard.

## 🚀 Live App

- 🌐 Hugging Face Space: [https://huggingface.co/spaces/GP05/nyc-311-response-predictor](https://huggingface.co/spaces/GP05/nyc-311-response-predictor)
- 🔗 Direct URL: [https://gp05-nyc-311-response-predictor.hf.space](https://gp05-nyc-311-response-predictor.hf.space/)

## 📦 What this repo includes

- `app.py` — Gradio app with two tabs:
	- **Dashboard Insights** (embedded Step 7 dashboard)
	- **Resolution Predictor** (interactive prediction form)
- `step7_dashboard.html` — interactive EDA dashboard generated from pipeline output
- `best_model.pkl` and `feature_cols.pkl` — model artifact + expected feature schema
- `Dockerfile` — deployment image for Hugging Face Docker Space
- `upload_to_hf.py` — helper script to upload deployment files to your Space

## 🤖 Model context

- Training window: Jan–Mar 2026
- Trained model: Random Forest Regressor
- Reported performance: R² ≈ 0.78–0.79
- Month selector supports 1–12; months outside Jan–Mar are extrapolations

## 💻 Run locally

From project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:7860`

## ☁️ Deploy/update Hugging Face Space

From project root:

```powershell
.\.venv\Scripts\Activate.ps1
$env:HF_TOKEN="your_new_token"
.\.venv\Scripts\python.exe upload_to_hf.py
Remove-Item Env:HF_TOKEN
```

## 🛠️ Troubleshooting

- **⚠️ Port in use (7860)**
	- Stop existing process on 7860, then rerun `python app.py`
- **🟡 Import warning in editor (yellow underline)**
	- Ensure VS Code interpreter is `.venv\Scripts\python.exe`
	- Reinstall deps: `python -m pip install -r requirements.txt`
- **🚧 Space build fails with missing file**
	- Re-run `upload_to_hf.py` and restart the Space

## 👤 Author

Built by **Gireeshee Pendela**.

- GitHub: https://github.com/GireesheePendela

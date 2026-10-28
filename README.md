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

**Data source:** [311 Service Requests from 2010 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9), published by [NYC Open Data](https://opendata.cityofnewyork.us/). This project uses complaints created January–March 2026.

## 🚀 Live App

- 🌐 Hugging Face Space: [https://huggingface.co/spaces/Gireeshee/nyc-311-response-predictor](https://huggingface.co/spaces/Gireeshee/nyc-311-response-predictor)
- 🔗 Direct URL: [https://gireeshee-nyc-311-response-predictor.hf.space](https://gireeshee-nyc-311-response-predictor.hf.space/)

## 📦 What this repo includes

- `app.py` — Gradio app with two tabs:
	- **Dashboard Insights** (embedded Step 7 dashboard)
	- **Resolution Predictor** (interactive prediction form)
- `step7_dashboard.html` — interactive EDA dashboard generated from pipeline output
- `best_model.pkl` and `feature_cols.pkl` — model artifact + expected feature schema
- `Dockerfile` — deployment image for Hugging Face Docker Space
- `upload_to_hf.py` — helper script to upload deployment files to your Space

## 🤖 Model context

- Data: 589,802 NYC 311 complaints ([NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9), Jan–Mar 2026), loaded into PostgreSQL and cleaned
- Target: `log1p(resolution_hours)`; metrics in hours are reported after converting back
- Month selector is limited to Jan–Mar, the months covered by the training data

## 🧪 Methodology

- **Split:** 60% train / 20% validation / 20% held-out test (`random_state=42`).
- **Model selection:** Linear Regression, Random Forest and XGBoost are compared on the validation set only. The winner is refit on train+validation and scored **once** on the test set.
- **Leakage control:** `problem_type` and `location_type` are target-encoded *after* the split, using training rows only. Training rows are encoded out-of-fold (5-fold) with smoothing toward the global mean, so no row's feature contains its own target. Imputation medians are also train-only. See `target_encoding.py`.
- **Single source of truth:** `step6_model.py` and `compress_model.py` write `model_metrics.json`; the dashboard reads its numbers from there.

| Model | Validation R² | Validation MAE (hrs) |
|---|---|---|
| Linear Regression | 0.703 | 40.0 |
| **Random Forest** | **0.787** | **34.8** |
| XGBoost | 0.785 | 35.4 |

| Final scores (held-out test) | R² | MAE (hrs) |
|---|---|---|
| Random Forest, 200 trees | 0.789 | 34.3 |
| Deployed Random Forest (25 trees, compressed to 0.7 MB for the Space) | 0.780 | 35.4 |

**Error analysis:** the dashboard breaks test-set error down by borough and agency. Absolute error is largest in the Bronx and Manhattan, where complaints take longest (median miss ≈ 9–10 h against a median resolution time of ≈ 22 h). Queens and Staten Island have the smallest misses in hours but the largest relative to their typical resolution time.

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

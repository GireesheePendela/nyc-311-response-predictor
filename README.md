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

**Data source:** [311 Service Requests from 2010 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9), published by [NYC Open Data](https://opendata.cityofnewyork.us/). This project uses complaints created January 1 – March 2, 2026.

## 🚀 Live App

- 🌐 Hugging Face Space: [https://huggingface.co/spaces/GP05/nyc-311-response-predictor](https://huggingface.co/spaces/GP05/nyc-311-response-predictor)
- 🔗 Direct URL: [https://gp05-nyc-311-response-predictor.hf.space](https://gp05-nyc-311-response-predictor.hf.space/)
- 📝 Write-up: [How long will your 311 complaint take?](https://gireesheependela.github.io/nyc-311-response-predictor/) (plain-language findings, built from `docs/`)

## 📦 What this repo includes

- `app.py` — Gradio app with two tabs:
	- **Dashboard Insights** (embedded Step 7 dashboard)
	- **Resolution Predictor** (interactive prediction form)
- `step7_dashboard.html` — interactive EDA dashboard generated from pipeline output
- `best_model.pkl` and `feature_cols.pkl` — model artifact + expected feature schema
- `Dockerfile` — deployment image for Hugging Face Docker Space
- `upload_to_hf.py` — helper script to upload deployment files to your Space

## 🤖 Model context

- Data: 589,802 NYC 311 complaints ([NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9), created Jan 1 – Mar 2, 2026), loaded into PostgreSQL and cleaned
- Target: `log1p(resolution_hours)`; metrics in hours are reported after converting back
- Month selector is limited to Jan–Feb; the data ends on March 2, so March is not covered

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

## 🔁 Reproduce the pipeline

The raw data is not committed (it is ~700k rows), so the steps below rebuild everything from scratch.

**1. Get the data**

- Open the [311 Service Requests dataset](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9) on NYC Open Data.
- Filter **Created Date** to 01/01/2026 – 03/02/2026 and export as CSV.
- Save it in a `data/` folder in the project root (any filename ending in `.csv`).

**2. Set up PostgreSQL**

- Install PostgreSQL and make sure the server is running.
- Copy `.env.example` to `.env` and fill in your credentials. `step2_load_postgres.py` creates the database named in `DB_NAME` if it does not exist.

**3. Install dependencies**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**4. Run the steps in order** (each has a matching `run_stepN.ps1` wrapper)

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_inspect.py` | Profiles the raw CSV: columns, missing values, date ranges |
| 2 | `step2_load_postgres.py` | Creates the database and bulk-loads the CSV into `nyc311_raw` |
| 3 | `step3_clean.py` | Cleans closed complaints into `nyc311_clean`; open ones go to `nyc311_open` |
| 4 | `step4_features.py` | Builds time, agency, borough and channel features into `nyc311_features` |
| 5 | `step5_eda.py` | Runs SQL analysis and saves `step5_eda_charts.png` |
| 6 | `step6_model.py` | Trains and compares models; saves `best_model.pkl`, `feature_cols.pkl`, `target_encodings.pkl`, `model_metrics.json` |
| — | `compress_model.py` | Refits a 25-tree Random Forest small enough for the Space and records its scores |
| 7 | `step7_visualization.py` | Builds the interactive `step7_dashboard.html` |
| 8 | `step8_api.py` | Optional FastAPI service exposing `POST /predict` |
| 9 | `step9_gallery_data.py` | Reads the CSV directly and writes `docs/findings.json` for the write-up page |

Then run `python app.py` to launch the app on the rebuilt model.

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
# once per computer: paste a Write token from huggingface.co/settings/tokens
.\.venv\Scripts\hf.exe auth login

.\.venv\Scripts\python.exe upload_to_hf.py
```

`upload_to_hf.py` also accepts a token in the `HF_TOKEN` environment variable.

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

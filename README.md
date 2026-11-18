# PRCL-0019 · FicZon Sales Effectiveness & Lead Quality Prediction

*DataMites™ data science capstone project PR-0019, based on a real-world business case.*

Predicts whether a new sales lead is **High potential** or **Low potential** from the information FicZon has when
the lead arrives (product, channel, location, sales agent, delivery mode, contact quality, time).
The project has the full analysis notebook, a **Flask** REST API, a **Streamlit** UI and Docker/Render deployment.

> **Data.** The raw CRM export (`Data/sales_data.csv`) and the project brief are provided by the institute and are
> not redistributed. Both are git-ignored, and the CSV is excluded from both Docker images. Only the trained model (`models/lead_quality_pipeline.joblib`, 54 KB) and aggregate metrics are
> shipped. Masked mobile numbers and e-mails are turned into two flags (number given? e-mail provider?) during
> cleaning and are never stored or sent anywhere.

**Live demo:** not deployed yet.

## Problem

**Project goals (from the brief):** (1) data exploration insights on sales effectiveness, and (2) an ML model to
predict the lead category (High potential / Low potential).

FicZon Inc. sells IT products and services. Leads arrive through calls, live chat, the website, referrals and
campaigns, and agents qualify them by hand. Sales effectiveness has dropped, so FicZon wants an ML model that scores
each new lead's quality, letting agents work the most promising leads first.

**Target.** The CRM `Status` (11 values) is grouped into two classes:

| Class | Statuses | Leads |
|---|---|---|
| **High potential (1)** | CONVERTED, converted, In Progress Positive, Potential, Long Term, Open | 2,931 (39.5%) |
| **Low potential (0)** | Junk Lead, Not Responding, Just Enquiry, In Progress Negative, LOST | 4,491 (60.5%) |

`Open` leads have not been worked yet. They count as live opportunities, so they go with High; there are only 82 of
them (1.1%).

## Results at a glance

All models share the same features and 5 stratified CV folds on the training set (80%). Class weights are on where
supported. The top three were tuned with `RandomizedSearchCV` and evaluated once on the held-out 20% test set.

| Model | CV accuracy | CV F1 | CV ROC-AUC | Tuned CV ROC-AUC | Test accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| Dummy (majority class) | 0.605 | 0.000 | 0.500 | – | – | – | – |
| KNN (k=25) | 0.700 | 0.587 | 0.761 | – | – | – | – |
| Decision Tree | 0.690 | 0.615 | 0.761 | – | – | – | – |
| Random Forest | 0.711 | 0.648 | 0.792 | – | – | – | – |
| Extra Trees | 0.706 | 0.657 | 0.793 | – | – | – | – |
| Logistic Regression | 0.708 | 0.667 | 0.797 | – | – | – | – |
| LightGBM | 0.711 | 0.662 | 0.800 | 0.801 | 0.750 | 0.673 | 0.821 |
| Gradient Boosting | 0.729 | 0.627 | 0.802 | 0.803 | 0.739 | 0.653 | 0.817 |
| **XGBoost (tuned), production** | 0.712 | 0.664 | 0.801 | **0.803** | **0.742** | **0.676** | **0.821** |

**Production model:** a tuned XGBoost pipeline with decision threshold **0.55**. The threshold maximises out-of-fold
macro-F1. Maximising F1 for High alone gave 0.40, which flagged most leads as High (recall 0.89, precision 0.55) and
would not help prioritisation.

| Production model on the test set (1,485 leads) | Value |
|---|---|
| Accuracy | **0.742** |
| Precision / Recall (High) | 0.670 / 0.683 |
| F1 (High) · macro-F1 | 0.676 · 0.731 |
| ROC-AUC | **0.821** |
| Confusion matrix [Low, High] | [[702, 197], [186, 400]] |
| Time hold-out (train Apr–Sep, test Oct–Nov 2018) | accuracy 0.727 · F1 0.723 · ROC-AUC 0.813 |
| Working the top 20% / 40% of scored leads finds | **41% / 68%** of all High leads |

**Class imbalance (60/40).** Class weights, SMOTE and random undersampling were compared inside CV. None of them
changes ROC-AUC (≈0.80), but all raise High-class recall from ~0.58 to 0.61–0.74. Class weights were kept because they
need no synthetic leads.

### Key findings

- **Product** is the strongest signal. Product 15 is the second-largest product (1,518 leads), yet only **6.5%** of
  its leads are High. Products 19, 10 and 20 run at 55–59%.
- **Location:** Bangalore **61%** and Chennai 51% High vs **18%** for the 2,500 "Other Locations" leads.
- **Channel:** referrals (87%) and existing clients (93%) are by far the best sources. The US website is almost all
  Low (2%).
- **Sales agent:** 26% (Agent-5) to 63% (Agent-9) High. This partly reflects lead allocation, and is useful for
  coaching.
- **Contact quality:** leads with no usable e-mail (`#VALUE!` in the export) are only 20% High vs ~47% for gmail/yahoo.
- **Time of day and weekday barely matter.** Month was left out on purpose: the High share rises in recent months
  because those leads are not closed out yet, and that pattern would not carry over to new leads.
- **Data quirk:** the 58 leads with no product and no location are almost all fresh `Open` leads. The API therefore
  **requires** product, source, location and delivery mode instead of scoring blank leads.

**Recommendations:** use the score to order the daily call queue; fix the web form that writes `#VALUE!` into e-mails
and make mobile number mandatory; review lead generation for product 15, "Other Locations" and the US website; grow the
referral and existing-client programmes; retrain monthly after closing out open leads.

Charts are in `reports/figures/`. Details, the full cleaning table and the challenges are in the notebook (§3, §13).

## Project layout

```
Data/sales_data.csv                  raw CRM export from the institute (NOT in git, NOT in images)
notebooks/FicZon_Lead_Quality.ipynb  load → target → cleaning → EDA → features → model comparison →
                                     imbalance → tuning + threshold → test eval → importance → time check → save
src/ficzon/features.py               cleaning + feature engineering shared by notebook, API and UI
models/                              lead_quality_pipeline.joblib, model_metadata.json, model_comparison.csv
reports/figures/                     charts saved by the notebook (aggregates only)
backend/                             Flask API (app.py, Dockerfile, requirements.txt)
frontend/                            Streamlit UI (app.py, Dockerfile, requirements.txt)
tests/test_api.py                    API tests (pytest)
docker-compose.yml, render.yaml      deployment
```

## 1. Re-run the notebook

Put the CSV from the institute at `Data/sales_data.csv`, then:

```bash
pip install scikit-learn==1.7.2 xgboost==3.3.0 lightgbm imbalanced-learn pandas seaborn scipy jupyter
```

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/FicZon_Lead_Quality.ipynb
```

This takes about 5 minutes on a laptop CPU. It writes the three files in `models/` and the charts in `reports/figures/`.

## 2. Run locally (no Docker)

Terminal 1, the API on http://localhost:5000:

```bash
pip install -r backend/requirements.txt
```

```bash
python backend/app.py
```

Terminal 2, the UI on http://localhost:8501:

```bash
pip install -r frontend/requirements.txt
```

```bash
streamlit run frontend/app.py
```

Tests:

```bash
python -m pytest tests -q
```

The UI has three tabs: **Score a lead** (form), **Score a CSV** (upload a CRM export or a tidy file, download the
scored and sorted leads; the file is processed in memory, and phone/e-mail are turned into flags before anything is
sent to the API) and **Model report**.

## 3. Run with Docker

```bash
docker compose up --build
```

UI → http://localhost:8501 · API → http://localhost:5000/health

## 4. Deploy to Render (free tier)

1. Push this folder to a GitHub repository. `.gitignore` keeps `Data/`, all CSVs (except
   `models/model_comparison.csv`) and the PDF brief out of git.
2. On Render: **New → Blueprint**, then pick the repo. `render.yaml` creates two Docker web services:
   `ficzon-lead-api` (Flask + gunicorn) and `ficzon-lead-ui` (Streamlit).
3. If Render gives the API a different URL, update `BACKEND_URL` on `ficzon-lead-ui` to match it.
4. Free services sleep after 15 minutes idle, so the first request takes about 30–60 seconds.

The API image uses `xgboost-cpu` (no CUDA libraries), so it stays small and well under the free tier's 512 MB.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | status, model name, threshold, test ROC-AUC |
| GET | `/api/v1/metadata` | metrics, confusion matrix, feature importance, allowed input values |
| POST | `/api/v1/predict` | JSON: one lead object, or `{"leads": [...]}` (up to 1,000) |

Lead fields: `created` ("YYYY-MM-DD HH:MM"), `product_id`, `source`, `sales_agent`, `location`, `delivery_mode`,
`has_mobile` (0/1), `email_type` (gmail / yahoo / microsoft / other / missing). `product_id`, `source`, `location` and
`delivery_mode` are required. Unseen values are handled safely.

```bash
curl -X POST http://localhost:5000/api/v1/predict -H "Content-Type: application/json" -d "{\"created\": \"2018-11-10 11:30\", \"product_id\": \"27\", \"source\": \"Referral\", \"sales_agent\": \"Sales-Agent-9\", \"location\": \"Bangalore\", \"delivery_mode\": \"Mode-4\", \"has_mobile\": 1, \"email_type\": \"gmail\"}"
```

Example response:

```json
{"prediction": "High potential", "probability_high": 0.97, "priority": "Hot", "threshold": 0.55, "latency_ms": 12.4}
```

`priority` is **Hot** for P(High) ≥ 0.65, **Warm** from the threshold up to 0.65, and **Cold** below the threshold.

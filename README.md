# Hospital Readmission Prediction

**Predicts which patients are at high risk of returning to the hospital within 30 days of discharge — before they leave.**

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)

---

## Overview

Hospitals face financial penalties from Medicare when too many of their patients return within 30 days of discharge. This project builds a machine learning classifier — trained on 101,766 real patient records from 130 U.S. hospitals — that identifies high-risk patients before they leave, giving care teams time to intervene. The best model (XGBoost) achieves an AUC of 95.5% and catches 87% of actual readmissions, enabling a modeled net annual benefit of ~$4.8M for a mid-size hospital.

---

## Business Problem

The Centers for Medicare & Medicaid Services (CMS) runs the Hospital Readmissions Reduction Program (HRRP), which penalizes hospitals whose patients are readmitted too frequently within 30 days of discharge. Penalties can reach **3% of total Medicare payments** — a multi-million-dollar hit for most hospitals. The problem is not that hospitals lack data; it is that no systematic process exists to identify *which specific patients* are at risk while they are still in the building. By scoring patients at discharge, care coordinators can act immediately: scheduling follow-up calls, arranging home health visits, or extending observation stays. Every readmission prevented reduces both direct treatment costs (~$14,400 per event) and the cumulative penalty exposure that CMS recalculates annually.

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| Name | Diabetes 130-US Hospitals for Years 1999–2008 |
| Source | UCI Machine Learning Repository / Kaggle |
| Records | 101,766 patient encounters |
| Features (raw) | 50 |
| Features (after engineering) | 91 |
| Time span | 10 years (1999–2008) |
| Target | `readmitted` — binary: readmitted within 30 days (1) vs. not (0) |
| Class balance | ~11% positive (readmitted within 30 days) |

**Download:** [Kaggle — Diabetes 130-US Hospitals](https://www.kaggle.com/datasets/brandao/diabetes)

Place the downloaded CSV at `data/raw/diabetic_data.csv`. Raw data files are excluded from this repository via `.gitignore` — only processed outputs are tracked.

---

## Project Structure

```
healthcare-readmission-prediction/
│
├── data/
│   ├── raw/                        # Source data — never modified (git-ignored)
│   └── processed/                  # Cleaned features and train/test splits
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis — understand the data
│   ├── 02_preprocessing.ipynb      # Feature engineering, encoding, SMOTE balancing
│   ├── 03_modeling.ipynb           # Model training, tuning, SHAP explainability
│   └── 04_business_impact.ipynb    # ROI analysis and executive-facing outputs
│
├── src/
│   ├── __init__.py                 # Makes src/ a Python package
│   ├── data_processing.py          # Reusable data cleaning functions
│   ├── features.py                 # Feature engineering functions
│   ├── model.py                    # Training, evaluation, and saving utilities
│   └── business_metrics.py         # Penalty savings and ROI calculations
│
├── outputs/
│   ├── figures/                    # All charts saved by notebooks (PNG)
│   ├── models/                     # Trained model files (XGBoost JSON, threshold)
│   └── model_comparison.csv        # Side-by-side metrics for all evaluated models
│
├── requirements.txt                # Python dependencies
├── CLAUDE.md                       # AI coding assistant instructions (git-ignored)
└── README.md                       # This file
```

---

## Methodology

- **01 — Exploratory Data Analysis:** Profiled all 50 features across 101,766 records. Identified severe class imbalance (11% positive), high missingness in `weight` (97%) and `payer_code` (40%), and strong signal in diagnosis categories and prior admission counts. Established that `readmitted` required binarization (collapse ">30 days" and "NO" into a single negative class).

- **02 — Preprocessing & Feature Engineering:** Mapped 17,000+ ICD-9 diagnosis codes into 9 clinical categories (Circulatory, Diabetes, Respiratory, etc.). Converted age brackets to numeric midpoints, encoded 40+ categorical columns, and dropped leakage-prone identifiers. Applied SMOTE on the training set only to address class imbalance, yielding a 50/50 balanced training corpus of 144,654 rows and 91 features.

- **03 — Modeling & Evaluation:** Trained a Logistic Regression baseline (AUC 95.1%) and an XGBoost classifier (AUC 95.5%). Tuned XGBoost via `RandomizedSearchCV` (20 iterations, 3-fold CV, optimizing AUC). Used SHAP `TreeExplainer` to identify the top predictors driving individual predictions. Optimized the classification threshold to 0.45 using the F1 score to balance recall and precision. Tracked all experiments with MLflow.

- **04 — Business Impact Analysis:** Translated model metrics into financial outcomes using CMS cost data and intervention cost estimates from healthcare literature. Modeled ROI across a range of classification thresholds to show how the business case changes with operational constraints. Produced an executive dashboard and a patient-outcome waffle chart suitable for non-technical stakeholders.

---

## Results

### Model Performance

| Model | AUC | Recall | Precision | F1 |
|-------|-----|--------|-----------|-----|
| Logistic Regression (baseline) | 0.951 | 86.9% | 99.5% | 92.8% |
| XGBoost (tuned, threshold = 0.45) | **0.955** | **86.9%** | **99.6%** | **92.8%** |

> **Recall** is the primary metric: it measures how many actual readmissions the model catches. A recall of 86.9% means 87 out of every 100 patients who *would* be readmitted are correctly flagged for intervention.

### Business Impact (500-bed hospital, ~15,000 annual discharges)

| Metric | Value |
|--------|-------|
| Patients flagged for intervention (annually) | ~1,434 |
| Readmissions prevented (est. 30% prevention rate) | ~430 |
| Total intervention program cost | ~$1.72M |
| Cost avoided (readmission treatment) | ~$6.19M |
| CMS HRRP penalty reduction | ~$110K |
| **Net annual benefit** | **~$4.8M** |
| **Return on investment** | **~180%** |

### Top Predictors (SHAP)

The features with the highest impact on individual predictions:

1. **Diagnosis category — Circulatory** (e.g., heart failure, coronary artery disease): strongest positive predictor of readmission
2. **Diagnosis category — Other**: catch-all for multi-morbidity patients; elevated risk
3. **Payer code — Unknown**: missing insurance information correlates with higher readmission rates, likely reflecting social determinants of health
4. **Race**: demographic signal; warrants careful fairness review before clinical deployment

---

## Key Visualizations

All figures are saved to `outputs/figures/` and generated by running the notebooks.

| Figure | Description |
|--------|-------------|
| `08_feature_correlations.png` | Correlation heatmap highlighting the features most associated with 30-day readmission |
| `11_shap_summary.png` | SHAP beeswarm plot showing which features drive the model's predictions and in which direction — the single most useful chart for clinical interpretation |
| `14_executive_dashboard.png` | Three-panel KPI summary (patients flagged, readmissions prevented, ROI) designed for a hospital board presentation |
| `15_threshold_business_tradeoff.png` | How net benefit, ROI, and intervention volume shift as the classification threshold changes — shows the business implications of a "dial" that care teams can adjust |

---

## Limitations & Future Work

- **Temporal split not enforced:** The train/test split is random rather than time-based. A production model should be trained on earlier years and evaluated on later ones to prevent data leakage across time and to reflect how the model would be deployed on future patients.

- **`weight` column dropped due to missingness:** Body weight is a clinically meaningful predictor of readmission risk, but 97% of records are missing this value. Future work should explore whether this field can be imputed from BMI or recovered from linked EHR data.

- **`payer_code` encodes socioeconomic status:** The model uses `payer_code` as a feature, but it is a proxy for insurance type and income level. Deployment in a clinical setting would require a fairness audit to ensure the model does not systematically disadvantage patients from lower-income groups.

- **Single-hospital business model assumption:** The ROI calculation assumes a single 500-bed hospital. Multi-site deployment would reduce per-patient intervention costs through economies of scale, and the model should be retrained on each hospital's own patient population before live use.

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/sarah-xyang/healthcare-readmission-prediction.git
cd healthcare-readmission-prediction
```

**2. Download the dataset from Kaggle**

Visit [Kaggle — Diabetes 130-US Hospitals](https://www.kaggle.com/datasets/brandao/diabetes), download `diabetic_data.csv`, and place it at:
```
data/raw/diabetic_data.csv
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the notebooks in order**
```
notebooks/01_eda.ipynb              # Explore the raw data
notebooks/02_preprocessing.ipynb   # Build the feature set
notebooks/03_modeling.ipynb        # Train and evaluate models
notebooks/04_business_impact.ipynb # Quantify the business case
```

Each notebook is self-contained and saves its outputs to `data/processed/`, `outputs/figures/`, and `outputs/models/` automatically.

---

## Author

Built as part of a data science portfolio project demonstrating end-to-end ML with business impact analysis.

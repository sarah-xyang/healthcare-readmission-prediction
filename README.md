# Healthcare Readmission Prediction

A machine learning model that predicts which hospital patients are at high risk of being readmitted within 30 days of discharge — helping clinical teams intervene early and reduce costly Medicare penalty fees.

## Overview

> TODO: Add a 2-3 sentence summary of the project, the dataset used, and the best model's performance.

## Business Problem

Hospitals that discharge Medicare patients who return within 30 days face financial penalties under the [Hospital Readmissions Reduction Program (HRRP)](https://www.cms.gov/Medicare/Medicare-Fee-for-Service-Payment/AcuteInpatientPPS/Readmissions-Reduction-Program). By identifying high-risk patients *before* they leave the hospital, care teams can schedule follow-up calls, arrange home health visits, or extend observation — reducing both readmissions and penalties.

> TODO: Add specific penalty figures and the estimated financial impact this model could address.

## Dataset

> TODO: Describe the dataset (source, number of records, key features, label definition).

## Project Structure

```
healthcare-readmission-prediction/
├── data/
│   ├── raw/          # Original downloaded data — never modified
│   └── processed/    # Cleaned data ready for modeling
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb   # Data cleaning and feature engineering
│   ├── 03_model.ipynb            # Model training and evaluation
│   └── 04_business_impact.ipynb # Translating results to business value
├── src/
│   ├── data_processing.py  # Reusable data cleaning functions
│   ├── features.py         # Feature engineering functions
│   ├── model.py            # Model training and evaluation functions
│   └── business_metrics.py # Business impact calculation functions
├── outputs/
│   ├── figures/    # Saved charts and visualizations
│   └── models/     # Saved trained model files
├── requirements.txt
└── CLAUDE.md
```

## Setup

> TODO: Add instructions for creating a virtual environment and installing dependencies.

```bash
pip install -r requirements.txt
```

## Usage

> TODO: Add step-by-step instructions for running the notebooks in order.

## Results

> TODO: Add key findings: model performance metrics, most important features, and estimated business impact.

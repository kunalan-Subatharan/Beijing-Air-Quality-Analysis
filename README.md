# 🌿 Beijing Air Quality Analysis
### CMP7005 — Programming for Data Analysis | Cardiff Metropolitan University

> **Student:** Kunalan Subatharan | **Student ID:** ST20274714
> **Module:** CMP7005 | **Institution:** Cardiff Metropolitan University

---

## 🌐 Live Web Application

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://beijing-air-quality-analysis-s4cfuwcd4qpfhwaovz3n6q.streamlit.app/)

---

## 📌 Project Overview

This project performs a comprehensive data-driven analysis of Beijing's air quality using hourly measurements recorded between **March 2013 and February 2017** across **4 monitoring stations** — 2 urban and 2 suburban.

The project covers the full data science pipeline:
- Raw data selection, merging and cleaning
- Exploratory Data Analysis (EDA)
- Machine learning model development (Random Forest & Linear Regression)
- An interactive Streamlit web application for real-time PM2.5 monitoring and forecasting

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Period** | March 2013 — February 2017 |
| **Stations** | Dongsi (Urban), Guanyuan (Urban), Changping (Suburban), Dingling (Suburban) |
| **Records** | ~140,000 hourly observations (after cleaning) |
| **Key Variables** | PM2.5, PM10, SO₂, NO₂, CO, O₃, Temperature, Pressure, Dew Point, Wind Speed, Rainfall |

---

## 🗂️ Repository Structure

```
Beijing-Air-Quality-Analysis/
│
├── data/                                         # Raw CSV files per station
│   ├── PRSA_Data_Changping_20130301-20170228.csv
│   ├── PRSA_Data_Dingling_20130301-20170228.csv
│   ├── PRSA_Data_Dongsi_20130301-20170228.csv
│   └── PRSA_Data_Guanyuan_20130301-20170228.csv
│
├── CMP7005_PRAC1_Beijing_Air_Quality.ipynb       # Main Jupyter Notebook (full analysis)
├── app.py                                        # Streamlit web application
├── beijing_air_quality_cleaned.csv               # Cleaned & merged dataset
├── requirements.txt                              # Python dependencies
├── scaler.joblib                                 # Saved StandardScaler for model
└── README.md                                     # This file
```

---

## 🔬 Jupyter Notebook — Analysis Pipeline

The notebook `CMP7005_PRAC1_Beijing_Air_Quality.ipynb` covers:

1. **Data Loading & Merging** — Reading 4 station CSVs and combining into one dataset
2. **Data Cleaning** — Handling missing values, outliers and type correction
3. **Feature Engineering** — Adding season, station type, AQI category columns
4. **Exploratory Data Analysis** — Distribution plots, temporal trends, correlation heatmaps, station comparisons
5. **Model Development** — Random Forest Regressor and Linear Regression for PM2.5 prediction
6. **Model Evaluation** — R², MAE, RMSE metrics, actual vs predicted plots, feature importance
7. **Key Findings** — WHO guideline exceedances, seasonal and hourly pollution patterns

---

## 🌐 Streamlit Web Application

The interactive dashboard (`app.py`) includes **6 pages**:

| Page | Description |
|---|---|
| 🏠 **Overview** | KPI cards, monthly PM2.5 trend, AQI distribution, station network summary |
| 📤 **Data Upload** | Upload your own CSV; all pages update automatically |
| 📋 **Dataset** | Schema, statistical summary, missing values audit, interactive data explorer |
| 📊 **Visualisation** | Distribution histograms, temporal patterns, station comparisons, bivariate explorer |
| 🤖 **Analytics** | Model performance, actual vs predicted, feature importance, live PM2.5 predictor |
| 📄 **Reports** | WHO AQI health guide, exceedance rates per station, top 10 worst pollution events |

---

## 🤖 Machine Learning Models

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| **Random Forest** | ~0.94 | ~10.5 µg/m³ | ~18.2 µg/m³ |
| Linear Regression | ~0.74 | ~22.1 µg/m³ | ~30.4 µg/m³ |

**Features used (15):** PM10, SO₂, NO₂, CO, O₃, Temperature, Pressure, Dew Point, Rainfall, Wind Speed, Hour, Month, Season, Station, Station Type

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/kunalan-Subatharan/Beijing-Air-Quality-Analysis.git
cd Beijing-Air-Quality-Analysis
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Dependencies

```
streamlit
pandas
numpy
plotly
scikit-learn
joblib
matplotlib
seaborn
```

Install all with: `pip install -r requirements.txt`

---

## 📈 Key Findings

- The annual mean PM2.5 across all stations is **~75 µg/m³ — 5× the WHO guideline** of 15 µg/m³
- Only **~39%** of hourly readings fall within the "Good" AQI category (≤35 µg/m³)
- **Winter** consistently records the highest pollution due to heating emissions and low wind
- **Urban stations** (Dongsi, Guanyuan) show significantly higher PM2.5 than suburban sites
- The Random Forest model achieves an **R² of ~0.94**, explaining 94% of PM2.5 variance

---

## 📋 WHO PM2.5 AQI Reference

| Category | PM2.5 Range | Health Impact |
|---|---|---|
| 🟢 Good | 0–35 µg/m³ | Little or no risk |
| 🟡 Moderate | 36–75 µg/m³ | Acceptable; sensitive groups at slight risk |
| 🟠 Unhealthy (SG) | 76–115 µg/m³ | Elderly and children may be affected |
| 🔴 Unhealthy | 116–150 µg/m³ | Everyone may experience effects |
| 🟣 Very Unhealthy | 151–250 µg/m³ | Health alert for all |
| ⚫ Hazardous | > 250 µg/m³ | Emergency conditions |

---

## 👤 Author

**Kunalan Subatharan**
Student ID: ST20274714
Cardiff Metropolitan University — MSc Data Science
Module: CMP7005 Programming for Data Analysis

---

## 📄 License

This project was developed for academic purposes as part of the CMP7005 module at Cardiff Metropolitan University.

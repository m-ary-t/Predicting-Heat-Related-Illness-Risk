# Predicting Heat-Related Illness Risk

![Heat-Related Illness Dashboard](images/dashboard_overview.png)

*Interactive public health analytics dashboard for forecasting heat-related illness risk across heat-prone U.S. regions.*

---

## Live Dashboard

https://predictinghri.streamlit.app/

---

## Project Overview

This project develops predictive machine learning models to estimate weekly heat-related illness (HRI) risk across heat-prone U.S. regions using weather, socioeconomic, and public health indicators.

The analysis integrates CDC, NOAA, ACS, and HUD datasets to identify relationships between environmental conditions, community vulnerability, and heat-related healthcare demand. Multiple statistical and machine learning models were evaluated to improve forecasting accuracy, interpretability, and public health applicability.

An interactive Streamlit dashboard was developed to visualize model outputs, regional trends, and healthcare risk indicators.

---

## Business Problem

Extreme heat events increasingly impact emergency healthcare systems and vulnerable populations across the United States.

Public health agencies and healthcare systems often face challenges in:
- anticipating surges in heat-related illness
- identifying high-risk communities
- allocating healthcare resources proactively
- monitoring environmental and socioeconomic vulnerability factors

This project addresses these challenges by building a data-driven forecasting framework capable of estimating heat-related illness risk using weather, socioeconomic, and public health indicators.

---

## Objectives

The goals of this project were to:

- Predict weekly heat-related illness risk across high-risk U.S. regions
- Identify key environmental and socioeconomic drivers of heat-related illness
- Compare statistical and machine learning forecasting approaches
- Improve predictive performance through feature engineering and model tuning
- Apply explainable AI techniques for healthcare interpretability
- Develop an interactive public health analytics dashboard

---

## Key Results

- Developed predictive models using CDC, NOAA, ACS, and HUD public health datasets
- Improved model RMSE through feature engineering, transformations, and model tuning
- Identified major environmental and socioeconomic drivers of heat-related illness risk using SHAP explainability analysis
- Built an interactive Streamlit dashboard for healthcare forecasting and public health decision support
- Demonstrated the value of explainable machine learning for healthcare risk forecasting

---

## Data Sources

This project integrates publicly available datasets from multiple healthcare, environmental, and socioeconomic sources.

### CDC Heat-Related Illness Data
Used for:
- weekly heat-related illness rates
- regional healthcare impact analysis
- target prediction variables

### NOAA Weather Data
Used for:
- maximum temperature
- minimum temperature
- environmental heat exposure indicators

### American Community Survey (ACS)
Used for:
- socioeconomic indicators
- poverty and income metrics
- demographic vulnerability measures

### HUD Homelessness Data
Used for:
- overall homelessness estimates
- unsheltered homelessness indicators
- population vulnerability analysis

---

## Repository Data Notice

## Repository Data

Processed analytical datasets used for modeling and dashboard development are included within the repository's `/data` directory.

The project integrates publicly available datasets from:
- CDC Heat & Health Tracker
- NOAA Climate Data
- American Community Survey (ACS)
- HUD Homelessness Data

The repository includes:
- cleaned analytical datasets
- modeling workflows
- dashboard application code
- visualizations
- explainability analyses
- project outputs

Some large raw source files used during development may be excluded due to repository size limitations.

---

## Methodology

### Data Pipeline

```text
CDC HRI Data
      ↓
NOAA Weather Integration
      ↓
ACS + HUD Socioeconomic Merge
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
SHAP Explainability Analysis
      ↓
Dashboard Visualization
```

---

## Feature Engineering

The project incorporated multiple feature engineering approaches to improve predictive performance and healthcare interpretability.

### Engineered Features Included
- temperature interaction terms
- nonlinear transformations
- socioeconomic vulnerability metrics
- homelessness indicators
- transformed environmental variables
- feature scaling and normalization

Feature selection and transformation techniques were evaluated to optimize predictive accuracy and reduce model complexity.

---

## Modeling Approaches

The following models were developed and evaluated:

### Statistical Models
- Linear Regression

### Machine Learning Models
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

### Explainability Methods
- SHAP (SHapley Additive Explanations)

---

## Model Performance

### Tuned Model Comparison

| Model | MAE | RMSE | R² |
|------|------|------|------|
| Linear Regression | 98.97 | 149.24 | 0.383 |
| XGBoost | 96.48 | 152.94 | 0.352 |
| Decision Tree | 91.79 | 152.94 | 0.352 |

### Key Findings
- Tuned Linear Regression achieved the strongest predictive performance
- Feature engineering improved forecasting accuracy
- Nonlinear transformations improved model behavior
- Socioeconomic vulnerability indicators contributed meaningfully to healthcare risk prediction

---

## Predictive Performance Visualization

### Actual vs Predicted Heat-Related Illness Values
![Predicted vs Actual](images/predicted_vs_actual.png)

The final tuned linear regression model demonstrated strong alignment between predicted and observed HRI values while capturing broader healthcare risk trends across regions.

---

## Explainability Analysis

SHAP explainability analysis was applied to improve transparency and interpretability of the predictive models.

The analysis identified:
- environmental exposure patterns
- temperature-related nonlinear effects
- socioeconomic vulnerability impacts
- homelessness-related healthcare risk contributions
- feature-level model behavior

### SHAP Summary Plot
![SHAP Summary](images/shap_summary.png)

### Major Predictive Drivers
- Temperature-income interaction effects
- Minimum temperature indicators
- Nonlinear temperature transformations
- Median household income
- Unsheltered homelessness estimates

---

## Dashboard Features

The interactive Streamlit dashboard allows users to:

- Explore regional heat-related illness trends
- Visualize model predictions
- Compare forecasting outputs
- Review healthcare vulnerability indicators
- Analyze temperature-risk relationships
- Examine explainability visualizations
- Generate manual and batch predictions

---

## Dashboard Preview

### Dashboard Overview
![Dashboard Overview](images/dashboard_overview.png)

---

## Public Health Impact

This project demonstrates how predictive analytics and public health data can support healthcare preparedness and environmental health planning.

Potential applications include:
- emergency department preparedness
- heat-risk forecasting
- vulnerable population identification
- healthcare resource allocation
- environmental health monitoring
- public health planning

---

## Technologies Used

### Programming & Analytics
- Python
- Jupyter Notebook

### Data Analysis & Machine Learning
- pandas
- numpy
- scikit-learn
- XGBoost

### Visualization & Dashboarding
- Streamlit
- Plotly
- Matplotlib
- Seaborn

### Explainable AI
- SHAP

---

## Repository Structure

```text
Predicting-Heat-Related-Illness-Risk/

│── README.md
│── requirements.txt

├── notebooks/
│     ├── heat_related_illness_modeling.ipynb
│     ├── shap_analysis.ipynb

├── streamlit_app/
│     ├── app.py

├── images/
│     ├── dashboard_overview.png
│     ├── tuned_model_performance.png
│     ├── predicted_vs_actual.png
│     ├── shap_summary.png
```

---

## How to Run the Project

### Clone Repository

```bash
git clone https://github.com/m-ary-t/Predicting-Heat-Related-Illness-Risk.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Streamlit Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

## Future Improvements

Potential future enhancements include:

- Incorporating additional climate and environmental indicators
- Adding real-time weather integrations
- Expanding forecasting granularity to state or county levels
- Integrating emergency department utilization datasets
- Applying deep learning forecasting approaches
- Developing real-time heat-health alert systems

---

## Contributors

- Mary Tekele
- Shahzeb Ather
- Jasmine Cheng

---

## Acknowledgements

Data sources and public health resources:
- CDC Heat & Health Tracker
- NOAA Climate Data
- American Community Survey (ACS)
- HUD Homelessness Data

---

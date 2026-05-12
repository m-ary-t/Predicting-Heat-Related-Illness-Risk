# Predicting Heat-Related Illness Risk Using Weather and Socioeconomic Indicators

## Project Overview

This project investigates whether weekly heat-related illness (HRI) rates in heat-prone U.S. regions can be predicted using weather and socioeconomic indicators. The analysis integrates public health, climate, and demographic data to build predictive machine learning models capable of forecasting HRI trends and identifying the most influential drivers of illness risk.

The project combines:
- Public health analytics
- Machine learning modeling
- Feature engineering
- Explainable AI (SHAP)
- Real-world policy and healthcare implications

## Research Question

Can we predict weekly heat-related illness (HRI) rates in heat-prone U.S. regions using weather and socioeconomic indicators?

## Data Sources

The project integrates multiple real-world datasets:

### CDC Heat & Health Tracker
- Weekly HRI emergency department visit counts
- Regional-level data
- Target variable for prediction

### NOAA Weather Data
- Daily weather observations
- Aggregated to weekly regional level
- Includes:
  - Maximum temperature
  - Minimum temperature
### American Community Survey (ACS)

Socioeconomic indicators including:
  - Median household income
  - Poverty rate
  - Unemployment rate
  - Population
### HUD Homelessness Data
  - Overall homelessness counts
  - Unsheltered homelessness counts

## Data Pipeline
1. Collect and clean data from all sources
2. Aggregate NOAA daily weather data to weekly regional level
3. Align all datasets by region and week/year
4. Merge weather, HRI, and socioeconomic data
5. Generate synthetic training data while preserving distributions and correlations
6. Train and evaluate predictive models

### Final dataset:
  - 1460 rows
  - 8–9 predictive features

## Models Evaluated
The following machine learning models were tested:
### Linear Regression
Used as an interpretable baseline model.
### Decision Tree Regressor
Used to capture nonlinear relationships without requiring feature transformations.
### XGBoost Regressor
Used as an advanced ensemble learning model to maximize predictive performance.

## Feature Engineering
To improve predictive performance and capture nonlinear behavior, several engineered features were created:
  - Polynomial temperature term (temp²)
  - Temperature-income interaction term
  - Log-transformed homelessness variable

Feature selection was also performed to reduce redundancy and multicollinearity.

## Model Evaluation Metrics
The models were evaluated using:
  - RMSE (primary metric)
  - MAE
  - R²
### Why RMSE?
RMSE was prioritized because it penalizes large prediction errors more heavily, which is important when forecasting extreme heat-related illness spikes that may impact hospital preparedness and public health response.

## Final Model
### Best Performing Model
Engineered Linear Regression
### Final Performance
  - RMSE ~> 141
  - MAE	~> 86
  - R²	~> 0.45

The final engineered linear model outperformed Decision Tree and XGBoost models on the real holdout dataset.

## SHAP Analysis
SHAP (SHapley Additive exPlanations) was used to interpret the final model.
### Key Findings
  - Temperature-related variables were the strongest predictors
  - HRI risk increases nonlinearly at extreme temperatures
  - Socioeconomic variables contributed additional contextual information

## Key Insights
  - Temperature is the dominant driver of heat-related illness
  - Nonlinear temperature effects significantly improve prediction performance
  - Socioeconomic variables add important contextual vulnerability information
  - Real-world HRI prediction is inherently noisy and variable, making perfect prediction difficult

## Technologies Used
  - Python
  - Pandas
  - NumPy
  - Scikit-learn
  - XGBoost
  - SHAP
  - Statsmodels
  - Matplotlib
  - Seaborn

## Future Improvements
Potential future work includes:
  - Incorporating humidity and heat index variables
  - Adding geographic/spatial modeling
  - Incorporating CDC's PLACES small-area health data
  - Testing time-series forecasting approaches
  - Expanding to additional CDC regions

## Authors
Mary Tekele | Shahzeb Ather | Jasmine Cheng

Graduate Data Analytics Project – Predictive Modeling & Public Health Analytics

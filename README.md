# Machine Learning Term Project

## Table of Contents
- [Project Description](#project-description)
- [Target Variable](#target-variable)
- [Models](#models)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
  - [Setup & Installation](#setup--installation)
  - [Running the Notebook](#running-the-notebook)
  - [Running the CLI Demo App](#running-the-cli-demo-app)

---

## Project Description

This project predicts hourly electricity consumption using real-time electricity consumption and production data from 1 June 2025 to 1 September 2025.

The task is formulated as a supervised regression problem.

## Target Variable

The target variable is `Tuketim`, the hourly electricity consumption value.

To avoid feature availability leakage, same-time consumption and same-time production values are not used as input features. Instead, the model uses calendar features, lagged consumption values, lagged production values, and rolling historical statistics.

## Models

The following models are implemented and compared:

- Decision Tree Regressor
- XGBoost Regressor

## Methodology

The project uses:

- Data loading and preprocessing
- Exploratory data analysis
- Calendar, lag, and rolling feature engineering
- Chronological train/test split
- TimeSeriesSplit validation
- Hyperparameter tuning
- RMSE, MAE, and R² evaluation
- Feature importance analysis

## Repository Structure

```text
data/raw/          Raw production and consumption datasets
notebooks/         Main analysis notebook
figures/results/   Generated figures and result tables
presentation/      Presentation PDF
```

## How to Run

### Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YusufAliDemirkiran/machine-learning-term-project.git
cd machine-learning-term-project

# 2. Install the required dependencies
pip install -r requirements.txt
```
### Running the Notebook
```
jupyter notebook notebooks/main_analysis.ipynb
```
### Running the CLI Demo App
```
python src/app.py -h
```
Run model comparison:

```
python src/app.py compare
```
Predict consumption for a timestamp inside the available dataset:
```
python src/app.py predict "2025-08-20 14:00"
```
Use the tuned Decision Tree instead of tuned XGBoost:
```
python src/app.py predict "2025-08-20 14:00" --model decision_tree
```
List saved result files:
```
python src/app.py figures
```
Open a saved figure:
```
python src/app.py open-figure final_model_rmse_comparison
```

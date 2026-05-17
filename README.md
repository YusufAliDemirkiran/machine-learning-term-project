# Machine Learning Term Project

## Project Description

This project predicts hourly total electricity generation using real-time production data from 1 June 2025 to 1 September 2025.

## Models

- Decision Tree Regressor
- XGBoost Regressor

## Target Variable

The target variable is `Toplam`, total electricity generation.

To avoid target leakage, same-time source production columns are not used directly to predict same-time total generation. Instead, calendar features, lag features, and rolling historical features are used.

## Repository Structure

```text
data/raw/       Raw dataset
notebooks/      Main analysis notebook
figures/        Generated figures
slides/         Presentation slides

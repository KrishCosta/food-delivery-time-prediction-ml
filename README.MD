# Food Delivery Time Prediction

A machine-learning project that predicts food delivery time using distance, traffic, weather, preparation time, courier experience, time of day and vehicle type.

## Project Contents

- `notebooks/food_delivery_model_development.ipynb` — analysis, training, evaluation and feature importance
- `models/food_delivery_model.pkl` — trained prediction pipeline
- `requirements.txt` — project dependencies

## Models Evaluated

Baseline Mean, Linear Regression, Ridge Regression, Decision Tree, Random Forest and Gradient Boosting.

The final model was selected using the lowest five-fold cross-validation MAE and evaluated on a separate test set.

## Run Locally

```bash
pip install -r requirements.txt
# src/spread_predictor/model.py

from typing import Any, Dict, List, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from spread_predictor.constants import LIST_MODEL_TYPES, TEST_SIZE


# train/test split
def train_test_split_ts(X, y, test_size=TEST_SIZE):
    # split indices
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # NOTE - look into indices misalignment - due to lagging?
    # align indices - will be some leading or trailing values misalignment
    # train
    if not X_train.index.equals(y_train.index):
        print("X_train, y_train indices are not aligned! Correcting...")
        X_train, y_train = X_train.align(y_train, join="inner", axis=0)
        print("Indices are now aligned!")
    else:
        print("X_train, y_train indices are aligned!")
    # test
    if not X_test.index.equals(y_test.index):
        print("X_test, y_test indices are not aligned! Correcting...")
        X_test, y_test = X_test.align(y_test, join="inner", axis=0)
        print("Indices are now aligned!")
    else:
        print("X_test, y_test indices are aligned!")

    return X_train, X_test, y_train, y_test


def standardize_features(X_train, y_train, X_test, y_test):
    """
    Computed after train_test_split_ts to avoid data leakage - not necessary for non-time-series data
    """
    # standardize X and y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = pd.DataFrame(
        scaler_X.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler_X.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    y_train = pd.Series(
        scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten(),
        index=y_train.index,
    )
    y_test = pd.Series(
        scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index
    )

    return X_train, X_test, y_train, y_test


# OLS regression
def train_ols(X_train, y_train):
    X_const = sm.add_constant(X_train)
    fit_model = sm.OLS(y_train, X_const).fit()

    return fit_model


# mixed effects model (random intercept by group)
def train_mixed_effects(X_train, y_train, group):
    data = X_train.copy()
    data["y"] = y_train
    model = MixedLM.from_formula(
        "y ~ " + " + ".join(X_train.columns), groups=group, data=data
    )
    fit_model = model.fit()
    return fit_model


# xgboost regressor
def train_xgboost(X_train, y_train):
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)  # returns results inplace
    return model


# SARIMAX model (auto-selected order)
def train_sarimax(y_train, exog_train):
    model_order = auto_arima(y_train, exogenous=exog_train, seasonal=False, trace=True)
    sarimax = SARIMAX(
        y_train,
        exog=exog_train,
        order=model_order.order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit_model = sarimax.fit(disp=False)
    return fit_model


# bayesian linear regression with PyMC
def train_bayesian_regression(X_train, y_train, draws=2000, tune=1000):
    X_shared = X_train.values
    y_shared = y_train.values

    with pm.Model() as model:
        # Priors for intercept and slopes
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        betas = pm.Normal("betas", mu=0, sigma=1, shape=X_shared.shape[1])

        # Expected value of outcome
        mu = intercept + pm.math.dot(X_shared, betas)

        # Likelihood
        sigma = pm.HalfNormal("sigma", sigma=1)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_shared)

        # Sample from posterior
        trace = pm.sample(
            draws=draws, tune=tune, target_accept=0.9, return_inferencedata=True
        )

    return model, trace


# predict with bayesian model
def predict_bayesian_regression(model, trace, X_test):
    X_test_shared = X_test.values
    with model:
        pm.set_data(
            {"betas": trace.posterior["betas"].mean(dim=("chain", "draw")).values}
        )
        pred_mu = trace.posterior["intercept"].mean().values + np.dot(
            X_test_shared, trace.posterior["betas"].mean(dim=("chain", "draw")).values
        )
    return pred_mu


# model evaluation metrics
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return (
        np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    )


def evaluate_model(
    model, X_test, y_test, model_type: Literal[LIST_MODEL_TYPES] = "ols", trace=None
):
    """
    Evaluate a model and compute metrics like RMSE, MAE, and R2.

    Parameters:
        model: The trained model to evaluate.
        X_test: Test features.
        y_test: Test target values.
        model_type: Type of model ('ols', 'mixed', 'xgboost', 'bayesian').
        trace: PyMC3 trace object (required for Bayesian models).

    Returns:
        y_pred: Predicted values.
        metrics: Dictionary containing RMSE, MAE, and R2.
    """
    if model_type == "ols" or model_type == "mixed":
        X_test_const = sm.add_constant(X_test)  # Add constant for intercept
        y_pred = model.predict(X_test_const)
    elif model_type == "xgboost":
        y_pred = model.predict(X_test)
    elif model_type == "bayesian":
        if trace is None:
            raise ValueError(
                "For Bayesian models, the 'trace' argument must be provided."
            )
        y_pred = predict_bayesian_regression(model, trace, X_test)
    else:
        raise ValueError("Unsupported model type for evaluation.")

    # compute evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"SMAPE: {smape:.4f}")

    return y_pred, {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "smape": smape}


# SARIMAX evaluation (slightly different)
def evaluate_sarimax(model, y_test, exog_test):
    forecast = model.get_prediction(start=0, end=len(y_test) - 1, exog=exog_test)
    y_pred = forecast.predicted_mean

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)

    print(f"SARIMAX RMSE: {rmse:.4f}")
    print(f"SARIMAX MAE: {mae:.4f}")
    print(f"SARIMAX MAPE: {mape:.4f}")
    print(f"SARIMAX SMAPE: {smape:.4f}")

    return y_pred, {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape}


# diagnostic plot
def plot_bayesian_trace(trace, var_names=["intercept", "betas", "sigma"]):
    az.plot_trace(trace, var_names=var_names)
    plt.tight_layout()
    plt.show()


# plot predictions vs actuals
def plot_predictions(y_true, y_pred, title="Predictions vs Actuals"):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label="Actual", alpha=0.7)
    plt.plot(y_true.index, y_pred, label="Predicted", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# plot residuals
def plot_residuals(y_true, y_pred, title="Residuals"):
    residuals = y_true - y_pred
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(residuals)
    axs[0].set_title(f"{title} Over Time")
    sns.histplot(residuals, kde=True, ax=axs[1])
    axs[1].set_title(f"{title} Distribution")
    plt.tight_layout()
    plt.show()

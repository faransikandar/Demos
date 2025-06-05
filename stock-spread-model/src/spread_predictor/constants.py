# src/spread_predictor/constants.py

import os

# PATH DEFINITIONS
# define the directory path - matters if using shell console or IDE
DIRECTORY = os.path.dirname(os.getcwd())
if DIRECTORY.split("/")[:-1] != "spread-model":
    DIRECTORY = os.path.abspath("../")

# define the data and artifacts paths
DATA_PATH = os.path.join(DIRECTORY, "data")
ARTIFACTS_PATH = os.path.join(DIRECTORY, "artifacts")
DATA_FILENAME_SPACEX_ORDERS = "spacex_order_data.csv"
DATA_FILENAME_YFINANCE = "yfinance_data.csv"
DATA_FILENAME_FRED = "fred_data.csv"

# VARIABLE DEFINITIONS
# define vars and types - for original datasets (more arise from feature engineering later)
VARS_SPACEX_ORDERS = [
    "direction",
    "date",
    "price",
    "size",
    "structure",
    "carry",
    "management_fee",
]
# VARS_BOOK_WEEKLY = ['size_total', 'book_imbalance', 'depth_midprice', 'price_volume_slope', 'order_size_distortion']

VARS_YF = {
    "vix": "^VIX",
    "spy": "SPY",
    "arkx": "ARKX",
    "xli": "XLI",
    "treasury_10y": "^TNX",
}
VARS_FRED = {
    "fed_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "unemp_u3": "UNRATE",
    "unemp_u6": "U6RATE",
    "m2": "M2SL",
}
VARS_GOOGLE = ["SpaceX"]

COL_DATE_KEY = "date"

VARS_DATES = ["date"]
VARS_PERCENT = ["carry", "management_fee"]
VARS_NUMERIC = (
    ["spread", "price", "size", "carry", "management_fee"]
    + list(VARS_YF.keys())
    + list(VARS_FRED.keys())
)
# VARS_NUMERIC_AGG = VARS_NUMERIC + VARS_BOOK_WEEKLY
VARS_CATEGORICAL = [
    x for x in VARS_SPACEX_ORDERS if x not in VARS_NUMERIC and x not in VARS_DATES
]
# VARS_CATEGORICAL_TS = VARS_CATEGORICAL + ['no_orders']
VARS_DUMMIES = ["direction", "structure"]

VARS_RENAME = {"managementFee": "management_fee"}

# # NOTE - capture the aggregation and imputation vars
# OTHER_VARS = [1,2,3]

# FEATURE ENGINEERING DEFINITIONS
LIST_WINDOWS = [7, 14, 28, 56]
WINDOW_DAYS = 7
MODEL_ORDER_LEVEL = False

# NOTE - if DROP_DAYS_WITH_INVALID_SPREAD is False -> need to define SPREAD_IMPUTATION_METHOD
DROP_DAYS_WITH_INVALID_SPREAD = True
LIST_SPREAD_IMPUTATION_METHOD = ["median", "zero", "ffill_bfill"]
SPREAD_IMPUTATION_METHOD = None
Y_TARGET = "spread_7d_future"
Y_TARGET_TS = "spread_7d"

# MODEL TRAINING DEFINITIONS
TEST_SIZE = 0.2
LIST_MODEL_TYPES = ["ols", "mixed", "xgboost", "bayesian", "sarimax"]

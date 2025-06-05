# src/spread_predictor/features.py

from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from spread_predictor.constants import *
from spread_predictor.data_loader import (fetch_fred_data, fetch_google_trends,
                                          fetch_yahoo_data)


def build_df_daily_calendar(
    df_orders: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """
    This builds the calendar index used to capture all days in range
    """
    date_index = pd.date_range(
        df_orders["date"].min(), df_orders["date"].max(), freq="D"
    )
    df_calendar = pd.DataFrame(index=date_index)
    return date_index, df_calendar


def build_df_exog(
    date_index: pd.DatetimeIndex,
    df_daily: pd.DataFrame,
    dict_yf_tickers: Dict = VARS_YF,
    dict_fred_series: Dict = VARS_FRED,
    list_gt_keywords: List = VARS_GOOGLE,
) -> pd.DataFrame:
    """
    This builds the df_exog with external variables for every day from start to end
    Inherits date_index, df_daily from build_df_daily_calendar
    """
    # fetch external macro/market data into df_daily
    start_date = date_index.min()  # df_orders[COL_DATE_KEY].min()
    end_date = date_index.max()  # df_orders[COL_DATE_KEY].max()

    # create copy of df_daily for clarity of return statement
    df_exog = df_daily.copy()

    # fetch yahoo! data + add to df_exog
    df_yf = fetch_yahoo_data(dict_yf_tickers=dict_yf_tickers, date_index=date_index)
    for col in df_yf.columns:
        df_exog[col] = df_yf[col]

    # preview YF data
    print(f"\nVARS_YF: {VARS_YF}")
    print(
        f"\n**Preview the Yahoo! Finance Data:**\n\nRecords: {df_yf.shape[0]}\nVariables: {df_yf.shape[1]}"
    )
    print(f"\n**Data Types:**\n\n{df_yf.dtypes}")
    print(f"\n**Null Data:**\n\n{df_yf.isnull().sum()}")
    print(f"\ndf_yf_head: {df_yf.head()}")

    # fetch FRED data + add to df_exog
    df_fred = fetch_fred_data(dict_fred_series=dict_fred_series, date_index=date_index)
    for col in df_fred.columns:
        df_exog[col] = df_fred[col]

    # preview FRED data
    print(f"VARS_FRED: {VARS_FRED}")
    print(
        f"\n**Preview the FRED Data:**\n\nRecords: {df_fred.shape[0]}\nVariables: {df_fred.shape[1]}"
    )
    print(f"\n**Data Types:**\n\n{df_fred.dtypes}")
    print(f"\n**Null Data:**\n\n{df_fred.isnull().sum()}")
    print(f"\ndf_fred_head: {df_fred.head()}")

    # # NOTE - TooManyRequestsError: The request failed: Google returned a response with code 429
    # # NOTE - look into rate limiting or other errors
    # # fetch google trends data
    # df_gt = fetch_google_trends(list_gt_keywords=list_gt_keywords, date_index=date_index)

    # # merge into daily DataFrame (if trends found)
    # if not df_gt.empty:
    #     for keyword in VARS_GOOGLE:
    #         df_exog[keyword.lower() + "_trend"] = df_gt[keyword]
    # else:
    #     print(f"No Google Trends data available for {VARS_GOOGLE}")

    # # preview google trends data
    # print(f'VARS_GOOGLE: {VARS_GOOGLE}')
    # print(f'\n**Preview the Google Trends Data:**\n\nRecords: {df_gt.shape[0]}\nVariables: {df_gt.shape[1]}')
    # print(f'\n**Data Types:**\n\n{df_gt.dtypes}')
    # print(f'\n**Null Data:**\n\n{df_gt.isnull().sum()}')
    # print(f"\ndf_gt_head: {df_gt.head()}")

    return df_exog


def compute_df_book_static(df_orders: pd.DataFrame) -> pd.DataFrame:
    """
    This computes book-level features for static features - i.e. last seen bid / ask - used for X modeling
    """
    df_orders = df_orders.sort_values("date").copy()
    df_orders["date"] = pd.to_datetime(df_orders["date"])

    date_index = pd.date_range(
        df_orders["date"].min(), df_orders["date"].max(), freq="D"
    )

    def safe_idxmax(group):
        if group["price"].notna().any():  # Check if there are any non-NaN values
            return group.loc[group["price"].idxmax()]
        else:
            return None  # Return None for empty or all-NaN groups

    # get the maximum price and corresponding size for bids
    # NOTE - don't want to drop in order to preserve index for df_last joins - ffill should work on None - otherwise ffill doesn't fulfill it's role
    bids_max = (
        df_orders[df_orders["direction"] == "buy"]
        .groupby("date")
        .apply(safe_idxmax)
        # .dropna()  # drop None results
    )
    bids_max = bids_max[["price", "size"]].rename(
        columns={"price": "bid_last_price_max", "size": "bid_last_size_max"}
    )

    # get the minimum price and corresponding size for asks
    # NOTE - don't want to drop in order to preserve index for df_last joins - ffill should work on None - otherwise ffill doesn't fulfill it's role
    asks_min = (
        df_orders[df_orders["direction"] == "sell"]
        .groupby("date")
        .apply(safe_idxmax)
        # .dropna()  # drop None results
    )
    asks_min = asks_min[["price", "size"]].rename(
        columns={"price": "ask_last_price_min", "size": "ask_last_size_min"}
    )

    # create a DataFrame for all dates and forward-fill missing values
    df_last = pd.DataFrame(index=date_index)
    df_last = df_last.join(bids_max, how="left").join(asks_min, how="left")
    # NOTE - even with ffill, will have some empty rows at beginning of df_last, unless you use .dropna() in assignment above - or bfill here
    df_last["bid_last_price_max"] = df_last["bid_last_price_max"].ffill()
    df_last["bid_last_size_max"] = df_last["bid_last_size_max"].ffill()
    df_last["ask_last_price_min"] = df_last["ask_last_price_min"].ffill()
    df_last["ask_last_size_min"] = df_last["ask_last_size_min"].ffill()

    # add the last seen date for bids and asks
    df_last["bid_last_date"] = bids_max.index.to_series().reindex(date_index).ffill()
    df_last["ask_last_date"] = asks_min.index.to_series().reindex(date_index).ffill()

    # calculate days since ask / bid
    df_last["days_since_bid"] = df_last.index - df_last["bid_last_date"]
    df_last["days_since_ask"] = df_last.index - df_last["ask_last_date"]
    df_last["days_ask_minus_bid"] = df_last["ask_last_date"] - df_last["bid_last_date"]

    # change days type datetime to float
    df_last["days_since_bid"] = df_last["days_since_bid"].dt.days.astype(
        "float", errors="ignore"
    )
    df_last["days_since_ask"] = df_last["days_since_ask"].dt.days.astype(
        "float", errors="ignore"
    )
    df_last["days_ask_minus_bid"] = df_last["days_ask_minus_bid"].dt.days.astype(
        "float", errors="ignore"
    )

    print(f"df_last shape: {df_last.shape}")
    print(f"\ndf_last dtypes: \n{df_last.dtypes}")
    print(f"\ndf_last isna sum: \n{df_last.isna().sum()}")
    print(f"\ndf_last head: \n{df_last.head()}")

    return df_last


def compute_df_book_rolling(df_orders: pd.DataFrame, window_days: int = WINDOW_DAYS):
    """
    This computes book-level features - used for X modeling
    """
    df_orders = df_orders.sort_values("date").copy()
    df_orders["date"] = pd.to_datetime(df_orders["date"])

    date_index = pd.date_range(
        df_orders["date"].min(), df_orders["date"].max(), freq="D"
    )
    results = []

    # NOTE - add counter for y_imputations for price_volume_slope fxn - count_nan_imputations = 0
    for current_date in date_index:
        window_start = current_date - timedelta(days=window_days - 1)
        window_df = df_orders[
            (df_orders["date"] >= window_start) & (df_orders["date"] <= current_date)
        ]

        bids = window_df[window_df["direction"] == "buy"].sort_values(
            "price", ascending=False
        )
        asks = window_df[window_df["direction"] == "sell"].sort_values("price")

        bid_size = bids["size"].sum()
        ask_size = asks["size"].sum()
        total_size = window_df["size"].sum()

        book_imbalance = (
            (bid_size - ask_size) / total_size if total_size > 0 else np.nan
        )
        depth_midprice = (
            (window_df["price"] * window_df["size"]).sum() / total_size
            if total_size > 0
            else np.nan
        )

        def price_volume_slope(book_side):
            """
            This models price responsiveness to volume (cum_size) - we return the coefficient of the slope to understand elasticity dynamics
            """
            if len(book_side) < 2:
                return np.nan
            book_side = book_side.copy()
            book_side["cum_size"] = book_side["size"].cumsum()
            X = book_side["cum_size"].values.reshape(-1, 1)
            y = book_side["price"].values

            y_nan_count = np.count_nonzero(np.isnan(y))

            # TODO - add logging for y_imputed ...and create a counter - would have to take code out of fxn
            if y_nan_count > 0:
                # print(f"y: {y}")
                # print(y_nan_count)
                y = np.where(
                    np.isnan(y), np.ma.array(y, mask=np.isnan(y)).mean(axis=0), y
                )
                # print(f"y_imputed: {y}")

            reg = LinearRegression().fit(X, y)
            return reg.coef_[0]

        slope_bid = price_volume_slope(bids)
        slope_ask = price_volume_slope(asks)

        results.append(
            {
                "date": current_date,
                f"book_imbalance_{window_days}d": book_imbalance,
                f"depth_midprice_{window_days}d": depth_midprice,
                f"slope_bid_{window_days}d": slope_bid,
                f"slope_ask_{window_days}d": slope_ask,
                f"bid_count_{window_days}d": len(bids),
                f"ask_count_{window_days}d": len(asks),
                f"bid_size_total_{window_days}d": bid_size,
                f"ask_size_total_{window_days}d": ask_size,
            }
        )

    df_book = pd.DataFrame(results).set_index("date")

    print(f"df_book shape: {df_book.shape}")
    print(f"\ndf_book dtypes: \n{df_book.dtypes}")
    print(f"\ndf_book isna sum: \n{df_book.isna().sum()}")
    # print(f"\ndf_book head: {df_book.head()}")

    return df_book


def compute_df_spread_rolling(
    df_orders: pd.DataFrame, window_days: int = WINDOW_DAYS
) -> pd.DataFrame:
    """
    This computes the spread_{n}day - used for y_target - otherwise, features for X captured by compute_df_book_features_rolling
    Cannot use features here in X b/c they are spread = ask - bid (endogeneity, if goal is causal inference) - unless using for ARIMA-type model - e.g. y(t+1) ~ y + Ax + Bx + ... + e
    """
    date_index = pd.date_range(
        df_orders["date"].min(), df_orders["date"].max(), freq="D"
    )
    max_bids = df_orders[df_orders["direction"] == "buy"].groupby("date")["price"].max()
    min_asks = (
        df_orders[df_orders["direction"] == "sell"].groupby("date")["price"].min()
    )

    df_spread = pd.DataFrame(index=date_index)
    df_spread["bid_max_1d"] = max_bids
    df_spread["ask_min_1d"] = min_asks
    df_spread[f"bid_max_{window_days}d"] = (
        df_spread["bid_max_1d"].rolling(f"{window_days}D", min_periods=1).max()
    )
    df_spread[f"ask_min_{window_days}d"] = (
        df_spread["ask_min_1d"].rolling(f"{window_days}D", min_periods=1).min()
    )
    df_spread[f"spread_{window_days}d"] = (
        df_spread[f"ask_min_{window_days}d"] - df_spread[f"bid_max_{window_days}d"]
    )
    df_spread[f"spread_{window_days}d_future"] = df_spread[
        f"spread_{window_days}d"
    ].shift(periods=-1)

    print(f"df_spread shape: {df_spread.shape}")
    print(f"\ndf_spread dtypes: \n{df_spread.dtypes}")
    print(f"\ndf_spread isna sum: \n{df_spread.isna().sum()}")
    # print(f"\ndf_spread head: {df_spread.head()}")

    return df_spread


def add_df_features_all(
    df_exog: pd.DataFrame,
    df_last: pd.DataFrame,
    df_orders: pd.DataFrame,
    list_windows: List = LIST_WINDOWS,
) -> pd.DataFrame:  # df_spread was in place of df_orders, daily in place of df_exog
    """
    This adds in all the rolling features for each rolling window
    - NOTE - df_exog is derived from build_df_daily_calendar - exog variables are added to this
    - df_exog included in arguments b/c should be pre-computed - avoid processing twice
    - df_last included in arguments b/c should be pre-computed - avoid processing twice
    - df_orders used to calculate df_book and df_spread - rolling features which need to iterate over windows
    """
    # start with empty df_all
    df_all = pd.DataFrame(index=df_exog.index)

    # ffill and bfill missing data for df_exog
    df_exog = df_exog.ffill().bfill()

    # store pre-computed features first
    dict_features_all = {"df_exog": df_exog, "df_last": df_last}

    # compute rolling features over windows
    for window in list_windows:
        df_book_temp = compute_df_book_rolling(df_orders, window_days=window)
        # df_book = df_book.add_suffix(f'_{window}d')
        df_spread_temp = compute_df_spread_rolling(df_orders, window_days=window)
        dict_features_all[f"df_book_{window}d"] = df_book_temp
        dict_features_all[f"df_spread_{window}d"] = df_spread_temp

    # ensure index alignment before joining, otherwise will break
    dict_features_all = {
        name: df.reindex(df_all.index) for name, df in dict_features_all.items()
    }

    # de-duplicate columns before concatenation
    list_dfs_to_concat = []
    set_cols_seen = set()
    for name, df in dict_features_all.items():
        # only add columns that haven't been added yet
        list_cols_unique = [col for col in df.columns if col not in set_cols_seen]
        list_dfs_to_concat.append(df[list_cols_unique])
        set_cols_seen.update(list_cols_unique)

    # concatenate all dfs
    df_all = pd.concat([df_all] + list_dfs_to_concat, axis=1)

    # compute rolling daily/exog features
    for window in list_windows:
        for col in [
            "vix",
            "spy",
            "arkx",
            "xli",
            "treasury_10y",
            "fed_rate",
            "cpi",
            "unemp_u3",
            "unemp_u6",
            "m2",
        ]:
            df_all[f"{col}_{window}d_avg"] = (
                df_exog[col]
                .reindex(df_all.index)
                .fillna(method="ffill")
                .rolling(window, min_periods=1)
                .mean()
            )

    if all(x in list_windows for x in [7, 28, 56]):
        # use ratios instead of raw imbalance / depth_midprice, to help avoid multicollinearity
        df_all["imbalance_ratio_7d_28d"] = (
            df_all["book_imbalance_7d"] / df_all["book_imbalance_28d"]
        )
        df_all["imbalance_ratio_7d_56d"] = (
            df_all["book_imbalance_7d"] / df_all["book_imbalance_56d"]
        )
        df_all["depth_midprice_ratio_7d_28d"] = (
            df_all["depth_midprice_7d"] / df_all["depth_midprice_28d"]
        )
        df_all["depth_midprice_ratio_7d_56d"] = (
            df_all["depth_midprice_7d"] / df_all["depth_midprice_56d"]
        )

    # similarly, add spread_volatility and spread_entropy to avoid direct multicollinearity
    for window in list_windows:
        df_all[f"spread_volatility_{window}d"] = (
            df_all[f"spread_{window}d"].rolling(window, min_periods=1).std()
        )
        df_all[f"spread_entropy_{window}d"] = np.log1p(
            df_all[f"spread_{window}d"].rolling(window, min_periods=1).std()
            / df_all[f"spread_{window}d"].rolling(window, min_periods=1).mean().abs()
        )
    # replace infinity vals, if any
    df_all = df_all.replace([np.inf, -np.inf], np.nan)

    print(f"df_all shape: {df_all.shape}")
    print(f"\ndf_all dtypes: \n{df_all.dtypes}")
    print(f"\ndf_all isna sum: \n{df_all.isna().sum()}")
    print(f"\ndf_all isinf sum: \n{np.isinf(df_all).sum()}")
    # print(f"\ndf_all head: {df_all.head()}")

    return df_all


def build_feature_matrix(
    df_all: pd.DataFrame,
    y_target: str = Y_TARGET,
    window_days: int = WINDOW_DAYS,
    list_windows: List[int] = LIST_WINDOWS,
    vars_cat: List = VARS_CATEGORICAL,
    model_order_level: bool = MODEL_ORDER_LEVEL,
    drop_days_with_invalid_spread=DROP_DAYS_WITH_INVALID_SPREAD,
    imputation_method: Literal[
        LIST_SPREAD_IMPUTATION_METHOD
    ] = SPREAD_IMPUTATION_METHOD,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    This builds the final feature matrix, for use in modeling. Includes:
    - Dropping irrelevant cols
    - Imputation methods
    - Splitting into X, y
    - Dummy encoding for categorical variables - to ensure consistency across all models
    """
    # validate imputation_method (optional, since Literal already restricts it)
    # TODO - can layer on / nuance / mix imputation methods more
    allowed_methods = [None, "median", "zero", "ffill_bfill"]
    if imputation_method not in allowed_methods:
        raise ValueError(f"Invalid imputation_method. Choose from {allowed_methods}")

    y_series = df_all[y_target]

    # start with base columns
    # list_generated_feature_cols = list(df_all.columns)
    list_generated_feature_cols = [
        col for col in df_all.columns if y_target not in col and "date" not in col
    ]

    X = df_all[list_generated_feature_cols].copy()
    y = y_series.copy()

    # # dummy encoding for categorical variables
    # # NOTE - some major re-engineering needed if you want to do this - can't aggregate...
    # # NOTE - would need to use df_orders as base (in order to not lose order-level data) and THEN expand dates...take care handling joins
    # if model_at_order_level:
    #     if vars_cat:
    #         X = pd.get_dummies(X, columns=vars_cat, drop_first=True)

    # imputation - missing data
    # imputation - exog cols
    # ffill and then bfill isna macro cols - e.g. fed rate
    # NOTE - mean imputation may be better in some cases
    list_macro_cols = [col for col in X.columns if "_avg" in col]
    X[list_macro_cols] = X[list_macro_cols].fillna(method="ffill")
    X[list_macro_cols] = X[list_macro_cols].fillna(method="bfill")

    # imputation - book static cols
    # impute large (or median) value for missing last / days_since features
    # NOTE - bid_ or ask_ 0 imputation only for full OLS or time-series - otherwise drop columns
    X[[col for col in X.columns if "last" in col or "days" in col]] = X[
        [col for col in X.columns if "last" in col or "days" in col]
    ].apply(lambda col: col.fillna(col.median()))

    # imputation - book-level feature cols
    list_book_cols = [
        col
        for col in X.columns
        if "book" in col or "slope" in col or "imbalance" in col
    ]
    # impute neutral value = 0 for missing book-level features
    X[list_book_cols] = X[list_book_cols].fillna(0)

    # imputation - book rolling cols + spread cols
    if drop_days_with_invalid_spread:
        X = X.loc[y.notna()]
        y = y.loc[y.notna()]
    else:
        if imputation_method == "median":
            X[
                [
                    col
                    for col in X.columns
                    if "bid_" in col or "ask_" in col or "spread_" in col
                ]
            ] = X[
                [
                    col
                    for col in X.columns
                    if "bid_" in col or "ask_" in col or "spread_" in col
                ]
            ].apply(
                lambda col: col.fillna(col.median())
            )
            y = y.fillna(value=y.median())
        elif imputation_method == "zero":
            X[
                [
                    col
                    for col in X.columns
                    if "bid_" in col or "ask_" in col or "spread_" in col
                ]
            ] = X[
                [
                    col
                    for col in X.columns
                    if "bid_" in col or "ask_" in col or "spread_" in col
                ]
            ].fillna(
                0
            )
            y = y.fillna(0)
        elif imputation_method == "ffill_bfill":
            X[
                [
                    col
                    for col in X.columns
                    if "bid_" in col or "ask_" in col or "spread_" in col
                ]
            ] = (
                X[
                    [
                        col
                        for col in X.columns
                        if "bid_" in col or "ask_" in col or "spread_" in col
                    ]
                ]
                .ffill()
                .bfill()
            )
            y = y.ffill().bfill()

    # drop irrelevant cols
    X = X.drop(columns=[col for col in X.columns if "_future" in col])

    print(f"X shape: {X.shape}")
    print(f"\nX dtypes: \n{X.dtypes}")
    print(f"\nX isna sum: \n{X.isna().sum()}")
    # print(f"\nX head: {X.head()}")

    print(f"y shape: {y.shape}")
    print(f"\ny dtypes: \n{y.dtypes}")
    print(f"\ny isna sum: \n{y.isna().sum()}")
    # print(f"\ny head: {y.head()}")

    return X, y

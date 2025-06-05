# src/spread_predictor/data_loader.py

import os
from typing import Any, Dict, List

import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
from pytrends.request import TrendReq


# create general class to clean any incoming df
class cleanData(object):
    """
    --------
    Main class to clean a df
    --------
    """

    def __init__(self, df) -> None:
        """
        ----------
        Initializes a clean_data object
        ----------

        Args:
            - df (pd DataFrame) - pd df to clean

        Attributes:
            - self.df (pd DataFrame)

        Returns:
            - None
        """

        # initialize df as copy
        self.df = df.copy()

    def clean_df(
        self,
        col_date_key: str,
        date_vars: List,
        pct_vars: List,
        cat_vars: List,
        numeric_vars: List,
        rename_vars: Dict,
        verbose=False,
    ) -> None:
        """
        ----------
        Cleans the df - lowercases, renames, and converts vars to their appropriate dtypes
        --------

        Args:
            - date_vars (list) - list of var names to convert to pd.datetime

        Returns:
            - self (pd DataFrame)

        """

        # convert camelCase to snake_case
        self.df = self.df.rename(columns=rename_vars)

        # make stuff lower case
        self.df.columns = [x.lower() for x in self.df.columns]

        # convert date vars into datetime
        for col in date_vars:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col])

        # convert percent vars into float
        for col in pct_vars:
            if col in self.df.columns:
                self.df[col] = self.df[col].str.rstrip("%").astype("float") / 100

        # convert other vars into appropriate type
        for col in cat_vars:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")
        for col in numeric_vars:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # sort values by date
        if col_date_key not in self.df.columns:
            raise KeyError(
                f"Column '{col_date_key}' not found in the DataFrame. Available columns: {self.df.columns.tolist()}"
            )
        self.df = self.df.sort_values(col_date_key)

        # print head, optionally
        if verbose:
            print(self.df.head())

        return self


def load_raw_orders(csv_path: os.path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def fetch_yahoo_data(
    dict_yf_tickers: Dict, date_index: pd.DatetimeIndex
) -> pd.DataFrame:
    # date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    start_date = date_index.min()
    end_date = date_index.max()
    df_yf = pd.DataFrame(index=date_index)
    for key, symbol in dict_yf_tickers.items():
        ticker_data = yf.download(symbol, start=start_date, end=end_date)
        df_yf[key] = ticker_data["Close"]
    return df_yf


def fetch_fred_data(dict_fred_series: dict, date_index: pd.DatetimeIndex):
    # date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    start_date = date_index.min()
    end_date = date_index.max()
    df_fred = pd.DataFrame(index=date_index)
    for key, series in dict_fred_series.items():
        fred_data = pdr.data.DataReader(series, "fred", start_date, end_date)
        df_fred[key] = fred_data.reindex(df_fred.index)
    return df_fred


def fetch_google_trends(
    list_gt_keywords: List, date_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    google trends - can only request data occasionally
    """
    # date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    # pytrends = TrendReq(hl='en-US', tz=360)
    # kw_list = ['SpaceX']
    # pytrends.build_payload(kw_list, timeframe='{} {}'.format(start_date.date(), end_date.date()))
    # df_gt = pytrends.interest_over_time()
    # df_gt = df_gt.rename(columns={'SpaceX': 'spacex_trend'})#.reset_index() #.fillna(0)
    # date_index = pd.date_range(start=start_date, end=end_date, freq='D')
    start_date = date_index.min()
    end_date = date_index.max()
    pytrends = TrendReq(hl="en-US", tz=360)
    timeframe = f"{start_date.date()} {end_date.date()}"
    pytrends.build_payload(kw_list=list_gt_keywords, timeframe=timeframe)
    trends = pytrends.interest_over_time()
    return trends[list_gt_keywords] if not trends.empty else pd.DataFrame()

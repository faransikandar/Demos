import pandas as pd
import streamlit as st

from spread_predictor.data_loader import fetch_yahoo_data, load_raw_orders
from spread_predictor.features import make_features
from spread_predictor.model import train_ols

# TODO
# - create simple make_features fxn to create lagged spread data - ffill and bfill or median imputation to fill dataframe
# - pull ARKX from YF - use as one-variable model

st.title("Finance Forecast Dashboard")

ticker = st.text_input("Enter a stock ticker", value="ARKX")
start_date = st.date_input("Start date", pd.to_datetime("2022-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2023-01-01"))

if st.button("Load Data"):
    df = fetch_yahoo_finance(ticker, str(start_date), str(end_date))
    st.write(df.tail())

    if "Close" in df.columns:
        df = df[["Close"]].rename(columns={"Close": "price_arkx"})
        df_feat = make_features(df.copy())
        st.line_chart(df_feat["spread_7d_future"])
        df_feat = df_feat.dropna()
        if len(df_feat) > 1:
            model = train_ols(X=df_feat["price_arkx"], y=df_feat[["spread_7d_future"]])
            pred = model.predict(df_feat[["price_arkx"]])
            st.line_chart(
                pd.DataFrame({"True": df_feat["spread_7d"], "Predicted": pred})
            )

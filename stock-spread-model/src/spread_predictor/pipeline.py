from dagster import job, op

from spread_predictor.data_loader import fetch_yahoo_data, load_raw_orders

# NOTE - @op is outdated - should be using @asset


@op
def get_raw_order_data():
    return load_raw_orders("data/spacex_orders.csv")


@op
def get_finance_data():
    return fetch_yahoo_data("ARKX", start="2022-01-01", end="2023-01-01")


@job
def etl_pipeline():
    local = get_raw_order_data()
    finance = get_finance_data()

import pandas as pd

from spread_predictor.features import make_features


def test_make_features():
    df = pd.DataFrame({"price": [1, 2, 3]})
    features = make_features(df)
    assert "moving_avg" in features.columns

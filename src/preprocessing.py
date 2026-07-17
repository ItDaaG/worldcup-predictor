import pandas as pd
from .features import add_h2h_feature, add_form_features, add_tournament_type
from .elo_system import calculate_elo_ratings
TRAINING_WINDOW = 15


# Filters to last TRAINING_WINDOW years of games
def _filter_to_training_window(df, TRAINING_WINDOW):
    
    cutoff_date = df["date"].max() - pd.DateOffset(years=TRAINING_WINDOW)
    return df[df['date'] >= cutoff_date].copy()


def _drop_unplayed_games(df):
     return df[df["home_score"].notnull()].copy()

def _drop_worldcup2026_games(df):
    is_wc_2026 = (df["tournament"] == "FIFA World Cup") & (
        df["date"].dt.year == 2026
    )

    return df[~is_wc_2026].copy()
    

def preprocess(df):
    df['date'] = pd.to_datetime(df['date'])
    df = add_tournament_type(df)
    df = calculate_elo_ratings(df)
    df = _drop_unplayed_games(df)
    df = add_h2h_feature(df)
    df = add_form_features(df)
    df = _filter_to_training_window(df, TRAINING_WINDOW)
    df = _drop_worldcup2026_games(df)
    return df

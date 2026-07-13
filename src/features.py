"""
Feature engineering for match prediction.
Builds rolling-form and head-to-head features from match history.

All features are computed using only information available BEFORE
the match being predicted (no leakage).
"""
import pandas as pd

ROLLING_WINDOW = 5


def _build_team_long_format(df):
    """
    Reshapes match-level data into one row per team per match
    (i.e. two rows per match: one from each team's perspective).
    Each row keeps the original match_id and an is_home flag so the
    results can be merged back safely later, without relying on
    (date, team) being a unique key.
    """
    df['match_id'] = df.index

    home = df[['match_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score', 'home_elo']].copy()
    home.columns = ['match_id', 'date', 'team', 'opponent', 'goals_scored', 'goals_conceded', 'elo']
    home['is_home'] = True

    away = df[['match_id', 'date', 'away_team', 'home_team', 'away_score', 'home_score', 'away_elo']].copy()
    away.columns = ['match_id', 'date', 'team', 'opponent', 'goals_scored', 'goals_conceded', 'elo']
    away['is_home'] = False

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(['team', 'date']).reset_index(drop=True)

    return long_df, df


def _add_rolling_team_stats(long_df, window=ROLLING_WINDOW):
    """
    Adds rolling averages/momentum per team, using only PAST games
    (shift(1) before rolling, so the current match is never included).
    """
    grouped = long_df.groupby('team')

    long_df['average_goals_scored'] = (
        grouped['goals_scored']
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    long_df['average_conceded_goals'] = (
        grouped['goals_conceded']
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    long_df['elo_momentum'] = (
        grouped['elo']
        .transform(lambda s: s.shift(1) - s.shift(window + 1))
    )

    return long_df


def add_form_features(df, window=ROLLING_WINDOW):
    """
    Adds average_goals_scored, average_conceded_goals, and elo_momentum
    for both home and away teams to the match-level dataframe.
    """
    long_df, df = _build_team_long_format(df)
    long_df = _add_rolling_team_stats(long_df, window)

    stat_cols = ['average_goals_scored', 'average_conceded_goals', 'elo_momentum']

    home_stats = long_df[long_df['is_home']][['match_id'] + stat_cols].copy()
    home_stats.columns = ['match_id'] + [f'home_{c}' for c in stat_cols]

    away_stats = long_df[~long_df['is_home']][['match_id'] + stat_cols].copy()
    away_stats.columns = ['match_id'] + [f'away_{c}' for c in stat_cols]

    df = df.merge(home_stats, on='match_id', how='left')
    df = df.merge(away_stats, on='match_id', how='left')
    df = df.drop(columns=['match_id'])

    return df



def add_h2h_feature(df, window=ROLLING_WINDOW):
    """
    Adds h2h_win_pct: The percentage of the last `window` meetings won by the HOME team.
    If teams have never met, defaults to 0.5 (neutral expectation).
    """

    df['pair_key'] = df.apply(lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1)

    h2h_results = []
    history = {} 

    for row in df.itertuples(index=False):
        pair_key = row.pair_key
        
        if pair_key in history:
            past_winners = history[pair_key][-window:]
            total_matches = len(past_winners)
            
            # Count home wins
            home_wins = sum(1 for winner in past_winners if winner == row.home_team) + \
                        sum(0.5 for winner in past_winners if winner is None)

            win_pct = home_wins / total_matches
        else:
            #No history = neutral expectation
            win_pct = 0.5
            
        h2h_results.append(win_pct)

        # Update history for the next game
        if row.home_score > row.away_score:
            winner = row.home_team
        elif row.away_score > row.home_score:
            winner = row.away_team
        else:
            winner = None

        history.setdefault(pair_key, []).append(winner)



    df['h2h_win_pct'] = h2h_results
    
    df = df.drop(columns=['pair_key'])

    return df

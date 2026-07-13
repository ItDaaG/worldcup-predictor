"""
Elo rating system for international football matches.
Based on the World Football Elo Ratings methodology.
"""

import pandas as pd

DEFAULT_ELO = 1500
HOME_ADVANTAGE = 100

TOURNAMENT_K_FACTORS = {
    'Major Tournament': 60,
    'Qualification': 40,
    'Other / Regional Cup': 30,
    'Friendly': 10
}


def get_goal_diff_multiplier(goal_diff):
    """
    The official World Football Elo margin-of-victory multiplier.
    """
    abs_diff = abs(goal_diff)

    if abs_diff <= 1:
        return 1.0
    elif abs_diff == 2:
        return 1.5
    elif abs_diff == 3:
        return 1.75
    else:
        return 1.75 + (abs_diff - 3) / 8.0


def get_expected_score(home_elo, away_elo, neutral):
    """
    Probability of the home team winning, adjusted for home advantage
    unless the match is played at a neutral venue.
    """
    home_adj = home_elo if neutral else home_elo + HOME_ADVANTAGE
    return 1 / (1 + 10 ** ((away_elo - home_adj) / 400))


def get_actual_score(home_score, away_score):
    """
    1 for a home win, 0.5 for a draw, 0 for a home loss.
    """
    if home_score > away_score:
        return 1.0
    elif home_score == away_score:
        return 0.5
    else:
        return 0.0


def update_elo(home_elo, away_elo, home_score, away_score, tournament_type, neutral):
    """
    Returns the updated (home_elo, away_elo) after a single match.
    """
    k = TOURNAMENT_K_FACTORS.get(tournament_type, TOURNAMENT_K_FACTORS['Friendly'])
    goal_diff = home_score - away_score
    multiplier = get_goal_diff_multiplier(goal_diff)

    expected_home = get_expected_score(home_elo, away_elo, neutral)
    actual_home = get_actual_score(home_score, away_score)

    change = k * multiplier * (actual_home - expected_home)

    new_home_elo = home_elo + change
    new_away_elo = away_elo - change

    return new_home_elo, new_away_elo


def calculate_elo_ratings(df):
    """
    Runs Elo over the full match history of dataset in chronological order.
    Returns the original dataframe with home_elo/away_elo columns added
    (ratings BEFORE each match is played.
    """
    df = df.sort_values('date').reset_index(drop=True)

    ratings = {}
    home_elos = []
    away_elos = []

    for row in df.itertuples(index=False):
        home_elo = ratings.get(row.home_team, DEFAULT_ELO)
        away_elo = ratings.get(row.away_team, DEFAULT_ELO)

        home_elos.append(home_elo)
        away_elos.append(away_elo)

        new_home_elo, new_away_elo = update_elo(
            home_elo, away_elo,
            row.home_score, row.away_score,
            row.tournament_type, row.neutral
        )

        ratings[row.home_team] = new_home_elo
        ratings[row.away_team] = new_away_elo

    df['home_elo'] = home_elos
    df['away_elo'] = away_elos

    return df
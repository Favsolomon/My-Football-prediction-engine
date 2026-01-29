
# src/config.py


# API Keys
ODDS_API_KEY = "a136f290325a43885ca0bccc99576edb"

# League Mappings (Understat)
LEAGUES_UNDERSTAT = {
    "Champions League": "UCL",
    "Premier League": "EPL",
    "La Liga": "La_Liga",
    "Serie A": "Serie_A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue_1",
    "Russian Premier League": "RFPL"
}

# League Mappings (Odds API)
LEAGUES_ODDS_API = {
    "Champions League": "soccer_uefa_champs_league",
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Russian Premier League": "soccer_russia_premier_league" 
}

# UCL Intellectual Mappings (Heritage & Value)
UCL_PEDIGREE = {
    "Real Madrid": 1.15, "AC Milan": 1.12, "Bayern Munich": 1.10,
    "Liverpool": 1.10, "Barcelona": 1.08, "Ajax": 1.07,
    "Inter": 1.06, "Manchester United": 1.06, "Juventus": 1.05,
    "Chelsea": 1.05, "FC Porto": 1.04, "Benfica": 1.03
}

SQUAD_VALUE_INDEX = {
    "Manchester City": 1.25, "Arsenal": 1.20, "Real Madrid": 1.22,
    "Paris Saint Germain": 1.18, "Bayern Munich": 1.15, "Liverpool": 1.15,
    "Barcelona": 1.12, "Chelsea": 1.10, "Inter": 1.08, "Bayer Leverkusen": 1.05,
    "Milan": 1.04, "Zenit St. Petersburg": 0.85, "Krasnodar": 0.75
}

# Senior Correction: League Quality Coefficients for Cross-Border Normalization
LEAGUE_COEFFICIENTS = {
    "Manchester City": 1.0, "Arsenal": 1.0, "Liverpool": 1.0,
    "Real Madrid": 0.98, "Barcelona": 0.96,
    "Bayern Munich": 0.95, "Bayer Leverkusen": 0.94,
    "Inter": 0.94, "Juventus": 0.93,
    "Paris Saint Germain": 0.90, "Zenit St. Petersburg": 0.78
}





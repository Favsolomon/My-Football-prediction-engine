# src/data_service.py
import pandas as pd
import numpy as np
import requests
import streamlit as st
import concurrent.futures
from understatapi import UnderstatClient
from .config import ODDS_API_KEY, LEAGUES_UNDERSTAT, LEAGUES_ODDS_API

class DataService:
    @staticmethod
    @st.cache_data(ttl=3600)
    def fetch_league_data(league_code, season="2025"):
        """Fetches and cleans league data from Understat."""
        if league_code == "UCL":
            return pd.DataFrame(), []
            
        try:
            with UnderstatClient() as client:
                matches = client.league(league=league_code).get_match_data(season=season)
            
            if not matches:
                return pd.DataFrame(), []

            processed_data = []
            teams = set()
            
            for m in matches:
                if not m.get('isResult'):
                    row = {
                        'Home': m['h']['title'], 'Away': m['a']['title'],
                        'xG': np.nan, 'xG.1': np.nan,
                        'Score': None, 'DateTime': m['datetime']
                    }
                else:
                    row = {
                        'Home': m['h']['title'], 'Away': m['a']['title'],
                        'xG': float(m['xG']['h']), 'xG.1': float(m['xG']['a']),
                        'Score': f"{m['goals']['h']}-{m['goals']['a']}",
                        'DateTime': m['datetime']
                    }
                processed_data.append(row)
                teams.add(m['h']['title'])
                teams.add(m['a']['title'])
                
            df = pd.DataFrame(processed_data)
            df['xG'] = pd.to_numeric(df['xG'], errors='coerce')
            df['xG.1'] = pd.to_numeric(df['xG.1'], errors='coerce')
            return df, sorted(list(teams))
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame(), []

    @staticmethod
    def normalize_team_name(name):
        """Maps diverse API names to Understat standards."""
        mapping = {
            "Man City": "Manchester City", "Man Utd": "Manchester United", "Man United": "Manchester United",
            "Real Madrid": "Real Madrid", "Atleti": "Atletico Madrid", "Atletico Madrid": "Atletico Madrid",
            "Bayern": "Bayern Munich", "Bayer": "Bayer Leverkusen", "Dortmund": "Borussia Dortmund",
            "AC Milan": "Milan", "Inter Milan": "Inter", "Tottenham": "Tottenham", "Spurs": "Tottenham",
            "PSG": "Paris Saint Germain", "Lille": "Lille", "Monaco": "Monaco", "Inter": "Inter",
            "St Petersburg": "Zenit St. Petersburg", "Krasnodar": "Krasnodar", "CSKA": "CSKA Moscow",
            "Wolves": "Wolverhampton Wanderers", "Brighton": "Brighton", "Newcastle": "Newcastle United",
            "West Ham": "West Ham", "Leicester": "Leicester", "Villa": "Aston Villa", "Forest": "Nottingham Forest"
        }
        return mapping.get(name, name)

    @staticmethod
    def fetch_ucl_fixtures(api_key, sport_key):
        """Fetches UCL fixtures from Odds API to provide 'Upcoming' list with diagnostic logic."""
        if not api_key: return pd.DataFrame(), "MISSING_KEY"
        
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions=eu&markets=h2h"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 401:
                return pd.DataFrame(), "QUOTA_EXCEEDED"
            elif r.status_code != 200:
                return pd.DataFrame(), f"API_ERROR_{r.status_code}"
                
            data = r.json()
            fixtures = []
            for m in data:
                fixtures.append({
                    'Home': m['home_team'], 'Away': m['away_team'],
                    'xG': np.nan, 'xG.1': np.nan, 'Score': None,
                    'DateTime': m['commence_time']
                })
            return pd.DataFrame(fixtures), "OK"
        except Exception as e: 
            return pd.DataFrame(), "CONNECTION_ERROR"

    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_team_logo(team_name):
        """Fetches team logo URL with strict timeout and fallback."""
        try:
            search_name = team_name.replace(" ", "%20")
            url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={search_name}"
            r = requests.get(url, timeout=1.5)
            data = r.json()
            if data and data.get('teams'):
                return data['teams'][0]['strBadge']
        except:
            pass
        return None

    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_live_odds(api_key, sport_key, home_team, away_team):
        """Fetches live H2H and Totals odds from The-Odds-API for model biasing."""
        if not api_key: return None
        try:
            # Fetch both h2h and totals
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h,totals&apiKey={api_key}"
            r = requests.get(url, timeout=5)
            data = r.json()
            
            for match in data:
                m_home, m_away = match['home_team'], match['away_team']
                # Flexible name matching
                if (home_team in m_home or m_home in home_team) and (away_team in m_away or m_away in away_team):
                    market_data = {'h2h': {}, 'totals': {}}
                    for bookmaker in match['bookmakers']:
                        for market in bookmaker['markets']:
                            if market['key'] == 'h2h':
                                for outcome in market['outcomes']:
                                    if outcome['name'] == m_home: market_data['h2h']['home'] = max(market_data['h2h'].get('home', 0), outcome['price'])
                                    elif outcome['name'] == m_away: market_data['h2h']['away'] = max(market_data['h2h'].get('away', 0), outcome['price'])
                                    elif outcome['name'] == 'Draw': market_data['h2h']['draw'] = max(market_data['h2h'].get('draw', 0), outcome['price'])
                            elif market['key'] == 'totals':
                                for outcome in market['outcomes']:
                                    if outcome['name'] == 'Over' and outcome['point'] == 2.5: 
                                        market_data['totals']['over25'] = max(market_data['totals'].get('over25', 0), outcome['price'])
                                    elif outcome['name'] == 'Over' and outcome['point'] == 1.5:
                                        market_data['totals']['over15'] = max(market_data['totals'].get('over15', 0), outcome['price'])
                    return market_data
        except: pass
        return None

    @staticmethod
    def parallel_ucl_fetch(season):
        """Ultra-fast parallel aggregator for UCL domestic data."""
        domestic_leagues = [c for c in LEAGUES_UNDERSTAT.values() if c != "UCL"]
        master_dfs = []
        
        def fetch_worker(code):
            d, _ = DataService.fetch_league_data(code, season)
            p, _ = DataService.fetch_league_data(code, str(int(season)-1))
            return [d, p]

        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            results = list(executor.map(fetch_worker, domestic_leagues))
            for res in results:
                master_dfs.extend(res)
                
        return pd.concat(master_dfs) if master_dfs else pd.DataFrame()
    def preload_competition_context(league_name, season="2025"):
        """Background loader worker for simultaneous league fetching (Data Only)."""
        is_ucl = league_name == "Champions League"
        league_code = LEAGUES_UNDERSTAT[league_name]
        status = "OK"
        
        if is_ucl:
            df = DataService.parallel_ucl_fetch(season)
            upcoming, status = DataService.fetch_ucl_fixtures(ODDS_API_KEY, LEAGUES_ODDS_API[league_name])
        else:
            d_df, _ = DataService.fetch_league_data(league_code, season)
            df = d_df
            upcoming = d_df[d_df['xg'].isna() if 'xg' in d_df.columns else d_df['xG'].isna()].copy()

        return league_name, {"df": df, "upcoming": upcoming, "status": status}

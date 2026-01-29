
# src/data_service.py
import pandas as pd
import numpy as np
import requests
from cachetools.func import ttl_cache
import concurrent.futures
from understatapi import UnderstatClient
from .config import ODDS_API_KEY, LEAGUES_UNDERSTAT, LEAGUES_ODDS_API

class DataService:
    @staticmethod
    @ttl_cache(ttl=3600)
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
            print(f"Error fetching data: {e}")
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
    def fetch_free_ucl_fixtures():
        """Senior Scraper: Extracts UCL fixtures with multi-month lookahead and robust parsing."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        # Check current month and Feb 2026 (next round)
        urls = [
            "https://www.skysports.com/champions-league-fixtures",
            "https://www.skysports.com/champions-league-scores-fixtures/2026-02-01"
        ]
        
        all_fixtures = []
        import re
        
        for url in urls:
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code != 200: continue
                html = r.text
                
                # Split by date headers
                date_blocks = re.split(r'class="fixres__header"', html)
                for block in date_blocks[1:]:
                    date_match = re.search(r'>(.*?)<', block)
                    if not date_match: continue
                    date_str = date_match.group(1).strip()
                    
                    # Split into individual match items
                    items = re.split(r'class="fixres__item"', block)
                    for item in items[1:]:
                        # Robust team extraction (handles both swap-text and simple spans)
                        teams = re.findall(r'class="swap-text--heavy">(.*?)</span>', item)
                        if len(teams) < 2:
                            # Fallback if swap-text is missing
                            teams = re.findall(r'class="fixres__team--(?:home|away)".*?<span>(.*?)</span>', item, re.DOTALL)
                        
                        if len(teams) >= 2:
                            home, away = teams[0].strip(), teams[1].strip()
                            # Check if it's already finished (has a score)
                            is_result = "fixres__score" in item
                            
                            if not is_result: # We only want upcoming fixtures
                                all_fixtures.append({
                                    'Home': home, 'Away': away,
                                    'xG': np.nan, 'xG.1': np.nan, 'Score': None,
                                    'DateTime': date_str
                                })
            except: continue
            
        df = pd.DataFrame(all_fixtures)
        # Deduplicate and return
        if not df.empty:
            df = df.drop_duplicates(subset=['Home', 'Away'])
            return df, "OK"
        return pd.DataFrame(), "NO_UPCOMING_MATCHES"

    @staticmethod
    @ttl_cache(ttl=86400)
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
    @ttl_cache(ttl=300)
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
    @ttl_cache(ttl=3600)
    def parallel_ucl_fetch(season="2024"):
        """Senior Aggregator: Builds a global team database across all 5 top leagues."""
        domestic_codes = [c for c in LEAGUES_UNDERSTAT.values() if c != "UCL"]
        master_dfs = []
        
        def fetch_worker(code):
            try:
<<<<<<< Updated upstream
                d, _ = DataService.fetch_league_data(code, season)
                # Only fetch previous season if current season has very little data (early season)
                if len(d) < 50:
                    p, _ = DataService.fetch_league_data(code, str(int(season)-1))
                    return [d, p]
                return [d]
            except:
                return []
=======
                # Fetch 2 seasons to ensure team coverage and H2H data
                s1, _ = DataService.fetch_league_data(code, season)
                s2, _ = DataService.fetch_league_data(code, str(int(season)-1))
                return [s1, s2]
            except: return []
>>>>>>> Stashed changes

        # Parallel fetch for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(domestic_codes)) as executor:
            results = list(executor.map(fetch_worker, domestic_codes))
            for res in results:
<<<<<<< Updated upstream
                master_dfs.extend(res)
                
        return pd.concat(master_dfs) if master_dfs else pd.DataFrame()
=======
                if res: master_dfs.extend(res)
        
        if not master_dfs: return pd.DataFrame()
        return pd.concat(master_dfs).sort_values('DateTime')
>>>>>>> Stashed changes
    @staticmethod
    @ttl_cache(ttl=1800)
    def preload_competition_context(league_name, season="2025"):
        """Background loader worker for simultaneous league fetching (Data Only)."""
        is_ucl = league_name == "Champions League"
        league_code = LEAGUES_UNDERSTAT[league_name]
        status = "OK"
        
        if is_ucl:
            df = DataService.parallel_ucl_fetch(season if int(season) < 2025 else "2024")
            upcoming, status = DataService.fetch_free_ucl_fixtures()
        else:
            d_df, _ = DataService.fetch_league_data(league_code, season)
            df = d_df
            upcoming = d_df[d_df['xg'].isna() if 'xg' in d_df.columns else d_df['xG'].isna()].copy()

        return league_name, {"df": df, "upcoming": upcoming, "status": status}

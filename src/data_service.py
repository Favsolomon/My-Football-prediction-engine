
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
            
            processed_data = []
            teams = set()
            cols = ['Home', 'Away', 'xG', 'xG.1', 'Score', 'DateTime']
            
            if not matches:
                return pd.DataFrame(columns=cols), []
            
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
            cols = ['Home', 'Away', 'xG', 'xG.1', 'Score', 'DateTime']
            return pd.DataFrame(columns=cols), []

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
    @ttl_cache(ttl=900) # Cache for 15 mins to respect API quotas
    def fetch_odds_batch(api_key, sport_key):
        """Fetch ALL odds for a league at once to minimize API calls and latency."""
        if not api_key or not sport_key: return {}
        
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h,totals&apiKey={api_key}"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200: 
                print(f"Odds API Error {r.status_code}: {r.text}")
                return {}
            
            data = r.json()
            odds_map = {} # Key: (Home, Away) normalized -> Odds Data
            
            for match in data:
                h = DataService.normalize_team_name(match['home_team'])
                a = DataService.normalize_team_name(match['away_team'])
                
                market_data = {'h2h': {}, 'totals': {}}
                
                # Best odds aggregation
                best_h = 0
                best_d = 0
                best_a = 0
                best_o25 = 0
                best_o15 = 0
                
                for bookmaker in match['bookmakers']:
                    for market in bookmaker['markets']:
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == match['home_team']: best_h = max(best_h, outcome['price'])
                                elif outcome['name'] == match['away_team']: best_a = max(best_a, outcome['price'])
                                elif outcome['name'] == 'Draw': best_d = max(best_d, outcome['price'])
                        elif market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                if outcome['name'] == 'Over' and outcome['point'] == 2.5: 
                                    best_o25 = max(best_o25, outcome['price'])
                                elif outcome['name'] == 'Over' and outcome['point'] == 1.5:
                                    best_o15 = max(best_o15, outcome['price'])
                                    
                market_data['h2h'] = {'home': best_h, 'away': best_a, 'draw': best_d}
                market_data['totals'] = {'over25': best_o25, 'over15': best_o15}
                
                odds_map[f"{h}_vs_{a}"] = market_data
                
            return odds_map
        except Exception as e:
            print(f"Batch Odds Fetch Error: {e}")
            return {}

    @staticmethod
    def get_odds_for_fixture(odds_map, home, away):
        """Robust lookup using direct normalized keys and fuzzy fallbacks."""
        if not odds_map: return None
        
        h_norm = DataService.normalize_team_name(home)
        a_norm = DataService.normalize_team_name(away)
        
        # 1. Direct Try
        key = f"{h_norm}_vs_{a_norm}"
        if key in odds_map: return odds_map[key]
        
        # 2. Fuzzy Match from 'fuzzywuzzy' logic (simplified here to avoid heavy deps if possible, but user installed it)
        # Using simple substring overlap first for speed
        for k, v in odds_map.items():
            k_home, k_away = k.split('_vs_')
            # Check overlap
            if (h_norm in k_home or k_home in h_norm) and (a_norm in k_away or k_away in a_norm):
                return v
                
        return None

    @staticmethod
    @ttl_cache(ttl=300)
    def fetch_live_odds(api_key, sport_key, home_team, away_team):
        """Fetches live H2H and Totals odds from The-Odds-API for model biasing."""
        if not api_key: return None
        try:
            # Fetch both h2h and totals
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h,totals&apiKey={api_key}"
            # Reverting to simple request if needed, but this method is now deprecated in favor of batch
            r = requests.get(url, timeout=5)
            # ... (logic omitted for brevity as we are moving to batch)
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
                # Fetch 2 seasons to ensure team coverage and H2H data
                s1, _ = DataService.fetch_league_data(code, season)
                s2, _ = DataService.fetch_league_data(code, str(int(season)-1))
                return [s1, s2]
            except: return []

        # Parallel fetch for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(domestic_codes)) as executor:
            results = list(executor.map(fetch_worker, domestic_codes))
            for res in results:
                if res: master_dfs.extend(res)
        
        if not master_dfs: return pd.DataFrame()
        return pd.concat(master_dfs).sort_values('DateTime')
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
            # Fetch current and previous 2 seasons to ensure 5 H2H matches
            seasons = [season, str(int(season)-1), str(int(season)-2)]
            season_dfs = []
            for s in seasons:
                d_df, _ = DataService.fetch_league_data(league_code, s)
                if not d_df.empty: season_dfs.append(d_df)
            
            df = pd.concat(season_dfs).sort_values('DateTime') if season_dfs else pd.DataFrame()
            
            # Upcoming matches are only from the current season
            current_df = season_dfs[0] if season_dfs else pd.DataFrame()
            if not current_df.empty:
                upcoming = current_df[current_df['xg'].isna() if 'xg' in current_df.columns else current_df['xG'].isna()].copy()
            else:
                upcoming = pd.DataFrame(columns=['Home', 'Away', 'xG', 'xG.1', 'Score', 'DateTime'])

        return league_name, {"df": df, "upcoming": upcoming, "status": status}

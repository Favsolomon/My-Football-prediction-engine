
# src/data_service.py
import pandas as pd
import numpy as np
import requests
from cachetools.func import ttl_cache
import concurrent.futures
from datetime import datetime, timedelta
from .config import ODDS_API_KEY, LEAGUES_UNDERSTAT, LEAGUES_ODDS_API

# ESPN league codes (replaces understatapi which is now blocked by Understat)
ESPN_LEAGUE_CODES = {
    "EPL":        "eng.1",
    "La_Liga":    "esp.1",
    "Serie_A":    "ita.1",
    "Bundesliga": "ger.1",
    "Ligue_1":    "fra.1",
    "RFPL":       "rus.1",
}

class DataService:

    @staticmethod
    @ttl_cache(ttl=3600)
    def fetch_league_data(league_code, season="2026"):
        """Fetches league data from the ESPN API using a rolling day-by-day window.
        Goals scored are used as a proxy for xG (same column names, compatible with engine).
        Understatapi was replaced because understat.com now blocks Python scrapers."""
        cols = ['Home', 'Away', 'xG', 'xG.1', 'Score', 'DateTime']

        if league_code == "UCL":
            return pd.DataFrame(columns=cols), []

        espn_code = ESPN_LEAGUE_CODES.get(league_code)
        if not espn_code:
            print(f"[DataService] No ESPN code for league_code='{league_code}'")
            return pd.DataFrame(columns=cols), []

        # Rolling window: 120 past days of results + 30 future days of fixtures
        today = datetime.now()
        start = today - timedelta(days=120)
        end   = today + timedelta(days=30)

        dates_to_query = []
        cur = start
        while cur <= end:
            dates_to_query.append(cur.strftime('%Y%m%d'))
            cur += timedelta(days=1)

        def fetch_date(date_str):
            try:
                url = (
                    f"https://site.api.espn.com/apis/site/v2/sports/soccer"
                    f"/{espn_code}/scoreboard?dates={date_str}"
                )
                r = requests.get(url, timeout=8)
                if r.status_code == 200:
                    return r.json().get('events', [])
            except Exception:
                pass
            return []

        # Parallel fetch — 30 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            day_results = list(executor.map(fetch_date, dates_to_query))

        seen_ids = set()
        processed_data = []
        teams = set()

        for events in day_results:
            for e in events:
                event_id = e.get('id')
                if event_id in seen_ids:
                    continue
                seen_ids.add(event_id)

                try:
                    comp = e['competitions'][0]
                    is_result = comp.get('status', {}).get('type', {}).get('completed', False)

                    home_team = away_team = None
                    home_score = away_score = None

                    for c in comp.get('competitors', []):
                        name = DataService.normalize_team_name(c['team']['displayName'])
                        score = c.get('score')
                        score_val = None
                        if isinstance(score, dict):
                            score_val = score.get('value')
                        elif score is not None:
                            try:
                                score_val = float(score)
                            except (TypeError, ValueError):
                                pass

                        if c['homeAway'] == 'home':
                            home_team, home_score = name, score_val
                        else:
                            away_team, away_score = name, score_val

                    if not home_team or not away_team:
                        continue

                    teams.add(home_team)
                    teams.add(away_team)
                    date_iso = e['date'].replace('T', ' ').replace('Z', '')

                    if is_result and home_score is not None and away_score is not None:
                        row = {
                            'Home': home_team, 'Away': away_team,
                            'xG': float(home_score), 'xG.1': float(away_score),
                            'Score': f"{int(home_score)}-{int(away_score)}",
                            'DateTime': date_iso
                        }
                    else:
                        row = {
                            'Home': home_team, 'Away': away_team,
                            'xG': np.nan, 'xG.1': np.nan,
                            'Score': None, 'DateTime': date_iso
                        }
                    processed_data.append(row)

                except Exception as row_err:
                    print(f"[DataService] Skipping malformed event: {row_err}")
                    continue

        if not processed_data:
            print(f"[DataService] WARNING: No events found for {league_code}")
            return pd.DataFrame(columns=cols), []

        df = pd.DataFrame(processed_data)
        df['xG']   = pd.to_numeric(df['xG'],   errors='coerce')
        df['xG.1'] = pd.to_numeric(df['xG.1'], errors='coerce')
        df = df.sort_values('DateTime').reset_index(drop=True)
        return df, sorted(list(teams))

    @staticmethod
    def normalize_team_name(name):
        """Maps diverse API names to consistent internal standards."""
        mapping = {
            # English
            "Man City": "Manchester City", "Man Utd": "Manchester United",
            "Man United": "Manchester United",
            "Wolves": "Wolverhampton Wanderers",
            "Brighton & Hove Albion": "Brighton",
            "Brighton and Hove Albion": "Brighton",
            "Newcastle United": "Newcastle United", "Newcastle": "Newcastle United",
            "West Ham United": "West Ham", "West Ham": "West Ham",
            "Leicester City": "Leicester", "Leicester": "Leicester",
            "Aston Villa": "Aston Villa", "Villa": "Aston Villa",
            "Nottingham Forest": "Nottingham Forest", "Forest": "Nottingham Forest",
            "Tottenham Hotspur": "Tottenham", "Tottenham": "Tottenham", "Spurs": "Tottenham",
            "Coventry City": "Coventry",
            # Spanish
            "Atletico Madrid": "Atletico Madrid", "Atleti": "Atletico Madrid",
            "Real Madrid": "Real Madrid",
            # German
            "Bayern Munich": "Bayern Munich", "Bayern": "Bayern Munich",
            "Borussia Dortmund": "Borussia Dortmund", "Dortmund": "Borussia Dortmund",
            "Bayer Leverkusen": "Bayer Leverkusen", "Bayer": "Bayer Leverkusen",
            # Italian
            "AC Milan": "Milan", "Inter Milan": "Inter", "Inter": "Inter",
            # French
            "Paris Saint-Germain": "Paris Saint Germain",
            "PSG": "Paris Saint Germain",
            "Lille": "Lille", "Monaco": "Monaco",
            # Russian
            "Zenit St. Petersburg": "Zenit St. Petersburg",
            "St Petersburg": "Zenit St. Petersburg",
            "Krasnodar": "Krasnodar", "CSKA Moscow": "CSKA Moscow", "CSKA": "CSKA Moscow",
            # UCL generic
            "Man City": "Manchester City",
        }
        return mapping.get(name, name)

    @staticmethod
    def fetch_free_ucl_fixtures():
        """Scrapes upcoming UCL fixtures from the ESPN JSON API (60-day lookahead)."""
        today  = datetime.now()
        future = today + timedelta(days=60)
        start_date = today.strftime('%Y%m%d')
        end_date   = future.strftime('%Y%m%d')

        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/soccer"
            f"/uefa.champions/scoreboard?dates={start_date}-{end_date}"
        )

        all_fixtures = []
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                for e in r.json().get('events', []):
                    try:
                        date_str = e['date'].replace('T', ' ').replace('Z', '')
                        competitors = e['competitions'][0]['competitors']
                        home_team = away_team = None
                        for c in competitors:
                            if c['homeAway'] == 'home':
                                home_team = c['team']['displayName']
                            else:
                                away_team = c['team']['displayName']
                        if home_team and away_team:
                            h_norm = DataService.normalize_team_name(home_team)
                            a_norm = DataService.normalize_team_name(away_team)
                            all_fixtures.append({
                                'Home': h_norm, 'Away': a_norm,
                                'xG': np.nan, 'xG.1': np.nan,
                                'Score': None, 'DateTime': date_str
                            })
                    except Exception:
                        continue
        except Exception as e:
            print(f"[DataService] ESPN UCL Fetch Error: {e}")

        df = pd.DataFrame(all_fixtures)
        if not df.empty:
            df = df.drop_duplicates(subset=['Home', 'Away'])
            return df, "OK"
        return pd.DataFrame(), "NO_UPCOMING_MATCHES"

    @staticmethod
    @ttl_cache(ttl=86400)
    def fetch_team_logo(team_name):
        """Fetches team badge URL from TheSportsDB with strict timeout."""
        try:
            search_name = team_name.replace(" ", "%20")
            url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={search_name}"
            r = requests.get(url, timeout=1.5)
            data = r.json()
            if data and data.get('teams'):
                return data['teams'][0]['strBadge']
        except Exception:
            pass
        return None

    @staticmethod
    @ttl_cache(ttl=900)
    def fetch_odds_batch(api_key, sport_key):
        """Fetch ALL odds for a league at once to minimise API calls and latency."""
        if not api_key or not sport_key:
            return {}

        url = (
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
            f"?regions=eu&markets=h2h,totals&apiKey={api_key}"
        )
        try:
            r = requests.get(url, timeout=5)
            if r.status_code != 200:
                print(f"[DataService] Odds API Error {r.status_code}: {r.text}")
                return {}

            odds_map = {}
            for match in r.json():
                h = DataService.normalize_team_name(match['home_team'])
                a = DataService.normalize_team_name(match['away_team'])

                best_h = best_d = best_a = best_o25 = best_o15 = 0

                for bookmaker in match['bookmakers']:
                    for market in bookmaker['markets']:
                        if market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == match['home_team']:
                                    best_h = max(best_h, outcome['price'])
                                elif outcome['name'] == match['away_team']:
                                    best_a = max(best_a, outcome['price'])
                                elif outcome['name'] == 'Draw':
                                    best_d = max(best_d, outcome['price'])
                        elif market['key'] == 'totals':
                            for outcome in market['outcomes']:
                                if outcome['name'] == 'Over' and outcome['point'] == 2.5:
                                    best_o25 = max(best_o25, outcome['price'])
                                elif outcome['name'] == 'Over' and outcome['point'] == 1.5:
                                    best_o15 = max(best_o15, outcome['price'])

                odds_map[f"{h}_vs_{a}"] = {
                    'h2h':    {'home': best_h, 'away': best_a, 'draw': best_d},
                    'totals': {'over25': best_o25, 'over15': best_o15}
                }

            return odds_map
        except Exception as e:
            print(f"[DataService] Batch Odds Fetch Error: {e}")
            return {}

    @staticmethod
    def get_odds_for_fixture(odds_map, home, away):
        """Robust lookup using direct normalized keys with substring fallback."""
        if not odds_map:
            return None

        h_norm = DataService.normalize_team_name(home)
        a_norm = DataService.normalize_team_name(away)

        key = f"{h_norm}_vs_{a_norm}"
        if key in odds_map:
            return odds_map[key]

        # Fuzzy substring fallback
        for k, v in odds_map.items():
            k_home, k_away = k.split('_vs_')
            if (h_norm in k_home or k_home in h_norm) and (a_norm in k_away or k_away in a_norm):
                return v

        return None

    @staticmethod
    @ttl_cache(ttl=300)
    def fetch_live_odds(api_key, sport_key, home_team, away_team):
        """Deprecated in favour of batch fetch — kept for API compatibility."""
        return None

    @staticmethod
    @ttl_cache(ttl=3600)
    def parallel_ucl_fetch(season="2026"):
        """Builds a global team database across all 5 top domestic leagues for UCL context."""
        domestic_codes = [c for c in LEAGUES_UNDERSTAT.values() if c != "UCL"]
        master_dfs = []

        def fetch_worker(code):
            try:
                df, _ = DataService.fetch_league_data(code, season)
                return [df]
            except Exception:
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(domestic_codes)) as executor:
            results = list(executor.map(fetch_worker, domestic_codes))
            for res in results:
                if res:
                    master_dfs.extend(res)

        if not master_dfs:
            return pd.DataFrame()
        valid = [d for d in master_dfs if not d.empty]
        return pd.concat(valid).sort_values('DateTime') if valid else pd.DataFrame()

    @staticmethod
    @ttl_cache(ttl=1800)
    def preload_competition_context(league_name, season="2026"):
        """Background loader: fetches match data and upcoming fixtures for a league."""
        is_ucl = league_name == "Champions League"
        league_code = LEAGUES_UNDERSTAT[league_name]
        status = "OK"

        if is_ucl:
            df = DataService.parallel_ucl_fetch(season)
            upcoming, status = DataService.fetch_free_ucl_fixtures()
        else:
            # Single rolling-window fetch (ESPN ignores season param internally)
            df, _ = DataService.fetch_league_data(league_code, season)

            # Upcoming = rows where xG is NaN (not yet played)
            if not df.empty and 'xG' in df.columns:
                upcoming = df[df['xG'].isna()].copy()
            else:
                upcoming = pd.DataFrame(columns=['Home', 'Away', 'xG', 'xG.1', 'Score', 'DateTime'])

        return league_name, {"df": df, "upcoming": upcoming, "status": status}

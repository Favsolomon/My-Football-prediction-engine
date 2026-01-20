import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
from understatapi import UnderstatClient

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
st.set_page_config(page_title="Match Predictor PRO", layout="wide", initial_sidebar_state="collapsed")

# API Keys
ODDS_API_KEY = "a136f290325a43885ca0bccc99576edb"

# League Mappings
LEAGUES_UNDERSTAT = {
    "Champions League": "UCL",
    "Premier League": "EPL",
    "La Liga": "La_Liga",
    "Serie A": "Serie_A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue_1",
    "Russian Premier League": "RFPL"
}

LEAGUES_ODDS_API = {
    "Champions League": "soccer_uefa_champions_league",
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

# Advanced Glassmorphism CSS
# Advanced Glassmorphism CSS (Dual-Mode Support)
APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    :root {
        /* Dark Mode */
        --app-bg: #0f172a;
        --app-text: #f8fafc;
        --app-subtext: #94a3b8;
        --card-bg: rgba(255, 255, 255, 0.05);
        --card-border: rgba(255, 255, 255, 0.12);
        --card-shadow: rgba(0, 0, 0, 0.5);
        --outcome-bg: rgba(255, 255, 255, 0.03);
        --badge-bg: rgba(255, 255, 255, 0.08);
        --score-bg: rgba(16, 185, 129, 0.15);
        --score-border: rgba(16, 185, 129, 0.3);
        --hero-grad: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        --value-grad: linear-gradient(135deg, #422006 0%, #0f172a 100%);
        --safety-grad: linear-gradient(135deg, #4b5563 0%, #1f2937 100%);
    }

    @media (prefers-color-scheme: light) {
        :root {
            /* Light Mode */
            --app-bg: #ffffff;
            --app-text: #0f172a;
            --app-subtext: #475569;
            --card-bg: #f8fafc;
            --card-border: #e2e8f0;
            --card-shadow: rgba(0, 0, 0, 0.05);
            --outcome-bg: #f1f5f9;
            --badge-bg: #e2e8f0;
            --score-bg: #ecfdf5;
            --score-border: #10b98144;
            --hero-grad: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            --value-grad: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            --safety-grad: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        }
    }

    /* Strict Text Visibility Overrides */
    .stApp, .stMarkdown, p, span, div, h1, h2, h3, h4, h5, h6, label, .stCaption, .stSelectbox label {
        color: var(--app-text) !important;
    }
    
    * { font-family: 'Outfit', sans-serif; }
    .stApp { background: var(--app-bg); }
    
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px var(--card-shadow);
    }

    .hero-card {
        background: var(--hero-grad);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 10px 25px var(--card-shadow);
        border: 1px solid var(--card-border);
    }
    
    .hero-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }

    .confidence-glow {
        height: 6px;
        background: #10b981;
        border-radius: 3px;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
        margin: 10px 0;
    }
    
    /* Match Outcome Horizontal Bar */
    .outcome-bar {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: var(--outcome-bg);
        border-radius: 12px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid var(--card-border);
    }
    
    .outcome-item { text-align: center; }
    .outcome-value { font-size: 1.1rem; font-weight: 800; color: var(--app-text); }
    
    .badge-circle {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        border-radius: 50%;
        background: var(--badge-bg);
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 4px;
        color: var(--app-text);
    }

    /* Heat Meter */
    .heat-meter-container {
        width: 100%;
        height: 12px;
        background: var(--badge-bg);
        border-radius: 6px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .heat-meter-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .heat-high { background: #10b981; box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }
    .heat-mid { background: #f59e0b; }
    .heat-low { background: #ef4444; }

    /* High-Tech Tactical Visuals */
    .tactical-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin-top: 15px;
    }
    .radar-bar-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .radar-bar-base {
        height: 6px;
        background: var(--badge-bg);
        border-radius: 3px;
        overflow: hidden;
        display: flex;
    }
    .radar-bar-h { background: #60a5fa; height: 100%; }
    .radar-bar-a { background: #f87171; height: 100%; }
    
    .momentum-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin: 0 4px;
    }
    .dot-win { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .dot-draw { background: #94a3b8; }
    .dot-loss { background: #ef4444; }

    .script-box {
        border-left: 3px solid #60a5fa;
        background: var(--outcome-bg);
        padding: 12px 15px;
        margin: 15px 0;
        font-size: 0.85rem;
        line-height: 1.6;
        font-style: italic;
        border-radius: 0 8px 8px 0;
        color: var(--app-subtext);
    }

    /* Hero League Discovery Grid */
    .league-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 25px;
        margin-top: 40px;
        padding-bottom: 60px;
    }
    
    .league-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 28px;
        padding: 45px 20px;
        text-align: center;
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 30px var(--card-shadow);
        border-bottom: 4px solid var(--card-border);
    }
    
    .league-card:hover {
        transform: translateY(-12px);
        border-color: #3b82f6;
        border-bottom-color: #3b82f6;
        box-shadow: 0 20px 45px rgba(59, 130, 246, 0.3);
    }
    
    .league-card-icon {
        font-size: 3.5rem;
        margin-bottom: 20px;
        display: block;
        filter: drop-shadow(0 0 10px rgba(59, 130, 246, 0.2));
    }
    
    .league-card-name {
        font-weight: 800;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--app-text);
    }
    
    .league-card-tag {
        font-size: 0.65rem;
        text-transform: uppercase;
        color: #60a5fa;
        font-weight: 800;
        margin-top: 10px;
        opacity: 0.8;
    }

    .change-btn {
        background: var(--badge-bg);
        border: 1px solid var(--card-border);
        padding: 6px 15px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
        color: var(--app-subtext);
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }
    .change-btn:hover {
        color: var(--app-text);
        border-color: #3b82f6;
    }
</style>
"""

# ==============================================================================
# DATA SERVICES
# ==============================================================================
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
            "St Petersburg": "Zenit St. Petersburg", "Krasnodar": "Krasnodar", "CSKA": "CSKA Moscow"
        }
        return mapping.get(name, name)

    @staticmethod
    def fetch_ucl_fixtures(api_key, sport_key):
        """Fetches UCL fixtures from Odds API to provide 'Upcoming' list."""
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions=eu&markets=h2h"
        try:
            r = requests.get(url)
            data = r.json()
            fixtures = []
            for m in data:
                fixtures.append({
                    'Home': m['home_team'], 'Away': m['away_team'],
                    'xG': np.nan, 'xG.1': np.nan, 'Score': None,
                    'DateTime': m['commence_time']
                })
            return pd.DataFrame(fixtures)
        except: return pd.DataFrame()

    @staticmethod
    @st.cache_data(ttl=86400)
    def fetch_team_logo(team_name):
        """Fetches team logo URL from TheSportsDB."""
        try:
            search_name = team_name.replace(" ", "%20")
            url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={search_name}"
            r = requests.get(url, timeout=2)
            data = r.json()
            if data['teams']:
                return data['teams'][0]['strBadge']
        except:
            pass
        return None

    @staticmethod
    @st.cache_data(ttl=300)
    def fetch_live_odds(api_key, sport_key, home_team, away_team):
        """Fetches live odds from The-Odds-API."""
        if not api_key: return None
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h&apiKey={api_key}"
            r = requests.get(url, timeout=5)
            data = r.json()
            
            for match in data:
                m_home, m_away = match['home_team'], match['away_team']
                if (home_team in m_home or m_home in home_team) and (away_team in m_away or m_away in away_team):
                    best_odds = {'home': 0, 'draw': 0, 'away': 0}
                    for bookmaker in match['bookmakers']:
                        for market in bookmaker['markets']:
                            if market['key'] == 'h2h':
                                for outcome in market['outcomes']:
                                    if outcome['name'] == m_home and outcome['price'] > best_odds['home']:
                                        best_odds['home'] = outcome['price']
                                    elif outcome['name'] == m_away and outcome['price'] > best_odds['away']:
                                        best_odds['away'] = outcome['price']
                                    elif outcome['name'] == 'Draw' and outcome['price'] > best_odds['draw']:
                                        best_odds['draw'] = outcome['price']
                    return best_odds
        except:
            pass
        return None

# ==============================================================================
# PREDICTION ENGINE
# ==============================================================================
class MatchPredictor:
    """Core logic engine for computing probabilities and value recommendations."""

    def get_league_stats(self, df):
        """Computes league average xG for home and away teams."""
        played = df.dropna(subset=['xG', 'xG.1'])
        if played.empty:
            return 1.3, 1.3  # Default baselines
        avg_home_xg = played['xG'].mean()
        avg_away_xg = played['xG.1'].mean()
        return avg_home_xg, avg_away_xg

    def get_league_table(self, df):
        """Generates a league table from matching history to calculate rankings and dominance."""
        table = {}
        played = df.dropna(subset=['Score'])
        
        for _, row in played.iterrows():
            try:
                h_score, a_score = map(int, row['Score'].split('-'))
                h_team, a_team = row['Home'], row['Away']
                
                for team in [h_team, a_team]:
                    if team not in table:
                        table[team] = {'pts': 0, 'gp': 0, 'h_pts': 0, 'h_gp': 0, 'a_pts': 0, 'a_gp': 0}
                
                table[h_team]['gp'] += 1
                table[a_team]['gp'] += 1
                
                if h_score > a_score:
                    table[h_team]['pts'] += 3
                    table[h_team]['h_pts'] += 3
                    table[h_team]['h_gp'] += 1
                    table[a_team]['a_gp'] += 1
                elif h_score < a_score:
                    table[a_team]['pts'] += 3
                    table[a_team]['a_pts'] += 3
                    table[a_team]['a_gp'] += 1
                    table[h_team]['h_gp'] += 1
                else:
                    table[h_team]['pts'] += 1
                    table[a_team]['pts'] += 1
                    table[h_team]['h_pts'] += 1
                    table[a_team]['a_pts'] += 1
                    table[h_team]['h_gp'] += 1
                    table[a_team]['a_gp'] += 1
            except:
                continue
        
        sorted_table = sorted(table.items(), key=lambda x: x[1]['pts'], reverse=True)
        return {team: {'rank': i+1, **stats} for i, (team, stats) in enumerate(sorted_table)}

    def calculate_elo(self, df):
        """Calculates a simple Elo rating for all teams based on match results."""
        elo = {team: 1500 for team in pd.concat([df['Home'], df['Away']]).unique()}
        played = df.dropna(subset=['Score'])
        K = 32
        
        for _, row in played.iterrows():
            try:
                h, a = row['Home'], row['Away']
                h_score, a_score = map(int, row['Score'].split('-'))
                
                # Expected outcomes
                r_h, r_a = elo[h], elo[a]
                e_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
                e_a = 1 / (1 + 10 ** ((r_h - r_a) / 400))
                
                # Actual outcomes
                s_h = 1 if h_score > a_score else 0.5 if h_score == a_score else 0
                s_a = 1 - s_h
                
                # Update Elo
                elo[h] += K * (s_h - e_h)
                elo[a] += K * (s_a - e_a)
            except:
                continue
        return elo

    def tau_adjustment(self, x, y, l_h, l_a, rho=-0.1):
        """Dixon-Coles adjustment function for low-scoring interdependence."""
        if x == 0 and y == 0: return 1 - (l_h * l_a * rho)
        elif x == 0 and y == 1: return 1 + (l_h * rho)
        elif x == 1 and y == 0: return 1 + (l_a * rho)
        elif x == 1 and y == 1: return 1 - rho
        return 1.0

    def run_monte_carlo(self, l_h, l_a, iterations=10000):
        """Runs 10,000 simulations to derive probabilistic outcomes and variance."""
        h_sims = np.random.poisson(l_h, iterations)
        a_sims = np.random.poisson(l_a, iterations)
        
        results = h_sims - a_sims
        h_wins = np.sum(results > 0)
        draws = np.sum(results == 0)
        a_wins = np.sum(results < 0)
        
        # Expected Points (xP)
        h_xp = (h_wins * 3 + draws * 1) / iterations
        a_xp = (a_wins * 3 + draws * 1) / iterations
        
        return {
            "h_win": h_wins / iterations, "draw": draws / iterations, "a_win": a_wins / iterations,
            "h_xp": h_xp, "a_xp": a_xp, "avg_goals": np.mean(h_sims + a_sims)
        }

    def calculate_strength(self, team, df, is_home, avg_home_xg, avg_away_xg, league_table=None, elo=None):
        """Calculates strength using Elo-weighted xG and League Quality Multipliers."""
        played = df.dropna(subset=['xG', 'xG.1'])
        team_matches = played[(played['Home'] == team) | (played['Away'] == team)].tail(8)
        
        # League Quality Coefficients
        LEAGUE_COEFFICIENTS = {
            "Manchester City": 1.0, "Arsenal": 1.0, "Liverpool": 1.0, # EPL Baseline
            "Real Madrid": 0.98, "Barcelona": 0.96, # La Liga
            "Bayern Munich": 0.95, "Bayer Leverkusen": 0.94, # Bundesliga
            "Inter": 0.94, "Juventus": 0.93, # Serie A
            "Paris Saint Germain": 0.90, # Ligue 1
            "Zenit St. Petersburg": 0.78 # RFPL
        }
        # Dynamic fallback based on typical league averages
        coeff = LEAGUE_COEFFICIENTS.get(team, 0.85) 
        
        # UCL Heritage & Economic Overrides
        pedigree = UCL_PEDIGREE.get(team, 1.0)
        squad_val = SQUAD_VALUE_INDEX.get(team, 1.0)
        
        if team_matches.empty:
            atk_strength, def_strength = 1.0, 1.0
        else:
            avg_league_elo = np.mean(list(elo.values())) if elo else 1500
            atk_vals, def_vals, weight_array = [], [], []
            
            for i, (_, row) in enumerate(team_matches.iterrows()):
                is_team_home = row['Home'] == team
                opponent = row['Away'] if is_team_home else row['Home']
                opp_elo = elo.get(opponent, 1500) if elo else 1500
                
                # Quality of Opposition Weight (Harder opponents = higher weight for performance)
                q_weight = (opp_elo / avg_league_elo) ** 1.5 
                # Recency weight (Standard 1-8 scale)
                r_weight = (i + 1) / len(team_matches)
                total_weight = q_weight * r_weight
                
                if is_team_home:
                    atk_vals.append(row['xG'])
                    def_vals.append(row['xG.1'])
                else:
                    atk_vals.append(row['xG.1'])
                    def_vals.append(row['xG'])
                weight_array.append(total_weight)
                    
            team_avg_atk = np.average(atk_vals, weights=weight_array) if atk_vals else 0
            team_avg_def = np.average(def_vals, weights=weight_array) if def_vals else 1.0
            
            # Cross-League Adjustments using League Quality Coefficients
            if is_home:
                atk_strength = (team_avg_atk / avg_home_xg) * coeff
                def_strength = (team_avg_def / avg_away_xg) * (2.0 - coeff)
            else:
                atk_strength = (team_avg_atk / avg_away_xg) * coeff
                def_strength = (team_avg_def / avg_home_xg) * (2.0 - coeff)

            # Apply UCL DNA (Pedigree) & Squad Depth (Value)
            atk_strength *= (pedigree * squad_val)
            def_strength /= (pedigree * squad_val) # Better squads have better defenses

        # 2. Add League Context & Elite Status
        if league_table and team in league_table:
            stats = league_table[team]
            # General Team Ranking Effect
            rank_boost = max(0.9, 1.1 - (stats['rank'] / 20 * 0.2)) 
            atk_strength *= rank_boost
            
            if stats['rank'] <= 4:
                atk_strength *= 1.05 
            
            # Current Form / Domestic Dominance
            if is_home and stats['h_gp'] > 0:
                h_ppg = stats['h_pts'] / stats['h_gp']
                if h_ppg > 2.0: atk_strength *= 1.1
            elif not is_home and stats['a_gp'] > 0:
                a_ppg = stats['a_pts'] / stats['a_gp']
                if a_ppg > 1.8: atk_strength *= 1.05
                
        return atk_strength, def_strength, pedigree, squad_val

    def get_h2h_history(self, home, away, df):
        """Analyzes Head-to-Head history for tactical trends."""
        h2h = df[((df['Home'] == home) & (df['Away'] == away)) | 
                 ((df['Home'] == away) & (df['Away'] == home))].dropna(subset=['Score']).tail(5)
        
        if h2h.empty: return None
        
        h_wins, a_wins, draws = 0, 0, 0
        total_goals = 0
        for _, row in h2h.iterrows():
            hs, ascore = map(int, row['Score'].split('-'))
            total_goals += (hs + ascore)
            if hs == ascore: draws += 1
            elif (row['Home'] == home and hs > ascore) or (row['Away'] == home and ascore > hs): h_wins += 1
            else: a_wins += 1

        return {
            "h_wins": h_wins, "a_wins": a_wins, "draws": draws,
            "avg_goals": total_goals / len(h2h),
            "recent": h2h.to_dict('records')
        }

    def predict_match(self, home_team, away_team, df):
        """Runs Advanced Analytical Pipeline: Dixon-Coles Correction + Monte Carlo."""
        avg_h_xg, avg_a_xg = self.get_league_stats(df)
        league_table = self.get_league_table(df)
        elo = self.calculate_elo(df)

        h_atk, h_def, h_ped, h_val = self.calculate_strength(home_team, df, True, avg_h_xg, avg_a_xg, league_table, elo)
        a_atk, a_def, a_ped, a_val = self.calculate_strength(away_team, df, False, avg_h_xg, avg_a_xg, league_table, elo)

        l_home = h_atk * a_def * avg_h_xg * 1.08  # Enhanced HFA
        l_away = a_atk * h_def * avg_a_xg * 0.92

        h_pmf = poisson.pmf(np.arange(10), l_home)
        a_pmf = poisson.pmf(np.arange(10), l_away)
        matrix = np.outer(h_pmf, a_pmf)

        for i in range(2):
            for j in range(2):
                matrix[i, j] *= self.tau_adjustment(i, j, l_home, l_away)

        mc = self.run_monte_carlo(l_home, l_away)
        h2h = self.get_h2h_history(home_team, away_team, df)

        # 4. H2H Tactical Weighting (User Feedback Integration)
        if h2h:
            # Win Rate Influence
            if h2h['h_wins'] > h2h['a_wins']: l_home *= 1.05
            elif h2h['a_wins'] > h2h['h_wins']: l_away *= 1.05

            # Scoring Trend Influence
            if h2h['avg_goals'] > 3.2:
                l_home *= 1.05
                l_away *= 1.05
            elif h2h['avg_goals'] < 1.8:
                l_home *= 0.92
                l_away *= 0.92

        h_win = np.sum(np.tril(matrix, -1))
        draw_prob = np.sum(np.diag(matrix))
        a_win = np.sum(np.triu(matrix, 1))

        return {
            "home": home_team, "away": away_team,
            "l_home": l_home, "l_away": l_away,
            "h_win": h_win, "draw": draw_prob, "a_win": a_win,
            "mc_h_win": mc['h_win'], "mc_draw": mc['draw'], "mc_a_win": mc['a_win'],
            "h_xp": mc['h_xp'], "a_xp": mc['a_xp'],
            "btts": (1 - h_pmf[0]) * (1 - a_pmf[0]),
            "over25": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)]),
            "under25": np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)]),
            "over15": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(2) for j in range(2-i)]),
            "under35": np.sum([h_pmf[i]*a_pmf[j] for i in range(4) for j in range(4-i)]),
            "predicted_score": f"{np.unravel_index(matrix.argmax(), matrix.shape)[0]}-{np.unravel_index(matrix.argmax(), matrix.shape)[1]}",
            "elo_h": elo.get(home_team, 1500), "elo_a": elo.get(away_team, 1500),
            "h2h": h2h, "ped_h": h_ped, "ped_a": a_ped, "val_h": h_val, "val_a": a_val
        }

    def get_recommendations(self, res, live_odds=None):
        """Combines Probabilities, Monte Carlo, and Market Odds for Edge Detection."""
        h_xg, a_xg = res['l_home'], res['l_away']
        primary_pick, secondary_pick = None, None
        primary_insight, secondary_insight = "", ""
        edge_data = None

        # 1. Market Edge Detection
        if live_odds:
            try:
                m_probs = {'home': 1/live_odds['home'], 'draw': 1/live_odds['draw'], 'away': 1/live_odds['away']}
                edges = {'Home': res['mc_h_win'] - m_probs['home'], 'Draw': res['mc_draw'] - m_probs['draw'], 'Away': res['mc_a_win'] - m_probs['away']}
                best_edge_key = max(edges, key=edges.get)
                if edges[best_edge_key] > 0.05:
                    edge_data = {"market": best_edge_key, "value": edges[best_edge_key]}
            except: pass

        # 2. Logic Hierarchy
        if a_xg > 2.25 and h_xg > 1.60:
            primary_pick = "BTTS (Yes)"
            primary_insight = "Offensive metrics are off the charts for both sides. A shootout is statistically inevitable."
        elif res['mc_h_win'] > 0.65:
            primary_pick = f"{res['home']} Win"
            primary_insight = f"{res['home']} dominance at home is backed by an elite Elo momentum of {int(res['elo_h'])}."
        elif res['mc_a_win'] > 0.60:
            primary_pick = f"{res['away']} Win"
            primary_insight = f"{res['away']} holds a major tactical edge. Their xG generation away is consistent."
        elif res['mc_h_win'] > res['mc_a_win']:
            primary_pick = "Home/Draw (1X)"
            primary_insight = "A tight match-up where home advantage provides the critical safety margin."
        else:
            primary_pick = "Away/Draw (X2)"
            primary_insight = "Model suggests the visitors are robust enough to avoid defeat."

        # 3. Secondary Pick (Value Finder)
        if edge_data:
            secondary_pick = f"VALUE: {edge_data['market']}"
            secondary_insight = f"Model identifies a {edge_data['value']:.1%} edge over bookmaker odds. High value detected."
        elif res['over25'] > 0.65:
            secondary_pick = "Over 2.5 Goals"
            secondary_insight = "Goal expectancy is high. Both defenses show vulnerabilities today."
        else:
            secondary_pick = "Under 3.5 Goals"
            secondary_insight = "A controlled tactical battle is expected with focus on defensive discipline."

        return {
            "primary_pick": primary_pick, "primary_insight": primary_insight,
            "secondary_pick": secondary_pick, "secondary_insight": secondary_insight,
            "edge": edge_data
        }

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
def inject_css():
    """Injects the global CSS theme."""
    st.markdown(APP_CSS, unsafe_allow_html=True)

def render_match_header(res, match_date):
    """Renders the FotMob-inspired match header with team logos and Elo Momentum."""
    logo_h = DataService.fetch_team_logo(res['home'])
    logo_a = DataService.fetch_team_logo(res['away'])
    img_h = f"<img src='{logo_h}' style='height: 80px;'>" if logo_h else ""
    img_a = f"<img src='{logo_a}' style='height: 80px;'>" if logo_a else ""

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 40px; margin-bottom: 20px;">
        <div style="text-align: center; flex: 1;">
            {img_h}
            <div style="font-weight: 800; font-size: 1.4rem; margin-top: 10px;">{res['home']}</div>
            <div style="display: flex; justify-content: center; gap: 5px; margin-top: 4px;">
                <div style="font-size: 0.7rem; color: #10b981; font-weight: 800; background: rgba(16,185,129,0.1); padding: 2px 6px; border-radius: 4px;">ELO: {int(res['elo_h'])}</div>
                {f'<div style="font-size: 0.7rem; color: #3b82f6; font-weight: 800; background: rgba(59,130,246,0.1); padding: 2px 6px; border-radius: 4px;">UCL DNA</div>' if res.get('ped_h', 1)>1.05 else ''}
            </div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.2rem; font-weight: 300; color: #94a3b8;">VS</div>
            <div style="font-size: 0.9rem; font-weight: 600; background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 20px; margin-top: 5px;">
                {match_date.strftime('%H:%M')}
            </div>
        </div>
        <div style="text-align: center; flex: 1;">
            {img_a}
            <div style="font-weight: 800; font-size: 1.4rem; margin-top: 10px;">{res['away']}</div>
            <div style="display: flex; justify-content: center; gap: 5px; margin-top: 4px;">
                <div style="font-size: 0.7rem; color: #10b981; font-weight: 800; background: rgba(16,185,129,0.1); padding: 2px 6px; border-radius: 4px;">ELO: {int(res['elo_a'])}</div>
                {f'<div style="font-size: 0.7rem; color: #3b82f6; font-weight: 800; background: rgba(59,130,246,0.1); padding: 2px 6px; border-radius: 4px;">UCL DNA</div>' if res.get('ped_a', 1)>1.05 else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_outcome_bar(res):
    """Renders the Monte Carlo powered H/D/A horizontal bar with xP."""
    st.markdown(f"""
    <div class="outcome-bar">
        <div class="outcome-item">
            <div class="badge-circle" style="color: #60a5fa;">H</div>
            <div class="outcome-value">{res['mc_h_win']:.0%}</div>
            <div style="font-size: 0.65rem; opacity: 0.6;">{res['h_xp']:.1f} xP</div>
        </div>
        <div class="outcome-item">
            <div class="badge-circle" style="color: #94a3b8;">D</div>
            <div class="outcome-value">{res['mc_draw']:.0%}</div>
        </div>
        <div class="outcome-item">
            <div class="badge-circle" style="color: #f87171;">A</div>
            <div class="outcome-value">{res['mc_a_win']:.0%}</div>
            <div style="font-size: 0.65rem; opacity: 0.6;">{res['a_xp']:.1f} xP</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_top_picks(recs):
    """Renders the High-Value Hero Cards with optional Value glow."""
    st.markdown("### 🏆 High-Value Picks")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="hero-card">
            <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.8; font-weight: 800;">Primary Pick</div>
            <div style="font-size: 1.3rem; font-weight: 800; margin: 8px 0;">{recs['primary_pick']}</div>
            <div class="confidence-glow" style="width: 85%;"></div>
            <div style="font-size: 0.85rem; font-style: italic; opacity: 0.9;">"{recs['primary_insight']}"</div>
        </div>
        """, unsafe_allow_html=True)

    if recs['secondary_pick']:
        is_value = "VALUE" in recs['secondary_pick']
        card_class = "value-grad" if is_value else "safety-grad"
        glow_color = "#f59e0b" if is_value else "#94a3b8"
        
        with col2:
            st.markdown(f"""
            <div class="hero-card" style="background: var(--{card_class}); border: 1px solid {glow_color}66;">
                <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.8; font-weight: 800;">Safety Pick</div>
                <div style="font-size: 1.3rem; font-weight: 800; margin: 8px 0;">{recs['secondary_pick']}</div>
                <div class="confidence-glow" style="width: 70%; background: {glow_color}; box-shadow: 0 0 10px {glow_color};"></div>
                <div style="font-size: 0.85rem; font-style: italic; opacity: 0.9;">"{recs['secondary_insight']}"</div>
            </div>
            """, unsafe_allow_html=True)

def render_analytics(res):
    """Renders Goals, Simulation, and Head-to-Head Tactical history."""
    with st.expander("🔬 Elite Simulation & Tactical History", expanded=True):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # 1. H2H Trends
        if res['h2h']:
            h2h = res['h2h']
            st.markdown(f"""
            <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 800; margin-bottom: 10px;">Head-to-Head History</div>
            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <div style="background: rgba(16,185,129,0.1); padding: 5px 15px; border-radius: 8px; font-weight: 800;">{res['home']} {h2h['h_wins']}W</div>
                <div style="background: rgba(255,255,255,0.05); padding: 5px 15px; border-radius: 8px;">{h2h['draws']}D</div>
                <div style="background: rgba(239,68,68,0.1); padding: 5px 15px; border-radius: 8px; font-weight: 800;">{res['away']} {h2h['a_wins']}W</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 2. Heat Meters
        h_xg, a_xg = res['l_home'], res['l_away']
        h_heat = "heat-high" if h_xg > 2.0 else "heat-mid" if h_xg > 1.2 else "heat-low"
        a_heat = "heat-high" if a_xg > 2.0 else "heat-mid" if a_xg > 1.2 else "heat-low"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>Offensive Momentum (xG)</span>
            <span style="font-weight: 800;">{h_xg:.2f} vs {a_xg:.2f}</span>
        </div>
        <div style="display: flex; gap: 4px;">
            <div class="heat-meter-container" style="flex: 1;"><div class="heat-meter-fill {h_heat}" style="width: {min(h_xg/3*100, 100)}%;"></div></div>
            <div class="heat-meter-container" style="flex: 1;"><div class="heat-meter-fill {a_heat}" style="width: {min(a_xg/3*100, 100)}%;"></div></div>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        # 3. Probabilities
        st.markdown("<div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; text-align: center;'>", unsafe_allow_html=True)
        for label, prob in [("BTTS", res['btts']), ("OVER 2.5", res['over25']), ("UNDER 2.5", res['under25'])]:
            st.markdown(f"<div><div style='font-size:0.7rem; color:#94a3b8;'>{label}</div><div style='font-size:1.1rem; font-weight:800;'>{prob:.0%}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"⚡ Dixon-Coles Correction + 10,000 Monte Carlo Simulations applied.")

def render_advanced_intelligence(res, df):
    """Renders high-tech tactical modules: Radar Comparison, Momentum Pulse, and Game Script."""
    st.markdown("### 🧬 Advanced Match Intelligence")
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # 1. Tactical Radar Comparison
        st.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 800; margin-bottom: 15px;">Advanced Competitive Indicators</div>', unsafe_allow_html=True)
        
        metrics = [
            ("UCL DNA (Heritage)", res.get('ped_h', 1.0)/2, res.get('ped_a', 1.0)/2),
            ("Economic Multiplier", res.get('val_h', 1.0)/2, res.get('val_a', 1.0)/2),
            ("Attack Conversion", res['l_home']/(res['l_home']+res['l_away']), res['l_away']/(res['l_home']+res['l_away'])),
            ("Win Probability", res['mc_h_win'], res['mc_a_win'])
        ]
        
        for label, h_val, a_val in metrics:
            h_pct = max(5, min(95, h_val * 100))
            a_pct = max(5, min(95, a_val * 100))
            st.markdown(f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 4px;">
                    <span style="color: #60a5fa;">{res['home']}</span>
                    <span>{label}</span>
                    <span style="color: #f87171;">{res['away']}</span>
                </div>
                <div class="radar-bar-base">
                    <div class="radar-bar-h" style="width: {h_pct}%;"></div>
                    <div style="width: 2px; background: rgba(0,0,0,0.3);"></div>
                    <div class="radar-bar-a" style="width: {a_pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Momentum Pulse (Last 6 Games)
        col1, col2 = st.columns(2)
        
        def get_momentum_dots(team, df):
            played = df[((df['Home'] == team) | (df['Away'] == team)) & df['Score'].notna()].tail(6)
            dots = ""
            for _, row in played.iterrows():
                h_score, a_score = map(int, row['Score'].split('-'))
                if row['Home'] == team:
                    res_char = 'win' if h_score > a_score else 'draw' if h_score == a_score else 'loss'
                else:
                    res_char = 'win' if a_score > h_score else 'draw' if h_score == a_score else 'loss'
                dots += f'<span class="momentum-dot dot-{res_char}"></span>'
            return dots

        with col1:
            st.markdown(f"""
            <div style="font-size: 0.7rem; color: #94a3b8; margin-bottom: 5px;">{res['home']} Momentum</div>
            <div>{get_momentum_dots(res['home'], df)}</div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="font-size: 0.7rem; color: #94a3b8; margin-bottom: 5px; text-align: right;">{res['away']} Momentum</div>
            <div style="text-align: right;">{get_momentum_dots(res['away'], df)}</div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. Predicted Game Script (Enhanced with UCL Intelligence)
        st.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 800;">AI Tactical Game Script</div>', unsafe_allow_html=True)
        
        ped_h, ped_a = res.get('ped_h', 1.0), res.get('ped_a', 1.0)
        val_h, val_a = res.get('val_h', 1.0), res.get('val_a', 1.0)
        
        script = ""
        # UCL Heritage logic
        if ped_h > 1.05 and ped_a <= 1.0:
            heritage_context = f"The immense European pedigree of {res['home']} will be a decisive factor tonight. History tends to repeat itself on these nights."
        elif ped_a > 1.05 and ped_h <= 1.0:
            heritage_context = f"{res['away']} brings their 'Champions League DNA' to this fixture, which often overcomes domestic form disparities."
        elif ped_h > 1.05 and ped_a > 1.05:
            heritage_context = "Two European giants collide. This is a clash of legacies where the weight of history is balanced between both sides."
        else: heritage_context = ""

        # Score & Flow logic
        if res['l_home'] > 2.0 and res['l_away'] > 1.5:
            flow = "A high-octane encounter expected. Both sides will trade blows early, with clinical finishing deciding the final 20 minutes."
        elif res['l_home'] > res['l_away'] + 1.0:
            flow = f"Total dominance predicted for {res['home']}. They will likely monopolize possession and force {res['away']} into a deep block."
        elif abs(res['l_home'] - res['l_away']) < 0.3:
            flow = "A tactical stalemate is brewing. Expect a midfield battle where the first goal will likely be the knockout blow."
        else:
            flow = f"A controlled tactical performance likely. {res['home']} holds the advantage, but {res['away']}'s transition threat is significant."
            
        final_script = f"{heritage_context} {flow}".strip()
        st.markdown(f'<div class="script-box">"{final_script}"</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_match_results(res, match_date, live_odds=None, df=None):
    """Orchestrates the rendering of the advanced analytical bundle."""
    st.markdown("---")
    recs = MatchPredictor().get_recommendations(res, live_odds)
    
    render_match_header(res, match_date)
    st.markdown(f"""
    <div class="predicted-score-box">
        <div style="font-size: 0.8rem; text-transform: uppercase; color: #10b981; letter-spacing: 0.1em; font-weight: 800;">Dixon-Coles Score</div>
        <div style="font-size: 2.3rem; font-weight: 800; color: var(--app-text);">{res['predicted_score']}</div>
    </div>
    """, unsafe_allow_html=True)

    render_outcome_bar(res)
    render_top_picks(recs)
    render_analytics(res)
    
    # New Advanced Section
    if df is not None:
        render_advanced_intelligence(res, df)

# ==============================================================================
# UI COMPONENTS (LANDING & DISCOVERY)
# ==============================================================================
def render_league_selector():
    """Renders the high-tech immersive league selection grid."""
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 40px;">
        <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 10px;">MATCH INTELLIGENCE</h1>
        <p style="font-size: 1.1rem; color: #94a3b8; letter-spacing: 0.2em;">SELECT YOUR THEATER OF OPERATION</p>
    </div>
    """, unsafe_allow_html=True)
    
    leagues = [
        ("Champions League", "🏆", "Elite European Nights"),
        ("Premier League", "🦁", "World's Best League"),
        ("La Liga", "🇪🇸", "The Home of Technicians"),
        ("Serie A", "🇮🇹", "Tactical Masterclass"),
        ("Bundesliga", "🇩🇪", "Pure Intensity"),
        ("Ligue 1", "🇫🇷", "The Talent Factory"),
        ("Russian Premier League", "🐻", "Northern Resistance")
    ]
    
    cols = st.columns(3)
    for i, (name, icon, tag) in enumerate(leagues):
        with cols[i % 3]:
            if st.button(name, key=f"btn_{name}", use_container_width=True):
                st.session_state.selected_league = name
                st.rerun()
            
            # Use HTML overlay for the gamified look while button handles logic
            st.markdown(f"""
            <div class="league-card" style="pointer-events: none; margin-top: -45px; margin-bottom: 25px;">
                <span class="league-card-icon">{icon}</span>
                <div class="league-card-name">{name}</div>
                <div class="league-card-tag">{tag}</div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# MAIN APP FLOW
# ==============================================================================
def main():
    inject_css()
    
    # State Management for Navigation
    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = None
    
    if not st.session_state.selected_league:
        render_league_selector()
        return

    # Sidebar (Simplified after selection)
    st.sidebar.markdown(f"### 🏟️ {st.session_state.selected_league}")
    if st.sidebar.button("◀ Return to Selector", use_container_width=True):
        st.session_state.selected_league = None
        st.rerun()
        
    st.sidebar.divider()
    season = st.sidebar.selectbox("Season", ["2025", "2024", "2023"], index=0)
    selected_league = st.session_state.selected_league
    
    # Rest of the App Logic (Same as before but using selected_league from state)
    st.title(f"⚽ {selected_league}")
    
    is_ucl = selected_league == "Champions League"
    df = pd.DataFrame()
    
    if is_ucl:
        with st.spinner("Compiling global dataset for UCL..."):
            master_dfs = []
            for l_name, l_code in LEAGUES_UNDERSTAT.items():
                if l_code != "UCL":
                    d_df, _ = DataService.fetch_league_data(l_code, season)
                    master_dfs.append(d_df)
            df = pd.concat(master_dfs) if master_dfs else pd.DataFrame()
            upcoming = DataService.fetch_ucl_fixtures(ODDS_API_KEY, LEAGUES_ODDS_API[selected_league])
    else:
        with st.spinner(f"Initiating {selected_league} Stream..."):
            df, _ = DataService.fetch_league_data(LEAGUES_UNDERSTAT[selected_league], season)
            upcoming = df[df['xG'].isna()].copy()
        
    if df.empty and not is_ucl:
        st.error("Protocol Error: Could not fetch data stream."); st.stop()

    if not upcoming.empty:
        upcoming['DateTime'] = pd.to_datetime(upcoming['DateTime'])
        upcoming = upcoming.sort_values('DateTime')
        
        options = [f"{r['Home']} vs {r['Away']} ({r['DateTime'].strftime('%Y-%m-%d %H:%M')})" for _, r in upcoming.iterrows()]
        selection = st.selectbox("📅 Choose Active Target", ["Select a Match..."] + options)
        
        if selection != "Select a Match...":
            try:
                match_str = selection.split(" (")[0]; home, away = match_str.split(" vs ")
                home_norm = DataService.normalize_team_name(home)
                away_norm = DataService.normalize_team_name(away)
                live_odds = DataService.fetch_live_odds(ODDS_API_KEY, LEAGUES_ODDS_API[selected_league], home, away)
                
                predictor = MatchPredictor()
                res = predictor.predict_match(home_norm, away_norm, df)
                match_date = upcoming[(upcoming['Home'] == home) & (upcoming['Away'] == away)].iloc[0]['DateTime']
                
                render_match_results(res, match_date, live_odds, df)
            except Exception as e:
                st.error(f"Analysis Failure: {e}")

    st.markdown("<div style='height: 500px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
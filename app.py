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
    "Premier League": "EPL",
    "La Liga": "La_Liga",
    "Serie A": "Serie_A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue_1",
    "Russian Premier League": "RFPL"
}

LEAGUES_ODDS_API = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Russian Premier League": "soccer_russia_premier_league" 
}

# Advanced Glassmorphism CSS
APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    * { font-family: 'Outfit', sans-serif; }
    
    .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    
    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Hero Card for Top Picks */
    .hero-card {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);
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
        box-shadow: 0 0 10px #10b981;
        margin: 10px 0;
    }
    
    /* Match Outcome Horizontal Bar */
    .outcome-bar {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    .outcome-item { text-align: center; }
    .outcome-value { font-size: 1.1rem; font-weight: 800; color: #f8fafc; }
    
    .badge-circle {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        border-radius: 50%;
        background: rgba(255,255,255,0.1);
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 4px;
    }

    /* Heat Meter */
    .heat-meter-container {
        width: 100%;
        height: 12px;
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .heat-meter-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease-in-out;
    }

    .heat-low { background: #10b981; }
    .heat-mid { background: #f59e0b; }
    .heat-high { background: #ef4444; box-shadow: 0 0 10px #ef4444; }

    /* Typography fixes for Streamlit */
    h1, h2, h3, h4, h5, h6, p, span, div { color: #f8fafc !important; }
    .stCaption { color: #94a3b8 !important; font-style: italic; }
    
    /* Predicted Score Box */
    .predicted-score-box {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin: 15px 0;
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

    def calculate_strength(self, team, df, is_home, avg_home_xg, avg_away_xg):
        """Calculates weighted offensive and defensive strength based on last 8 matches."""
        played = df.dropna(subset=['xG', 'xG.1'])
        team_matches = played[(played['Home'] == team) | (played['Away'] == team)].tail(8)
        
        if team_matches.empty:
            return 1.0, 1.0
            
        atk_xg = []
        def_xg = []
        
        for _, row in team_matches.iterrows():
            if row['Home'] == team:
                atk_xg.append(row['xG'])
                def_xg.append(row['xG.1'])
            else:
                atk_xg.append(row['xG.1'])
                def_xg.append(row['xG'])
                
        weights = np.arange(1, len(atk_xg) + 1)
        team_avg_atk = np.average(atk_xg, weights=weights)
        team_avg_def = np.average(def_xg, weights=weights)
        
        # Strength relative to league
        if is_home:
            atk_strength = team_avg_atk / avg_home_xg
            def_strength = team_avg_def / avg_away_xg
        else:
            atk_strength = team_avg_atk / avg_away_xg
            def_strength = team_avg_def / avg_home_xg
            
        return atk_strength, def_strength

    def predict_match(self, home_team, away_team, df):
        """Runs Poisson simulation to generate all market probabilities."""
        avg_h_xg, avg_a_xg = self.get_league_stats(df)
        
        h_atk, h_def = self.calculate_strength(home_team, df, True, avg_h_xg, avg_a_xg)
        a_atk, a_def = self.calculate_strength(away_team, df, False, avg_h_xg, avg_a_xg)
        
        # Expected Goals = Team Attack * Opponent Defense * League Average
        l_home = h_atk * a_def * avg_h_xg
        l_away = a_atk * h_def * avg_a_xg
        
        # Add Home Advantage boost (slight)
        l_home *= 1.05
        l_away *= 0.95
        
        h_pmf = poisson.pmf(np.arange(10), l_home)
        a_pmf = poisson.pmf(np.arange(10), l_away)
        matrix = np.outer(h_pmf, a_pmf)

        h_win = np.sum(np.tril(matrix, -1))
        draw_prob = np.sum(np.diag(matrix))
        a_win = np.sum(np.triu(matrix, 1))
        
        return {
            "home": home_team, "away": away_team,
            "l_home": l_home, "l_away": l_away,
            "h_win": h_win, "draw": draw_prob, "a_win": a_win,
            "dc_1x": h_win + draw_prob,
            "dc_x2": a_win + draw_prob,
            "dc_12": h_win + a_win,
            "btts": (1 - h_pmf[0]) * (1 - a_pmf[0]),
            "over15": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(2) for j in range(2-i)]),
            "under15": np.sum([h_pmf[i]*a_pmf[j] for i in range(2) for j in range(2-i)]),
            "over25": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)]),
            "under25": np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)]),
            "over35": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(4) for j in range(4-i)]),
            "under35": np.sum([h_pmf[i]*a_pmf[j] for i in range(4) for j in range(4-i)]),
            "h_over15": 1 - h_pmf[0] - h_pmf[1],
            "a_over15": 1 - a_pmf[0] - a_pmf[1],
            "predicted_score": f"{np.unravel_index(matrix.argmax(), matrix.shape)[0]}-{np.unravel_index(matrix.argmax(), matrix.shape)[1]}"
        }

    def get_recommendations(self, res):
        """Applies the logic hierarchy to select primary and secondary picks."""
        h_xg = res['l_home']
        a_xg = res['l_away']
        
        primary_pick = None
        secondary_pick = None
        primary_insight = ""
        secondary_insight = ""

        # Rule 1: BTTS Override
        if a_xg > 2.15 and h_xg > 1.50:
            primary_pick = "BTTS (Yes)"
            primary_insight = "Both teams show elite offensive data today. Expect net-bulging action from both ends."
        
        # Rule 2: Away Optimization
        elif res['a_win'] > res['h_win'] and res['a_win'] > res['draw']:
            if a_xg > 2.15:
                primary_pick = f"{res['away']} Over 1.5 Goals"
                primary_insight = f"{res['away']} attack is firing on all cylinders. They have too much firepower for the home defense."
            else:
                primary_pick = "Away/Draw (X2)"
                primary_insight = f"{res['away']} holds the tactical edge here. This pick provides a vital safety net for a tight game."

        # Rule 3: Home Optimization
        elif res['h_win'] > res['a_win'] and res['h_win'] > res['draw']:
            if h_xg > 2.49:
                primary_pick = f"{res['home']} Over 1.5 Goals"
                primary_insight = f"{res['home']} dominates the xG metrics at home. A multi-goal performance is statistically mapped."
            else:
                primary_pick = "Home/Draw (1X)"
                primary_insight = f"Home advantage and defensive stability favor the hosts. Expect them to avoid defeat in this setup."
                
        # Fallback
        if not primary_pick:
            primary_pick = f"{res['home']} or {res['away']} Win"
            primary_insight = "Both teams are volatile and seek the win. A draw looks unlikely based on current form."

        # Secondary Pick (Safety)
        if "Over" not in primary_pick and res['over15'] > 0.75:
            secondary_pick = "Over 1.5 Goals"
            secondary_insight = "High probability of at least two goals. Both squads' recent trends support a scoring rhythm."

        return {
            "primary_pick": primary_pick,
            "primary_insight": primary_insight,
            "secondary_pick": secondary_pick,
            "secondary_insight": secondary_insight
        }

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
def inject_css():
    """Injects the global CSS theme."""
    st.markdown(APP_CSS, unsafe_allow_html=True)

def render_match_header(res, match_date):
    """Renders the FotMob-inspired match header with team logos."""
    logo_h = DataService.fetch_team_logo(res['home'])
    logo_a = DataService.fetch_team_logo(res['away'])
    img_h = f"<img src='{logo_h}' style='height: 80px;'>" if logo_h else ""
    img_a = f"<img src='{logo_a}' style='height: 80px;'>" if logo_a else ""

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 40px; margin-bottom: 30px;">
        <div style="text-align: center; flex: 1;">
            {img_h}
            <div style="font-weight: 800; font-size: 1.4rem; margin-top: 10px;">{res['home']}</div>
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
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_outcome_bar(res):
    """Renders the de-emphasized H/D/A horizontal bar."""
    st.markdown(f"""
    <div class="outcome-bar">
        <div class="outcome-item">
            <div class="badge-circle" style="color: #60a5fa;">H</div>
            <div class="outcome-value">{res['h_win']:.0%}</div>
        </div>
        <div class="outcome-item">
            <div class="badge-circle" style="color: #94a3b8;">D</div>
            <div class="outcome-value">{res['draw']:.0%}</div>
        </div>
        <div class="outcome-item">
            <div class="badge-circle" style="color: #f87171;">A</div>
            <div class="outcome-value">{res['a_win']:.0%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_top_picks(recs):
    """Renders the High-Value Hero Cards."""
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
        with col2:
            st.markdown(f"""
            <div class="hero-card" style="background: linear-gradient(135deg, #4b5563 0%, #1f2937 100%); box-shadow: none;">
                <div style="font-size: 0.7rem; text-transform: uppercase; opacity: 0.8; font-weight: 800;">Safety Pick</div>
                <div style="font-size: 1.3rem; font-weight: 800; margin: 8px 0;">{recs['secondary_pick']}</div>
                <div class="confidence-glow" style="width: 70%; background: #94a3b8; box-shadow: none;"></div>
                <div style="font-size: 0.85rem; font-style: italic; opacity: 0.9;">"{recs['secondary_insight']}"</div>
            </div>
            """, unsafe_allow_html=True)

def render_analytics(res):
    """Renders the Goals & Analytics section with Heat Meters."""
    h_xg, a_xg = res['l_home'], res['l_away']
    with st.expander("📈 Goals & Analytics (Heat Meters)", expanded=True):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Heat Meter Classes
        h_heat = "heat-high" if h_xg > 2.0 else "heat-mid" if h_xg > 1.2 else "heat-low"
        a_heat = "heat-high" if a_xg > 2.0 else "heat-mid" if a_xg > 1.2 else "heat-low"
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>{res['home']} Attack (xG)</span>
            <span style="font-weight: 800;">{h_xg:.2f}</span>
        </div>
        <div class="heat-meter-container">
            <div class="heat-meter-fill {h_heat}" style="width: {min(h_xg/3*100, 100)}%;"></div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 15px; margin-bottom: 5px;">
            <span>{res['away']} Attack (xG)</span>
            <span style="font-weight: 800;">{a_xg:.2f}</span>
        </div>
        <div class="heat-meter-container">
            <div class="heat-meter-fill {a_heat}" style="width: {min(a_xg/3*100, 100)}%;"></div>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        for label, prob in [("BTTS", res['btts']), ("OVER 2.5", res['over25']), ("UNDER 2.5", res['under25'])]:
            with c1 if label == "BTTS" else c2 if label == "OVER 2.5" else c3:
                st.markdown(f"""
                <div style='text-align:center;'>
                    <div style='font-size:0.7rem; color:#94a3b8;'>{label}</div>
                    <div style='font-size:1.2rem; font-weight:800;'>{prob:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_match_results(res, match_date):
    """Orchestrates the rendering of all match analysis components."""
    st.markdown("---")
    
    # Analyze recommendations
    recs = MatchPredictor().get_recommendations(res)
    
    # 1. Header
    render_match_header(res, match_date)

    # 2. Predicted Score
    st.markdown(f"""
    <div class="predicted-score-box">
        <div style="font-size: 0.8rem; text-transform: uppercase; color: #10b981; letter-spacing: 0.1em; font-weight: 800;">Predicted Score</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #f8fafc;">{res['predicted_score']}</div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Probability Bar
    render_outcome_bar(res)

    # 4. Hero Picks
    render_top_picks(recs)

    # 5. Goal Analytics
    render_analytics(res)

# ==============================================================================
# MAIN APP FLOW
# ==============================================================================
def main():
    """Main entry point for the Streamlit application."""
    inject_css()
    st.title("⚽ Football Prediction Engine")
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    season = st.sidebar.selectbox("Season", ["2025", "2024", "2023"], index=0)
    selected_league = st.sidebar.selectbox("Select League", list(LEAGUES_UNDERSTAT.keys()))
    
    # Data Retrieval
    with st.spinner(f"Fetching {selected_league}..."):
        df, teams = DataService.fetch_league_data(LEAGUES_UNDERSTAT[selected_league], season)
        
    if df.empty:
        st.error("Could not fetch data.")
        st.stop()

    # Match Selection Filter
    upcoming = df[df['xG'].isna()].copy()
    if not upcoming.empty:
        upcoming['DateTime'] = pd.to_datetime(upcoming['DateTime'])
        upcoming = upcoming.sort_values('DateTime')
        
        match_options = [
            f"{r['Home']} vs {r['Away']} ({r['DateTime'].strftime('%Y-%m-%d %H:%M')})" 
            for _, r in upcoming.iterrows()
        ]
        selection = st.selectbox("📅 Choose Upcoming Match", ["Select a Match..."] + match_options)
        
        if selection != "Select a Match...":
            try:
                # Analysis Pipeline
                match_str = selection.split(" (")[0]
                home, away = match_str.split(" vs ")
                
                predictor = MatchPredictor()
                res = predictor.predict_match(home, away, df)
                match_date = upcoming[(upcoming['Home'] == home) & (upcoming['Away'] == away)].iloc[0]['DateTime']
                
                # Render results
                render_match_results(res, match_date)
                
            except Exception as e:
                st.error(f"Analysis Error: {e}")

    # Layout Spacing
    st.markdown("<div style='height: 500px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
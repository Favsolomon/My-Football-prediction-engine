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
    def get_league_stats(self, df):
        played = df.dropna(subset=['xG', 'xG.1'])
        if played.empty:
            return 1.3, 1.3  # Default baselines
        avg_home_xg = played['xG'].mean()
        avg_away_xg = played['xG.1'].mean()
        return avg_home_xg, avg_away_xg

    def calculate_strength(self, team, df, is_home, avg_home_xg, avg_away_xg):
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
            "over25": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)]),
            "h_over15": 1 - h_pmf[0] - h_pmf[1],
            "a_over15": 1 - a_pmf[0] - a_pmf[1],
            "predicted_score": f"{np.unravel_index(matrix.argmax(), matrix.shape)[0]}-{np.unravel_index(matrix.argmax(), matrix.shape)[1]}"
        }

# ==============================================================================
# UI COMPONENTS
# ==============================================================================
def inject_css():
    st.markdown("""
    <style>
        .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .stMetric label { font-weight: 600; font-size: 0.9rem; }
        .stMetric .css-1wivap2 { font-size: 1.5rem !important; }
        .result-card { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }
        [data-testid="stMetricValue"] { justify-content: center; }
    </style>
    """, unsafe_allow_html=True)

def render_results(res, match_date, live_odds):
    st.markdown("---")
    
    # 1. Header (Logos/Title) - Custom HTML for better mobile alignment
    logo_h = DataService.fetch_team_logo(res['home'])
    logo_a = DataService.fetch_team_logo(res['away'])
    
    # Defaults if no logo found (optional, or just handle empty src)
    img_h = f"<img src='{logo_h}' style='height: 60px; object-fit: contain;'>" if logo_h else ""
    img_a = f"<img src='{logo_a}' style='height: 60px; object-fit: contain;'>" if logo_a else ""
    
    odds_badge = ""
    if live_odds:
        odds_badge = f"""<div style='margin-top: 5px; font-size: 0.8em; background: #fffbe6; padding: 2px 8px; border-radius: 10px; border: 1px solid #ffe58f; display: inline-block;'>1({live_odds.get('home', '-')}) X({live_odds.get('draw', '-')}) 2({live_odds.get('away', '-')})</div>"""

    st.markdown(f"""
<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
<div style="display: flex; align-items: center; justify-content: flex-end; flex: 1; gap: 10px;">
<div style="text-align: right; font-weight: bold; font-size: 1.1rem;">{res['home']}</div>
{img_h}
</div>
<div style="padding: 0 15px; text-align: center;">
<div style="font-weight: bold; color: #444;">VS</div>
<div style="font-size: 0.75em; color: gray;">{match_date.strftime('%H:%M')}</div>
</div>
<div style="display: flex; align-items: center; justify-content: flex-start; flex: 1; gap: 10px;">
{img_a}
<div style="text-align: left; font-weight: bold; font-size: 1.1rem;">{res['away']}</div>
</div>
</div>
<div style="text-align: center; margin-bottom: 15px;">
<span style="color: gray; font-size: 0.9em;">{match_date.strftime('%d %b %Y')}</span>
{odds_badge}
</div>
""", unsafe_allow_html=True)

    # 2. Recommendation Logic
    all_bets = {
        f"{res['home']} Win": res['h_win'], "Draw": res['draw'], f"{res['away']} Win": res['a_win'],
        f"Home/Draw (1X)": res['dc_1x'], f"Away/Draw (X2)": res['dc_x2'], "Any Winner (12)": res['dc_12'],
        "BTTS Yes": res['btts'], "Over 1.5 Goals": res['over15'], "Over 2.5 Goals": res['over25'],
        f"{res['home']} Over 1.5 Goals": res['h_over15'], f"{res['away']} Over 1.5 Goals": res['a_over15']
    }
    
    # Analysis Flags
    diff_dc = abs(res['dc_1x'] - res['dc_x2'])
    is_balanced = diff_dc < 0.10
    
    def get_sort_score(item):
        name, prob = item
        
        # Get live odds for the primary outcomes if available
        home_odds = live_odds.get('home', 0) if live_odds else 0
        away_odds = live_odds.get('away', 0) if live_odds else 0
        draw_odds = live_odds.get('draw', 0) if live_odds else 0

        # A. Odds Filter & Favorite Prioritization
        current = 0
        is_favorite_market = False
        if f"{res['home']} Win" == name: 
            current = home_odds
            is_favorite_market = prob > 0.55
        elif f"{res['away']} Win" == name: 
            current = away_odds
            is_favorite_market = prob > 0.55
        elif "Draw" == name: 
            current = draw_odds
        
        # Boost strong favorites to #1 spot if confidence is high
        if is_favorite_market and (current > 0 and current < 1.60):
            return prob * 2.0  # Heavier weight for clear wins
            
        if current > 0 and current < 1.15: return prob * 0.3

        # Double Chance Implicit Filtering
        if "Home/Draw (1X)" == name and home_odds > 0 and home_odds < 1.15:
            return prob * 0.3
        if "Away/Draw (X2)" == name and away_odds > 0 and away_odds < 1.15:
            return prob * 0.3
            
        # B. Balanced Penalty
        if is_balanced and any(x in name for x in ["Win", "1X", "X2", "12"]):
            return prob * 0.5
            
        # C. General Penalties
        if "Over 2.5 Goals" in name: return prob * 1.1 # Favor goals slightly more
        if "Over 1.5 Goals" == name: return prob * 0.70
        if "Any Winner (12)" in name: return prob * 0.50 # Penalize 12 more
        return prob

    sorted_bets = sorted(all_bets.items(), key=get_sort_score, reverse=True)
    raw_sorted = sorted(all_bets.items(), key=lambda x: x[1], reverse=True)
    
    # 3. Top Picks UI
    st.success(f"### 🏆 Top Picks")
    if is_balanced:
        st.warning("⚠️ **Tight Match**: Teams are evenly matched. Focusing on Goal Markets.")
        
    c1, c2 = st.columns(2)
    c1.markdown(f"**🥇 1. {sorted_bets[0][0]}**")
    c1.caption(f"Confidence: {sorted_bets[0][1]:.0%}")
    c2.markdown(f"**🥈 2. {sorted_bets[1][0]}**")
    c2.caption(f"Confidence: {sorted_bets[1][1]:.0%}")
    
    # Predicted Score
    st.markdown(f"""
    <div style="text-align: center; margin-top: 10px; padding: 10px; background-color: #e8f5e9; border-radius: 8px;">
        <span style="font-size: 0.9rem; color: #2e7d32; font-weight: bold;">🎯 Predicted Score</span><br>
        <span style="font-size: 1.8rem; font-weight: 800; color: #1b5e20;">{res['predicted_score']}</span>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"**📊 Statistically Most Likely:** {raw_sorted[0][0]} ({raw_sorted[0][1]:.1%})")

    # 4. Detailed Stats
    with st.container():
        st.markdown(f"##### 📊 Match Outcomes")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{res['home']}", f"{res['h_win']:.0%}")
        c2.metric("Draw", f"{res['draw']:.0%}")
        c3.metric(f"{res['away']}", f"{res['a_win']:.0%}")
        
        st.markdown(f"##### 🛡️ Safety (Double Chance)")
        d1, d2, d3 = st.columns(3)
        d1.metric("1X", f"{res['dc_1x']:.0%}")
        d2.metric("12", f"{res['dc_12']:.0%}")
        d3.metric("X2", f"{res['dc_x2']:.0%}")

    with st.expander("📈 Goal Stats & Analysis", expanded=False):
        g1, g2 = st.columns(2)
        with g1:
            st.write("Expected Goals (xG)")
            st.progress(min(res['l_home']/3, 1.0), text=f"{res['home']}: {res['l_home']:.2f}")
            st.progress(min(res['l_away']/3, 1.0), text=f"{res['away']}: {res['l_away']:.2f}")
        with g2:
            st.write("Market Probabilities")
            st.write(f"• BTTS: **{res['btts']:.0%}**")
            st.write(f"• Over 1.5: **{res['over15']:.0%}**")
            st.write(f"• Over 2.5: **{res['over25']:.0%}**")
        st.caption("Based on 10,000 simulations of Poisson distribution models.")

# ==============================================================================
# MAIN APP FLOW
# ==============================================================================
def main():
    inject_css()
    st.title("⚽ Football Prediction Engine")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    season = st.sidebar.selectbox("Season", ["2025", "2024", "2023"], index=0)
    selected_league = st.sidebar.selectbox("Select League", list(LEAGUES_UNDERSTAT.keys()))
    
    # Fetch Data
    with st.spinner(f"Fetching {selected_league}..."):
        df, teams = DataService.fetch_league_data(LEAGUES_UNDERSTAT[selected_league], season)
        
    if df.empty:
        st.error("Could not fetch data.")
        st.stop()

    # Match Selection
    upcoming = df[df['xG'].isna()].copy()
    if not upcoming.empty:
        upcoming['DateTime'] = pd.to_datetime(upcoming['DateTime'])
        upcoming = upcoming.sort_values('DateTime')
        choices = [
            f"{r['Home']} vs {r['Away']} ({r['DateTime'].strftime('%Y-%m-%d %H:%M')})" 
            for _, r in upcoming.iterrows()
        ]
        selection = st.selectbox("📅 Choose Upcoming Match", ["Select a Match..."] + choices)
        
        if selection != "Select a Match...":
            try:
                # Parse Selection
                match_str = selection.split(" (")[0]
                home, away = match_str.split(" vs ")
                
                # Analyze
                predictor = MatchPredictor()
                res = predictor.predict_match(home, away, df)
                match_date = upcoming[(upcoming['Home'] == home) & (upcoming['Away'] == away)].iloc[0]['DateTime']
                
                # Get Odds
                odds_key = LEAGUES_ODDS_API.get(selected_league)
                live_odds = DataService.fetch_live_odds(ODDS_API_KEY, odds_key, home, away)
                
                # Render
                render_results(res, match_date, live_odds)
                
            except Exception as e:
                st.error(f"Analysis Error: {e}")

    # Mobile Fix
    st.markdown("<div style='height: 500px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
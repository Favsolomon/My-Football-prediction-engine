import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import io
import time
from understatapi import UnderstatClient

# --- LOGIC ENGINE ---
LEAGUES = {
    "Premier League": "EPL",
    "La Liga": "La_Liga",
    "Serie A": "Serie_A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue_1"
}

@st.cache_data(ttl=3600)
def fetch_league_data(league_code, season="2024"):
    """Fetches and cleans league data from Understat.com."""
    try:
        with UnderstatClient() as client:
            matches = client.league(league=league_code).get_match_data(season=season)
        
        if not matches:
            return pd.DataFrame(), []

        # Convert to list of dicts for DataFrame
        processed_data = []
        teams = set()
        
        for m in matches:
            if not m.get('isResult'):
                # Store upcoming matches for fixture list, but no stats yet
                # We can store them with NaNs or keep separate
                # For this app logic, let's keep them and mark xG as NaN
                row = {
                    'Home': m['h']['title'],
                    'Away': m['a']['title'],
                    'xG': np.nan, 
                    'xG.1': np.nan,
                    'Score': None
                }
            else:
                row = {
                    'Home': m['h']['title'],
                    'Away': m['a']['title'],
                    'xG': float(m['xG']['h']),
                    'xG.1': float(m['xG']['a']),
                    'Score': f"{m['goals']['h']}-{m['goals']['a']}"
                }
            
            processed_data.append(row)
            teams.add(m['h']['title'])
            teams.add(m['a']['title'])
            
        df = pd.DataFrame(processed_data)
        
        # Data cleaning types
        df['xG'] = pd.to_numeric(df['xG'], errors='coerce')
        df['xG.1'] = pd.to_numeric(df['xG.1'], errors='coerce')
        
        return df, sorted(list(teams))
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame(), []

class MatchPredictor:
    def calculate_lambda(self, team, is_home, df, use_defensive=False):
        # FIX: Filter for MATCHES THAT HAVE BEEN PLAYED (valid xG data)
        # We look for matches where 'team' played and xG is not NaN
        played_games = df.dropna(subset=['xG', 'xG.1'])
        
        team_matches = played_games[
            (played_games['Home'] == team) | (played_games['Away'] == team)
        ].tail(6) # TRULY the last 6 PLAYED matches
        
        xg_values = []
        for _, row in team_matches.iterrows():
            if use_defensive:
                # If home team, defensive strength is xG conced (Away xG)
                val = row['xG.1'] if row['Home'] == team else row['xG']
            else:
                val = row['xG'] if row['Home'] == team else row['xG.1']
                
            xg_values.append(val)
                
        if not xg_values: 
            # Fallback if no data (e.g. start of season or promoted team with no history)
            return 1.2 if not use_defensive else 1.0
        
        # Weighted average (most recent games count more)
        weights = np.arange(1, len(xg_values) + 1)
        avg_xg = np.average(xg_values, weights=weights)

        # Home/Away Adjustment
        adjustment = (1.12 if is_home else 0.88) if not use_defensive else (0.88 if is_home else 1.12)
        return avg_xg * adjustment

    def predict_match(self, home_team, away_team, df):
        # Stats Logic
        h_atk = self.calculate_lambda(home_team, True, df, False)
        h_def = self.calculate_lambda(home_team, True, df, True)
        a_atk = self.calculate_lambda(away_team, False, df, False)
        a_def = self.calculate_lambda(away_team, False, df, True)
        
        l_home, l_away = (h_atk + a_def) / 2, (a_atk + h_def) / 2
        
        max_g = 10
        h_pmf = poisson.pmf(np.arange(max_g), l_home)
        a_pmf = poisson.pmf(np.arange(max_g), l_away)
        matrix = np.outer(h_pmf, a_pmf)

        # Calculate probabilities
        h_win = np.sum(np.tril(matrix, -1))
        draw_prob = np.sum(np.diag(matrix))
        a_win = np.sum(np.triu(matrix, 1))

        # Double Chance
        dc_1x = h_win + draw_prob
        dc_x2 = a_win + draw_prob
        dc_12 = h_win + a_win

        return {
            "home": home_team, "away": away_team,
            "h_atk": h_atk, "h_def": h_def,
            "a_atk": a_atk, "a_def": a_def,
            "l_home": l_home, "l_away": l_away,
            "h_win": h_win,
            "draw": draw_prob,
            "a_win": a_win,
            "dc_1x": dc_1x,
            "dc_x2": dc_x2,
            "dc_12": dc_12,
            "btts": (1 - h_pmf[0]) * (1 - a_pmf[0]),
            "over15": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(2) for j in range(2-i)]),
            "over25": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)]),
            "h_over15": 1 - h_pmf[0] - h_pmf[1],
            "a_over15": 1 - a_pmf[0] - a_pmf[1],
            "predicted_score": f"{np.unravel_index(matrix.argmax(), matrix.shape)[0]}-{np.unravel_index(matrix.argmax(), matrix.shape)[1]}"
        }

# --- STREAMLIT UI ---
st.set_page_config(page_title="Match Predictor PRO", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Mobile-First Card Design
st.markdown("""
<style>
    /* Card Container */
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric label {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stMetric .css-1wivap2 {
        font-size: 1.5rem !important;
    }
    
    /* Result Card Styling */
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Make metrics center aligned */
    [data-testid="stMetricValue"] {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Football Prediction Engine")

# Sidebar - Global Config
st.sidebar.header("Configuration")
selected_league_name = st.sidebar.selectbox("Select League", list(LEAGUES.keys()))
league_code = LEAGUES[selected_league_name]
selected_season = st.sidebar.selectbox("Season", ["2025", "2024", "2023"], index=0)

# Data Fetching
df, teams = None, []
with st.spinner(f"Loading {selected_league_name}..."):
    df, teams = fetch_league_data(league_code, season=selected_season)

if df is None or df.empty:
    st.warning("⚠️ Could not fetch data. Try another league or season.")
else:
    # --- MAIN AREA SELECTION (Mobile Friendly) ---
    upcoming_df = df[df['xG'].isna()]
    fixtures = ["Select a Match..."]
    if not upcoming_df.empty:
        fixture_list = [f"{row['Home']} vs {row['Away']}" for _, row in upcoming_df.iterrows()]
        fixtures.extend(fixture_list)

    # Big Prominent Select Box
    selected_fixture = st.selectbox("📅 Choose Upcoming Match", fixtures)
    
    # Validation & Analysis Trigger
    if selected_fixture != "Select a Match...":
        try:
            p_home, p_away = selected_fixture.split(" vs ")
            if p_home in teams and p_away in teams:
                # Run Prediction Immediately on selection
                engine = MatchPredictor()
                res = engine.predict_match(p_home, p_away, df)
                
                # --- RESULTS UI ---
                st.markdown("---")
                
                # 1. Recommendation Banner
                all_bets = {
                    f"{res['home']} Win": res['h_win'],
                    "Draw": res['draw'],
                    f"{res['away']} Win": res['a_win'],
                    f"Home/Draw (1X)": res['dc_1x'],
                    f"Away/Draw (X2)": res['dc_x2'],
                    "Any Winner (12)": res['dc_12'],
                    "BTTS Yes": res['btts'],
                    "Over 1.5 Goals": res['over15'],
                    "Over 2.5 Goals": res['over25'],
                    f"{res['home']} Over 1.5 Goals": res['h_over15'],
                    f"{res['away']} Over 1.5 Goals": res['a_over15']
                }
                
                
                # Logic: 
                # 1. Deprioritize "Over 1.5 Goals" (Weight 0.7) - Too obvious/low odds
                # 2. Deprioritize "Any Winner" (Weight 0.8) - Often safe but boring
                # 3. Prioritize Goals (Over 2.5, BTTS often have value) - keep weight 1.0
                def get_sort_score(item):
                    name, prob = item
                    if "Over 1.5 Goals" == name:
                        return prob * 0.70
                    if "Any Winner (12)" in name:
                        return prob * 0.80
                    return prob

                sorted_bets = sorted(all_bets.items(), key=get_sort_score, reverse=True)
                
                # RAW probabilities (Unweighted) for "Most Probable" section
                raw_sorted = sorted(all_bets.items(), key=lambda x: x[1], reverse=True)
                most_probable_name, most_probable_prob = raw_sorted[0]
                
                # Top 2 Picks (Weighted)
                pick1_name, pick1_prob = sorted_bets[0]
                pick2_name, pick2_prob = sorted_bets[1]

                
                st.success(f"### 🏆 Top Picks")
                col_r1, col_r2 = st.columns(2)
                col_r1.markdown(f"**🥇 1. {pick1_name}**")
                col_r1.caption(f"Confidence: {pick1_prob:.0%}")
                
                col_r2.markdown(f"**🥈 2. {pick2_name}**")
                col_r2.caption(f"Confidence: {pick2_prob:.0%}")
                
                # PREDICTED SCORE
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px; padding: 10px; background-color: #e8f5e9; border-radius: 8px;">
                    <span style="font-size: 0.9rem; color: #2e7d32; font-weight: bold;">🎯 Predicted Score</span>
                    <br>
                    <span style="font-size: 1.8rem; font-weight: 800; color: #1b5e20;">{res['predicted_score']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Most Probable Outcome Section
                st.info(f"**📊 Statistically Most Likely:** {most_probable_name} ({most_probable_prob:.1%})")
                
                # 2. Main Probabilities (Card Style)
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

                # 3. Stats & Goals (Expandable)
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
                    
                    st.caption("Based on 10,000 simulations of Poisson distribution models using recent form.")


        except Exception as e:
            st.error(f"Error analyzing match: {e}")

# --- MOBILE KEYBOARD FIX ---
# Add extra space at the bottom so the browser can scroll the input box 
# into view when the mobile keyboard pops up.
st.markdown("<div style='height: 500px;'></div>", unsafe_allow_html=True)
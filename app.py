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
    "La Liga": "La_liga",
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
            "over25": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)])
        }

# --- STREAMLIT UI ---
st.set_page_config(page_title="Match Predictor PRO", layout="wide")

st.title("⚽ Fav Football Prediction Engine v4.1 (Smart Bet)")

# sidebar
st.sidebar.header("Match Setup")

# 1. League Selection
selected_league_name = st.sidebar.selectbox("Select League", list(LEAGUES.keys()))
league_code = LEAGUES[selected_league_name]

# 2. Match Details
st.sidebar.subheader("Configuration")

# Season Selection
selected_season = st.sidebar.selectbox("Season", ["2025", "2024", "2023"], index=0)

# Data Fetching (Cached or Manual)
df, teams = None, []

# Try fetching live
with st.spinner(f"Fetching {selected_league_name} data ({selected_season}) from Understat.com..."):
    df, teams = fetch_league_data(league_code, season=selected_season)

# Fallback UI if fetch fails
if df is None or df.empty:
    st.warning("⚠️ Could not fetch updated data. Switching to Offline Mode.")
    
    if st.button("Load Demo Data (Example)"):
        # Create dummy data for demo
        data = {
            'Home': ['Arsenal', 'Liverpool', 'Man City', 'Aston Villa'] * 3,
            'Away': ['Man City', 'Aston Villa', 'Arsenal', 'Liverpool'] * 3,
            'xG': [1.5, 2.1, 1.8, 1.2] * 3,
            'xG.1': [1.1, 0.9, 1.4, 1.3] * 3,
            'Score': ['1-1', '2-0', '1-2', '1-1'] * 3
        }
        df = pd.DataFrame(data)
        teams = sorted(list(set(df['Home'].unique().tolist() + df['Away'].unique().tolist())))
        st.success("Demo data loaded!")

if df is not None and not df.empty:
    # Identify Upcoming Matches (where Score/xG is NaN for actual data, or just create fake ones)
    upcoming_df = df[df['xG'].isna()]
    
    # Create list of upcoming fixtures
    fixtures = ["Custom Match"]
    if not upcoming_df.empty:
        # Sort by index (date order roughly)
        fixture_list = [f"{row['Home']} vs {row['Away']}" for _, row in upcoming_df.iterrows()]
        fixtures.extend(fixture_list)

    # 3. Fixture Selection
    selected_fixture = st.sidebar.selectbox("Select Upcoming Match", fixtures)
    
    # Determine default indices based on selection
    idx_home = 0
    idx_away = 1 if len(teams) > 1 else 0
    
    if selected_fixture != "Custom Match":
        try:
            p_home, p_away = selected_fixture.split(" vs ")
            if p_home in teams and p_away in teams:
                idx_home = teams.index(p_home)
                idx_away = teams.index(p_away)
        except:
            pass
    
    # 4. Team Selection
    home_team = st.sidebar.selectbox("Home Team", teams, index=idx_home)
    away_team = st.sidebar.selectbox("Away Team", teams, index=idx_away)

    if st.sidebar.button("Analyze Match", type="primary"):
        if home_team == away_team:
            st.error("Please select two different teams!")
        else:
            engine = MatchPredictor()
            res = engine.predict_match(home_team, away_team, df)
            
            # --- Results Display ---
            st.subheader(f"📊 {res['home']} vs {res['away']} ({selected_league_name})")
            
            # Row 1: Main Outcomes
            col1, col2, col3 = st.columns(3)
            col1.metric("Home Win", f"{res['h_win']:.1%}")
            col2.metric("Draw", f"{res['draw']:.1%}")
            col3.metric("Away Win", f"{res['a_win']:.1%}")

            # Row 2: Double Chance
            c1, c2, c3 = st.columns(3)
            c1.metric("1X (Home/Draw)", f"{res['dc_1x']:.1%}", help="Win if Home wins or Draw")
            c2.metric("12 (Any Winner)", f"{res['dc_12']:.1%}", help="Win if either team wins")
            c3.metric("X2 (Away/Draw)", f"{res['dc_x2']:.1%}", help="Win if Away wins or Draw")
            
            st.divider()
            
            # Goals Analysis
            g1, g2 = st.columns(2)
            with g1:
                st.write("**Expected Goals (Model)**")
                st.progress(min(res['l_home']/3, 1.0), text=f"{res['home']}: {res['l_home']:.2f}")
                st.progress(min(res['l_away']/3, 1.0), text=f"{res['away']}: {res['l_away']:.2f}")
            
            with g2:
                st.write("**Market Insights**")
                st.write(f"✅ **BTTS Yes:** {res['btts']:.1%}")
                st.write(f"📈 **Over 1.5 Goals:** {res['over15']:.1%}")
                st.write(f"🔥 **Over 2.5 Goals:** {res['over25']:.1%}")

            # Breakdown Section
            with st.expander("🕵️ See How Calculation Works"):
                st.markdown(f"""
                ### 1. Team Strength (Last 6 Games)
                The model calculated the following **Attack** and **Defense** ratings based on weighted average xG:
                
                *   **{res['home']} Attack:** {res['h_atk']:.2f}
                *   **{res['home']} Defense:** {res['h_def']:.2f}
                *   **{res['away']} Attack:** {res['a_atk']:.2f}
                *   **{res['away']} Defense:** {res['a_def']:.2f}
                
                ### 2. Math Formula (Poisson)
                We combine these to find the **Total Expected Goals** for this match:
                
                *   **{res['home']} Expected Goals** = ({res['home']} Attack + {res['away']} Defense) / 2 = **{res['l_home']:.2f}**
                *   **{res['away']} Expected Goals** = ({res['away']} Attack + {res['home']} Defense) / 2 = **{res['l_away']:.2f}**
                
                We then simulate the match to get all probabilities.
                """)

            # Verdict Logic - SMART RECOMMENDER
            # Create a dictionary of all possible bets with their probabilities
            all_bets = {
                f"{res['home']} Win": res['h_win'],
                "Draw": res['draw'],
                f"{res['away']} Win": res['a_win'],
                f"{res['home']} or Draw (1X)": res['dc_1x'],
                f"{res['away']} or Draw (X2)": res['dc_x2'],
                "Any Winner (12)": res['dc_12'],
                "BTTS Yes": res['btts'],
                "BTTS No": 1 - res['btts'],
                "Over 1.5 Goals": res['over15'],
                "Under 1.5 Goals": 1 - res['over15'],
                "Over 2.5 Goals": res['over25'],
                "Under 2.5 Goals": 1 - res['over25']
            }
            
            # Sort bets by probability (highest first)
            sorted_bets = sorted(all_bets.items(), key=lambda item: item[1], reverse=True)
            
            # Get the #1 safest bet
            best_bet_name, best_bet_prob = sorted_bets[0]
            
            # Find a "Value Bet" (highest probability that isn't a double chance or extremely low odds < 1.10)
            # Simple logic: First outcome > 55% that isn't Over 0.5 or 1.5 if they are too obvious?
            # Let's just pick the safest one for now as requested.
            
            st.success(f"**🏆 Model Recommendation:** The safest statistical pick is **{best_bet_name}** ({best_bet_prob:.1%})")
            
            # Show top 3 probabilities
            st.write("**Top 3 Probabilities:**")
            for name, prob in sorted_bets[:3]:
                st.write(f"- {name}: **{prob:.1%}**")
else:
    if df is not None: # Means df is empty but not None
         st.error("Data loaded but empty. Check the source.")
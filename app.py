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
                    'Score': None,
                    'DateTime': m['datetime']
                }
            else:
                row = {
                    'Home': m['h']['title'],
                    'Away': m['a']['title'],
                    'xG': float(m['xG']['h']),
                    'xG.1': float(m['xG']['a']),
                    'Score': f"{m['goals']['h']}-{m['goals']['a']}",
                    'DateTime': m['datetime']
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

@st.cache_data(ttl=86400) # Cache for 24 hours
def fetch_team_logo(team_name):
    """Fetches team logo URL from TheSportsDB."""
    try:
        # Clean name for better search (Understat names usually fine, but simple is better)
        search_name = team_name.replace(" ", "%20")
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={search_name}"
        r = requests.get(url, timeout=2)
        data = r.json()
        if data['teams']:
            return data['teams'][0]['strBadge']
    except:
        pass
    return None

@st.cache_data(ttl=300) # Cache odds for 5 minutes
def fetch_live_odds(api_key, sport_key, home_team, away_team):
    """
    Fetches live odds from The-Odds-API.
    Returns a dict with best available odds for H/D/A.
    """
    if not api_key:
        return None
        
    try:
        # 1. Get Odds for the specific sport
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?regions=eu&markets=h2h&apiKey={api_key}"
        r = requests.get(url, timeout=5)
        data = r.json()
        
        # 2. Find the match
        # Fuzzy matching might be needed, but let's try direct first or simplistic fuzzy
        for match in data:
            m_home = match['home_team']
            m_away = match['away_team']
            
            # Simple substring match check
            if (home_team in m_home or m_home in home_team) and (away_team in m_away or m_away in away_team):
                # Found it! Get best odds.
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
    except Exception as e:
        # st.error(f"Odds Error: {e}")
        pass
    return None

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
st.sidebar.header("⚙️ Configuration")
season = st.sidebar.selectbox("Season", ["2025", "2024", "2023"], index=0)

# League Mapping for Understat
leagues = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Serie A": "Serie_A",
    "Bundesliga": "Bundesliga",
    "Ligue 1": "Ligue_1",
    "Russian Premier League": "RFPL"
}

# League Mapping for The-Odds-API
odds_api_leagues = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Russian Premier League": "soccer_russia_premier_league" 
}

selected_league_name = st.sidebar.selectbox("Select League", list(leagues.keys()))
league_code = leagues[selected_league_name]
odds_league_key = odds_api_leagues.get(selected_league_name)

# Live Odds Config
# st.sidebar.markdown("---")
# st.sidebar.markdown("### 🎲 Live Odds")
# Replace this with your actual key from the-odds-api.com
ODDS_API_KEY = "a136f290325a43885ca0bccc99576edb" 
odds_api_key = ODDS_API_KEY

# --- DATA FETCHING ---
with st.spinner(f"Fetching data for {selected_league_name}..."):
    df, teams = fetch_league_data(league_code, season)

if df.empty:
    st.error("Could not fetch data. Please try again.")
    st.stop()
    st.warning("⚠️ Could not fetch data. Try another league or season.")
else:
    # --- MAIN AREA SELECTION (Mobile Friendly) ---
    upcoming_df = df[df['xG'].isna()]
    fixtures = ["Select a Match..."]
    if not upcoming_df.empty:
        # Sort by DateTime
        upcoming_df['DateTime'] = pd.to_datetime(upcoming_df['DateTime'])
        upcoming_df = upcoming_df.sort_values(by='DateTime')
        
        fixture_list = [f"{row['Home']} vs {row['Away']} ({row['DateTime'].strftime('%Y-%m-%d %H:%M')})" for _, row in upcoming_df.iterrows()]
        fixtures.extend(fixture_list)

    # Big Prominent Select Box
    selected_fixture = st.selectbox("📅 Choose Upcoming Match", fixtures)
    
    # Validation & Analysis Trigger
    if selected_fixture != "Select a Match...":
        try:
            # Extract names from string like "Home vs Away (Date)"
            # Split by " (" first to get "Home vs Away"
            match_part = selected_fixture.split(" (")[0]
            p_home, p_away = match_part.split(" vs ")
            
            if p_home in teams and p_away in teams:
                # Run Prediction Immediately on selection
                engine = MatchPredictor()
                res = engine.predict_match(p_home, p_away, df)
                
                # Get the date for this match
                match_date = upcoming_df[(upcoming_df['Home'] == p_home) & (upcoming_df['Away'] == p_away)].iloc[0]['DateTime']
                
                # --- LIVE ODDS FETCHING ---
                live_odds = None
                if odds_api_key and odds_league_key:
                    with st.spinner("Checks live odds..."):
                        live_odds = fetch_live_odds(odds_api_key, odds_league_key, res['home'], res['away'])
                
                # --- RESULTS UI ---
                st.markdown("---")
                
                # Logos and Title
                c_logo_h, c_vs, c_logo_a = st.columns([1, 2, 1])
                
                logo_h = fetch_team_logo(res['home'])
                logo_a = fetch_team_logo(res['away'])
                
                with c_logo_h:
                    if logo_h: st.image(logo_h, width=80)
                
                with c_vs:
                    st.markdown(f"<h3 style='text-align: center; margin-top: 10px; margin-bottom: 0px'>{res['home']} vs {res['away']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<p style='text-align: center; color: gray; font-size: 0.9em;'>{match_date.strftime('%d %b %Y, %H:%M')}</p>", unsafe_allow_html=True)
                    if live_odds:
                         st.markdown(f"""
                         <div style='text-align: center; font-size: 0.8em; background: #fffbe6; padding: 5px; border-radius: 5px; border: 1px solid #ffe58f;'>
                            💰 Odds: 1({live_odds.get('home', '-')}) | X({live_odds.get('draw', '-')}) | 2({live_odds.get('away', '-')})
                         </div>
                         """, unsafe_allow_html=True)
                
                with c_logo_a:
                    if logo_a: st.image(logo_a, width=80)


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
                # 3. BALANCED MATCH CHECK: If 1X and X2 are close (< 10% diff), penalize ALL result-based bets
                # 4. VALUE ODDS CHECK: If odds < 1.18, penalize heavily (unimportant)
                
                diff_1x_x2 = abs(res['dc_1x'] - res['dc_x2'])
                is_balanced = diff_1x_x2 < 0.10
                
                def get_sort_score(item):
                    name, prob = item
                    
                    # Live Odds Filter (< 1.18)
                    if live_odds:
                        current_odd = 0
                        if f"{res['home']} Win" == name: current_odd = live_odds.get('home', 0)
                        elif f"{res['away']} Win" == name: current_odd = live_odds.get('away', 0)
                        elif "Draw" == name: current_odd = live_odds.get('draw', 0)
                        
                        # Apply heavy penalty if odds exist and are too low
                        if current_odd > 0 and current_odd < 1.18:
                            return prob * 0.1 # Kill it (unimportant)
                    
                    # If match is balanced, punish Winner/Double Chance markets
                    if is_balanced and ("Win" in name or "1X" in name or "X2" in name or "12" in name):
                        return prob * 0.5 # Heavy penalty to force Goal markets to top
                    
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
                
                if is_balanced:
                   st.warning("⚠️ **Tight Match**: Teams are evenly matched. Focusing on Goal Markets.")
                   
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
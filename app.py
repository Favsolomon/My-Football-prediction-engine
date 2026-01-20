import streamlit as st
import pandas as pd
import concurrent.futures
from src.config import init_page, ODDS_API_KEY, LEAGUES_UNDERSTAT, LEAGUES_ODDS_API
from src.ui import inject_css, render_tactical_tabs, render_match_results
from src.data_service import DataService
from src.engine import MatchPredictor

# 1. Initialize Page Config (Strict First Call)
init_page()

def main():
    # 2. Inject Design Theme
    inject_css()
    
    # 3. Initialize Session State
    if 'selected_league' not in st.session_state:
        st.session_state.selected_league = "Champions League"
    if 'master_store' not in st.session_state:
        st.session_state.master_store = {}
    if 'season' not in st.session_state:
        st.session_state.season = "2025"

    # 4. Global Sidebar Identity
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 20px; background: rgba(59,130,246,0.1); border-radius: 16px; margin-bottom: 25px; border: 1px solid rgba(59,130,246,0.2);'>
            <div style='font-size: 2.5rem; margin-bottom: 10px;'>🧠</div>
            <div style='font-weight: 800; font-size: 1.1rem; color: #60a5fa; letter-spacing: 0.05em;'>TACTICAL CORE</div>
            <div style='font-size: 0.6rem; opacity: 0.8; letter-spacing: 0.2em; color: #10b981;'>SYSTEM: ONLINE</div>
        </div>
    """, unsafe_allow_html=True)
    
    new_season = st.sidebar.selectbox("Active Season", ["2025", "2024"], index=0)
    if new_season != st.session_state.season:
        st.session_state.season = new_season
        st.session_state.master_store = {} # Force reload
        st.rerun()

    # 5. IMMEDIATE UI RENDER (Tabs)
    render_tactical_tabs()
    
    # 6. DATA LOADING LOGIC (Priority & Stealth Sync)
    selected_league = st.session_state.selected_league
    
    # Priority Loading for Active League
    if selected_league not in st.session_state.master_store:
        with st.spinner(f"📡 Syncing {selected_league}..."):
            name, data = DataService.preload_competition_context(selected_league, st.session_state.season)
            st.session_state.master_store[name] = data
            st.rerun()

    # Stealth Background Sync
    remaining = [l for l in LEAGUES_UNDERSTAT.keys() if l not in st.session_state.master_store]
    if remaining:
        st.caption("🔄 Secondary Tactical Feeds Syncing in Background...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(DataService.preload_competition_context, l, st.session_state.season): l for l in remaining}
            for future in concurrent.futures.as_completed(futures):
                name, data = future.result()
                st.session_state.master_store[name] = data

    # 7. Analyze Selected Competition
    master_data = st.session_state.master_store.get(selected_league, {})
    df = master_data.get("df", pd.DataFrame())
    upcoming = master_data.get("upcoming", pd.DataFrame())
    status = master_data.get("status", "OK")
    is_ucl = selected_league == "Champions League"

    if df.empty and not is_ucl:
        st.error(f"Analysis Failure: Could not synchronize {selected_league} data."); st.stop()

    if status == "QUOTA_EXCEEDED":
        st.warning("⚠️ **Usage Quota Reached**: The free API tier limit for today's match data has been exhausted. Champions League fixtures will be restored once the quota resets. Domestic leagues (EPL, La Liga, etc.) remain fully functional.")
    elif upcoming.empty:
        st.info(f"📡 No upcoming {selected_league} matches recorded.")
    else:
        # Sort and Display Fixtures
        if 'DateTime' in upcoming.columns:
            upcoming['DateTime'] = pd.to_datetime(upcoming['DateTime'])
            upcoming = upcoming.sort_values('DateTime')
        
        options = [f"{r['Home']} vs {r['Away']} ({r['DateTime'].strftime('%Y-%m-%d %H:%M')})" for _, r in upcoming.iterrows()]
        selection = st.selectbox("📅 Select Tactical Target", ["Select a Match..."] + options)
        
        if selection != "Select a Match...":
            try:
                match_str = selection.split(" (")[0]; home, away = match_str.split(" vs ")
                home_norm = DataService.normalize_team_name(home)
                away_norm = DataService.normalize_team_name(away)
                
                # Dynamic Logic execution
                live_odds = DataService.fetch_live_odds(ODDS_API_KEY, LEAGUES_ODDS_API[selected_league], home, away)
                predictor = MatchPredictor()
                res = predictor.predict_match(home_norm, away_norm, df, is_ucl, live_odds)
                match_date = upcoming[(upcoming['Home'] == home) & (upcoming['Away'] == away)].iloc[0]['DateTime']
                
                render_match_results(res, match_date, df)
            except Exception as e:
                st.error(f"Intelligence Failure: {e}")

    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
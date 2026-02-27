
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from src.data_service import DataService
from src.engine import MatchPredictor
from src.config import ODDS_API_KEY, LEAGUES_UNDERSTAT, LEAGUES_ODDS_API
import uvicorn
import concurrent.futures
from datetime import datetime, timedelta

# --- Server-Side Verification Cache ---
# Simple in-memory cache to prevent re-fetching data when calling both endpoints in parallel
CANDIDATES_CACHE = {
    "date": None,
    "data": [],
    "is_strict": True
}

app = FastAPI(title="Betly AI API")

from pathlib import Path

# Mount static files
static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    season: str = "2025"

# --- Prediction Cache (Global Search) ---
GLOBAL_FIXTURES_CACHE = {
    "data": None,
    "timestamp": None
}

@app.get("/api/fixtures/all")
async def get_all_fixtures(season: str = "2025"):
    # Return cached data if valid (10 minutes)
    now = datetime.now()
    if GLOBAL_FIXTURES_CACHE["data"] and GLOBAL_FIXTURES_CACHE["timestamp"]:
        elapsed = (now - GLOBAL_FIXTURES_CACHE["timestamp"]).total_seconds()
        if elapsed < 600:
            return GLOBAL_FIXTURES_CACHE["data"]

    def fetch_league(name):
        try:
            _, data = DataService.preload_competition_context(name, season)
            upcoming = data.get("upcoming", pd.DataFrame())
            if not upcoming.empty:
                fixtures = upcoming.replace({np.nan: None}).to_dict(orient="records")
                # Add league name to each fixture for searched results context
                for f in fixtures:
                    f['LeagueName'] = name
                return fixtures
        except:
            return []
        return []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_league, LEAGUES_UNDERSTAT.keys()))
    
    # Flatten the list of lists
    all_fixtures = [item for sublist in results for item in sublist]
    
    # Store in cache
    GLOBAL_FIXTURES_CACHE["data"] = {"fixtures": all_fixtures}
    GLOBAL_FIXTURES_CACHE["timestamp"] = datetime.now()
    
    return GLOBAL_FIXTURES_CACHE["data"]

@app.get("/")
@app.get("/index.html")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'index.html'))

@app.get("/top_picks.html")
async def read_top_picks():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'top_picks.html'))

@app.get("/api/leagues")
async def get_leagues():
    return [{"name": name, "code": code} for name, code in LEAGUES_UNDERSTAT.items()]

@app.get("/api/fixtures/{league_name}")
async def get_fixtures(league_name: str, season: str = "2025"):
    if league_name not in LEAGUES_UNDERSTAT:
        raise HTTPException(status_code=404, detail="League not found")
    
    try:
        # Utilizing the optimized preload logic
        _, data = DataService.preload_competition_context(league_name, season)
        upcoming = data.get("upcoming", pd.DataFrame())
        
        if upcoming.empty:
            return {"fixtures": [], "status": data.get("status", "OK")}
        
        # Convert DataFrame to list of dicts, handling NaN for JSON compatibility
        fixtures = upcoming.replace({np.nan: None}).to_dict(orient="records")
        return {"fixtures": fixtures, "status": data.get("status", "OK")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
def predict_match(request: MatchPredictionRequest):
    try:
        # 1. Fetch data context
        _, data = DataService.preload_competition_context(request.league, request.season)
        df = data.get("df")
        is_ucl = request.league == "Champions League"
        
        # 2. Normalize names
        home_norm = DataService.normalize_team_name(request.home_team)
        away_norm = DataService.normalize_team_name(request.away_team)
        
        # 3. Fetch live odds
        sport_key = LEAGUES_ODDS_API.get(request.league)
        live_odds = DataService.fetch_live_odds(ODDS_API_KEY, sport_key, request.home_team, request.away_team) if sport_key else None
        
        # 4. Predict
        predictor = MatchPredictor()
        res = predictor.predict_match(home_norm, away_norm, df, is_ucl, live_odds)
        
        if not res:
            raise HTTPException(status_code=404, detail="Could not generate prediction for these teams.")

        # 5. Get recommendations
        recs = predictor.get_recommendations(res)
        
        # 6. Metadata (logos, etc.)
        res['home_logo'] = DataService.fetch_team_logo(request.home_team) or ""
        res['away_logo'] = DataService.fetch_team_logo(request.away_team) or ""
        
        return {
            "prediction": res,
            "recommendations": recs
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Helper for fetching candidates with caching ---
def get_daily_candidates(season, min_edge=0.03):
    """Fetches candidates for next 3 days, using cache if available."""
    global CANDIDATES_CACHE
    today_str = datetime.now().date().isoformat()
    
    # Check cache
    if CANDIDATES_CACHE["date"] == today_str and len(CANDIDATES_CACHE["data"]) > 0:
        cached_is_strict = CANDIDATES_CACHE["is_strict"]
        if cached_is_strict or (not cached_is_strict and min_edge <= 0):
             return [c for c in CANDIDATES_CACHE["data"] if c['edge'] >= min_edge]

    # ... If cache miss, proceed to fetch ...
    print(f"Cache miss for {today_str}. Fetching fresh data (3-day window)...")
    predictor = MatchPredictor()
    candidates = []

    def process_league(league_name):
        league_candidates = []
        try:
            # Use a specific season for UCL if needed
            season_to_fetch = season if int(season) < 2025 or league_name != "Champions League" else "2024"
            _, data = DataService.preload_competition_context(league_name, season_to_fetch)
            upcoming = data.get("upcoming", pd.DataFrame())
            df = data.get("df")
            is_ucl = league_name == "Champions League"
            
            if upcoming.empty: return []
            
            # Filter for Next 3 Days (Today + 2)
            today = datetime.now().date()
            lookahead = today + pd.Timedelta(days=2)
            
            upcoming['MatchDate'] = pd.to_datetime(upcoming['DateTime']).dt.date
            # Keep matches in window [Today, Today+2]
            upcoming_window = upcoming[(upcoming['MatchDate'] >= today) & (upcoming['MatchDate'] <= lookahead)]
            
            if upcoming_window.empty: return []
            
            # Pre-fetch ODDS for the whole league once
            sport_key = LEAGUES_ODDS_API.get(league_name)
            league_odds_map = DataService.fetch_odds_batch(ODDS_API_KEY, sport_key) if sport_key else {}
        
            # For each match, generate candidates
            for _, f in upcoming_window.iterrows(): 
                home_norm = DataService.normalize_team_name(f['Home'])
                away_norm = DataService.normalize_team_name(f['Away'])
                match_date_str = str(f['MatchDate'])
                
                res = predictor.predict_match(home_norm, away_norm, df, is_ucl)
                if not res: continue
            
                outcomes = [
                    {"market": "H2H", "selection": f"{f['Home']} Win", "true_prob": res['h_win']},
                    {"market": "H2H", "selection": f"{f['Away']} Win", "true_prob": res['a_win']},
                    {"market": "Goals", "selection": "Over 2.5", "true_prob": res['over25']},
                    {"market": "Goals", "selection": "Over 1.5", "true_prob": res['over15']},
                ]
                
                # Robust lookup using the new batch map
                odds = DataService.get_odds_for_fixture(league_odds_map, f['Home'], f['Away'])
                
                for oc in outcomes:
                    price = 0
                    if odds:
                        if oc['market'] == "H2H":
                            price = odds['h2h'].get('home') if "Home" in oc['selection'] else odds['h2h'].get('away')
                        if oc['market'] == "Goals":
                            price = odds['totals'].get('over25') if "2.5" in oc['selection'] else odds['totals'].get('over15')
                    
                    if not price or price <= 1.01:
                        # No real odds available — include pick but without odds/edge data
                        oc['decimal_odds'] = None
                        oc['implied_prob'] = None
                        oc['edge'] = 0
                        league_candidates.append({
                            **oc,
                            "fixture": f"{f['Home']} vs {f['Away']}",
                            "match_date": match_date_str,
                            "league": league_name,
                            "edge_percent": 0,
                            "independence_factor": "Diversified",
                            "has_odds": False
                        })
                        continue
                    
                    oc['decimal_odds'] = price
                    oc['implied_prob'] = 1/price
                    oc['edge'] = oc['true_prob'] - oc['implied_prob']
                    
                    # Store all candidates; sorting will prioritize positive edges later
                    league_candidates.append({
                        **oc,
                        "fixture": f"{f['Home']} vs {f['Away']}",
                        "match_date": match_date_str, # Store date for filtering
                        "league": league_name,
                        "edge_percent": round(oc['edge'] * 100, 1),
                        "independence_factor": "Diversified",
                        "has_odds": True
                    })
        except Exception as e:
            print(f"Error processing {league_name}: {e}")
        return league_candidates

    leagues_to_process = list(LEAGUES_UNDERSTAT.keys())
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(leagues_to_process)) as executor:
        future_to_league = {executor.submit(process_league, name): name for name in leagues_to_process}
        for future in concurrent.futures.as_completed(future_to_league):
            candidates.extend(future.result())
            
    # Update Cache
    CANDIDATES_CACHE["date"] = today_str
    CANDIDATES_CACHE["data"] = candidates
    CANDIDATES_CACHE["is_strict"] = False # We cache everything >= 0 edge
    
    # Return filtered by requested edge
    return [c for c in candidates if c['edge'] >= min_edge]


@app.get("/api/top-picks")
def get_top_picks(season: str = "2025", offset: int = 0):
    """Returns the top mathematical edges for the specified offset without accumulator constraints."""
    # Pass a negative min_edge to capture all candidates in cache
    candidates = get_daily_candidates(season, min_edge=-1.0)

    target_date = datetime.now().date() + timedelta(days=offset)
    target_str = str(target_date)
    
    # Try target date first
    target_picks = [c for c in candidates if c.get('match_date') == target_str]
    
    # Custom sort key: Prefer real odds picks, then positive edge, then highest true_prob
    sort_key = lambda x: (1 if x.get('has_odds') else 0, x['edge_percent'] if x['edge_percent'] > 0 else 0, x['true_prob'])
    
    # Deduplicate: Keep only the single best pick per fixture
    def dedupe_by_fixture(picks):
        seen = {}
        for p in picks:
            fx = p['fixture']
            if fx not in seen or sort_key(p) > sort_key(seen[fx]):
                seen[fx] = p
        return list(seen.values())

    if len(target_picks) > 0:
        target_picks = dedupe_by_fixture(target_picks)
        # Only keep good probability picks (>= 60% true probability)
        target_picks = [p for p in target_picks if p['true_prob'] >= 0.60]
        target_picks = sorted(target_picks, key=sort_key, reverse=True)
        return {
            "picks": target_picks, # Dynamic count based on available quality picks
            "statistical_rationale": f"High value or mathematically safe selections for {target_str}."
        }
    
    if offset > 0:
        return {"picks": [], "message": f"NO MATCHES SCHEDULED ON {target_str}"}
        
    # If no picks today, expand to 3-day window
    if candidates:
        all_picks = dedupe_by_fixture(candidates)
        all_picks = [p for p in all_picks if p['true_prob'] >= 0.60]
        all_picks = sorted(all_picks, key=sort_key, reverse=True)
        return {
            "picks": all_picks,
            "statistical_rationale": "Rolling 3-day window best selections."
        }
        
    return {"picks": [], "message": "NO MATCHES AVAILABLE IN SYSTEM."}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


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
from datetime import datetime

# --- Server-Side Verification Cache ---
# Simple in-memory cache to prevent re-fetching data when calling both endpoints in parallel
CANDIDATES_CACHE = {
    "date": None,
    "data": [],
    "is_strict": True
}

app = FastAPI(title="Betly AI API")

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

@app.get("/")
@app.get("/index.html")
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'index.html'))

@app.get("/accumulator.html")
async def read_accumulator():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'accumulator.html'))

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
                    
                    if not price or price <= 1.01: continue
                    
                    oc['decimal_odds'] = price
                    oc['implied_prob'] = 1/price
                    oc['edge'] = oc['true_prob'] - oc['implied_prob']
                    
                    # Store ALL valid candidates (Edge >= 0) in cache
                    if oc['edge'] >= 0.0:
                        league_candidates.append({
                            **oc,
                            "fixture": f"{f['Home']} vs {f['Away']}",
                            "match_date": match_date_str, # Store date for filtering
                            "league": league_name,
                            "edge_percent": round(oc['edge'] * 100, 1),
                            "independence_factor": "Diversified"
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
def get_top_picks(season: str = "2025"):
    """Returns top 6 picks. Prioritizes Today, extends to Future if needed."""
    # 1. Fetch wide net (Today + 2 days), strict edge
    candidates = get_daily_candidates(season, min_edge=0.03) 
    
    if not candidates:
        # Try relaxed if strict returns nothing
        candidates = get_daily_candidates(season, min_edge=0.0)

    # 2. Filter logic: Try to fill with TODAY's games first
    today_str = str(datetime.now().date())
    today_picks = [c for c in candidates if c.get('match_date') == today_str]
    future_picks = [c for c in candidates if c.get('match_date') != today_str]
    
    # Sort both lists by edge
    today_picks.sort(key=lambda x: x['edge'], reverse=True)
    future_picks.sort(key=lambda x: x['edge'], reverse=True)
    
    final_picks = []
    
    # Take all good today picks (up to 6)
    final_picks.extend(today_picks)
    
    # If we have space, fill with future picks (best edges)
    if len(final_picks) < 6:
        needed = 6 - len(final_picks)
        final_picks.extend(future_picks[:needed])
        
    # Sanity check (> 45% prob)
    final_picks = [p for p in final_picks if p['true_prob'] > 0.45]
    
    return {"top_picks": final_picks[:6]}


@app.get("/api/accumulator")
def get_smart_accumulator(season: str = "2025"):
    """Runs the optimizer. Prioritizes Today, allows Future if needed."""
    from src.engine import AccumulatorOptimizer
    
    candidates = get_daily_candidates(season, min_edge=0.03)
    optimizer = AccumulatorOptimizer()
    
    # Try finding acca with TODAY's matches only first
    today_str = str(datetime.now().date())
    today_candidates = [c for c in candidates if c.get('match_date') == today_str]
    
    result = None
    rationale_suffix = ""
    
    # 1. Strict + Today
    if len(today_candidates) >= 3:
        result = optimizer.find_optimal(today_candidates, relaxed_mode=False)
        
    # 2. Relaxed + Today (if strict failed)
    if not result:
        # Re-fetch relaxed
        all_relaxed = get_daily_candidates(season, min_edge=0.00)
        today_relaxed = [c for c in all_relaxed if c.get('match_date') == today_str]
        if len(today_relaxed) >= 3:
            result = optimizer.find_optimal(today_relaxed, relaxed_mode=True)
            rationale_suffix = " (Relaxed constraints)"

    # 3. Multi-Day (if Today failed completely)
    if not result:
        # Use ALL candidates (Today + Future)
        print("Today's matches insufficient. Expanding to 3-day window...")
        if candidates:
            result = optimizer.find_optimal(candidates, relaxed_mode=False)
            rationale_suffix = " (Multi-day Selection)"
            
        if not result: 
            # Relaxed + Multi-Day
             all_relaxed = get_daily_candidates(season, min_edge=0.00)
             if all_relaxed:
                 result = optimizer.find_optimal(all_relaxed, relaxed_mode=True)
                 rationale_suffix = " (Multi-day, Relaxed)"

    if not result:
        return {"accumulator": None, "message": "NO BET – Market Too Efficient (No Value Found)"}
        
    result['legs'] = list(result['legs'])
    return {
        "accumulator": result,
        "statistical_rationale": f"Optimized across {len(LEAGUES_UNDERSTAT)} leagues. Selection prioritizes alpha edges.{rationale_suffix}"
    }
        

def from_src_engine_accumulator_optimizer():
    from src.engine import AccumulatorOptimizer
    return AccumulatorOptimizer()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

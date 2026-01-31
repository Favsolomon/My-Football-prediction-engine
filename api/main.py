
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

@app.get("/api/accumulator")
def get_smart_accumulator(season: str = "2025"):
    """Senior Quant logic for cross-league accumulator optimization."""
    try:
        import concurrent.futures
        from src.engine import AccumulatorOptimizer
        from datetime import datetime # Moved import here for broader scope
        
        predictor = MatchPredictor()
        
        # Helper to fetch candidates with adjustable strictness
        def fetch_candidates(min_edge=0.03):
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
                    
                    # Filter for TODAY ONLY as requested by user
                    today = datetime.now().date()
                    upcoming['MatchDate'] = pd.to_datetime(upcoming['DateTime']).dt.date
                    upcoming_today = upcoming[upcoming['MatchDate'] == today]
                    
                    if upcoming_today.empty: return []
                    
                    # Pre-fetch ODDS for the whole league once
                    sport_key = LEAGUES_ODDS_API.get(league_name)
                    league_odds_map = DataService.fetch_odds_batch(ODDS_API_KEY, sport_key) if sport_key else {}
                
                # For each match, generate candidates
                for _, f in upcoming_today.head(10).iterrows(): 
                    home_norm = DataService.normalize_team_name(f['Home'])
                    away_norm = DataService.normalize_team_name(f['Away'])
                    
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
                        
                        # Fallback ONLY if logic fails, but now we have real odds or 0 (skip)
                        # User wants sync with real odds. If no odds, we skip to avoid fake accumulation.
                        # Unless it's a very clear favorite in our model? 
                        # To comply with user request "not real life odd... sync with real odd", 
                        # let's be strict: if price is 0 (missing), we skip this leg or penalize it heavily.
                        # However, for robustness, if we miss odds, let's just ignore this market for the accumulator.
                        if not price or price <= 1.01: continue
                        
                        oc['decimal_odds'] = price
                        oc['implied_prob'] = 1/price
                        oc['edge'] = oc['true_prob'] - oc['implied_prob']
                        
                        if oc['edge'] >= min_edge:
                            league_candidates.append({
                                **oc,
                                "fixture": f"{f['Home']} vs {f['Away']}",
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
            return candidates

        # 1. First Pass: Strict Mode (Edge >= 3%)
        candidates = fetch_candidates(min_edge=0.03)
        optimizer = AccumulatorOptimizer()
        result = None
        
        if candidates:
            result = optimizer.find_optimal(candidates, relaxed_mode=False)
            
        # 2. Second Pass: Relaxed Mode (Edge >= 0%) if strict failed
        # This occurs if strictly no good bets exist. We loosen constraints to provide *something* useful.
        rationale_suffix = ""
        if not result:
            print("Strict mode failed. Retrying with relaxed constraints...")
            # Re-fetch with lower edge threshold (allow break-even bets)
            candidates = fetch_candidates(min_edge=0.00) 
            if candidates:
                result = optimizer.find_optimal(candidates, relaxed_mode=True)
                rationale_suffix = " (Constraints relaxed to ensure matchday coverage)"
        
        if not result:
            return {"accumulator": None, "message": "NO BET – Market Too Efficient at Target Odds"}
            
        result['legs'] = list(result['legs'])
        return {
            "accumulator": result,
            "statistical_rationale": f"Optimized across {len(LEAGUES_UNDERSTAT)} leagues. Selection prioritizes alpha edges.{rationale_suffix}"
        }
        
    except Exception as e:
        print(f"Accumulator Fatal Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def from_src_engine_accumulator_optimizer():
    from src.engine import AccumulatorOptimizer
    return AccumulatorOptimizer()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

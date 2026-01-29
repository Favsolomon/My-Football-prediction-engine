
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
async def read_index():
    return FileResponse(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'index.html'))

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
async def predict_match(request: MatchPredictionRequest):
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
        
        # 5. Get recommendations
        recs = predictor.get_recommendations(res)
        
        # 6. Metadata (logos, etc.)
        res['home_logo'] = DataService.fetch_team_logo(request.home_team)
        res['away_logo'] = DataService.fetch_team_logo(request.away_team)
        
        return {
            "prediction": res,
            "recommendations": recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

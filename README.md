
# ⚽ Agozie Match Predictor (Understat Integration)

A powerful web application that predicts football match outcomes using live Expected Goals (xG) data from **Understat.com**. Built with a modern FastAPI backend and a high-performance Vanilla JS frontend.

## Features

- **Live Data**: Fetches real-time xG data for the current season (2025/26) from Understat.
- **Top 5 Leagues + UCL**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, and Champions League.
- **Smart Prediction**: 
  - Uses Poisson distribution and Monte Carlo simulations to calculate Win/Draw/Loss probabilities.
  - **UCL DNA**: Proprietary weighting for European heritage in Champions League fixtures.
  - **Tactical Insights**: Provides three levels of picks (Primary, Tactical Margin, and Safety Picks).
- **Match Intelligence**: Advanced indicators like economic multipliers and momentum pulse.

## How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API Server**:
   ```bash
   python api/main.py
   ```

3. **Access the App**:
   Open `http://localhost:8000` in your browser.

## Technologies

- **FastAPI**: Modern, high-performance Python web framework.
- **Vanilla JS**: Lightweight, lightning-fast frontend logic.
- **CSS3**: Premium UI with glassmorphism and micro-animations.
- **Pandas/NumPy**: Heavy-duty data processing.
- **SciPy**: Advanced statistical modeling.
- **UnderstatAPI**: Python client for fetching football analytics.
- **The-Odds-API**: Real-time market sentiment and odds biasing.

## License

MIT License.


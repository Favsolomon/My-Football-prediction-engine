# My-Football-prediction-engine

# ⚽ Football Prediction Engine (Understat Integration)

A powerful Streamlit web application that predicts football match outcomes using live Expected Goals (xG) data from **Understat.com**.

## Features

- **Live Data**: Fetches real-time xG data for the current season (2025/26) from Understat.
- **Top 5 Leagues**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1.
- **Smart Prediction**: 
  - Uses Poisson distribution to calculate Win/Draw/Loss probabilities.
  - **Double Chance**: Calculates 1X, X2, 12 probabilities.
  - **Holistic Recommendation**: Automatically picks the statistically safest bet across all markets (Win, Double Chance, Over/Under, BTTS).
- **Historical Data**: Option to switch back to 2024 or 2023 seasons.

## How to Run Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Technologies

- **Streamlit**: fast Web UI.
- **Pandas/NumPy**: Data manipulation.
- **SciPy**: Poisson probability distribution.
- **UnderstatAPI**: Python client for fetching football analytics.

## License

MIT License.

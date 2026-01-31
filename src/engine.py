
# src/engine.py
import numpy as np
import pandas as pd
from .config import UCL_PEDIGREE, SQUAD_VALUE_INDEX

def poisson_pmf(k_array, lam):
    """Native numpy implementation of Poisson PMF to avoid scipy dependency."""
    # P(k; lam) = lam^k * e^-lam / k!
    # Using small k (0-9) so factorials are trivial.
    factorials = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880])
    k_array = np.array(k_array, dtype=int)
    
    # Safe guard for larger k if ever used, but we only use range(10)
    facts = np.array([factorials[k] if k < 10 else np.prod(np.arange(1, k+1)) for k in k_array])
    
    return (np.power(lam, k_array) * np.exp(-lam)) / facts

class MatchPredictor:
    """Core logic engine for computing probabilities and value recommendations."""

    def get_league_stats(self, df):
        """Computes league average xG for home and away teams."""
        played = df.dropna(subset=['xG', 'xG.1'])
        if played.empty:
            return 1.3, 1.3  # Default baselines
        avg_home_xg = played['xG'].mean()
        avg_away_xg = played['xG.1'].mean()
        return avg_home_xg, avg_away_xg

    def get_league_table(self, df):
        """Generates a league table from matching history to calculate rankings and dominance."""
        table = {}
        played = df.dropna(subset=['Score'])
        
        for _, row in played.iterrows():
            try:
                h_score, a_score = map(int, row['Score'].split('-'))
                h_team, a_team = row['Home'], row['Away']
                
                for team in [h_team, a_team]:
                    if team not in table:
                        table[team] = {'pts': 0, 'gp': 0, 'h_pts': 0, 'h_gp': 0, 'a_pts': 0, 'a_gp': 0}
                
                table[h_team]['gp'] += 1
                table[a_team]['gp'] += 1
                
                if h_score > a_score:
                    table[h_team]['pts'] += 3
                    table[h_team]['h_pts'] += 3
                    table[h_team]['h_gp'] += 1
                    table[a_team]['a_gp'] += 1
                elif h_score < a_score:
                    table[a_team]['pts'] += 3
                    table[a_team]['a_pts'] += 3
                    table[a_team]['a_gp'] += 1
                    table[h_team]['h_gp'] += 1
                else:
                    table[h_team]['pts'] += 1
                    table[a_team]['pts'] += 1
                    table[h_team]['h_pts'] += 1
                    table[a_team]['a_pts'] += 1
                    table[h_team]['h_gp'] += 1
                    table[a_team]['a_gp'] += 1
            except:
                continue
        
        sorted_table = sorted(table.items(), key=lambda x: x[1]['pts'], reverse=True)
        return {team: {'rank': i+1, **stats} for i, (team, stats) in enumerate(sorted_table)}

    def calculate_elo(self, df):
        """Calculates a simple Elo rating for all teams based on match results."""
        elo = {team: 1500 for team in pd.concat([df['Home'], df['Away']]).unique()}
        played = df.dropna(subset=['Score'])
        K = 32
        
        for _, row in played.iterrows():
            try:
                h, a = row['Home'], row['Away']
                h_score, a_score = map(int, row['Score'].split('-'))
                
                # Expected outcomes
                r_h, r_a = elo[h], elo[a]
                e_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
                e_a = 1 / (1 + 10 ** ((r_h - r_a) / 400))
                
                # Actual outcomes
                s_h = 1 if h_score > a_score else 0.5 if h_score == a_score else 0
                s_a = 1 - s_h
                
                # Update Elo
                elo[h] += K * (s_h - e_h)
                elo[a] += K * (s_a - e_a)
            except:
                continue
        return elo

    def tau_adjustment(self, x, y, l_h, l_a, rho=-0.1):
        """Dixon-Coles adjustment function for low-scoring interdependence."""
        if x == 0 and y == 0: return 1 - (l_h * l_a * rho)
        elif x == 0 and y == 1: return 1 + (l_h * rho)
        elif x == 1 and y == 0: return 1 + (l_a * rho)
        elif x == 1 and y == 1: return 1 - rho
        return 1.0

    def run_monte_carlo(self, l_h, l_a, iterations=10000):
        """Runs 10,000 simulations to derive probabilistic outcomes and variance."""
        h_sims = np.random.poisson(l_h, iterations)
        a_sims = np.random.poisson(l_a, iterations)
        
        results = h_sims - a_sims
        h_wins = np.sum(results > 0)
        draws = np.sum(results == 0)
        a_wins = np.sum(results < 0)
        
        # Expected Points (xP)
        h_xp = (h_wins * 3 + draws * 1) / iterations
        a_xp = (a_wins * 3 + draws * 1) / iterations
        
        return {
            "h_win": h_wins / iterations, "draw": draws / iterations, "a_win": a_wins / iterations,
            "h_xp": h_xp, "a_xp": a_xp, "avg_goals": np.mean(h_sims + a_sims)
        }

    def calculate_strength(self, team, df, is_home, avg_home_xg, avg_away_xg, league_table=None, elo=None, is_ucl=False):
        """Calculates strength using Elo-weighted xG and context-aware Multipliers."""
        played = df.dropna(subset=['xG', 'xG.1'])
        team_matches = played[(played['Home'] == team) | (played['Away'] == team)].tail(5)
        clinical_idx = 1.0
        
        if is_ucl:
            coeff = getattr(self, 'LEAGUE_COEFFICIENTS', {}).get(team, 0.85) 
            if not coeff or coeff == 0.85:
                # Fallback to importing from config if not on self (Senior failsafe)
                from .config import LEAGUE_COEFFICIENTS
                coeff = LEAGUE_COEFFICIENTS.get(team, 0.85)
            
            pedigree = UCL_PEDIGREE.get(team, 1.0)
            squad_val = SQUAD_VALUE_INDEX.get(team, 1.0)
        else:
            coeff = 1.0
            pedigree = 1.0
            squad_val = 1.0
        
        if team_matches.empty:
            atk_strength, def_strength = 1.0, 1.0
        else:
            avg_league_elo = np.mean(list(elo.values())) if elo else 1500
            atk_vals, def_vals, weight_array = [], [], []
            
            for i, (_, row) in enumerate(team_matches.iterrows()):
                is_team_home = row['Home'] == team
                opponent = row['Away'] if is_team_home else row['Home']
                opp_elo = elo.get(opponent, 1500) if elo else 1500
                
                q_weight = (opp_elo / avg_league_elo) ** 1.5 
                r_weight = (i + 1) / len(team_matches)
                total_weight = q_weight * r_weight
                
                if is_team_home:
                    atk_vals.append(row['xG'])
                    def_vals.append(row['xG.1'])
                else:
                    atk_vals.append(row['xG.1'])
                    def_vals.append(row['xG'])
                weight_array.append(total_weight)
                    
            team_avg_atk = np.average(atk_vals, weights=weight_array) if atk_vals else 0
            team_avg_def = np.average(def_vals, weights=weight_array) if def_vals else 1.0
            
            # --- MOMENTUM FACTOR (Trajectory Analysis) ---
            momentum_multiplier = 1.0
            if len(atk_vals) >= 3:
                try:
                    # Calculate slope of offensive production (recent xG)
                    x_axis = np.arange(len(atk_vals))
                    slope, _ = np.polyfit(x_axis, atk_vals, 1)
                    
                    # Sigmoid-style smoothing for momentum
                    if slope > 0.15: momentum_multiplier = 1.06   # Strong uptrend (Ascending)
                    elif slope > 0.05: momentum_multiplier = 1.03  # Slight uptrend
                    elif slope < -0.15: momentum_multiplier = 0.94 # Sharp decline (Collapsing)
                    elif slope < -0.05: momentum_multiplier = 0.97 # Slight decline
                    print(f"Momentum Analysis for {team}: Slope={slope:.4f}, Multiplier={momentum_multiplier:.2f}")
                except Exception as e:
                    print(f"Momentum Calculation Error for {team}: {e}")
                    pass
            
            # Clinical Factor (xG Conversion Efficiency)
            total_actual = 0
            total_expected = 0
            for _, row in team_matches.iterrows():
                if row['Score']:
                    hs, ascore = map(int, row['Score'].split('-'))
                    is_team_h = row['Home'] == team
                    total_actual += hs if is_team_h else ascore
                    total_expected += row['xG'] if is_team_h else row['xG.1']
            
            clinical_idx = 1.0
            if total_expected > 0:
                eff = total_actual / total_expected
                # smoothed impact: 25% of the deviation from 1.0, capped at +/- 8%
                clinical_idx = 1.0 + (eff - 1.0) * 0.25
                clinical_idx = max(0.92, min(clinical_idx, 1.08))
                print(f"Clinical Analysis for {team}: Eff={eff:.2f}, Multiplier={clinical_idx:.2f}")
            
            if is_home:
                atk_strength = (team_avg_atk / avg_home_xg) * coeff
                def_strength = (team_avg_def / avg_away_xg) * (2.0 - coeff)
            else:
                atk_strength = (team_avg_atk / avg_away_xg) * coeff
                def_strength = (team_avg_def / avg_home_xg) * (2.0 - coeff)
            
            atk_strength *= clinical_idx
            atk_strength *= momentum_multiplier
            atk_strength *= (pedigree * squad_val)
            def_strength /= (pedigree * squad_val)

        if league_table and team in league_table:
            stats = league_table[team]
            # Steeper rank-based scaling (Rank 1: ~1.13x, Rank 20: ~0.85x)
            rank_boost = 1.15 - (stats['rank'] / 20 * 0.3) 
            atk_strength *= rank_boost
            
            # Tiered Quality Adjustments
            if stats['rank'] <= 4:
                atk_strength *= 1.10  # Elite offensive efficiency
                def_strength *= 0.88  # Elite defensive discipline (lower is better)
            elif stats['rank'] >= 18:
                atk_strength *= 0.85  # Bottom-tier finishing struggle
                def_strength *= 1.25  # Relegation-zone defensive fragility (higher is worse)
            
            if is_home and stats['h_gp'] > 0:
                h_ppg = stats['h_pts'] / stats['h_gp']
                if h_ppg > 2.1: atk_strength *= 1.08 # Fortified home ground
            elif not is_home and stats['a_gp'] > 0:
                a_ppg = stats['a_pts'] / stats['a_gp']
                if a_ppg > 1.9: atk_strength *= 1.08 # Dominant away force
        
        # Guard rails for stability
        atk_strength = max(0.5, min(atk_strength, 1.8))
        def_strength = max(0.5, min(def_strength, 1.8))
                
        return atk_strength, def_strength, pedigree, squad_val, clinical_idx

    def get_h2h_history(self, home, away, df):
        """Analyzes Head-to-Head history for tactical trends."""
        h2h = df[((df['Home'] == home) & (df['Away'] == away)) | 
                 ((df['Home'] == away) & (df['Away'] == home))].dropna(subset=['Score']).tail(5)
        
        if h2h.empty: return None
        
        h_wins, a_wins, draws = 0, 0, 0
        total_goals = 0
        for _, row in h2h.iterrows():
            hs, ascore = map(int, row['Score'].split('-'))
            total_goals += (hs + ascore)
            if hs == ascore: draws += 1
            elif (row['Home'] == home and hs > ascore) or (row['Away'] == home and ascore > hs): h_wins += 1
            else: a_wins += 1

        return {
            "h_wins": h_wins, "a_wins": a_wins, "draws": draws,
            "avg_goals": total_goals / len(h2h),
            "recent": h2h.to_dict('records')
        }

    def predict_match(self, home_team, away_team, df, is_ucl=False, live_odds=None):
        """Runs Advanced Analytical Pipeline: Dixon-Coles Correction + Monte Carlo + Market Bias."""
        avg_h_xg, avg_a_xg = self.get_league_stats(df)
        league_table = self.get_league_table(df)
        elo = self.calculate_elo(df)

        h_atk, h_def, h_ped, h_val, h_clin = self.calculate_strength(home_team, df, True, avg_h_xg, avg_a_xg, league_table, elo, is_ucl)
        a_atk, a_def, a_ped, a_val, a_clin = self.calculate_strength(away_team, df, False, avg_h_xg, avg_a_xg, league_table, elo, is_ucl)

        l_home = h_atk * a_def * avg_h_xg * 1.08
        l_away = a_atk * h_def * avg_a_xg * 0.92

        # 1. Market Sentiment Bias (SUBTLE preference to heavy favorites / goal lines)
        # User Feedback: Analytical core matters most. These multipliers are refinements, not overrides.
        if live_odds:
            try:
                # Apply subtle bias to Market Favorites
                if 'h2h' in live_odds:
                    h_prob = 1/live_odds['h2h'].get('home', 10)
                    a_prob = 1/live_odds['h2h'].get('away', 10)
                    
                    if h_prob > 0.65: l_home *= 1.04 # Slight Market Favorite Refinement (4%)
                    elif a_prob > 0.60: l_away *= 1.04
                
                # Apply subtle bias to Over 1.5 (Minor scoring favor)
                if 'totals' in live_odds and live_odds['totals'].get('over15', 10) < 1.35:
                    l_home *= 1.02 # Subtle scoring nudge (2%)
                    l_away *= 1.02
            except: pass

        h_pmf = poisson_pmf(np.arange(10), l_home)
        a_pmf = poisson_pmf(np.arange(10), l_away)
        matrix = np.outer(h_pmf, a_pmf)

        for i in range(2):
            for j in range(2):
                matrix[i, j] *= self.tau_adjustment(i, j, l_home, l_away)

        mc = self.run_monte_carlo(l_home, l_away)
        h2h = self.get_h2h_history(home_team, away_team, df)

        if h2h:
            if h2h['h_wins'] > h2h['a_wins']: l_home *= 1.05
            elif h2h['a_wins'] > h2h['h_wins']: l_away *= 1.05

        h_win = np.sum(np.tril(matrix, -1))
        draw_prob = np.sum(np.diag(matrix))
        a_win = np.sum(np.triu(matrix, 1))

        return {
            "home": home_team, "away": away_team,
            "l_home": l_home, "l_away": l_away,
            "h_win": h_win, "draw": draw_prob, "a_win": a_win,
            "mc_h_win": mc['h_win'], "mc_draw": mc['draw'], "mc_a_win": mc['a_win'],
            "h_xp": mc['h_xp'], "a_xp": mc['a_xp'],
            "btts": (1 - h_pmf[0]) * (1 - a_pmf[0]),
            "over25": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)]),
            "under25": np.sum([h_pmf[i]*a_pmf[j] for i in range(3) for j in range(3-i)]),
            "over15": 1 - np.sum([h_pmf[i]*a_pmf[j] for i in range(2) for j in range(2-i)]),
            "under35": np.sum([h_pmf[i]*a_pmf[j] for i in range(4) for j in range(4-i)]),
            "predicted_score": f"{np.unravel_index(matrix.argmax(), matrix.shape)[0]}-{np.unravel_index(matrix.argmax(), matrix.shape)[1]}",
            "elo_h": elo.get(home_team, 1500), "elo_a": elo.get(away_team, 1500),
            "rank_h": league_table.get(home_team, {}).get('rank', 'N/A'),
            "rank_a": league_table.get(away_team, {}).get('rank', 'N/A'),
            "pts_h": league_table.get(home_team, {}).get('pts', 0),
            "pts_a": league_table.get(away_team, {}).get('pts', 0),
            "h2h": h2h, "ped_h": h_ped, "ped_a": a_ped, "val_h": h_val, "val_a": a_val,
            "clin_h": h_clin, "clin_a": a_clin,
            "is_ucl": is_ucl
        }

    def get_recommendations(self, res):
        """Statistical engine providing three levels of tactical recommendations."""
        h_xg, a_xg = res['l_home'], res['l_away']
        
        # 1. Primary Pick (The Core Direction)
        if a_xg > 2.25 and h_xg > 1.60:
            primary_pick = "BTTS (Yes)"
            primary_insight = "Both teams are scoring for fun lately. Expect goals at both ends."
        elif res['mc_h_win'] > 0.65:
            primary_pick = f"{res['home']} Win"
            primary_insight = f"{res['home']} are in great form and should handle this easily."
        elif res['mc_a_win'] > 0.60:
            primary_pick = f"{res['away']} Win"
            primary_insight = f"{res['away']} are playing very well and look likely to win."
        elif res['mc_h_win'] > res['mc_a_win']:
            primary_pick = "Home/Draw (1X)"
            primary_insight = "A close one, but the home team is very likely to avoid a loss."
        else:
            primary_pick = "Away/Draw (X2)"
            primary_insight = "The away team is strong enough to at least get a draw here."

        # 2. Risky Value Pick (The 'High Growth' Analytical Choice)
        # Focuses on 55-68% range outcomes that aren't the primary
        tactical_pick = None
        tactical_insight = ""
        
        t_candidates = []
        if primary_pick != "BTTS (Yes)": t_candidates.append(("BTTS (Yes)", "Both teams have a strong habit of scoring recently.", res['btts']))
        if primary_pick != "Over 2.5 Goals": t_candidates.append(("Over 2.5 Goals", "A good chance we see at least three goals here.", res['over25']))
        if "Win" not in primary_pick:
            t_candidates.append((f"{res['home']} Win", "Low-tier teams often struggle here; home win is plausible.", res['mc_h_win']))
            t_candidates.append((f"{res['away']} Win", "Visitors have the tactical edge to take the full points.", res['mc_a_win']))
        
        # Filter for 'Risky but Probable' (~55-70%)
        t_viable = [c for c in t_candidates if 0.50 < c[2] < 0.70]
        if t_viable:
            t_viable.sort(key=lambda x: x[2], reverse=True)
            tactical_pick, tactical_insight = t_viable[0][0], t_viable[0][1]
        else:
            # Fallback for Tactical
            tactical_pick = "Over 2.5 Goals" if res['over25'] > 0.45 else "Under 3.5 Goals"
            tactical_insight = "A high-risk analytical choice based on recent variance patterns."

        # 3. Safety Pick (The High Probability Conservative Choice)
        # Focuses on >70% probability locks
        safety_candidates = [
            ("Over 1.5 Goals", "Very likely to see at least two goals in this match.", res['over15']),
            ("Under 3.5 Goals", "Expect a disciplined defensive performance from both.", res['under35']),
        ]
        if res['mc_h_win'] > 0.75: safety_candidates.append(("Home or Draw", "The home side looks extremely hard to beat.", res['mc_h_win'] + res['mc_draw']))
        if res['mc_a_win'] > 0.70: safety_candidates.append(("Away or Draw", "The visitors should at least secure a point.", res['mc_a_win'] + res['mc_draw']))
        
        safety_candidates.sort(key=lambda x: x[2], reverse=True)
        safety_pick, safety_insight = safety_candidates[0][0], safety_candidates[0][1]

        return {
            "primary": {"pick": primary_pick, "insight": primary_insight, "type": "Primary Pick"},
            "tactical": {"pick": tactical_pick, "insight": tactical_insight, "type": "Risky Value Pick"},
            "safety": {"pick": safety_pick, "insight": safety_insight, "type": "Safety Pick"}
        }

class AccumulatorOptimizer:
    """Professional-grade combinatorial optimizer for multi-leg betting."""
    
    def __init__(self, target_odds=3.0, range_low=2.8, range_high=3.3):
        self.target_odds = target_odds
        self.range_low = range_low
        self.range_high = range_high

    def get_correlation(self, leg1, leg2):
        """Heuristic ρ calculation based on league and market overlap."""
        same_league = leg1['league'] == leg2['league']
        same_market = leg1['market'] == leg2['market']
        
        if same_league and same_market: return 0.35
        if same_league: return 0.20
        if same_market: return 0.15
        return 0.05

    def calculate_score(self, legs, combined_prob, combined_odds):
        """Standardized Risk-Adjusted Utility Score."""
        ev = (combined_prob * combined_odds) - 1
        if ev <= 0: return -1
        
        leg_count = len(legs)
        max_rho = 0
        for i in range(leg_count):
            for j in range(i + 1, leg_count):
                max_rho = max(max_rho, self.get_correlation(legs[i], legs[j]))
        
        if max_rho > 0.40: return -1 # Hard constraint
        
        score = (ev * np.sqrt(combined_prob)) / ((leg_count ** 1.5) * (1 + max_rho))
        return score

    def compute_confidence(self, legs, combined_prob, combined_odds, avg_rho):
        """Composite 0-100 Confidence Score."""
        # 1. Edge Strength (30%)
        alphas = [l['true_prob'] - l['implied_prob'] for l in legs]
        avg_alpha = np.mean(alphas)
        edge_pts = min(avg_alpha * 500, 30)
        
        # 2. Probability Density (25%)
        prob_pts = combined_prob * 25
        
        # 3. Odds Proximity (20%)
        proximity_pts = 20 - abs(self.target_odds - combined_odds) * 20
        
        # 4. Diversification (15%)
        div_pts = 15 * (1 - avg_rho)
        
        # 5. Leg Efficiency (10%)
        leg_pts = max(0, 10 - abs(len(legs) - 3) * 5)
        
        return int(edge_pts + prob_pts + proximity_pts + div_pts + leg_pts)

    def find_optimal(self, candidate_legs, relaxed_mode=False):
        """Combinatorial search for the highest Score(A)."""
        import itertools
        best_acca = None
        best_score = -999 # Allow searching for best possible even if negative in extremely relaxed mode
        
        # Limit legs to 5
        leg_limit = min(5, len(candidate_legs))
        
        for r in range(1, leg_limit + 1):
            for combo in itertools.combinations(candidate_legs, r):
                # Verify no two legs from same match
                matches = [l['fixture'] for l in combo]
                if len(set(matches)) < r: continue
                
                c_odds = np.prod([l['decimal_odds'] for l in combo])
                
                # In relaxed mode, slightly widen the odds range if needed
                low = self.range_low if not relaxed_mode else 2.5
                high = self.range_high if not relaxed_mode else 4.0
                
                if not (low <= c_odds <= high): continue
                
                # Apply leg penalties (-0.02 alpha per leg beyond 3)
                adj_prob_multiplier = 1.0
                if r > 3:
                    penalty = (r - 3) * 0.02
                    adj_prob_multiplier = (1.0 - penalty)
                
                c_prob = np.prod([l['true_prob'] for l in combo]) * adj_prob_multiplier
                
                # In relaxed mode, we might accept negative EV if user insists on a bet
                # but let's try to keep it at least > -3% EV
                ev = (c_prob * c_odds) - 1
                if not relaxed_mode and ev <= 0: continue
                
                # Calculate utility score
                # Re-implement simplified score here to avoid strict EV checks in calculate_score
                # for relaxed mode
                
                avg_rho = 0
                max_rho = 0
                if r > 1:
                    rhos = [self.get_correlation(combo[i], combo[j]) for i in range(r) for j in range(i+1, r)]
                    avg_rho = np.mean(rhos)
                    max_rho = max(rhos)
                
                if max_rho > 0.40 and not relaxed_mode: continue
                
                score = (ev * np.sqrt(c_prob)) / ((r ** 1.5) * (1 + max_rho))
                
                if score > best_score:
                    # Select fewer legs if scores are nearly identical
                    if best_acca and abs(score - best_score) < 0.05 and r > len(best_acca['legs']):
                        continue
                    
                    best_score = score
                    
                    conf = self.compute_confidence(combo, c_prob, c_odds, avg_rho)
                    ev_pct = (c_prob * c_odds - 1) * 100
                    kelly = (ev_pct / 100) / (c_odds - 1) if c_odds > 1 else 0
                    
                    band = "Reject"
                    if conf >= 90: band = "High Confidence"
                    elif conf >= 70: band = "Solid Play"
                    elif conf >= 50: band = "Marginal"
                    else: band = "Speculative" # New band for relaxed mode
                    
                    best_acca = {
                        "target_odds": self.target_odds,
                        "actual_odds": round(c_odds, 2),
                        "legs": combo,
                        "combined_probability": round(c_prob, 3),
                        "expected_value_percent": round(ev_pct, 1),
                        "kelly_fraction": round(max(0.1, kelly * 0.25 * 10), 2), # Ensure min stake for engagement
                        "confidence_score": conf,
                        "confidence_band": band,
                        "risk_factors": [
                            f"{'High' if avg_rho > 0.2 else 'Low'} correlation across legs",
                            "Model variance at higher leg counts" if r > 3 else "Stable leg count"
                        ],
                        "score": score
                    }
        
        # Strict Mode Threshold
        if not relaxed_mode:
            if not best_acca or best_acca['confidence_score'] < 50:
                return None
        
        # Relaxed Mode: Return best found, even if low confidence
        if relaxed_mode and best_acca:
             best_acca['risk_factors'].append("Relaxed constraints applied")
             return best_acca
             
        return best_acca




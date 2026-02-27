
# src/engine.py
import numpy as np
import pandas as pd
from .config import UCL_PEDIGREE, SQUAD_VALUE_INDEX

def poisson_pmf(k_array, lam):
    """Native numpy implementation of Poisson PMF to avoid scipy dependency.
    Covers k=0..11 for a 12×12 probability matrix, ensuring negligible
    probability mass is lost even when λ > 3.5."""
    # FIX 1: Extended factorials to k=0..11
    factorials = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800])
    k_array = np.array(k_array, dtype=int)
    
    # Safe guard for larger k if ever used, but we only use range(12)
    facts = np.array([factorials[k] if k < 12 else np.prod(np.arange(1, k+1)) for k in k_array])
    
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
        # FIX 2: Reduced K from 32 to 20 for more stable Elo ratings
        K = 20
        
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
        """Dixon-Coles adjustment function for low-scoring interdependence.
        FIX 3: Returns 1.0 when either λ exceeds 2.0, as the correction
        was derived for typical goal rates (λ ≈ 1.0–1.6) and is
        mathematically unsound outside that range."""
        # FIX 3: Guard against high λ values
        if l_h > 2.0 or l_a > 2.0:
            return 1.0
        if x == 0 and y == 0: return 1 - (l_h * l_a * rho)
        elif x == 0 and y == 1: return 1 + (l_h * rho)
        elif x == 1 and y == 0: return 1 + (l_a * rho)
        elif x == 1 and y == 1: return 1 - rho
        return 1.0

    def get_venue_form(self, team, df, is_home):
        """
        Analyzes performance specifically at the relevant venue (Home or Away).
        Returns: (avg_goals_scored, avg_goals_conceded, consistency_score)
        """
        if df.empty: return 1.0, 1.0, 1.0
        
        # Filter for recent games at this specific venue
        if is_home:
            venue_games = df[df['Home'] == team].dropna(subset=['Score']).tail(5)
            if venue_games.empty: return 1.5, 1.0, 1.0 # Default assumption
            
            goals_for = venue_games['Score'].apply(lambda x: int(x.split('-')[0]))
            goals_against = venue_games['Score'].apply(lambda x: int(x.split('-')[1]))
        else:
            venue_games = df[df['Away'] == team].dropna(subset=['Score']).tail(5)
            if venue_games.empty: return 1.0, 1.5, 1.0 # Default assumption
            
            goals_for = venue_games['Score'].apply(lambda x: int(x.split('-')[1]))
            goals_against = venue_games['Score'].apply(lambda x: int(x.split('-')[0]))
            
        avg_gf = goals_for.mean()
        avg_ga = goals_against.mean()
        
        # Consistency: Standard Deviation of Goal Difference (Low std dev = High Consistency)
        # We invert it so higher score = more consistent
        gd_std = (goals_for - goals_against).std()
        consistency = 1.0
        if not np.isnan(gd_std):
            # Scale: 0.0 std (perfectly consistent) -> 1.1x multiplier
            # 3.0 std (very erratic) -> 0.9x multiplier
            consistency = max(0.9, min(1.1, 1.1 - (gd_std * 0.06)))
            
        return avg_gf, avg_ga, consistency

    def calculate_strength(self, team, df, is_home, avg_home_xg, avg_away_xg, league_table=None, elo=None, is_ucl=False, own_venue_stats=None, opponent_venue_stats=None):
        """Calculates strength using Elo-weighted xG and context-aware multipliers.
        FIX 4: Venue double-counting removed — base strength uses overall xG only.
        FIX 5: League table collapsed into single rank_factor formula.
        FIX 6: Clinical & momentum merged into single form_factor.
        FIX 7: Pedigree & squad value merged into capped club_quality.
        FIX 8: Elo quality weight exponent linearised."""
        played = df.dropna(subset=['xG', 'xG.1'])
        
        # 1. Overall Form (Last 5 Games Anywhere)
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
            
        # Calculate Overall Metrics
        team_avg_atk_overall = 0
        team_avg_def_overall = 1.0
        # FIX 6: Store momentum as component, not multiplier
        momentum_component = 1.0
        clinical_component = 1.0
        
        if not team_matches.empty:
            avg_league_elo = np.mean(list(elo.values())) if elo else 1500
            atk_vals, def_vals, weight_array = [], [], []
            
            for i, (_, row) in enumerate(team_matches.iterrows()):
                is_team_home = row['Home'] == team
                opponent = row['Away'] if is_team_home else row['Home']
                opp_elo = elo.get(opponent, 1500) if elo else 1500
                
                # FIX 8: Linear quality weight (exponent 1.0 instead of 1.5)
                q_weight = opp_elo / avg_league_elo
                r_weight = (i + 1) / len(team_matches)
                total_weight = q_weight * r_weight
                
                # Check actual production
                prod_atk = row['xG'] if is_team_home else row['xG.1']
                prod_def = row['xG.1'] if is_team_home else row['xG']
                
                atk_vals.append(prod_atk)
                def_vals.append(prod_def)
                weight_array.append(total_weight)
                    
            team_avg_atk_overall = np.average(atk_vals, weights=weight_array) if atk_vals else 0
            team_avg_def_overall = np.average(def_vals, weights=weight_array) if def_vals else 1.0
            
            # --- MOMENTUM FACTOR (Trajectory Analysis) ---
            # FIX 6: Continuous mapping instead of discrete thresholds
            if len(atk_vals) >= 3:
                try:
                    x_axis = np.arange(len(atk_vals))
                    slope, _ = np.polyfit(x_axis, atk_vals, 1)
                    
                    # Continuous: slope of +0.20 → ×1.06, slope of -0.20 → ×0.94
                    momentum_component = 1.0 + max(-0.06, min(0.06, slope / 0.20 * 0.06))
                except: pass
            
            # Clinical Factor (xG Conversion Efficiency)
            total_actual = 0
            total_expected = 0
            for _, row in team_matches.iterrows():
                if row['Score']:
                    hs, ascore = map(int, row['Score'].split('-'))
                    is_team_h = row['Home'] == team
                    total_actual += hs if is_team_h else ascore
                    total_expected += row['xG'] if is_team_h else row['xG.1']
            
            if total_expected > 0:
                eff = total_actual / total_expected
                # smoothed impact: 25% of the deviation from 1.0, capped at +/- 8%
                clinical_component = 1.0 + max(-0.08, min(0.08, (eff - 1.0) * 0.25))

        # FIX 4 (Part 1): No blending — use overall xG directly for base strength.
        # The venue signal is carried solely by the venue_multiplier block below.
        team_avg_atk = team_avg_atk_overall
        team_avg_def = team_avg_def_overall
        
        # Recalculate basic strengths
        if is_home:
            atk_strength = (team_avg_atk / avg_home_xg) * coeff
            def_strength = (team_avg_def / avg_away_xg) * (2.0 - coeff)
        else:
            atk_strength = (team_avg_atk / avg_away_xg) * coeff
            def_strength = (team_avg_def / avg_home_xg) * (2.0 - coeff)
            
        # FIX 6: Apply merged form_factor (60% momentum + 40% clinical) once
        form_factor = 0.60 * momentum_component + 0.40 * clinical_component
        atk_strength *= form_factor
        # Keep clinical_idx for the return value so the UI display is unaffected
        clinical_idx = clinical_component

        # FIX 7: Merge Pedigree & Squad Value into single capped club_quality
        club_quality = max(pedigree, squad_val) * 0.6 + min(pedigree, squad_val) * 0.4
        club_quality = min(club_quality, 1.25)  # absolute ceiling
        atk_strength *= club_quality
        def_strength /= club_quality

        # --- RELATIVE VENUE ANALYSIS (Matchup Specific) ---
        # FIX 4 (Part 2): venue_multiplier is the sole carrier of the venue signal.
        venue_multiplier = 1.0
        
        if opponent_venue_stats and own_venue_stats:
            v_gf, _, _ = own_venue_stats
            _, opp_v_ga, _ = opponent_venue_stats
            
            # Matchup Logic: My Venue Attack vs Opponent Venue Defense
            if is_home:
                # Home Fortress Logic (remains strong as it's crowd/familiarity driven)
                if v_gf > 1.8 and opp_v_ga > 1.8:
                    venue_multiplier = 1.12 # Fortress exploiting weakness
                elif v_gf < 1.0 and opp_v_ga < 1.0:
                    venue_multiplier = 0.90 # Stoppage expected
            else:
                # Away Team Logic (Dampened as per user feedback: Travel fatigue is minimal)
                # We focus purely on tactical disadvantage of being away, not fatigue.
                if v_gf > 1.6 and opp_v_ga > 1.6: 
                    venue_multiplier = 1.04 # Competent visitor (Reduced from 1.08)
                elif v_gf < 0.8:
                    venue_multiplier = 0.96 # Poor away form (Reduced penalty from 0.92)
        
        # FIX 4 (Part 2): Consistency modulates venue effect instead of stacking independently
        if own_venue_stats:
            _, _, v_consist = own_venue_stats
            venue_multiplier = 1.0 + (venue_multiplier - 1.0) * v_consist
            
        atk_strength *= venue_multiplier
            
        # FIX 5: Collapsed league table into single rank_factor formula
        if league_table and team in league_table:
            stats = league_table[team]
            n_teams = max(len(league_table), 20)
            rank_factor = 1.15 - (stats['rank'] / n_teams * 0.40)
            rank_factor = max(0.80, min(1.15, rank_factor))
            
            # Fold PPG signal smoothly into rank_factor
            if is_home and stats['h_gp'] > 0:
                h_ppg = stats['h_pts'] / stats['h_gp']
                ppg_adj = (h_ppg - 1.5) / 1.5 * 0.04  # range: approx -0.04 to +0.04
                rank_factor = max(0.80, min(1.15, rank_factor + ppg_adj))
            elif not is_home and stats['a_gp'] > 0:
                a_ppg = stats['a_pts'] / stats['a_gp']
                ppg_adj = (a_ppg - 1.0) / 1.5 * 0.03
                rank_factor = max(0.80, min(1.12, rank_factor + ppg_adj))
            
            atk_strength *= rank_factor
            
            # Symmetric inverse for defense
            def_rank_factor = 2.0 - rank_factor
            def_strength *= def_rank_factor
        
        # Guard rails
        atk_strength = max(0.5, min(atk_strength, 2.0)) # Increased cap to 2.0 to allow for high performers
        def_strength = max(0.4, min(def_strength, 2.0))
                
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
        """Runs Advanced Analytical Pipeline: Dixon-Coles Correction + Exact Matrix Probabilities.
        FIX 9:  H2H adjustment applied before matrix build.
        FIX 10: Market odds no longer contaminate λ values.
        FIX 11: Monte Carlo replaced with exact analytical probabilities."""
        avg_h_xg, avg_a_xg = self.get_league_stats(df)
        league_table = self.get_league_table(df)
        elo = self.calculate_elo(df)

        # Pre-fetch Venue Stats for Cross-Reference
        h_v_gf, h_v_ga, h_v_consist = self.get_venue_form(home_team, df, True)
        a_v_gf, a_v_ga, a_v_consist = self.get_venue_form(away_team, df, False)

        # Pass Opponent's Venue Stats AND Own Venue Stats to calculation
        h_atk, h_def, h_ped, h_val, h_clin = self.calculate_strength(
            home_team, df, True, avg_h_xg, avg_a_xg, league_table, elo, is_ucl, 
            own_venue_stats=(h_v_gf, h_v_ga, h_v_consist),
            opponent_venue_stats=(a_v_gf, a_v_ga, a_v_consist)
        )
        
        a_atk, a_def, a_ped, a_val, a_clin = self.calculate_strength(
            away_team, df, False, avg_h_xg, avg_a_xg, league_table, elo, is_ucl,
            own_venue_stats=(a_v_gf, a_v_ga, a_v_consist),
            opponent_venue_stats=(h_v_gf, h_v_ga, h_v_consist)
        )

        l_home = h_atk * a_def * avg_h_xg * 1.08
        l_away = a_atk * h_def * avg_a_xg * 0.92

        # FIX 10: Market odds removed from λ calculation.
        # live_odds is still accepted and passed through in the result dict
        # for use by the accumulator optimizer and UI edge calculation.

        # FIX 9: H2H adjustment BEFORE matrix build so it actually affects probabilities
        h2h = self.get_h2h_history(home_team, away_team, df)
        if h2h:
            n_games = h2h['h_wins'] + h2h['a_wins'] + h2h['draws']
            h2h_scale = 0.01 + 0.03 * (n_games / 5)
            # At 5 games: scale = 0.04 (×1.04 max)
            # At 1 game:  scale = 0.016 (barely moves λ)
            if h2h['h_wins'] > h2h['a_wins']:
                l_home *= (1.0 + h2h_scale)
            elif h2h['a_wins'] > h2h['h_wins']:
                l_away *= (1.0 + h2h_scale)

        # FIX 1: 12×12 matrix (k=0..11)
        h_pmf = poisson_pmf(np.arange(12), l_home)
        a_pmf = poisson_pmf(np.arange(12), l_away)
        matrix = np.outer(h_pmf, a_pmf)

        for i in range(2):
            for j in range(2):
                matrix[i, j] *= self.tau_adjustment(i, j, l_home, l_away)

        # FIX 11: Exact analytical probabilities from the matrix (no Monte Carlo)
        h_win = np.sum(np.tril(matrix, -1))
        draw_prob = np.sum(np.diag(matrix))
        a_win = np.sum(np.triu(matrix, 1))

        # Analytically derived expected points
        h_xp = (h_win * 3) + (draw_prob * 1)
        a_xp = (a_win * 3) + (draw_prob * 1)

        return {
            "home": home_team, "away": away_team,
            "l_home": l_home, "l_away": l_away,
            "h_win": h_win, "draw": draw_prob, "a_win": a_win,
            # FIX 11: mc_ keys point to matrix-derived values (no frontend breakage)
            "mc_h_win": h_win, "mc_draw": draw_prob, "mc_a_win": a_win,
            "h_xp": h_xp, "a_xp": a_xp,
            "mc_h_xp": h_xp, "mc_a_xp": a_xp,
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
            "is_ucl": is_ucl,
            "live_odds": live_odds
        }

    def get_recommendations(self, res):
        """Statistical engine providing two high-probability tactical recommendations.
        FIX 11: All mc_ keys now reference exact matrix-derived probabilities."""
        h_xg, a_xg = res['l_home'], res['l_away']
        
        # 1. Main Prediction (The Core Direction)
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

        # 2. Safety Recommendation (The High Probability Conservative Choice)
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
            "primary": {"pick": primary_pick, "insight": primary_insight, "type": "Main Prediction"},
            "safety": {"pick": safety_pick, "insight": safety_insight, "type": "Safety Recommendation"}
        }


# src/ui.py
import streamlit as st
from .styles import APP_CSS
from .data_service import DataService
from .engine import MatchPredictor

def inject_css():
    """Injects the global CSS theme."""
    st.markdown(APP_CSS, unsafe_allow_html=True)

def render_match_header(res, match_date):
    """Renders the FotMob-inspired match header with resilient asset fallback."""
    logo_h = DataService.fetch_team_logo(res['home'])
    logo_a = DataService.fetch_team_logo(res['away'])
    
    # CSS Badge Fallback for speed
    def get_badge(logo, name):
        if logo: return f"<img src='{logo}' style='height: 70px; width: 70px; object-fit: contain;' loading='lazy'>"
        return f"<div style='height: 70px; width: 70px; border-radius: 50%; background: var(--hero-grad); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 900; color: white; margin: 0 auto;'>{name[0]}</div>"

    img_h = get_badge(logo_h, res['home'])
    img_a = get_badge(logo_a, res['away'])

    # TIGHTENED HTML STRING (No double newlines to prevent markdown leakage)
    ucl_dna_h = f'<div style="font-size: 0.65rem; color: #3b82f6; font-weight: 800; background: rgba(59,130,246,0.1); padding: 2px 6px; border-radius: 4px;">UCL DNA</div>' if (res.get('is_ucl') and res.get('ped_h', 1)>1.05) else ''
    ucl_dna_a = f'<div style="font-size: 0.65rem; color: #3b82f6; font-weight: 800; background: rgba(59,130,246,0.1); padding: 2px 6px; border-radius: 4px;">UCL DNA</div>' if (res.get('is_ucl') and res.get('ped_a', 1)>1.05) else ''

    standings_h = f'<div style="font-size: 0.7rem; color: #94a3b8; font-weight: 600; margin-top: 2px;">Pos. {res["rank_h"]} • {res["pts_h"]} pts</div>'
    standings_a = f'<div style="font-size: 0.7rem; color: #94a3b8; font-weight: 600; margin-top: 2px;">Pos. {res["rank_a"]} • {res["pts_a"]} pts</div>'

    header_html = f"""<div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;"><div style="text-align: center; flex: 1; min-width: 120px;">{img_h}<div style="font-weight: 800; font-size: 1.2rem; margin-top: 10px;">{res['home']}</div>{standings_h}<div style="display: flex; justify-content: center; gap: 5px; margin-top: 4px;">{ucl_dna_h}</div></div><div style="text-align: center;"><div style="font-size: 1rem; font-weight: 300; color: #94a3b8;">VS</div><div style="font-size: 0.8rem; font-weight: 600; background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 20px; margin-top: 5px;">{match_date.strftime('%H:%M')}</div></div><div style="text-align: center; flex: 1; min-width: 120px;">{img_a}<div style="font-weight: 800; font-size: 1.2rem; margin-top: 10px;">{res['away']}</div>{standings_a}<div style="display: flex; justify-content: center; gap: 5px; margin-top: 4px;">{ucl_dna_a}</div></div></div>"""
    st.markdown(header_html, unsafe_allow_html=True)

def render_outcome_bar(res):
    """Renders the Monte Carlo powered H/D/A horizontal bar with xP."""
    bar_html = f"""<div class="outcome-bar"><div class="outcome-item"><div class="badge-circle" style="color: #60a5fa;">H</div><div class="outcome-value">{res['mc_h_win']:.0%}</div><div style="font-size: 0.65rem; opacity: 0.6;">{res['h_xp']:.1f} xP</div></div><div class="outcome-item"><div class="badge-circle" style="color: #94a3b8;">D</div><div class="outcome-value">{res['mc_draw']:.0%}</div></div><div class="outcome-item"><div class="badge-circle" style="color: #f87171;">A</div><div class="outcome-value">{res['mc_a_win']:.0%}</div><div style="font-size: 0.65rem; opacity: 0.6;">{res['a_xp']:.1f} xP</div></div></div>"""
    st.markdown(bar_html, unsafe_allow_html=True)

def render_top_picks(recs):
    """Renders the High-Value Hero Cards in a responsive 3-pillar layout."""
    st.markdown("### 🏆 Tactical Insights")
    
    types = [
        ('primary', 'hero-grad', '#60a5fa'),
        ('tactical', 'value-grad', '#f59e0b'),
        ('safety', 'safety-grad', '#94a3b8')
    ]
    
    cards_html = ""
    for key, grad, glow in types:
        data = recs[key]
        cards_html += f"""<div class="hero-card" style="background: var(--{grad}); border: 1px solid {glow}44; min-height: 180px; display: flex; flex-direction: column; justify-content: space-between;"><div><div style="font-size: 0.65rem; text-transform: uppercase; opacity: 0.8; font-weight: 800; letter-spacing: 0.05em;">{data['type']}</div><div style="font-size: 1.1rem; font-weight: 800; margin: 8px 0; line-height: 1.2;">{data['pick']}</div></div><div><div class="confidence-glow" style="width: {'90%' if key=='primary' else '75%' if key=='tactical' else '60%'}; background: {glow}; box-shadow: 0 0 10px {glow};"></div><div style="font-size: 0.8rem; font-style: italic; opacity: 0.9; margin-top: 8px;">"{data['insight']}"</div></div></div>"""
    
    st.markdown(f'<div class="picks-grid">{cards_html}</div>', unsafe_allow_html=True)

def render_analytics(res):
    """Renders Goals, Simulation, and Tactical indicators."""
    with st.expander("🔬 Elite Simulation & Tactical History", expanded=True):
        # 1. Tactical Potency Gauge (xG Momentum)
        h_xg, a_xg = res['l_home'], res['l_away']
        
        def get_gauge_html(val, color, team_name):
            segments = 15
            active = max(1, min(int(val / 3 * segments), segments))
            seg_html = ""
            for i in range(segments):
                is_active = i < active
                active_class = "segment-active" if is_active else ""
                pulse_class = "segment-pulse" if i == active - 1 else ""
                style = f"color: {color}; background: {color if is_active else 'var(--badge-bg)'};"
                seg_html += f'<div class="potency-segment {active_class} {pulse_class}" style="{style}"></div>'
            
            return f'<div class="potency-label-row"><div class="potency-title">{team_name} Potency</div><div class="potency-value" style="color: {color};">{val:.2f} xG</div></div><div class="potency-gauge-container">{seg_html}</div>'

        h_color = "#10b981" if h_xg > 1.8 else "#f59e0b" if h_xg > 1.2 else "#60a5fa"
        a_color = "#10b981" if a_xg > 1.8 else "#f59e0b" if a_xg > 1.2 else "#f87171"

        st.markdown(get_gauge_html(h_xg, h_color, res['home']), unsafe_allow_html=True)
        st.markdown(get_gauge_html(a_xg, a_color, res['away']), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. Probabilities
        st.markdown("<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 10px; text-align: center;'>", unsafe_allow_html=True)
        for label, prob in [("BTTS", res['btts']), ("OVER 2.5", res['over25']), ("UNDER 2.5", res['under25'])]:
            st.markdown(f"<div><div style='font-size:0.7rem; color:#94a3b8;'>{label}</div><div style='font-size:1.1rem; font-weight:800;'>{prob:.0%}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 4. AI Tactical Game Script
        st.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 800; margin-top: 20px;">AI Tactical Game Script</div>', unsafe_allow_html=True)
        ped_h, ped_a = res.get('ped_h', 1.0), res.get('ped_a', 1.0)
        
        heritage = ""
        if res.get('is_ucl'):
            if ped_h > 1.05 and ped_a <= 1.0:
                heritage = f"The immense European pedigree of {res['home']} will be a decisive factor tonight."
            elif ped_a > 1.05 and ped_h <= 1.0:
                heritage = f"{res['away']} brings their 'Champions League DNA' to this fixture."

        if res['l_home'] > res['l_away'] + 1.0:
            flow = f"Total dominance predicted for {res['home']}."
        elif abs(res['l_home'] - res['l_away']) < 0.3:
            flow = "A tactical stalemate is brewing. Expect a midfield battle."
        else:
            flow = f"A controlled tactical performance likely. {res['home']} holds the advantage."
            
        st.markdown(f'<div class="script-box">"{heritage} {flow}"</div>', unsafe_allow_html=True)
        st.caption(f"⚡ High-precision tactical analysis applied for elite accuracy.")

def render_advanced_intelligence(res, df):
    """Renders high-tech tactical modules: Radar Comparison, Momentum Pulse, and Game Script."""
    st.markdown("### 🧬 Advanced Match Intelligence")
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # 1. Tactical Radar Comparison
        st.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 800; margin-bottom: 15px;">Advanced Competitive Indicators</div>', unsafe_allow_html=True)
        
        metrics = [
            ("UCL DNA (Heritage)", res.get('ped_h', 1.0)/2, res.get('ped_a', 1.0)/2),
            ("Economic Multiplier", res.get('val_h', 1.0)/2, res.get('val_a', 1.0)/2),
            ("Attack Conversion", res['l_home']/(res['l_home']+res['l_away']), res['l_away']/(res['l_home']+res['l_away'])),
            ("Win Probability", res['mc_h_win'], res['mc_a_win'])
        ]
        
        for label, h_val, a_val in metrics:
            h_pct = max(5, min(95, h_val * 100))
            a_pct = max(5, min(95, a_val * 100))
            st.markdown(f'<div style="margin-bottom: 12px;"><div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 4px; flex-wrap: wrap; gap: 4px;"><span style="color: #60a5fa;">{res["home"]}</span><span>{label}</span><span style="color: #f87171;">{res["away"]}</span></div><div class="radar-bar-base"><div class="radar-bar-h" style="width: {h_pct}%;"></div><div style="width: 2px; background: rgba(0,0,0,0.3);"></div><div class="radar-bar-a" style="width: {a_pct}%;"></div></div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. Momentum Pulse (Last 6 Games)
        col1, col2 = st.columns(2)
        
        def get_momentum_dots(team, df):
            played = df[((df['Home'] == team) | (df['Away'] == team)) & df['Score'].notna()].tail(6)
            dots = ""
            for _, row in played.iterrows():
                h_score, a_score = map(int, row['Score'].split('-'))
                if row['Home'] == team:
                    res_char = 'win' if h_score > a_score else 'draw' if h_score == a_score else 'loss'
                else:
                    res_char = 'win' if a_score > h_score else 'draw' if h_score == a_score else 'loss'
                dots += f'<span class="momentum-dot dot-{res_char}"></span>'
            return dots

        with col1:
            st.markdown(f'<div style="font-size: 0.7rem; color: #94a3b8; margin-bottom: 5px;">{res["home"]} Momentum</div><div>{get_momentum_dots(res["home"], df)}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div style="font-size: 0.7rem; color: #94a3b8; margin-bottom: 5px; text-align: right;">{res["away"]} Momentum</div><div style="text-align: right;">{get_momentum_dots(res["away"], df)}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. Predicted Game Script (Enhanced with UCL Intelligence)
        st.markdown('<div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 800;">AI Tactical Game Script</div>', unsafe_allow_html=True)
        
        ped_h, ped_a = res.get('ped_h', 1.0), res.get('ped_a', 1.0)
        
        script = ""
        # UCL Heritage logic
        if res.get('is_ucl'):
            if ped_h > 1.05 and ped_a <= 1.0:
                heritage_context = f"The immense European pedigree of {res['home']} will be a decisive factor tonight. History tends to repeat itself on these nights."
            elif ped_a > 1.05 and ped_h <= 1.0:
                heritage_context = f"{res['away']} brings their 'Champions League DNA' to this fixture, which often overcomes domestic form disparities."
            elif ped_h > 1.05 and ped_a > 1.05:
                heritage_context = "Two European giants collide. This is a clash of legacies where the weight of history is balanced between both sides."
            else: heritage_context = ""
        else: heritage_context = ""

        # Score & Flow logic
        if res['l_home'] > 2.0 and res['l_away'] > 1.5:
            flow = "A high-octane encounter expected. Both sides will trade blows early, with clinical finishing deciding the final 20 minutes."
        elif res['l_home'] > res['l_away'] + 1.0:
            flow = f"Total dominance predicted for {res['home']}. They will likely monopolize possession and force {res['away']} into a deep block."
        elif abs(res['l_home'] - res['l_away']) < 0.3:
            flow = "A tactical stalemate is brewing. Expect a midfield battle where the first goal will likely be the knockout blow."
        else:
            flow = f"A controlled tactical performance likely. {res['home']} holds the advantage, but {res['away']}'s transition threat is significant."
            
        final_script = f"{heritage_context} {flow}".strip()
        st.markdown(f'<div class="script-box">"{final_script}"</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_match_results(res, match_date, df=None):
    """Orchestrates the rendering of the advanced analytical bundle."""
    st.markdown("---")
    recs = MatchPredictor().get_recommendations(res)
    
    render_match_header(res, match_date)
    st.markdown(f'<div class="predicted-score-box"><div style="font-size: 0.8rem; text-transform: uppercase; color: #10b981; letter-spacing: 0.1em; font-weight: 800;">PREDICTED SCORE</div><div style="font-size: 2.3rem; font-weight: 800; color: var(--app-text);">{res["predicted_score"]}</div></div>', unsafe_allow_html=True)

    render_outcome_bar(res)
    render_top_picks(recs)
    render_analytics(res)

def render_tactical_tabs():
    """Renders the horizontal Masterboard tabs at the top."""
    tabs = [
        ("Champions League", "🏆"), ("Premier League", "🦁"), ("La Liga", "🇪🇸"),
        ("Serie A", "🇮🇹"), ("Bundesliga", "🇩🇪"), ("Ligue 1", "🇫🇷"), ("Russian Premier League", "🇷🇺")
    ]
    
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    for name, icon in tabs:
        if st.button(f"{icon} {name}", key=f"tab_{name}"):
            st.session_state.selected_league = name
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

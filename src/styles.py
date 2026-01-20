# src/styles.py

APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    :root {
        /* Dark Mode */
        --app-bg: #0f172a;
        --app-text: #f8fafc;
        --app-subtext: #94a3b8;
        --card-bg: rgba(255, 255, 255, 0.05);
        --card-border: rgba(255, 255, 255, 0.12);
        --card-shadow: rgba(0, 0, 0, 0.5);
        --outcome-bg: rgba(255, 255, 255, 0.03);
        --badge-bg: rgba(255, 255, 255, 0.08);
        --score-bg: rgba(16, 185, 129, 0.15);
        --score-border: rgba(16, 185, 129, 0.3);
        --hero-grad: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        --value-grad: linear-gradient(135deg, #422006 0%, #0f172a 100%);
        --safety-grad: linear-gradient(135deg, #4b5563 0%, #1f2937 100%);
    }

    @media (prefers-color-scheme: light) {
        :root {
            /* Light Mode */
            --app-bg: #ffffff;
            --app-text: #0f172a;
            --app-subtext: #475569;
            --card-bg: #f8fafc;
            --card-border: #e2e8f0;
            --card-shadow: rgba(0, 0, 0, 0.05);
            --outcome-bg: #f1f5f9;
            --badge-bg: #e2e8f0;
            --score-bg: #ecfdf5;
            --score-border: #10b98144;
            --hero-grad: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            --value-grad: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            --safety-grad: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        }
    }

    /* Strict Text Visibility Overrides */
    .stApp, .stMarkdown, p, span, div, h1, h2, h3, h4, h5, h6, label, .stCaption, .stSelectbox label {
        color: var(--app-text) !important;
    }
    
    * { 
        font-family: 'Outfit', sans-serif; 
        box-sizing: border-box;
    }
    .stApp { 
        background: var(--app-bg); 
        max-width: 100vw;
        overflow-x: hidden;
    }
    
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px var(--card-shadow);
        margin-bottom: 20px;
        width: 100%;
    }

    .hero-card {
        background: var(--hero-grad);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 10px 25px var(--card-shadow);
        border: 1px solid var(--card-border);
    }
    
    .hero-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        pointer-events: none;
    }

    .confidence-glow {
        height: 6px;
        background: #10b981;
        border-radius: 3px;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
        margin: 10px 0;
    }
    
    /* Match Outcome Horizontal Bar */
    .outcome-bar {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background: var(--outcome-bg);
        border-radius: 12px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid var(--card-border);
    }
    
    .outcome-item { text-align: center; }
    .outcome-value { font-size: 1.1rem; font-weight: 800; color: var(--app-text); }
    
    .badge-circle {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        border-radius: 50%;
        background: var(--badge-bg);
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 4px;
        color: var(--app-text);
    }

    /* Tactical Potency Gauge (Segmented) */
    .potency-label-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .potency-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.05em;
        color: var(--app-subtext);
    }
    .potency-value {
        font-family: 'monospace';
        font-weight: 800;
        color: #10b981;
    }
    .potency-gauge-container {
        display: flex;
        gap: 3px;
        width: 100%;
        height: 14px;
        margin-bottom: 12px;
    }
    .potency-segment {
        flex: 1;
        background: var(--badge-bg);
        border-radius: 2px;
        transition: all 0.3s ease;
    }
    .segment-active {
        box-shadow: 0 0 10px currentColor;
    }
    .segment-pulse {
        animation: energy-pulse 1.5s infinite alternate;
    }
    @keyframes energy-pulse {
        from { opacity: 0.6; filter: brightness(1); }
        to { opacity: 1; filter: brightness(1.5); }
    }

    /* Keep Legacy Heat Meter for fallback or secondary use if needed */
    .heat-meter-container {
        width: 100%;
        height: 12px;
        background: var(--badge-bg);
        border-radius: 6px;
        overflow: hidden;
        margin: 8px 0;
    }
    .heat-meter-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .heat-high { background: #10b981; box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }
    .heat-mid { background: #f59e0b; }
    .heat-low { background: #ef4444; }

    /* High-Tech Tactical Visuals */
    .tactical-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        margin-top: 15px;
    }
    .radar-bar-container {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .radar-bar-base {
        height: 6px;
        background: var(--badge-bg);
        border-radius: 3px;
        overflow: hidden;
        display: flex;
    }
    .radar-bar-h { background: #60a5fa; height: 100%; }
    .radar-bar-a { background: #f87171; height: 100%; }
    
    .momentum-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin: 0 4px;
    }
    .dot-win { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .dot-draw { background: #94a3b8; }
    .dot-loss { background: #ef4444; }

    .script-box {
        border-left: 3px solid #60a5fa;
        background: var(--outcome-bg);
        padding: 12px 15px;
        margin: 15px 0;
        font-size: 0.85rem;
        line-height: 1.6;
        font-style: italic;
        border-radius: 0 8px 8px 0;
        color: var(--app-subtext);
    }

    /* Master Tab Navigation */
    .tab-container {
        display: flex;
        gap: 10px;
        padding: 10px 0;
        overflow-x: auto;
        margin-bottom: 25px;
        border-bottom: 1px solid var(--card-border);
        scrollbar-width: none;
    }
    .tab-container::-webkit-scrollbar { display: none; }
    
    .tab-item {
        padding: 10px 20px;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        cursor: pointer;
        white-space: nowrap;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        gap: 8px;
        opacity: 0.7;
    }
    .tab-item:hover {
        opacity: 1;
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.05);
        transform: translateY(-2px);
    }
    .tab-active {
        background: var(--hero-grad) !important;
        color: white !important;
        opacity: 1;
        border-color: transparent;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .tab-icon { font-size: 1.1rem; }
</style>
"""

const API_BASE = '/api';

async function loadAccumulator() {
    const loading = document.getElementById('loading-state');
    const noBet = document.getElementById('no-bet-state');
    const content = document.getElementById('result-content');

    try {
        // Parallel fetch for speed
        const [picksRes, accaRes] = await Promise.all([
            fetch(`${API_BASE}/top-picks`),
            fetch(`${API_BASE}/optimize-acca`)
        ]);

        const picksData = await picksRes.json();
        const accaData = await accaRes.json();

        loading.style.display = 'none';

        if (accaData.accumulator) {
            // Scenario 1: Optimal Acca Found
            content.style.display = 'block';
            renderAccumulator(accaData, picksData.picks);
        } else {
            // Scenario 2: Market Efficiency (No Acca) but we have Top Picks
            if (picksData.picks && picksData.picks.length > 0) {
                content.style.display = 'block';
                renderTopPicks(picksData.picks, picksData.statistical_rationale);

                // Show warning banner
                const statusBanner = document.getElementById('status-banner');
                statusBanner.style.display = 'block';
                statusBanner.style.background = 'rgba(255, 215, 0, 0.15)';
                statusBanner.style.border = '1px solid rgba(255, 215, 0, 0.3)';
                statusBanner.innerHTML = `
                    <div class="status-icon">⚠️</div>
                    <div class="status-text">
                        <strong>Market Efficiency Alert</strong><br>
                        Stable accumulator not found. Top picks maintained.
                    </div>
                 `;

                const rationaleBox = content.querySelector('.rationale-box');
                if (rationaleBox && accaData.message) {
                    rationaleBox.textContent = `"${accaData.message}"`;
                }
            } else {
                // Scenario 3: Total Failure
                content.style.display = 'none';
                noBet.style.display = 'block';
            }
        }
    } catch (e) {
        console.error("Accumulator load failed", e);
        loading.style.display = 'none';
        noBet.style.display = 'block';
        noBet.innerHTML = '<div class="no-bet-title">SYNC FAILURE</div><p>Intelligence node is currently unreachable.</p>';
    }
}

function renderTopPicks(picks, rationale) {
    const content = document.getElementById('result-content');
    const legsContainer = content.querySelector('.legs-container');
    const ticketTitle = content.querySelector('.ticket-title');
    const rationaleBox = content.querySelector('.rationale-box');
    const ticketId = content.querySelector('.ticket-id');

    ticketTitle.textContent = "TOP PICKS (LIVE FEED)";
    ticketId.textContent = `QUANT-SELECT-${new Date().toISOString().slice(0, 10)}`;

    legsContainer.innerHTML = picks.map(pick => `
        <div class="leg-item" style="border-left: 3px solid #ffd700;">
            <div class="leg-main">
                <span class="leg-fixture">${pick.fixture} ${getDateBadge(pick.match_date)}</span>
                <span class="leg-selection">${pick.selection}</span>
            </div>
            <div class="leg-meta">
                <span class="leg-odds">@${pick.decimal_odds.toFixed(2)}</span>
                <span class="leg-edge" style="color: #4ade80;">Edge: +${pick.edge_percent}%</span>
            </div>
        </div>
    `).join('');

    rationaleBox.textContent = rationale ? `"${rationale}"` : "Real-time alpha screening compliant with Kelly criterion.";
}

function renderAccumulator(data, existingTopPicks) {
    const acca = data.accumulator;
    const content = document.getElementById('result-content');
    const legsContainer = content.querySelector('.legs-container');
    const ticketTitle = content.querySelector('.ticket-title');
    const rationaleBox = content.querySelector('.rationale-box');
    const ticketId = content.querySelector('.ticket-id');

    ticketTitle.textContent = "ANALYST'S ACCUMULATOR 🎯";
    ticketId.textContent = `QUANT-ACCA-${new Date().toISOString().slice(0, 10)}`;
    rationaleBox.textContent = `"${data.statistical_rationale}"`;

    const legsHtml = acca.legs.map((leg, index) => `
        <div class="leg-item">
            <div class="leg-number">${index + 1}</div>
            <div class="leg-main">
                <span class="leg-fixture">${leg.fixture} ${getDateBadge(leg.match_date)}</span>
                <span class="leg-selection">${leg.selection}</span>
            </div>
            <div class="leg-meta">
                <span class="leg-odds">@${leg.decimal_odds.toFixed(2)}</span>
                <span class="leg-edge">+${leg.edge_percent}% Edge</span>
            </div>
        </div>
    `).join('');

    const statsHtml = `
        <div class="footer-stats">
            <div class="stat-item">
                <div class="stat-label">Total Odds</div>
                <div class="stat-val">${acca.actual_odds.toFixed(2)}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Expected Value</div>
                <div class="stat-val">+${acca.expected_value_percent}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">True Probability</div>
                <div class="stat-val">${(acca.combined_probability * 100).toFixed(1)}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Fractional Stake</div>
                <div class="stat-val">${acca.kelly_fraction} Units</div>
            </div>
        </div>
    `;

    legsContainer.innerHTML = legsHtml + statsHtml;

    if (existingTopPicks && existingTopPicks.length > 0) {
        renderAlternativePicks(existingTopPicks, legsContainer, rationaleBox);
    }
}

function renderAlternativePicks(picks, container, rationaleBox) {
    let secondaryDiv = document.getElementById('secondary-picks');
    if (!secondaryDiv) {
        secondaryDiv = document.createElement('div');
        secondaryDiv.id = 'secondary-picks';
        secondaryDiv.style.marginTop = '2rem';
        secondaryDiv.style.borderTop = '1px solid rgba(255,255,255,0.1)';
        secondaryDiv.style.paddingTop = '1rem';
        secondaryDiv.innerHTML = `
            <div class="ticket-title" style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;">
                ALTERNATIVE HIGH-VALUE SELECTIONS
            </div>
            <div class="legs-container-secondary legs-container"></div>
        `;
        container.parentNode.insertBefore(secondaryDiv, rationaleBox);
    }

    secondaryDiv.querySelector('.legs-container-secondary').innerHTML = picks.map(pick => `
        <div class="leg-item" style="border-left: 3px solid #ffd700;">
            <div class="leg-main">
                <span class="leg-fixture">${pick.fixture}</span>
                <span class="leg-selection">${pick.selection}</span>
            </div>
            <div class="leg-meta">
                <span class="leg-odds">@${pick.decimal_odds.toFixed(2)}</span>
                <span class="leg-edge" style="color: #4ade80;">Edge: +${pick.edge_percent}%</span>
            </div>
        </div>
    `).join('');
}

function getDateBadge(dateStr) {
    const todayStr = new Date().toISOString().slice(0, 10);
    if (!dateStr || dateStr === todayStr) return '';
    const d = new Date(dateStr);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const diffDays = Math.ceil((d - today) / (1000 * 60 * 60 * 24));
    let label = diffDays === 1 ? "TOMORROW" : new Intl.DateTimeFormat('en', { weekday: 'short' }).format(d).toUpperCase();
    return `<span style="font-size: 0.6em; background: rgba(59, 130, 246, 0.2); color: #60a5fa; padding: 2px 6px; border-radius: 4px; margin-left: 8px; vertical-align: middle;">${label}</span>`;
}

// Run on load
document.addEventListener('DOMContentLoaded', loadAccumulator);

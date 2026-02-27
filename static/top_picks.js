const API_BASE = '/api';
let currentDateOffset = 0;

async function loadTopPicks(offset = 0) {
    currentDateOffset = offset;
    const loading = document.getElementById('loading-state');
    const noBet = document.getElementById('no-bet-state');
    const content = document.getElementById('result-content');
    const nextDayBtn = document.getElementById('next-day-btn');

    loading.style.display = 'block';
    noBet.style.display = 'none';
    content.style.display = 'none';
    if (nextDayBtn) nextDayBtn.style.display = 'none';

    try {
        const res = await fetch(`${API_BASE}/top-picks?offset=${offset}`);
        const data = await res.json();

        loading.style.display = 'none';

        if (data.picks && data.picks.length > 0) {
            content.style.display = 'block';
            renderTopPicks(data.picks, data.statistical_rationale);
            updateNavigation(offset);
        } else {
            content.style.display = 'none';
            noBet.style.display = 'block';
            const reason = noBet.querySelector('p');
            if (reason) reason.textContent = data.message || "Market Too Efficient (No Value Found)";

            if (nextDayBtn && offset < 5) {
                nextDayBtn.style.display = 'inline-flex';
                const d = new Date();
                d.setDate(d.getDate() + offset + 1);
                const options = { weekday: 'short', month: 'short', day: 'numeric' };
                const nextDateStr = new Intl.DateTimeFormat('en', options).format(d).toUpperCase();
                nextDayBtn.innerHTML = `REVEAL ${nextDateStr} &nbsp; →`;
            }
        }
    } catch (e) {
        console.error("Top picks load failed", e);
        loading.style.display = 'none';
        noBet.style.display = 'block';
        noBet.innerHTML = '<div class="no-bet-title">SYNC FAILURE</div><p>Intelligence node is currently unreachable.</p>';
    }
}

function renderTopPicks(picks, rationale) {
    const content = document.getElementById('result-content');
    const legsContainer = content.querySelector('.legs-container');
    const rationaleBox = content.querySelector('.rationale-box');
    const ticketId = content.querySelector('.ticket-id');

    if (ticketId) ticketId.textContent = `TOP-PICKS-${new Date().toISOString().slice(0, 10)}`;

    legsContainer.innerHTML = picks.map((pick, i) => `
        <div class="leg-item" style="border-left: 3px solid #ffd700;">
            <div class="leg-number">${i + 1}</div>
            <div class="leg-main">
                <span class="leg-fixture">${pick.fixture} ${getDateBadge(pick.match_date)}</span>
                <span class="leg-selection">${pick.selection}</span>
            </div>
            <div class="leg-meta">
                ${pick.decimal_odds ? `<span class="leg-odds">@${parseFloat(pick.decimal_odds).toFixed(2)}</span>` : ''}
                <span class="leg-edge" style="color: ${pick.edge_percent > 0 ? '#4ade80' : '#fbbf24'};">${pick.edge_percent > 0 ? 'Edge: +' + pick.edge_percent + '%' : 'Prob: ' + (pick.true_prob * 100).toFixed(0) + '%'}</span>
            </div>
        </div>
    `).join('');

    if (rationaleBox) {
        rationaleBox.textContent = rationale ? `"${rationale}"` : "Daily curated high-value selections.";
    }
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

function searchNextDay() {
    loadTopPicks(currentDateOffset + 1);
}

function searchPrevDay() {
    if (currentDateOffset > 0) {
        loadTopPicks(currentDateOffset - 1);
    }
}

function updateNavigation(offset) {
    const prevBtn = document.getElementById('prev-day-btn-results');
    const nextBtn = document.getElementById('next-day-btn-results');
    const dateDisplay = document.getElementById('current-date-display');

    if (prevBtn) {
        prevBtn.style.visibility = offset > 0 ? 'visible' : 'hidden';
        prevBtn.disabled = offset <= 0;
    }
    if (nextBtn) {
        nextBtn.style.visibility = offset < 6 ? 'visible' : 'hidden'; // Limit to 6 days ahead
        nextBtn.disabled = offset >= 6;
    }

    if (dateDisplay) {
        const d = new Date();
        d.setDate(d.getDate() + offset);
        const options = { weekday: 'short', month: 'short', day: 'numeric' };
        const dateStr = new Intl.DateTimeFormat('en', options).format(d);
        dateDisplay.textContent = offset === 0 ? "TODAY" : dateStr.toUpperCase();
    }
}

// Run on load
document.addEventListener('DOMContentLoaded', () => loadTopPicks(0));

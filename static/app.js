const API_BASE = '/api';
let currentLeague = 'Premier League';
let leagues = [];
let historyData = [];
let globalFixtures = [];

// Initialize the application
document.addEventListener('DOMContentLoaded', init);

async function init() {
    try {
        loadHistory(); // Load from local storage

        // ⚡ SPEED GAIN: Only wait for the essential UI components
        const [leaguesRes, fixturesRes] = await Promise.all([
            fetch(`${API_BASE}/leagues`),
            fetch(`${API_BASE}/fixtures/${encodeURIComponent(currentLeague)}`)
        ]);

        leagues = await leaguesRes.json();
        renderTabs();

        const fixturesData = await fixturesRes.json();
        renderFixtures(fixturesData);

        // 🔍 DEFERRED LOADING: Fetch search data in background after UI is visible
        fetchGlobalFixtures();

    } catch (e) {
        console.error("Initialization failed", e);
        document.getElementById('fixture-list').innerHTML =
            '<div style="color: #f87171; padding: 20px;">Intelligence Synch Failure</div>';
    }
}

async function fetchGlobalFixtures() {
    try {
        console.log("Syncing global search database in background...");
        const res = await fetch(`${API_BASE}/fixtures/all`);
        const data = await res.json();
        globalFixtures = data.fixtures || [];
        console.log(`Global search database synced: ${globalFixtures.length} matches indexed.`);
    } catch (e) {
        console.warn("Global search sync failed, search may be limited to current league.", e);
    }
}

// --- NAVIGATION & MENU ---
function toggleMenu() {
    document.querySelector('.sidebar').classList.toggle('open');
    document.querySelector('.overlay').classList.toggle('open');
}

function switchView(viewName) {
    toggleMenu(); // Close menu
    if (viewName === 'tactical') {
        document.getElementById('view-tactical').style.display = 'block';
        document.getElementById('view-history').style.display = 'none';

        // Re-render fixtures if needed or just show the view
        // The view state is preserved in DOM, so just switching display is fine
    } else {
        document.getElementById('view-tactical').style.display = 'none';
        document.getElementById('view-history').style.display = 'block';
        renderHistory();
    }
}

// --- HISTORY LOGIC ---
function loadHistory() {
    const stored = localStorage.getItem('betly_history');
    if (stored) {
        historyData = JSON.parse(stored);
    }
}

function saveHistory() {
    localStorage.setItem('betly_history', JSON.stringify(historyData));
}

function addToHistory(pred, recs) {
    // Check if already exists (prevent dupes on re-click)
    const id = `${pred.home}-${pred.away}-${new Date().toDateString()}`;
    if (historyData.some(h => h.id === id)) return;

    const item = {
        id: id,
        date: new Date().toISOString(),
        home: pred.home,
        away: pred.away,
        league: currentLeague,
        predicted_score: pred.predicted_score,
        primary_pick: recs.primary.pick,
        result: '?',
        status: 'pending' // pending, won, lost
    };
    historyData.unshift(item); // Add to top
    if (historyData.length > 50) historyData.pop(); // Limit to 50
    saveHistory();
}

function renderHistory(filter = 'all') {
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;

    let filtered = historyData;
    if (filter !== 'all') {
        filtered = historyData.filter(h => h.status === filter);
    }

    if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; padding: 30px; color: var(--text-secondary);">No records found.</td></tr>';
        return;
    }

    tbody.innerHTML = filtered.map(item => `
        <tr>
            <td>${new Date(item.date).toLocaleDateString()}</td>
            <td><span style="font-weight:600; color: var(--text-primary);">${item.home}</span> vs ${item.away}</td>
            <td>
                <div style="font-weight:700;">${item.primary_pick}</div>
                <div style="font-size:0.75rem; opacity:0.7;">Score: ${item.predicted_score}</div>
            </td>
            <td>
                <div style="display:flex; gap:5px;">
                    <button onclick="updateStatus('${item.id}', 'won')" style="background:rgba(16,185,129,0.2); border:none; color:#10b981; cursor:pointer; padding:2px 6px; border-radius:4px;">✔</button>
                    <button onclick="updateStatus('${item.id}', 'lost')" style="background:rgba(244,63,94,0.2); border:none; color:#f43f5e; cursor:pointer; padding:2px 6px; border-radius:4px;">✘</button>
                </div>
            </td>
            <td><span class="status-badge status-${item.status}">${item.status}</span></td>
        </tr>
    `).join('');

    // Update filter buttons
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    // Ideally we'd set active class on the clicked button based on the filter argument
    // But for now keeping it simple as per original
}

function updateStatus(id, newStatus) {
    const item = historyData.find(h => h.id === id);
    if (item) {
        item.status = newStatus;
        item.result = newStatus === 'won' ? 'Correct' : 'Incorrect';
        saveHistory();
        renderHistory();
    }
}

function filterHistory(status) {
    renderHistory(status);
}

function clearHistory() {
    if (confirm("Clear all prediction history?")) {
        historyData = [];
        saveHistory();
        renderHistory();
    }
}

function renderTabs() {
    const container = document.getElementById('league-tabs');
    if (!container) return;

    container.innerHTML = leagues.map(l => `
        <div class="tab ${l.name === currentLeague ? 'active' : ''}" onclick="switchLeague('${l.name}')">
            ${l.name}
        </div>
    `).join('');
}

async function switchLeague(name) {
    currentLeague = name;
    renderTabs();
    document.getElementById('results-display').style.display = 'none';
    await loadFixtures(name);
}

async function loadFixtures(league) {
    const list = document.getElementById('fixture-list');
    list.innerHTML = Array(3).fill('<div class="fixture-item loading-shimmer" style="height: 70px;"></div>').join('');

    try {
        const res = await fetch(`${API_BASE}/fixtures/${encodeURIComponent(league)}`);
        const data = await res.json();
        renderFixtures(data);
    } catch (e) {
        list.innerHTML = '<div style="color: #f87171; padding: 20px;">Intelligence Synch Failure</div>';
    }
}

function handleSearch(query) {
    const list = document.getElementById('fixture-list');
    const q = query.toLowerCase().trim();

    // If search is empty, go back to currently selected league tab
    if (!q) {
        loadFixtures(currentLeague);
        return;
    }

    // Filter global fixtures by team name or league name
    const filtered = globalFixtures.filter(f =>
        f.Home.toLowerCase().includes(q) ||
        f.Away.toLowerCase().includes(q) ||
        (f.LeagueName && f.LeagueName.toLowerCase().includes(q))
    );

    renderFixtures({ fixtures: filtered }, true); // pass true to indicate it's a search result
}

function renderFixtures(data, isSearchResult = false) {
    const list = document.getElementById('fixture-list');

    // Filter out past fixtures
    const now = new Date();
    const upcomingFixtures = data.fixtures.filter(f => {
        const matchDate = new Date(f.DateTime);
        return matchDate >= now;
    });

    if (upcomingFixtures.length === 0) {
        list.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--text-secondary);">${isSearchResult ? 'No matches match your search.' : 'No upcoming matches found.'}</div>`;
        return;
    }

    list.innerHTML = upcomingFixtures.map(f => {
        const d = new Date(f.DateTime);
        // Format: DD/MM/YYYY HH:MM
        const day = String(d.getDate()).padStart(2, '0');
        const month = String(d.getMonth() + 1).padStart(2, '0');
        const year = d.getFullYear();
        const hours = String(d.getHours()).padStart(2, '0');
        const mins = String(d.getMinutes()).padStart(2, '0');
        const formatted = `${day}/${month}/${year} ${hours}:${mins}`;

        // Show league name badge if it's a search result from various leagues
        const leagueBadge = isSearchResult && f.LeagueName ? `<div style="font-size: 0.65rem; color: var(--accent-blue); font-weight: 800; text-transform: uppercase; margin-bottom: 2px;">${f.LeagueName}</div>` : '';

        return `
        <div class="fixture-item" onclick="getPrediction('${f.Home}', '${f.Away}', '${f.LeagueName || currentLeague}')">
            ${leagueBadge}
            <div class="f-teams">${f.Home} vs ${f.Away}</div>
            <div class="f-meta">${formatted}</div>
        </div>
    `}).join('');
}


async function getPrediction(home, away, leagueOverride = null) {
    const leagueToUse = leagueOverride || currentLeague;
    const display = document.getElementById('results-display');
    display.style.display = 'block';
    display.style.opacity = '0.5';

    try {
        // Ensure API call uses the current host and protocol for online compatibility
        const api_url = `${window.location.protocol}//${window.location.host}${API_BASE}/predict`;

        const res = await fetch(api_url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ home_team: home, away_team: away, league: leagueToUse })
        });

        if (!res.ok) {
            const errorBody = await res.text();
            throw new Error(`Server Error (${res.status}): ${errorBody}`);
        }

        const data = await res.json();
        renderResults(data);

        // Save to History automatically
        addToHistory(data.prediction, data.recommendations);

        display.style.opacity = '1';
        display.scrollIntoView({ behavior: 'smooth' });
    } catch (e) {
        console.error("Betly Intelligence Sync Error:", e);
        alert("Intelligence Sync Interrupted:\n" + e.message);
        display.style.opacity = '1';
    }
}

function renderResults(data) {
    if (!data || !data.prediction) {
        console.error("Invalid prediction data:", data);
        return;
    }
    const pred = data.prediction;
    const recs = data.recommendations;

    // Header
    const homeLogo = pred.home_logo || '';
    const awayLogo = pred.away_logo || '';

    document.getElementById('match-header').innerHTML = `
        <div class="team-box">
            <img src="${homeLogo}" class="team-logo" onerror="this.src='https://via.placeholder.com/80?text=${pred.home[0]}'">
            <div class="team-name">${pred.home}</div>
        </div>
        <div class="vs-divider">VS</div>
        <div class="team-box">
            <img src="${awayLogo}" class="team-logo" onerror="this.src='https://via.placeholder.com/80?text=${pred.away[0]}'">
            <div class="team-name">${pred.away}</div>
        </div>
    `;

    // Score
    document.getElementById('predicted-score-val').innerText = pred.predicted_score;

    // Probability Gauge
    const hPerc = (pred.mc_h_win * 100).toFixed(0);
    const dPerc = (pred.mc_draw * 100).toFixed(0);
    const aPerc = (pred.mc_a_win * 100).toFixed(0);

    const gaugeContainer = document.createElement('div');
    gaugeContainer.className = 'prob-gauge-container';
    gaugeContainer.innerHTML = `
        <div class="prob-gauge">
            <div class="gauge-h" style="width: ${hPerc}%"></div>
            <div class="gauge-d" style="width: ${dPerc}%"></div>
            <div class="gauge-a" style="width: ${aPerc}%"></div>
        </div>
        <div class="gauge-labels">
            <span class="label-h">${pred.home} ${hPerc}%</span>
            <span>DRAW ${dPerc}%</span>
            <span class="label-a">${aPerc}% ${pred.away}</span>
        </div>
        <div style="text-align: center; font-size: 0.65rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 5px; font-weight: 600;">win probability</div>
    `;

    const scoreBox = document.querySelector('.score-box');
    // Remove existing gauge if any
    const oldGauge = document.querySelector('.prob-gauge-container');
    if (oldGauge) oldGauge.remove();
    scoreBox.parentNode.insertBefore(gaugeContainer, scoreBox.nextSibling);

    // Picks
    const types = [
        { key: 'primary', grad: 'var(--hero-grad)', class: 'pick-primary' },
        { key: 'safety', grad: 'var(--safety-grad)', class: 'pick-safety' },
        { key: 'tactical', grad: 'var(--value-grad)', class: 'pick-risky' }
    ];

    document.getElementById('picks-grid').innerHTML = types.map(t => {
        const rec = recs[t.key];
        return `
            <div class="pick-card ${t.class}" style="background: ${t.grad}">
                <div>
                    <div class="pick-type">${rec.type}</div>
                    <div class="pick-val">${rec.pick}</div>
                </div>
                <div class="pick-insight">"${rec.insight}"</div>
            </div>
        `;
    }).join('');

    // H2H History Section
    const h2hContainer = document.getElementById('h2h-history');
    if (pred.h2h && pred.h2h.recent && pred.h2h.recent.length > 0) {
        const resultsHtml = pred.h2h.recent.map(m => `
            <div style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-radius: 12px; font-size: 0.8rem; display: flex; justify-content: space-between; align-items: center; border: 1px solid rgba(255,255,255,0.03);">
                <span style="color: var(--text-secondary); flex: 1;">${m.DateTime.split(' ')[0]}</span>
                <span style="flex: 2; text-align: center; font-weight: 600;">${m.Home} ${m.Score} ${m.Away}</span>
            </div>
        `).reverse().join('');

        h2hContainer.innerHTML = `
            <div style="margin-top: 30px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3 style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary);">Conflict History</h3>
                    <div style="font-size: 0.7rem; background: var(--accent-blue); padding: 2px 8px; border-radius: 10px; color: white;">LAST 5</div>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    ${resultsHtml}
                </div>
            </div>
        `;
        h2hContainer.style.display = 'block';
    } else {
        h2hContainer.style.display = 'none';
    }

    document.getElementById('analytics-summary').innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)); gap: 15px; text-align: center;">
            <div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">BTTS</div>
                <div style="font-size: 1.1rem; font-weight: 800;">${(pred.btts * 100).toFixed(0)}%</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">OVER 1.5</div>
                <div style="font-size: 1.1rem; font-weight: 800;">${(pred.over15 * 100).toFixed(0)}%</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">OVER 2.5</div>
                <div style="font-size: 1.1rem; font-weight: 800;">${(pred.over25 * 100).toFixed(0)}%</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">1X</div>
                <div style="font-size: 1.1rem; font-weight: 800;">${((pred.mc_h_win + pred.mc_draw) * 100).toFixed(0)}%</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: var(--text-secondary);">X2</div>
                <div style="font-size: 1.1rem; font-weight: 800;">${((pred.mc_a_win + pred.mc_draw) * 100).toFixed(0)}%</div>
            </div>
        </div>
    `;
}

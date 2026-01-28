# Test web scraping approach for UCL fixtures
# This is a fallback that doesn't require any API key
import requests
from datetime import datetime

print("Testing alternative approaches for UCL fixtures...")

# Approach 1: Try transfermarkt (public schedule page)
print("\n=== Testing Sofascore Public API ===")
try:
    # Sofascore has a public API for fixtures
    url = "https://api.sofascore.com/api/v1/unique-tournament/7/season/52162/events/next/0"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        events = data.get('events', [])
        print(f"Events found: {len(events)}")
        for e in events[:5]:
            home = e.get('homeTeam', {}).get('name', '?')
            away = e.get('awayTeam', {}).get('name', '?')
            ts = e.get('startTimestamp', 0)
            date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if ts else '?'
            print(f"  {home} vs {away} - {date}")
except Exception as ex:
    print(f"Error: {ex}")

# Approach 2: Try ESPN public API  
print("\n=== Testing ESPN Public API ===")
try:
    # ESPN has public endpoints
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard"
    r = requests.get(url, timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        events = data.get('events', [])
        print(f"Events found: {len(events)}")
        for e in events[:5]:
            name = e.get('name', '?')
            date = e.get('date', '?')
            print(f"  {name} - {date}")
except Exception as ex:
    print(f"Error: {ex}")

# Approach 3: Try FotMob public API
print("\n=== Testing FotMob Public API ===")
try:
    # FotMob has a public API
    url = "https://www.fotmob.com/api/leagues?id=42"  # 42 is Champions League
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        matches = data.get('matches', {}).get('allMatches', [])
        print(f"Matches found: {len(matches)}")
        upcoming = [m for m in matches if m.get('status', {}).get('finished') == False][:5]
        for m in upcoming:
            home = m.get('home', {}).get('name', '?')
            away = m.get('away', {}).get('name', '?')
            print(f"  {home} vs {away}")
except Exception as ex:
    print(f"Error: {ex}")

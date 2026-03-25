# test_benzinga.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BENZINGA_API")
if not API_KEY:
    print("ERROR: BENZINGA_API not found in .env")
    exit(1)

BASE_URL = "https://api.benzinga.com/api/v2/news"

# Test 1 — single article fetch (cheapest possible call)
print("Test 1: Fetching 1 general news article...")
resp = requests.get(BASE_URL, params={
    "token":    API_KEY,
    "pageSize": 1,
    "channels": "News",
}, timeout=10)
print(f"  Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    articles = data if isinstance(data, list) else data.get("result", [])
    if articles:
        a = articles[0]
        print(f"  OK — got article: '{a.get('title', 'no title')[:80]}'")
        print(f"       date: {a.get('created', 'N/A')}")
    else:
        print("  WARNING: 200 OK but no articles returned")
else:
    print(f"  FAIL — response: {resp.text[:300]}")

# Test 2 — ticker-specific news
print("\nTest 2: Fetching AAPL news...")
resp2 = requests.get(BASE_URL, params={
    "token":    API_KEY,
    "pageSize": 3,
    "tickers":  "AAPL",
}, timeout=10)
print(f"  Status: {resp2.status_code}")
if resp2.status_code == 200:
    data2 = resp2.json()
    articles2 = data2 if isinstance(data2, list) else data2.get("result", [])
    print(f"  OK — {len(articles2)} article(s) returned")
    for a in articles2:
        print(f"    - {a.get('title', '')[:70]}")
else:
    print(f"  FAIL — response: {resp2.text[:300]}")

print("\nDone.")

# Port `limit_test` → `main` (excluding Crucix)

Goal: apply all changes made on the `limit_test` branch to `main`, **excluding** the Crucix OSINT sidecar integration. Crucix is optional and cleanly isolated, so it can be stripped without affecting the rest.

## Branch context
- Source branch: `limit_test`
- Target branch: `main`
- Commits on `limit_test` not in `main`:
  ```
  f0c1494 fixed bugs
  10a2654 applied tons of changes
  68f7ba6 fixed some stuff
  c2d75bf changed settings
  17c49a4 added crucix
  cb94a53 added crucix to limit test
  e9674cd fixed some bugs
  682a766 fix some bugs
  5e00ffe make it more adaptable to crisis market environment
  ed74f2a added technical design document
  59382b3 changed the pipeline to be more adaptive
  a8bca9e removed diversity enforcer
  4182018 removed removal of same category
  ```

---

## ❌ EXCLUDE (Crucix — do NOT port)

Delete / never create these:
- `Crucix-master/` — entire vendored Node sidecar (~18k lines, 80+ files)
- `crucix_adapter.py` — Python client for the sidecar (459 lines)

Strip Crucix hunks out of these shared files:
- **`pipeline_orchestrator.py`** — remove the Stage 1c OSINT enrichment block (approx. lines 264–316 on `limit_test`). Specifically:
  - The entire `# ── Stage 1c: Crucix OSINT enrichment ──` block
  - `crucix_snapshot`, `crucix_macro_ctx`, `_crucix` locals
  - `from crucix_adapter import CrucixAdapter` import
  - The `articles["crucix_osint"] = crucix_df` injection
  - The `if crucix_macro_ctx:` merge that adds `summary["crucix_macro"]` / appends `_crucix.to_summary_text(...)`
- **`commands.txt`** — keep only the "Run Chequita (pipeline only, no Crucix)" section. Drop all Crucix sidecar terminal commands, the `cd Crucix-master` blocks, and the `CrucixAdapter` health check.
- **`TECHNICAL_DESIGN_DOCUMENT.md`** — scrub any Crucix mentions, or omit the doc entirely.
- **`README.md`** — scrub Crucix mentions from the updated README.
- **`tests/test_all_fixes.py`** — if any test references Crucix, remove those cases; keep the rest.

Verification: after porting, run `git grep -i crucix` — it should return **zero** matches.

---

## ✅ INCLUDE (features to port to main)

### 1. Pipeline orchestration overhaul
File: `pipeline_orchestrator.py` (+182 lines net, excluding Crucix block)
- Adaptive pipeline restructuring (commit `59382b3` — "changed the pipeline to be more adaptive")
- Crisis-market adaptability changes (commit `5e00ffe`)
- Remove the diversity enforcer stage (commit `a8bca9e`)
- Remove the same-category removal step (commit `4182018`)
- Bug fixes from `682a766`, `e9674cd`, `68f7ba6`, `f0c1494`, `10a2654`

### 2. Backtester expansion
File: `backtester.py` (+184 lines, nearly doubled from "applied tons of changes" — `10a2654`)
- Port the full expanded backtester logic.

### 3. Strategy selection
File: `strategy_selector.py` (+145 lines)
- Port all updates to the selection logic.

### 4. Regime classifier (crisis-market aware)
File: `regime_classifier.py` (+215 lines)
- New crisis-market regime detection logic tied to `5e00ffe`.

### 5. Report generator rewrite
File: `report_generator.py` (+587 lines — large rewrite)
- Port the full updated reporting pipeline.

### 6. Signal / alpha / diagnostics engines
- `ml_signal_engine.py`
- `alpha_engine.py`
- `diagnostics_engine.py`

### 7. Risk / portfolio
- `monte_carlo_engine.py`
- `portfolio_optimizer.py`

### 8. Data & execution layer
- `ohlcv_fetcher.py`
- `ticker_screener.py`
- `execution_advisor.py`

### 9. Tests
- New file: `tests/test_all_fixes.py` (~775 lines) — port it, but first remove any Crucix-specific assertions.

### 10. Documentation
- `README.md` updates (Crucix mentions stripped)
- `TECHNICAL_DESIGN_DOCUMENT.md` (new, ~771 lines — Crucix mentions stripped) *(optional)*

### 11. Settings
- `.claude/settings.json` / `.claude/settings.local.json` — minor tweaks, port if desired.

---

## Suggested porting procedure

```bash
# 1. Start from main
git checkout main
git pull
git checkout -b port-limit-test-no-crucix

# 2. Cherry-pick the clearly Crucix-free commits first
git cherry-pick 4182018 a8bca9e 59382b3 ed74f2a 5e00ffe 682a766 e9674cd

# 3. Cherry-pick the mixed commits with -n and manually drop Crucix hunks
git cherry-pick -n 10a2654 68f7ba6 f0c1494 c2d75bf
#    - Manually edit pipeline_orchestrator.py to remove the Stage 1c block
#    - Reject any Crucix-master/ or crucix_adapter.py additions
#    - Clean commands.txt, README.md, TECHNICAL_DESIGN_DOCUMENT.md

# 4. SKIP entirely:
#    17c49a4 ("added crucix")
#    cb94a53 ("added crucix to limit test")

# 5. Verify Crucix is fully gone
git grep -i crucix   # must return nothing
ls Crucix-master 2>/dev/null && echo "ERROR: Crucix-master still exists"

# 6. Run tests
pytest tests/test_all_fixes.py

# 7. Commit and push
git commit -m "feat: port limit_test adaptive pipeline + crisis regime + backtester expansion (no Crucix)"
git push -u origin port-limit-test-no-crucix
```

---

## Summary of behavioral changes landing on main

1. **Adaptive pipeline** that reshapes itself based on market conditions instead of running fixed stages.
2. **Crisis-market regime detection** via expanded `regime_classifier.py`, threaded through strategy selection.
3. **Removed diversity enforcer** and **removed same-category removal** — the pipeline no longer artificially diversifies or drops same-category candidates.
4. **Backtester nearly doubled in scope** — richer metrics / simulation paths.
5. **Report generator large rewrite** — new reporting outputs.
6. **New comprehensive test suite** `tests/test_all_fixes.py`.
7. **Misc bug fixes** across signal, alpha, diagnostics, Monte Carlo, portfolio optimizer, OHLCV fetcher, ticker screener, execution advisor.

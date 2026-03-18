"""
NCAA March Madness 2026 Prediction Pipeline — Competition-Level (Men + Women)
==============================================================================
Trains TWO completely separate models — Men's and Women's.
Men and women never play each other (ID ranges 1xxx vs 3xxx never overlap).
Output: single submission.csv with all 2026 M+W predictions.

2026 MEN'S SEEDS — HARDCODED FROM OFFICIAL BRACKET
====================================================
EAST REGION
  1  Duke               16 Siena
  8  Ohio St.            9 TCU
  5  St. John's         12 Northern Iowa
  4  Kansas             13 Cal Baptist
  6  Louisville         11 South Florida
  3  Michigan St.       14 North Dakota St.
  7  UCLA               10 UCF
  2  UConn              15 Furman

WEST REGION
  1  Arizona            16 Long Island Univ.
  8  Villanova           9 Utah State
  5  Wisconsin          12 High Point
  4  Arkansas           13 Hawai'i
  6  BYU                11 Texas / NC State  (First Four)
  3  Gonzaga            14 Kennesaw State
  7  Miami FL           10 Missouri
  2  Purdue             15 Queens Univ.

MIDWEST REGION
  1  Michigan           16 UMBC / Howard      (First Four)
  8  Georgia             9 Saint Louis
  5  Texas Tech         12 Akron
  4  Alabama            13 Hofstra
  6  Tennessee          14 Wright State
  7  Kentucky           10 Santa Clara
  2  Iowa State         15 Tennessee State
  11 Miami OH / SMU     (First Four)

SOUTH REGION
  1  Florida            16 Prairie View A&M / Lehigh  (First Four)
  8  Clemson             9 Iowa
  5  Vanderbilt         12 McNeese
  4  Nebraska           13 Troy
  6  North Carolina     11 VCU
  3  Illinois           14 Penn
  7  Saint Mary's       10 Texas A&M
  2  Houston            15 Idaho

First Four play-in games (these teams share a seed slot):
  W: UMBC vs Howard         → winner gets 16-seed in Midwest
  W: Texas vs NC State      → winner gets 11-seed in West
  S: Prairie View vs Lehigh → winner gets 16-seed in South
  MW: Miami(OH) vs SMU      → winner gets 11-seed in Midwest

FEATURES
  Win Quality   : win_pct, conf_win_pct, sos, avg_opp_winrate
  Efficiency    : off_rtg, def_rtg, net_rtg
  Shooting      : fg_pct, fg3_pct, ft_pct, efg_pct, ts_pct
  Ball Control  : ast_to, to_poss, stl_poss, blk_poss
  Rebounding    : orr, drr, reb_diff
  Recent Form   : last-5 win_pct, score_diff, off_rtg
  Elo           : elo + elo_win_prob
  Seeding       : seed (1-16) + seed_diff   ← #4 most predictive signal
  Massey        : rank_mean/median/min/max/std  (men only)
  All above     : _diff / _t1 / _t2 versions

TRICKS
  Isotonic calibration, clipping [0.025, 0.975]
  4-model ensemble: XGBoost 40% + LightGBM 25% + LogReg 20% + Elo 15%

HOW TO RUN
----------
  pip install pandas numpy scikit-learn xgboost lightgbm joblib
  python march_madness_pipeline.py

OUTPUTS
  submission.csv    — combined M+W 2026 predictions
  cv_results_M.csv  — men's Brier score per CV fold
  cv_results_W.csv  — women's Brier score per CV fold
"""

import os, warnings, subprocess
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
DATA_DIR      = "."
OUTPUT        = "submission.csv"

CV_START      = 2010
CV_END        = 2025
MIN_TRAIN_YRS = 5

ELO_K         = 20
ELO_INIT      = 1500
ELO_CARRYOVER = 0.4
RECENT_N      = 5

CLIP_LO, CLIP_HI = 0.025, 0.975

W_XGB, W_LGBM, W_LR, W_ELO = 0.40, 0.25, 0.20, 0.15

MEN_ID_MIN,   MEN_ID_MAX   = 1000, 1999
WOMEN_ID_MIN, WOMEN_ID_MAX = 3000, 3999


# ══════════════════════════════════════════════════════════
# 2026 MEN'S SEEDS
# Keyed by the team name spelling used in MTeamSpellings.csv
# (all lowercase). The script resolves these to TeamIDs at
# runtime using the MTeams.csv + MTeamSpellings.csv files,
# so you never need to hardcode numeric IDs.
#
# Play-in teams are given seed 11 or 16 — both teams in a
# First Four game share the same seed number. The winner
# will carry that seed into the bracket; the loser's seed
# feature is irrelevant because they don't make the field.
# ══════════════════════════════════════════════════════════
SEEDS_2026_M_BY_NAME = {
    # EAST
    "duke":              1,
    "siena":            16,
    "ohio st":           8,
    "tcu":               9,
    "st john's":         5,   # also spelled "saint john's" in some files
    "northern iowa":    12,
    "kansas":            4,
    "cal baptist":      13,
    "louisville":        6,
    "south florida":    11,
    "michigan st":       3,
    "north dakota st":  14,
    "ucla":              7,
    "ucf":              10,
    "connecticut":       2,   # UConn's official Kaggle name is "Connecticut"
    "furman":           15,

    # WEST
    "arizona":           1,
    "liu":              16,   # Long Island University — check MTeamSpellings
    "villanova":         8,
    "utah st":           9,
    "wisconsin":         5,
    "high point":       12,
    "arkansas":          4,
    "hawaii":           13,
    "byu":               6,
    "texas":            11,   # First Four
    "nc state":         11,   # First Four
    "gonzaga":           3,
    "kennesaw st":      14,
    "miami fl":          7,   # Miami (FL) — NOT Miami (OH)
    "missouri":         10,
    "purdue":            2,
    "queens nc":        15,   # Queens University (Charlotte, NC)

    # MIDWEST
    "michigan":          1,
    "umbc":             16,   # First Four
    "howard":           16,   # First Four
    "georgia":           8,
    "saint louis":       9,
    "texas tech":        5,
    "akron":            12,
    "alabama":           4,
    "hofstra":          13,
    "tennessee":         6,
    "wright st":        14,
    "kentucky":          7,
    "santa clara":      10,
    "iowa st":           2,
    "virginia":          3,
    "tennessee st":     15,
    "miami oh":         11,   # First Four — Miami (Ohio)
    "smu":              11,   # First Four

    # SOUTH
    "florida":           1,
    "prairie view":     16,   # First Four
    "lehigh":           16,   # First Four
    "clemson":           8,
    "iowa":              9,
    "vanderbilt":        5,
    "mcneese st":       12,
    "nebraska":          4,
    "troy":             13,
    "north carolina":    6,
    "vcu":              11,
    "illinois":          3,
    "penn":             14,
    "saint mary's":      7,
    "texas a&m":        10,
    "houston":           2,
    "idaho":            15,
}

# ══════════════════════════════════════════════════════════
# 2026 WOMEN'S SEEDS — HARDCODED FROM OFFICIAL BRACKET
# ══════════════════════════════════════════════════════════
# REGIONAL 1  (UConn's region)
#   1  UConn              16 Cal Baptist
#   8  Clemson             9 Syracuse
#   5  Kentucky           12 Colorado State
#   4  Minnesota          13 Green Bay
#   6  Notre Dame         11 Fairfield
#   3  Duke               14 Charleston
#   7  Illinois           10 Arizona State
#   2  Vanderbilt         15 Fairleigh Dickinson
#
# REGIONAL 2  (UCLA's region)
#   1  UCLA               16 Samford
#   8  Iowa State          9 USC
#   5  Maryland           12 Gonzaga
#   4  North Carolina     13 Idaho
#   6  Baylor             11 Nebraska
#   3  Ohio State         14 UC San Diego
#   7  NC State           10 Colorado
#   2  Iowa               15 High Point
#
# REGIONAL 3  (Texas's region)
#   1  Texas              16 UTSA
#   8  Oklahoma State      9 Virginia Tech
#   5  Ole Miss           12 Murray State
#   4  Oklahoma           13 Western Illinois
#   6  Washington         11 Rhode Island
#   3  TCU                14 Vermont
#   7  Georgia            10 Tennessee
#   2  LSU                15 Holy Cross
#
# REGIONAL 4  (South Carolina's region)
#   1  South Carolina     16 Southern
#   8  Oregon              9 Princeton
#   5  Michigan State     12 James Madison
#   4  West Virginia      13 Miami OH
#   6  Alabama            11 South Dakota State
#   3  Louisville         14 Howard
#   7  Texas Tech         10 Villanova
#   2  Michigan           15 Jacksonville
# ══════════════════════════════════════════════════════════
SEEDS_2026_W_BY_NAME = {
    # ── REGIONAL 1 — UConn's region ──────────────────────
    # 1 UConn            16 Cal Baptist
    # 8 Clemson           9 Syracuse
    # 5 Kentucky         12 Colorado State
    # 4 Minnesota        13 Green Bay
    # 6 Notre Dame       11 Fairfield
    # 3 Duke             14 Charleston
    # 7 Illinois         10 Arizona State
    # 2 Vanderbilt       15 Fairleigh Dickinson
    "connecticut":         1,   # UConn Huskies
    "cal baptist":        16,
    "clemson":             8,
    "syracuse":            9,
    "kentucky":            5,
    "colorado st":        12,
    "minnesota":           4,
    "green bay":          13,
    "notre dame":          6,
    "fairfield":          11,
    "duke":                3,
    "charleston":         14,
    "illinois":            7,
    "arizona st":         10,
    "vanderbilt":          2,
    "virginia":            10,   # First Four — 10-seed
    "fairleigh dickinson": 15,

    # ── REGIONAL 2 — UCLA's region ───────────────────────
    # 1 UCLA             16 Samford
    # 8 Iowa State        9 USC
    # 5 Maryland         12 Gonzaga
    # 4 North Carolina   13 Idaho
    # 6 Baylor           11 Nebraska
    # 3 Ohio State       14 UC San Diego
    # 7 NC State         10 Colorado
    # 2 Iowa             15 High Point
    "ucla":                1,
    "samford":            16,
    "iowa st":             8,
    "usc":                 9,   # Southern California — check WTeamSpellings
    "maryland":            5,
    "gonzaga":            12,
    "north carolina":      4,
    "idaho":              13,
    "baylor":              6,
    "nebraska":           11,
    "ohio st":             3,
    "uc san diego":       14,
    "nc state":            7,
    "colorado":           10,
    "iowa":                2,
    "high point":         15,

    # ── REGIONAL 3 — Texas's region ──────────────────────
    # 1 Texas            16 UTSA / Stephen F. Austin  (First Four)
    # 8 Oklahoma State    9 Virginia Tech
    # 5 Ole Miss         12 Murray State
    # 4 Oklahoma         13 Western Illinois
    # 6 Washington       11 Rhode Island / Richmond   (First Four)
    # 3 TCU              14 Vermont
    # 7 Georgia          10 Tennessee
    # 2 LSU              15 Holy Cross
    "texas":               1,
    "utsa":               16,   # First Four
    "stephen f. austin":  16,   # First Four — SFA Ladyjacks
    "oklahoma st":         8,
    "virginia tech":       9,
    "ole miss":            5,   # check WTeamSpellings — may be "mississippi"
    "murray st":          12,
    "oklahoma":            4,
    "western illinois":   13,
    "washington":          6,
    "rhode island":       11,   # First Four
    "richmond":           11,   # First Four
    "tcu":                 3,
    "vermont":            14,
    "georgia":             7,
    "tennessee":          10,
    "lsu":                 2,
    "holy cross":         15,

    # ── REGIONAL 4 — South Carolina's region ─────────────
    # 1 South Carolina   16 Southern / Missouri State  (First Four)
    # 8 Oregon            9 Princeton
    # 5 Michigan State   12 James Madison
    # 4 West Virginia    13 Miami OH
    # 6 Alabama          11 South Dakota State
    # 3 Louisville       14 Howard
    # 7 Texas Tech       10 Villanova
    # 2 Michigan         15 Jacksonville
    "south carolina":      1,
    "southern":           16,   # First Four — Southern University Jaguars
    "missouri st":        16,   # First Four — Missouri State Lady Bears
    "oregon":              8,
    "princeton":           9,
    "michigan st":         5,
    "james madison":      12,
    "west virginia":       4,
    "miami oh":           13,   # Miami (Ohio) RedHawks — NOT Miami FL
    "alabama":             6,
    "south dakota st":    11,
    "louisville":          3,
    "howard":             14,
    "texas tech":          7,
    "villanova":          10,
    "michigan":            2,
    "jacksonville":       15,
}




# ══════════════════════════════════════════════════════════
# RESOLVE NAME → TEAM ID
# Matches seed dict keys against MTeams + MTeamSpellings
# ══════════════════════════════════════════════════════════
def resolve_seed_names_to_ids(seed_name_dict, data_dir):
    """
    Returns {TeamID: seed_number} by fuzzy-matching team name
    strings against MTeams.csv and MTeamSpellings.csv.
    Prints a warning for any name it cannot resolve.
    """
    def load_if_exists(name):
        p = os.path.join(data_dir, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    teams    = load_if_exists("MTeams.csv")
    spellings = load_if_exists("MTeamSpellings.csv")

    # Build a lookup: lowercase spelling -> TeamID
    lookup = {}
    if not teams.empty:
        for _, row in teams.iterrows():
            lookup[row["TeamName"].lower().strip()] = row["TeamID"]
    if not spellings.empty:
        for _, row in spellings.iterrows():
            lookup[row["TeamNameSpelling"].lower().strip()] = row["TeamID"]

    resolved = {}
    unresolved = []

    for name, seed in seed_name_dict.items():
        key = name.lower().strip()
        if key in lookup:
            resolved[lookup[key]] = seed
        else:
            # Try partial match
            matches = [(k, v) for k, v in lookup.items() if key in k or k in v if isinstance(v, int)]
            partial = [(k, v) for k, v in lookup.items() if key in k]
            if partial:
                best_k, best_v = partial[0]
                resolved[best_v] = seed
                print(f"  [SEED] '{name}' → partial match '{best_k}' (ID={best_v}, seed={seed})")
            else:
                unresolved.append(name)

    if unresolved:
        print(f"\n  [SEED WARNING] Could not resolve {len(unresolved)} team name(s):")
        for u in unresolved:
            print(f"    '{u}' (seed={seed_name_dict[u]}) — check MTeamSpellings.csv")
        print("  → These teams will have seed=NaN in predictions.")
        print("  → Fix: add the correct spelling to SEEDS_2026_M_BY_NAME above.\n")

    print(f"  [SEED] Resolved {len(resolved)}/{len(seed_name_dict)} team names to TeamIDs")
    return resolved


# ══════════════════════════════════════════════════════════
# GPU DETECTION
# ══════════════════════════════════════════════════════════
def detect_gpu():
    for cmd, label in [("nvidia-smi","NVIDIA"),("rocm-smi","AMD")]:
        try:
            r = subprocess.run([cmd], capture_output=True, timeout=5)
            if r.returncode == 0:
                print(f"✓ {label} GPU detected — XGB & LGBM will use GPU")
                return "cuda", True
        except Exception:
            pass
    print("○ No GPU detected — CPU mode")
    return "cpu", False

XGB_DEVICE, USE_GPU = detect_gpu()


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def load(name):
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)

def try_load(name):
    try:
        return load(name)
    except FileNotFoundError:
        return None

def parse_seeds(seeds_df):
    """Convert Kaggle seed strings ('W01','X12a') to integer 1-16."""
    df = seeds_df.copy()
    df["seed"] = (df["Seed"]
                  .str[1:]
                  .str.replace(r"[ab]$", "", regex=True)
                  .astype(int))
    return df[["Season","TeamID","seed"]]


# ══════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════
def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.025,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
        device=XGB_DEVICE, tree_method="hist",
    )

def make_lgbm():
    params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.025,
        num_leaves=31, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=-1, objective="binary",
    )
    if USE_GPU:
        params["device"] = "gpu"
    return lgb.LGBMClassifier(**params)

def make_lr():
    return LogisticRegression(C=0.05, max_iter=2000, random_state=42)

class EloModel:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
    def fit(self, elo_probs, y):
        self.iso.fit(elo_probs, y); return self
    def predict(self, elo_probs):
        return self.iso.predict(elo_probs)


# ══════════════════════════════════════════════════════════
# TRAIN + PREDICT
# ══════════════════════════════════════════════════════════
def train_and_predict(train_df, test_df, feat_cols):
    X_tr = train_df[feat_cols].fillna(0).values
    X_te = test_df[feat_cols].fillna(0).values
    y_tr = train_df["label"].values
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)
    xgb_cal  = CalibratedClassifierCV(make_xgb(),  method="isotonic", cv=3)
    lgbm_cal = CalibratedClassifierCV(make_lgbm(), method="isotonic", cv=3)
    lr_cal   = CalibratedClassifierCV(make_lr(),   method="sigmoid",  cv=3)
    xgb_cal.fit(X_tr, y_tr);   p_xgb  = xgb_cal.predict_proba(X_te)[:,1]
    lgbm_cal.fit(X_tr, y_tr);  p_lgbm = lgbm_cal.predict_proba(X_te)[:,1]
    lr_cal.fit(X_tr_s, y_tr);  p_lr   = lr_cal.predict_proba(X_te_s)[:,1]
    elo_m = EloModel().fit(train_df["elo_win_prob"].fillna(0.5).values, y_tr)
    p_elo = elo_m.predict(test_df["elo_win_prob"].fillna(0.5).values)
    return np.clip(W_XGB*p_xgb + W_LGBM*p_lgbm + W_LR*p_lr + W_ELO*p_elo,
                   CLIP_LO, CLIP_HI)


# ══════════════════════════════════════════════════════════
# GENDER PIPELINE
# ══════════════════════════════════════════════════════════
def run_gender_pipeline(gender, seeds_2026_by_id=None):
    g = gender
    print(f"\n{'━'*62}")
    print(f"  PIPELINE: {'MEN' if g=='M' else 'WOMEN'}  "
          f"(TeamIDs {'1000-1999' if g=='M' else '3000-3999'})")
    print(f"{'━'*62}")

    # ── Load data ────────────────────────────────────────
    print(f"[{g}] Loading data...")
    detailed    = load(f"{g}RegularSeasonDetailedResults.csv")
    compact     = load(f"{g}RegularSeasonCompactResults.csv")
    tourney_res = load(f"{g}NCAATourneyCompactResults.csv")
    conferences = try_load(f"{g}TeamConferences.csv")
    seeds_hist  = try_load(f"{g}NCAATourneySeeds.csv")   # historical seeds
    massey      = try_load("MMasseyOrdinals.csv") if g == "M" else None

    use_conf   = conferences is not None
    use_massey = massey is not None
    use_seeds  = seeds_hist is not None or seeds_2026_by_id is not None

    print(f"[{g}]   Detailed         : {detailed.shape}")
    print(f"[{g}]   Conference data  : {'yes' if use_conf   else 'no'}")
    print(f"[{g}]   Massey ordinals  : {'yes' if use_massey else 'no'}")
    print(f"[{g}]   Historical seeds : {'yes' if seeds_hist is not None else 'no'}")
    print(f"[{g}]   2026 seeds       : {'YES — {len(seeds_2026_by_id)} teams' if seeds_2026_by_id else 'not provided'}")

    if seeds_hist is not None:
        seeds_df = parse_seeds(seeds_hist)
    else:
        seeds_df = None

    # ── Elo ──────────────────────────────────────────────
    print(f"[{g}] Computing Elo...")
    def compute_elo(compact_df):
        elo, records = {}, []
        for season, grp in compact_df.sort_values(["Season","DayNum"]).groupby("Season"):
            for tid in elo:
                elo[tid] = ELO_INIT*(1-ELO_CARRYOVER) + elo[tid]*ELO_CARRYOVER
            for _, row in grp.iterrows():
                w, l  = row["WTeamID"], row["LTeamID"]
                rw    = elo.get(w, ELO_INIT)
                rl    = elo.get(l, ELO_INIT)
                exp_w = 1 / (1 + 10**((rl-rw)/400))
                elo[w] = rw + ELO_K*(1-exp_w)
                elo[l] = rl + ELO_K*(0-(1-exp_w))
            for tid, rating in elo.items():
                records.append({"Season":season,"TeamID":tid,"elo":rating})
        return pd.DataFrame(records)
    elo_df = compute_elo(compact)

    # ── Strength of schedule ─────────────────────────────
    print(f"[{g}] Computing SOS...")
    def compute_sos(compact_df):
        rows_wp = []
        for _, row in compact_df.iterrows():
            rows_wp.append({"Season":row["Season"],"TeamID":row["WTeamID"],"win":1})
            rows_wp.append({"Season":row["Season"],"TeamID":row["LTeamID"],"win":0})
        wp = pd.DataFrame(rows_wp).groupby(["Season","TeamID"])["win"].mean().reset_index()
        wp.columns = ["Season","TeamID","win_pct_raw"]
        wp_dict = wp.set_index(["Season","TeamID"])["win_pct_raw"].to_dict()
        opp_rows = []
        for _, row in compact_df.iterrows():
            w,l,s = row["WTeamID"],row["LTeamID"],row["Season"]
            opp_rows.append({"Season":s,"TeamID":w,"opp":l})
            opp_rows.append({"Season":s,"TeamID":l,"opp":w})
        odf = pd.DataFrame(opp_rows)
        odf["opp_win_pct"] = odf.apply(
            lambda r: wp_dict.get((r["Season"],r["opp"]),0.5), axis=1)
        sos = odf.groupby(["Season","TeamID"])["opp_win_pct"].mean().reset_index()
        sos.columns = ["Season","TeamID","sos"]
        sos["avg_opp_winrate"] = sos["sos"]
        return sos
    sos_df = compute_sos(compact)

    # ── Conference win % ─────────────────────────────────
    def compute_conf_winpct(compact_df, conferences_df):
        if conferences_df is None: return None
        conf_map = conferences_df.set_index(["Season","TeamID"])["ConfAbbrev"].to_dict()
        rows = []
        for _, row in compact_df.iterrows():
            s,w,l = row["Season"],row["WTeamID"],row["LTeamID"]
            if conf_map.get((s,w)) == conf_map.get((s,l)) and conf_map.get((s,w)):
                rows.append({"Season":s,"TeamID":w,"conf_win":1})
                rows.append({"Season":s,"TeamID":l,"conf_win":0})
        if not rows: return None
        cdf = pd.DataFrame(rows)
        out = cdf.groupby(["Season","TeamID"])["conf_win"].mean().reset_index()
        out.columns = ["Season","TeamID","conf_win_pct"]
        return out
    conf_wp = compute_conf_winpct(compact, conferences)

    # ── Box-score stats ──────────────────────────────────
    print(f"[{g}] Building team stats...")
    def build_team_stats(detailed_df):
        def stack(df, mine, opp):
            d = df.copy()
            d["TeamID"] = d[f"{mine}TeamID"]
            d["Win"]    = 1 if mine=="W" else 0
            d = d.rename(columns={
                f"{mine}Score":"pts", f"{mine}FGM":"fgm",  f"{mine}FGA":"fga",
                f"{mine}FGM3":"fgm3", f"{mine}FGA3":"fga3",
                f"{mine}FTM":"ftm",   f"{mine}FTA":"fta",
                f"{mine}OR":"or_",    f"{mine}DR":"dr",
                f"{mine}Ast":"ast",   f"{mine}TO":"to_",
                f"{mine}Stl":"stl",   f"{mine}Blk":"blk",
                f"{opp}Score":"opp_pts",
                f"{opp}OR":"opp_or",  f"{opp}DR":"opp_dr",
            })
            return d[["Season","TeamID","Win","pts","fgm","fga","fgm3","fga3",
                       "ftm","fta","or_","dr","ast","to_","stl","blk",
                       "opp_pts","opp_or","opp_dr"]]
        combined = pd.concat(
            [stack(detailed_df,"W","L"), stack(detailed_df,"L","W")],
            ignore_index=True)
        agg = combined.groupby(["Season","TeamID"]).agg(
            games=("Win","count"), wins=("Win","sum"),
            pts=("pts","mean"),   fgm=("fgm","mean"),   fga=("fga","mean"),
            fgm3=("fgm3","mean"), fga3=("fga3","mean"),
            ftm=("ftm","mean"),   fta=("fta","mean"),
            or_=("or_","mean"),   dr=("dr","mean"),
            ast=("ast","mean"),   to_=("to_","mean"),
            stl=("stl","mean"),   blk=("blk","mean"),
            opp_pts=("opp_pts","mean"),
            opp_or=("opp_or","mean"), opp_dr=("opp_dr","mean"),
        ).reset_index()
        s = lambda x: x.replace(0, np.nan)
        agg["win_pct"]    = agg["wins"] / agg["games"]
        agg["poss"]       = agg["fga"]  + 0.44*agg["fta"] - agg["or_"] + agg["to_"]
        agg["off_rtg"]    = agg["pts"]     / s(agg["poss"]) * 100
        agg["def_rtg"]    = agg["opp_pts"] / s(agg["poss"]) * 100
        agg["net_rtg"]    = agg["off_rtg"] - agg["def_rtg"]
        agg["fg_pct"]     = agg["fgm"]  / s(agg["fga"])
        agg["fg3_pct"]    = agg["fgm3"] / s(agg["fga3"])
        agg["ft_pct"]     = agg["ftm"]  / s(agg["fta"])
        agg["efg_pct"]    = (agg["fgm"] + 0.5*agg["fgm3"]) / s(agg["fga"])
        agg["ts_pct"]     = agg["pts"]  / s(2*(agg["fga"] + 0.44*agg["fta"]))
        agg["ast_to"]     = agg["ast"]  / s(agg["to_"])
        agg["to_poss"]    = agg["to_"]  / s(agg["poss"])
        agg["stl_poss"]   = agg["stl"]  / s(agg["poss"])
        agg["blk_poss"]   = agg["blk"]  / s(agg["poss"])
        agg["orr"]        = agg["or_"]  / s(agg["or_"] + agg["opp_dr"])
        agg["drr"]        = agg["dr"]   / s(agg["dr"]  + agg["opp_or"])
        agg["reb_diff"]   = agg["dr"]   - agg["or_"]
        agg["score_diff"] = agg["pts"]  - agg["opp_pts"]
        return agg
    team_stats = build_team_stats(detailed)

    # ── Recent form ──────────────────────────────────────
    print(f"[{g}] Building recent form...")
    def _recent_one(season, grp, n):
        gl = {}
        for _, row in grp.iterrows():
            for side, opp in [("W","L"),("L","W")]:
                tid  = row[f"{side}TeamID"]
                pts  = row[f"{side}Score"]; opts = row[f"{opp}Score"]
                fga  = row[f"{side}FGA"];   fta  = row[f"{side}FTA"]
                or_  = row[f"{side}OR"];    to_  = row[f"{side}TO"]
                poss = fga + 0.44*fta - or_ + to_
                gl.setdefault(tid,[]).append({
                    "win": 1 if side=="W" else 0,
                    "diff": pts-opts,
                    "ortg": pts/poss*100 if poss>0 else 100.0,
                })
        return [{"Season":season,"TeamID":tid,
                 "recent_win_pct":   np.mean([gg["win"]  for gg in games[-n:]]),
                 "recent_score_diff":np.mean([gg["diff"] for gg in games[-n:]]),
                 "recent_off_rtg":   np.mean([gg["ortg"] for gg in games[-n:]])}
                for tid, games in gl.items()]
    groups  = list(detailed.sort_values("DayNum").groupby("Season"))
    results = Parallel(n_jobs=-1)(delayed(_recent_one)(s,g2,RECENT_N) for s,g2 in groups)
    recent_form = pd.DataFrame([r for batch in results for r in batch])

    # ── Massey ───────────────────────────────────────────
    if use_massey:
        print(f"[{g}] Building Massey rankings...")
        pre = massey[massey["RankingDayNum"]>=128].copy()
        pre = (pre.sort_values("RankingDayNum")
                  .groupby(["Season","TeamID","SystemName"]).last().reset_index())
        massey_feats = pre.groupby(["Season","TeamID"])["OrdinalRank"].agg(
            rank_mean="mean",rank_median="median",
            rank_min="min",  rank_max="max",
            rank_std="std",  rank_count="count",
        ).reset_index()
        massey_feats["rank_std"] = massey_feats["rank_std"].fillna(0)
    else:
        massey_feats = None

    # ── Stat column list ─────────────────────────────────
    base_cols = [
        "win_pct","score_diff","sos","avg_opp_winrate",
        "off_rtg","def_rtg","net_rtg",
        "fg_pct","fg3_pct","ft_pct","efg_pct","ts_pct",
        "ast_to","to_poss","stl_poss","blk_poss",
        "orr","drr","reb_diff","elo",
        "recent_win_pct","recent_score_diff","recent_off_rtg",
        "seed",
    ]
    conf_cols   = ["conf_win_pct"] if (use_conf and conf_wp is not None) else []
    massey_cols = ["rank_mean","rank_median","rank_min","rank_max","rank_std"] \
                  if use_massey else []
    all_stat_cols = base_cols + conf_cols + massey_cols

    # ── Merge features ───────────────────────────────────
    def get_team_features(season, seed_override=None):
        """
        seed_override: dict {TeamID: seed_int} — used for 2026 predictions
                       using the officially announced seeds.
        """
        df = team_stats[team_stats["Season"]==season].copy()
        df = df.merge(elo_df[elo_df["Season"]==season][["TeamID","elo"]],
                      on="TeamID",how="left")
        df = df.merge(sos_df[sos_df["Season"]==season][["TeamID","sos","avg_opp_winrate"]],
                      on="TeamID",how="left")
        if conf_wp is not None:
            df = df.merge(conf_wp[conf_wp["Season"]==season][["TeamID","conf_win_pct"]],
                          on="TeamID",how="left")
        df = df.merge(
            recent_form[recent_form["Season"]==season][
                ["TeamID","recent_win_pct","recent_score_diff","recent_off_rtg"]],
            on="TeamID",how="left")
        if massey_feats is not None:
            mf = massey_feats[massey_feats["Season"]==season].drop(columns=["Season"])
            df = df.merge(mf,on="TeamID",how="left")

        # Seeds — priority: override dict > historical CSV > NaN
        if seed_override is not None:
            seed_series = pd.DataFrame(
                list(seed_override.items()), columns=["TeamID","seed"])
            df = df.merge(seed_series, on="TeamID", how="left")
        elif seeds_df is not None:
            s_s = seeds_df[seeds_df["Season"]==season][["TeamID","seed"]]
            df  = df.merge(s_s, on="TeamID", how="left")
        else:
            df["seed"] = np.nan
        return df

    # ── Build matchup rows ───────────────────────────────
    def build_matchup_rows(games_df, season, team_feats):
        tf   = team_feats.set_index("TeamID")
        rows = []
        for _, row in games_df[games_df["Season"]==season].iterrows():
            for t1,t2,label in [
                (row["WTeamID"],row["LTeamID"],1),
                (row["LTeamID"],row["WTeamID"],0),
            ]:
                if t1 not in tf.index or t2 not in tf.index: continue
                s1,s2 = tf.loc[t1], tf.loc[t2]
                feat  = {"Season":season,"T1":t1,"T2":t2,"label":label}
                for c in all_stat_cols:
                    v1 = float(s1[c]) if c in s1.index and not pd.isna(s1.get(c,np.nan)) else np.nan
                    v2 = float(s2[c]) if c in s2.index and not pd.isna(s2.get(c,np.nan)) else np.nan
                    feat[f"{c}_t1"]   = v1
                    feat[f"{c}_t2"]   = v2
                    feat[f"{c}_diff"] = (v1-v2) if not(np.isnan(v1) or np.isnan(v2)) else np.nan
                r1 = float(s1["elo"]) if "elo" in s1.index else ELO_INIT
                r2 = float(s2["elo"]) if "elo" in s2.index else ELO_INIT
                feat["elo_win_prob"] = 1/(1+10**((r2-r1)/400))
                rows.append(feat)
        return pd.DataFrame(rows)

    def build_season_dataset(season):
        tf = get_team_features(season)
        tg = tourney_res[tourney_res["Season"]==season]
        return build_matchup_rows(tg, season, tf), tf

    # ── Pre-build season datasets ────────────────────────
    print(f"[{g}] Pre-building season datasets...")
    season_data = {}
    for s in range(2003, 2026):
        try:
            df, tf = build_season_dataset(s)
            if not df.empty:
                season_data[s] = (df, tf)
        except Exception:
            pass

    all_df    = pd.concat([season_data[s][0] for s in season_data], ignore_index=True)
    exclude   = {"Season","T1","T2","label","elo_win_prob"}
    feat_cols = [c for c in all_df.columns
                 if c not in exclude and pd.api.types.is_numeric_dtype(all_df[c])]
    seed_fc   = sum(1 for c in feat_cols if "seed" in c)
    print(f"[{g}]   Seasons : {min(season_data)}–{max(season_data)}")
    print(f"[{g}]   Features: {len(feat_cols)}  (seed features: {seed_fc})")

    # ── CV ───────────────────────────────────────────────
    print(f"\n[{g}] Expanding-window CV (2010–2025):")
    cv_results = []
    for test_season in range(CV_START, CV_END+1):
        train_seasons = [s for s in season_data if s < test_season]
        if len(train_seasons) < MIN_TRAIN_YRS: continue
        train_df = pd.concat([season_data[s][0] for s in train_seasons], ignore_index=True)
        test_df  = season_data.get(test_season, (pd.DataFrame(),))[0]
        if test_df.empty: continue
        cols  = [c for c in feat_cols if c in test_df.columns]
        extra = ["elo_win_prob","label"]
        preds = train_and_predict(
            train_df[cols+extra].fillna(0),
            test_df[cols+extra].fillna(0), cols)
        brier = brier_score_loss(test_df["label"].values, preds)
        cv_results.append({
            "test_season":test_season,
            "train_seasons":len(train_seasons),
            "n_tourney_games":len(test_df)//2,
            "brier_score":round(brier,5),
        })
        print(f"  [{g}] {test_season}  train={len(train_seasons):2d}  "
              f"games={len(test_df)//2:3d}  Brier={brier:.5f}")

    cv_df      = pd.DataFrame(cv_results)
    mean_brier = cv_df["brier_score"].mean() if not cv_df.empty else float("nan")
    print(f"[{g}] Mean Brier (CV 2010–2025): {mean_brier:.5f}")
    cv_df.to_csv(os.path.join(DATA_DIR, f"cv_results_{g}.csv"), index=False)

    # ── Final model ──────────────────────────────────────
    print(f"[{g}] Training final model...")
    all_train  = pd.concat(
        [season_data[s][0] for s in season_data if s < 2026],
        ignore_index=True)
    cols_final = [c for c in feat_cols if c in all_train.columns]
    X_all      = all_train[cols_final].fillna(0).values
    y_all      = all_train["label"].values
    scaler_f   = StandardScaler()
    X_all_s    = scaler_f.fit_transform(X_all)

    xgb_f  = CalibratedClassifierCV(make_xgb(),  method="isotonic", cv=5)
    lgbm_f = CalibratedClassifierCV(make_lgbm(), method="isotonic", cv=5)
    lr_f   = CalibratedClassifierCV(make_lr(),   method="sigmoid",  cv=5)
    elo_f  = EloModel().fit(all_train["elo_win_prob"].fillna(0.5).values, y_all)
    xgb_f.fit(X_all, y_all); lgbm_f.fit(X_all, y_all); lr_f.fit(X_all_s, y_all)

    # ── 2026 team features with official seeds ───────────
    print(f"[{g}] Building 2026 team features...")
    tf_2026   = get_team_features(2025, seed_override=seeds_2026_by_id)
    tf_idx    = tf_2026.set_index("TeamID")

    seeded_count = tf_2026["seed"].notna().sum() if "seed" in tf_2026.columns else 0
    print(f"[{g}]   2026 teams with announced seeds: {seeded_count}")

    # ── Predict all 2026 matchups ────────────────────────
    team_ids_2026 = sorted(tf_2026["TeamID"].unique())
    n_mu          = len(team_ids_2026)*(len(team_ids_2026)-1)//2
    print(f"[{g}] Predicting {n_mu:,} matchups...")

    def predict_matchup(t1, t2):
        if t1 not in tf_idx.index or t2 not in tf_idx.index: return 0.5
        s1,s2 = tf_idx.loc[t1], tf_idx.loc[t2]
        feat  = {}
        for c in all_stat_cols:
            v1 = float(s1[c]) if c in s1.index and not pd.isna(s1.get(c,np.nan)) else np.nan
            v2 = float(s2[c]) if c in s2.index and not pd.isna(s2.get(c,np.nan)) else np.nan
            feat[f"{c}_t1"]   = v1
            feat[f"{c}_t2"]   = v2
            feat[f"{c}_diff"] = (v1-v2) if not(np.isnan(v1) or np.isnan(v2)) else np.nan
        r1 = float(s1.get("elo", ELO_INIT))
        r2 = float(s2.get("elo", ELO_INIT))
        ep  = 1/(1+10**((r2-r1)/400))
        feat["elo_win_prob"] = ep
        row_df = pd.DataFrame([feat])
        row_v  = row_df[cols_final].fillna(0).values
        row_s  = scaler_f.transform(row_v)
        p = (W_XGB *xgb_f.predict_proba(row_v)[0,1]
           + W_LGBM*lgbm_f.predict_proba(row_v)[0,1]
           + W_LR  *lr_f.predict_proba(row_s)[0,1]
           + W_ELO *elo_f.predict(np.array([ep]))[0])
        return float(np.clip(p, CLIP_LO, CLIP_HI))

    records = [{"ID":f"2026_{t1}_{t2}", "Pred":predict_matchup(t1,t2)}
               for t1,t2 in combinations(team_ids_2026, 2)]
    pred_df = pd.DataFrame(records)
    print(f"[{g}] Done  range=[{pred_df['Pred'].min():.3f}, "
          f"{pred_df['Pred'].max():.3f}]  mean_Brier={mean_brier:.5f}")
    return pred_df, cv_df


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
print("═"*62)
print("  NCAA March Madness 2026 — Men + Women Prediction Pipeline")
print("═"*62)

# ── Resolve 2026 men's seed names → TeamIDs ─────────────
print("\nResolving 2026 men's seed names to TeamIDs...")
seeds_2026_m = resolve_seed_names_to_ids(SEEDS_2026_M_BY_NAME, DATA_DIR, gender="M")

# ── Resolve 2026 women's seed names → TeamIDs ────────────
print("\nResolving 2026 women's seed names to TeamIDs...")
seeds_2026_w = resolve_seed_names_to_ids(SEEDS_2026_W_BY_NAME, DATA_DIR, gender="W")

# ── Print verification tables ─────────────────────────────
def _print_seed_table(seed_dict, gender):
    p = os.path.join(DATA_DIR, f"{gender}Teams.csv")
    tdf = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    idn = dict(zip(tdf["TeamID"], tdf["TeamName"])) if not tdf.empty else {}
    label = "MEN'S" if gender == "M" else "WOMEN'S"
    print(f"\n2026 {label} SEEDS (as resolved by the pipeline):")
    print(f"  {'TeamID':>6}  {'Seed':>4}  Team Name")
    print(f"  {'------':>6}  {'----':>4}  ---------")
    for tid, seed in sorted(seed_dict.items(), key=lambda x: x[1]):
        name = idn.get(tid, "??? — check spelling")
        print(f"  {tid:>6}  {seed:>4}  {name}")
    print(f"  → {len(seed_dict)} teams seeded  |  please verify against official bracket")

_print_seed_table(seeds_2026_m, "M")
_print_seed_table(seeds_2026_w, "W")

# ── Run pipelines ─────────────────────────────────────────
men_preds,   men_cv   = run_gender_pipeline("M", seeds_2026_by_id=seeds_2026_m)
women_preds, women_cv = run_gender_pipeline("W", seeds_2026_by_id=seeds_2026_w)

# ── Combine ──────────────────────────────────────────────
print("\nCombining M + W predictions...")

def get_gender_from_id(tid):
    if MEN_ID_MIN <= tid <= MEN_ID_MAX:     return "M"
    if WOMEN_ID_MIN <= tid <= WOMEN_ID_MAX: return "W"
    return "?"

try:
    sub_template = load("SampleSubmissionStage2.csv")
    print(f"Template found: {sub_template.shape[0]:,} rows")
    sub_template["t1"] = sub_template["ID"].apply(lambda x: int(x.split("_")[1]))
    sub_template["t2"] = sub_template["ID"].apply(lambda x: int(x.split("_")[2]))
    sub_template["g1"] = sub_template["t1"].apply(get_gender_from_id)
    sub_template["g2"] = sub_template["t2"].apply(get_gender_from_id)
    cross = sub_template[sub_template["g1"] != sub_template["g2"]]
    if not cross.empty:
        print(f"WARNING: {len(cross)} cross-gender rows → set to 0.5")
    all_preds = pd.concat([men_preds, women_preds], ignore_index=True)
    pred_map  = dict(zip(all_preds["ID"], all_preds["Pred"]))
    sub_template["Pred"] = sub_template["ID"].map(pred_map).fillna(0.5)
    final_sub = sub_template[["ID","Pred"]].copy()
except FileNotFoundError:
    print("No template found — stacking M + W directly.")
    final_sub = pd.concat([men_preds, women_preds], ignore_index=True)

final_sub = final_sub[final_sub["ID"].str.startswith("2026_")].copy()
final_sub.to_csv(os.path.join(DATA_DIR, OUTPUT), index=False)

m_out = final_sub[final_sub["ID"].apply(lambda x: int(x.split("_")[1])) <= MEN_ID_MAX]
w_out = final_sub[final_sub["ID"].apply(lambda x: int(x.split("_")[1])) >= WOMEN_ID_MIN]

print("\n" + "═"*62)
print("  FINAL SUMMARY")
print("═"*62)
print(f"  Men's CV mean Brier   : {men_cv['brier_score'].mean():.5f}")
print(f"  Women's CV mean Brier : {women_cv['brier_score'].mean():.5f}")
print(f"  Total predictions     : {len(final_sub):,}")
print(f"    Men's   matchups    : {len(m_out):,}")
print(f"    Women's matchups    : {len(w_out):,}")
print(f"  Pred range (M)        : [{m_out['Pred'].min():.3f}, {m_out['Pred'].max():.3f}]")
print(f"  Pred range (W)        : [{w_out['Pred'].min():.3f}, {w_out['Pred'].max():.3f}]")
print(f"\n  Saved → {OUTPUT}, cv_results_M.csv, cv_results_W.csv")
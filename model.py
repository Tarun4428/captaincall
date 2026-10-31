"""
CaptainCall - model.py
======================
Data loading, the matchup model and the validation code.

Question the project answers:
    "A batter is at the crease in a given phase of an IPL innings.
     Which of my bowlers should bowl the next over?"

Approach (all in plain pandas / numpy / scipy):
    1. Turn ball-by-ball data into one row per delivery.
    2. Estimate batter and bowler effects PER PHASE with shrinkage
       (empirical-Bayes style): players with few balls are pulled
       towards the league average for that phase.
    3. Add a head-to-head adjustment, also shrunk, because a specific
       batter-vs-bowler record is usually only a handful of balls.
    4. Report uncertainty (80% ranges) and flag low-confidence matchups.
    5. Validate on a held-out season against simple baselines.

Run `python model.py` to rebuild the cache and print the validation report.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
# Folder with the raw Cricsheet CSV files. Override with the environment
# variable CAPTAINCALL_DATA if you keep the files somewhere else.
RAW_DIR = Path(os.environ.get("CAPTAINCALL_DATA", DATA_DIR / "ipl"))
# A compact processed copy. It is what gets committed to GitHub so the
# deployed app does not need the thousands of raw files.
CACHE_FILE = DATA_DIR / f"{RAW_DIR.name}_processed.csv.gz"

PHASES = [
    "Powerplay (overs 1-6)",
    "Middle (overs 7-15)",
    "Death (overs 16-20)",
]
BALLS_PER_OVER = 6
Z80 = 1.2816  # z-value for an 80% two-sided interval

# Dismissals that are not credited to the bowler.
NOT_BOWLER_WICKETS = {
    "run out",
    "retired hurt",
    "retired out",
    "retired not out",
    "obstructing the field",
    "hit the ball twice",
}


# --------------------------------------------------------------------------
# 1. Loading Cricsheet ball-by-ball CSV files
# --------------------------------------------------------------------------
def _season_to_int(value) -> float:
    """'2007/08' -> 2007, '2024' -> 2024, '2008-04-18' -> 2008."""
    m = re.search(r"(\d{4})", str(value))
    return float(m.group(1)) if m else np.nan


def _read_new_format(path: Path) -> pd.DataFrame:
    """Cricsheet 'CSV2' format: one header row, one row per ball."""
    df = pd.read_csv(path, dtype=str)
    if "striker" not in df.columns or "bowler" not in df.columns:
        raise ValueError(f"{path.name}: unexpected columns {list(df.columns)}")
    if "match_id" not in df.columns:
        df["match_id"] = path.stem
    return df


def _read_old_format(path: Path) -> pd.DataFrame:
    """Older Cricsheet CSV: 'info' lines followed by 'ball' lines."""
    season, date, rows = None, None, []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if not r:
                continue
            if r[0] == "info" and len(r) >= 3:
                if r[1] == "season":
                    season = r[2]
                elif r[1] == "date" and date is None:
                    date = r[2]
            elif r[0] == "ball":
                r = r + [""] * (18 - len(r))
                rows.append(r[:18])
    cols = [
        "tag", "innings", "ball", "batting_team", "striker", "non_striker",
        "bowler", "runs_off_bat", "extras", "wides", "noballs", "byes",
        "legbyes", "penalty", "wicket_type", "player_dismissed",
        "other_wicket_type", "other_player_dismissed",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["match_id"] = path.stem
    df["season"] = season if season is not None else date
    return df


def _read_match_file(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip().lower()
    if first.startswith("version") or first.startswith("info"):
        return _read_old_format(path)
    return _read_new_format(path)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw rows to the columns the model needs."""
    df = df.copy()
    for col in ["runs_off_bat", "wides", "noballs"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["innings"] = pd.to_numeric(df["innings"], errors="coerce")
    df = df[df["innings"].isin([1, 2])]          # drop super overs
    df = df.dropna(subset=["striker", "bowler"])

    # "0.1", "19.6" -> over index 0..19 (string split avoids float issues)
    df["over"] = df["ball"].astype(str).str.split(".").str[0].astype(int)
    df["season"] = df["season"].map(_season_to_int)
    df = df.dropna(subset=["season"])

    out = pd.DataFrame(
        {
            "match_id": df["match_id"].astype(str),
            "season": df["season"].astype(int),
            "innings": df["innings"].astype(int),
            "over": df["over"],
            "batter": df["striker"].astype(str),
            "bowler": df["bowler"].astype(str),
            # runs charged to the bowler: bat runs + wides + no-balls
            "runs": (df["runs_off_bat"] + df["wides"] + df["noballs"]).astype(int),
            "legal": ((df["wides"] == 0) & (df["noballs"] == 0)).astype(int),
        }
    )
    wt = df["wicket_type"].fillna("").str.strip().str.lower()
    out["wicket"] = ((wt != "") & ~wt.isin(NOT_BOWLER_WICKETS)).astype(int).to_numpy()
    out["phase"] = np.select([out["over"] < 6, out["over"] < 15], [0, 1], default=2)
    return out.reset_index(drop=True)


def load_data(force: bool = False) -> pd.DataFrame:
    """
    Load the processed delivery table.
    - If raw Cricsheet files exist and the cache is older, rebuild the cache.
    - If only the cache exists (e.g. on Streamlit Cloud), use the cache.
    """
    raw_files: list[Path] = []
    if RAW_DIR.exists():
        raw_files = sorted(
            p for p in RAW_DIR.glob("*.csv") if not p.name.endswith("_info.csv")
        )

    if CACHE_FILE.exists() and not force:
        newest_raw = max((p.stat().st_mtime for p in raw_files), default=0.0)
        if CACHE_FILE.stat().st_mtime >= newest_raw:
            return pd.read_csv(CACHE_FILE)

    if not raw_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}. Download the IPL ball-by-ball CSV "
            "zip from cricsheet.org and extract it into that folder."
        )

    frames = [_normalise(_read_match_file(p)) for p in raw_files]
    df = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False, compression="gzip")
    return df


# --------------------------------------------------------------------------
# 2. The matchup model
# --------------------------------------------------------------------------
@dataclass
class Params:
    """Shrinkage strengths, measured in 'pseudo-balls' of league-average data."""
    k_bat_r: float = 60.0     # batter effect on runs
    k_bowl_r: float = 100.0   # bowler effect on runs
    k_bat_w: float = 250.0    # batter effect on wickets (rare event -> more shrinkage)
    k_bowl_w: float = 250.0   # bowler effect on wickets
    k_h2h_r: float = 60.0     # head-to-head adjustment, runs
    k_h2h_w: float = 150.0    # head-to-head adjustment, wickets


class MatchupModel:
    """
    Expected runs per ball and wicket probability per ball for a
    (batter, bowler, phase) combination:

        rate = league_phase_rate
               x batter_phase_factor      (shrunk)
               x bowler_phase_factor      (shrunk)
               x head_to_head_factor      (shrunk, pooled across phases)
    """

    def __init__(self, params: Params | None = None):
        self.p = params or Params()

    # ---- fitting ---------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "MatchupModel":
        g = df.groupby("phase").agg(
            runs=("runs", "sum"), balls=("legal", "sum"), wk=("wicket", "sum")
        )
        self.mu = g["runs"] / g["balls"]       # league runs per ball, by phase
        self.wr = g["wk"] / g["balls"]         # league wicket prob per ball
        self.sigma = df.groupby("phase")["runs"].std()

        self.bat_r, self.bat_w = self._factors(df, "batter", self.p.k_bat_r, self.p.k_bat_w)
        self.bowl_r, self.bowl_w = self._factors(df, "bowler", self.p.k_bowl_r, self.p.k_bowl_w)
        self.bat_pb = df.groupby(["batter", "phase"])["legal"].sum()
        self.bowl_pb = df.groupby(["bowler", "phase"])["legal"].sum()
        self.pairs = self._fit_pairs(df)
        return self

    def _factors(self, df, who, k_r, k_w):
        g = df.groupby([who, "phase"]).agg(
            runs=("runs", "sum"), balls=("legal", "sum"), wk=("wicket", "sum")
        )
        ph = g.index.get_level_values("phase")
        mu = self.mu.reindex(ph).to_numpy()
        wr = self.wr.reindex(ph).to_numpy()
        balls = g["balls"].to_numpy()
        f_r = ((g["runs"].to_numpy() + k_r * mu) / (balls + k_r)) / mu
        f_w = ((g["wk"].to_numpy() + k_w * wr) / (balls + k_w)) / wr
        return pd.Series(f_r, index=g.index), pd.Series(f_w, index=g.index)

    def _fit_pairs(self, df):
        """Head-to-head multipliers: actual vs expected, shrunk towards 1."""
        pr, pw, _ = self.predict(
            df["batter"].to_numpy(), df["bowler"].to_numpy(),
            df["phase"].to_numpy(), use_h2h=False,
        )
        legal = df["legal"].to_numpy()
        tmp = pd.DataFrame(
            {
                "batter": df["batter"].to_numpy(),
                "bowler": df["bowler"].to_numpy(),
                "A": df["runs"].to_numpy(),          # actual runs
                "E": pr * legal,                     # expected runs
                "W": df["wicket"].to_numpy(),        # actual wickets
                "EW": pw * legal,                    # expected wickets
                "n": legal,                          # legal balls
            }
        )
        g = tmp.groupby(["batter", "bowler"]).sum()
        g = g[g["n"] > 0].copy()
        c_r = self.p.k_h2h_r * g["E"] / g["n"]       # pseudo-evidence, runs
        c_w = self.p.k_h2h_w * g["EW"] / g["n"]      # pseudo-evidence, wickets
        g["m_r"] = (g["A"] + c_r) / (g["E"] + c_r)
        g["m_w"] = (g["W"] + c_w) / (g["EW"] + c_w)
        return g[["n", "m_r", "m_w"]]

    # ---- prediction ------------------------------------------------------
    @staticmethod
    def _lookup(series, names, phases, fill=1.0):
        idx = pd.MultiIndex.from_arrays([np.asarray(names), np.asarray(phases)])
        return series.reindex(idx).fillna(fill).to_numpy()

    def predict(self, batter, bowler, phase, use_batter=True, use_bowler=True,
                use_h2h=True):
        """Return (runs per ball, wicket prob per ball, head-to-head legal balls)."""
        batter, bowler, phase = map(np.asarray, (batter, bowler, phase))
        pr = self.mu.reindex(phase).to_numpy().astype(float)
        pw = self.wr.reindex(phase).to_numpy().astype(float)
        if use_bowler:
            pr = pr * self._lookup(self.bowl_r, bowler, phase)
            pw = pw * self._lookup(self.bowl_w, bowler, phase)
        if use_batter:
            pr = pr * self._lookup(self.bat_r, batter, phase)
            pw = pw * self._lookup(self.bat_w, batter, phase)
        n = np.zeros(len(pr))
        if use_h2h and use_batter and use_bowler:
            look = self.pairs.reindex(pd.MultiIndex.from_arrays([batter, bowler]))
            n = look["n"].fillna(0).to_numpy()
            pr = pr * look["m_r"].fillna(1.0).to_numpy()
            pw = pw * look["m_w"].fillna(1.0).to_numpy()
        return pr, np.clip(pw, 1e-4, 0.6), n

    def matchup_table(self, batter: str, bowlers, phase: int, lam: float) -> pd.DataFrame:
        """
        Rank bowlers against one batter in one phase.

        net_cost = expected runs per over - lam * expected wickets per over
        (lower is better for the bowling side). Ranges are approximate 80%
        intervals on the EXPECTED value, not on a single over's outcome.
        """
        bowlers = list(bowlers)
        k = len(bowlers)
        pr, pw, n = self.predict(
            np.array([batter] * k), np.array(bowlers), np.full(k, phase)
        )
        runs_over = BALLS_PER_OVER * pr
        se_r = BALLS_PER_OVER * self.sigma.loc[phase] / np.sqrt(n + self.p.k_h2h_r)

        strength = n + self.p.k_h2h_w
        a, b = pw * strength, (1 - pw) * strength
        wk_over = BALLS_PER_OVER * pw
        wk_lo = BALLS_PER_OVER * beta_dist.ppf(0.10, a, b)
        wk_hi = BALLS_PER_OVER * beta_dist.ppf(0.90, a, b)
        se_w = (wk_hi - wk_lo) / (2 * Z80)

        net = runs_over - lam * wk_over
        net_se = np.sqrt(se_r**2 + (lam * se_w) ** 2)

        out = pd.DataFrame(
            {
                "bowler": bowlers,
                "runs_over": runs_over,
                "runs_lo": runs_over - Z80 * se_r,
                "runs_hi": runs_over + Z80 * se_r,
                "wk_over": wk_over,
                "wk_lo": wk_lo,
                "wk_hi": wk_hi,
                "net_cost": net,
                "net_lo": net - Z80 * net_se,
                "net_hi": net + Z80 * net_se,
                "h2h_balls": n.astype(int),
                "bowler_phase_balls": self._lookup(
                    self.bowl_pb, bowlers, np.full(k, phase), fill=0.0
                ).astype(int),
            }
        )
        out["confidence"] = np.select(
            [out["h2h_balls"] >= 30, out["h2h_balls"] >= 12], ["High", "Medium"],
            default="Low (mostly prior)",
        )
        return out.sort_values("net_cost").reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Validation on a held-out season
# --------------------------------------------------------------------------
VARIANTS = {
    "League average by phase": dict(use_batter=False, use_bowler=False, use_h2h=False),
    "Bowler record only (naive captain)": dict(use_batter=False, use_bowler=True, use_h2h=False),
    "Bowler + batter (no head-to-head)": dict(use_batter=True, use_bowler=True, use_h2h=False),
    "CaptainCall (adds head-to-head)": dict(use_batter=True, use_bowler=True, use_h2h=True),
}
K_R_GRID = [10, 30, 60, 120, 240]
K_W_GRID = [30, 100, 300, 1000]


def _score(model: MatchupModel, d: pd.DataFrame, **variant):
    """Return (runs MSE per ball, wicket log-loss per ball) on legal balls."""
    t = d[d["legal"] == 1]
    pr, pw, _ = model.predict(
        t["batter"].to_numpy(), t["bowler"].to_numpy(), t["phase"].to_numpy(), **variant
    )
    y_r = t["runs"].to_numpy()
    y_w = t["wicket"].to_numpy()
    mse = float(np.mean((y_r - pr) ** 2))
    ll = float(-np.mean(y_w * np.log(pw) + (1 - y_w) * np.log(1 - pw)))
    return mse, ll, pr, pw, y_r, y_w


def lift_table(pr, pw, y_r, y_w) -> pd.DataFrame:
    """Do 'bowler-favoured' matchups really go better for the bowler?"""
    q = pd.qcut(pr, 5, labels=False, duplicates="drop")
    d = pd.DataFrame({"q": q, "pr": pr, "pw": pw, "y_r": y_r, "y_w": y_w})
    g = d.groupby("q").agg(
        balls=("pr", "size"), pred_r=("pr", "mean"), act_r=("y_r", "mean"),
        pred_w=("pw", "mean"), act_w=("y_w", "mean"),
    )
    out = pd.DataFrame(
        {
            "Group (by predicted runs/ball)": [f"Group {i + 1}" for i in range(len(g))],
            "Balls": g["balls"].to_numpy(),
            "Predicted runs/over": BALLS_PER_OVER * g["pred_r"].to_numpy(),
            "Actual runs/over": BALLS_PER_OVER * g["act_r"].to_numpy(),
            "Predicted wickets/over": BALLS_PER_OVER * g["pred_w"].to_numpy(),
            "Actual wickets/over": BALLS_PER_OVER * g["act_w"].to_numpy(),
        }
    )
    return out


def tune_and_validate(df: pd.DataFrame) -> dict:
    """
    Time-ordered validation (no peeking at the future):
      * tune the head-to-head shrinkage on season T-1 using seasons < T-1
      * then fit on seasons < T with those settings and test on season T
    """
    seasons = sorted(df["season"].unique())
    if len(seasons) < 4:
        raise ValueError("Need at least 4 seasons of data to validate the model.")
    test_s, val_s = seasons[-1], seasons[-2]
    train_tune = df[df["season"] < val_s]
    val = df[df["season"] == val_s]

    best_r = min(
        K_R_GRID,
        key=lambda k: _score(MatchupModel(Params(k_h2h_r=k)).fit(train_tune), val)[0],
    )
    best_w = min(
        K_W_GRID,
        key=lambda k: _score(MatchupModel(Params(k_h2h_w=k)).fit(train_tune), val)[1],
    )
    params = Params(k_h2h_r=best_r, k_h2h_w=best_w)

    train = df[df["season"] < test_s]
    test = df[df["season"] == test_s]
    model = MatchupModel(params).fit(train)

    rows, full_pred = [], None
    for name, kw in VARIANTS.items():
        mse, ll, pr, pw, y_r, y_w = _score(model, test, **kw)
        rows.append({"Model": name, "Runs/ball error (MSE)": mse, "Wicket log-loss": ll})
        if name.startswith("CaptainCall"):
            full_pred = (pr, pw, y_r, y_w)
    metrics = pd.DataFrame(rows)
    base = metrics.iloc[0]
    metrics["Runs error vs league avg (%)"] = (
        (base["Runs/ball error (MSE)"] - metrics["Runs/ball error (MSE)"])
        / base["Runs/ball error (MSE)"] * 100
    )
    metrics["Wicket log-loss vs league avg (%)"] = (
        (base["Wicket log-loss"] - metrics["Wicket log-loss"])
        / base["Wicket log-loss"] * 100
    )
    return {
        "test_season": int(test_s),
        "tuning_season": int(val_s),
        "train_seasons": f"{int(seasons[0])}-{int(test_s) - 1}",
        "n_test_balls": int((test["legal"] == 1).sum()),
        "params": asdict(params),
        "metrics": metrics,
        "lift": lift_table(*full_pred),
    }


# --------------------------------------------------------------------------
# 4. Helpers for the app
# --------------------------------------------------------------------------
def recent_players(df: pd.DataFrame, role: str, n_seasons: int = 3,
                   min_balls: int = 60) -> pd.Series:
    """Players ('batter' or 'bowler') active in the last seasons, by balls."""
    last = sorted(df["season"].unique())[-n_seasons:]
    s = df[df["season"].isin(last)].groupby(role)["legal"].sum()
    return s[s >= min_balls].sort_values(ascending=False)


def top_bowlers_in_phase(df: pd.DataFrame, phase: int, season: int, k: int = 6) -> list:
    d = df[(df["season"] == season) & (df["phase"] == phase)]
    return list(d.groupby("bowler")["legal"].sum().sort_values(ascending=False).head(k).index)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data(force="--rebuild" in sys.argv)
    print(f"Loaded {len(data):,} deliveries, "
          f"{data['match_id'].nunique():,} matches, seasons "
          f"{int(data['season'].min())}-{int(data['season'].max())}")
    result = tune_and_validate(data)
    print(f"\nTrained on {result['train_seasons']}, tested on {result['test_season']} "
          f"({result['n_test_balls']:,} legal balls)")
    print("Tuned parameters:", result["params"])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n--- Model comparison (higher % = better than league average) ---")
        print(result["metrics"].round(4).to_string(index=False))
        print("\n--- Do predicted groups separate in reality? ---")
        print(result["lift"].round(3).to_string(index=False))

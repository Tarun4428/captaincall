"""
make_sample_data.py - FOR TESTING ONLY
======================================
Creates SYNTHETIC (made-up) matches in the Cricsheet CSV format so you can
check that the pipeline and the app run before you download the real IPL data.

    python make_sample_data.py
    # PowerShell:  $env:CAPTAINCALL_DATA="data/sample_ipl"; streamlit run app.py
    # Mac/Linux:   CAPTAINCALL_DATA=data/sample_ipl streamlit run app.py

NEVER submit results from this data. The hackathon needs real sports data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data" / "sample_ipl"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

batters = [f"Sample Batter {i:02d}" for i in range(1, 21)]
bowlers = [f"Sample Bowler {i:02d}" for i in range(1, 13)]
bat_skill = dict(zip(batters, rng.normal(0, 0.12, len(batters))))
bowl_skill = dict(zip(bowlers, rng.normal(0, 0.12, len(bowlers))))
pair_effect = {(b, w): rng.normal(0, 0.10) for b in batters for w in bowlers}

RUNS = np.array([0, 1, 2, 3, 4, 6])
BASE_P = np.array([0.34, 0.36, 0.08, 0.01, 0.13, 0.08])
PHASE_TILT = {0: 0.10, 1: -0.05, 2: 0.25}   # death overs score faster
PHASE_WK = {0: 0.040, 1: 0.045, 2: 0.060}

COLUMNS = [
    "match_id", "season", "start_date", "venue", "innings", "ball",
    "batting_team", "bowling_team", "striker", "non_striker", "bowler",
    "runs_off_bat", "extras", "wides", "noballs", "byes", "legbyes",
    "penalty", "wicket_type", "player_dismissed", "other_wicket_type",
    "other_player_dismissed",
]

match_id = 1000
for season in range(2018, 2026):
    for _ in range(30):
        match_id += 1
        rows = []
        for innings in (1, 2):
            for over in range(20):
                phase = 0 if over < 6 else (1 if over < 15 else 2)
                bowler = bowlers[rng.integers(0, len(bowlers))]
                for ball in range(6):
                    striker = batters[rng.integers(0, len(batters))]
                    theta = (PHASE_TILT[phase] + bat_skill[striker]
                             - bowl_skill[bowler] + pair_effect[(striker, bowler)])
                    w = BASE_P * np.exp(theta * RUNS)
                    runs = int(rng.choice(RUNS, p=w / w.sum()))
                    p_wk = PHASE_WK[phase] * np.exp(
                        bat_skill[striker] * -1 + bowl_skill[bowler] * -1
                        - 0.5 * pair_effect[(striker, bowler)]
                    )
                    wicket = rng.random() < p_wk
                    wide = int(rng.random() < 0.03)
                    rows.append({
                        "match_id": match_id, "season": str(season),
                        "start_date": f"{season}-04-01", "venue": "Sample Ground",
                        "innings": innings, "ball": f"{over}.{ball + 1}",
                        "batting_team": "Team A" if innings == 1 else "Team B",
                        "bowling_team": "Team B" if innings == 1 else "Team A",
                        "striker": striker, "non_striker": batters[0],
                        "bowler": bowler,
                        "runs_off_bat": 0 if wide else runs,
                        "extras": wide, "wides": wide or "", "noballs": "",
                        "byes": "", "legbyes": "", "penalty": "",
                        "wicket_type": "caught" if wicket else "",
                        "player_dismissed": striker if wicket else "",
                        "other_wicket_type": "", "other_player_dismissed": "",
                    })
        pd.DataFrame(rows, columns=COLUMNS).to_csv(OUT / f"{match_id}.csv", index=False)

print(f"Wrote synthetic sample matches to {OUT}")

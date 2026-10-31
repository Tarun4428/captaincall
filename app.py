"""
CaptainCall - app.py
Streamlit dashboard: "Who should bowl the next over?"

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    PHASES,
    RAW_DIR,
    MatchupModel,
    Params,
    load_data,
    recent_players,
    top_bowlers_in_phase,
    tune_and_validate,
)

st.set_page_config(page_title="CaptainCall - IPL bowling matchups", page_icon="🏏", layout="wide")


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading ball-by-ball data...")
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_data(show_spinner="Validating the model on a held-out season...")
def get_validation():
    try:
        return tune_and_validate(get_data())
    except ValueError:
        return None


@st.cache_resource(show_spinner="Fitting the final model...")
def get_model() -> MatchupModel:
    val = get_validation()
    params = Params(**val["params"]) if val else Params()
    return MatchupModel(params).fit(get_data())


# --------------------------------------------------------------------------
# Load data (with a friendly message if it is missing)
# --------------------------------------------------------------------------
try:
    df = get_data()
except FileNotFoundError:
    st.title("🏏 CaptainCall")
    st.error("No data found yet.")
    st.markdown(
        f"1. Download the **IPL ball-by-ball CSV** zip from cricsheet.org.\n"
        f"2. Extract all files into `{RAW_DIR}`.\n"
        f"3. Refresh this page."
    )
    st.stop()

model = get_model()
validation = get_validation()
latest_season = int(df["season"].max())

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🏏 CaptainCall")
st.caption(
    "Who should bowl the next over? Ranks your bowlers against the batter at the crease, "
    "with uncertainty ranges and honest low-confidence flags."
)

# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Match situation")
    phase_name = st.selectbox("Phase of the innings", PHASES, index=2)
    phase = PHASES.index(phase_name)

    batters = recent_players(df, "batter", n_seasons=3, min_balls=100)
    batter = st.selectbox("Batter at the crease", list(batters.index), index=0)

    bowler_pool = recent_players(df, "bowler", n_seasons=3, min_balls=60)
    defaults = [
        b for b in top_bowlers_in_phase(df, phase, latest_season, 6) if b in bowler_pool.index
    ]
    chosen = st.multiselect(
        "Bowlers with overs left",
        list(bowler_pool.index),
        default=defaults,
        key=f"bowlers_{phase}",
    )

    lam = st.slider(
        "Value of a wicket (in runs)",
        min_value=0,
        max_value=30,
        value=12,
        help="How many runs of cost one wicket is worth to you. This is an assumption you "
        "control, not something the data decides. 0 = only care about runs conceded.",
    )

tab_rec, tab_val, tab_how = st.tabs(["Recommendation", "Model validation", "How it works"])

# --------------------------------------------------------------------------
# Tab 1: recommendation
# --------------------------------------------------------------------------
with tab_rec:
    if len(chosen) < 2:
        st.info("Pick at least two bowlers in the sidebar to compare them.")
    else:
        tbl = model.matchup_table(batter, chosen, phase, lam)
        best = tbl.iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Suggested bowler", best["bowler"])
        c2.metric("Expected runs / over", f"{best['runs_over']:.1f}")
        c3.metric("Expected wickets / over", f"{best['wk_over']:.2f}")

        if len(tbl) > 1 and tbl.loc[1, "net_lo"] <= best["net_hi"]:
            st.warning(
                f"Close call: the ranges for **{best['bowler']}** and "
                f"**{tbl.loc[1, 'bowler']}** overlap. The data cannot separate them "
                "confidently, so use match context as the tie-breaker."
            )
        if str(best["confidence"]).startswith("Low"):
            st.warning(
                f"Little head-to-head history between {batter} and {best['bowler']}: "
                "this estimate leans mostly on each player's general record in this phase."
            )

        # Chart: net cost with 80% range
        fig = go.Figure(
            go.Bar(
                x=tbl["net_cost"],
                y=tbl["bowler"],
                orientation="h",
                error_x=dict(
                    type="data",
                    array=(tbl["net_hi"] - tbl["net_cost"]).tolist(),
                    arrayminus=(tbl["net_cost"] - tbl["net_lo"]).tolist(),
                ),
                marker_color=["#2e7d32"] + ["#90a4ae"] * (len(tbl) - 1),
            )
        )
        fig.update_layout(
            title=f"Net cost per over vs {batter} ({phase_name}) - lower is better",
            xaxis_title=f"Expected runs per over minus {lam} x expected wickets per over",
            yaxis_title="",
            height=120 + 60 * len(tbl),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig)

        show = pd.DataFrame(
            {
                "Bowler": tbl["bowler"],
                "Exp. runs/over": tbl["runs_over"].round(1),
                "Runs 80% range": [
                    f"{lo:.1f} - {hi:.1f}" for lo, hi in zip(tbl["runs_lo"], tbl["runs_hi"])
                ],
                "Exp. wickets/over": tbl["wk_over"].round(2),
                "Wickets 80% range": [
                    f"{lo:.2f} - {hi:.2f}" for lo, hi in zip(tbl["wk_lo"], tbl["wk_hi"])
                ],
                "Net cost": tbl["net_cost"].round(1),
                "Head-to-head balls": tbl["h2h_balls"],
                "Balls bowled in phase": tbl["bowler_phase_balls"],
                "Confidence": tbl["confidence"],
            }
        )
        st.dataframe(show, hide_index=True)
        st.caption(
            "Ranges are approximate 80% intervals on the EXPECTED value (not on one over's "
            "outcome; a single over is always very noisy). Confidence is based on the number "
            "of balls this exact batter has faced from this bowler."
        )

# --------------------------------------------------------------------------
# Tab 2: validation
# --------------------------------------------------------------------------
with tab_val:
    if validation is None:
        st.info("Validation needs at least 4 seasons of data.")
    else:
        v = validation
        st.subheader(f"Tested on the {v['test_season']} season (never seen in training)")
        st.write(
            f"Trained on seasons {v['train_seasons']}. Head-to-head shrinkage was tuned on "
            f"{v['tuning_season']} using only earlier seasons. Test set: "
            f"{v['n_test_balls']:,} legal balls. Tuned settings: "
            f"runs k = {v['params']['k_h2h_r']:g}, wickets k = {v['params']['k_h2h_w']:g}."
        )

        m = v["metrics"].copy()
        for col in m.columns[1:]:
            m[col] = m[col].round(2 if "%" in col else 4)
        st.dataframe(m, hide_index=True)
        st.caption(
            "Lower error and log-loss are better. The % columns show the improvement over simply "
            "using the league average for that phase (positive = better). Single-ball outcomes "
            "are extremely noisy, so even a good model improves per-ball error only modestly. "
            "The table below shows whether the ranking is still useful in practice."
        )

        lift = v["lift"]
        st.subheader("Do bowler-favoured matchups really go better for the bowler?")
        st.dataframe(lift.round(3), hide_index=True)
        fig2 = go.Figure()
        fig2.add_bar(x=lift.iloc[:, 0], y=lift["Predicted runs/over"], name="Predicted")
        fig2.add_bar(x=lift.iloc[:, 0], y=lift["Actual runs/over"], name="Actual")
        fig2.update_layout(
            barmode="group",
            yaxis_title="Runs per over",
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig2)
        st.caption(
            "Test balls are split into five equal groups by the model's predicted runs per ball "
            "(Group 1 = most bowler-friendly). If the model is useful, actual runs should rise "
            "from Group 1 to Group 5 and track the predictions."
        )

# --------------------------------------------------------------------------
# Tab 3: method and limitations
# --------------------------------------------------------------------------
with tab_how:
    st.markdown(
        """
### Method
1. **Data:** ball-by-ball IPL records, one row per delivery. Runs charged to the bowler are bat
   runs + wides + no-balls. Run-outs and retirements are not counted as bowler wickets.
2. **Phase effects:** the league average runs per ball and wicket probability are computed
   separately for the powerplay, middle overs and death overs.
3. **Shrinkage:** each batter's and bowler's rate in a phase is pulled towards the league average
   in proportion to how few balls they have (like adding "pseudo-balls" of average data).
   This stops a bowler with 12 lucky balls looking like a star.
4. **Head-to-head:** a batter-vs-bowler record is compared with what the model expected, and the
   result is shrunk heavily towards "no special effect". It only moves the estimate when
   there is real history.
5. **Uncertainty:** ranges come from the effective sample size behind each estimate, and a
   Beta distribution for wickets.
6. **Validation:** time-ordered. Train on earlier seasons, test on a later one, compare with simple
   baselines a captain might use.

### Limitations
- Overs are treated as independent; match state (score, wickets in hand, target) is not modelled.
- The model does not know pitch, dew, bowler type, field settings or fatigue.
- The wicket value slider is a user assumption.
- Ranges are approximate and cover the expected value, not one over's outcome.
- Players with little data in a phase are mostly estimated from the league average.

### Data credit
Ball-by-ball data from Cricsheet (cricsheet.org). Check and follow its licence terms.
        """
    )

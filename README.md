# CaptainCall: who should bowl the next over?

Built for the AQX Sports Analytics Data Bowl 4.0.

**Problem:** In T20 cricket a captain must pick the next bowler against the batter at the crease,
usually from gut feel or a bowler's overall economy rate. Head-to-head records are too small to
trust on their own, and nothing tells the captain how sure the numbers are.

**Solution:** An IPL dashboard that ranks the available bowlers against a chosen batter in a chosen
phase (powerplay, middle, death). It uses shrinkage (empirical-Bayes style) so small samples do not
mislead, adds a heavily shrunk head-to-head adjustment, shows 80% ranges, and flags close calls and
low-confidence matchups. It is validated on a held-out season against simple baselines.

**Actionable impact:** A captain or analyst gets a ranked bowler list plus an honest signal of when
the data cannot separate two options.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# put the Cricsheet IPL CSV files in data/ipl/
python model.py                   # builds the cache, prints the validation report
streamlit run app.py
```

## Files
- `model.py`: loading, matchup model, validation (also runnable from the terminal)
- `app.py`: Streamlit dashboard
- `make_sample_data.py`: synthetic data for testing only, never for results

## Data
Ball-by-ball IPL data from Cricsheet (https://cricsheet.org). Please follow its licence terms and credit it.

## Limitations
Match state (score, wickets in hand, target) is not modelled; pitch, dew, bowler type and fatigue are
ignored; the wicket-value slider is a user assumption; ranges describe the expected value, not one over.

## Licence
MIT (see LICENSE).

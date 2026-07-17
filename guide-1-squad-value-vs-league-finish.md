# Guide 1: Predicting League Finish from Squad Value (Regression)

**Goal:** Does spending more on players lead to a better league finish? We build a
model that takes a club's total squad market value and predicts where they'll
finish in the league table.

This guide is written so you can follow it top to bottom, run every cell
yourself, and understand *why* each step exists — not just copy-paste.

---

## 0. The big picture (read this first)

Here is the whole project in one paragraph, so you never lose the thread:

The dataset gives us two things that don't directly talk to each other: a list
of match results (`games.csv`) and a list of clubs with their money value
(`clubs.csv`). Nobody hands us a "league table." **We have to build the league
table ourselves** from raw match results — that's a real data engineering
task. Once we have a league table (points, position) and a squad value for
each club, we join them into one table and ask: can `squad_value` predict
`position`? That's a regression problem, because position is a number
(1st, 2nd, 3rd... not a category).

---

## 1. Folder structure

Create this structure. Empty folders won't show up in git — that's fine,
they'll fill up as we go.

```
squad-value-vs-finish/
├── data/
│   └── raw/                  # original CSVs, never edited by hand
├── notebooks/
│   └── 01_model.ipynb        # the whole build lives here
├── models/
│   └── model.pkl             # saved trained model (created later)
├── app/
│   └── app.py                # streamlit app
├── requirements.txt
└── README.md
```

**Why this shape?** `data/raw` never gets modified — if you mess up a
cleaning step, you can always reload the original file. `notebooks/` is
where we think out loud. `models/` and `app/` are the "finished product"
folders, separate from the messy exploration.

---

## 2. GitHub setup

On github.com:
1. Click **New repository**
2. Name it `squad-value-vs-finish`
3. Tick **Add a README**
4. Create it

On your machine, in a terminal (or VSCode's terminal, `` Ctrl+` ``):

```bash
git clone https://github.com/<your-username>/squad-value-vs-finish.git
cd squad-value-vs-finish
```

Now build the folders from Step 1 inside this cloned repo:

```bash
mkdir -p data/raw notebooks models app
```

---

## 3. Local environment setup

We use a **virtual environment** so this project's packages don't clash with
any other project's packages on your machine.

```bash
python -m venv venv
```

Activate it:

- Mac/Linux: `source venv/bin/activate`
- Windows (PowerShell): `venv\Scripts\Activate.ps1`

You'll know it worked because your terminal prompt now starts with `(venv)`.

Create `requirements.txt` in the project root:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
joblib
streamlit
```

Install everything:

```bash
pip install -r requirements.txt
```

**What each package is for**, so it's not a mystery:
- `pandas` — loading and manipulating tables of data
- `numpy` — number crunching under the hood
- `matplotlib` / `seaborn` — charts
- `scikit-learn` — the actual machine learning models
- `jupyter` — the notebook interface we'll code in
- `joblib` — saves the trained model to a file
- `streamlit` — turns our model into a clickable web app

Commit this so far:

```bash
git add .
git commit -m "Initial project structure and requirements"
git push
```

---

## 4. Get the data

1. Go to https://www.kaggle.com/datasets/davidcariboo/player-scores
2. Click **Download** (you'll need a free Kaggle account)
3. Unzip it
4. Copy at minimum these two files into `data/raw/`:
   - `games.csv`
   - `clubs.csv`

We are **not** uploading this data to GitHub — sports datasets like this
can be large and it's good practice to keep raw data out of git. Create a
`.gitignore` file in the project root with:

```
venv/
data/raw/*.csv
```

---

## 5. Launch Jupyter

```bash
jupyter notebook
```

This opens in your browser. Navigate into `notebooks/`, create a new
notebook, and rename it `01_model.ipynb`.

Every code block below is one cell. Run them **in order**, top to bottom —
don't skip around. If a cell errors, read the error message; it usually
tells you exactly what's wrong (wrong column name, wrong file path, etc.).

---

## Part A — Load and inspect the data

```python
import pandas as pd

# Load the two raw files we need
games = pd.read_csv("../data/raw/games.csv")
clubs = pd.read_csv("../data/raw/clubs.csv")

# Always look at what you loaded before doing anything else
print(games.shape)   # (rows, columns) — how big is this table?
print(clubs.shape)
```

**Why `.shape` first?** It's the fastest sanity check that the file loaded
correctly. If you expected thousands of rows and got 3, something's wrong
before you've wasted time analyzing it.

```python
games.head()
```

```python
clubs.head()
```

**Why `.head()`?** Shows the first 5 rows so you can *see* the data — column
names, what kind of values live in them, whether anything looks broken at a
glance.

```python
games.info()
```

```python
clubs.info()
```

**What `.info()` tells you:** the data type of every column (number, text,
date) and how many non-null values each column has. This is your first
clue about missing data — if a column has fewer non-null values than the
total row count, some rows are missing that value.

> **Beginner checkpoint:** before continuing, run `games.columns.tolist()`
> and `clubs.columns.tolist()` and actually read the list. This dataset is
> updated weekly by its maintainer, so column names can shift slightly over
> time. The code in this guide uses the most common column names
> (`home_club_goals`, `away_club_goals`, `total_market_value`, etc.) — if
> yours differ, that's fine, just swap the name in. **Never assume — always
> verify against what's actually in front of you.** This habit will save you
> hours of confusion later.

---

## Part B — Data cleaning (skip nothing)

This is the part beginners rush through and shouldn't. A model trained on
dirty data gives you confident, wrong answers.

### B1. Duplicates

```python
# Count exact duplicate rows
print("Duplicate games:", games.duplicated().sum())
print("Duplicate clubs:", clubs.duplicated().sum())

# Remove them if any exist
games = games.drop_duplicates()
clubs = clubs.drop_duplicates()
```

**Why this matters:** if the same match appears twice, that team's win gets
counted twice when we build the league table two steps from now — an
inflated, wrong points total.

### B2. Missing values

```python
# How many missing (null) values per column?
print(games.isnull().sum())
```

```python
print(clubs.isnull().sum())
```

**Why this matters:** a missing `home_club_goals` means we can't tell who
won that match — we can't safely guess a football score, so we must decide
whether to drop that row or investigate it. Guessing invented scores would
poison the league table with fake results.

```python
# Drop games where we don't know the final score — we can't use them
games = games.dropna(subset=["home_club_goals", "away_club_goals"])
```

For `clubs`, a missing `total_market_value` means we simply can't use that
club in this project (no input = no prediction possible for that row):

```python
clubs = clubs.dropna(subset=["total_market_value"])
```

### B3. Correct data types

```python
print(games.dtypes)
```

Goals should be whole numbers and dates should be dates, not text:

```python
games["date"] = pd.to_datetime(games["date"])
games["home_club_goals"] = games["home_club_goals"].astype(int)
games["away_club_goals"] = games["away_club_goals"].astype(int)
```

**Why this matters:** if `date` is stored as plain text, you can't filter
"games after 2020" or sort chronologically — pandas would sort it
alphabetically instead of by actual date order.

### B4. Outlier check

```python
games[["home_club_goals", "away_club_goals"]].describe()
```

**Reading `.describe()`:** `mean` is the average, `50%` is the median
(middle value), `max` is the highest score in the data. A real football
match rarely has double-digit goals — if `max` shows something like `40`,
that's almost certainly a data entry error worth investigating, not a real
scoreline.

### B5. Category consistency

```python
# Check that competition names/ids don't have inconsistent spellings
print(games["competition_id"].value_counts())
```

**Why this matters:** if the same league appears as both `"GB1"` and
`"gb1"` (different case) due to a scraping glitch, pandas treats them as
two different leagues, silently splitting your data and skewing results.
This step catches that.

---

## Part C — Build the league table (the core data engineering step)

There is no ready-made "league table" file. We build one from match
results. This is the single most important lesson in this project: **real
data science is often building the thing you need from raw material, not
finding it pre-made.**

**The logic, in plain words:**
For a chosen season and league, go through every match. The home team and
away team each earn points based on the result: **3 points for a win, 1
for a draw, 0 for a loss.** Add up every team's points across all their
matches in that season. Sort teams by total points, highest first — that
sorted order *is* the league table, and each team's row number is their
final **position**.

```python
# Focus on one league + one season to start simple.
# GB1 = Premier League in this dataset's coding; adjust the season to one
# that's fully finished in your data.
season_games = games[
    (games["competition_id"] == "GB1") &
    (games["season"] == 2022)
].copy()

print(season_games.shape)
```

Now we compute each match's result from each team's point of view. A
single match produces **two rows of outcome** — one for the home side, one
for the away side:

```python
# Build a "home team perspective" table
home = season_games[["home_club_id", "home_club_goals", "away_club_goals"]].copy()
home.columns = ["club_id", "goals_for", "goals_against"]

# Build an "away team perspective" table
away = season_games[["away_club_id", "away_club_goals", "home_club_goals"]].copy()
away.columns = ["club_id", "goals_for", "goals_against"]

# Stack them into one long table: one row per team per match
team_matches = pd.concat([home, away], ignore_index=True)
team_matches.head()
```

**Why split into "home" and "away" then stack them?** Every match involves
two teams, but our source data has one row per *match*, not per *team*.
We need one row per *team per match* to count each team's individual wins,
draws and losses — so we reshape the same information from two
perspectives and combine it.

```python
def points_for_result(row):
    """3 points for a win, 1 for a draw, 0 for a loss."""
    if row["goals_for"] > row["goals_against"]:
        return 3
    elif row["goals_for"] == row["goals_against"]:
        return 1
    else:
        return 0

team_matches["points"] = team_matches.apply(points_for_result, axis=1)
```

**Why a function instead of writing this inline?** It reads like English
and can be tested/reused. `axis=1` tells pandas to apply it row by row.

```python
# Now group by club and total everything up across the whole season
league_table = team_matches.groupby("club_id").agg(
    played=("points", "count"),
    goals_for=("goals_for", "sum"),
    goals_against=("goals_against", "sum"),
    points=("points", "sum"),
).reset_index()

# Goal difference is a classic tiebreaker in real league tables
league_table["goal_difference"] = league_table["goals_for"] - league_table["goals_against"]

# Sort: most points first, goal difference breaks ties
league_table = league_table.sort_values(
    by=["points", "goal_difference"], ascending=False
).reset_index(drop=True)

# Position = row number after sorting, starting at 1
league_table["position"] = league_table.index + 1

league_table.head(20)
```

**What each column means:**
- `played` — matches played that season
- `goals_for` — total goals this team scored
- `goals_against` — total goals scored against them
- `points` — total points (3/1/0 per match, summed)
- `goal_difference` — goals_for minus goals_against; a form of "how
  dominant," used to break ties on points
- `position` — final league rank, 1 = champions

---

## Part D — Bring in squad value

Now we attach each club's squad value to the league table we just built.

```python
squad_values = clubs[["club_id", "name", "total_market_value"]]

final_table = league_table.merge(squad_values, on="club_id", how="left")
final_table.head(20)
```

**Why `how="left"`?** We keep every row from `league_table` (our real
match data) and attach squad value where it matches. If a club_id has no
match in `clubs`, we'd rather see a missing value and investigate than
silently lose that team's row.

```python
# Check: did the merge leave any clubs without a value?
print(final_table["total_market_value"].isnull().sum())

# Drop any club we couldn't attach a value to — we can't use it as a training example
final_table = final_table.dropna(subset=["total_market_value"])
```

> **Note on scope:** `clubs.csv` stores each club's most recently recorded
> squad value, not a separate historical value for every past season. That
> means this project answers "does current squad value relate to that
> club's most recent finished season's position" rather than tracking value
> season-by-season across many years. That's a real limitation of the
> data, worth knowing rather than hiding.

---

## Part E — Exploratory Data Analysis (EDA)

```python
import matplotlib.pyplot as plt

plt.scatter(final_table["total_market_value"], final_table["position"])
plt.xlabel("Squad value (EUR)")
plt.ylabel("League position")
plt.gca().invert_yaxis()  # position 1 (best) at the top of the chart
plt.title("Squad Value vs League Position")
plt.show()
```

**What are we looking for?** If money buys success, we'd expect a
downward-sloping cloud of points: higher value, lower (better) position
number. If the cloud looks random, value alone doesn't explain finish —
which is itself a real, useful finding.

```python
final_table[["total_market_value", "position"]].corr()
```

**Reading correlation:** a number between -1 and 1. Close to -1 means
"as value goes up, position number goes down" (i.e., value predicts
success strongly). Close to 0 means barely related. This single number is
a preview of what our model is about to try to learn properly.

---

## Part F — Feature engineering

```python
# Our one input (feature) for now: squad value
X = final_table[["total_market_value"]]

# What we're predicting (target): league position
y = final_table["position"]
```

**Why start with just one feature?** So you can clearly see whether *this
one thing* explains the outcome, before adding complexity. You can extend
this later with `average_age`, `squad_size`, `foreigners_number` from
`clubs.csv` — same process, just add more columns to `X`.

---

## Part G — Train/test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why split the data at all?** If we test the model on the same data it
learned from, it can just "memorize" answers and look perfect while
actually being useless on new clubs. `test_size=0.2` holds back 20% of
rows the model never sees during training, purely to check its real
performance afterward. `random_state=42` just makes the split repeatable
— you and your student will get the exact same split every time you run
this.

---

## Part H — Baseline model: Linear Regression

```python
from sklearn.linear_model import LinearRegression

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)
```

**Why start with the simplest possible model?** Linear Regression assumes
a straight-line relationship between value and position. It's fast, easy
to explain, and gives you a floor to beat. If a complicated model can't
beat this, the complexity wasn't worth it.

---

## Part I — Better model: Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)
```

**What's different here?** A Random Forest builds many decision trees on
random slices of the data and averages their answers. It can capture
bends and curves in the relationship that a straight line can't — useful
if, say, the very richest clubs behave differently from mid-table ones.
`n_estimators=200` means 200 trees are built and averaged.

---

## Part J — Evaluate both models properly

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")

evaluate("Linear Regression", y_test, baseline_predictions)
evaluate("Random Forest", y_test, rf_predictions)
```

**What each metric means, in plain words:**
- **MAE (Mean Absolute Error)** — on average, how many league positions
  off is the model? MAE of 3 means predictions are typically 3 places off
  the real finish.
- **RMSE (Root Mean Squared Error)** — similar to MAE but punishes big
  misses harder (a 10-place miss hurts RMSE much more than two 5-place
  misses). Use this when big errors are especially bad.
- **R² (R-squared)** — how much of the variation in league position is
  explained by squad value, from 0 (none) to 1 (perfectly explained).
  An R² of 0.6 means squad value explains 60% of why teams finish where
  they do; the rest comes from other factors (form, injuries, manager,
  luck).

Whichever model has lower MAE/RMSE and higher R² is the one we keep.

---

## Part K — Save the model

```python
import joblib

# Save whichever model performed better — assume Random Forest here
joblib.dump(rf_model, "../models/model.pkl")
```

**Why save it?** Training takes time and the notebook won't run every time
someone wants a prediction. We save the trained model to a file once, and
the Streamlit app below just loads that file instantly.

Commit your work:

```bash
git add .
git commit -m "Add league table build, cleaning, model training and evaluation"
git push
```

---

## Part L — Streamlit app

Create `app/app.py`:

```python
import streamlit as st
import joblib
import numpy as np

# Load the model we trained and saved earlier
model = joblib.load("../models/model.pkl")

st.title("Squad Value → League Position Predictor")
st.write(
    "Enter a club's total squad market value to estimate "
    "where they'd finish in the league."
)

# A number input the user can type into or adjust with arrows
squad_value = st.number_input(
    "Squad market value (EUR)",
    min_value=0,
    value=100_000_000,
    step=1_000_000,
)

if st.button("Predict league position"):
    prediction = model.predict(np.array([[squad_value]]))
    st.success(f"Estimated league position: {round(prediction[0])}")
```

**Line by line, what's happening:**
- `joblib.load(...)` — brings back the exact trained model from Part K,
  no retraining needed
- `st.title` / `st.write` — page heading and description text
- `st.number_input` — a box the user types a number into
- `st.button` — nothing happens until this is clicked; keeps the app from
  predicting on every keystroke
- `model.predict(...)` — the model expects a 2D array (rows of features),
  so we wrap the single number in `[[ ]]`

Run it locally:

```bash
cd app
streamlit run app.py
```

This opens in your browser at `localhost:8501`.

---

## Part M — Deploy to Streamlit Community Cloud

1. Make sure `app/app.py`, `models/model.pkl`, and `requirements.txt` are
   all committed and pushed to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick your `squad-value-vs-finish` repo, branch
   `main`, and file path `app/app.py`.
4. Click **Deploy**.

Streamlit installs your `requirements.txt` on their server and runs your
app — you get a public URL your student can open on any device.

> **Common snag:** the app's file path (`../models/model.pkl`) is relative
> to where `app.py` runs from. If Streamlit Cloud can't find the model
> file, switch to a path relative to the project root, e.g.
> `models/model.pkl`, and run the app with
> `streamlit run app/app.py` from the project root instead of from inside
> `app/`.

---

## Recap: what was actually learned here

1. Loading and inspecting real-world data before trusting it
2. A full data cleaning pass: duplicates, missing values, wrong types,
   outliers, inconsistent categories
3. Real data engineering — building a table that didn't exist from raw
   match data (this is most of what data scientists actually do day to
   day)
4. Joining two different data sources on a shared key
5. EDA to sanity-check an idea before modeling it
6. Train/test splitting and why it matters
7. Baseline vs improved model comparison
8. Regression metrics: MAE, RMSE, R² and what each one actually tells you
9. Saving a model and serving it through a real, deployed web app

---

## Further reading

- Pandas `groupby`: https://pandas.pydata.org/docs/user_guide/groupby.html
- scikit-learn Linear Regression: https://scikit-learn.org/stable/modules/linear_model.html
- scikit-learn Random Forest: https://scikit-learn.org/stable/modules/ensemble.html#forest
- Understanding R², MAE, RMSE (plain-English): https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
- Streamlit docs: https://docs.streamlit.io
- Transfermarkt dataset source/schema: https://github.com/dcaribou/transfermarkt-datasets

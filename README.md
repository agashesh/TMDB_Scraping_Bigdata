# TMDB Movie Revenue Predictor

A data collection and predictive modeling project that pulls movie metadata
from [The Movie Database (TMDB)](https://www.themoviedb.org/) API and uses it
to predict box office revenue with Simple and Multiple Linear Regression.
Built for the Big Data & Predictive Analytics course (ST152), Informatics,
Universitas Amikom Yogyakarta.

This project was co-developed with the help of Claude (Anthropic). Claude
helped scaffold the scraping/API code, the regression pipeline, and the
Streamlit dashboard. Data collection decisions, interpretation of the
regression results, and the final report were done by me.

## Features

- Collects movie metadata for ~1,600+ titles via the official TMDB API
  (`discover/movie` + `movie/{id}` endpoints)
- Filters out incomplete records (zero budget/revenue/runtime) during cleaning
- Correlation analysis across 6 numeric variables before modeling
- Simple Linear Regression (budget → revenue) and Multiple Linear Regression
  (budget, runtime, popularity, vote_average, vote_count → revenue)
- Classical assumption checks for the multiple regression model:
  multicollinearity (VIF), heteroscedasticity (Breusch-Pagan), residual
  normality (Shapiro-Wilk)
- Interactive Streamlit dashboard with filters, charts, and a live
  "predict your own movie" form

## Requirements

- **Python** 3.10 or later
- A free TMDB API key (v3 auth) — register at themoviedb.org
- Dependencies listed in `requirements.txt`

## Installation

```bash
git clone <this-repo-url>
cd tmdb-revenue-predictor
pip install -r requirements.txt
```

## Usage

The data collection and analysis steps are notebooks rather than CLI scripts,
since they were built and run in Google Colab.

```bash
# 1. Open scraping_TMDB.ipynb, add your TMDB API key, run all cells
#    -> produces movies_dataset.csv

# 2. Open analysis_regression.ipynb, run all cells
#    -> correlation heatmap, regression models, assumption tests, evaluation

# 3. Run the dashboard locally
streamlit run app.py
```

The dashboard is also deployed here: [dashboard link]

## Dataset Schema

`movies_dataset.csv` contains the following columns after cleaning:

| Column | Description | Example |
|---|---|---|
| id | TMDB movie ID | 27205 |
| title | Movie title | Inception |
| budget | Production budget (USD) | 160000000 |
| revenue | Box office revenue (USD) | 836848102 |
| runtime | Runtime in minutes | 148 |
| popularity | TMDB popularity score | 32.4 |
| vote_average | Average user rating (0–10) | 8.4 |
| vote_count | Number of user votes | 34000 |
| release_date | Release date | 2010-07-15 |
| genres | Semicolon-free, comma-separated genre list | Action, Science Fiction |

Rows with `budget`, `revenue`, or `runtime` equal to 0 are dropped during
cleaning, since a large share of entries on TMDB are missing this data.

## Expected Runtime

Since TMDB's rate limit is generous for an official API (no fixed cap since
December 2019, informally observed around 40–50 requests/second), the
scraping notebook uses a light 0.3 second delay between requests rather than
long delays meant for anti-bot evasion. Collecting ~2,500 candidate IDs and
their details takes roughly **10–15 minutes** end to end.

## Project Structure

```
tmdb-revenue-predictor/
├── scraping_TMDB.ipynb        # TMDB API calls, ID collection, cleaning, CSV export
├── analysis_regression.ipynb  # EDA, correlation, regression, assumption tests
├── movies_dataset.csv         # Cleaned dataset (1,589 rows)
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Results Summary

| Metric | Simple LR (budget only) | Multiple LR (5 variables) |
|---|---|---|
| R² | 0.4274 | 0.5954 |
| MAE | $131,398,235 | $113,804,543 |
| RMSE | $228,436,635 | $192,010,872 |

Budget is the strongest predictor of revenue, followed by vote_count; runtime
has almost no effect. The multiple regression model explains more variance,
but residuals show heteroscedasticity and non-normality — expected given how
right-skewed movie revenue data tends to be (a small number of blockbusters
dominate the distribution).

## Limitations

- **TMDB data only** — no data merged from IMDb, Box Office Mojo, or other
  sources, so figures like inflation-adjusted revenue aren't available.
- **Linear models only** — no attempt at polynomial, tree-based, or
  log-transformed regression, which would likely fit the skewed revenue
  distribution better.
- **No genre-level modeling** — genres are collected but not one-hot encoded
  or used as predictors in the current models.

## Ethical Considerations

> This project is for academic purposes only. All movie data belongs to TMDB
> and its contributors. TMDB was chosen specifically because its API is free
> for non-commercial use and its Terms of Use permit this kind of data
> collection — unlike sites such as IMDb, which explicitly prohibit scraping
> without written permission. No scraping of IMDb, Goodreads, or any site
> that disallows automated data collection was performed for this project.

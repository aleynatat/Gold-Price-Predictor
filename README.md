# Gold-Price-Predictor

# AI Gold Price Predictor (End-to-End ML Pipeline)

## Project Overview
This project is an end-to-end Machine Learning web application that predicts tomorrow's Gold Close Price based on today's market data. The project includes extensive exploratory data analysis (EDA), feature engineering, rigorous model benchmarking, hyperparameter tuning, and a FastAPI backend serving a responsive frontend.

## Why Lasso Regression?
During the model benchmarking phase, 11 different algorithms were tested (including Random Forest, XGBoost, LightGBM, SVR, and Linear Models). 
* **The Tree-Based Failure:** Advanced ensemble models like Random Forest and XGBoost completely failed at predicting new all-time highs due to their inability to extrapolate beyond the maximum values seen in the training data.
* **The Linear Victory:** Simple linear models captured the trend perfectly. 
* **Feature Selection Magic:** Our champion model, **Lasso Regression**, zeroed out 39 out of 40 engineered features (oil prices, platinum, volume, dates), proving the *Random Walk Hypothesis* for daily predictions: **The best predictor of tomorrow's price is today's price.**

## Tech Stack
* **Data Science:** Pandas, NumPy, Scikit-Learn
* **Machine Learning:** Lasso Regression, TimeSeriesSplit CV, GridSearchCV
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Frontend:** HTML, JavaScript, CSS (Fetch API)

## Limitations & Future Work (Product Vision)
I acknowledge that predicting asset prices based solely on yesterday's price is mathematically sound for a baseline, but practically limited. In reality, financial markets are highly driven by real-world events, not just historical numbers.

**Future improvements will include:**

* Global News & Sentiment Analysis (NLP): Integrating live world news to gauge geopolitical tensions, wars, and global events that instantly affect gold prices.
* Macro Indicators: Adding Federal Reserve interest rate decisions and DXY (Dollar Index) to the prediction pipeline.
* Live API Integration: Replacing manual price input with automated live data fetching via yfinance API.

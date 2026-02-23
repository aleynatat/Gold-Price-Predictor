import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('financial_regression.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.info())

# Drop columns with more than 30% missing values
threshold = int(len(df) * 0.8)
df = df.dropna(thresh=threshold, axis=1)

print("Remaining column count:", len(df.columns))
print(df["date"].unique())

# --- FIXED BUG: MOVED DATETIME ENGINEERING BEFORE DUPLICATE CHECK ---
df['date'] = pd.to_datetime(df['date'])

# Type should be 'datetime64'
print(df['date'].dtype)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Day of the week (0=Monday, 6=Sunday)
df['day_of_week'] = df['date'].dt.dayofweek

# Is it a weekend?
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# Is month start/end?
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

df = df.drop(columns=['date'])
print(df.info())

# Drop rows with NaN values completely
df = df.dropna()

# New shape of the dataset
print("New shape of the dataset (Rows, Columns):", df.shape)

# Check for multiple records on the exact same date
date_duplicates = df.duplicated(subset=['year', 'month', 'day']).sum()
print(f"Duplicate records for the same day: {date_duplicates}")

# Find completely identical rows across all columns
total_duplicates = df.duplicated().sum()
print(f"Total duplicate rows in the dataset: {total_duplicates}")

# Drop duplicates if any exist
if total_duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate data successfully cleaned! New row count:", len(df))
else:
    print("No duplicate data found to clean!")
# --------------------------------------------------------------------

# Combine Year, Month, Day to create a 'Date' axis for visualization
actual_dates = pd.to_datetime(df[['year', 'month', 'day']])

# Plotting Gold Close Prices Over Time
plt.figure(figsize=(14, 6))
import matplotlib.dates as mdates

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.plot(actual_dates, df['gold close'], color='gold', label='Gold Close Price')
plt.title('Gold Close Prices Over Time')
plt.xlabel('Date (Year)')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Select gold-related columns for correlation
gold_columns = df[['gold open', 'gold high', 'gold low', 'gold close']]
correlation = gold_columns.corr()

print("--- Correlation Between Gold Columns ---")
print(correlation)

# Boxplot for Outlier Detection
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['gold close'], color='orange')
plt.title('Gold Close Price - Outlier Check')
plt.show()

# Subset of commodity close prices
close_prices = df[['gold close', 'oil close', 'platinum close', 'palladium close']]
correlation_matrix = close_prices.corr()

# Find columns containing 'open', 'high', or 'low' (except 'high-low')
cols_to_drop = [col for col in df.columns if ('open' in col or 'high' in col or 'low' in col) and col != 'high-low']

# Drop those columns
df = df.drop(cols_to_drop, axis=1)

print("Remaining Columns:\n", df.columns)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Between Commodity Close Prices')
plt.show()

# Pairplot
sns.pairplot(close_prices)
plt.show()

# 1st Quartile (Q1)
Q1 = df['gold close'].quantile(0.25)

# 3rd Quartile (Q3)
Q3 = df['gold close'].quantile(0.75)

# IQR (Interquartile Range)
IQR = Q3 - Q1

# Lower and Upper Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['gold close'] < lower_bound) | (df['gold close'] > upper_bound)]

print(f"Number of days outside normal market conditions: {len(outliers)}")
print(f"Lower bound: {lower_bound:.2f} USD | Upper bound: {upper_bound:.2f} USD")

# TRAIN-TEST SPLIT
# 1. Create Target Variable
df['target_gold_close'] = df['gold close'].shift(-1)
df = df.dropna()

# 2. X and y Split
X = df.drop('target_gold_close', axis=1)
y = df['target_gold_close']

# 3. Chronological Split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print(f"Training: {len(X_train)} days | Testing: {len(X_test)} days\n")

# 4. SCALING
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1),
    "KNN": KNeighborsRegressor(),
    "SVR (Support Vector)": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": score
    })

results_df = pd.DataFrame(results).sort_values(by='MAE')

print(results_df)

# HYPERPARAMETER TUNING

# Time Series Splitter
tscv = TimeSeriesSplit(n_splits=3)

# Models and Parameters
param_grids = {
    "Lasso": {
        "model": Lasso(max_iter=10000),
        "params": {'alpha': [0.001, 0.01, 0.1, 1, 10]}
    },
    "Ridge": {
        "model": Ridge(),
        "params": {'alpha': [0.1, 1, 10, 100, 200]}
    },
    "Random Forest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None]
        }
    }
}

best_results = []

print("\n----- Hyperparameter Tuning -----\n")

for name, config in param_grids.items():
    print(f"Searching for the best parameters for {name}...")

    grid_search = GridSearchCV(
        config["model"],
        config["params"],
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test_scaled)

    new_mae = mean_absolute_error(y_test, predictions)
    new_score = r2_score(y_test, predictions)
    new_mse = mean_squared_error(y_test, predictions)
    new_rmse = np.sqrt(mean_squared_error(y_test, predictions))

    best_results.append({
        "Model": name,
        "Best Parameters": grid_search.best_params_,
        "Tuning MAE": new_mae,
        "Tuning MSE": new_mse,
        "Tuning RMSE": new_rmse,
        "Tuning R2 Score": new_score
    })

tuning_df = pd.DataFrame(best_results).sort_values(by="Tuning MAE")
print("\n--- TUNING RESULTS ---\n")
print(tuning_df.to_string(index=False))

# FEATURE IMPORTANCE (CHAMPION MODEL: LASSO)
final_model = Lasso(alpha=0.1, max_iter=10000)
final_model.fit(X_train_scaled, y_train)

# Extract Coefficients
coefficients = final_model.coef_
column_names = X.columns

importance_df = pd.DataFrame({
    'Feature (Column Name)': column_names,
    'Weight (Impact)': coefficients
})

# Filter out features with exactly 0 impact
importance_df = importance_df[importance_df['Weight (Impact)'] != 0]

# Sort by absolute impact
importance_df['Absolute_Impact'] = importance_df['Weight (Impact)'].abs()
importance_df = importance_df.sort_values(by='Absolute_Impact', ascending=False)

print(f"Out of {len(X.columns)} total columns, only {len(importance_df)} were used by the model!\n")
print(importance_df[['Feature (Column Name)', 'Weight (Impact)']].head(10))

# Plotting Feature Importance
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature (Column Name)'][:10], importance_df['Weight (Impact)'][:10], color='coral')
plt.gca().invert_yaxis()
plt.title('Most Important Features for the Champion Model (Lasso)')
plt.xlabel('Impact on Price (Weight)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

import pickle
with open("financial_regression_model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": final_model,
            "scaler": scaler,
        },f)

pd.DataFrame(X_test_scaled).to_csv("financial_regression_X_test_scaled.csv", index=False)
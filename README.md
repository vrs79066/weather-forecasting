# Weather Forecasting: Traditional Regression vs Neural Networks

Forecasts the daily average temperature of Pune, India, 1 and 3 days ahead, and
compares traditional regression models with a neural network, after extensive
feature engineering.

## What the project does
1. Loads hourly weather data and converts it to daily averages.
2. Engineers features: lags (1, 2, 3, 7 days), rolling means/std (3, 7, 14 days),
   day-to-day differences, and cyclical (sin/cos) season features.
3. Splits the data chronologically (oldest 80% train, newest 20% test, no shuffling).
4. Trains and tunes (GridSearchCV with TimeSeriesSplit) these models:
   Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, and an MLP neural network.
5. Compares them with two baselines (persistence and monthly climatology) using MAE, RMSE and R².
6. Runs an ablation test: raw features vs engineered features.

## Project structure
```
weather_project/
├── weather_forecast.py     # main script (data, features, models, evaluation, plots)
├── download_weather.py     # optional: download daily data from Open-Meteo
├── requirements.txt        # Python libraries
├── data/
│   └── pune.csv            # hourly weather data for Pune
└── results_1day/, results_3day/   # saved tables and charts
```

## How to run
```bash
pip install -r requirements.txt
python weather_forecast.py --csv data/pune.csv --date-col date_time --target tempC \
    --drop FeelsLikeC HeatIndexC WindChillC --out results_1day
python weather_forecast.py --csv data/pune.csv --date-col date_time --target tempC \
    --drop FeelsLikeC HeatIndexC WindChillC --horizon 3 --out results_3day
```
Optional LSTM (needs `pip install torch`): add `--lstm`.

## Results
<!-- Paste YOUR numbers from results_*/model_comparison.csv here -->
| Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| ... | ... | ... | ... |

## Data source
<!-- Write where pune.csv came from (website/Kaggle link, author, license) -->

## Limitations
One city only; daily averages hide hourly detail; limited hyperparameter search.

## Requirements
Python 3.10+, pandas, numpy, scikit-learn, matplotlib (see `requirements.txt`).

# Air Quality Index Analysis

This project explores station-level air-quality measurements and estimates pollutant averages with a linear regression model.

## Files

- `air_quality_analysis.py` — data cleaning, exploratory charts, model training, and evaluation.
- `air quality index.csv` — the dataset used by the script.
- `requirements.txt` — Python packages needed to run the analysis.

## Setup and run

Install the packages listed in `requirements.txt`, then run:

```bash
python air_quality_analysis.py
```

The script looks for `air quality index.csv` beside itself or in the current working directory. It checks for the expected columns and reports a clear error if the dataset is missing or incomplete.

The analysis uses pollutant minimum and maximum values plus encoded city, state, and pollutant category fields to predict `pollutant_avg`. It prints R² and RMSE scores and displays distribution, outlier, relationship, correlation, city-ranking, and prediction charts.

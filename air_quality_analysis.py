from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


sns.set_theme(style="whitegrid")


# Load the CSV next to this script, or from the current working directory.
DATA_FILE = "air quality index.csv"
SCRIPT_DIR = Path(__file__).resolve().parent
candidates = [SCRIPT_DIR / DATA_FILE, Path.cwd() / DATA_FILE]
data_path = next((path for path in candidates if path.is_file()), None)

if data_path is None:
    locations = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {DATA_FILE!r}. Place it next to this script or in the "
        f"current working directory. Checked: {locations}"
    )


df = pd.read_csv(data_path)

required_columns = {
    "pollutant_min",
    "pollutant_max",
    "pollutant_avg",
    "pollutant_id",
    "city",
    "state",
}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(
        "The CSV is missing required columns: " + ", ".join(sorted(missing_columns))
    )

# Convert the model and chart fields to numbers, then remove incomplete records.
numeric_columns = ["pollutant_min", "pollutant_max", "pollutant_avg"]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.drop_duplicates().dropna(subset=list(required_columns))
if df.empty:
    raise ValueError("No complete air-quality records remain after cleaning the CSV.")
if len(df) < 6:
    raise ValueError("At least 6 complete records are needed so the test set can support the R2 metric.")

print("Shape:", df.shape)
df.info()
print(df.describe())


# Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["pollutant_avg"], kde=True)
plt.title("Pollutant Average Distribution")
plt.show()


# Outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[["pollutant_min", "pollutant_max", "pollutant_avg"]])
plt.title("Outlier Detection")
plt.show()


# Scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=df["pollutant_min"], y=df["pollutant_avg"])
plt.title("Min vs Avg")

plt.subplot(1, 2, 2)
sns.scatterplot(x=df["pollutant_max"], y=df["pollutant_avg"])
plt.title("Max vs Avg")

plt.tight_layout()
plt.show()


# Correlation heatmap. Select numeric columns explicitly for pandas compatibility.
plt.figure(figsize=(8, 5))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Top cities
top_cities = (
    df.groupby("city")["pollutant_avg"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
top_cities.plot(kind="bar", figsize=(10, 5))
plt.title("Top 10 Polluted Cities")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Pollutant type distribution. No redundant hue/legend arguments are needed.
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="pollutant_id")
plt.title("Pollutant Types")
plt.tight_layout()
plt.show()


# Machine learning
categorical_columns = ["city", "state", "pollutant_id"]
for column in categorical_columns:
    df[f"{column}_enc"] = LabelEncoder().fit_transform(df[column].astype(str))

features = [
    "pollutant_min",
    "pollutant_max",
    "city_enc",
    "state_enc",
    "pollutant_id_enc",
]
X = df[features]
y = df["pollutant_avg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Results
print("\nR2:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred,
}).round(2)
results["Error"] = results["Actual"] - results["Predicted"]

print("\nActual vs Predicted:\n")
print(results.head(20))


# Model visuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Scatter")

plt.subplot(1, 2, 2)
sns.histplot(results["Error"], kde=True)
plt.title("Error Distribution")

plt.tight_layout()
plt.show()


# First 20 actual values versus predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.to_numpy()[:20], label="Actual", marker="o")
plt.plot(y_pred[:20], label="Predicted", marker="x")
plt.legend()
plt.title("First 20 Comparison")
plt.tight_layout()
plt.show()

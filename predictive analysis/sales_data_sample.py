
# SALES PREDICTION USING HISTORICAL DATA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 1: LOAD DATASET

df = pd.read_csv("sales_data_sample.csv", encoding="cp1252")

print("First Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# STEP 2: DATA CLEANING

df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")

df = df.dropna(subset=["ORDERDATE", "SALES"])
df = df.drop_duplicates()

print("\nDataset After Cleaning:")
print(df.shape)


# STEP 3: BASIC SALES ANALYSIS

print("\nTotal Sales:", round(df["SALES"].sum(), 2))
print("Average Sales:", round(df["SALES"].mean(), 2))
print("Highest Sale:", round(df["SALES"].max(), 2))
print("Lowest Sale:", round(df["SALES"].min(), 2))


# STEP 4: SALES BY PRODUCT CATEGORY

product_sales = df.groupby("PRODUCTLINE")["SALES"].sum()

print("\nSales By Product Category:")
print(product_sales.sort_values(ascending=False))

plt.figure(figsize=(10, 5))

product_sales.sort_values(ascending=False).plot(kind="bar")

plt.title("Sales By Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# STEP 5: MONTHLY SALES ANALYSIS

monthly_sales = df.groupby(
    pd.Grouper(key="ORDERDATE", freq="MS")
)["SALES"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Historical Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Revenue")
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# STEP 6: SPLIT DATA INTO TRAINING AND TESTING

train = monthly_sales.iloc[:-6]
test = monthly_sales.iloc[-6:]

print("\nTraining Records:", len(train))
print("Testing Records:", len(test))


# STEP 7: PREPARE DATA FOR LINEAR REGRESSION

def create_features(dates, start=0):

    time = np.arange(start, start + len(dates))
    month = dates.month

    features = pd.DataFrame({
        "time": time,
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12)
    })

    return features


X_train = create_features(train.index)
X_test = create_features(test.index, len(train))


# STEP 8: TRAIN LINEAR REGRESSION MODEL

model = LinearRegression()

model.fit(X_train, train.values)

regression_predictions = model.predict(X_test)


# STEP 9: SEASONAL FORECAST

seasonal_predictions = monthly_sales.shift(12).loc[test.index]


# STEP 10: EVALUATE BOTH MODELS

def evaluate_model(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(mean_squared_error(actual, predicted))

    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


linear_results = evaluate_model(test, regression_predictions)

seasonal_results = evaluate_model(test, seasonal_predictions)


print("\nMODEL PERFORMANCE COMPARISON")

results = pd.DataFrame({
    "Model": ["Linear Regression", "Seasonal Naive"],
    "MAE": [linear_results[0], seasonal_results[0]],
    "RMSE": [linear_results[1], seasonal_results[1]],
    "R2 Score": [linear_results[2], seasonal_results[2]]
})

print(results.to_string(index=False))


# STEP 11: SELECT THE BETTER MODEL

if linear_results[0] < seasonal_results[0]:

    best_model = "Linear Regression"

else:

    best_model = "Seasonal Naive"


print("\nSelected Forecasting Model:", best_model)


# STEP 12: COMPARE ACTUAL AND PREDICTED SALES

comparison = pd.DataFrame({
    "Month": test.index,
    "Actual Sales": test.values,
    "Linear Regression": regression_predictions,
    "Seasonal Naive": seasonal_predictions.values
})

print("\nACTUAL VS PREDICTED SALES")
print(comparison.to_string(index=False))


plt.figure(figsize=(12, 6))

plt.plot(
    test.index,
    test.values,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    test.index,
    regression_predictions,
    marker="s",
    label="Linear Regression"
)

plt.plot(
    test.index,
    seasonal_predictions.values,
    marker="^",
    label="Seasonal Naive"
)

plt.title("Actual Sales vs Model Predictions")
plt.xlabel("Month")
plt.ylabel("Sales Revenue")

plt.legend()
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# STEP 13: PREDICT FUTURE SALES

future_dates = pd.date_range(
    start=monthly_sales.index[-1] + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS"
)


if best_model == "Linear Regression":

    future_features = create_features(
        future_dates,
        start=len(monthly_sales)
    )

    final_model = LinearRegression()

    final_model.fit(
        create_features(monthly_sales.index),
        monthly_sales.values
    )

    future_predictions = final_model.predict(future_features)

else:

    previous_year_dates = future_dates - pd.DateOffset(years=1)

    future_predictions = monthly_sales.reindex(
        previous_year_dates
    ).values


# STEP 14: DISPLAY FUTURE PREDICTIONS

forecast = pd.DataFrame({
    "Month": future_dates,
    "Predicted Sales": future_predictions
})

print("\nFUTURE SALES FORECAST")
print(forecast.to_string(index=False))


# STEP 15: SAVE PREDICTIONS TO CSV

forecast.to_csv(
    "future_sales_prediction.csv",
    index=False
)

print("\nForecast saved successfully!")


# STEP 16: VISUALIZE FUTURE SALES

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o",
    label="Historical Sales"
)

plt.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Historical Sales and Future Sales Prediction")
plt.xlabel("Month")
plt.ylabel("Sales Revenue")

plt.legend()
plt.grid(True)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# STEP 17: PROJECT CONCLUSION

print("\nPROJECT CONCLUSION")

print("Total Historical Sales:", round(monthly_sales.sum(), 2))

print("Best Performing Model:", best_model)

print("Average Monthly Sales:", round(monthly_sales.mean(), 2))

print("Average Predicted Sales:", round(np.mean(future_predictions), 2))

print("\nSales prediction project completed successfully!")

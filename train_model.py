import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


df = pd.read_csv("data/processed_energy_data.csv")


X = df.drop('Appliances', axis=1)
y = df['Appliances']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)


model.fit(X_train, y_train)



predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(y_test, predictions)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")


# Actual vs Predicted

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values[:200],
    label='Actual'
)

plt.plot(
    predictions[:200],
    label='Predicted'
)

plt.title("Actual vs Predicted Energy Usage")
plt.xlabel("Samples")
plt.ylabel("Energy Consumption")

plt.legend()

plt.show()



importance = model.feature_importances_

feature_names = X.columns

plt.figure(figsize=(10,6))

plt.barh(feature_names, importance)

plt.title("Feature Importance")

plt.show()


joblib.dump(
    model,
    "models/xgboost_energy_model.pkl"
)

print("Model saved!")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/energydata_complete.csv")


df['date'] = pd.to_datetime(df['date'])

df['hour'] = df['date'].dt.hour
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday

# Weekend feature
df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)


# FEATURE ENGINEERING

# Lag features
df['lag_1'] = df['Appliances'].shift(1)
df['lag_6'] = df['Appliances'].shift(6)
df['lag_12'] = df['Appliances'].shift(12)

# Rolling features
df['rolling_mean_6'] = df['Appliances'].rolling(window=6).mean()
df['rolling_mean_12'] = df['Appliances'].rolling(window=12).mean()

# Peak hour indicator
df['peak_hour'] = df['hour'].apply(
    lambda x: 1 if 17 <= x <= 22 else 0
)


df.dropna(inplace=True)


features = [
    'T1', 'RH_1',
    'hour', 'day', 'month',
    'weekday', 'is_weekend',
    'lag_1', 'lag_6', 'lag_12',
    'rolling_mean_6',
    'rolling_mean_12',
    'peak_hour'
]

target = 'Appliances'

X = df[features]
y = df[target]



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

processed_df = pd.DataFrame(X_scaled, columns=features)
processed_df[target] = y.values


processed_df.to_csv("data/processed_energy_data.csv", index=False)

print("Preprocessing complete!")
print(processed_df.head())
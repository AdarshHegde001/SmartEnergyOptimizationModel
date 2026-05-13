import streamlit as st
import pandas as pd
import numpy as np
import joblib

#load model
model = joblib.load(
    "models/xgboost_energy_model.pkl"
)


st.set_page_config(
    page_title="Smart Energy Optimizer",
    layout="wide"
)

st.title("⚡ Smart Energy Optimization Dashboard")

st.markdown(
    "Predict and optimize appliance energy consumption using XGBoost"
)


st.sidebar.header("Input Parameters")

temperature = st.sidebar.slider(
    "Temperature (T1)",
    0.0,
    50.0,
    22.0
)

humidity = st.sidebar.slider(
    "Humidity (RH_1)",
    0.0,
    100.0,
    50.0
)

hour = st.sidebar.slider(
    "Hour",
    0,
    23,
    18
)

day = st.sidebar.slider(
    "Day",
    1,
    31,
    15
)

month = st.sidebar.slider(
    "Month",
    1,
    12,
    6
)

weekday = st.sidebar.slider(
    "Weekday",
    0,
    6,
    2
)

is_weekend = 1 if weekday >= 5 else 0

lag_1 = st.sidebar.number_input(
    "Previous Energy Usage",
    value=100.0
)

lag_6 = st.sidebar.number_input(
    "6-Step Lag Energy",
    value=120.0
)

lag_12 = st.sidebar.number_input(
    "12-Step Lag Energy",
    value=110.0
)

rolling_mean_6 = st.sidebar.number_input(
    "Rolling Mean 6",
    value=115.0
)

rolling_mean_12 = st.sidebar.number_input(
    "Rolling Mean 12",
    value=118.0
)

peak_hour = 1 if 17 <= hour <= 22 else 0


input_data = np.array([[
    temperature,
    humidity,
    hour,
    day,
    month,
    weekday,
    is_weekend,
    lag_1,
    lag_6,
    lag_12,
    rolling_mean_6,
    rolling_mean_12,
    peak_hour
]])


prediction = model.predict(input_data)[0]


st.subheader("Predicted Energy Consumption")

st.metric(
    label="Predicted Appliance Energy",
    value=f"{prediction:.2f} Wh"
)


st.subheader("Optimization Suggestions")

if prediction > 300:
    st.error(
        "⚠ High predicted energy consumption detected!"
    )

    st.write(
        "- Reduce heavy appliance usage"
    )

    st.write(
        "- Avoid AC during peak hours"
    )

    st.write(
        "- Shift appliances to off-peak periods"
    )

elif prediction > 150:
    st.warning(
        "Moderate energy consumption predicted."
    )

    st.write(
        "- Monitor appliance usage"
    )

else:
    st.success(
        "✅ Efficient energy usage predicted!"
    )

if peak_hour:
    st.warning(
        "⚡ You are currently in peak usage hours."
    )



cost = prediction * 0.012

st.subheader("Estimated Electricity Cost")

st.metric(
    label="Estimated Cost",
    value=f"${cost:.2f}"
)


st.markdown("---")

st.caption(
    "AI-Based Smart Energy Optimization System using XGBoost"
)
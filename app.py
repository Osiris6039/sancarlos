# app.py - Deep Learning AI Forecasting App (Streamlit Cloud Ready)

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# File paths
data_file = "data.csv"
event_file = "events.csv"

# Create files if not exist
if not os.path.exists(data_file):
    pd.DataFrame(columns=["Date", "Sales", "Customers", "Weather", "AddOnSales"]).to_csv(data_file, index=False)

if not os.path.exists(event_file):
    pd.DataFrame(columns=["EventDate", "EventName", "LastYearSales", "LastYearCustomers"]).to_csv(event_file, index=False)

# Auth
def authenticate(username, password):
    return username == "admin" and password == "forecast123"

st.set_page_config(page_title="DL AI Forecast", layout="wide")
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if not st.session_state.authenticated:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

st.title("🤖 Deep Learning AI Forecasting")

# Load data
data = pd.read_csv(data_file)
events = pd.read_csv(event_file)
data["Date"] = pd.to_datetime(data["Date"])
today = pd.Timestamp.today().normalize()

# Input form
st.header("📥 Daily Data")
with st.form("daily_form", clear_on_submit=True):
    date = st.date_input("Date")
    sales = st.number_input("Sales", 0)
    customers = st.number_input("Customers", 0)
    weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy"])
    addon = st.number_input("Add-On Sales", 0)
    if st.form_submit_button("Submit"):
        new_row = pd.DataFrame([{"Date": date, "Sales": sales, "Customers": customers, "Weather": weather, "AddOnSales": addon}])
        data = pd.concat([data, new_row], ignore_index=True)
        data.to_csv(data_file, index=False)
        st.success("Saved!")

# Event input
st.header("📅 Future Events")
with st.form("event_form", clear_on_submit=True):
    edate = st.date_input("Event Date")
    ename = st.text_input("Event Name")
    esales = st.number_input("Last Year Sales", 0)
    ecustomers = st.number_input("Last Year Customers", 0)
    if st.form_submit_button("Submit Event"):
        new_event = pd.DataFrame([{"EventDate": edate.strftime('%Y-%m-%d'), "EventName": ename, "LastYearSales": esales, "LastYearCustomers": ecustomers}])
        events = pd.concat([events, new_event], ignore_index=True)
        events.to_csv(event_file, index=False)
        st.success("Event added!")

# View recent data
st.subheader("📋 Last 10 Days")
recent = data[data["Date"] >= (today - pd.Timedelta(days=10))]
st.dataframe(recent.sort_values("Date", ascending=False).reset_index(drop=True))

# Forecasting functions
def prepare_data(df):
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["AddOnFlag"] = df["AddOnSales"].apply(lambda x: 1 if x > 0 else 0)
    df = pd.get_dummies(df[["DayOfYear", "Weather", "AddOnFlag"]])
    return df

def build_model(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_models(data):
    X = prepare_data(data.copy())
    y_sales = data["Sales"]
    y_cust = data["Customers"]
    m_sales = build_model(X.shape[1])
    m_cust = build_model(X.shape[1])
    es = EarlyStopping(monitor="loss", patience=5)
    m_sales.fit(X, y_sales, epochs=100, verbose=0, callbacks=[es])
    m_cust.fit(X, y_cust, epochs=100, verbose=0, callbacks=[es])
    m_sales.save("model_sales.h5")
    m_cust.save("model_customers.h5")
    return m_sales, m_cust, X.columns

def forecast_next(model_sales, model_customers, columns, events, days=10):
    result = []
    for i in range(days):
        fdate = today + pd.Timedelta(days=i)
        row = {"DayOfYear": fdate.dayofyear, "Weather": "Sunny", "AddOnFlag": 0}
        if not events[events["EventDate"] == fdate.strftime('%Y-%m-%d')].empty:
            row["AddOnFlag"] = 1
        df = pd.DataFrame([row])
        df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
        s = model_sales.predict(df)[0][0]
        c = model_customers.predict(df)[0][0]
        result.append((fdate.strftime('%Y-%m-%d'), round(s), round(c)))
    return pd.DataFrame(result, columns=["Date", "Forecasted Sales", "Forecasted Customers"])

# Forecast section
st.header("🔮 Forecast (Next 10 Days)")
if st.button("Run Forecast"):
    if len(data) < 10:
        st.warning("At least 10 records required.")
    else:
        if os.path.exists("model_sales.h5") and os.path.exists("model_customers.h5"):
            sm = load_model("model_sales.h5")
            cm = load_model("model_customers.h5")
            cols = prepare_data(data).columns
        else:
            sm, cm, cols = train_models(data)
        fcast = forecast_next(sm, cm, cols, events)
        st.dataframe(fcast)
        st.download_button("📥 Download CSV", fcast.to_csv(index=False), "forecast.csv")
        st.line_chart(fcast.set_index("Date"))

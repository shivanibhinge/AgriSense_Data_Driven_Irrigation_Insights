# app.py
# Streamlit Smart Water Usage Project Website (Module 1 -> Modules 2-4 scaffolding)
# Requirements: streamlit, pandas, requests, scikit-learn, matplotlib, seaborn, mlxtend
# pip install streamlit pandas requests scikit-learn matplotlib seaborn mlxtend

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

# For apriori (ARM)
# from mlxtend.frequent_patterns import apriori, association_rules
# from mlxtend.preprocessing import TransactionEncoder

st.set_page_config(layout="wide", page_title="Smart Water Usage — Project Website")

# ---------------------
# Helper functions
# ---------------------
def fetch_nasa_power(lat, lon, start="20230101", end="20231231", parameters="T2M_MAX,T2M_MIN,RH2M,PRECTOT,EVPTRNS"):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "start": start,
        "end": end,
        "latitude": lat,
        "longitude": lon,
        "parameters": parameters,
        "community": "AG",
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "properties" not in data or "parameter" not in data["properties"]:
        st.warning("NASA POWER returned an unexpected structure or empty data.")
        return pd.DataFrame()
    params_data = data["properties"]["parameter"]
    dates = sorted(next(iter(params_data.values())).keys())
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    for var, series in params_data.items():
        df[var.lower()] = [series[d] for d in dates]
    return df

def fetch_weatherbit_ag(lat, lon, start, end, api_key):
    # weatherbit / history/agweather endpoint requires API key and may limit ranges
    url = "https://api.weatherbit.io/v2.0/history/agweather"
    params = {"lat": lat, "lon": lon, "start_date": start, "end_date": end, "key": api_key}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Weatherbit error {r.status_code}: {r.text}")
        return pd.DataFrame()
    data = r.json()
    df = pd.DataFrame(data.get("data", []))
    # ensure date column name
    if 'valid_date' in df.columns:
        df['valid_date'] = pd.to_datetime(df['valid_date'])
    return df

def json_to_df(soil_json):
    # Accepts dict or JSON with key "data" or list of records
    if isinstance(soil_json, dict) and "data" in soil_json:
        df = pd.DataFrame(soil_json["data"])
    elif isinstance(soil_json, list):
        df = pd.DataFrame(soil_json)
    else:
        # maybe full JSON from Weatherbit already parsed differently
        df = pd.DataFrame([soil_json])
    # normalize date fields
    for c in ['valid_date','date','timestamp_local','timestamp_utc']:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
            except:
                pass
    return df

def clean_and_merge(df_weather, df_soil, keep_cols=None):
    # Standardize names to lower, unify date columns
    w = df_weather.copy()
    s = df_soil.copy()
    w.columns = w.columns.str.lower()
    s.columns = s.columns.str.lower()

    # Find date columns
    if 'date' not in w.columns:
        # try other
        for c in w.columns:
            if 'time' in c or 'valid' in c:
                w['date'] = pd.to_datetime(w[c])
                break
    if 'valid_date' in s.columns:
        s['date_soil'] = pd.to_datetime(s['valid_date'])
    elif 'date' in s.columns:
        s['date_soil'] = pd.to_datetime(s['date'])
    elif 'timestamp_local' in s.columns:
        s['date_soil'] = pd.to_datetime(s['timestamp_local']).dt.date
        s['date_soil'] = pd.to_datetime(s['date_soil'])
    else:
        # assume index is date-like
        s = s.reset_index()
        try:
            s['date_soil'] = pd.to_datetime(s['index'])
        except:
            s['date_soil'] = pd.NaT

    w['date'] = pd.to_datetime(w['date'])
    # Rename common useful columns (safe mapping)
    rename_map = {
        't2m_max':'t2m_max','t2m_min':'t2m_min','rh2m':'rh2m','prectot':'precipitation','prectotcorr':'precipitation',
        'evptrns':'evapotranspiration','evapotranspiration':'evapotranspiration'
    }
    w = w.rename(columns=rename_map)
    # soil volumetric columns often: v_soilm_0_10cm
    # take common soil moisture column names
    possible_soil_cols = [c for c in s.columns if 'soil' in c or 'v_soilm' in c or 'soilm' in c]
    # Merge on date
    df = pd.merge(w, s, left_on='date', right_on='date_soil', how='inner')
    # drop duplicates & NaNs
    df = df.drop(columns=[c for c in ['date_soil','index'] if c in df.columns], errors='ignore')
    # convert numeric columns
    for c in df.columns:
        if df[c].dtype == object:
            # try to convert
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                pass
    # drop rows where all numeric are NaN
    numcols = df.select_dtypes(include='number').columns
    if len(numcols)>0:
        df = df.dropna(subset=numcols, how='all')
    # fill small gaps
    df = df.sort_values('date').reset_index(drop=True)
    df[numcols] = df[numcols].fillna(method='ffill').fillna(method='bfill')
    # optional select columns
    if keep_cols:
        cols = [c for c in keep_cols if c in df.columns]
        df = df[['date']+cols] if 'date' in df.columns else df[cols]
    return df

def plot_and_save(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

st.title("Smart Water Usage — Predicting irrigation needs using weather, soil and vegetation")

# Sidebar: config
# st.sidebar.header("Project Configuration")
# lat = st.sidebar.number_input("Latitude", value=36.77, format="%.5f")
# lon = st.sidebar.number_input("Longitude", value=-119.41, format="%.5f")
# start_date = st.sidebar.date_input("Start date", value=datetime(2023,1,1))
# end_date = st.sidebar.date_input("End date", value=datetime(2023,12,31))
# start_str = start_date.strftime("%Y%m%d")
# end_str = end_date.strftime("%Y%m%d")

# st.sidebar.markdown("**Soil data options**")
# weatherbit_key = st.sidebar.text_input("Weatherbit API Key (optional)", value="")
# soil_upload = st.sidebar.file_uploader("Or upload soil JSON/CSV (optional)", type=['json','csv'])

# Layout tabs
tabs = st.tabs(["Introduction","Data Prep","EDA","Clustering","ARM (Assoc Rules)","DT / NB","SVM / Ensemble","Regression","Conclusions","About Me"])

# Introduction tab
with tabs[0]:
    st.header("Introduction")
    st.markdown("""
    **Topic:** Smart Water Usage Prediction — Predicting irrigation needs using weather, soil, and vegetation data.

    Agriculture consumes the majority of global freshwater resources. Optimizing irrigation timing and amount can substantially reduce water use while maintaining crop yields. Smart water usage prediction combines weather variables (temperature, rainfall, evapotranspiration) with soil moisture observations to determine when irrigation is necessary. This project will focus on building a reproducible pipeline — collecting data via APIs, cleaning & merging datasets, performing exploratory data analysis, and applying unsupervised and supervised machine learning to predict irrigation-relevant targets.
    """)
    st.subheader("Introduction")
    st.write("""
    Water is the essential resource for food production. However, inefficient irrigation practices waste large amounts of water and results in higher costs for farmers. With climate variability increasing the frequency of droughts and extreme rainfall events, data-driven irrigation scheduling can improve reliability and sustainability. This project will explore on how daily climate and soil data can be combined to make informed decisions on irrigation timing decisions that conserve water while supporting crop health.
    The aim is to represent a iterable data science process that collects real API data, cleans and transforms it then conducts exploratory analysis to identify key factors of soil moisture and evapotranspiration and then will apply machine learning techniques to predict irrigation needs. The final website will present results which are easy to understand for stakeholders (like farmers, policy makers, and the public) and can understand and act upon.
    """)
    st.subheader("Ten research questions addressed")
    questions = [
        "How does daily precipitation affect soil moisture at 0–10 cm depth?",
        "Which weather features (temperature, humidity, wind) most influence evapotranspiration?",
        "Can we cluster days/periods with similar water demand profiles?",
        "Does soil moisture lag precipitation by a consistent number of days?",
        "What are the principal components that explain variance in the combined dataset?",
        "Can a decision tree accurately predict high/low irrigation need?",
        "How does seasonal variation affect soil moisture and ET?",
        "Which features are most important for predicting soil moisture?",
        "Can association rules reveal combinations of conditions that signal irrigation need?",
        "How much water could be saved with an automated irrigation decision system?"
    ]
    for i,q in enumerate(questions,1):
        st.write(f"{i}. {q}")

with tabs[1]:
    st.header("Data Prep")

    st.header("1️⃣ Data Gathering Overview")

    st.markdown("""
    The data for this project was collected from two APIs:

    1. **NASA POWER API** – provides daily weather parameters such as maximum and minimum temperature, humidity, precipitation, and evapotranspiration.
    2. **Weatherbit AgWeather API** – provides soil moisture, evapotranspiration, temperature, precipitation, and wind data specific to agricultural applications.

    The data was gathered for **California (Latitude: 36.77, Longitude: -119.41)** for the years **2018–2023**.  
    NASA data was received in a tabular format (CSV-like), while the Weatherbit API returned **JSON**, which was flattened into a DataFrame.

    Both datasets were merged on the **date field** to create one unified dataset combining daily soil and weather parameters.
    """)
    st.image(
        "images/dataset_merged.png",
        caption="Preview of the merged weather–soil dataset (first few rows)",
        use_container_width=True
    )

    st.header("2️⃣ Data Cleaning and Merging")

    st.markdown("""
    After gathering data from the two APIs, the following cleaning and preparation steps were applied to ensure the datasets were consistent, accurate, and ready for analysis:

    1. **Standardizing Column Names**  
       All column names were converted to lowercase for uniformity.

    2. **Date Formatting**  
       The date columns from both datasets were converted to `datetime` format. The Weatherbit dataset column `valid_date` was renamed to `date` to match the NASA dataset, allowing accurate merging.

    3. **Dropping Unnecessary Columns**  
       Metadata columns such as timestamps, soil density, and other non-relevant columns were removed.

    4. **Handling Missing Values**  
       Checked and handled if any missing values were present.

    5. **Merging Datasets**  
       The NASA POWER and Weatherbit datasets were merged on the `date` column using an inner join, producing a single dataset containing both weather and soil parameters for each day.

    **Resulting Dataset:**  
    The merged dataset now contains daily observations combining weather and soil parameters according to given latitude and longitude values along with the start and end date . Key features include:

    - **Weather features:** maximum and minimum temperature, relative humidity, corrected precipitation, evapotranspiration.  
    - **Soil features:** volumetric soil moisture at different depths, average temperature, surface evapotranspiration, wind speed.

    This dataset forms the basis for exploratory data analysis, visualization, and predictive modeling in subsequent modules.
    """)

    st.header("3️⃣ Key Parameters in the Final Dataset")

    st.markdown("""
    | Category | Feature | Description |
    |-----------|----------|-------------|
    | **Weather** | `T2M_MAX` | Max air temperature (°C) |
    |  | `T2M_MIN` | Min air temperature (°C) |
    |  | `RH2M` | Relative humidity (%) |
    |  | `PRECTOTCORR` | Corrected precipitation (mm/day) |
    |  | `EVPTRNS` | Evapotranspiration (mm/day) |
    | **Soil** | `v_soilm_0_10cm` | Volumetric soil moisture (0–10 cm) |
    |  | `v_soilm_10_40cm` | Soil moisture (10–40 cm) |
    |  | `evapotranspiration` | Surface evapotranspiration (mm/day) |
    |  | `precip` | Precipitation (mm/day) |
    |  | `temp_2m_avg` | Mean temperature (°C) |
    |  | `wind_10m_spd_avg` | Wind speed at 10 m (m/s) |
    """, unsafe_allow_html=True)

    st.info("These features together describe the climate–soil interaction that determines optimal irrigation timing.")


#eda tab
with tabs[2]:
    # Exploratory Data Analysis (EDA)
    st.header("4️⃣ Exploratory Data Analysis (EDA)")

    st.markdown("""
    Exploratory Data Analysis (EDA) was performed to understand trends, variability,
    and relationships between weather conditions and soil moisture parameters.
    The following visualizations highlight key patterns observed in the dataset.
    """)

    # Visualizations
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image(
            "images/daily_weather_soil_moisture_trends.png",
            caption="Daily maximum and minimum temperature trends (2018–2023)",
            use_container_width=True
        )

    with col2:
        st.markdown("""
    **Daily Weather Soil Moisture Trends**

    This time-series plot displays how daily maximum temperature, rainfall, and soil moisture change over time. A clear seasonal pattern is visible in temperature, with higher values in summer and lower values in winter. Rainfall occurs in irregular spikes, and increases in rainfall are often followed by small rises in soil moisture, indicating that precipitation directly influences soil water availability.
    """)

    st.divider()

    # Visualization 2
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image(
            "images/monthly_soil_moisture.png",
            caption="Distribution of daily precipitation values",
            use_container_width=True
        )

    with col2:
        st.markdown("""
    **Monthly Soil Moisture Trends**

    This boxplot displays the distribution and variability of soil moisture across months which captures the daily fluctuations within each month rather than just averages.
Soil moisture levels are highest and most stable during winter months, indicating sufficient natural water availability from precipitation and reduced evapotranspiration. As the year progresses into late spring and summer, soil moisture decreases sharply and becomes more variable, reflecting higher evaporation rates and increased plant water uptake.
The lowest soil moisture levels occur during July through September, coinciding with peak evapotranspiration
    """)

    st.divider()

    # Visualization 3
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image(
            "images/max_temp_rainfall.png",
            caption="Soil moisture variation across depth layers",
            use_container_width=True
        )

    with col2:
        st.markdown("""
    **Maximum Temperature compared to Rainfall**

    This scatter plot compares daily maximum temperature with daily rainfall amounts. Most heavy rainfall events occur at lower to moderate temperatures, while very high temperatures are generally associated with little or no rainfall. This suggests an inverse relationship where hotter days tend to be drier.
    """)

    st.divider()

    # Visualization 4
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.image(
            "images/rainfall_soil_moisture.png",
            caption="Correlation heatmap of weather and soil parameters",
            use_container_width=True
        )

    with col2:
        st.markdown("""
    **Rainfall Soil Moisture**

    This scatter plot illustrates the relationship between rainfall and shallow soil moisture. Higher rainfall values are generally associated with higher soil moisture levels, while near-zero rainfall often corresponds to low soil moisture. However, variability shows that soil moisture is also influenced by factors such as evaporation and prior conditions.
    """)

    col1, col2 = st.columns([1.2, 1])
    st.divider()
    with col1:
        st.image(
            "images/avg_evaporation.png",
            caption="Correlation heatmap of weather and soil parameters",
            use_container_width=True
        )

    with col2:
        st.markdown("""
    **Evapotranspiration graph over months**

    This bar chart shows the average daily evapotranspiration (evaporation+transpiration) for each month of the year. It is the combined water loss from soil evaporation and plant transpiration it makes it a key indicator of irrigation demand. 
ET values increase steadily from winter to early summer, reaching a peak during June and July. This pattern reflects rising temperatures, longer daylight hours, and increased plant activity during the growing season. After midsummer, ET gradually declines through the fall and reaches its lowest levels in December and January.
These seasonal trends reflect periods of high water demand mostly in spring and summer
    """)

    st.info("Additional EDA plots are provided in the project repository and linked under the Code section.")

# Clustering tab
with tabs[3]:
    st.header("Clustering - Upcoming")


#ARM tab
with tabs[4]:
    st.header("Association Rule Mining (ARM)")


with tabs[5]:
    st.header("Decision Trees & Naive Bayes")

#SVM / Ensemble tab
with tabs[6]:
    st.header("SVM & Ensemble Methods")

#Regression tab
with tabs[7]:
    st.header("Regression (linear baseline and tree)")

#Conclusions tab
with tabs[8]:
    st.header("Conclusions")

# -------- About Me tab --------
with tabs[9]:
    st.header("About Me")
    st.markdown("""
    **Name:** Shivani Bhinge  
    **Project:** Smart Water Usage Prediction  
    """)

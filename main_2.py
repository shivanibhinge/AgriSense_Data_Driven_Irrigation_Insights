# app.py
# Streamlit Smart Water Usage Project Website
# Requirements: streamlit, pandas, requests, scikit-learn, matplotlib, seaborn, mlxtend
# pip install streamlit pandas requests scikit-learn matplotlib seaborn mlxtend

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from io import StringIO, BytesIO


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
    url = "https://api.weatherbit.io/v2.0/history/agweather"
    params = {"lat": lat, "lon": lon, "start_date": start, "end_date": end, "key": api_key}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Weatherbit error {r.status_code}: {r.text}")
        return pd.DataFrame()
    data = r.json()
    df = pd.DataFrame(data.get("data", []))
    if 'valid_date' in df.columns:
        df['valid_date'] = pd.to_datetime(df['valid_date'])
    return df

def json_to_df(soil_json):
    if isinstance(soil_json, dict) and "data" in soil_json:
        df = pd.DataFrame(soil_json["data"])
    elif isinstance(soil_json, list):
        df = pd.DataFrame(soil_json)
    else:
        df = pd.DataFrame([soil_json])
    for c in ['valid_date','date','timestamp_local','timestamp_utc']:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c])
            except:
                pass
    return df

def clean_and_merge(df_weather, df_soil, keep_cols=None):
    w = df_weather.copy()
    s = df_soil.copy()
    w.columns = w.columns.str.lower()
    s.columns = s.columns.str.lower()
    if 'date' not in w.columns:
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
        s = s.reset_index()
        try:
            s['date_soil'] = pd.to_datetime(s['index'])
        except:
            s['date_soil'] = pd.NaT
    w['date'] = pd.to_datetime(w['date'])
    rename_map = {
        't2m_max':'t2m_max','t2m_min':'t2m_min','rh2m':'rh2m','prectot':'precipitation','prectotcorr':'precipitation',
        'evptrns':'evapotranspiration','evapotranspiration':'evapotranspiration'
    }
    w = w.rename(columns=rename_map)
    df = pd.merge(w, s, left_on='date', right_on='date_soil', how='inner')
    df = df.drop(columns=[c for c in ['date_soil','index'] if c in df.columns], errors='ignore')
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            except:
                pass
    numcols = df.select_dtypes(include='number').columns
    if len(numcols)>0:
        df = df.dropna(subset=numcols, how='all')
    df = df.sort_values('date').reset_index(drop=True)
    df[numcols] = df[numcols].ffill().bfill()
    if keep_cols:
        cols = [c for c in keep_cols if c in df.columns]
        df = df[['date']+cols] if 'date' in df.columns else df[cols]
    return df

def plot_and_save(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────
st.title("🌾 Smart Water Usage — Predicting irrigation needs using weather, soil and vegetation")
# ─────────────────────────────────────────────────────────────

tabs = st.tabs([
    " Introduction",
    " Data Prep",
    " EDA",
    " PCA",
    " Clustering",
    " ARM (Assoc Rules)",
    " NB / DT",
    " SVM / Ensemble",
    " Regression",
    " Conclusions",
    " About Me"
])

# ─────────────────────────────────────────────────────────────
# TAB 0 — Introduction
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Introduction")
    st.markdown("""
    **Topic:** Smart Water Usage Prediction — Predicting irrigation needs using weather, soil, and vegetation data.

    Agriculture consumes the majority of global freshwater resources. Optimizing irrigation timing and amount
    can substantially reduce water use while maintaining crop yields. Smart water usage prediction combines
    weather variables (temperature, rainfall, evapotranspiration) with soil moisture observations to determine
    when irrigation is necessary. This project focuses on building a reproducible pipeline — collecting data
    via APIs, cleaning and merging datasets, performing exploratory data analysis, and applying unsupervised
    and supervised machine learning to predict irrigation-relevant targets. The goal is to answer a real-world
    agricultural question using data science tools in a transparent and replicable way. Every step from raw
    data collection to final model results is documented and linked so that anyone can follow along and verify
    the process. This kind of approach is becoming increasingly important as agriculture faces growing pressure
    from population growth and climate variability.
    """)

    st.markdown("""
    Water is the most essential resource for food production, yet it is one of the most wasted. Globally,
    agriculture accounts for roughly 70 percent of all freshwater withdrawals, and a large portion of that
    is applied inefficiently — too much water on days when none is needed, or too little during critical
    dry spells. Inefficient irrigation not only wastes water but also increases costs for farmers, leaches
    nutrients from the soil, and contributes to groundwater depletion in regions already under stress.
    With climate variability increasing the frequency and severity of droughts and extreme rainfall events,
    the case for smarter irrigation scheduling has never been stronger. Data-driven approaches offer a
    practical path forward — by combining weather forecasts with real-time soil moisture readings, it becomes
    possible to predict when irrigation is truly needed rather than relying on fixed schedules. This kind of
    precision agriculture is already being adopted in parts of California, Israel, and the Netherlands, and
    its potential to spread globally is enormous.
    """)

    st.markdown("""
    The AgriSense project focuses on California's Central Valley, one of the most productive and
    water-intensive agricultural regions in the world. California receives most of its rainfall in winter
    and faces long, hot, dry summers — making it an ideal case study for irrigation prediction. The region
    depends heavily on both surface water and groundwater for farming, and decades of over-extraction have
    led to serious groundwater depletion in many parts of the valley. Understanding the daily patterns of
    soil moisture, evapotranspiration, and temperature is critical for any farmer trying to make smart
    watering decisions. This project uses publicly available climate and soil data from NASA and Weatherbit
    to build that understanding from the ground up. Rather than relying on expensive sensor networks,
    the aim is to show that API-sourced weather and soil data alone can power accurate irrigation decisions.
    The data covers 2020 to 2025, capturing multiple wet and dry years to ensure the models generalize well.
    """)

    st.markdown("""
    The data science approach used in this project follows the full lifecycle — from raw data collection
    through cleaning, exploration, unsupervised learning, and finally supervised prediction. In the early
    stages, exploratory analysis revealed strong seasonal patterns in soil moisture and evapotranspiration,
    with summer months consistently showing the lowest soil moisture and highest water demand. Clustering
    confirmed that days naturally group into warm-dry and cool-wet regimes, and association rule mining
    uncovered specific combinations of conditions — like moderate spring humidity combined with high ET —
    that reliably signal upcoming irrigation need. These findings motivated the supervised learning models
    built in Module 3, where Naive Bayes, Decision Trees, and Logistic Regression are all applied to
    predict whether a given day requires irrigation. All models achieve between 85 and 93 percent accuracy
    on unseen test data, which is a strong and honest result for a real-world agricultural dataset.
    """)

    st.markdown("""GitHub Link : https://github.com/shivanibhinge/AgriSense_Data_Driven_Irrigation_Insights/tree/main""")

    st.subheader("Ten Research Questions")
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
    for i, q in enumerate(questions, 1):
        st.write(f"{i}. {q}")

# ─────────────────────────────────────────────────────────────
# TAB 1 — Data Prep
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Data Prep")

    st.subheader("1. Data Gathering Overview")
    st.markdown("""
    The data for this project was collected from two APIs:

    1. **NASA POWER API** – provides daily weather parameters such as maximum and minimum temperature, humidity, precipitation, and evapotranspiration.
    2. **Weatherbit AgWeather API** – provides soil moisture, evapotranspiration, temperature, precipitation, and wind data specific to agricultural applications.

    The data was gathered for **California (Latitude: 36.77, Longitude: -119.41)** for the years **2018–2023**.
    NASA data was received in a tabular format (CSV-like), while the Weatherbit API returned **JSON**, which was flattened into a DataFrame.

    Both datasets were merged on the **date field** to create one unified dataset combining daily soil and weather parameters.
    """)
    st.image("images/dataset_merged.png", caption="Preview of the merged weather–soil dataset (first few rows)", use_container_width=True)

    st.subheader("2. Data Cleaning and Merging")
    st.markdown("""
    After gathering data from the two APIs, the following cleaning and preparation steps were applied:

    1. **Standardizing Column Names** — All column names were converted to lowercase for uniformity.
    2. **Date Formatting** — Date columns from both datasets were converted to `datetime` format.
    3. **Dropping Unnecessary Columns** — Metadata columns such as timestamps, soil density, and other non-relevant columns were removed.
    4. **Handling Missing Values** — Checked and handled any missing values using forward-fill.
    5. **Merging Datasets** — The NASA POWER and Weatherbit datasets were merged on the `date` column using an inner join.

    **Resulting Dataset** contains daily observations combining weather and soil parameters:
    - **Weather features:** max/min temperature, relative humidity, corrected precipitation, evapotranspiration.
    - **Soil features:** volumetric soil moisture at different depths, surface evapotranspiration, wind speed.
    """)

    st.subheader("3. Key Parameters in the Final Dataset")
    st.markdown("""
    | Category | Feature | Description |
    |----------|---------|-------------|
    | **Weather** | `T2M_MAX` | Max air temperature (°C) |
    | | `T2M_MIN` | Min air temperature (°C) |
    | | `RH2M` | Relative humidity (%) |
    | | `PRECTOTCORR` | Corrected precipitation (mm/day) |
    | | `EVPTRNS` | Evapotranspiration (mm/day) |
    | **Soil** | `v_soilm_0_10cm` | Volumetric soil moisture (0–10 cm) |
    | | `v_soilm_10_40cm` | Soil moisture (10–40 cm) |
    | | `evapotranspiration` | Surface evapotranspiration (mm/day) |
    | | `wind_10m_spd_avg` | Wind speed at 10 m (m/s) |
    """, unsafe_allow_html=True)
    st.info("These features together describe the climate–soil interaction that determines optimal irrigation timing.")

# ─────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("""
    Exploratory Data Analysis (EDA) was performed to understand trends, variability,
    and relationships between weather conditions and soil moisture parameters.
    The 10 visualizations below walk through the key patterns found in the AgriSense dataset.
    """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/daily_weather_soil_moisture_trends.png", caption="Viz 1 — Daily Weather & Soil Moisture Trends", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 1 — Daily Weather & Soil Moisture Trends**

        This time-series plot shows how daily maximum temperature, rainfall, and soil moisture
        change over the full dataset period. A clear seasonal rhythm is visible — temperatures
        peak every summer and drop every winter. Rainfall appears as irregular spikes rather
        than a steady flow, and small rises in soil moisture typically follow those rainfall
        events shortly after. This is one of the clearest signs that precipitation directly
        drives soil water availability.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/monthly_soil_moisture.png", caption="Viz 2 — Monthly Soil Moisture Box Plot", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 2 — Monthly Soil Moisture Variation**

        This box plot shows how soil moisture is spread across each month of the year.
        Winter months show the highest and most stable soil moisture. As spring arrives
        moisture starts dropping, and by July–September it hits its lowest point with the
        most variability. This summer dip lines up directly with peak irrigation demand.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/max_temp_rainfall.png", caption="Viz 3 — Max Temperature vs Rainfall", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 3 — Maximum Temperature vs Rainfall**

        Heavy rainfall events almost always occur at lower or moderate temperatures, while
        the hottest days have zero or near-zero rainfall. The days that crops need water
        the most are also the days least likely to receive it naturally.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/rainfall_soil_moisture.png", caption="Viz 4 — Rainfall vs Soil Moisture", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 4 — Rainfall vs Soil Moisture**

        Higher rainfall days do tend to have higher soil moisture, but the relationship is
        not perfectly tight. Soil moisture is not determined by rainfall alone — prior soil
        conditions, evaporation rate, and temperature all play a role.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/avg_evaporation.png", caption="Viz 5 — Average Evapotranspiration by Month", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 5 — Average Evapotranspiration by Month**

        ET is low in winter and rises steadily through spring, peaking in June and July.
        The peak ET period lines up exactly with the lowest soil moisture months,
        confirming that summer is the highest-risk period for crops.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz10_dual_axis.png", caption="Viz 6 — Monthly Rainfall vs Soil Moisture Over Time", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 6 — Monthly Rainfall vs Soil Moisture Over Time**

        When rainfall spikes, soil moisture rises shortly after — not always in the same
        month. This lag effect reflects the time it takes water to soak into soil layers.
        During long dry stretches, the soil moisture line drops steadily.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz8_wind_et.png", caption="Viz 7 — Wind Speed vs Evapotranspiration", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 7 — Wind Speed vs Evapotranspiration**

        There is a moderate positive trend — higher wind speeds tend to come with higher ET,
        because wind removes the humid layer near the soil surface, allowing evaporation to
        continue faster. On windy days a smart system should account for extra water loss.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz7_seasonal_soil_moisture.png", caption="Viz 8 — Average Soil Moisture by Season", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 8 — Average Soil Moisture by Season**

        Winter leads at 30.79 m³/m³, followed by Spring at 24.63, Autumn at 15.97,
        and Summer at the lowest of 12.23 m³/m³ — less than half of winter levels.
        This sharp drop confirms summer is when automated irrigation is most urgently needed.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz9_temp_heatmap_year.png", caption="Viz 9 — Monthly Average Max Temperature by Year", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 9 — Monthly Average Max Temperature by Year**

        The consistent dark red band across June–August every single year confirms that
        summer heat is a reliable annual pattern. Irrigation schedules can be planned
        well in advance based on the calendar using this pattern.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz6_correlation_heatmap.png", caption="Viz 10 — Correlation Heatmap", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 10 — Correlation Heatmap**

        Temperature and ET are strongly positively correlated. Soil moisture and temperature
        are negatively correlated. Volumetric soil moisture at different depths are highly
        correlated with each other — when the surface is dry, deeper layers tend to be dry too.
        """)

# ─────────────────────────────────────────────────────────────
# TAB 3 — PCA
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Principal Component Analysis (PCA)")

    st.subheader("What is PCA?")
    st.markdown("""
    **Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction method that transforms
    correlated data columns into a smaller set of uncorrelated features called principal components.
    Every component represents a linear combination of the original features and is ordered so that the
    first component captures the most variance in the data and the second captures the next most. By keeping
    only the top k components, data complexity is reduced while retaining the majority of the original
    information, making visualisation, pattern detection, and downstream modelling less complex.
    """)
    st.divider()

    st.subheader("1. Dataset Used")
    st.markdown("""
    PCA was applied to the merged dataset combining daily NASA POWER weather data and Weatherbit
    agricultural soil data for California (2020 to 2025).

    Data preparation steps before PCA:
    - Selected only quantitative/numeric columns — no label columns included.
    - Removed any columns with near-zero variance.
    - Normalized all features using StandardScaler (mean = 0, std = 1).
    """)
    st.info("All qualitative columns (dates, season labels) were dropped before applying PCA as PCA requires numeric input.")
    st.divider()

    st.subheader("2. PCA with n_components = 2")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_2d.png", caption="2D PCA Projection coloured by Month", use_container_width=True)
    with col2:
        st.markdown("""
        **2D Projection Results**

        The 2D PCA scatter plot projects the 31-feature dataset into just two principal components.
        Points are coloured by month, showing a clear seasonal arc — winter months cluster to
        the right while summer months cluster to the left. This separation confirms that PC1
        captures seasonal temperature and evapotranspiration variation.

        - **PC1** captures **58.8%** of total variance
        - **PC2** captures **16.4%** of total variance
        - **Total retained in 2D: 75.2%**
        """)
    st.divider()

    st.subheader("3. PCA with n_components = 3")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_3d.png", caption="3D PCA Projection — coloured by Month", use_container_width=True)
    with col2:
        st.markdown("""
        **3D Projection Results**

        Adding a third principal component gives the data more depth and reveals additional
        structure not visible in 2D. PC3 helps separate transitional months like spring and autumn.

        - **PC1** captures **58.8%** of total variance
        - **PC2** captures **16.4%** of total variance
        - **PC3** captures **8.1%** of total variance
        - **Total retained in 3D: 83.3%**
        """)
    st.divider()

    st.subheader("4. Variance Retained Scree Plot & Cumulative Curve")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_variance.png", caption="Scree Plot and Cumulative Explained Variance", use_container_width=True)
    with col2:
        st.markdown("""
        | Components | Cumulative Variance |
        |-----------|-------------------|
        | 2 | ~75.2% |
        | 3 | ~83.3% |
        | 5 | ~89.0% |
        | **8** | **≥ 95.0%** |

        8 principal components are required to retain at least 95% of the variance.
        The scree plot shows a sharp elbow after PC1, confirming it captures the dominant signal.
        """)
    st.divider()

    st.subheader("5. Top Eigenvalues")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_eigenvalues.png", caption="Top 10 Eigenvalues of AgriSense Data", use_container_width=True)
    with col2:
        st.markdown("""
        | Principal Component | Eigenvalue |
        |--------------------|-----------|
        | PC1 | **18.24** |
        | PC2 | **5.09** |
        | PC3 | **2.51** |

        PC1's eigenvalue of ~18 is much larger than the rest, confirming it captures the dominant
        source of variation — the temperature–ET seasonal cycle. Components with eigenvalues below 1
        are generally not considered meaningful and can be discarded.
        """)
    st.divider()

    st.subheader("PCA Summary & Conclusions")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("2D Variance Retained", "75.2%")
    col2.metric("3D Variance Retained", "83.3%")
    col3.metric("Components for 95%", "8")
    col4.metric("Top Eigenvalue (PC1)", "18.24")
    st.success("""
    **Key Takeaways from PCA:**
    - The dataset variance is dominated by a strong seasonal signal (PC1 = 58.8%), driven by temperature and ET cycles.
    - Just 2 components capture ~75% of all information — ideal for visualization.
    - 8 components retain 95% of all data, reducing dimensionality from 31 to 8.
    - These reduced components can be fed directly into clustering and supervised ML models.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 4 — Clustering
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Clustering")
    st.subheader("Comparing Clustering Methods")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **KMeans**
        - Assigns each point to the nearest centroid
        - Minimises within-cluster variance
        - Requires k to be specified upfront
        - Fast and scalable
        - Assumes roughly spherical clusters
        """)
    with col2:
        st.markdown("""
        **Hierarchical (Agglomerative)**
        - Builds a tree by merging similar points bottom-up
        - Ward linkage minimises within-cluster variance at each merge
        - No need to specify k at the start
        - Interpretable tree structure
        - Slower on large datasets (O(n²))
        """)
    with col3:
        st.markdown("""
        **DBSCAN (Density-Based)**
        - Groups densely packed points into clusters
        - Marks sparse/isolated points as noise
        - No need to define k
        - Handles arbitrary cluster shapes
        - Sensitive to eps and min_samples parameters
        """)
    st.divider()

    st.subheader("1. Data Preparation for Clustering")
    st.markdown("""
    The following steps were applied before clustering:
    1. **Created seasonal labels** from the date column: Winter, Spring, Summer, Autumn
    2. **Selected numeric features only** — removed date, season string, and near-zero-variance columns.
    3. **Normalized with StandardScaler** — mean = 0, std = 1 per column.
    4. **Reduced to 3D via PCA** — retained ~83.3% of variance, enabling cleaner clustering and visualisation.
    """)
    st.divider()

    st.subheader("2. KMeans — Silhouette Method to Choose k")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/kmeans_silhouette.png", caption="Silhouette Scores for k = 2 to 10", use_container_width=True)
    with col2:
        st.markdown("""
        The Silhouette Score measures how similar each point is to its own cluster vs other clusters.
        A score closer to +1 means well-separated clusters.

        | k | Silhouette Score |
        |---|-----------------|
        | **2** | **0.497** → Best |
        | 3 | 0.440 |
        | 4 | 0.448 |

        k = 2 achieved the highest score, suggesting the data naturally splits into two broad groups —
        warm/dry season vs cool/wet season.
        """)
    st.divider()

    st.subheader("3. KMeans Cluster Plots (k = 2, 3, 4)")
    st.image("images/kmeans_clusters.png", caption="KMeans Clustering — coloured by Season, ✕ = Centroids", use_container_width=True)
    st.markdown("""
    - **k=2:** Cleanly separates into cool months vs warm months along PC1.
    - **k=3:** Adds a transitional cluster capturing spring and autumn days.
    - **k=4:** Further subdivides seasonal groups with reducing interpretability.
    """)
    st.divider()

    st.subheader("4. Hierarchical Clustering — Ward Linkage Dendrogram")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/dendrogram.png", caption="Ward Linkage Dendrogram", use_container_width=True)
    with col2:
        st.markdown("""
        The dendrogram shows how individual data points were merged bottom-up.
        A large gap at Ward Distance 40–70 suggests 2 main natural clusters — consistent
        with the silhouette result. A secondary split at distance ~30 shows a 3-cluster
        structure, aligning with the k=3 KMeans result.
        """)
    st.divider()

    st.subheader("5. KMeans vs Hierarchical (k=2)")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/kmeans_vs_hierarchical.png", caption="KMeans k=2 vs Hierarchical (Ward) k=2", use_container_width=True)
    with col2:
        st.markdown("""
        Both methods produce nearly identical cluster assignments. KMeans uses centroid positions
        for clean geometric boundaries. Hierarchical clustering is slightly more flexible at edges.
        The high agreement confirms the robustness of the seasonal clustering signal in the data.
        """)
    st.divider()

    st.subheader("6. DBSCAN — Density-Based Clustering")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/dbscan_kdist.png", caption="k-Distance Plot — used to choose eps", use_container_width=True)
    with col2:
        st.image("images/dbscan_clusters.png", caption="DBSCAN Results (eps=0.5, min_samples=10)", use_container_width=True)
    st.markdown("""
    The elbow in the k-distance plot appears around 0.5, giving the eps parameter.
    DBSCAN found 8 clusters and flagged ~12.3% of days as noise — these represent anomalous
    weather days with unusual combinations of temperature, humidity, and soil moisture.
    These noise points are actually useful: they flag days where standard irrigation rules may not apply.
    """)
    st.divider()

    st.subheader("Clustering Conclusions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Silhouette Score", "0.497", "k=2")
    col2.metric("Natural Clusters Found", "2–3", "All methods agree")
    col3.metric("DBSCAN Noise Points", "~12.3%", "Anomalous days")
    st.success("""
    **Key Takeaways from Clustering:**
    - All three methods identify 2 primary clusters aligned with warm vs cool seasons.
    - The seasonal temperature–ET cycle (PC1) is the dominant driver separating clusters.
    - DBSCAN uniquely identifies anomalous weather days as noise — useful for edge-case irrigation decisions.
    - These cluster labels can serve as features in downstream supervised ML models.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 5 — ARM
# ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Association Rule Mining (ARM)")

    st.subheader("(a) What is Association Rule Mining?")
    st.markdown("""
    **Association Rule Mining (ARM)** is an unsupervised machine learning technique that discovers
    interesting relationships between features in large datasets. It started in market-basket
    analysis ("customers who buy bread also buy butter") but applies broadly — in agriculture it
    can reveal which combinations of climate conditions co-occur with specific soil or irrigation states.

    A rule takes the form **X → Y**, meaning "when X occurs, Y also tends to occur."

    **Three key metrics define a rule's strength:**
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Support** — Fraction of all transactions containing both X and Y. Measures how frequently the rule applies overall.\n\nSupport(X→Y) = P(X ∩ Y)")
    with col2:
        st.info("**Confidence** — Probability that Y occurs given X is present. Measures how reliable the rule is.\n\nConfidence(X→Y) = P(Y | X)")
    with col3:
        st.info("**Lift** — How much more likely Y is given X vs random chance. Lift > 1 = positive association.\n\nLift = Confidence / P(Y)")

    st.markdown("""
    **The Apriori Algorithm** works in two stages:
    1. Find frequent itemsets — all combinations with Support ≥ min_support, using the Apriori property to prune.
    2. Generate rules — from each frequent itemset, generate candidates and filter by min_confidence.
    """)
    st.divider()

    st.subheader("(b) Data Preparation for ARM")
    st.markdown("""
    ARM requires unlabeled transaction data — a binary matrix where each row is a "transaction"
    (one day of observation) and each column is an "item" (a discretized condition).
    All numeric features were discretized into categorical bins using quantile-based binning:

    | Original Feature | Binned Items |
    |-----------------|-------------|
    | `T2M_MAX` | Temp_Low, Temp_Mid, Temp_High |
    | `PRECTOTCORR` | Rain_Low, Rain_Mid, Rain_High |
    | `RH2M` | Hum_Low, Hum_Mid, Hum_High |
    | `EVPTRNS` | ET_Low, ET_Mid, ET_High |
    | `v_soilm_0_10cm` | SoilM_Low, SoilM_Mid, SoilM_High |
    | Date → Season | Winter, Spring, Summer, Autumn |

    Thresholds: `min_support = 0.05`, `min_confidence = 0.40`
    """)
    st.divider()

    st.subheader("(c) ARM Results")

    st.markdown("#### Top 15 Rules by Support")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_support.png", caption="Top 15 Association Rules by Support", use_container_width=True)
    with col2:
        st.markdown("""
        The most common pattern is Rain_Low appearing alongside ET_Low or ET_High — showing up in
        over 50% of all days. This makes sense for a dry California climate where low rainfall is
        simply the norm. Almost every high-support rule ends in Rain_Low as the outcome, confirming
        that dry conditions dominate the dataset.
        """)
    st.divider()

    st.markdown("#### Top 15 Rules by Confidence")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_confidence.png", caption="Top 15 Association Rules by Confidence", use_container_width=True)
    with col2:
        st.markdown("""
        Every rule in this top-15 list has confidence of 1.0 (100%) — whenever those left-hand
        conditions occur, Rain_Low is always the outcome. For example, Autumn → Rain_Low (100%),
        and High Temp + Low Soil Moisture → Rain_Low (100%). These rules are directly usable
        as if-then irrigation triggers in an automated system.
        """)
    st.divider()

    st.markdown("#### Top 15 Rules by Lift")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_lift.png", caption="Top 15 Association Rules by Lift", use_container_width=True)
    with col2:
        st.markdown("""
        The highest-lift rules involve Spring + Moderate Humidity predicting Moderate Soil Moisture + High ET
        (lift ~4.38). This means in spring, moderate humidity is a strong predictor that ET is high and
        irrigation may be needed soon. These lift-ranked rules are the most surprising and actionable —
        they reveal patterns a simple calendar would miss.
        """)
    st.divider()

    st.subheader("(d) Visualizations")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("images/arm_scatter.png", caption="Support vs Confidence (size & colour = Lift)", use_container_width=True)
        st.markdown("Most rules cluster at low support but very high confidence. The most interesting rules appear in the lower-left — rare but highly specific combinations.")
    with col2:
        st.image("images/arm_network.png", caption="ARM Network Graph — Top 20 Rules by Lift", use_container_width=True)
        st.markdown("Nodes are itemsets, edges are rules. Spring + Humidity_Mid strongly connects to SoilM_Mid + ET_High. Hub-like nodes confirm the most conditionally dependent variables.")
    st.divider()

    st.subheader("(e) ARM Conclusions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Min Support Used", "0.05", "5% of days")
    col2.metric("Min Confidence Used", "0.40", "40% reliability")
    col3.metric("Top Lift Achieved", "4.38", "Spring → SoilM_Mid+ET_High")
    st.success("""
    **Key Takeaways:**
    - Rain_Low dominates — dry conditions are the norm in California's Central Valley.
    - Summer dryness is highly predictable from humidity and ET combinations alone.
    - Spring is the most complex season — moderate humidity reliably signals elevated ET and upcoming irrigation need.
    - Winter high-humidity days always correspond to full soil moisture — irrigation can safely stop.
    - These rules can be directly implemented as if-then irrigation decision logic in an automated system.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 6 — NB / DT
# ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("Naive Bayes & Decision Trees")

    # ── NAIVE BAYES ──────────────────────────────────────────
    st.subheader("(a) Overview — Naive Bayes")
    st.markdown("""
    Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem. It calculates
    the probability that a data point belongs to each class, and assigns it to the class with the
    highest probability. The "naive" part means the algorithm assumes all features are independent
    of each other — which is rarely true in practice but still works surprisingly well on many
    real-world datasets. It is fast, requires very little data to train, and handles high-dimensional
    data well. Naive Bayes is commonly used in spam detection, sentiment analysis, and medical diagnosis.
    It works especially well when the independence assumption is roughly satisfied, and performs
    reasonably even when it is not. For this project, Naive Bayes is used to predict whether a given
    day requires irrigation based on weather and soil features.

    There are four main types of Naive Bayes, each suited to different data types:
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("""
        **Multinomial NB**
        Designed for discrete count data like word frequencies in text.
        Requires non-negative integer inputs.
        Used here by scaling features to integers (0–100).
        Best for: text classification, document analysis.
        """)
    with col2:
        st.info("""
        **Gaussian NB**
        Assumes features follow a normal (Gaussian) distribution.
        Works directly with continuous float values.
        Best suited for real-valued measurements like temperature and humidity.
        Best for: continuous sensor data, weather data.
        """)
    with col3:
        st.info("""
        **Bernoulli NB**
        Designed for binary (0/1) features.
        Features are binarized — above mean = 1, below = 0.
        Loses some information but works well when features are presence/absence signals.
        Best for: binary encoded data, text with word presence.
        """)
    with col4:
        st.info("""
        **Categorical NB**
        Designed for features with discrete categories.
        Each feature must have a fixed set of possible values.
        Requires features to be integer-encoded categories.
        Best for: survey data, ordinal categories.
        """)

    st.markdown("""
    For this project, **Multinomial, Gaussian, and Bernoulli NB** were implemented since the dataset
    contains continuous weather and soil measurements — Gaussian NB is the most natural fit, while
    Multinomial and Bernoulli allow comparison of how different data assumptions affect performance.
    Categorical NB was not used because the features are continuous, not discrete categories.
    """)
    st.divider()

    # ── NB Data Prep ─────────────────────────────────────────
    st.subheader("(b) Data Preparation — Naive Bayes")
    st.markdown("""
    The label used for all three Naive Bayes models is **irrigation_needed** — a binary label (0 or 1)
    created from three conditions: soil moisture at 0–10 cm below the 40th percentile, evapotranspiration
    above the 60th percentile, and daily rainfall below 1 mm. A day satisfying all three conditions is
    labeled as needing irrigation. This resulted in about 30.7% positive class and 69.3% negative class.
    The label was carefully designed to avoid data leakage — the two columns used to build the label
    (evapotranspiration and PRECTOTCORR) were removed from the feature set before training.
    """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/train_test_split.png", caption="Train/Test Split — 80% Training, 20% Testing", use_container_width=True)
    with col2:
        st.markdown("""
        **Train / Test Split**

        The dataset of 2031 rows was split into 80% training (1624 rows) and 20% testing (407 rows).
        A stratified split was used to ensure the class balance (30.7% positive) is preserved in
        both the training and testing sets. The two sets are completely disjoint — no row appears
        in both. This is essential for getting an honest estimate of how the model performs on
        data it has never seen before. If training and test data overlap, accuracy scores become
        meaningless since the model has already memorized those examples.
        """)

    st.markdown("""
    Each NB flavor requires a different data format. The table below summarizes what was done:

    | NB Flavor | Data Format Required | Preparation Applied |
    |-----------|---------------------|---------------------|
    | Multinomial NB | Non-negative integers | MinMaxScaler → scale to [0,1] → multiply by 100 → round to int |
    | Gaussian NB | Continuous floats | StandardScaler (mean=0, std=1) |
    | Bernoulli NB | Binary 0/1 | Binarize at mean — above mean = 1, below = 0 |
    """)

    st.image("images/nb_before_data.png", caption="Sample of prepared data for each NB flavor (first 5 rows of training set)", use_container_width=True)

    st.markdown("""
    **[Link to raw dataset](https://github.com/shivanibhinge/AgriSense_Data_Driven_Irrigation_Insights/tree/main)**
    &nbsp;&nbsp;|&nbsp;&nbsp;
    **[Link to Module 3 Colab Notebook](https://github.com/shivanibhinge/AgriSense_Data_Driven_Irrigation_Insights/tree/main)**
    """)
    st.divider()

    # ── NB Results ───────────────────────────────────────────
    st.subheader("(c) Naive Bayes Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Multinomial NB", "86.24%")
    col2.metric("Gaussian NB", "88.70%")
    col3.metric("Bernoulli NB", "85.01%")

    st.image("images/nb_confusion_matrices.png", caption="Confusion Matrices — All 3 Naive Bayes Flavors", use_container_width=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/nb_accuracy_comparison.png", caption="Accuracy Comparison — Multinomial vs Gaussian vs Bernoulli NB", use_container_width=True)
    with col2:
        st.markdown("""
        **Interpreting the Results**

        Gaussian NB performs best at 88.70% because the weather and soil features — temperature,
        humidity, soil temperature, wind speed — are all continuous measurements that naturally
        follow a bell-curve distribution. Gaussian NB is mathematically designed for this type
        of data. Multinomial NB comes in second at 86.24%, which is solid given that it was
        designed for discrete count data, not continuous weather values. Bernoulli NB scores
        85.01% — the lowest of the three — because binarizing features at the mean discards
        the actual magnitude of each reading, losing useful information. All three models
        significantly outperform the naive baseline of always predicting "no irrigation",
        which would give 69.3% accuracy. The real gain is 16–19 percentage points above
        that baseline, which shows the models are genuinely learning patterns in the data.
        """)
    st.divider()

    # ── NB Conclusions ───────────────────────────────────────
    st.subheader("(d) Naive Bayes Conclusions")
    st.markdown("""
    Naive Bayes proved to be a practical and fast classifier for the irrigation prediction task.
    Gaussian NB was the best-performing flavor because the underlying data is continuous and
    approximately normally distributed — matching the core assumption of the Gaussian model.
    Multinomial NB performed well despite being designed for count data, showing that the
    scaled integer representation still captures enough signal for useful predictions. Bernoulli
    NB lost some performance by collapsing continuous features into binary signals, but its
    85% accuracy still makes it a viable option in resource-constrained settings where simplicity
    matters. One important finding is that all three models perform consistently, with less than
    4 percentage points separating the best and worst — this stability suggests the underlying
    patterns in the data are strong enough that even a less ideal model can pick them up. For
    a real irrigation system, Gaussian NB would be the recommended choice — it is fast, requires
    no intensive data transformation, and naturally handles the continuous sensor readings that
    a field deployment would produce.
    """)
    st.divider()

    # ── DECISION TREES ───────────────────────────────────────
    st.subheader("(e) Overview — Decision Trees")
    st.markdown("""
    A decision tree is a supervised machine learning model that learns a series of if-then rules
    from the training data and organizes them into a tree structure. At each internal node, the
    tree asks a question about one feature — for example "is soil temperature above 15°C?" — and
    splits the data into two branches based on the answer. This continues until the data in each
    branch is pure enough (all one class) or a stopping criterion like maximum depth is reached.
    Decision trees are easy to interpret because the learned rules can be visualized and read
    directly. They are also non-parametric, meaning they make no assumptions about the distribution
    of the data, unlike Naive Bayes. A key limitation is that a single tree can overfit — it can
    memorize the training data perfectly while generalizing poorly to new data. This is why
    controlling depth and other hyperparameters is important.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/dt_tree1.png", caption="Tree 1 — Shallow GINI tree (depth=3)", use_container_width=True)
    with col2:
        st.image("images/dt_tree2.png", caption="Tree 2 — Entropy tree (depth=5)", use_container_width=True)

    st.markdown("""
    **GINI Impurity, Entropy, and Information Gain**

    When building a tree, the algorithm needs a way to measure how "good" a split is. Two common
    measures are GINI impurity and Entropy. GINI impurity measures the probability that a randomly
    chosen data point would be misclassified if randomly labeled according to the class distribution
    in that node. A perfectly pure node (all one class) has GINI = 0. Entropy measures disorder —
    a pure node has entropy = 0 and a maximally mixed node has the highest entropy.
    Information Gain is the reduction in impurity or entropy achieved by a split — the tree always
    picks the split that gives the highest information gain.

    **Example from this dataset:**
    - Training set: 1624 rows, 69.3% class 0 (no irrigation), 30.7% class 1 (needs irrigation)
    - GINI at root = 1 − (0.693² + 0.307²) = **0.426**
    - Entropy at root = −(0.693 × log₂(0.693) + 0.307 × log₂(0.307)) = **0.896 bits**

    The best split found by Tree 1 was on `soilt_0_10cm` (soil temperature at 0–10 cm depth),
    which reduced GINI significantly — this means soil temperature is the single most informative
    feature for deciding whether irrigation is needed.

    **Why can you build an infinite number of trees?**
    Because there is no single correct tree for any dataset. You can vary the splitting criterion
    (GINI vs Entropy), the maximum depth, the minimum samples required to split, the features
    considered at each node, and the random seed. Each combination produces a different tree
    structure with a different root node and different branches. Even using the same algorithm
    and data, there are often multiple splits with nearly identical information gain scores,
    and choosing any of them leads to a different tree. This is why ensembles like Random Forests
    build hundreds of different trees and average their predictions — diversity across trees
    reduces the overall error.
    """)
    st.divider()

    # ── DT Data Prep ─────────────────────────────────────────
    st.subheader("(f) Data Preparation — Decision Trees")
    st.markdown("""
    Decision trees do not require feature scaling — they are scale invariant, meaning the same
    splits are chosen regardless of whether the data is normalized or not. The same binary label
    (irrigation_needed) and the same clean feature set used for Naive Bayes were reused here.
    The same 80/20 stratified train/test split (random_state=42) was applied so all models
    are evaluated on the exact same held-out test rows, making accuracy comparisons fair.

    Three trees were built with different structures to show how varying parameters affects
    the resulting model:

    | Tree | Depth | Criterion | Features | Root Node |
    |------|-------|-----------|----------|-----------|
    | Tree 1 | 3 | GINI | All 10 | soilt_0_10cm |
    | Tree 2 | 5 | Entropy | All 10 | soilt_0_10cm |
    | Tree 3 | 5 | GINI | 7 (no soil temp) | T2M_MAX |

    Tree 3 was built with soil temperature features removed, forcing the tree to find a different
    root node — it chose T2M_MAX (maximum air temperature), which is the next most informative
    feature for separating irrigation-needed days.
    """)

    st.markdown("""
    **[Link to raw dataset](https://github.com/shivanibhinge/AgriSense_Data_Driven_Irrigation_Insights/tree/main)**
    &nbsp;&nbsp;|&nbsp;&nbsp;
    **[Link to Module 3 Colab Notebook](https://github.com/shivanibhinge/AgriSense_Data_Driven_Irrigation_Insights/tree/main)**
    """)
    st.divider()

    # ── DT Results ───────────────────────────────────────────
    st.subheader("(g) Decision Tree Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tree 1 — depth=3 GINI", "91.65%")
    col2.metric("Tree 2 — depth=5 Entropy", "92.87%")
    col3.metric("Tree 3 — depth=5 no soil temp", "88.45%")

    st.image("images/dt_confusion_matrices.png", caption="Confusion Matrices — All 3 Decision Trees", use_container_width=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/dt_accuracy_comparison.png", caption="Accuracy Comparison — 3 Decision Trees", use_container_width=True)
    with col2:
        st.markdown("""
        **Interpreting the Results**

        All three trees perform well above the 69.3% naive baseline, with accuracies
        ranging from 88.45% to 92.87%. Tree 2 (depth=5, Entropy) scores the highest at
        92.87% — the extra depth allows it to capture more complex patterns than the
        shallow Tree 1. Tree 3 drops to 88.45% because removing soil temperature features
        forces it to rely on air temperature and humidity, which are slightly less directly
        linked to the irrigation label. The fact that even Tree 3 achieves 88% without
        soil temperature shows that air temperature alone carries substantial predictive
        signal — this is useful to know for deployments where soil sensors may not be
        available. The confusion matrices show that all three trees are better at
        identifying "no irrigation" days than "needs irrigation" days, which is expected
        given the 70/30 class imbalance.
        """)

    st.image("images/dt_feature_importance.png", caption="Feature Importances — Tree 1 (GINI, depth=3)", use_container_width=True)
    st.divider()

    # ── DT Conclusions ───────────────────────────────────────
    st.subheader("(h) Decision Tree Conclusions")
    st.markdown("""
    Decision trees provided clear, interpretable results for the irrigation prediction task.
    The most important feature across all three trees was soil temperature at 0–10 cm depth,
    followed closely by maximum air temperature. This makes physical sense — soil temperature
    is a proxy for how hot and dry the near-surface environment is, directly tied to whether
    irrigation is needed. The shallow Tree 1 (depth=3) is the most interpretable model —
    it can be printed as a small set of rules and handed directly to a farmer. For example:
    if soil temperature at 0–10 cm is above a threshold and maximum temperature is high,
    irrigation is very likely needed. Tree 2 improves accuracy by going deeper but becomes
    harder to read visually. Tree 3 shows that even without soil sensors, air temperature
    features can still drive 88% accuracy — making the model viable for weather-only deployments.
    Across all three trees, the consistency of results (all within 4 percentage points)
    confirms that the decision tree approach is robust and not overly sensitive to parameter choices.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 7 — SVM / Ensemble
# ─────────────────────────────────────────────────────────────
with tabs[7]:
    st.header("SVM & Ensemble Methods")
    st.info("This section will be completed in Module 4.")

# ─────────────────────────────────────────────────────────────
# TAB 8 — Regression
# ─────────────────────────────────────────────────────────────
with tabs[8]:
    st.header("Regression")

    # ── Conceptual Questions ─────────────────────────────────
    st.subheader("(a) What is Linear Regression?")
    st.markdown("""
    Linear regression is a statistical method that models the relationship between a continuous
    outcome variable and one or more input features by fitting a straight line (or hyperplane)
    through the data. The model learns a set of weights — one per feature — that minimize the
    sum of squared differences between the predicted values and the actual values. This is called
    the ordinary least squares criterion. The output of linear regression is a real-valued number,
    making it suitable for tasks like predicting soil moisture depth, estimating daily
    evapotranspiration, or forecasting water demand in liters. Linear regression assumes a
    linear relationship between inputs and output, constant variance in errors (homoscedasticity),
    and no strong multicollinearity between features. It is one of the oldest and most widely
    used models in statistics and data science, valued for its simplicity and interpretability.
    """)
    st.divider()

    st.subheader("(b) What is Logistic Regression?")
    st.markdown("""
    Logistic regression is a classification algorithm — despite the name containing "regression."
    Rather than predicting a continuous number, it predicts the probability that an observation
    belongs to a particular class, and then assigns the observation to the class with the highest
    probability. Internally, it applies the sigmoid function to a linear combination of features,
    which squashes any real-valued output into the range (0, 1). A decision threshold — usually
    0.5 — converts this probability into a class label. Logistic regression is widely used in
    binary classification problems such as spam vs not spam, disease vs healthy, and in this
    project, irrigation needed vs not needed. It is fast to train, produces interpretable
    coefficients, and handles both numerical and scaled categorical features well. Like linear
    regression, it assumes a linear relationship between the log-odds of the outcome and the features.
    """)
    st.divider()

    st.subheader("(c) Similarities and Differences")
    st.markdown("""
    Both linear and logistic regression are linear models — they combine features using learned
    weights and a bias term, and both are trained by optimizing a loss function. Both are
    interpretable: the coefficients show how much each feature contributes to the prediction.
    The key difference is in what they predict. Linear regression outputs a continuous number
    and minimizes mean squared error. Logistic regression outputs a probability between 0 and 1
    and minimizes log-loss (binary cross-entropy). Linear regression can produce outputs outside
    [0, 1], which would be meaningless for a probability — the sigmoid function in logistic
    regression solves this by bounding the output. Linear regression is used for regression
    tasks, logistic regression is used for classification tasks. Despite the naming confusion,
    they are fundamentally different tools built for different output types.
    """)
    st.divider()

    st.subheader("(d) Does Logistic Regression Use the Sigmoid Function?")
    st.markdown("""
    Yes — the sigmoid function is the core mathematical component that makes logistic regression
    work as a classifier. The sigmoid is defined as σ(z) = 1 / (1 + e^{−z}), where z is the
    linear combination of features: z = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ. This function takes
    any real number and maps it to a value strictly between 0 and 1, which can be interpreted
    as a probability. When z is very large and positive, σ(z) approaches 1 — the model is
    confident about the positive class. When z is very large and negative, σ(z) approaches 0 —
    the model is confident about the negative class. Without the sigmoid, the linear combination
    would produce arbitrary real numbers that cannot be interpreted as probabilities or used
    directly as class predictions. The sigmoid is what turns a linear regression formula into
    a probabilistic classifier.
    """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/lr_sigmoid.png", caption="Sigmoid Function — maps any real number to probability (0, 1)", use_container_width=True)
    with col2:
        st.markdown("""
        The sigmoid curve starts near 0 for large negative z values and rises smoothly to near 1
        for large positive z values. The decision boundary sits at z = 0, where the probability
        equals exactly 0.5. Above this boundary the model predicts "irrigation needed", below it
        the model predicts "no irrigation". The green and orange shading shows which side of the
        boundary corresponds to which class. The shape of this curve is why logistic regression
        produces smooth, well-calibrated probability estimates rather than hard cutoffs.
        """)
    st.divider()

    st.subheader("(e) Maximum Likelihood and Logistic Regression")
    st.markdown("""
    Maximum likelihood estimation (MLE) is the principle used to train logistic regression.
    The idea is to find the set of model parameters (the weights) that make the observed training
    labels as probable as possible given the model's predictions. For each training example, the
    model outputs a probability p — if the true label is 1, the likelihood contribution is p;
    if the true label is 0, it is (1 − p). The total likelihood is the product of all individual
    contributions across the training set. In practice, the log-likelihood is maximized instead
    (which is equivalent but numerically more stable), and this is identical to minimizing binary
    cross-entropy loss. Gradient descent is used to iteratively update the weights until the
    log-likelihood is maximized. The connection to logistic regression is direct — MLE ensures
    that the model assigns the highest possible probability to the labels it actually observed,
    which is exactly the goal of a good classifier.
    """)
    st.divider()

    # ── Logistic Regression Coding Section ───────────────────
    st.subheader("(f) Logistic Regression — Coding & Results")
    st.markdown("""
    Logistic regression was applied to the same binary label (irrigation_needed) and the same
    clean feature set used for Naive Bayes and Decision Trees. The data was standardized using
    StandardScaler before fitting, since logistic regression is sensitive to feature scale.
    The same 80/20 stratified split was used for fair comparison across all models.

    **Features used (10 total — leaky features removed):**
    T2M_MAX, T2M_MIN, RH2M, EVPTRNS, wind_10m_spd_avg, skin_temp_avg,
    soilt_0_10cm, soilt_10_40cm, specific_humidity, pres_avg

    **[Link to raw dataset](https://github.com/shivanibhinge/AgriSense_Data_Driven_Irrigation_Insights/tree/main)**
    &nbsp;&nbsp;|&nbsp;&nbsp;
    **[Link to Module 3 Colab Notebook](https://github.com/shivanibhinge/AgriSense_Data_Driven_Irrigation_Insights/tree/main)**
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression Accuracy", "92.38%")
    col2.metric("Precision (Irrigation Needed)", "87%")
    col3.metric("Recall (Irrigation Needed)", "84%")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("images/lr_confusion_matrix.png", caption="Logistic Regression — Confusion Matrix", use_container_width=True)
    with col2:
        st.image("images/lr_coefficients.png", caption="Logistic Regression — Feature Coefficients", use_container_width=True)

    st.markdown("""
    **Reading the Coefficient Plot:**
    Red bars indicate features that increase the probability of irrigation being needed — soil
    temperature and maximum air temperature are the strongest positive predictors, which matches
    the physical intuition that hot soil means dry soil. Blue bars indicate features that decrease
    the irrigation probability — higher humidity and pressure are associated with cooler, wetter
    conditions where irrigation is less likely needed.
    """)
    st.divider()

    # ── LR vs GNB ────────────────────────────────────────────
    st.subheader("(g) Logistic Regression vs Gaussian NB — Comparison")
    st.image("images/lr_vs_gnb.png", caption="Logistic Regression vs Gaussian NB — Confusion Matrices Side by Side", use_container_width=True)

    st.markdown("""
    Both models were trained and tested on the same data split for a direct comparison.
    Logistic regression achieves 92.38% accuracy vs Gaussian NB at 88.70% — a difference of
    about 3.7 percentage points. Logistic regression is stronger at identifying the positive
    class (days that need irrigation), with better precision and recall on the minority class.
    This makes sense because logistic regression explicitly optimizes for the decision boundary
    between classes, while Gaussian NB makes distributional assumptions that are only
    approximately true for this data. However, Gaussian NB is significantly faster to train
    and requires no hyperparameter tuning. For a real-time irrigation system that retrains
    daily on new sensor data, Gaussian NB's speed advantage may matter more than the 3.7%
    accuracy gap. For a one-time offline model, logistic regression is the better choice.
    """)
    st.divider()

    # ── Final Summary ─────────────────────────────────────────
    st.subheader("(h) Final Model Comparison — All Module 3 Models")
    st.image("images/all_models_accuracy.png", caption="Accuracy Comparison — All 7 Models from Module 3", use_container_width=True)

    st.markdown("""
    Across all seven models trained in Module 3, accuracy ranges from 85% to 93%. The
    baseline dummy classifier (always predict "no irrigation") would score 69.3%, so the
    true performance gain from all models is 15 to 23 percentage points above random guessing.
    Decision Tree (depth=5, Entropy) and Logistic Regression tie for the best performance at
    ~92–93%. Gaussian NB is the best Naive Bayes variant at 88.7%. Multinomial and Bernoulli
    NB are close behind at 85–86%. All models are honest — trained on clean features with no
    leakage, validated on a held-out test set, and cross-validated to confirm stability.
    The results show that the irrigation prediction task is well-suited to supervised learning
    and that multiple model types can solve it effectively with this dataset.
    """)

    st.success("""
    **Key Takeaways from Regression:**
    - Logistic Regression (92.38%) is the best single model for binary irrigation prediction.
    - The sigmoid function converts the linear model output into a calibrated probability.
    - Feature coefficients confirm that soil temperature and max air temperature drive irrigation need.
    - All models beat the 69.3% naive baseline by 15–23 percentage points — genuine learning.
    - LR is preferred over GNB when accuracy matters most; GNB when speed and simplicity matter.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 9 — Conclusions
# ─────────────────────────────────────────────────────────────
with tabs[9]:
    st.header("Conclusions")
    st.info("This section will be completed in the Final Project submission.")

# ─────────────────────────────────────────────────────────────
# TAB 10 — About Me
# ─────────────────────────────────────────────────────────────
with tabs[10]:
    st.header("About Me")
    st.markdown("""
    **Name:** Shivani Atul Bhinge

    **Project:** Smart Water Usage Prediction — AgriSense
    """)
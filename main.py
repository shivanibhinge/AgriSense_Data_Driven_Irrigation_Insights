# app.py
# Streamlit Smart Water Usage Project Website (Module 1 -> Modules 2-4 scaffolding)
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
    " DT / NB",
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

    Agriculture consumes the majority of global freshwater resources. Optimizing irrigation timing and amount can substantially reduce water use while maintaining crop yields. Smart water usage prediction combines weather variables (temperature, rainfall, evapotranspiration) with soil moisture observations to determine when irrigation is necessary. This project will focus on building a reproducible pipeline — collecting data via APIs, cleaning & merging datasets, performing exploratory data analysis, and applying unsupervised and supervised machine learning to predict irrigation-relevant targets.
    """)
    st.markdown("""
    Water is the essential resource for food production. However, inefficient irrigation practices waste large amounts of water and results in higher costs for farmers. With climate variability increasing the frequency of droughts and extreme rainfall events, data-driven irrigation scheduling can improve reliability and sustainability. This project will explore on how daily climate and soil data can be combined to make informed decisions on irrigation timing decisions that conserve water while supporting crop health.

    The aim is to represent an iterative data science process that collects real API data, cleans and transforms it then conducts exploratory analysis to identify key factors of soil moisture and evapotranspiration and then will apply machine learning techniques to predict irrigation needs. The final website will present results which are easy to understand for stakeholders (like farmers, policy makers, and the public) and can understand and act upon.
    """)

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

    st.subheader("1️. Data Gathering Overview")
    st.markdown("""
    The data for this project was collected from two APIs:

    1. **NASA POWER API** – provides daily weather parameters such as maximum and minimum temperature, humidity, precipitation, and evapotranspiration.
    2. **Weatherbit AgWeather API** – provides soil moisture, evapotranspiration, temperature, precipitation, and wind data specific to agricultural applications.

    The data was gathered for **California (Latitude: 36.77, Longitude: -119.41)** for the years **2018–2023**.  
    NASA data was received in a tabular format (CSV-like), while the Weatherbit API returned **JSON**, which was flattened into a DataFrame.

    Both datasets were merged on the **date field** to create one unified dataset combining daily soil and weather parameters.
    """)
    st.image("images/dataset_merged.png", caption="Preview of the merged weather–soil dataset (first few rows)", use_container_width=True)

    st.subheader("2️. Data Cleaning and Merging")
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

    st.subheader("3 Key Parameters in the Final Dataset")
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
    """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/daily_weather_soil_moisture_trends.png", caption="Daily Weather & Soil Moisture Trends", use_container_width=True)
    with col2:
        st.markdown("""
        **Daily Weather Soil Moisture Trends**

        This time-series plot displays how daily maximum temperature, rainfall, and soil moisture change over time. A clear seasonal pattern is visible in temperature, with higher values in summer and lower values in winter. Rainfall occurs in irregular spikes, and increases in rainfall are often followed by small rises in soil moisture.
        """)

    st.divider()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/monthly_soil_moisture.png", caption="Monthly Soil Moisture Box Plot", use_container_width=True)
    with col2:
        st.markdown("""
        **Monthly Soil Moisture Trends**

        This boxplot displays the distribution and variability of soil moisture across months. Soil moisture levels are highest and most stable during winter months. As the year progresses into late spring and summer, soil moisture decreases sharply and becomes more variable, reflecting higher evaporation rates and increased plant water uptake.
        """)

    st.divider()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/max_temp_rainfall.png", caption="Max Temperature vs Rainfall", use_container_width=True)
    with col2:
        st.markdown("""
        **Maximum Temperature vs Rainfall**

        Most heavy rainfall events occur at lower to moderate temperatures, while very high temperatures are generally associated with little or no rainfall — suggesting an inverse relationship between heat and precipitation.
        """)

    st.divider()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/rainfall_soil_moisture.png", caption="Rainfall vs Soil Moisture", use_container_width=True)
    with col2:
        st.markdown("""
        **Rainfall vs Soil Moisture**

        Higher rainfall values are generally associated with higher soil moisture levels. However, variability shows that soil moisture is also influenced by evaporation and prior soil conditions.
        """)

    st.divider()
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/avg_evaporation.png", caption="Average Evapotranspiration by Month", use_container_width=True)
    with col2:
        st.markdown("""
        **Average Evapotranspiration by Month**

        ET values increase steadily from winter to early summer, reaching a peak during June and July. This pattern reflects rising temperatures, longer daylight hours, and increased plant activity during the growing season — indicating periods of highest irrigation demand.
        """)

# ─────────────────────────────────────────────────────────────
# TAB 3 — PCA  (NEW)
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Principal Component Analysis (PCA)")

    # ── What is PCA ──────────────────────────────────────────
    st.subheader("What is PCA?")
    st.markdown("""
    **Principal Component Analysis (PCA)** is an unsupervised dimensionality reduction method that transforms the data columns which are correlated to smaller set of uncorelated features called principle component.
    Every component represents a linear combination of the original features and ordered so that the first component captures the
    most variance in the data and the second captures the next most. By keeping only the top k components,
    we can reduce data complexity while maintaining the majority of the original information and making
    visualisation, pattern detection, and downstream modelling less complex.
    """)

    st.divider()

    # ── Dataset used ─────────────────────────────────────────
    st.subheader("1. Dataset Used")
    st.markdown("""
    PCA applied to the Merged dataset which combines daily NASA POWER
    weather data and Weatherbit agricultural soil data for California (from 2020 to 2023).

    -> Data preparation before applying PCA: 
    - Selected only quantitative/numeric columns and no label columns included.
    - Removed any columns with close to zero variance.
    - Normalized all features using StandardScaler (mean = 0, std = 1).
    """)

    st.info("All qualitative columns (like - dates, season labels) were dropped before applying PCA as PCA requires numeric input.")

    st.divider()
#2d
    st.subheader("2️. PCA with n_components = 2")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_2d.png", caption="2D PCA Projection coloured by Month", use_container_width=True)
    with col2:
        st.markdown("""
        **2D Projection Results**

        The 2D PCA scatter plot projects the all the 31-feature dataset into just two principal components.
        Here points are coloured by month, showing a clear seasonal arc ie winter months (blue/purple)
        cluster to the right while summer months (red/orange) cluster to the left.

        This separation confirms that PC1 captures seasonal temperature and evapotranspiration variation,
        which are the main drivers of irrigation demand in agricultural settings.

        - **PC1** captures **58.5%** of total variance
        - **PC2** captures **15.8%** of total variance
        - **Total retained in 2D: 74.3%**
        """)

    st.divider()

    # ── 3D PCA ───────────────────────────────────────────────
    st.subheader("3️. PCA with n_components = 3")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_3d.png", caption="3D PCA Projection — coloured by Month", use_container_width=True)
    with col2:
        st.markdown("""
        **3D Projection Results**

        Adding third principal component gives the data more depth and suggests additional structure
        not visible in the 2D view. The seasonal colour gradient is maintained and PC3 helps
        separate transitional months (like spring/autumn) also.

        - **PC1** captures **58.5%** of total variance
        - **PC2** captures **15.8%** of total variance
        - **PC3** captures **8.0%** of total variance
        - **Total retained in 3D: 82.3%**

        The 3D representation is especially useful for clustering tasks, providing cleaner
        view between seasonal groups.
        """)

    st.divider()

    # ── Variance Analysis ────────────────────────────────────
    st.subheader("4. Variance Retained Scree Plot & Cumulative Curve")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_variance.png", caption="Scree Plot and Cumulative Explained Variance", use_container_width=True)
    with col2:
        st.markdown("""
        **Plot for how many components do we need to retain 95% of the data?**

        From the cumulative explained variance curve:

        | Components | Cumulative Variance |
        |-----------|-------------------|
        | 2 | ~74.3% |
        | 3 | ~82.3% |
        | 5 | ~89.0% |
        | **8** | **≥ 95.0%** |

        Hence 8 principal components are required to retain at least 95% of the
        variance from the data.

        The scree plot shows a sharp elbow after PC1, suggesting that the first component
        is main and subsequent components each contribute reducing amounts of information.
        """)

    st.divider()

    # ── Eigenvalues ──────────────────────────────────────────
    st.subheader("5️. Top Eigenvalues")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_eigenvalues.png", caption="Top 10 Eigenvalues of AgriSense Data", use_container_width=True)
    with col2:
        st.markdown("""
        **Top 3 Eigenvalues:**

        | Principal Component | Eigenvalue |
        |--------------------|-----------|
        | PC1 | **18.14** |
        | PC2 | **4.89** |
        | PC3 | **2.46** |

        Eigenvalues represent how much variance each principal component explains.
        PC1's eigenvalue of ~18 is larger than the rest this confirms it
        captures the dominant source of variation in the data ie most likely the
        temperature–evapotranspiration seasonal cycle which is the core driver
        of agricultural water demand.

        Components with eigenvalues < 1  are generally not considered
        meaningful and can be discarded.
        """)

    st.divider()

    # ── PCA Summary ──────────────────────────────────────────
    st.subheader("PCA Summary & Conclusions")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("2D Variance Retained", "74.3%")
    col2.metric("3D Variance Retained", "82.3%")
    col3.metric("Components for 95%", "8")
    col4.metric("Top Eigenvalue (PC1)", "18.14")

    st.success("""
    **Key Takeaways from PCA:**
    - The dataset's variance is dominated by a strong seasonal signal (PC1 = 58.5%), driven by temperature and ET cycles.
    - Just 2 components capture ~74% of all information that is ideal for visualization.
    - 8 components are enough to retain 95% of all data, reducing dimensionality from 31 to 8.
    - These reduced components can be fed directly into clustering and supervised ML models to improve performance and reduce overfitting.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 4 — Clustering
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🔶 Clustering")

    # ── Overview ─────────────────────────────────────────────
    st.subheader("Comparing Clustering Methods")
    st.markdown("""
        Three clustering approaches were applied to the AgriSense dataset and compared:
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            **KMeans**
            - Assigns each point to the nearest centroid
            - Minimises within-cluster variance (ie inertia)
            - Requires k for optimal clusters
            - Fast and scalable
            - Assumes roughly spherical clusters
            """)
    with col2:
        st.markdown("""
            **Hierarchical (Agglomerative)**
            - Builds a tree (dendrogram) by merging similar points bottom-up
            - Ward linkage used ie minimises within-cluster variance at each merge
            - No need to specify k at the start
            - Interpretable tree structure
            - Can be slower on large datasets (O(n²))
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

    # ── Data Prep ────────────────────────────────────────────
    st.subheader("1️. Data Preparation for clustering")
    st.markdown("""
        The following steps were applied before clustering:

        1. **Created seasonal labels** from the date column: Winter, Spring, Summer, Autumn
        2. **Selected numeric features only** ie removed date, season string, and any close to zero-variance columns.
        3. **Normalized with StandardScaler** — mean = 0, std = 1 per column.
        4. **Reduced to 3D via PCA** — retained **~82.3%** of variance, enabling cleaner clustering and 2D/3D visualisation.
        """)

    st.divider()

# KMeans Silhouette
    st.subheader("2️. KMeans — Silhouette Method to Choose k")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/kmeans_silhouette.png", caption="Silhouette Scores for k = 2 to 10", use_container_width=True)
    with col2:
        st.markdown("""
            **k selection method**

            The **Silhouette Score** measures how similar each point is to its own cluster compared to other clusters.
            A score closer to **+1** means well-separated clusters; closer to **0** means they are overlapping.

            | k | Silhouette Score |
            |---|-----------------|
            | **2** | **0.494** ->  Best |
            | 3 | 0.425 |
            | 5 | 0.398 |
            | 6 | 0.379 |

            The top 3 values chosen are:  k = 2, 3, and 5 based on highest silhouette scores.

            k = 2 achieved the highest score of **0.494**, this suggests the data naturally separated into
            two broad groups ie most likely **warm/dry season vs cool/wet season**.
            """)

    st.divider()

# KMeans Cluster Plots
    st.subheader("3️. KMeans Cluster Plots (k = 2, 3, 5)")
    st.image("images/kmeans_clusters.png",
             caption="KMeans Clustering for k=2, k=3, k=5 — coloured by Season, ✕ = Centroids",
             use_container_width=True)
    st.markdown("""
        **Observations:**
        - **k=2:** Cleanly separates the data into two halves along PC1 and the dominant seasonal axis. The centroid positions
          confirm one cluster covers cool months and the other covers warm months.
        - **k=3:** Adds a transitional cluster in the centre, capturing seasons like spring and autumn days that sit between the two extremes.
        - **k=5:** Further subdivides the seasonal groups but with reducing interpretability clusters begin to overlap.

        Across all three values, **summer and winter** points consistently group separately, confirming
        that seasonal temperature and ET cycles are the primary drivers of irrigation demand patterns in the dataset.
        """)

    st.divider()

    # Dendrogram
    st.subheader("4️. Hierarchical Clustering - Ward Linkage Dendrogram")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/dendrogram.png", caption="Ward Linkage Dendrogram",
                 use_container_width=True)
    with col2:
        st.markdown("""
            **From the Dendrogram:**

            The dendrogram shows how individual data points were merged.
            The **y-axis (Ward Distance)** shows the dissimilarity at which two clusters merged
            a large jump indicates a natural cluster boundary.

            **Key observations:**
            - There is a clear large gap at ~**Ward Distance 40–70**, suggesting **2 main natural clusters** — consistent with the silhouette result.
            - A secondary split at ~distance 30 shows a **3-cluster structure**, aligning with the k=3 KMeans result.
            - The orange group (left) represents a close, well-separated cluster likely corresponding to peak summer days.
            - The green group (right) is larger and more spread, representing the cooler months.

            **Comparison to KMeans:** Both methods agree on 2–3 natural clusters. Hierarchical clustering
            reveals the merging path more clearly, while KMeans provides cleaner boundaries.
            """)

    st.divider()

    # KMeans vs Hierarchical
    st.subheader("5️. KMeans vs Hierarchical (k=2)")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/kmeans_vs_hierarchical.png", caption="KMeans k=2 vs Hierarchical (Ward) k=2",
                 use_container_width=True)
    with col2:
        st.markdown("""
            **Comparison at k=2:**

            Both methods produce nearly identical cluster assignments on this dataset:
            - The left cluster (PC1 < 0) groups cool season days ie winter and some autumn/spring.
            - The right cluster (PC1 > 0) groups warm season days ie summer and late spring.

            **Key differences:**
            - KMeans uses **centroid positions** (shown in ✕ markers) to define boundaries that are clean and geometric.
            - Hierarchical uses **Ward linkage merging** ie slightly more flexible at the boundary edges.
            - A small number of transitional days near PC1 ≈ 0 are assigned differently between the two methods.

            Therefore, the **high agreement confirms the robustness** of the seasonal clustering signal in the data.
            """)

    st.divider()

    # DBSCAN
    st.subheader("6️. DBSCAN - Density-Based Clustering")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/dbscan_kdist.png", caption="k-Distance Plot — used to choose eps value",
                 use_container_width=True)
    with col2:
        st.image("images/dbscan_clusters.png", caption="DBSCAN Results (eps=0.5, min_samples=10)",
                 use_container_width=True)

    st.markdown("""
        **Choosing eps using the k-Distance Plot:**

        The k-distance plot sorts all points by their distance to the 10th nearest neighbour.
        The elbow of this curve (where distance starts rising steeply) gives a good estimate for eps.
        Here the elbow appears around 0.5, which was used as the eps parameter.

        **DBSCAN Results (eps=0.5, min_samples=10):**
        - **Cluster 0 (blue):** The largest dense region - covers the majority of normal seasonal days.
        - **Cluster 1 (teal):** A second dense region - likely transitional spring/autumn days.
        - **Cluster 2 (brown):** A smaller isolated cluster - possibly low-temperature winter outlier days.
        - **Noise (gray):** Sparse points that do not belong to any cluster - these represent **anomalous weather days**
          with unusual combinations of temperature, humidity, and soil moisture.

        **Comparison to KMeans/Hierarchical:**
        DBSCAN does not force every point into a cluster, this makes it useful for **outlier detection**.
        The noise points flagged by DBSCAN could represent days of unexpected season
        or sensor anomalies ie highly valuable information for smart irrigation decision systems.
        """)

    st.divider()

    #Conclusions
    st.subheader("Clustering Conclusions")

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Silhouette Score", "0.494", "k=2")
    col2.metric("Natural Clusters Found", "2–3", "All methods agree")
    col3.metric("DBSCAN Noise Points", "~15–20%", "Anomalous days")

    st.success("""
        **Key Takeaways from Clustering:**
        - All three methods identify **2 primary clusters** aligned with warm vs cool seasons.
        - The **seasonal temperature–ET cycle** (captured in PC1) is the dominant driver separating clusters.
        - KMeans and Hierarchical produce very similar results, validating cluster numbers.
        - DBSCAN uniquely identifies **anomalous weather days as noise** which is useful for detecting unusual irrigation demand events.
        - These cluster labels (warm vs cool) can serve as features or targets in downstream supervised ML models.
        """)


# TAB 5 — ARM
with tabs[5]:
    st.header("🔗 Association Rule Mining (ARM)")

    # ── (a) Overview ─────────────────────────────────────────
    st.subheader("(a) What is Association Rule Mining?")
    st.markdown("""
        **Association Rule Mining (ARM)** is an unsupervised machine learning technique that discovers
        interesting relationships between features in large datasets. It started in market-basket
        analysis ("eg: customers who buy bread also buy butter") but applies widely including agriculture,
        where it can show which combinations of climate conditions co-occur with specific soil or
        irrigation states.

        A rule takes the form **X -> Y**, meaning *"when X occurs, Y also tends to occur."*

        **Three key metrics define a rule's strength:**
        """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
            **Support**
            Fraction of all transactions (days) containing both X and Y.
            Measures how *frequently* the rule applies overall.

            Support(X→Y) = P(X ∩ Y)
            """)
    with col2:
        st.info("""
            **Confidence**
            Probability that Y occurs given X is present.
            Measures how reliable the rule is.

            Confidence(X->Y) = P(Y | X)
            """)
    with col3:
        st.info("""
            **Lift**
            How much more likely Y is given X, vs random chance.
            Lift > 1 = positive association and Lift = 1 = independent.

            Lift = Confidence / P(Y)
            """)

    st.markdown("""
        **The Apriori Algorithm** works in two stages:
        1. **Find frequent itemsets** - all combinations of items with Support ≥ min_support threshold,
           using the *Apriori property* to prune: any subset of a frequent itemset must also be frequent.
        2. **Generate rules** - from each frequent itemset, generate candidate rules and filter by min_confidence.

        This reduces the search space compared to brute force, making ARM scalable to large datasets.
        """)

    st.divider()

    #(b) Data Prep
    st.subheader("(b) Data Preparation for ARM")
    st.markdown("""
        ARM requires **unlabeled transaction data** ie a binary matrix where each row is a "transaction"
        (here, one day of weather/soil observation) and each column is an "item" (a discretized condition).

        ARM cannot use continuous values directly. All numeric features discretized into
        categorical bins using quantile-based binning:

        | Original Feature | Binned Items Created |
        |-----------------|---------------------|
        | `T2M_MAX` (temperature) | `Temp_Low`, `Temp_Mid`, `Temp_High` |
        | `PRECTOTCORR` (rainfall) | `Rain_Low`, `Rain_Mid`, `Rain_High` |
        | `RH2M` (humidity) | `Hum_Low`, `Hum_Mid`, `Hum_High` |
        | `EVPTRNS` (ET) | `ET_Low`, `ET_Mid`, `ET_High` |
        | `v_soilm_0_10cm` (soil moisture) | `SoilM_Low`, `SoilM_Mid`, `SoilM_High` |
        | Date → Season | `Winter`, `Spring`, `Summer`, `Autumn` |

        Each day becomes a transaction like: `['Summer', 'Temp_High', 'Rain_Low', 'ET_High', 'SoilM_Low', 'Hum_Low']`

        **Thresholds used:**
        - `min_support = 0.05` (rule must apply to at least 5% of all days)
        - `min_confidence = 0.40` (rule must be correct at least 40% of the time)
        """)

    st.divider()

    #(c) Results
    st.subheader("(c) ARM Results")

    # Top 15 by Support
    st.markdown("#### Top 15 Rules by Support")
    st.markdown(
        "Support tells us how often a rule shows up across all days in our dataset — think of it as how common a pattern is.")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_support.png", caption="Top 15 Association Rules by Support", use_container_width=True)
    with col2:
        st.markdown("""
            **Interpretation**

            The most common pattern by far is **Rain_Low appearing alongside ET_Low or ET_High** —
            showing up in over 50% of all days. This makes sense for a dry California climate where
            low rainfall is simply the norm.

            Almost every other high-support rule also ends in **Rain_Low** as the outcome, regardless
            of whether it's a hot day, a humid day, or a day with high soil moisture. This tells us
            that **dry conditions dominate the dataset**, and that low rainfall is a baseline condition
            rather than a special event.

            The one standout rule is at the bottom ie **Rain_Low + ET_Low -> SoilM_Low** with a lift
            of 1.84, meaning dry days with low evaporation are nearly twice as likely to also have
            low soil moisture.
            """)

    st.divider()

    # Top 15 by Confidence
    st.markdown("#### Top 15 Rules by Confidence")
    st.markdown(
        "Confidence tells us how reliable a rule is - if the conditions on the left are true, how often is the outcome on the right also true?")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_confidence.png", caption="Top 15 Association Rules by Confidence",
                 use_container_width=True)
    with col2:
        st.markdown("""
            **Interpretations:**

            Every single rule in this top-15 list has a confidence of **1.0 (100%)** suggesting
            whenever those conditions occur, Rain_Low is *always* the outcome. That's as reliable
            as a rule can get.

            examples:
            - **Autumn → Rain_Low** (100% confident): Autumn days in this region are always dry.
            - **Summer + Low Humidity -> Rain_Low** (100% confident): Hot dry summer days never bring rain.
            - **High Temp + Low Soil Moisture → Rain_Low** (100% confident): When the soil is already
              dry and it's hot, there's no rainfall to save it which is a clear irrigation trigger.

            These rules are practically usable as simple **if-then irrigation rules** that a farmer
            or automated system could apply directly.
            """)

    st.divider()

    # Top 15 by Lift
    st.markdown("####Top 15 Rules by Lift")
    st.markdown(
        "Lift tells us how much stronger a relationship is compared to pure chance a lift of 3.8 means the outcome is nearly 4× more likely than random.")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_lift.png", caption="Top 15 Association Rules by Lift", use_container_width=True)
    with col2:
        st.markdown("""
            **Interpretation:**

            The highest-lift rules (lift ~3.8) involve **Spring + Moderate Humidity predicting
            Moderate Soil Moisture + High ET**. This is genuinely useful — it means that in spring,
            if humidity is moderate, you can reliably predict that the soil has some moisture but
            evaporation is also high, so irrigation may still be needed soon.

            strong patterns:
            - **Summer + Low ET + Low Humidity → High Temp + Low Soil Moisture** (lift 3.44,
              confidence 91.5%): A hot dry summer day almost always means the soil is running
              critically low — water now.
            - **High Humidity + Winter + High ET → High Soil Moisture + Low Temp** (lift 3.45,
              confidence 88.8%): Cold wet winter days keep the soil full — no irrigation needed.

            These lift-ranked rules are the most *surprising and actionable* ones — they go beyond
            the obvious and reveal non-random patterns that a simple calendar or rule-of-thumb would miss.
            """)

    st.divider()

    # Visualizations
    st.subheader("(d) Visualizations")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("images/arm_scatter.png",
                 caption="Support vs Confidence scatter plot (size & colour = Lift)",
                 use_container_width=True)
        st.markdown("""
            **Support vs Confidence Plot**

            Each point represents one association rule. The colour and size both encode **Lift** —
            darker red and larger circles indicate stronger non-random associations.

            - Most rules cluster at **low support (0.05–0.15)** but very **high confidence (0.8–1.0)**,
              meaning the rules are reliable even if they apply to a subset of days.
            - Rules with support > 0.3 tend to involve `Rain_Low` (the most common condition in this
              dry California climate) and have lift ≈ 1.
            - The most interesting rules appear in the **lower-left** —
              rare but highly specific combinations.
            """)

    with col2:
        st.image("images/arm_network.png",
                 caption="ARM Network Graph — Top 20 Rules by Lift (arrow = rule direction, colour = lift strength)",
                 use_container_width=True)
        st.markdown("""
            **Association Rule Network**

            Nodes represent itemsets (conditions) and directed edges represent rules (X → Y).
            Edge colour and thickness both encode **Lift** — darker/thicker = stronger association.

            **Key clusters visible:**
            - `Spring + Humidity_Mid` strongly connects to `SoilM_Mid + ET_High + Temp_Mid` (lift ~3.84)
            - `Summer + ET_Low + Humidity_Low` strongly connects to `Temp_High + SoilM_Low` (lift ~3.44)
            - `Humidity_High + Winter + ET_High` connects to `SoilM_High + Temp_Low` (lift ~3.45)

            The hub-like nodes (`SoilM_Mid, ET_High` and `Rain_Low` combinations) appear in many rules,
            confirming they are the most conditionally dependent variables in the dataset.
            """)

    st.divider()

# (e) Conclusions
    st.subheader("(e) ARM Conclusions")

    col1, col2, col3 = st.columns(3)
    col1.metric("Min Support Used", "0.05", "5% of days")
    col2.metric("Min Confidence Used", "0.40", "40% reliability")
    col3.metric("Top Lift Achieved", "3.84", "Spring → SoilM_Mid+ET_High")

    st.success("""
        **Key Takeaways from Association Rule Mining:**

        - **Rain_Low dominates** — nearly every high-support rule has `Rain_Low` as a consequent,
          showing the dry California climate where low rainfall is the norm (support > 0.33 for all temperature/humidity bins).

        - **Summer dryness is highly predictable** - `Summer + ET_Low + Humidity_Low -> Temp_High + SoilM_Low`
          (lift = 3.44, confidence = 91.5%) is a strong rule: low humidity summer days almost always
          mean high temperatures and critically low soil moisture.

        - **Spring is the most complex season** — `Spring + Humidity_Mid → SoilM_Mid + ET_High + Temp_Mid`
          (lift = 3.84) has the highest lift in the dataset, suggesting moderate spring humidity
          is a strong predictor of moderate soil moisture and elevated ET which is useful for pre-season irrigation planning.

        - **Winter cold means full soil** — `Humidity_High + Winter + ET_High → SoilM_High + Temp_Low`
          (lift = 3.45) confirms that high-humidity winter days are reliably associated with high soil
          moisture, meaning irrigation can be safely suspended in winter.

        - These rules can be directly implemented as **if-then irrigation decision rules** in an automated
          smart irrigation system, providing interpretable and actionable triggers without requiring a black-box model.
        """)

# ─────────────────────────────────────────────────────────────
# TAB 6 — DT / NB
# ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("🌳 Decision Trees & Naive Bayes")

# ─────────────────────────────────────────────────────────────
# TAB 7 — SVM / Ensemble
# ─────────────────────────────────────────────────────────────
with tabs[7]:
    st.header("⚙️ SVM & Ensemble Methods")

# ─────────────────────────────────────────────────────────────
# TAB 8 — Regression
# ─────────────────────────────────────────────────────────────
with tabs[8]:
    st.header("📈 Regression")

# ─────────────────────────────────────────────────────────────
# TAB 9 — Conclusions
# ─────────────────────────────────────────────────────────────
with tabs[9]:
    st.header("✅ Conclusions")

# ─────────────────────────────────────────────────────────────
# TAB 10 — About Me
# ─────────────────────────────────────────────────────────────
with tabs[10]:
    st.header("👤 About Me")
    st.markdown("""
    **Name:** Shivani Atul Bhinge
    **Project:** Smart Water Usage Prediction
    """)
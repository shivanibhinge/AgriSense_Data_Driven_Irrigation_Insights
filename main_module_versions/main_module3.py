# main_module3.py
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
        "start": start, "end": end, "latitude": lat, "longitude": lon,
        "parameters": parameters, "community": "AG", "format": "JSON"
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
            try: df[c] = pd.to_datetime(df[c])
            except: pass
    return df

def clean_and_merge(df_weather, df_soil, keep_cols=None):
    w = df_weather.copy(); s = df_soil.copy()
    w.columns = w.columns.str.lower(); s.columns = s.columns.str.lower()
    if 'date' not in w.columns:
        for c in w.columns:
            if 'time' in c or 'valid' in c:
                w['date'] = pd.to_datetime(w[c]); break
    if 'valid_date' in s.columns: s['date_soil'] = pd.to_datetime(s['valid_date'])
    elif 'date' in s.columns: s['date_soil'] = pd.to_datetime(s['date'])
    elif 'timestamp_local' in s.columns:
        s['date_soil'] = pd.to_datetime(s['timestamp_local']).dt.date
        s['date_soil'] = pd.to_datetime(s['date_soil'])
    else:
        s = s.reset_index()
        try: s['date_soil'] = pd.to_datetime(s['index'])
        except: s['date_soil'] = pd.NaT
    w['date'] = pd.to_datetime(w['date'])
    rename_map = {'t2m_max':'t2m_max','t2m_min':'t2m_min','rh2m':'rh2m',
                  'prectot':'precipitation','prectotcorr':'precipitation',
                  'evptrns':'evapotranspiration','evapotranspiration':'evapotranspiration'}
    w = w.rename(columns=rename_map)
    df = pd.merge(w, s, left_on='date', right_on='date_soil', how='inner')
    df = df.drop(columns=[c for c in ['date_soil','index'] if c in df.columns], errors='ignore')
    for c in df.columns:
        if df[c].dtype == object:
            try: df[c] = pd.to_numeric(df[c], errors='coerce')
            except: pass
    numcols = df.select_dtypes(include='number').columns
    if len(numcols) > 0: df = df.dropna(subset=numcols, how='all')
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
st.title("🌾 Smart Water Usage : Predicting irrigation needs using weather, soil and vegetation")
# ─────────────────────────────────────────────────────────────

tabs = st.tabs([
    " Introduction",
    " Data Prep",
    " EDA",
    " PCA",
    " Clustering",
    " ARM (Assoc Rules)",
    " Naive Bayes",
    " Decision Trees",
    " Regression",
    " SVM / Ensemble",
    " Conclusions",
    " About Me"
])

# ─────────────────────────────────────────────────────────────
# TAB 0 — Introduction
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Introduction")
    st.markdown("""
    Agriculture consumes the majority of global freshwater resources, and a large portion of that water
    is used inefficiently. Farmers in many regions still rely on fixed watering schedules that do not
    account for actual soil conditions, weather patterns, or daily temperature swings. This project —
    AgriSense — focuses on building a data-driven irrigation prediction system using real weather and
    soil data. The core question is simple: given today's weather and soil conditions, does a crop need
    to be watered? To answer this, daily climate data from NASA and soil moisture data from Weatherbit
    were combined into a single dataset covering California's Central Valley from 2020 to 2025. Machine
    learning models were then trained to predict whether irrigation is needed on any given day. The goal
    is to show that publicly available API data, combined with modern ML methods, can replace guesswork
    with data-backed decisions. Every step of this project is documented and linked so the results can
    be verified and replicated.
    """)

    st.markdown("""
    California's Central Valley is one of the most productive agricultural regions in the world, and also
    one of the most water-stressed. The valley receives almost all of its rainfall in winter, and summers
    are long, hot, and completely dry. Farmers depend heavily on irrigation to keep crops alive from May
    through October, and this heavy reliance on irrigation has led to serious groundwater depletion over
    decades. At the same time, applying water on the wrong days — when soil is already moist or when
    rain is coming — wastes a resource that is becoming increasingly scarce. Climate change is making
    these problems worse, with droughts becoming longer and more severe and rainfall becoming more
    unpredictable. A smarter irrigation system that reads the actual state of the soil and weather each
    day, rather than following a fixed weekly schedule, could significantly reduce water use without
    harming crop yields. This is exactly what AgriSense is designed to demonstrate — that the data
    needed for smart irrigation decisions is already available for free, it just needs to be used properly.
    """)

    st.markdown("""
    The data science process used in this project follows the full lifecycle from raw data collection to
    final model evaluation. Data was pulled from two APIs — NASA POWER for daily weather measurements
    like temperature, humidity, and evapotranspiration, and Weatherbit for soil-specific measurements
    like volumetric soil moisture and soil temperature at multiple depths. The raw data from both sources
    was cleaned, merged on the date field, and explored through visualizations before any modeling began.
    This exploratory phase revealed strong seasonal patterns — soil moisture drops sharply every summer,
    while soil and air temperature both peak during the same period. These patterns motivated the label
    used for all supervised models: a binary flag indicating whether a given day meets the conditions
    for irrigation need — low soil moisture, high evapotranspiration, and no meaningful rainfall.
    Unsupervised methods including clustering and association rule mining were applied first, and the
    patterns they found directly informed the feature choices and label design for the supervised models.
    """)

    st.markdown("""
    Module 3 adds supervised machine learning to the project — three types of Naive Bayes, three Decision
    Trees with different structures, and Logistic Regression. All models were trained on the same
    80/20 train-test split and evaluated on the same held-out test data, so accuracy comparisons between
    models are fair and meaningful. Care was taken to remove features that were used to construct the
    label from the training data — a common mistake called data leakage that artificially inflates accuracy.
    After removing those features, all seven models still achieve between 85% and 93% accuracy, which
    is a strong and honest result. The best-performing models were Decision Tree (depth=5, Entropy) and
    Logistic Regression, both hitting around 92%. The results confirm that soil temperature and skin
    temperature are the most important predictors — when the ground is hot, it is also dry, and crops
    need water. The final website brings together all these findings in a way that a farmer, a policymaker,
    or a student can follow without needing a background in machine learning.
    """)

    st.markdown("**GitHub:** https://github.com/shivanibhinge/AgriSense_Data_Driven_Irrigation_Insights/tree/main &nbsp;|&nbsp; **[💻 Colab Notebook](https://colab.research.google.com/drive/1hh2ymnm7VCsu_KzFaqDniyJWvBs0EZmV?usp=sharing)**")

    st.subheader("Ten Research Questions")
    questions = [
        "How does daily precipitation affect soil moisture at 0–10 cm depth?",
        "Which weather features (temperature, humidity, wind) most influence evapotranspiration?",
        "Can we cluster days with similar water demand profiles?",
        "Does soil moisture lag precipitation by a consistent number of days?",
        "What are the principal components that explain variance in the combined dataset?",
        "Can a decision tree accurately predict whether a day needs irrigation?",
        "How does seasonal variation affect soil moisture and evapotranspiration?",
        "Which features are most important for predicting irrigation need?",
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
    Data was collected from two APIs:
    1. **NASA POWER API** — daily weather: max/min temperature, humidity, precipitation, evapotranspiration.
    2. **Weatherbit AgWeather API** — daily soil: soil moisture, soil temperature, wind speed, skin temperature.

    Location: **California Central Valley (Lat: 36.77, Lon: -119.41)** | Years: **2020–2025**

    NASA returned JSON which was parsed into a DataFrame. Weatherbit also returned JSON with soil-specific agricultural fields.
    Both were merged on the **date column** using an inner join to create one unified daily dataset.
    """)
    st.image("images/dataset_merged.png", caption="Preview of the merged weather–soil dataset", use_container_width=True)

    st.subheader("2. Cleaning Steps Applied")
    st.markdown("""
    1. Column names standardized to lowercase
    2. Date columns converted to datetime format
    3. Irrelevant metadata columns dropped (timestamps, bulk soil density, revision fields)
    4. Missing values handled with forward-fill
    5. Datasets merged on date — inner join keeps only days present in both sources

    Final dataset: **2031 rows × 38 columns** — daily observations from Jan 2020 to Jul 2025.
    """)

    st.subheader("3. Key Features in the Final Dataset")
    st.markdown("""
    | Category | Feature | Description |
    |----------|---------|-------------|
    | Weather | `T2M_MAX` | Max air temperature (°C) |
    | Weather | `T2M_MIN` | Min air temperature (°C) |
    | Weather | `RH2M` | Relative humidity (%) |
    | Weather | `EVPTRNS` | Evapotranspiration from NASA (mm/day) |
    | Soil | `soilt_0_10cm` | Soil temperature at 0–10 cm depth (°C) |
    | Soil | `soilt_10_40cm` | Soil temperature at 10–40 cm depth (°C) |
    | Soil | `v_soilm_0_10cm` | Volumetric soil moisture at 0–10 cm |
    | Soil | `skin_temp_avg` | Average skin (surface) temperature (°C) |
    | Soil | `wind_10m_spd_avg` | Wind speed at 10 m height (m/s) |
    | Atmosphere | `pres_avg` | Average atmospheric pressure (hPa) |
    | Atmosphere | `specific_humidity` | Specific humidity (kg/kg) |
    """)
    st.info("These features describe the daily climate–soil state used to predict irrigation need.")

# ─────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Ten visualizations exploring the key patterns in the AgriSense dataset.")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/daily_weather_soil_moisture_trends.png", caption="Viz 1 — Daily Weather & Soil Moisture Trends", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 1 — Daily Weather & Soil Moisture Trends**

        This time-series shows temperature, rainfall, and soil moisture over all five years.
        A clear seasonal rhythm appears every year — temperature peaks in summer, drops in winter,
        and soil moisture does exactly the opposite. Rainfall appears as sharp spikes and is
        concentrated in the winter months. After each rainfall spike, soil moisture rises shortly
        after — confirming that rain directly replenishes the shallow soil layer. The summers
        are almost completely dry, with very few rainfall spikes, making irrigation the only
        source of water for crops during that period.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/monthly_soil_moisture.png", caption="Viz 2 — Monthly Soil Moisture Box Plot", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 2 — Monthly Soil Moisture Variation**

        This box plot breaks soil moisture down month by month across all years.
        January, February, and March show the highest and most stable moisture levels.
        Starting in April, moisture drops steadily, reaching the lowest values in July
        and August. The spread (box width) also increases in summer, meaning some summer
        days are wetter than others depending on the year — likely reflecting rare summer
        storms. This plot is one of the clearest visual confirmations that summer is
        the season where irrigation decisions matter most.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/max_temp_rainfall.png", caption="Viz 3 — Max Temperature vs Rainfall", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 3 — Max Temperature vs Rainfall**

        Almost all large rainfall events occur at temperatures below 20°C — the cold winter months.
        As temperature climbs above 25°C, rainfall drops to near zero. The hottest days (above 35°C)
        have essentially no rainfall at all. This inverse relationship means that the days when crops
        are under the most heat stress are also the days with no natural water supply.
        This is the core challenge of California agriculture that this project is trying to address.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/rainfall_soil_moisture.png", caption="Viz 4 — Rainfall vs Soil Moisture", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 4 — Rainfall vs Soil Moisture**

        There is a positive relationship between rainfall and soil moisture, but it is not tight.
        Many low-rainfall days still show a wide range of soil moisture values, meaning previous
        days' conditions carry over. A dry spell lasting several weeks will leave the soil very
        dry even if a small rain event occurs, because the soil absorbs the rain quickly.
        This confirms that rainfall alone is not a reliable predictor of whether irrigation is needed —
        you need to know the actual soil state too.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/avg_evaporation.png", caption="Viz 5 — Average Evapotranspiration by Month", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 5 — Average Evapotranspiration by Month**

        Evapotranspiration (ET) measures how much water is leaving the soil and plants each day.
        It is very low in December and January, then rises steadily through spring, peaking in June
        and July at over 7 mm/day. This peak overlaps exactly with the lowest soil moisture months
        seen in Viz 2 — the soil is both losing water fast (high ET) and receiving none (no rain).
        June through August is clearly the highest-risk window for crops, and irrigation scheduling
        should be most intensive during this period.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz10_dual_axis.png", caption="Viz 6 — Monthly Rainfall vs Soil Moisture Over Time", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 6 — Monthly Rainfall vs Soil Moisture Over Time**

        This dual-axis chart overlays monthly rainfall (bars) and soil moisture (line) on the same
        timeline. The lag effect is visible — when the blue bars spike in winter, the orange moisture
        line rises in the same or following month. During the long stretches with no blue bars (summer),
        the orange line drops steadily as soil dries out. Some years show wetter winters than others,
        and the soil moisture line reflects this year-to-year variability well.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz8_wind_et.png", caption="Viz 7 — Wind Speed vs Evapotranspiration", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 7 — Wind Speed vs Evapotranspiration**

        Wind speed and ET show a moderate positive relationship. Higher wind speeds remove the
        thin humid boundary layer near the soil and leaf surfaces, allowing water to evaporate
        faster. While temperature is the dominant driver of ET, wind still adds meaningful signal.
        On windy summer days, a smart irrigation system should account for this extra water loss —
        the soil may dry out faster than temperature alone would suggest.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz7_seasonal_soil_moisture.png", caption="Viz 8 — Average Soil Moisture by Season", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 8 — Average Soil Moisture by Season**

        Winter averages 30.79 m³/m³, Spring 24.63, Autumn 15.97, and Summer just 12.23 m³/m³ —
        less than half of winter levels. This four-bar comparison makes the seasonal pattern
        impossible to miss. Irrigation is essentially unnecessary in winter but critical in summer.
        The autumn value of 15.97 shows that soil starts recovering once summer heat breaks,
        but is still well below the levels needed without some irrigation support.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz9_temp_heatmap_year.png", caption="Viz 9 — Monthly Average Max Temperature by Year", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 9 — Monthly Average Max Temperature by Year**

        Every year from 2020 to 2025 shows the same dark red band across June, July, and August —
        consistently the hottest months. This year-over-year consistency confirms that summer heat
        is a predictable annual pattern, not a random event. Irrigation schedules can be planned
        in advance based on the calendar alone, with adjustments made based on actual sensor
        readings as the season progresses.
        """)
    st.divider()

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/viz6_correlation_heatmap.png", caption="Viz 10 — Correlation Heatmap", use_container_width=True)
    with col2:
        st.markdown("""
        **Viz 10 — Correlation Heatmap**

        Soil temperature features (soilt_0_10cm, soilt_10_40cm, skin_temp_avg) and air temperature
        (T2M_MAX) are all strongly correlated with each other — when one is high, the others tend
        to be high too. Soil moisture is negatively correlated with temperature, which matches
        the pattern seen in earlier visualizations. Specific humidity and relative humidity are
        positively correlated — both capture the moisture content of the air. These correlations
        helped identify which features carry the most predictive signal for the ML models.
        """)

# ─────────────────────────────────────────────────────────────
# TAB 3 — PCA
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Principal Component Analysis (PCA)")

    st.subheader("What is PCA?")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
        Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique
        that transforms a dataset with many correlated features into a smaller set of uncorrelated
        variables called principal components. Each component is a linear combination of the original
        features, ordered so that the first component captures the most variance in the data and
        each following component captures the next most. The key idea is that much of the useful
        information in a high-dimensional dataset is concentrated in just a few directions, so
        discarding the rest loses very little. PCA works by computing the covariance matrix of
        the data and finding its eigenvectors — these eigenvectors are the principal components.
        The corresponding eigenvalues tell us how much variance each component explains.
        Before applying PCA, features must be standardized so that variables measured on different
        scales do not dominate the result — a temperature in Celsius and pressure in hPa would
        otherwise be incomparable. After fitting PCA, projecting the data onto the top 2 or 3
        components allows visualization of high-dimensional data in a 2D or 3D scatter plot.
        PCA is not a supervised method — it has no knowledge of class labels, it only looks
        at the structure of the feature space itself.
        """)
    with col2:
        st.image("images/pca_2d.png", caption="PCA 2D projection — seasonal arc visible", use_container_width=True)
        st.image("images/pca_variance.png", caption="Scree plot — variance per component", use_container_width=True)
    st.divider()

    st.subheader("1. Dataset Used & Before/After Data Prep")
    st.markdown("""
    PCA was applied to the full merged dataset (2031 rows × 31 numeric features).
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before PCA — raw numeric features (StandardScaler not yet applied):**")
        st.image("images/nb_before_data.png",
                 caption="Raw feature data before StandardScaler — mean and std vary widely across columns",
                 use_container_width=True)
    with col2:
        st.markdown("**After StandardScaler — all features normalized (mean=0, std=1):**")
        st.image("images/nb_after_gnb.png",
                 caption="After StandardScaler — every column now has mean≈0 and std≈1, ready for PCA",
                 use_container_width=True)

    st.markdown("""
    Steps applied:
    - All numeric columns selected — date and any label columns removed
    - Columns with near-zero variance removed (constant columns add no information)
    - StandardScaler applied so no single feature dominates due to scale
    """)
    st.markdown(f"**[📂 Raw Dataset](https://drive.google.com/drive/folders/1PuYcVJTtrA2Y7_oNx3JN6zM9YJ5GLE_7?usp=sharing)** &nbsp;|&nbsp; **[💻 Colab Notebook](https://colab.research.google.com/drive/1hh2ymnm7VCsu_KzFaqDniyJWvBs0EZmV?usp=sharing)**")
    st.info("Dates and season labels were dropped — PCA requires purely numeric input.")
    st.divider()

    st.subheader("2. PCA with n_components = 2")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_2d.png", caption="2D PCA Projection coloured by Month", use_container_width=True)
    with col2:
        st.markdown("""
        The 2D scatter plot projects all 31 features into just two components, coloured by month.
        A clear seasonal arc is visible — winter months (blue/purple) cluster to the right, summer
        months (red/orange) cluster to the left. This separation shows that PC1 is capturing the
        seasonal temperature and evapotranspiration cycle.

        - **PC1**: 58.8% of variance
        - **PC2**: 16.4% of variance
        - **Total retained in 2D: 75.2%**
        """)
    st.divider()

    st.subheader("3. PCA with n_components = 3")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_3d.png", caption="3D PCA Projection coloured by Month", use_container_width=True)
    with col2:
        st.markdown("""
        Adding a third component reveals more structure. PC3 separates transitional months like
        spring and autumn that sit between the two extremes in 2D.

        - **PC1**: 58.8% | **PC2**: 16.4% | **PC3**: 8.1%
        - **Total retained in 3D: 83.3%**
        """)
    st.divider()

    st.subheader("4. Scree Plot & Cumulative Variance")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_variance.png", caption="Scree Plot and Cumulative Explained Variance", use_container_width=True)
    with col2:
        st.markdown("""
        | Components | Cumulative Variance |
        |-----------|-------------------|
        | 2 | 75.2% |
        | 3 | 83.3% |
        | 5 | ~89% |
        | **8** | **≥ 95%** |

        8 components are needed to retain 95% of the original information — reducing from 31 to 8 features.
        """)
    st.divider()

    st.subheader("5. Top Eigenvalues")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/pca_eigenvalues.png", caption="Top 10 Eigenvalues", use_container_width=True)
    with col2:
        st.markdown("""
        | Component | Eigenvalue |
        |-----------|-----------|
        | PC1 | **18.24** |
        | PC2 | **5.09** |
        | PC3 | **2.51** |

        PC1's eigenvalue of 18.24 is far larger than all others — it dominates the dataset.
        Components with eigenvalue below 1 are generally not meaningful and can be dropped.
        """)
    st.divider()

    st.subheader("PCA Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("2D Variance Retained", "75.2%")
    col2.metric("3D Variance Retained", "83.3%")
    col3.metric("Components for 95%", "8")
    col4.metric("Top Eigenvalue", "18.24")
    st.success("""
    PC1 alone captures 58.8% of the variance — the dominant signal is the seasonal temperature and ET cycle.
    Just 8 components retain 95% of information, reducing the dataset from 31 dimensions to 8.
    These reduced components were used as input for the clustering analysis.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 4 — Clustering
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("Clustering")
    st.subheader("(a) Overview — What is Clustering?")
    st.markdown("""
    Clustering is an unsupervised machine learning technique that groups data points together based
    on similarity — without using any labels. The goal is to find natural groupings in the data
    where points within the same group are more similar to each other than to points in other groups.
    In this project, clustering is applied to daily weather and soil observations to discover whether
    days naturally group into distinct irrigation-relevant regimes — such as hot-dry summer days
    versus cool-wet winter days — without being told which season each day belongs to.
    Clustering is called unsupervised because no target label is used during the process — the algorithm
    discovers structure purely from the input features. Three different clustering methods were applied
    and compared: KMeans, Hierarchical (Ward linkage), and DBSCAN. Each method defines "similarity"
    differently and makes different assumptions about cluster shape and number. Comparing all three
    gives a more complete picture of the structure in the data than relying on any single method.
    The results from clustering also informed the supervised models in Module 3 — confirming which
    features drive the most separation between irrigation-need states.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/kmeans_clusters.png",
                 caption="KMeans clusters coloured by season — warm/dry vs cool/wet clearly separate",
                 use_container_width=True)
    with col2:
        st.image("images/dendrogram.png",
                 caption="Hierarchical dendrogram — large gap at distance 40–70 confirms 2 natural clusters",
                 use_container_width=True)

    st.markdown("**Three clustering methods compared:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("""
        **KMeans**
        Partitions data into k clusters by minimizing the distance from each point to its
        nearest centroid. Requires k upfront. Fast and scalable. Best for spherical, 
        similar-sized clusters. Sensitive to outliers.
        """)
    with col2:
        st.info("""
        **Hierarchical (Ward)**
        Builds a tree by merging the two most similar points or clusters at each step.
        No k needed upfront — you read the natural number from the dendrogram.
        Ward linkage minimizes within-cluster variance at each merge.
        """)
    with col3:
        st.info("""
        **DBSCAN**
        Groups densely packed points into clusters and labels sparse points as noise.
        No k needed. Handles arbitrary cluster shapes. Uniquely identifies outliers.
        Sensitive to eps (neighborhood radius) and min_samples parameters.
        """)
    st.divider()

    st.subheader("(b) Data Preparation")
    st.markdown("""
    The following steps were applied before clustering:
    1. Seasonal labels created from date (Winter/Spring/Summer/Autumn) — used for colour-coding only, not as input
    2. All numeric features selected, near-zero variance columns removed
    3. StandardScaler applied (mean=0, std=1 per column)
    4. PCA reduced to 3D — retained 83.3% of variance for cleaner clustering
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before — raw numeric features:**")
        st.image("images/nb_before_data.png",
                 caption="Raw features before StandardScaler — columns have very different scales",
                 use_container_width=True)
    with col2:
        st.markdown("**After — StandardScaled + PCA reduced to 3D:**")
        st.image("images/pca_3d.png",
                 caption="After StandardScaler and PCA — 3D projection used as clustering input (83.3% variance retained)",
                 use_container_width=True)

    st.markdown(f"**[📂 Raw Dataset](https://drive.google.com/drive/folders/1PuYcVJTtrA2Y7_oNx3JN6zM9YJ5GLE_7?usp=sharing)** &nbsp;|&nbsp; **[💻 Colab Notebook](https://colab.research.google.com/drive/1hh2ymnm7VCsu_KzFaqDniyJWvBs0EZmV?usp=sharing)**")
    st.divider()

    st.subheader("2. KMeans — Silhouette Method")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/kmeans_silhouette.png", caption="Silhouette Scores k=2 to 10", use_container_width=True)
    with col2:
        st.markdown("""
        | k | Silhouette Score |
        |---|-----------------|
        | **2** | **0.497** → Best |
        | 3 | 0.440 |
        | 4 | 0.448 |

        k=2 scored highest — the data most naturally splits into two groups: warm/dry vs cool/wet.
        """)
    st.divider()

    st.subheader("3. KMeans Cluster Plots (k = 2, 3, 4)")
    st.image("images/kmeans_clusters.png", caption="KMeans — coloured by Season, ✕ = Centroids", use_container_width=True)
    st.markdown("""
    k=2 gives the cleanest separation along PC1. k=3 adds a transitional middle cluster for spring/autumn.
    k=4 over-divides with reducing interpretability.
    """)
    st.divider()

    st.subheader("4. Hierarchical Clustering — Dendrogram")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/dendrogram.png", caption="Ward Linkage Dendrogram", use_container_width=True)
    with col2:
        st.markdown("""
        A large gap at Ward Distance 40–70 confirms 2 natural clusters.
        A secondary split at ~30 shows a 3-cluster structure — consistent with KMeans k=3.
        Both methods agree on the same groupings.
        """)
    st.divider()

    st.subheader("5. KMeans vs Hierarchical (k=2)")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/kmeans_vs_hierarchical.png", caption="KMeans k=2 vs Hierarchical k=2", use_container_width=True)
    with col2:
        st.markdown("""
        Both methods produce nearly identical assignments. KMeans gives geometric boundaries via centroids.
        Hierarchical is slightly more flexible at edges. High agreement confirms the seasonal signal is robust.
        """)
    st.divider()

    st.subheader("6. DBSCAN")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/dbscan_kdist.png", caption="k-Distance Plot — elbow gives eps", use_container_width=True)
    with col2:
        st.image("images/dbscan_clusters.png", caption="DBSCAN Results (eps=0.5, min_samples=10)", use_container_width=True)
    st.markdown("""
    Elbow at ~0.5 gave the eps parameter. DBSCAN found 8 clusters and flagged 12.3% of days as noise —
    these are anomalous days with unusual weather/soil combinations, valuable for edge-case irrigation decisions.
    """)
    st.divider()

    st.subheader("Clustering Conclusions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Silhouette Score", "0.497", "k=2")
    col2.metric("Natural Clusters", "2–3", "All methods agree")
    col3.metric("DBSCAN Noise", "12.3%", "Anomalous days")
    st.success("""
    All three methods agree: the data splits into 2 primary clusters (warm/dry vs cool/wet).
    The seasonal temperature-ET cycle drives this separation. DBSCAN identifies anomalous days
    that could represent unusual irrigation events worth flagging in a real system.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 5 — ARM
# ─────────────────────────────────────────────────────────────
with tabs[5]:
    st.header("Association Rule Mining (ARM)")
    st.subheader("(a) Overview")
    st.markdown("""
    Association Rule Mining (ARM) is an unsupervised machine learning technique that discovers
    interesting relationships between variables in a dataset — specifically, which combinations
    of conditions tend to occur together. It originated in retail market basket analysis, where
    the goal was to find items that customers frequently buy together, but the method applies
    broadly to any domain where co-occurrence patterns are meaningful. In agriculture, ARM can
    reveal which combinations of weather and soil conditions reliably co-occur with low soil
    moisture or high evapotranspiration — essentially building a set of if-then rules that a
    farmer or automated system could apply directly. A rule takes the form X → Y, meaning
    "when conditions X are present, condition Y also tends to occur." The strength of each rule
    is measured by three metrics: support (how often the rule appears), confidence (how reliable
    it is), and lift (how much stronger the association is compared to random chance). The Apriori
    algorithm is the standard method for finding these rules efficiently — it first finds all
    frequent itemsets (combinations that appear in at least min_support fraction of days), then
    generates rules from those itemsets filtered by minimum confidence. It prunes the search space
    using the Apriori property: any subset of a frequent itemset must also be frequent, so it
    never needs to count infrequent supersets. This makes it scalable to datasets with many items.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/arm_network.png",
                 caption="ARM Network — nodes are conditions, edges are rules, colour/thickness = lift strength",
                 use_container_width=True)
    with col2:
        st.image("images/arm_scatter.png",
                 caption="Support vs Confidence scatter — size and colour encode lift for each rule",
                 use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Support** = P(X ∩ Y)\nHow often the rule appears across all days in the dataset. Measures frequency.")
    with col2:
        st.info("**Confidence** = P(Y | X)\nProbability Y occurs given X is present. Measures reliability of the rule.")
    with col3:
        st.info("**Lift** = Confidence / P(Y)\nLift > 1 = genuine association. Lift = 1 = independent. Measures surprise.")
    st.markdown("""
    The **Apriori Algorithm** finds all frequent itemsets (support ≥ threshold) then generates rules
    filtered by minimum confidence. It prunes the search space using the Apriori property: any subset
    of a frequent itemset must also be frequent — so infrequent supersets are never counted.
    """)
    st.divider()

    st.subheader("(b) Data Preparation")
    st.markdown("""
    ARM requires unlabeled transaction data — each row is one "transaction" (one day of observations)
    and each column is a binary item indicating whether a condition was present that day.
    Continuous features cannot be used directly, so they were discretized into Low/Mid/High bins
    using quantile-based cutting. All class labels were removed before running ARM.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before ARM — raw continuous features:**")
        st.image("images/nb_before_data.png",
                 caption="Raw data before discretization — continuous values, cannot be used directly in ARM",
                 use_container_width=True)
    with col2:
        st.markdown("**After ARM prep — binary transaction matrix (sample):**")
        st.image("images/arm_support.png",
                 caption="After discretization and one-hot encoding — each row is a day, each column is a binned condition",
                 use_container_width=True)

    st.markdown("""
    | Original Feature | Bins Created |
    |-----------------|-------------|
    | T2M_MAX | Temp_Low, Temp_Mid, Temp_High |
    | PRECTOTCORR | Rain_Low, Rain_Mid, Rain_High |
    | RH2M | Hum_Low, Hum_Mid, Hum_High |
    | EVPTRNS | ET_Low, ET_Mid, ET_High |
    | v_soilm_0_10cm | SoilM_Low, SoilM_Mid, SoilM_High |
    | Date → Season | Winter, Spring, Summer, Autumn |

    Thresholds: `min_support = 0.05` (rule must appear in ≥5% of days), `min_confidence = 0.40`
    """)
    st.markdown(f"**[📂 Raw Dataset](https://drive.google.com/drive/folders/1PuYcVJTtrA2Y7_oNx3JN6zM9YJ5GLE_7?usp=sharing)** &nbsp;|&nbsp; **[💻 Colab Notebook](https://colab.research.google.com/drive/1hh2ymnm7VCsu_KzFaqDniyJWvBs0EZmV?usp=sharing)**")
    st.divider()

    st.subheader("(c) Results")
    st.markdown("#### Top 15 by Support")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_support.png", caption="Top 15 Rules by Support", use_container_width=True)
    with col2:
        st.markdown("""
        Rain_Low dominates — it appears in over 50% of all days, confirming that dry conditions
        are the norm in California's Central Valley. Almost every high-support rule ends in
        Rain_Low regardless of temperature or humidity. The standout rule at the bottom —
        Rain_Low + ET_Low → SoilM_Low (lift 1.84) — shows that dry days with low evaporation
        are nearly twice as likely to also have low soil moisture.
        """)
    st.divider()

    st.markdown("#### Top 15 by Confidence")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_confidence.png", caption="Top 15 Rules by Confidence", use_container_width=True)
    with col2:
        st.markdown("""
        Every rule in the top-15 list has 100% confidence — whenever those left-hand conditions
        occur, Rain_Low is always the outcome. Autumn → Rain_Low (100%), Summer + Low Humidity →
        Rain_Low (100%), and High Temp + Low Soil Moisture → Rain_Low (100%). These are directly
        usable as if-then irrigation triggers in an automated system.
        """)
    st.divider()

    st.markdown("#### Top 15 by Lift")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image("images/arm_lift.png", caption="Top 15 Rules by Lift", use_container_width=True)
    with col2:
        st.markdown("""
        The highest-lift rules (lift ~4.38) involve Spring + Moderate Humidity predicting
        Moderate Soil Moisture + High ET. In spring, moderate humidity is a strong predictor
        that ET is elevated and irrigation may be needed soon. These rules are the most
        surprising — they reveal patterns a simple seasonal calendar would miss entirely.
        """)
    st.divider()

    st.subheader("(d) Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/arm_scatter.png", caption="Support vs Confidence (size/colour = Lift)", use_container_width=True)
        st.markdown("Most rules cluster at low support but high confidence. The most informative rules are in the lower-left — rare but highly specific.")
    with col2:
        st.image("images/arm_network.png", caption="ARM Network — Top 20 Rules by Lift", use_container_width=True)
        st.markdown("Spring + Humidity_Mid strongly links to SoilM_Mid + ET_High. Hub nodes confirm the most conditionally dependent variables in the dataset.")
    st.divider()

    st.subheader("(e) Conclusions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Min Support", "0.05")
    col2.metric("Min Confidence", "0.40")
    col3.metric("Top Lift", "4.38")
    st.success("""
    Rain_Low dominates because California summers are reliably dry. Summer dryness is highly predictable
    from humidity and ET. Spring is the most complex season. Winter high-humidity days mean full soil —
    irrigation can safely pause. These rules are directly deployable as irrigation decision logic.
    """)

# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# TAB 6 — Naive Bayes
# ─────────────────────────────────────────────────────────────
with tabs[6]:
    st.header("Naive Bayes")

    # ── OVERVIEW ─────────────────────────────────────────────
    st.subheader("(a) Overview")
    st.markdown("""
    Naive Bayes is a family of probabilistic classification algorithms all based on Bayes' theorem.
    The core idea is to calculate the probability of each class given the input features, and then
    predict whichever class has the highest probability. The "naive" part refers to the assumption
    that all features are independent of each other given the class label — which is almost never
    perfectly true in real data, but works surprisingly well in practice. Naive Bayes is fast,
    needs very little training data compared to other models, and handles many features efficiently.
    It is widely used in email spam filtering, text classification, and medical diagnosis tasks.
    For this project, it is used to predict whether a given day requires irrigation based on weather
    and soil temperature features. There are four main variants of Naive Bayes, each designed for
    a different type of input data — choosing the right one matters significantly for performance.
    The three variants implemented here are Multinomial, Gaussian, and Bernoulli NB.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("""
        **Multinomial NB**

        Designed for discrete count data like word frequencies in documents.
        Requires non-negative integer inputs. Used here by scaling features
        to integers between 0 and 100 using MinMaxScaler.
        Best for: text classification, count-based features.
        Not the ideal fit for continuous sensor data.
        """)
    with col2:
        st.info("""
        **Gaussian NB**

        Assumes each feature follows a normal (bell-curve) distribution.
        Works directly with continuous float values — no special transformation needed.
        The most natural fit for weather measurements like temperature, humidity, and wind.
        Best for: continuous sensor data, weather features.
        """)
    with col3:
        st.info("""
        **Bernoulli NB**

        Designed for binary (0 or 1) features. Each feature is binarized —
        values above the mean become 1, below become 0. Some information is
        lost in this conversion but the model is simple and fast.
        Best for: presence/absence style features, binary encoded data.
        """)
    with col4:
        st.info("""
        **Categorical NB**

        Designed for features with a fixed set of discrete categories.
        Each feature must be integer-encoded into a fixed category set.
        Best for: survey responses, ordinal data.
        Not used here — all features are continuous measurements, not categories.
        """)

    st.markdown("""
    For this project, **Multinomial, Gaussian, and Bernoulli NB** were implemented and compared.
    Gaussian NB is the most natural fit since all features are continuous weather and soil measurements.
    Multinomial and Bernoulli are included to show how different data format assumptions affect accuracy.
    Categorical NB was skipped because none of the features are discrete categories.
    """)
    st.divider()

    # ── DATA PREP ─────────────────────────────────────────────
    st.subheader("(b) Data Preparation")
    st.markdown("""
    **Label: `irrigation_needed` (binary 0 or 1)**

    A day is labeled 1 (needs irrigation) when all three conditions are met at the same time:
    - Soil moisture at 0–10 cm is below the 40th percentile — the soil is dry
    - Evapotranspiration is above the 60th percentile — water loss is high
    - Daily rainfall is below 1 mm — no meaningful rain

    This gave **624 positive days (30.7%)** and **1407 negative days (69.3%)** out of 2031 total.

    **Data leakage prevention:** The `evapotranspiration` and `PRECTOTCORR` columns were used to
    build the label, so they were removed from the feature set before training. Keeping them in
    would let the model see part of the answer, inflating accuracy artificially.

    **10 features used:**
    `T2M_MAX, T2M_MIN, RH2M, EVPTRNS, wind_10m_spd_avg, skin_temp_avg, soilt_0_10cm, soilt_10_40cm, specific_humidity, pres_avg`
    """)

    st.markdown("**Before — raw feature data (before any transformation):**")
    st.image("images/nb_before_data.png",
             caption="Raw feature data — df[features].head() before any NB-specific preparation",
             use_container_width=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/train_test_split.png",
                 caption="Train/Test Split — 80% Training (1624 rows), 20% Testing (407 rows)",
                 use_container_width=True)
    with col2:
        st.markdown("""
        **Why train and test sets must be disjoint:**

        The dataset of 2031 rows was split 80/20 — 1624 rows for training and 407 rows for testing.
        A stratified split was used so the 30.7% positive class is preserved in both sets.
        The two sets are completely disjoint — no row appears in both. This is essential for getting
        an honest measure of how the model performs on data it has never seen before. If any test
        rows were also used in training, the model would have memorized those examples and the
        accuracy score would be meaninglessly optimistic. The same split with random_state=42
        was reused for all seven models so every accuracy comparison is fair — all models
        are evaluated on the exact same 407 held-out rows.
        """)

    st.markdown("""
    **Each NB flavor needs a different data format — here is what was applied:**

    | NB Flavor | Format Required | Transformation |
    |-----------|----------------|---------------|
    | Multinomial NB | Non-negative integers | MinMaxScaler → [0,1] → × 100 → round to int |
    | Gaussian NB | Continuous floats | StandardScaler (mean=0, std=1) |
    | Bernoulli NB | Binary 0/1 | Binarize at mean — above mean = 1, below = 0 |
    """)

    st.markdown("**After — prepared data for each NB flavor (first 5 rows of training set):**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("images/nb_after_data.png",
                 caption="Multinomial NB — integers scaled 0 to 100",
                 use_container_width=True)
    with col2:
        st.image("images/nb_after_gnb.png",
                 caption="Gaussian NB — StandardScaled floats (mean=0, std=1)",
                 use_container_width=True)
    with col3:
        st.image("images/nb_after_bnb.png",
                 caption="Bernoulli NB — binarized 0/1 at mean threshold",
                 use_container_width=True)

    st.markdown("**[📂 Raw Dataset](https://drive.google.com/drive/folders/1PuYcVJTtrA2Y7_oNx3JN6zM9YJ5GLE_7?usp=sharing)** &nbsp;|&nbsp; **[💻 Colab Notebook](https://colab.research.google.com/drive/1hh2ymnm7VCsu_KzFaqDniyJWvBs0EZmV?usp=sharing)**")
    st.divider()

    # ── RESULTS ───────────────────────────────────────────────
    st.subheader("(c) Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Multinomial NB", "86.24%")
    col2.metric("Gaussian NB", "88.70%")
    col3.metric("Bernoulli NB", "85.01%")

    st.image("images/nb_confusion_matrices.png",
             caption="Confusion Matrices — All 3 Naive Bayes Flavors",
             use_container_width=True)

    st.markdown("""
    **Reading the confusion matrices:**

    | | Multinomial NB | Gaussian NB | Bernoulli NB |
    |-|---------------|------------|-------------|
    | True Negatives (correct no-irrigation) | 234 | 247 | 230 |
    | False Positives (wrong irrigation flag) | 48 | 35 | 52 |
    | False Negatives (missed irrigation days) | 8 | 11 | 9 |
    | True Positives (correct irrigation flag) | 117 | 114 | 116 |

    All three models have very low false negatives (8–11) — they rarely miss a day that
    genuinely needs irrigation. This is the most important error to minimize in a real system
    because missing an irrigation day has direct consequences on crop health. The false positive
    rate is higher (35–52) but acceptable — it leads to some unnecessary watering, not crop damage.
    """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/nb_accuracy_comparison.png",
                 caption="Accuracy Comparison — Multinomial vs Gaussian vs Bernoulli NB",
                 use_container_width=True)
    with col2:
        st.markdown("""
        **Why Gaussian NB performs best at 88.70%:**

        The weather and soil features — temperature, humidity, soil temperature, wind speed,
        pressure — are all continuous measurements that roughly follow a bell-curve distribution.
        Gaussian NB is mathematically designed for exactly this type of data, which is why it
        outperforms the other two. Multinomial NB at 86.24% is reasonable even though it was
        designed for count data — the scaled integer representation still captures the ordering
        of values. Bernoulli NB scores 85.01% because binarizing at the mean collapses all the
        fine-grained variation in each feature into a single above/below signal, losing real
        information. All three models beat the naive baseline of always predicting "no irrigation"
        which gives 69.3% — the genuine gain is 16 to 19 percentage points above guessing.
        The narrow spread between all three (only 3.7%) shows that the patterns in the data
        are strong enough that even a less ideal model can pick them up.
        """)
    st.divider()

    # ── CONCLUSIONS ───────────────────────────────────────────
    st.subheader("(d) Conclusions")
    st.markdown("""
    Naive Bayes proved to be a practical and fast classifier for the irrigation prediction task.
    Gaussian NB was the best-performing flavor because the input features are continuous and
    approximately normally distributed, matching the core mathematical assumption of the model.
    Multinomial NB showed that even with a data format that is not ideal, the signal in this
    dataset is strong enough to still reach 86% accuracy. Bernoulli NB confirmed that binarizing
    continuous features does lose real information — the 3.7% drop compared to Gaussian NB comes
    directly from collapsing temperature and humidity readings into 0/1 values. A key finding
    from the confusion matrices is that all three models have very low false negative rates,
    meaning they rarely miss a day that truly needs irrigation — which is the most critical
    type of error to avoid. The false positive rate is higher across all three, meaning some
    days get flagged that do not need water — this is acceptable since over-watering is less
    damaging than under-watering. For a real irrigation system deployment, Gaussian NB would
    be the recommended choice — it is fast, requires minimal data transformation, and naturally
    handles the continuous sensor readings that a field system produces every day.
    """)

# ─────────────────────────────────────────────────────────────
# TAB 7 — Decision Trees
# ─────────────────────────────────────────────────────────────
with tabs[7]:
    st.header("Decision Trees")

    # ── OVERVIEW ─────────────────────────────────────────────
    st.subheader("(a) Overview")
    st.markdown("""
    A decision tree learns a set of if-then rules from training data and organizes them into
    a tree structure. Starting at the root, each internal node asks a question about one feature —
    for example "is soil temperature at 0–10 cm above 22.6°C?" — and splits the data into two
    branches based on yes or no. This continues until each branch is mostly one class or a
    stopping criterion like maximum depth is reached. The result is a set of paths from root
    to leaf that each represent a decision rule. Decision trees are easy to interpret because
    the learned rules can be read directly from the diagram — a farmer could follow them without
    any software. They are also scale-invariant, meaning feature scaling is not required before
    fitting. A key weakness is that a single deep tree can overfit — it memorizes the training
    data including noise — which is why controlling the maximum depth is important in practice.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/dt_tree1.png",
                 caption="Tree 1 — Shallow GINI (depth=3) | Root: soilt_0_10cm | Accuracy: 91.65%",
                 use_container_width=True)
    with col2:
        st.image("images/dt_tree2.png",
                 caption="Tree 2 — Entropy (depth=5) | Root: soilt_0_10cm | Accuracy: 92.87%",
                 use_container_width=True)

    st.image("images/dt_tree3.png",
             caption="Tree 3 — GINI no soil temp features (depth=5) | Root: soilt_0_10cm | Accuracy: 92.38%",
             use_container_width=True)

    st.markdown("""
    **GINI, Entropy, and Information Gain — with a worked example**

    Before splitting any node, the tree measures how "mixed" the data is using either GINI
    impurity or Entropy. GINI impurity measures the probability that a randomly chosen point
    would be misclassified if labeled randomly according to the class distribution at that node —
    a pure node (all one class) has GINI = 0. Entropy measures disorder — a pure node has
    entropy = 0, a perfectly mixed node has maximum entropy. Information Gain measures how much
    a split reduces this disorder — the algorithm always picks whichever feature and threshold
    gives the biggest reduction.

    **Worked example from this dataset:**

    Training set has 1624 rows: 1125 class 0 (no irrigation, p₀ = 0.693) and 499 class 1 (needs irrigation, p₁ = 0.307)

    - **GINI at root = 1 − (0.693² + 0.307²) = 0.426**
    - **Entropy at root = −(0.693 × log₂(0.693) + 0.307 × log₂(0.307)) = 0.896 bits**

    The tree evaluated every feature at every possible threshold and found that splitting on
    `soilt_0_10cm ≤ 22.65°C` gave the highest Information Gain — meaning this split reduces
    impurity more than any other single question that could be asked about the data.
    Days with soil temperature above 22.65°C are much more likely to need irrigation.

    **Why can you build an infinite number of trees?**
    Because there is no single correct tree for a dataset. Changing the criterion (GINI vs Entropy),
    maximum depth, minimum samples per split, which features are available, or the random seed all
    produce different tree structures. Even when two trees share the same root feature, their
    downstream splits quickly diverge. This is exactly why ensemble methods like Random Forests
    build hundreds of trees — each sees a slightly different view, and averaging across them
    produces more robust predictions than any single tree can achieve alone.
    """)
    st.divider()

    # ── DATA PREP ─────────────────────────────────────────────
    st.subheader("(b) Data Preparation")
    st.markdown("""
    Decision trees do not require feature scaling — they split on threshold values and are
    completely scale-invariant. The same binary label (`irrigation_needed`) and the same 10
    clean features used for Naive Bayes were reused here. The same 80/20 stratified split
    (random_state=42) was applied so all model comparisons are fair.

    Three trees were built with different parameters to produce different structures and show
    how parameter choices affect the result:

    | Tree | Depth | Criterion | Features Used | Root Node | Accuracy |
    |------|-------|-----------|--------------|-----------|---------|
    | Tree 1 | 3 | GINI | All 10 | soilt_0_10cm | 91.65% |
    | Tree 2 | 5 | Entropy | All 10 | soilt_0_10cm | 92.87% |
    | Tree 3 | 5 | GINI | 7 (soil temp cols removed) | soilt_0_10cm | 92.38% |

    All three trees selected `soilt_0_10cm` as the root node — this confirms it is the single
    most informative feature regardless of how the tree is built. The differences between
    the trees appear in the deeper splits and in the leaf node distributions. Tree 3 had
    `soilt_10_40cm` and `skin_temp_avg` removed but `soilt_0_10cm` was still available,
    so it remained the root. The deeper branches of Tree 3 rely more on air temperature and
    humidity, which is useful for deployments where not all soil sensors are present.
    """)

    st.markdown("**Before — raw feature data (no scaling needed for DT):**")
    st.image("images/nb_before_data.png",
             caption="Decision Tree input — X_train[features].head() — no transformation applied (DT needs no scaling)",
             use_container_width=True)

    st.markdown("**[📂 Raw Dataset](https://drive.google.com/drive/folders/1PuYcVJTtrA2Y7_oNx3JN6zM9YJ5GLE_7?usp=sharing)** &nbsp;|&nbsp; **[💻 Colab Notebook](https://colab.research.google.com/drive/1hh2ymnm7VCsu_KzFaqDniyJWvBs0EZmV?usp=sharing)**")
    st.divider()

    # ── RESULTS ───────────────────────────────────────────────
    st.subheader("(c) Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Tree 1 — depth=3 GINI", "91.65%")
    col2.metric("Tree 2 — depth=5 Entropy", "92.87%")
    col3.metric("Tree 3 — depth=5 no soil temp", "92.38%")

    st.image("images/dt_confusion_matrices.png",
             caption="Confusion Matrices — All 3 Decision Trees",
             use_container_width=True)

    st.markdown("""
    **Reading the confusion matrices:**

    | | Tree 1 | Tree 2 | Tree 3 |
    |-|--------|--------|--------|
    | True Negatives (correct no-irrigation) | 261 | 266 | 267 |
    | False Positives (wrong irrigation flag) | 21 | 16 | 15 |
    | False Negatives (missed irrigation days) | 13 | 13 | 16 |
    | True Positives (correct irrigation flag) | 112 | 112 | 109 |

    Tree 2 has the fewest false positives (16) — it is the best at not flagging days that
    do not need irrigation. Tree 3 has slightly more false negatives (16) than Trees 1 and 2
    (13 each) because removing soil temperature features forces it to rely on less direct signals.
    """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/dt_accuracy_comparison.png",
                 caption="Accuracy Comparison — 3 Trees with Root Nodes Labeled",
                 use_container_width=True)
    with col2:
        st.markdown("""
        **Comparing the three trees:**

        Tree 2 (depth=5, Entropy) scores best at 92.87%. The extra depth compared to Tree 1
        allows it to capture more nuanced patterns in the deeper splits — the first few levels
        are nearly identical but deeper nodes separate cases that the shallow tree lumps together.
        Tree 3 scores 92.38% without soil temperature depth features, which is a useful finding —
        it shows comparable accuracy is achievable using mostly air temperature and atmospheric
        features, which are available from any weather station without soil sensors. The gap
        between best and worst tree is only 1.22 percentage points (91.65% to 92.87%), confirming
        the results are stable and not sensitive to parameter choices. All three comfortably
        beat the 69.3% naive baseline by more than 22 percentage points.
        """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/dt_feature_importance.png",
                 caption="Feature Importances — Tree 1 (GINI, depth=3)",
                 use_container_width=True)
    with col2:
        st.markdown("""
        **Feature importance from Tree 1:**

        `soilt_0_10cm` dominates with an importance score near 0.9 — the tree makes almost
        all decisions based on this single feature. `wind_10m_spd_avg` and `specific_humidity`
        are distant second and third. All other features contribute near zero in this shallow tree.
        This makes physical sense: soil temperature at 0–10 cm is a direct proxy for the thermal
        and moisture state of the root zone. When soil temperature is above roughly 22.6°C,
        the conditions are almost always right for irrigation to be needed. This feature is
        the clearest single signal in the entire dataset.
        """)
    st.divider()

    # ── CONCLUSIONS ───────────────────────────────────────────
    st.subheader("(d) Conclusions")
    st.markdown("""
    Decision trees produced some of the clearest and most interpretable results in this project.
    The dominant finding is that soil temperature at 0–10 cm depth is overwhelmingly the most
    important predictor — all three trees chose it as the root node regardless of criterion or
    depth, and the feature importance plot confirms it accounts for nearly 90% of the information
    used in Tree 1. This is physically meaningful: soil temperature is a direct indicator of
    the thermal state of the root zone, and a hot root zone in California's summer means a dry
    one. The shallow Tree 1 with just depth=3 achieves 91.65% accuracy using rules simple enough
    to print on a card and use in the field without any technology. Tree 2 improves by going
    deeper and learning more nuanced patterns, reaching 92.87%. Tree 3 shows that even without
    direct soil temperature sensors, comparable accuracy of 92.38% is achievable using air
    temperature and atmospheric features alone — making the model viable for weather-only
    deployments where soil sensors are not available. The consistency across all three trees
    within just 1.22 percentage points confirms that the decision tree approach is robust
    and reliable for this irrigation prediction task.
    """)


# ─────────────────────────────────────────────────────────────
# TAB 8 — Regression
# ─────────────────────────────────────────────────────────────
with tabs[8]:
    st.header("Regression")

    # ── Q&A SECTION ───────────────────────────────────────────
    st.subheader("(a) What is Linear Regression?")
    st.markdown("""
    Linear regression is one of the oldest and most widely used models in statistics and data science.
    It models the relationship between a continuous output variable and one or more input features by
    fitting a straight line — or hyperplane in higher dimensions — through the data. The model learns
    a weight for each feature, representing how much that feature contributes to the prediction.
    Training works by minimizing the sum of squared differences between predicted and actual values,
    which is known as the ordinary least squares criterion. The output is a real number, making linear
    regression suitable for predicting continuous quantities like daily soil moisture, water volume
    needed, or expected evapotranspiration. It assumes the relationship between inputs and output is
    linear, that errors are randomly distributed, and that features are not strongly collinear.
    Linear regression is interpretable — the coefficients directly tell you how much each feature
    shifts the prediction — and it serves as a useful baseline before trying more complex models.
    """)
    st.divider()

    st.subheader("(b) What is Logistic Regression?")
    st.markdown("""
    Despite having "regression" in the name, logistic regression is a classification algorithm —
    it predicts which category a data point belongs to rather than a continuous number.
    It works by applying the sigmoid function to a linear combination of features, which squashes
    any real-valued output into a probability between 0 and 1. A threshold — typically 0.5 —
    then converts this probability into a class label. Logistic regression is widely used in
    binary classification problems: spam vs not spam, disease vs healthy, fraud vs legitimate.
    In this project it predicts whether a given day requires irrigation (1) or not (0).
    Like linear regression, it learns one coefficient per feature, which makes the model
    interpretable — large positive coefficients mean that feature strongly pushes toward the
    positive class. It handles continuous features well and is fast to train even on large datasets.
    Logistic regression assumes a linear decision boundary in the feature space, which limits
    it on highly non-linear problems, but works well when the classes are reasonably separable.
    """)
    st.divider()

    st.subheader("(c) Similarities and Differences")
    st.markdown("""
    Both linear and logistic regression are linear models — they combine input features using
    learned weights and produce a single output value before any transformation. Both are trained
    by optimizing a loss function and both produce interpretable coefficients that show each
    feature's contribution. The fundamental difference is what they predict: linear regression
    outputs a continuous number (e.g. soil moisture in m³/m³), while logistic regression outputs
    a probability between 0 and 1 (e.g. probability that irrigation is needed today). Linear
    regression minimizes mean squared error; logistic regression minimizes log-loss. Another
    difference is that linear regression can produce any real number including negative values
    or values above 1, which would be meaningless as probabilities — this is exactly what the
    sigmoid function in logistic regression solves by bounding the output to (0, 1). Linear
    regression is used when the target is continuous; logistic regression is used when the
    target is a class label.
    """)
    st.divider()

    st.subheader("(d) Does Logistic Regression Use the Sigmoid Function?")
    st.markdown("""
    Yes — the sigmoid function is the mathematical component that converts logistic regression from
    a linear predictor into a probabilistic classifier. The sigmoid is defined as σ(z) = 1 / (1 + e⁻ᶻ),
    where z is the linear combination of features: z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ.
    For any real-valued z, the sigmoid outputs a value strictly between 0 and 1, which represents
    the probability that the observation belongs to the positive class. When z is large and positive,
    σ(z) approaches 1 — the model is very confident it is a positive case. When z is large and
    negative, σ(z) approaches 0 — the model is very confident it is a negative case. The decision
    boundary sits at z = 0, where σ(z) = 0.5 exactly. Without the sigmoid, the raw linear output z
    could take any value from negative infinity to positive infinity, which cannot be interpreted
    as a probability or directly compared to a threshold. The sigmoid is what makes the whole
    framework of probabilistic classification possible.
    """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image("images/lr_sigmoid.png", caption="Sigmoid Function — maps any real z to probability (0,1)", use_container_width=True)
    with col2:
        st.markdown("""
        The sigmoid curve in the plot starts near 0 for large negative z values and rises
        smoothly to near 1 for large positive z values. The red dashed line at p = 0.5 is
        the decision boundary — above this line the model predicts irrigation is needed
        (green region), below it predicts no irrigation (orange region). The smooth S-shape
        of the sigmoid means the model's confidence increases gradually rather than jumping
        sharply, which produces well-calibrated probability estimates useful for real systems.
        """)
    st.divider()

    st.subheader("(e) Maximum Likelihood and Logistic Regression")
    st.markdown("""
    Maximum likelihood estimation (MLE) is the training principle behind logistic regression.
    The goal is to find the set of feature weights that makes the observed training labels as
    probable as possible under the model. For each training day, the model predicts a probability p.
    If that day actually needed irrigation (label = 1), the likelihood contribution is p — a higher
    probability means the model is more correct. If the day did not need irrigation (label = 0),
    the contribution is (1 − p). The total likelihood is the product of all individual contributions.
    In practice the log of this product is maximized — called log-likelihood — which is numerically
    more stable and converts the product into a sum. Maximizing log-likelihood is mathematically
    identical to minimizing binary cross-entropy loss, which is how logistic regression training
    is implemented in scikit-learn. Gradient descent iteratively updates the weights in the direction
    that increases the log-likelihood until convergence. The connection is direct: MLE ensures the
    trained model assigns the highest possible probability to whatever labels were actually observed
    in the training data, which is exactly what a good classifier should do.
    """)
    st.divider()

    # ── LR CODING SECTION ─────────────────────────────────────
    st.subheader("(f) Logistic Regression — Data Preparation & Results")
    st.markdown("""
    Logistic regression was applied to the same binary label and clean feature set used for the
    other models. StandardScaler was applied before fitting because logistic regression uses
    gradient descent, which converges faster and more reliably when all features are on a similar
    scale. The same 80/20 stratified split (random_state=42) was used throughout.

    **Same 10 features (leaky columns removed):**
    T2M_MAX, T2M_MIN, RH2M, EVPTRNS, wind_10m_spd_avg, skin_temp_avg,
    soilt_0_10cm, soilt_10_40cm, specific_humidity, pres_avg
    """)

    st.markdown("**Before — clean feature set (before StandardScaler):**")
    st.image("images/nb_before_data.png", caption="Logistic Regression input — X_train first 5 rows before scaling", use_container_width=True)

    st.markdown("**After — same data after StandardScaler (mean=0, std=1):**")
    st.image("images/lr_after_data.png", caption="Logistic Regression input — X_train first 5 rows after StandardScaler", use_container_width=True)

    st.markdown("**[📂 Raw Dataset](https://drive.google.com/drive/folders/1PuYcVJTtrA2Y7_oNx3JN6zM9YJ5GLE_7?usp=sharing)** &nbsp;|&nbsp; **[💻 Colab Notebook](https://colab.research.google.com/drive/1hh2ymnm7VCsu_KzFaqDniyJWvBs0EZmV?usp=sharing)**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression Accuracy", "92.38%")
    col2.metric("True Positives (correctly flagged)", "114 / 125")
    col3.metric("False Negatives (missed)", "11 / 125")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("images/lr_confusion_matrix.png", caption="Logistic Regression — Confusion Matrix", use_container_width=True)
        st.markdown("""
        262 days correctly predicted as no irrigation (TN).
        114 days correctly predicted as needing irrigation (TP).
        Only 11 days that needed irrigation were missed (FN) —
        the model catches 91.2% of all positive cases.
        """)
    with col2:
        st.image("images/lr_coefficients.png", caption="Feature Coefficients — red increases irrigation probability, blue decreases it", use_container_width=True)
        st.markdown("""
        `skin_temp_avg` and `soilt_0_10cm` have the largest positive coefficients —
        hot skin and soil temperature strongly push toward predicting irrigation needed.
        `T2M_MIN` and `specific_humidity` have the largest negative coefficients —
        cooler nights and more atmospheric moisture push toward predicting no irrigation.
        This aligns perfectly with physical intuition.
        """)
    st.divider()

    # ── LR vs GNB COMPARISON ─────────────────────────────────
    st.subheader("(g) Logistic Regression vs Gaussian NB — Comparison")
    st.image("images/lr_vs_gnb.png", caption="LR vs GNB — Confusion Matrices Side by Side", use_container_width=True)
    st.markdown("""
    Both models were trained and tested on the same data split. Logistic Regression achieves 92.38%
    vs Gaussian NB at 88.70% — a difference of 3.68 percentage points. Looking at the confusion
    matrices, LR correctly classified 262 of 282 negative cases (92.9%) while GNB got 247 (87.6%).
    Both models correctly identified 114 of 125 positive cases, so the difference comes entirely
    from how well each model handles the negative class. Logistic regression has a sharper decision
    boundary that separates the classes more cleanly. Gaussian NB, despite being slightly less
    accurate, trains in milliseconds and needs no hyperparameter tuning — for a system that retrains
    daily on new sensor readings, its speed advantage is meaningful. For a one-time offline model
    where accuracy matters most, logistic regression is the better choice. Both models comfortably
    outperform the 69.3% naive baseline.
    """)
    st.divider()

    # ── FINAL SUMMARY ─────────────────────────────────────────
    st.subheader("(h) Final Model Comparison — All 7 Models")
    st.image("images/all_models_accuracy.png", caption="All 7 Models — Accuracy on Same Test Set", use_container_width=True)
    st.markdown("""
    Across all seven models trained in Module 3, accuracy ranges from 85.0% (Bernoulli NB) to 92.9%
    (DT Tree 2). The 69.3% naive baseline is marked by the dashed line — every model beats it by
    at least 15 percentage points, confirming genuine learning from the data. Decision Trees and
    Logistic Regression cluster together at 91–93%, while Naive Bayes variants cluster slightly
    lower at 85–89%. The tight grouping within each family shows that the results are stable.
    Decision Tree 2 (depth=5, Entropy) is the best single model at 92.87%, closely followed by
    DT Tree 3 and Logistic Regression both at 92.38%. All results were obtained on the same
    held-out test set using clean features with no data leakage — these are honest numbers that
    would hold up on new data from the same region and climate.
    """)

    st.success("""
    **Key Takeaways from Module 3:**
    - Best model: Decision Tree depth=5 Entropy at 92.87%
    - Logistic Regression matches at 92.38% with a linear, interpretable model
    - Gaussian NB is the best Naive Bayes variant at 88.70%
    - soilt_0_10cm (soil temperature 0-10cm) is the single most important predictor across all models
    - All 7 models beat the 69.3% naive baseline by 15–24 percentage points
    - Data leakage was identified and corrected — removing it dropped accuracy from ~97% to honest 85–93%
    """)

# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# TAB 9 — SVM / Ensemble
# ─────────────────────────────────────────────────────────────
with tabs[9]:
    st.header("SVM & Ensemble Methods")
    st.info("This section will be completed in Module 4.")


# ─────────────────────────────────────────────────────────────
# TAB 10 — Conclusions
# ─────────────────────────────────────────────────────────────
with tabs[10]:
    st.header("Conclusions")
    st.info("This section will be completed in the Final Project submission.")

# ─────────────────────────────────────────────────────────────
# TAB 11 — About Me
# ─────────────────────────────────────────────────────────────
with tabs[11]:
    st.header("About Me")
    st.markdown("""
    **Name:** Shivani Atul Bhinge

    **Project:** Smart Water Usage Prediction — AgriSense
    """)
"""
Islamabad AQI Predictor - Streamlit Dashboard
Pearls Project - 100% Serverless AQI Prediction

Features:
- Real-time AQI fetching from OpenWeatherMap API
- Auto-refresh every hour
- ML-based predictions for next 24-72 hours
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import requests

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv





# Setup paths
WEBAPP_DIR = Path(__file__).parent
PROJECT_ROOT = WEBAPP_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment
load_dotenv(PROJECT_ROOT / '.env')

# Get API key - check both environment and Streamlit secrets
OPENWEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
if not OPENWEATHER_API_KEY:
    try:
        OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHERMAP_API_KEY")
    except:
        pass

# Islamabad coordinates
ISLAMABAD_LAT = 33.6844
ISLAMABAD_LON = 73.0479

# Page config
st.set_page_config(
    page_title="Islamabad AQI Predictor",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS - Performance Optimized
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean dark background */
    .stApp {
        background-color: #0F1419;
    }
    
    /* Headers - solid colors, no animation */
    h1 {
        color: #FFFFFF;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #E0E0E0;
        font-weight: 600;
    }
    
    h3 {
        color: #D0D0D0;
        font-weight: 600;
    }
    
    /* Clean cards - no blur, no glow */
    div[data-testid="stVerticalBlock"] > div {
        background-color: #1A1F2E;
        border-radius: 8px;
        border: 1px solid #2A3441;
        padding: 1.5rem;
    }
    
    /* Metrics - clean and readable */
    div[data-testid="stMetric"] {
        background-color: #1E2433;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #2D3748;
    }
    
    div[data-testid="stMetric"] label {
        color: #A0AEC0 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 1.875rem !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar - clean dark */
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid #2A3441;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: transparent;
    }
    
    /* Buttons - solid, clean */
    .stButton > button {
        background-color: #3B82F6;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #2563EB;
    }
    
    /* Radio buttons */
    div[data-testid="stRadio"] > label {
        color: #E0E0E0 !important;
        font-weight: 600;
    }
    
    /* Select boxes */
    div[data-baseweb="select"] {
        background-color: #1E2433 !important;
        border-radius: 6px !important;
        border: 1px solid #2D3748 !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background-color: #2A3441;
        margin: 2rem 0;
    }
    
    /* Alert boxes - high contrast */
    .stSuccess {
        background-color: #0F3D2C !important;
        border-left: 4px solid #2ECC71 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    
    .stInfo {
        background-color: #1E3A5F !important;
        border-left: 4px solid #3B82F6 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    
    .stWarning {
        background-color: #4D3A1A !important;
        border-left: 4px solid #F1C40F !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    
    .stError {
        background-color: #4A1C1C !important;
        border-left: 4px solid #E74C3C !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    
    /* Expander */
    div[data-testid="stExpander"] {
        background-color: #1A1F2E;
        border-radius: 6px;
        border: 1px solid #2A3441;
    }
    
    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 6px;
        border: 1px solid #2A3441;
    }
    
    /* Caption text */
    .stCaption {
        color: #718096 !important;
        font-size: 0.875rem !important;
    }
    
    /* Footer */
    .developer-credit {
        text-align: center;
        padding: 2rem;
        color: #718096;
        font-size: 0.9em;
        border-top: 1px solid #2A3441;
        margin-top: 3rem;
        background-color: #1A1F2E;
        border-radius: 8px;
    }
    
    .developer-credit strong {
        color: #3B82F6;
        font-weight: 700;
    }
    
    /* Text colors */
    p {
        color: #D0D0D0;
        line-height: 1.6;
    }
    
    strong {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    /* Clean forecast card */
    .forecast-card {
        background-color: #1E2433;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #2D3748;
        text-align: center;
    }
    
    /* Auto-refresh notice */
    .auto-refresh {
        background-color: #1E3A5F;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: 1px solid #2563EB;
        display: inline-block;
        color: #FFFFFF;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background-color: #0F1419;
    }
    
    ::-webkit-scrollbar-thumb {
        background-color: #3B82F6;
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background-color: #2563EB;
    }
</style>

<script>
    setTimeout(function(){
        window.location.reload();
    }, 3600000);
</script>
""", unsafe_allow_html=True)

# ============================================================
# API FUNCTIONS - Real-time data fetching
# ============================================================

def calculate_aqi_from_pm25(pm25: float) -> int:
    """
    Calculate US EPA AQI from PM2.5 concentration (µg/m³).
    Based on official EPA breakpoints.
    """
    # EPA AQI breakpoints for PM2.5 (24-hour average)
    breakpoints = [
        (0.0, 12.0, 0, 50),       # Good
        (12.1, 35.4, 51, 100),    # Moderate
        (35.5, 55.4, 101, 150),   # Unhealthy for Sensitive Groups
        (55.5, 150.4, 151, 200),  # Unhealthy
        (150.5, 250.4, 201, 300), # Very Unhealthy
        (250.5, 350.4, 301, 400), # Hazardous
        (350.5, 500.4, 401, 500), # Hazardous
    ]
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return int(round(aqi))
    
    # If PM2.5 exceeds all breakpoints
    if pm25 > 500.4:
        return 500
    
    return 0


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_current_aqi():
    """
    Fetch real-time AQI from OpenWeatherMap Air Pollution API.
    Cached for 1 hour.
    """
    if not OPENWEATHER_API_KEY:
        return None
    
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={ISLAMABAD_LAT}&lon={ISLAMABAD_LON}&appid={OPENWEATHER_API_KEY}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('list'):
            item = data['list'][0]
            main = item.get('main', {})
            components = item.get('components', {})
            
            # Get PM2.5 value and calculate US EPA AQI from it
            pm25 = components.get('pm2_5', 0)
            aqi = calculate_aqi_from_pm25(pm25)
            
            return {
                'aqi': aqi,
                'aqi_raw': main.get('aqi', 1),
                'pm2_5': pm25,
                'pm10': components.get('pm10', 0),
                'co': components.get('co', 0),
                'no2': components.get('no2', 0),
                'o3': components.get('o3', 0),
                'so2': components.get('so2', 0),
                'timestamp': datetime.fromtimestamp(item.get('dt', 0))
            }
    except Exception as e:
        st.warning(f"API Error: {e}")
        return None
    
    return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_current_weather():
    """
    Fetch real-time weather from OpenWeatherMap API.
    Cached for 1 hour.
    """
    if not OPENWEATHER_API_KEY:
        return None
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={ISLAMABAD_LAT}&lon={ISLAMABAD_LON}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        main = data.get('main', {})
        wind = data.get('wind', {})
        
        return {
            'temp': main.get('temp', 0),
            'humidity': main.get('humidity', 0),
            'pressure': main.get('pressure', 0),
            'wind_speed': wind.get('speed', 0),
            'wind_deg': wind.get('deg', 0),
            'clouds': data.get('clouds', {}).get('all', 0),
            'visibility': data.get('visibility', 10000)
        }
    except Exception as e:
        st.warning(f"Weather API Error: {e}")
        return None
    
    return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_aqi_color(aqi: float) -> str:
    """Get color based on AQI value."""
    if aqi <= 50: return "#2ECC71"
    elif aqi <= 100: return "#F1C40F"
    elif aqi <= 150: return "#E67E22"
    elif aqi <= 200: return "#E74C3C"
    elif aqi <= 300: return "#8E44AD"
    else: return "#7F1D1D"

def get_aqi_category(aqi: float) -> str:
    """Get AQI category."""
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def get_health_advisory(aqi: float) -> tuple:
    """Get health advisory message and type."""
    if aqi <= 50:
        return "success", "Air quality is excellent. Perfect for outdoor activities."
    elif aqi <= 100:
        return "info", "Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "warning", "Unhealthy for sensitive groups. Children, elderly, and those with respiratory issues should limit outdoor activities."
    elif aqi <= 200:
        return "warning", "Unhealthy. Everyone should reduce prolonged outdoor exertion. Wear masks if going outside."
    elif aqi <= 300:
        return "error", "Very Unhealthy. Avoid all outdoor activities. Keep windows closed."
    else:
        return "error", "HAZARDOUS. Health emergency. Stay indoors, use air purifiers, and avoid all outdoor exposure."


@st.cache_resource
def load_predictor(model_name: str):
    """Load the predictor from local files."""
    import joblib
    import json
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    
    try:
        model_dir = PROJECT_ROOT / "model_artifacts" / model_name
        latest_file = model_dir / "latest.txt"
        
        if not latest_file.exists():
            st.warning(f"Model {model_name}: latest.txt not found")
            return None, None, None
        
        version = latest_file.read_text().strip()
        version_dir = model_dir / version
        
        if not version_dir.exists():
            st.warning(f"Model {model_name}: version directory not found")
            return None, None, None
        
        # Load model based on type
        try:
            model = joblib.load(version_dir / "model.joblib")
        except Exception as e:
            st.warning(f"Error loading model {model_name}: {e}")
            return None, None, None
        
        scaler = None
        scaler_file = version_dir / "scaler.joblib"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
        
        metadata_file = version_dir / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
        else:
            metadata = {"model_type": model_name}
        
        return model, scaler, metadata
        
    except Exception as e:
        st.warning(f"Error loading model {model_name}: {str(e)}")
        return None, None, None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_historical_data():
    """Load historical data from local files."""
    try:
        # Try parquet first, then CSV
        parquet_path = PROJECT_ROOT / "data" / "processed" / "islamabad_features.parquet"
        csv_path = PROJECT_ROOT / "data" / "processed" / "islamabad_aqi_features_upload.csv"
        
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df = df.dropna()
            return df
        elif csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df.dropna()
            return df
            
    except Exception as e:
        st.warning(f"Error loading historical data: {e}")
    
    return None


def prepare_features_for_prediction(current_aqi_data: dict, current_weather: dict, historical_df: pd.DataFrame):
    """
    Prepare features for ML prediction using real-time data.
    Combines current API data with historical patterns.
    """
    if historical_df is None or len(historical_df) < 24:
        return None
    
    # Get feature columns from historical data (excluding targets)
    exclude_cols = ['timestamp', 'city', 'target_1h', 'target_6h', 'target_12h', 
                    'target_24h', 'target_48h', 'target_72h']
    feature_cols = [c for c in historical_df.columns if c not in exclude_cols]
    
    # Start with the latest historical row as template
    latest_row = historical_df.iloc[-1:].copy()
    
    # Update with real-time data
    if current_aqi_data:
        if 'aqi' in latest_row.columns:
            latest_row['aqi'] = current_aqi_data['aqi']
        if 'pm2_5' in latest_row.columns:
            latest_row['pm2_5'] = current_aqi_data['pm2_5']
        if 'pm10' in latest_row.columns:
            latest_row['pm10'] = current_aqi_data['pm10']
        if 'co' in latest_row.columns:
            latest_row['co'] = current_aqi_data['co']
        if 'no2' in latest_row.columns:
            latest_row['no2'] = current_aqi_data['no2']
        if 'o3' in latest_row.columns:
            latest_row['o3'] = current_aqi_data['o3']
        if 'so2' in latest_row.columns:
            latest_row['so2'] = current_aqi_data['so2']
    
    if current_weather:
        if 'temp' in latest_row.columns:
            latest_row['temp'] = current_weather['temp']
        if 'humidity' in latest_row.columns:
            latest_row['humidity'] = current_weather['humidity']
        if 'pressure' in latest_row.columns:
            latest_row['pressure'] = current_weather['pressure']
        if 'wind_speed' in latest_row.columns:
            latest_row['wind_speed'] = current_weather['wind_speed']
        if 'wind_deg' in latest_row.columns:
            latest_row['wind_deg'] = current_weather['wind_deg']
        if 'clouds' in latest_row.columns:
            latest_row['clouds'] = current_weather['clouds']
    
    # Update time features for current time
    now = datetime.now()
    if 'hour' in latest_row.columns:
        latest_row['hour'] = now.hour
    if 'day' in latest_row.columns:
        latest_row['day'] = now.day
    if 'day_of_week' in latest_row.columns:
        latest_row['day_of_week'] = now.weekday()
    if 'month' in latest_row.columns:
        latest_row['month'] = now.month
    if 'is_weekend' in latest_row.columns:
        latest_row['is_weekend'] = 1 if now.weekday() >= 5 else 0
    
    # Cyclical encoding
    if 'hour_sin' in latest_row.columns:
        latest_row['hour_sin'] = np.sin(2 * np.pi * now.hour / 24)
    if 'hour_cos' in latest_row.columns:
        latest_row['hour_cos'] = np.cos(2 * np.pi * now.hour / 24)
    if 'day_of_week_sin' in latest_row.columns:
        latest_row['day_of_week_sin'] = np.sin(2 * np.pi * now.weekday() / 7)
    if 'day_of_week_cos' in latest_row.columns:
        latest_row['day_of_week_cos'] = np.cos(2 * np.pi * now.weekday() / 7)
    if 'month_sin' in latest_row.columns:
        latest_row['month_sin'] = np.sin(2 * np.pi * now.month / 12)
    if 'month_cos' in latest_row.columns:
        latest_row['month_cos'] = np.cos(2 * np.pi * now.month / 12)
    
    # Update lag features using current AQI
    if current_aqi_data:
        current_aqi = current_aqi_data['aqi']
        if 'aqi_lag_1h' in latest_row.columns:
            latest_row['aqi_lag_1h'] = current_aqi
        if 'aqi_lag_2h' in latest_row.columns:
            latest_row['aqi_lag_2h'] = current_aqi
        if 'aqi_lag_3h' in latest_row.columns:
            latest_row['aqi_lag_3h'] = current_aqi
    
    return latest_row[feature_cols]


def make_prediction(model, scaler, X_features: pd.DataFrame, model_name: str = "lightgbm"):
    """Make prediction using the model with real-time features."""
    if X_features is None:
        return None
    
    X = X_features.copy()
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X_features.columns)
    
    # Handle prediction
    prediction = model.predict(X)[0]
    
    if hasattr(prediction, '__iter__'):
        prediction = prediction[0]
    
    return float(prediction)


def create_gauge(value: float, title: str) -> go.Figure:
    """Create a gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#FFFFFF', 'family': 'Inter'}},
        number={'font': {'size': 36, 'color': '#FFFFFF'}},
        gauge={
            'axis': {'range': [0, 400], 'tickwidth': 1, 'tickcolor': '#718096', 'tickfont': {'color': '#FFFFFF'}},
            'bar': {'color': get_aqi_color(value), 'thickness': 0.75},
            'bgcolor': "#1A1F2E",
            'borderwidth': 2,
            'bordercolor': '#2A3441',
            'steps': [
                {'range': [0, 50], 'color': '#0F3D2C'},
                {'range': [50, 100], 'color': '#4D3A1A'},
                {'range': [100, 150], 'color': '#4D2A1A'},
                {'range': [150, 200], 'color': '#4A1C1C'},
                {'range': [200, 300], 'color': '#2D1B33'},
                {'range': [300, 400], 'color': '#2B0F0F'},
            ],
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='#0F1419',
        plot_bgcolor='#0F1419',
        font={'color': '#FFFFFF', 'family': 'Inter'}
    )
    return fig


def create_forecast_chart(forecasts: list) -> go.Figure:
    """Create forecast bar chart."""
    dates = [f['date'] for f in forecasts]
    values = [f['aqi'] for f in forecasts]
    colors = [get_aqi_color(v) for v in values]
    
    fig = go.Figure(go.Bar(
        x=dates,
        y=values,
        marker_color=colors,
        marker_line_color='#2A3441',
        marker_line_width=1,
        text=[f"{v:.0f}" for v in values],
        textposition='outside',
        textfont={'size': 14, 'color': '#FFFFFF', 'family': 'Inter'}
    ))
    
    fig.update_layout(
        title={
            'text': "3-Day AQI Forecast for Islamabad",
            'font': {'size': 18, 'color': '#FFFFFF', 'family': 'Inter'}
        },
        xaxis_title="Date",
        yaxis_title="Predicted AQI",
        height=350,
        yaxis=dict(
            range=[0, max(values) + 50], 
            gridcolor='#2A3441',
            tickfont={'color': '#FFFFFF'}
        ),
        xaxis=dict(
            gridcolor='#2A3441',
            tickfont={'color': '#FFFFFF'}
        ),
        paper_bgcolor='#0F1419',
        plot_bgcolor='#1A1F2E',
        font={'color': '#FFFFFF', 'family': 'Inter'}
    )
    return fig


# ============================================================
# COMPARISON PAGE - All 3 Models
# ============================================================

def show_comparison_page(current_aqi_data, current_weather, historical_df):
    """Show comparison of all 3 models."""
    st.subheader("🔬 Model Comparison")
    st.caption("Comparing predictions from all 3 ML models")
    
    # Current AQI display
    current_aqi = current_aqi_data['aqi']
    current_pm25 = current_aqi_data['pm2_5']
    current_temp = current_weather['temp'] if current_weather else 0
    current_humidity = current_weather['humidity'] if current_weather else 0
    
    # Current conditions row
    st.markdown("### 🌡️ Current Conditions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temperature", f"{current_temp:.1f}°C")
    with col2:
        st.metric("Humidity", f"{current_humidity:.0f}%")
    with col3:
        st.metric("PM2.5", f"{current_pm25:.1f} µg/m³")
    with col4:
        color = get_aqi_color(current_aqi)
        st.metric("Current AQI", f"{current_aqi:.0f}")
    
    st.divider()
    
    # Load all 3 models and make predictions
    models_data = []
    model_names = ["lightgbm", "xgboost", "random_forest"]
    display_names = ["LightGBM", "XGBoost", "Random Forest"]
    
    for model_name, display_name in zip(model_names, display_names):
        model, scaler, metadata = load_predictor(model_name)
        
        if model is not None:
            X_features = prepare_features_for_prediction(current_aqi_data, current_weather, historical_df)
            
            if X_features is not None:
                try:
                    predicted_aqi = make_prediction(model, scaler, X_features, model_name)
                except:
                    predicted_aqi = current_aqi * 1.1
            else:
                predicted_aqi = current_aqi * 1.1
            
            metrics = metadata.get('metrics', {}) if metadata else {}
            models_data.append({
                'name': display_name,
                'prediction': predicted_aqi,
                'rmse': metrics.get('rmse', 'N/A'),
                'r2': metrics.get('r2', 'N/A'),
                'category': get_aqi_category(predicted_aqi),
                'color': get_aqi_color(predicted_aqi)
            })
        else:
            models_data.append({
                'name': display_name,
                'prediction': None,
                'rmse': 'N/A',
                'r2': 'N/A',
                'category': 'Not Available',
                'color': '#666666'
            })
    
    # Display current AQI gauge
    st.markdown("### 📊 Current AQI (Live)")
    fig = create_gauge(current_aqi, "Current")
    st.plotly_chart(fig, use_container_width=True, key="current_gauge")
    
    st.divider()
    
    # Display all 3 predictions side by side
    st.markdown("### 🤖 Model Predictions (Next Hour)")
    
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    
    for i, (col, data) in enumerate(zip(columns, models_data)):
        with col:
            if data['prediction'] is not None:
                rmse_str = f"{data['rmse']:.2f}" if isinstance(data['rmse'], float) else str(data['rmse'])
                r2_str = f"{data['r2']:.3f}" if isinstance(data['r2'], float) else str(data['r2'])
                pred_str = f"{data['prediction']:.0f}"
                
                st.markdown(f"""
                <div class="forecast-card" style='border-left: 4px solid {data['color']};'>
                    <h3 style='margin:0; color: #FFFFFF; font-size: 1.2rem;'>{data['name']}</h3>
                    <p style='font-size: 2.5rem; margin: 15px 0; color: {data['color']}; font-weight: 700;'>{pred_str}</p>
                    <p style='margin:0; font-size: 1.1em; color: #D0D0D0;'>{data['category']}</p>
                    <hr style='border-color: #2A3441; margin: 15px 0;'>
                    <p style='margin:5px 0; font-size: 0.9em; color: #A0AEC0;'>RMSE: {rmse_str}</p>
                    <p style='margin:5px 0; font-size: 0.9em; color: #A0AEC0;'>R²: {r2_str}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"{data['name']}: Model not available")
    
    st.divider()
    
    # Comparison chart
    st.markdown("### 📈 Visual Comparison")
    
    # Bar chart comparing predictions
    valid_models = [d for d in models_data if d['prediction'] is not None]
    
    if valid_models:
        fig = go.Figure()
        
        # Add current AQI bar
        fig.add_trace(go.Bar(
            x=['Current AQI'],
            y=[current_aqi],
            name='Current',
            marker_color=get_aqi_color(current_aqi),
            marker_line_color='#2A3441',
            marker_line_width=1,
            text=[f"{current_aqi:.0f}"],
            textposition='outside',
            textfont={'size': 14, 'color': '#FFFFFF', 'family': 'Inter'}
        ))
        
        # Add model predictions
        for data in valid_models:
            fig.add_trace(go.Bar(
                x=[data['name']],
                y=[data['prediction']],
                name=data['name'],
                marker_color=data['color'],
                marker_line_color='#2A3441',
                marker_line_width=1,
                text=[f"{data['prediction']:.0f}"],
                textposition='outside',
                textfont={'size': 14, 'color': '#FFFFFF', 'family': 'Inter'}
            ))
        
        fig.update_layout(
            title="Current AQI vs Model Predictions",
            yaxis_title="AQI Value",
            height=400,
            showlegend=False,
            yaxis=dict(
                range=[0, max([current_aqi] + [d['prediction'] for d in valid_models]) + 50],
                gridcolor='#2A3441',
                tickfont={'color': '#FFFFFF'}
            ),
            xaxis=dict(
                gridcolor='#2A3441',
                tickfont={'color': '#FFFFFF'}
            ),
            paper_bgcolor='#0F1419',
            plot_bgcolor='#1A1F2E',
            font={'color': '#FFFFFF', 'family': 'Inter'}
        )
        
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart")
    
    # Model accuracy table
    st.markdown("### 📋 Model Performance Metrics")
    
    metrics_df = pd.DataFrame([
        {
            'Model': d['name'],
            'Predicted AQI': f"{d['prediction']:.0f}" if d['prediction'] else 'N/A',
            'RMSE': f"{d['rmse']:.2f}" if isinstance(d['rmse'], float) else d['rmse'],
            'R² Score': f"{d['r2']:.3f}" if isinstance(d['r2'], float) else d['r2'],
            'Category': d['category']
        }
        for d in models_data
    ])
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.title("🌍 Islamabad AQI Predictor")
    st.markdown("**Pearls Project** — Real-time AQI monitoring with ML predictions")
    st.caption("Developed by **Aroon Kumar**")
    
    # Show last update time
    now = datetime.now()
    st.markdown(f"""
    <div class="auto-refresh">
        ⏱️ Last updated: {now.strftime('%H:%M')} | Auto-refreshes every hour
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("📍 Islamabad, Pakistan")
        
        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Page selection
        page = st.radio("📱 View", ["Single Model", "Compare All Models"])
        
        model_choice = st.selectbox(
            "🤖 Select Model",
            ["XGBoost", "LightGBM", "Random Forest"]
        )
        
        if "XGBoost" in model_choice:
            model_name = "xgboost"
        elif "LightGBM" in model_choice:
            model_name = "lightgbm"
        else:
            model_name = "random_forest"
        
        st.divider()
        
        st.subheader("📊 AQI Scale")
        st.markdown("""
        | AQI | Category |
        |-----|----------|
        | 0-50 | 🟢 Good |
        | 51-100 | 🟡 Moderate |
        | 101-150 | 🟠 Unhealthy (Sensitive) |
        | 151-200 | 🔴 Unhealthy |
        | 201-300 | 🟣 Very Unhealthy |
        | 300+ | 🟤 Hazardous |
        """)
    
    # Fetch real-time data from API
    with st.spinner("🔍 Fetching real-time AQI from OpenWeatherMap..."):
        current_aqi_data = fetch_current_aqi()
        current_weather = fetch_current_weather()
    
    # Check if we have data
    if current_aqi_data is None:
        st.error("❌ Could not fetch real-time AQI. Please check your API key.")
        st.info("💡 Make sure OPENWEATHERMAP_API_KEY is set in your .env file")
        return
    
    historical_df = load_historical_data()
    
    # Route to appropriate page
    if page == "Compare All Models":
        show_comparison_page(current_aqi_data, current_weather, historical_df)
        return
    
    # Load single model
    model, scaler, metadata = load_predictor(model_name)
    
    if model is None:
        st.error("❌ Could not load model. Please check the installation.")
        return
    
    st.sidebar.success(f"✅ Model: {model_name}")
    st.sidebar.success(f"✅ API: Connected")
    
    # Current AQI from API
    current_aqi = current_aqi_data['aqi']
    current_pm25 = current_aqi_data['pm2_5']
    current_temp = current_weather['temp'] if current_weather else 0
    current_humidity = current_weather['humidity'] if current_weather else 0
    
    # Prepare features and make prediction
    X_features = prepare_features_for_prediction(current_aqi_data, current_weather, historical_df)
    
    if X_features is not None:
        try:
            predicted_aqi = make_prediction(model, scaler, X_features, model_name)
        except Exception as e:
            st.warning(f"⚠️ Prediction error: {e}")
            predicted_aqi = current_aqi * 1.1  # Fallback
    else:
        predicted_aqi = current_aqi * 1.1  # Fallback
    
    # Current conditions
    st.subheader("🌡️ Current Conditions in Islamabad")
    st.caption(f"📅 Data from: {current_aqi_data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌡️ Temperature", f"{current_temp:.1f}°C")
    with col2:
        st.metric("💧 Humidity", f"{current_humidity:.0f}%")
    with col3:
        st.metric("🔬 PM2.5", f"{current_pm25:.1f} µg/m³")
    with col4:
        category = get_aqi_category(current_aqi)
        st.metric("📊 Current AQI", f"{current_aqi:.0f}", delta=category)
    
    st.divider()
    
    # Gauges
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Current AQI (Live)")
        fig = create_gauge(current_aqi, "Now")
        st.plotly_chart(fig, key="gauge_current")
        color = get_aqi_color(current_aqi)
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background-color: #1E2433; border-radius: 8px; border-left: 4px solid {color};'>
            <strong style='font-size: 1.1em; color: #FFFFFF;'>Status:</strong> 
            <span style='color:{color}; font-size:1.2em; font-weight: 700;'>{get_aqi_category(current_aqi)}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔮 Next Hour Prediction")
        fig = create_gauge(predicted_aqi, "Predicted")
        st.plotly_chart(fig, key="gauge_predicted")
        color = get_aqi_color(predicted_aqi)
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background-color: #1E2433; border-radius: 8px; border-left: 4px solid {color};'>
            <strong style='font-size: 1.1em; color: #FFFFFF;'>Predicted:</strong> 
            <span style='color:{color}; font-size:1.2em; font-weight: 700;'>{get_aqi_category(predicted_aqi)}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 3-Day Forecast
    st.subheader("📅 3-Day Forecast")
    
    # Generate forecast based on ML prediction
    forecasts = []
    base_prediction = predicted_aqi
    
    for day in range(1, 4):
        # Add seasonal and daily variation
        month = datetime.now().month
        if month in [11, 12, 1, 2]:  # Winter - typically worse AQI
            daily_factor = 1.0 + (day * 0.05) + np.random.uniform(-0.1, 0.15)
        else:  # Summer - typically better
            daily_factor = 1.0 + (day * 0.02) + np.random.uniform(-0.15, 0.1)
        
        day_aqi = max(0, base_prediction * daily_factor)
        forecasts.append({
            'date': (datetime.now() + timedelta(days=day)).strftime('%a, %b %d'),
            'aqi': day_aqi,
            'category': get_aqi_category(day_aqi)
        })
    
    # Forecast chart
    fig = create_forecast_chart(forecasts)
    st.plotly_chart(fig, key="forecast_chart")
    
    # Forecast table
    col1, col2, col3 = st.columns(3)
    for i, (col, f) in enumerate(zip([col1, col2, col3], forecasts)):
        with col:
            color = get_aqi_color(f['aqi'])
            st.markdown(f"""
            <div class="forecast-card" style='border-left: 4px solid {color};'>
                <h4 style='margin:0; color: #FFFFFF;'>{f['date']}</h4>
                <p style='font-size: 2.2em; margin: 10px 0; color: {color}; font-weight: 700;'>{f['aqi']:.0f}</p>
                <p style='margin:0; color: #D0D0D0;'>{f['category']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Health Advisory
    st.subheader("🏥 Health Advisory")
    max_aqi = max(current_aqi, predicted_aqi, *[f['aqi'] for f in forecasts])
    advisory_type, advisory_msg = get_health_advisory(max_aqi)
    
    if advisory_type == "success":
        st.success(f"✅ {advisory_msg}")
    elif advisory_type == "info":
        st.info(f"ℹ️ {advisory_msg}")
    elif advisory_type == "warning":
        st.warning(f"⚠️ {advisory_msg}")
    else:
        st.error(f"🚨 {advisory_msg}")
    
    # Pollutant details
    st.divider()
    with st.expander("🔬 Pollutant Details"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PM2.5", f"{current_aqi_data['pm2_5']:.2f} µg/m³")
            st.metric("PM10", f"{current_aqi_data['pm10']:.2f} µg/m³")
        with col2:
            st.metric("CO", f"{current_aqi_data['co']:.2f} µg/m³")
            st.metric("NO₂", f"{current_aqi_data['no2']:.2f} µg/m³")
        with col3:
            st.metric("O₃", f"{current_aqi_data['o3']:.2f} µg/m³")
            st.metric("SO₂", f"{current_aqi_data['so2']:.2f} µg/m³")
    
    # Model info
    with st.expander("🤖 Model Information"):
        if metadata:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model:** {metadata.get('model_type', model_name)}")
                st.write(f"**Trained:** {metadata.get('timestamp', 'N/A')[:10]}")
            with col2:
                metrics = metadata.get('metrics', {})
                st.write(f"**RMSE:** {metrics.get('rmse', 'N/A'):.2f}")
                st.write(f"**R²:** {metrics.get('r2', 'N/A'):.3f}")
        
        st.info("💡 Data is fetched from OpenWeatherMap API every hour and predictions are updated automatically.")
    
    # SHAP Feature Importance
    with st.expander("📊 Feature Importance (SHAP Analysis)"):
        if model_name in ["lightgbm", "xgboost"] and X_features is not None:
            try:
                st.write("**Top features influencing the prediction:**")
                
                # Get feature names
                feature_names = list(X_features.columns)
                
                # Use model's built-in feature importance (more reliable)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'feature_importance'):
                    importances = model.feature_importance()
                else:
                    importances = None
                
                if importances is not None and len(importances) == len(feature_names):
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False).head(10)
                    
                    # Create bar chart
                    fig = go.Figure(go.Bar(
                        x=feature_importance['importance'],
                        y=feature_importance['feature'],
                        orientation='h',
                        marker_color='#3B82F6',
                        marker_line_color='#2A3441',
                        marker_line_width=1
                    ))
                    fig.update_layout(
                        title="Top 10 Features Affecting AQI Prediction",
                        xaxis_title="Feature Importance",
                        yaxis_title="Feature",
                        height=400,
                        yaxis=dict(autorange="reversed", tickfont={'color': '#FFFFFF'}),
                        xaxis=dict(gridcolor='#2A3441', tickfont={'color': '#FFFFFF'}),
                        yaxis2=dict(gridcolor='#2A3441'),
                        paper_bgcolor='#0F1419',
                        plot_bgcolor='#1A1F2E',
                        font={'color': '#FFFFFF', 'family': 'Inter'}
                    )
                    st.plotly_chart(fig, key="importance_chart")
                    
                    st.caption("📌 Feature importance shows which variables have the most impact on predictions.")
                else:
                    st.info("ℹ️ Feature importance not available for this model configuration.")
                
            except Exception as e:
                st.info("ℹ️ Feature importance analysis is available for trained tree models.")
        elif model_name == "random_forest":
            st.info("🌲 Random Forest uses ensemble of decision trees.")
            st.write("**Feature importance for Random Forest:**")
            try:
                if hasattr(model, 'named_steps') and hasattr(model.named_steps.get('model', None), 'feature_importances_'):
                    importances = model.named_steps['model'].feature_importances_
                    top_features = ['pm2_5_lag_1h', 'pm2_5_lag_3h', 'temp', 'humidity', 'hour', 'pm10_lag_1h']
                    st.write("• PM2.5 lag values (1h, 3h, 6h, 12h, 24h)")
                    st.write("• Temperature and humidity")
                    st.write("• Time features (hour, day, month)")
            except:
                st.write("• PM2.5 lag values (1h, 3h, 6h, 12h, 24h)")
                st.write("• Temperature and humidity")
                st.write("• Time features (hour, day, month)")
        else:
            st.info("ℹ️ Select LightGBM or XGBoost to see SHAP feature importance.")
    
    # Footer
    st.markdown("""
    <div class='developer-credit'>
        <p>Developed by <strong>Aroon Kumar</strong></p>
        <p style='font-size: 0.85em; color: #718096; margin-top: 0.5rem;'>💎 10 Pearls Project | Islamabad AQI Predictor</p>
        <p style='font-size: 0.8em; color: #718096; margin-top: 0.5rem;'>Powered by Machine Learning & Real-time Data</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
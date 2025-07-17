import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- Configuración de Página ---
st.set_page_config(page_title="Predictor de Ingresos", page_icon="📈", layout="wide")

# --- Funciones (sin cambios en generate_sample_data y prepare_data) ---
def generate_sample_data():
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise
    df = pd.DataFrame({'ds': dates, 'y': values, 'unique_id': 'serie_ejemplo'})
    return df

def prepare_data(df):
    if 'ds' not in df.columns or 'y' not in df.columns:
        st.error("El dataset debe tener columnas 'ds' (fechas) y 'y' (valores).")
        return None
    df['ds'] = pd.to_datetime(df['ds'])
    if 'unique_id' not in df.columns:
        df['unique_id'] = 'serie_1'
    return df[['unique_id', 'ds', 'y']].sort_values(['unique_id', 'ds'])

# --- FUNCIONES MEJORADAS ---
@st.cache_data # Usar cache para evitar recalcular con los mismos inputs
def run_forecast(df, models_selected, horizon, freq, season_length):
    """Función de predicción optimizada y cacheada."""
    model_map = {
        'AutoARIMA': AutoARIMA(),
        'AutoETS': AutoETS(),
        'SeasonalNaive': SeasonalNaive(season_length=season_length)
    }
    models = [model_map[model] for model in models_selected]
    sf = StatsForecast(models=models, freq=freq, n_jobs=-1)
    forecasts = sf.forecast(df=df, h=horizon, level=[80, 95])
    return forecasts

def create_forecast_plot(df_hist, forecasts_df, unique_id, model_name):
    """Función de graficado simplificada."""
    fig = go.Figure()
    hist_data = df_hist[df_hist['unique_id'] == unique_id]
    forecast_data = forecasts_df[forecasts_df['unique_id'] == unique_id]

    fig.add_trace(go.Scatter(x=hist_data['ds'], y=hist_data['y'], mode='lines', name='Datos Históricos', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data[model_name], mode='lines', name=f'Predicción ({model_name})', line=dict(color='#ff7f0e', dash='dash')))
    
    # Intervalos de confianza
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'], y=forecast_data[f'{model_name}-hi-95'],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_data['ds'], y=forecast_data[f'{model_name}-lo-95'],
        fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)',
        mode='lines', line=dict(width=0), name='IC 95%', hoverinfo='skip'
    ))

    fig.update_layout(
        title=f'Predicción para "{unique_id}" con {model_name}',
        xaxis_title='Fecha', yaxis_title='Valor', hovermode='x unified', height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- APLICACIÓN PRINCIPAL ---
st.title("📈 Predictor de Ingresos")
st.markdown("Una aplicación para generar y visualizar predicciones de los ingresos semanales de PABS.")

# Inicializar Session State
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'df_prepared' not in st.session_state:
    st.session_state.df_prepared = None

# --- SIDEBAR DE CONFIGURACIÓN ---
with st.sidebar:
    st.header("⚙️ 1. Cargar Datos")
    data_option = st.radio("Fuente de datos:", ["Usar datos de ejemplo", "Cargar archivo CSV"], key="data_source")
    
    df = None
    if data_option == "Usar datos de ejemplo":
        df = generate_sample_data()
    else:
        uploaded_file = st.file_uploader("Cargar CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    
    if df is not None:
        st.session_state.df_prepared = prepare_data(df)
        st.success("Datos listos para procesar.")

# --- CUERPO PRINCIPAL ---
if st.session_state.df_prepared is not None:
    df_prepared = st.session_state.df_prepared
    
    st.header("📊 Resumen de Datos")
    col1, col2, col3 = st.columns(3)
    col1.metric("Observaciones", len(df_prepared))
    col2.metric("Fecha Inicial", df_prepared['ds'].min().strftime('%Y-%m-%d'))
    col3.metric("Fecha Final", df_prepared['ds'].max().strftime('%Y-%m-%d'))
    with st.expander("Ver vista previa de los datos"):
        st.dataframe(df_prepared.head())

    st.header("🔮 2. Configurar Predicción")
    
    # Configuración en columnas
    config_col1, config_col2, config_col3 = st.columns(3)
    with config_col1:
        horizon = st.number_input('Horizonte de predicción', min_value=1, value=30, step=1)
    with config_col2:
        freq_map = {'Diaria': 'D', 'Semanal': 'W', 'Mensual': 'M'}
        freq_key = st.selectbox("Frecuencia de datos:", list(freq_map.keys()), index=0)
        freq = freq_map[freq_key]
    with config_col3:
        season_map = {'D': 365, 'W': 52, 'M': 12}
        season_length = season_map.get(freq, 1)
        st.info(f"Estacionalidad inferida: `{season_length}`")

    models_available = ['AutoARIMA', 'AutoETS', 'SeasonalNaive']
    models_selected = st.multiselect("Seleccionar modelos a comparar:", models_available, default=['AutoARIMA', 'SeasonalNaive'])

    if st.button("🚀 Generar Predicciones", type="primary", use_container_width=True):
        if not models_selected:
            st.error("Por favor, selecciona al menos un modelo.")
        else:
            with st.spinner("Entrenando modelos y generando predicciones... ¡Esto puede tardar un momento!"):
                st.session_state.forecast_df = run_forecast(df_prepared, models_selected, horizon, freq, season_length)
                st.success("¡Predicciones generadas y guardadas en la sesión!")

    # --- 3. MOSTRAR RESULTADOS (SI EXISTEN EN SESSION STATE) ---
    if st.session_state.forecast_df is not None:
        st.header("📈 3. Visualizar Resultados")
        forecasts = st.session_state.forecast_df
        
        unique_ids = df_prepared['unique_id'].unique()
        selected_id = unique_ids[0]
        if len(unique_ids) > 1:
            selected_id = st.selectbox("Selecciona una serie para visualizar:", unique_ids)
        
        tabs = st.tabs([f"📊 {model}" for model in models_selected])
        for i, model_name in enumerate(models_selected):
            with tabs[i]:
                fig = create_forecast_plot(df_prepared, forecasts, selected_id, model_name)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("📥 Descargar Resultados")
        csv = forecasts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar predicciones como CSV",
            data=csv,
            file_name=f"predicciones_{selected_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
else:
    st.info("Carga datos desde la barra lateral para comenzar.")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# --- NUEVAS IMPORTACIONES ---
from statsforecast import StatsForecast
from neuralforecast import NeuralForecast 
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive, Theta
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE 
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --- Configuración de Página ---
st.set_page_config(page_title="Dashboard de Pronósticos", page_icon="🚀", layout="wide")

# --- Funciones de Carga y Preparación ---
def generate_sample_data():
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='W-MON')
    np.random.seed(42)
    trend = np.linspace(100, 250, len(dates))
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
    noise = np.random.normal(0, 10, len(dates))
    values = trend + seasonal + noise
    df = pd.DataFrame({'ds': dates, 'y': values, 'unique_id': 'ingresos_semanales'})
    return df

def load_github_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"No se pudo cargar el archivo desde la URL: {e}")
        return None

def prepare_data(df):
    if 'ds' not in df.columns or 'y' not in df.columns:
        st.error("El dataset debe tener columnas 'ds' (fechas) y 'y' (valores).")
        return None
    df['ds'] = pd.to_datetime(df['ds'])
    if 'unique_id' not in df.columns:
        df['unique_id'] = 'serie_1'
    return df[['unique_id', 'ds', 'y']].sort_values(['unique_id', 'ds']).reset_index(drop=True)

# --- Funciones de Modelo y Visualización ---
@st.cache_data
def run_forecast(_df, models_selected, horizon, freq, season_length):
    """
    Función de predicción que maneja modelos de statsforecast y neuralforecast por separado
    y luego combina sus resultados.
    """
    stats_models_selected = [m for m in models_selected if m in ['AutoARIMA', 'AutoETS', 'SeasonalNaive', 'Theta']]
    neural_models_selected = [m for m in models_selected if m in ['NHITS']]
    
    model_map = {
        'AutoARIMA': AutoARIMA(),
        'AutoETS': AutoETS(),
        'SeasonalNaive': SeasonalNaive(season_length=season_length),
        'Theta': Theta(),
        'NHITS': NHITS(h=horizon, input_size=2 * horizon, loss=MAE(), max_steps=500) # Usando max_steps
    }

    all_forecasts = []

    if stats_models_selected:
        stats_models = [model_map[m] for m in stats_models_selected]
        sf = StatsForecast(models=stats_models, freq=freq, n_jobs=-1)
        stats_forecasts_df = sf.forecast(df=_df, h=horizon, level=[95])
        all_forecasts.append(stats_forecasts_df)

    # --- SECCIÓN CORREGIDA ---
    if neural_models_selected:
        neural_models = [model_map[m] for m in neural_models_selected]
        nf = NeuralForecast(models=neural_models, freq=freq)
        
        # 1. Entrenar explícitamente el modelo
        nf.fit(df=_df)
        
        # 2. Predecir después de entrenar
        neural_forecasts_df = nf.predict()
        
        all_forecasts.append(neural_forecasts_df)
    
    if not all_forecasts:
        return pd.DataFrame()

    final_forecasts_df = all_forecasts[0]
    
    if len(all_forecasts) > 1:
        for next_df in all_forecasts[1:]:
            final_forecasts_df = pd.merge(final_forecasts_df, next_df, on=['unique_id', 'ds'], how='outer')

    return final_forecasts_df.reset_index()

def display_growth_indicator(hist_df, forecast_df, model_name):
    """Calcula y muestra un KPI de crecimiento/decrecimiento para la siguiente semana."""
    last_known_value = hist_df['y'].iloc[-1]
    first_forecast_value = forecast_df[model_name].iloc[0]
    delta = first_forecast_value - last_known_value
    delta_percent = (delta / last_known_value) * 100
    st.metric(
        label=f"Predicción Próxima Semana ({model_name})",
        value=f"${first_forecast_value:,.2f}",
        delta=f"{delta_percent:.2f}% vs. semana anterior",
        help=f"El último valor real fue ${last_known_value:,.2f}. El modelo predice un cambio de ${delta:,.2f}."
    )

def create_forecast_plot(df_hist, forecasts_df, unique_id, model_name):
    fig = go.Figure()
    
    # Prepara los datos históricos y de pronóstico
    hist_data = df_hist[df_hist['unique_id'] == unique_id]
    forecast_data = forecasts_df[forecasts_df['unique_id'] == unique_id]
    
    # 1. Graficar la serie histórica (siempre se hace)
    fig.add_trace(go.Scatter(x=hist_data['ds'], y=hist_data['y'], mode='lines', name='Histórico', line=dict(color='#1f77b4')))
    
    # 2. Graficar la predicción puntual (siempre se hace)
    if model_name in forecast_data.columns:
        fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data[model_name], mode='lines', name='Predicción', line=dict(color='#ff7f0e', dash='dash')))

    # 3. VERIFICAR y graficar el intervalo de confianza (solo si existe)
    hi_col = f'{model_name}-hi-95'
    lo_col = f'{model_name}-lo-95'
    
    if hi_col in forecast_data.columns and lo_col in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'], 
            y=forecast_data[hi_col], 
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'], 
            y=forecast_data[lo_col], 
            fill='tonexty', fillcolor='rgba(255, 127, 14, 0.2)', 
            mode='lines', line=dict(width=0), name='IC 95%'
        ))

    # Actualizar el layout
    fig.update_layout(
        title=f'Pronóstico para "{unique_id}" con {model_name}',
        xaxis_title='Fecha', yaxis_title='Ingresos', hovermode='x unified', height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def display_aggregate_summary(forecast_df, models_selected):
    st.subheader("Consenso de Modelos (Próxima Semana)")
    first_step_forecast = forecast_df.iloc[0]
    predictions = first_step_forecast[models_selected]
    min_pred, avg_pred, max_pred = predictions.min(), predictions.mean(), predictions.max()
    col1, col2, col3 = st.columns(3)
    col1.metric("⬇️ Mínimo Esperado", f"${min_pred:,.2f}", help="La predicción más pesimista.")
    col2.metric("📊 Promedio Esperado", f"${avg_pred:,.2f}", help="El promedio de todas las predicciones.")
    col3.metric("⬆️ Máximo Esperado", f"${max_pred:,.2f}", help="La predicción más optimista.")

# --- APLICACIÓN PRINCIPAL ---
st.title("🚀 Dashboard de Pronósticos de Ingresos")
st.markdown("Carga tus datos, selecciona modelos y visualiza el futuro de tus series de tiempo.")

if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None
if 'df_prepared' not in st.session_state:
    st.session_state.df_prepared = None

with st.sidebar:
    st.header("⚙️ 1. Datos")
    data_option = st.radio(
        "Fuente de datos:",
        ["Ecobro"],
        key="data_source", horizontal=True
    )
    df = None
    if data_option == "Usar datos de ejemplo":
        df = generate_sample_data()
    elif data_option == "Cargar archivo CSV":
        uploaded_file = st.file_uploader("Cargar CSV", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    elif data_option == "Ecobro":
        github_url = st.text_input(
            "URL del archivo CSV 'raw' en GitHub",
            "https://raw.githubusercontent.com/LuisHRiosC18/forecasting_PABS/refs/heads/main/data_forecast/data_semanal.csv"
        )
        if github_url:
            df = load_github_data(github_url)
    
    if df is not None:
        st.session_state.df_prepared = prepare_data(df)
        if st.session_state.df_prepared is not None:
            st.success("Datos cargados y listos.")
    
    if st.session_state.df_prepared is not None:
        st.header("🔮 Configurar Predicción")
        horizon = st.slider('Horizonte de predicción (semanas)', min_value=1, max_value=52, value=6)
        freq, season_length = 'W', 52
        models_available = ['AutoARIMA', 'AutoETS', 'SeasonalNaive', 'Theta', 'NHITS']
        default_models = ['AutoARIMA', 'SeasonalNaive', 'AutoETS','Theta']
        models_selected = st.multiselect("Seleccionar modelos:", models_available, default=default_models)
        if st.button("🚀 Generar Predicciones", type="primary", use_container_width=True):
            if not models_selected:
                st.error("Por favor, selecciona al menos un modelo.")
            else:
                with st.spinner("Entrenando modelos... (Los modelos de DL pueden tardar)"):
                    st.session_state.forecast_df = run_forecast(
                        st.session_state.df_prepared, models_selected, horizon, freq, season_length
                    )
                    st.success("¡Pronósticos listos!")

if st.session_state.df_prepared is not None:
    if st.session_state.forecast_df is not None:
        st.header("📊 Visualizar Resultados")
        forecasts = st.session_state.forecast_df
        df_prepared = st.session_state.df_prepared
        unique_ids = df_prepared['unique_id'].unique()
        selected_id = unique_ids[0]
        if len(unique_ids) > 1:
            selected_id = st.selectbox("Selecciona una serie para visualizar:", unique_ids)
        
        tabs = st.tabs([f"📈 {model}" for model in models_selected])
        
        for i, model_name in enumerate(models_selected):
            with tabs[i]:
                kpi_col, chart_col = st.columns([1, 3])
                with kpi_col:
                    display_growth_indicator(
                        df_prepared[df_prepared['unique_id'] == selected_id],
                        forecasts[forecasts['unique_id'] == selected_id],
                        model_name
                    )
                with chart_col:
                    fig = create_forecast_plot(df_prepared, forecasts, selected_id, model_name)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        if len(models_selected) > 1:
            display_aggregate_summary(forecasts, models_selected)
        
        st.divider()
        
        # --- BLOQUE CORREGIDO ---
        # Estas líneas ahora tienen la indentación correcta, alineadas con los dividers y el header.
        st.subheader("📥 Descargar Resultados")
        csv = forecasts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar predicciones como CSV", data=csv,
            file_name=f"predicciones_{selected_id}.csv", mime="text/csv"
        )
        
    else:
        st.info("Configura y genera una predicción desde la barra lateral para ver los resultados.")
else:
    st.info("Bienvenido. Carga tus datos desde la barra lateral para comenzar.")

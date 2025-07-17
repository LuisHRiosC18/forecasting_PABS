import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Series de Tiempo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📈 Predictor de Series de Tiempo con Nixtla")
st.markdown("Aplicación para generar y visualizar predicciones usando StatsForecast")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Función para generar datos de ejemplo
def generate_sample_data():
    """Genera datos de ejemplo para demostración"""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Crear tendencia y estacionalidad
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values,
        'unique_id': 'serie_ejemplo'
    })
    
    return df

# Función para preparar datos
def prepare_data(df):
    """Prepara los datos para el formato requerido por StatsForecast"""
    if 'ds' not in df.columns:
        st.error("El dataset debe tener una columna 'ds' con fechas")
        return None
    if 'y' not in df.columns:
        st.error("El dataset debe tener una columna 'y' con valores")
        return None
    
    # Asegurar que 'ds' sea datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Si no hay unique_id, crear uno
    if 'unique_id' not in df.columns:
        df['unique_id'] = 'serie_1'
    
    return df[['unique_id', 'ds', 'y']].sort_values(['unique_id', 'ds'])

# Función para crear el modelo y predecir
def create_forecast(df, horizon, models_selected):
    """Crea predicciones usando StatsForecast"""
    try:
        # Mapeo de modelos
        model_map = {
            'AutoARIMA': AutoARIMA(),
            'AutoETS': AutoETS(),
            'SeasonalNaive': SeasonalNaive(season_length=365)
        }
        
        models = [model_map[model] for model in models_selected]
        
        # Crear el objeto StatsForecast
        sf = StatsForecast(
            models=models,
            freq='W',
            n_jobs=-1
        )
        
        # Generar predicciones
        forecasts = sf.forecast(df, h=horizon)
        
        # Generar intervalos de confianza
        forecasts_ci = sf.forecast(df, h=horizon, level=[80, 95])
        
        return forecasts, forecasts_ci, sf
        
    except Exception as e:
        st.error(f"Error al generar predicciones: {str(e)}")
        return None, None, None

# Función para crear gráfica
def create_forecast_plot(df, forecasts, forecasts_ci, unique_id, model_name):
    """Crea gráfica interactiva de la predicción"""
    
    # Filtrar datos para la serie específica
    df_filtered = df[df['unique_id'] == unique_id].copy()
    forecasts_filtered = forecasts[forecasts['unique_id'] == unique_id].copy()
    forecasts_ci_filtered = forecasts_ci[forecasts_ci['unique_id'] == unique_id].copy()
    
    # Crear figura
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=df_filtered['ds'],
        y=df_filtered['y'],
        mode='lines',
        name='Datos Históricos',
        line=dict(color='blue')
    ))
    
    # Predicciones
    if model_name in forecasts_filtered.columns:
        fig.add_trace(go.Scatter(
            x=forecasts_filtered['ds'],
            y=forecasts_filtered[model_name],
            mode='lines',
            name=f'Predicción {model_name}',
            line=dict(color='red', dash='dash')
        ))
        
        # Intervalos de confianza si están disponibles
        if f'{model_name}-lo-95' in forecasts_ci_filtered.columns:
            fig.add_trace(go.Scatter(
                x=forecasts_ci_filtered['ds'],
                y=forecasts_ci_filtered[f'{model_name}-hi-95'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecasts_ci_filtered['ds'],
                y=forecasts_ci_filtered[f'{model_name}-lo-95'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255,0,0,0.2)',
                fill='tonexty',
                name='IC 95%',
                hoverinfo='skip'
            ))
    
    # Configurar diseño
    fig.update_layout(
        title=f'Predicción para {unique_id} - Modelo: {model_name}',
        xaxis_title='Fecha',
        yaxis_title='Valor',
        hovermode='x unified',
        height=500
    )
    
    return fig

# Interfaz principal
def main():
    # Opción para cargar datos
    data_option = st.sidebar.radio(
        "Fuente de datos:",
        ["Usar datos de ejemplo", "Cargar archivo CSV"]
    )
    
    df = None
    
    if data_option == "Usar datos de ejemplo":
        df = generate_sample_data()
        st.sidebar.success("Datos de ejemplo cargados")
        
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Cargar archivo CSV",
            type=['csv'],
            help="El archivo debe tener columnas: 'ds' (fechas) y 'y' (valores)"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("Archivo cargado exitosamente")
            except Exception as e:
                st.sidebar.error(f"Error al cargar archivo: {str(e)}")
    
    if df is not None:
        # Preparar datos
        df_prepared = prepare_data(df)
        
        if df_prepared is not None:
            # Mostrar información de los datos
            st.subheader("📊 Información de los Datos")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Número de observaciones", len(df_prepared))
            with col2:
                st.metric("Fecha inicial", df_prepared['ds'].min().strftime('%Y-%m-%d'))
            with col3:
                st.metric("Fecha final", df_prepared['ds'].max().strftime('%Y-%m-%d'))
            
            # Mostrar vista previa de datos
            with st.expander("Vista previa de los datos"):
                st.dataframe(df_prepared.head(10))
            
            # Configuración de predicción
            st.subheader("🔮 Configuración de Predicción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                horizon = st.number_input('Horizonte de predicción (semanas)', 
                                          min_value=1, value=12, 
                                          step=1)
                #horizon = st.slider(
                    #"Horizonte de predicción (días)",
                    #min_value=1,
                    #max_value=365,
                    #value=30
                #)
            
            with col2:
                models_available = ['AutoARIMA', 'AutoETS', 'SeasonalNaive']
                models_selected = st.multiselect(
                    "Seleccionar modelos",
                    models_available,
                    default=['AutoARIMA']
                )
            
            # Botón para generar predicciones
            if st.button("🚀 Generar Predicciones", type="primary"):
                if not models_selected:
                    st.error("Selecciona al menos un modelo")
                else:
                    with st.spinner("Generando predicciones..."):
                        forecasts, forecasts_ci, sf = create_forecast(
                            df_prepared, horizon, models_selected
                        )
                        
                        if forecasts is not None:
                            st.success("Predicciones generadas exitosamente!")
                            
                            # Obtener series únicas
                            unique_ids = df_prepared['unique_id'].unique()
                            
                            # Crear tabs para cada serie
                            if len(unique_ids) == 1:
                                unique_id = unique_ids[0]
                                
                                # Crear tabs para cada modelo
                                tabs = st.tabs(models_selected)
                                
                                for i, model_name in enumerate(models_selected):
                                    with tabs[i]:
                                        fig = create_forecast_plot(
                                            df_prepared, forecasts, forecasts_ci, 
                                            unique_id, model_name
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Mostrar métricas de la predicción
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            forecast_mean = forecasts[forecasts['unique_id'] == unique_id][model_name].mean()
                                            st.metric("Promedio de predicción", f"{forecast_mean:.2f}")
                                        with col2:
                                            forecast_std = forecasts[forecasts['unique_id'] == unique_id][model_name].std()
                                            st.metric("Desviación estándar", f"{forecast_std:.2f}")
                            
                            else:
                                # Para múltiples series, crear selectbox
                                selected_series = st.selectbox(
                                    "Seleccionar serie:",
                                    unique_ids
                                )
                                
                                tabs = st.tabs(models_selected)
                                
                                for i, model_name in enumerate(models_selected):
                                    with tabs[i]:
                                        fig = create_forecast_plot(
                                            df_prepared, forecasts, forecasts_ci, 
                                            selected_series, model_name
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # Opción para descargar predicciones
                            st.subheader("📥 Descargar Resultados")
                            csv = forecasts.to_csv(index=False)
                            st.download_button(
                                label="Descargar predicciones CSV",
                                data=csv,
                                file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

if __name__ == "__main__":
    main()

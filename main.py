import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
from io import StringIO

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Sismos en México",
    page_icon="🌋",
    layout="wide"
)

# URL del archivo CSV en GitHub (asegúrate de usar la URL raw)
CSV_URL = "https://raw.githubusercontent.com/tu_usuario/tu_repositorio/main/datos_sismos.csv"

# Título de la aplicación
st.title("🌍 Análisis de Sismos en México")
st.markdown("""
Esta aplicación analiza datos históricos de sismos en México, mostrando estadísticas, gráficas y predicciones.
""")

# --- Funciones (se mantienen casi igual que en el código original) ---
def clasificar_sismo(magnitud):
    """Clasifica un sismo según su magnitud"""
    if pd.isna(magnitud):
        return 'Desconocido'
    magnitud = float(magnitud)
    if magnitud < 3.0:
        return 'Micro (<3.0)'
    elif 3.0 <= magnitud < 4.0:
        return 'Menor (3.0-3.9)'
    elif 4.0 <= magnitud < 5.0:
        return 'Ligero (4.0-4.9)'
    elif 5.0 <= magnitud < 6.0:
        return 'Moderado (5.0-5.9)'
    elif 6.0 <= magnitud < 7.0:
        return 'Fuerte (6.0-6.9)'
    elif 7.0 <= magnitud < 8.0:
        return 'Mayor (7.0-7.9)'
    else:
        return 'Gran terremoto (≥8.0)'

def descargar_csv_desde_github():
    """Descarga el archivo CSV desde GitHub y lo carga en un DataFrame"""
    try:
        response = requests.get(CSV_URL)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        df = pd.read_csv(
            csv_data,
            encoding='utf-8',
            sep=',',
            parse_dates={'Fecha_Hora': ['Fecha', 'Hora']},
            dayfirst=True,
            dtype={
                'Magnitud': float,
                'Latitud': float,
                'Longitud': float,
                'Profundidad': float
            },
            na_values=['no calculable'],
            skipinitialspace=True
        )
        return df
    except Exception as e:
        st.error(f"Error al descargar o procesar el archivo CSV: {str(e)}")
        return None

def procesar_archivo(df):
    """Procesa el DataFrame de sismos"""
    try:
        # Verificar columnas esenciales
        required_columns = ['Fecha_Hora', 'Magnitud', 'Latitud', 'Longitud']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Faltan columnas esenciales: {', '.join(missing_cols)}")
            return None
        
        # Limpieza de datos
        df = df.dropna(subset=['Magnitud', 'Latitud', 'Longitud'])
        
        # Convertir profundidad si existe
        if 'Profundidad' in df.columns:
            df['Profundidad'] = pd.to_numeric(df['Profundidad'], errors='coerce')
        
        # Extraer año y mes para filtrado
        df['Año'] = df['Fecha_Hora'].dt.year
        df['Mes'] = df['Fecha_Hora'].dt.month
        df['Hora'] = df['Fecha_Hora'].dt.hour
        
        # Añadir clasificación
        df['Clasificacion'] = df['Magnitud'].apply(clasificar_sismo)
        
        return df
    except Exception as e:
        st.error(f"Error al procesar archivo: {str(e)}")
        return None

# --- Funciones para gráficas y estadísticas (similares al original) ---
def generar_graficas(df, año_inicio=None, año_fin=None):
    """Genera gráficas interactivas con Plotly"""
    if año_inicio and año_fin:
        df_filtrado = df[(df['Año'] >= int(año_inicio)) & (df['Año'] <= int(año_fin))]
        titulo_rango = f" ({año_inicio}-{año_fin})"
    else:
        df_filtrado = df
        titulo_rango = " (Todos los años)"

    # 1. Histograma de magnitudes
    fig_hist = px.histogram(
        df_filtrado, x='Magnitud', 
        title=f'Distribución de Magnitudes{titulo_rango}',
        labels={'Magnitud': 'Magnitud (Richter)'},
        nbins=30,
        color_discrete_sequence=['#EF553B']
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # 2. Mapa de sismos
    fig_map = px.scatter_geo(
        df_filtrado, 
        lat='Latitud', 
        lon='Longitud',
        color='Magnitud',
        size='Magnitud',
        title=f'Ubicación de Sismos{titulo_rango}',
        hover_name='Referencia de localizacion' if 'Referencia de localizacion' in df.columns else None,
        projection='natural earth',
        scope='north america'
    )
    st.plotly_chart(fig_map, use_container_width=True)

def generar_estadisticas(df, año_inicio=None, año_fin=None):
    """Muestra estadísticas descriptivas en Streamlit"""
    if año_inicio and año_fin:
        df_filtrado = df[(df['Año'] >= int(año_inicio)) & (df['Año'] <= int(año_fin))]
    else:
        df_filtrado = df

    # Mostrar métricas clave
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de sismos", len(df_filtrado))
    col2.metric("Magnitud máxima", round(df_filtrado['Magnitud'].max(), 1))
    col3.metric("Magnitud promedio", round(df_filtrado['Magnitud'].mean(), 1))

    # Mostrar tabla de clasificación
    st.subheader("Clasificación de sismos")
    st.table(df_filtrado['Clasificacion'].value_counts().reset_index().rename(
        columns={'index': 'Clasificación', 'Clasificacion': 'Número de sismos'}))

# --- Interfaz de Streamlit ---
def main():
    # Descargar y procesar datos
    df = descargar_csv_desde_github()
    if df is not None:
        df = procesar_archivo(df)
    
    if df is not None:
        # Selector de rango de años
        años_disponibles = sorted(df['Año'].unique())
        año_inicio = st.sidebar.selectbox(
            "Año de inicio",
            años_disponibles,
            index=0
        )
        año_fin = st.sidebar.selectbox(
            "Año final",
            años_disponibles,
            index=len(años_disponibles) - 1
        )

        # Validar rango
        if año_inicio > año_fin:
            st.error("El año de inicio debe ser menor o igual al año final.")
        else:
            # Mostrar gráficas y estadísticas
            generar_graficas(df, año_inicio, año_fin)
            generar_estadisticas(df, año_inicio, año_fin)

if __name__ == '__main__':
    main()

from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import tempfile
import logging
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# Crear la aplicación Flask
app = Flask(__name__)
app.secret_key = 'clave_secreta_sismica'

# Configuración
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
ALLOWED_EXTENSIONS = {'csv'}

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Verifica si la extensión del archivo está permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def procesar_archivo(file_path):
    """Procesa el archivo CSV de sismos"""
    try:
        # Leer el archivo CSV
        df = pd.read_csv(
            file_path,
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
        
        # Verificar columnas esenciales
        required_columns = ['Fecha_Hora', 'Magnitud', 'Latitud', 'Longitud']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas esenciales: {', '.join(missing_cols)}")
        
        # Limpieza de datos
        df = df.dropna(subset=['Magnitud', 'Latitud', 'Longitud'])
        
        # Convertir profundidad si existe
        if 'Profundidad' in df.columns:
            df['Profundidad'] = pd.to_numeric(df['Profundidad'], errors='coerce')
        
        # Extraer año para filtrado
        df['Año'] = df['Fecha_Hora'].dt.year
        
        # Añadir clasificación
        df['Clasificacion'] = df['Magnitud'].apply(clasificar_sismo)
        
        return df
    
    except Exception as e:
        logger.error(f"Error al procesar archivo: {str(e)}")
        raise

def generar_predicciones(df):
    """Genera predicciones simples basadas en datos históricos"""
    try:
        # Agrupar por año y contar sismos
        sismos_por_año = df.groupby('Año').size().reset_index(name='Conteo')
        
        # Preparar datos para modelo
        X = sismos_por_año['Año'].values.reshape(-1, 1)
        y = sismos_por_año['Conteo'].values
        
        # Entrenar modelo lineal simple
        modelo = LinearRegression()
        modelo.fit(X, y)
        
        # Predecir para los próximos 3 años
        ultimo_año = sismos_por_año['Año'].max()
        años_futuros = np.array([ultimo_año + 1, ultimo_año + 2, ultimo_año + 3]).reshape(-1, 1)
        predicciones = modelo.predict(años_futuros).astype(int)
        
        # Calcular tendencia
        tendencia = "aumentando" if modelo.coef_[0] > 0 else "disminuyendo"
        
        return {
            'tendencia': tendencia,
            'predicciones': {
                año: pred for año, pred in zip(años_futuros.flatten(), predicciones)
            },
            'coeficiente': round(float(modelo.coef_[0]), 2)
        }
        
    except Exception as e:
        logger.error(f"Error al generar predicciones: {str(e)}")
        return None

def analizar_patrones_temporales(df):
    """Analiza patrones temporales en los sismos"""
    try:
        # Extraer mes y hora
        df['Mes'] = df['Fecha_Hora'].dt.month
        df['Hora'] = df['Fecha_Hora'].dt.hour
        
        # Meses con más sismos
        meses_sismos = df['Mes'].value_counts().head(3).index.tolist()
        meses_nombres = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                        'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        meses_top = [meses_nombres[m-1] for m in meses_sismos]
        
        # Horas con más sismos
        horas_sismos = df['Hora'].value_counts().head(3).index.tolist()
        
        return {
            'meses_mas_sismos': meses_top,
            'horas_mas_sismos': horas_sismos,
            'sismos_noche': len(df[(df['Hora'] >= 22) | (df['Hora'] <= 6)]) / len(df) * 100
        }
        
    except Exception as e:
        logger.error(f"Error al analizar patrones temporales: {str(e)}")
        return None

def generar_graficas(df, año_inicio=None, año_fin=None):
    """Genera gráficas interactivas con filtrado por años"""
    try:
        # Filtrar por rango de años si se especifica
        if año_inicio and año_fin:
            df_filtrado = df[(df['Año'] >= int(año_inicio)) & (df['Año'] <= int(año_fin))]
            titulo_rango = f" ({año_inicio}-{año_fin})"
        else:
            df_filtrado = df
            titulo_rango = " (Todos los años)"
        
        graficas = {}
        
        # 1. Histograma de magnitudes
        fig_hist = px.histogram(
            df_filtrado, x='Magnitud', 
            title=f'Distribución de Magnitudes{titulo_rango}',
            labels={'Magnitud': 'Magnitud (Richter)'},
            nbins=30,
            color_discrete_sequence=['#EF553B'],
            category_orders={"Clasificacion": [
                'Micro (<3.0)', 'Menor (3.0-3.9)', 'Ligero (4.0-4.9)', 
                'Moderado (5.0-5.9)', 'Fuerte (6.0-6.9)', 'Mayor (7.0-7.9)', 
                'Gran terremoto (≥8.0)'
            ]}
        )
        fig_hist.update_layout(
            bargap=0.1,
            xaxis_title="Magnitud",
            yaxis_title="Número de sismos",
            plot_bgcolor='rgba(240,240,240,1)'
        )
        graficas['histograma'] = fig_hist.to_html(full_html=False)
        
        # 2. Serie temporal
        fig_ts = px.scatter(
            df_filtrado, x='Fecha_Hora', y='Magnitud',
            title=f'Sismos por Fecha{titulo_rango}',
            labels={'Fecha_Hora': 'Fecha', 'Magnitud': 'Magnitud (Richter)'},
            color='Magnitud',
            color_continuous_scale='reds',
            hover_data=['Referencia de localizacion'] if 'Referencia de localizacion' in df.columns else None
        )
        fig_ts.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Magnitud",
            hovermode='x unified',
            plot_bgcolor='rgba(240,240,240,1)'
        )
        graficas['serie_temporal'] = fig_ts.to_html(full_html=False)
        
        # 3. Mapa de sismos (versión mejorada)
        fig_map = px.scatter_geo(
            df_filtrado, 
            lat='Latitud', 
            lon='Longitud',
            color='Magnitud',
            size='Magnitud',
            title=f'Ubicación de Sismos{titulo_rango}',
            hover_name='Referencia de localizacion' if 'Referencia de localizacion' in df.columns else None,
            projection='natural earth',
            color_continuous_scale='reds',
            scope='north america',
            height=600  # Aumentamos la altura del mapa
        )
        
        # Configuración específica para México
        fig_map.update_geos(
            showcountries=True, 
            countrycolor="Black",
            showsubunits=True, 
            subunitcolor="grey",
            resolution=50,  # Mayor resolución
            showcoastlines=True,
            coastlinecolor="RebeccaPurple",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue",
            showlakes=True,
            lakecolor="blue"
        )
        
        # Ajustar la vista centrada en México con zoom
        fig_map.update_layout(
            geo=dict(
                center=dict(lat=23, lon=-102),
                lataxis_range=[14, 33],  # Ajuste de latitud para México
                lonaxis_range=[-118, -86],  # Ajuste de longitud para México
                projection_scale=5  # Zoom más cercano
            ),
            margin=dict(l=0, r=0, t=40, b=0)  # Márgenes ajustados
        )
        
        # Añadir controles de zoom mejorados
        fig_map.update_layout(
            dragmode="zoom",
            mapbox_style="open-street-map",
            hovermode="closest"
        )
        
        graficas['mapa'] = fig_map.to_html(full_html=False, config={
            'responsive': True,
            'displayModeBar': True
        })
        
        # 4. Sismos por año (comparativa)
        if año_inicio and año_fin:
            df_comparativa = df[(df['Año'] >= int(año_inicio)-5) & (df['Año'] <= int(año_fin)+5)]
            sismos_por_año = df_comparativa.groupby('Año').size().reset_index(name='Conteo')
            
            fig_comparativa = px.bar(
                sismos_por_año, 
                x='Año', 
                y='Conteo',
                title=f'Comparativa de Sismos por Año (Rango seleccionado: {año_inicio}-{año_fin})',
                labels={'Año': 'Año', 'Conteo': 'Número de sismos'},
                color='Conteo',
                color_continuous_scale='reds'
            )
            fig_comparativa.update_layout(
                xaxis_title="Año",
                yaxis_title="Número de sismos",
                plot_bgcolor='rgba(240,240,240,1)'
            )
            # Resaltar el rango seleccionado
            fig_comparativa.add_vrect(
                x0=int(año_inicio)-0.5, x1=int(año_fin)+0.5,
                fillcolor="yellow", opacity=0.2,
                line_width=0
            )
            graficas['comparativa'] = fig_comparativa.to_html(full_html=False)
        
        return graficas
    
    except Exception as e:
        logger.error(f"Error al generar gráficas: {str(e)}")
        raise

def generar_estadisticas(df, año_inicio=None, año_fin=None):
    """Genera estadísticas descriptivas y análisis predictivos"""
    try:
        # Filtrar por rango de años si se especifica
        if año_inicio and año_fin:
            df_filtrado = df[(df['Año'] >= int(año_inicio)) & (df['Año'] <= int(año_fin))]
        else:
            df_filtrado = df
        
        stats = {
            'total_sismos': len(df_filtrado),
            'periodo': f"{df_filtrado['Fecha_Hora'].min().strftime('%Y')}-{df_filtrado['Fecha_Hora'].max().strftime('%Y')}",
            'magnitud_maxima': round(df_filtrado['Magnitud'].max(), 1),
            'magnitud_promedio': round(df_filtrado['Magnitud'].mean(), 1),
            'sismos_fuertes': len(df_filtrado[df_filtrado['Magnitud'] >= 6]),
            'ultimo_sismo': {
                'fecha': df_filtrado['Fecha_Hora'].max().strftime('%d/%m/%Y %H:%M'),
                'magnitud': round(df_filtrado.loc[df_filtrado['Fecha_Hora'].idxmax(), 'Magnitud'], 1),
                'ubicacion': df_filtrado.loc[df_filtrado['Fecha_Hora'].idxmax(), 'Referencia de localizacion'] 
                             if 'Referencia de localizacion' in df_filtrado.columns else "No disponible"
            },
            'clasificacion': df_filtrado['Clasificacion'].value_counts().to_dict()
        }
        
        if 'Profundidad' in df_filtrado.columns:
            stats.update({
                'profundidad_maxima': round(df_filtrado['Profundidad'].max(), 1),
                'profundidad_promedio': round(df_filtrado['Profundidad'].mean(), 1),
                'sismos_superficiales': len(df_filtrado[df_filtrado['Profundidad'] < 30]) / len(df_filtrado) * 100
            })
        
        # Añadir análisis predictivos si hay suficientes datos
        if len(df_filtrado) > 10 and len(df_filtrado['Año'].unique()) > 3:
            stats['predicciones'] = generar_predicciones(df_filtrado)
            stats['patrones_temporales'] = analizar_patrones_temporales(df_filtrado)
        
        return stats
    
    except Exception as e:
        logger.error(f"Error al generar estadísticas: {str(e)}")
        return {}

@app.route('/', methods=['GET', 'POST'])
def index():
    current_year = datetime.now().year
    if request.method == 'POST':
        try:
            # Verificar archivo
            if 'file' not in request.files:
                return render_template('index.html', error="No se seleccionó ningún archivo", current_year=current_year)
            
            file = request.files['file']
            
            if file.filename == '':
                return render_template('index.html', error="Por favor seleccione un archivo CSV", current_year=current_year)
            
            if not allowed_file(file.filename):
                return render_template('index.html', error="Solo se permiten archivos CSV", current_year=current_year)
            
            # Obtener años para filtrado
            año_inicio = request.form.get('año_inicio')
            año_fin = request.form.get('año_fin')
            
            # Validar años
            if año_inicio and año_fin:
                try:
                    if int(año_inicio) > int(año_fin):
                        return render_template('index.html', error="El año de inicio debe ser menor al año final", current_year=current_year)
                except ValueError:
                    return render_template('index.html', error="Los años deben ser números enteros", current_year=current_year)
            
            # Procesar archivo temporal
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)
            
            try:
                # Procesar archivo
                df = procesar_archivo(temp_path)
                
                # Validar rango de años
                if año_inicio and año_fin:
                    min_year = df['Año'].min()
                    max_year = df['Año'].max()
                    if int(año_inicio) < min_year or int(año_fin) > max_year:
                        return render_template('index.html', 
                            error=f"El archivo contiene datos de {min_year} a {max_year}",
                            current_year=current_year)
                
                # Generar gráficas y estadísticas
                graficas = generar_graficas(df, año_inicio, año_fin)
                stats = generar_estadisticas(df, año_inicio, año_fin)
                
                # Obtener años disponibles para el selector
                años_disponibles = sorted(df['Año'].unique())
                
                return render_template('resultados.html',
                    graficas=graficas,
                    stats=stats,
                    año_inicio=año_inicio,
                    año_fin=año_fin,
                    años_disponibles=años_disponibles,
                    min_year=min(años_disponibles),
                    max_year=max(años_disponibles),
                    current_year=current_year,
                    show_resumen=True)
            
            except Exception as e:
                logger.error(f"Error al procesar archivo: {str(e)}", exc_info=True)
                return render_template('index.html', 
                                     error=f"Error al procesar el archivo: {str(e)}",
                                     current_year=current_year)
            finally:
                # Limpiar archivo temporal
                try:
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                except Exception as e:
                    logger.error(f"Error al limpiar archivos temporales: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error general: {str(e)}", exc_info=True)
            return render_template('index.html', 
                                error=f"Error inesperado: {str(e)}",
                                current_year=current_year)
    
    return render_template('index.html', current_year=current_year)

if __name__ == '__main__':
    # Configuración para producción
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('DEBUG', 'False') == 'True',
        threaded=True  # Mejor manejo de múltiples solicitudes
    )

"""
============================================================================
PROYECTO: SEGMENTACIÓN DE CLIENTES CON K-MEANS
Archivo: config.py
Descripción: Configuración centralizada del proyecto
============================================================================
"""

import os
from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================

# Directorio raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Directorios de resultados
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
REPORTS_DIR = RESULTS_DIR / 'reports'
MODELS_DIR = RESULTS_DIR / 'models'

# Subdirectorios de figuras
EXPLORACION_DIR = FIGURES_DIR / '01_exploracion'
CLUSTERING_DIR = FIGURES_DIR / '02_clustering'
BUSINESS_DIR = FIGURES_DIR / '03_business'

# ============================================================================
# ARCHIVOS DE DATOS
# ============================================================================

# Dataset principal
DATASET_FILE = RAW_DATA_DIR / 'Mall_Customers.csv'

# Archivos de salida
OUTPUT_CSV = PROCESSED_DATA_DIR / 'clientes_segmentados.csv'
STATS_CSV = REPORTS_DIR / 'cluster_statistics.csv'
REPORT_TXT = REPORTS_DIR / 'reporte_segmentacion.txt'

# Modelos guardados
MODEL_FILE = MODELS_DIR / 'kmeans_final.pkl'
SCALER_FILE = MODELS_DIR / 'scaler.pkl'

# ============================================================================
# PARÁMETROS DEL MODELO
# ============================================================================

# K-Means
K_OPTIMO = 5  # Número de clusters (ajustar según método del codo)
RANDOM_STATE = 42  # Para reproducibilidad
N_INIT = 10  # Número de inicializaciones
MAX_ITER = 300  # Máximo de iteraciones

# Método del Codo
K_MIN = 1  # K mínimo a probar
K_MAX = 11  # K máximo a probar

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================

# Tamaño de figuras
FIGURE_SIZE = (14, 9)
FIGURE_SIZE_SMALL = (10, 6)
FIGURE_SIZE_LARGE = (16, 10)

# DPI para guardar imágenes
DPI = 300

# Colores
COLOR_PALETTE = 'husl'  # Paleta de colores para clusters

# Estilo de gráficos
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# ============================================================================
# COLUMNAS DEL DATASET
# ============================================================================

# Columnas originales del dataset
COL_CUSTOMER_ID = 'CustomerID'
COL_GENDER = 'Gender'
COL_AGE = 'Age'
COL_INCOME = 'Annual Income (k$)'
COL_SPENDING = 'Spending Score (1-100)'

# Columnas creadas
COL_GENDER_ENCODED = 'Gender_Encoded'
COL_CLUSTER = 'Cluster'
COL_CLUSTER_NAME = 'Cluster_Nombre'

# Features para clustering (modelo simple - 2D)
FEATURES_SIMPLE = [COL_INCOME, COL_SPENDING]

# Features para clustering (modelo completo)
FEATURES_COMPLETO = [COL_AGE, COL_GENDER_ENCODED, COL_INCOME, COL_SPENDING]

# ============================================================================
# PERFILES DE CLUSTERS (Ajustar según resultados)
# ============================================================================

# Estos nombres son ejemplos típicos, ajustar según tus centroides
CLUSTER_PROFILES = {
    0: {
        'nombre': 'Precavidos de Ingresos Medios',
        'descripcion': 'Ingresos medios, spending moderado',
        'caracteristicas': [
            'Ingresos: 50-60k$',
            'Spending: 40-50',
            'Edad: Variada'
        ]
    },
    1: {
        'nombre': 'VIP High Spenders',
        'descripcion': 'Altos ingresos, alto spending',
        'caracteristicas': [
            'Ingresos: 70-90k$',
            'Spending: 75-90',
            'Edad: 30-45'
        ]
    },
    2: {
        'nombre': 'Conservadores de Alto Ingreso',
        'descripcion': 'Altos ingresos, bajo spending',
        'caracteristicas': [
            'Ingresos: 70-90k$',
            'Spending: 10-25',
            'Edad: 40-60'
        ]
    },
    3: {
        'nombre': 'Jóvenes Gastadores',
        'descripcion': 'Bajos ingresos, alto spending',
        'caracteristicas': [
            'Ingresos: 20-40k$',
            'Spending: 70-90',
            'Edad: 18-30'
        ]
    },
    4: {
        'nombre': 'Oportunidad de Crecimiento',
        'descripcion': 'Bajos ingresos, bajo spending',
        'caracteristicas': [
            'Ingresos: 20-40k$',
            'Spending: 10-30',
            'Edad: Variada'
        ]
    }
}

# ============================================================================
# ESTRATEGIAS DE MARKETING
# ============================================================================

MARKETING_STRATEGIES = {
    'VIP High Spenders': """
    🎯 ESTRATEGIA:
       • Programa de lealtad PREMIUM con beneficios exclusivos
       • Acceso temprano a lanzamientos y colecciones limitadas
       • Personal shopper y servicios de atención personalizada
       • Eventos VIP y experiencias únicas
       • Comunicación directa vía email/WhatsApp
    
    💰 INVERSIÓN: Alta
    📈 POTENCIAL: Muy Alto (son tu base más rentable)
    """,
    
    'Jóvenes Gastadores': """
    🎯 ESTRATEGIA:
       • Marketing en redes sociales (Instagram, TikTok)
       • Influencer marketing con microinfluencers
       • Descuentos por volumen y bundles
       • Programa de referidos con incentivos
       • Gamificación (puntos, badges, desafíos)
    
    💰 INVERSIÓN: Media
    📈 POTENCIAL: Alto (pueden convertirse en VIP futuro)
    """,
    
    'Conservadores de Alto Ingreso': """
    🎯 ESTRATEGIA:
       • Demostración de VALOR y CALIDAD sobre precio
       • Testimonios y casos de éxito
       • Garantías extendidas y políticas de devolución generosas
       • Marketing educativo (webinars, guías, comparativas)
       • Email marketing con contenido de valor
    
    💰 INVERSIÓN: Media-Alta
    📈 POTENCIAL: Alto (tienen el dinero, falta convencerlos)
    """,
    
    'Oportunidad de Crecimiento': """
    🎯 ESTRATEGIA:
       • Productos de entrada (loss leaders)
       • Línea económica de calidad
       • Programas de financiamiento y pagos flexibles
       • Descuentos por primera compra
       • Comunicación por WhatsApp y SMS (bajo costo)
    
    💰 INVERSIÓN: Baja
    📈 POTENCIAL: Medio (volumen vs. margen)
    """,
    
    'Precavidos de Ingresos Medios': """
    🎯 ESTRATEGIA:
       • Promociones estacionales y flash sales
       • Descuentos por cantidad (2x1, 3x2)
       • Productos de rango medio con buen value
       • Programa de puntos acumulables
       • Email marketing con ofertas personalizadas
    
    💰 INVERSIÓN: Media
    📈 POTENCIAL: Medio-Alto (segmento estable)
    """
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_directorios():
    """Crea todos los directorios necesarios del proyecto."""
    directorios = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXPLORACION_DIR,
        CLUSTERING_DIR,
        BUSINESS_DIR,
        REPORTS_DIR,
        MODELS_DIR
    ]
    
    for directorio in directorios:
        directorio.mkdir(parents=True, exist_ok=True)
    
    print("✅ Directorios creados correctamente")


def verificar_dataset():
    """Verifica que el dataset existe."""
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"\n❌ ERROR: No se encuentra el archivo {DATASET_FILE}\n"
            f"Por favor, descarga 'Mall_Customers.csv' de Kaggle y colócalo en:\n"
            f"{RAW_DATA_DIR}\n"
        )
    print(f"✅ Dataset encontrado: {DATASET_FILE}")


def get_cluster_name(cluster_id):
    """Obtiene el nombre de un cluster por su ID."""
    return CLUSTER_PROFILES.get(cluster_id, {}).get('nombre', f'Cluster {cluster_id}')


if __name__ == '__main__':
    print("=" * 80)
    print("CONFIGURACIÓN DEL PROYECTO")
    print("=" * 80)
    
    print("\nDirectorio base:", BASE_DIR)
    print("Directorio de datos:", DATA_DIR)
    print("Directorio de resultados:", RESULTS_DIR)
    
    print("\nCreando directorios...")
    crear_directorios()
    
    print("\nVerificando dataset...")
    try:
        verificar_dataset()
    except FileNotFoundError as e:
        print(e)
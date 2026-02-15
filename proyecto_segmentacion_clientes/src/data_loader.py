"""
============================================================================
PROYECTO: SEGMENTACIÓN DE CLIENTES CON K-MEANS
Archivo: data_loader.py
Descripción: Funciones para cargar y validar datos
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Importar configuración
from config import (
    DATASET_FILE, 
    COL_CUSTOMER_ID, 
    COL_GENDER, 
    COL_AGE, 
    COL_INCOME, 
    COL_SPENDING
)


def cargar_dataset(filepath=None):
    """
    Carga el dataset de clientes.
    
    Parámetros:
    -----------
    filepath : str o Path, opcional
        Ruta al archivo CSV. Si es None, usa la ruta por defecto de config.py
        
    Retorna:
    --------
    df : pandas.DataFrame
        DataFrame con los datos cargados
    """
    if filepath is None:
        filepath = DATASET_FILE
    
    print("=" * 80)
    print("CARGANDO DATASET")
    print("=" * 80)
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset cargado exitosamente")
        print(f"   Archivo: {filepath}")
        print(f"   Filas: {len(df)}")
        print(f"   Columnas: {len(df.columns)}")
        return df
    
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encuentra el archivo {filepath}")
        print("\nPor favor:")
        print("1. Descarga 'Mall_Customers.csv' de Kaggle:")
        print("   https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python")
        print(f"2. Colócalo en: {DATASET_FILE.parent}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR al cargar el archivo: {e}")
        sys.exit(1)


def validar_dataset(df):
    """
    Valida que el dataset tenga la estructura correcta.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame a validar
        
    Retorna:
    --------
    bool
        True si el dataset es válido, False en caso contrario
    """
    print("\n" + "=" * 80)
    print("VALIDANDO DATASET")
    print("=" * 80)
    
    # Columnas esperadas
    columnas_esperadas = [
        COL_CUSTOMER_ID,
        COL_GENDER,
        COL_AGE,
        COL_INCOME,
        COL_SPENDING
    ]
    
    # Verificar columnas
    columnas_faltantes = set(columnas_esperadas) - set(df.columns)
    if columnas_faltantes:
        print(f"❌ ERROR: Faltan columnas: {columnas_faltantes}")
        return False
    
    print("✅ Todas las columnas están presentes")
    
    # Verificar valores nulos
    nulos = df.isnull().sum()
    if nulos.sum() > 0:
        print(f"⚠️  ADVERTENCIA: Se encontraron {nulos.sum()} valores nulos:")
        print(nulos[nulos > 0])
    else:
        print("✅ No hay valores nulos")
    
    # Verificar tipos de datos
    print("\n📊 Tipos de datos:")
    print(df.dtypes)
    
    # Verificar rangos de valores
    print("\n📈 Rangos de valores:")
    print(f"   Edad: {df[COL_AGE].min()} - {df[COL_AGE].max()}")
    print(f"   Ingresos: {df[COL_INCOME].min()} - {df[COL_INCOME].max()} k$")
    print(f"   Spending Score: {df[COL_SPENDING].min()} - {df[COL_SPENDING].max()}")
    
    # Verificar distribución de género
    print(f"\n👥 Distribución de género:")
    print(df[COL_GENDER].value_counts())
    
    return True


def mostrar_info_dataset(df):
    """
    Muestra información detallada del dataset.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame a analizar
    """
    print("\n" + "=" * 80)
    print("INFORMACIÓN DEL DATASET")
    print("=" * 80)
    
    print("\n📋 Primeras 5 filas:")
    print(df.head())
    
    print("\n📊 Información general:")
    print(df.info())
    
    print("\n📈 Estadísticas descriptivas:")
    print(df.describe())
    
    print("\n💾 Uso de memoria:")
    print(f"   {df.memory_usage(deep=True).sum() / 1024:.2f} KB")


def detectar_outliers(df, columna, metodo='iqr'):
    """
    Detecta outliers en una columna usando el método IQR.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos
    columna : str
        Nombre de la columna a analizar
    metodo : str, opcional
        Método para detectar outliers ('iqr' o 'zscore')
        
    Retorna:
    --------
    outliers : pandas.Series
        Serie booleana indicando qué filas son outliers
    """
    if metodo == 'iqr':
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = (df[columna] < limite_inferior) | (df[columna] > limite_superior)
        
        n_outliers = outliers.sum()
        if n_outliers > 0:
            print(f"\n⚠️  {n_outliers} outliers detectados en '{columna}' (método IQR)")
            print(f"   Límite inferior: {limite_inferior:.2f}")
            print(f"   Límite superior: {limite_superior:.2f}")
        else:
            print(f"\n✅ No se detectaron outliers en '{columna}'")
        
        return outliers
    
    elif metodo == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[columna]))
        outliers = z_scores > 3
        
        n_outliers = outliers.sum()
        if n_outliers > 0:
            print(f"\n⚠️  {n_outliers} outliers detectados en '{columna}' (método Z-Score)")
        else:
            print(f"\n✅ No se detectaron outliers en '{columna}'")
        
        return outliers


def generar_dataset_sintetico(n_clientes=200, guardar=False):
    """
    Genera un dataset sintético similar al Mall Customers dataset.
    Útil si no puedes descargar de Kaggle.
    
    Parámetros:
    -----------
    n_clientes : int
        Número de clientes a generar
    guardar : bool
        Si True, guarda el dataset en data/raw/
        
    Retorna:
    --------
    df : pandas.DataFrame
        Dataset sintético generado
    """
    print("\n" + "=" * 80)
    print("GENERANDO DATASET SINTÉTICO")
    print("=" * 80)
    
    np.random.seed(42)
    
    data = {
        COL_CUSTOMER_ID: range(1, n_clientes + 1),
        COL_GENDER: np.random.choice(['Male', 'Female'], n_clientes),
        COL_AGE: np.random.randint(18, 70, n_clientes),
        COL_INCOME: np.random.randint(15, 140, n_clientes),
        COL_SPENDING: np.random.randint(1, 100, n_clientes)
    }
    
    df = pd.DataFrame(data)
    
    print(f"✅ Dataset sintético generado: {n_clientes} clientes")
    
    if guardar:
        DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_FILE, index=False)
        print(f"✅ Dataset guardado en: {DATASET_FILE}")
    
    return df


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal para probar el módulo."""
    # Cargar dataset
    df = cargar_dataset()
    
    # Validar dataset
    if validar_dataset(df):
        print("\n✅ Dataset validado correctamente")
    
    # Mostrar información
    mostrar_info_dataset(df)
    
    # Detectar outliers
    print("\n" + "=" * 80)
    print("DETECCIÓN DE OUTLIERS")
    print("=" * 80)
    
    detectar_outliers(df, COL_AGE)
    detectar_outliers(df, COL_INCOME)
    detectar_outliers(df, COL_SPENDING)


if __name__ == '__main__':
    main()
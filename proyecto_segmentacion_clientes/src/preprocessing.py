"""
============================================================================
PROYECTO: SEGMENTACIÓN DE CLIENTES CON K-MEANS
Archivo: preprocessing.py
Descripción: Funciones para preprocesar datos antes del clustering
============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Importar configuración
from config import (
    COL_GENDER,
    COL_GENDER_ENCODED,
    COL_AGE,
    COL_INCOME,
    COL_SPENDING,
    FEATURES_SIMPLE,
    FEATURES_COMPLETO,
    SCALER_FILE
)


class PreprocessorClientes:
    """
    Clase para preprocesar datos de clientes.
    """
    
    def __init__(self):
        """Inicializa el preprocesador."""
        self.label_encoder = LabelEncoder()
        self.scaler_simple = StandardScaler()
        self.scaler_completo = StandardScaler()
        self.df_processed = None
        self.X_simple = None
        self.X_completo = None
        self.X_simple_scaled = None
        self.X_completo_scaled = None
    
    
    def codificar_genero(self, df):
        """
        Codifica la variable categórica 'Gender' a numérica.
        
        Parámetros:
        -----------
        df : pandas.DataFrame
            DataFrame con columna 'Gender'
            
        Retorna:
        --------
        df : pandas.DataFrame
            DataFrame con columna 'Gender_Encoded' añadida
        """
        print("\n" + "=" * 80)
        print("CODIFICANDO VARIABLE CATEGÓRICA")
        print("=" * 80)
        
        # Crear copia para no modificar original
        df = df.copy()
        
        # Codificar género
        df[COL_GENDER_ENCODED] = self.label_encoder.fit_transform(df[COL_GENDER])
        
        print("\n✅ Variable 'Gender' codificada:")
        print(df[[COL_GENDER, COL_GENDER_ENCODED]].drop_duplicates().sort_values(COL_GENDER))
        
        return df
    
    
    def seleccionar_features(self, df, tipo='simple'):
        """
        Selecciona las features para el modelo.
        
        Parámetros:
        -----------
        df : pandas.DataFrame
            DataFrame con los datos
        tipo : str
            'simple' para modelo 2D (Ingresos + Spending)
            'completo' para modelo multidimensional (todas las features)
            
        Retorna:
        --------
        X : numpy.ndarray
            Array con las features seleccionadas
        """
        print("\n" + "=" * 80)
        print("SELECCIÓN DE FEATURES")
        print("=" * 80)
        
        if tipo == 'simple':
            features = FEATURES_SIMPLE
            print(f"\n📊 Modelo SIMPLE (2D) - Features seleccionadas:")
        else:
            features = FEATURES_COMPLETO
            print(f"\n📊 Modelo COMPLETO - Features seleccionadas:")
        
        for i, feature in enumerate(features, 1):
            print(f"   {i}. {feature}")
        
        X = df[features].values
        print(f"\n✅ Shape del array: {X.shape}")
        print(f"   ({X.shape[0]} clientes, {X.shape[1]} features)")
        
        return X
    
    
    def escalar_datos(self, X, tipo='simple', fit=True):
        """
        Escala los datos usando StandardScaler.
        
        Parámetros:
        -----------
        X : numpy.ndarray
            Array con los datos a escalar
        tipo : str
            'simple' o 'completo'
        fit : bool
            Si True, ajusta el scaler. Si False, solo transforma.
            
        Retorna:
        --------
        X_scaled : numpy.ndarray
            Array con los datos escalados
        """
        print("\n" + "=" * 80)
        print("ESCALADO DE DATOS")
        print("=" * 80)
        
        scaler = self.scaler_simple if tipo == 'simple' else self.scaler_completo
        
        if fit:
            X_scaled = scaler.fit_transform(X)
            print("\n✅ Scaler ajustado y datos transformados")
        else:
            X_scaled = scaler.transform(X)
            print("\n✅ Datos transformados con scaler existente")
        
        # Verificar escalado
        print(f"\n📊 Verificación del escalado:")
        print(f"   Media de cada feature: {X_scaled.mean(axis=0)}")
        print(f"   Desviación estándar:   {X_scaled.std(axis=0)}")
        print(f"\n   ✅ Media ≈ 0 y Std ≈ 1 indica escalado correcto")
        
        # Comparación antes/después (primeras 3 filas)
        print(f"\n📋 Comparación (primeras 3 filas):")
        print(f"\n   ANTES del escalado:")
        print(X[:3])
        print(f"\n   DESPUÉS del escalado:")
        print(X_scaled[:3])
        
        return X_scaled
    
    
    def preprocesar_completo(self, df, tipo='simple'):
        """
        Ejecuta todo el pipeline de preprocesamiento.
        
        Parámetros:
        -----------
        df : pandas.DataFrame
            DataFrame con los datos originales
        tipo : str
            'simple' para modelo 2D, 'completo' para todas las features
            
        Retorna:
        --------
        X_scaled : numpy.ndarray
            Datos preprocesados y escalados
        df_processed : pandas.DataFrame
            DataFrame con variables codificadas
        """
        print("\n" + "=" * 80)
        print("PIPELINE DE PREPROCESAMIENTO COMPLETO")
        print("=" * 80)
        
        # 1. Codificar género
        self.df_processed = self.codificar_genero(df)
        
        # 2. Seleccionar features
        if tipo == 'simple':
            self.X_simple = self.seleccionar_features(self.df_processed, tipo='simple')
            # 3. Escalar datos
            self.X_simple_scaled = self.escalar_datos(self.X_simple, tipo='simple', fit=True)
            return self.X_simple_scaled, self.df_processed
        else:
            self.X_completo = self.seleccionar_features(self.df_processed, tipo='completo')
            # 3. Escalar datos
            self.X_completo_scaled = self.escalar_datos(self.X_completo, tipo='completo', fit=True)
            return self.X_completo_scaled, self.df_processed
    
    
    def guardar_scaler(self, tipo='simple'):
        """
        Guarda el scaler en disco para uso posterior.
        
        Parámetros:
        -----------
        tipo : str
            'simple' o 'completo'
        """
        scaler = self.scaler_simple if tipo == 'simple' else self.scaler_completo
        
        SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\n✅ Scaler guardado en: {SCALER_FILE}")
    
    
    def cargar_scaler(self, tipo='simple'):
        """
        Carga el scaler desde disco.
        
        Parámetros:
        -----------
        tipo : str
            'simple' o 'completo'
        """
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        
        if tipo == 'simple':
            self.scaler_simple = scaler
        else:
            self.scaler_completo = scaler
        
        print(f"✅ Scaler cargado desde: {SCALER_FILE}")
    
    
    def transformar_nuevo_cliente(self, nuevo_cliente, tipo='simple'):
        """
        Preprocesa datos de un nuevo cliente para predicción.
        
        Parámetros:
        -----------
        nuevo_cliente : dict o list
            Datos del nuevo cliente
            Si es dict: {'Annual Income (k$)': 70, 'Spending Score (1-100)': 85}
            Si es list: [70, 85]
        tipo : str
            'simple' o 'completo'
            
        Retorna:
        --------
        cliente_scaled : numpy.ndarray
            Datos del cliente escalados
        """
        # Convertir a array si es dict
        if isinstance(nuevo_cliente, dict):
            if tipo == 'simple':
                cliente_array = np.array([[
                    nuevo_cliente[COL_INCOME],
                    nuevo_cliente[COL_SPENDING]
                ]])
            else:
                cliente_array = np.array([[
                    nuevo_cliente[COL_AGE],
                    nuevo_cliente[COL_GENDER_ENCODED],
                    nuevo_cliente[COL_INCOME],
                    nuevo_cliente[COL_SPENDING]
                ]])
        else:
            cliente_array = np.array([nuevo_cliente])
        
        # Escalar
        scaler = self.scaler_simple if tipo == 'simple' else self.scaler_completo
        cliente_scaled = scaler.transform(cliente_array)
        
        return cliente_scaled


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def mostrar_estadisticas_features(df, features):
    """
    Muestra estadísticas de las features seleccionadas.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos
    features : list
        Lista de nombres de features
    """
    print("\n" + "=" * 80)
    print("ESTADÍSTICAS DE FEATURES")
    print("=" * 80)
    
    for feature in features:
        print(f"\n📊 {feature}:")
        print(f"   Media: {df[feature].mean():.2f}")
        print(f"   Mediana: {df[feature].median():.2f}")
        print(f"   Std: {df[feature].std():.2f}")
        print(f"   Min: {df[feature].min():.2f}")
        print(f"   Max: {df[feature].max():.2f}")


def detectar_valores_extremos(df, features, threshold=3):
    """
    Detecta valores extremos usando Z-score.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos
    features : list
        Lista de features a analizar
    threshold : float
        Umbral de Z-score para considerar extremo
        
    Retorna:
    --------
    extremos : pandas.DataFrame
        DataFrame con los valores extremos
    """
    from scipy import stats
    
    print("\n" + "=" * 80)
    print("DETECCIÓN DE VALORES EXTREMOS")
    print("=" * 80)
    
    extremos_dict = {}
    
    for feature in features:
        z_scores = np.abs(stats.zscore(df[feature]))
        extremos_mask = z_scores > threshold
        n_extremos = extremos_mask.sum()
        
        if n_extremos > 0:
            print(f"\n⚠️  {n_extremos} valores extremos en '{feature}'")
            extremos_dict[feature] = df[extremos_mask][[feature]]
        else:
            print(f"\n✅ No hay valores extremos en '{feature}'")
    
    return extremos_dict


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal para probar el módulo."""
    from data_loader import cargar_dataset
    
    # Cargar datos
    df = cargar_dataset()
    
    # Crear preprocesador
    prep = PreprocessorClientes()
    
    # Preprocesar (modelo simple)
    print("\n" + "=" * 80)
    print("PROBANDO PREPROCESAMIENTO - MODELO SIMPLE")
    print("=" * 80)
    X_scaled, df_processed = prep.preprocesar_completo(df, tipo='simple')
    
    # Guardar scaler
    prep.guardar_scaler(tipo='simple')
    
    # Probar transformación de nuevo cliente
    print("\n" + "=" * 80)
    print("PROBANDO TRANSFORMACIÓN DE NUEVO CLIENTE")
    print("=" * 80)
    
    nuevo_cliente = {COL_INCOME: 70, COL_SPENDING: 85}
    print(f"\nNuevo cliente: {nuevo_cliente}")
    
    cliente_scaled = prep.transformar_nuevo_cliente(nuevo_cliente, tipo='simple')
    print(f"Cliente escalado: {cliente_scaled}")
    
    print("\n✅ Preprocesamiento completado exitosamente")


if __name__ == '__main__':
    main()
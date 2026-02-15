"""
============================================================================
PROYECTO: SEGMENTACIÓN DE CLIENTES CON K-MEANS
Archivo: clustering.py
Descripción: Implementación del modelo K-Means y funciones relacionadas
============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle

# Importar configuración
from config import (
    K_OPTIMO,
    RANDOM_STATE,
    N_INIT,
    MAX_ITER,
    K_MIN,
    K_MAX,
    MODEL_FILE,
    COL_CLUSTER,
    COL_CLUSTER_NAME,
    CLUSTER_PROFILES
)


class ModeloKMeans:
    """
    Clase para entrenar y gestionar el modelo K-Means.
    """
    
    def __init__(self, n_clusters=K_OPTIMO, random_state=RANDOM_STATE):
        """
        Inicializa el modelo K-Means.
        
        Parámetros:
        -----------
        n_clusters : int
            Número de clusters
        random_state : int
            Semilla para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.modelo = None
        self.labels = None
        self.centroides = None
        self.inercia = None
        self.iteraciones = None
    
    
    def metodo_del_codo(self, X, k_min=K_MIN, k_max=K_MAX, verbose=True):
        """
        Implementa el método del codo para encontrar K óptimo.
        
        Parámetros:
        -----------
        X : numpy.ndarray
            Datos escalados
        k_min : int
            K mínimo a probar
        k_max : int
            K máximo a probar
        verbose : bool
            Si True, imprime resultados
            
        Retorna:
        --------
        inertias : list
            Lista de inercias para cada K
        k_range : range
            Rango de K probados
        """
        if verbose:
            print("\n" + "=" * 80)
            print("MÉTODO DEL CODO - DETERMINACIÓN DE K ÓPTIMO")
            print("=" * 80)
            print(f"\nProbando K desde {k_min} hasta {k_max-1}...")
            print("─" * 60)
        
        inertias = []
        k_range = range(k_min, k_max)
        
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=N_INIT,
                max_iter=MAX_ITER
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            if verbose:
                print(f"K = {k:2d} → Inercia = {kmeans.inertia_:10.2f}")
        
        if verbose:
            print("─" * 60)
            print("\n💡 Busca el 'codo' en el gráfico donde la inercia deja de bajar rápidamente")
        
        return inertias, k_range
    
    
    def calcular_silhouette(self, X, k_min=2, k_max=K_MAX, verbose=True):
        """
        Calcula el Silhouette Score para diferentes valores de K.
        
        Parámetros:
        -----------
        X : numpy.ndarray
            Datos escalados
        k_min : int
            K mínimo (debe ser >= 2)
        k_max : int
            K máximo
        verbose : bool
            Si True, imprime resultados
            
        Retorna:
        --------
        silhouette_scores : list
            Lista de Silhouette Scores
        k_range : range
            Rango de K probados
        """
        if verbose:
            print("\n" + "=" * 80)
            print("SILHOUETTE SCORE - MÉTRICA COMPLEMENTARIA")
            print("=" * 80)
            print(f"\nCalculando Silhouette Score para K desde {k_min} hasta {k_max-1}...")
            print("─" * 60)
        
        silhouette_scores = []
        k_range = range(k_min, k_max)
        
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=N_INIT
            )
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            
            if verbose:
                print(f"K = {k:2d} → Silhouette Score = {score:.4f}")
        
        if verbose:
            print("─" * 60)
            best_k = k_range[np.argmax(silhouette_scores)]
            best_score = max(silhouette_scores)
            print(f"\n🏆 Mejor K según Silhouette: {best_k} (score = {best_score:.4f})")
            print("\n💡 Silhouette Score:")
            print("   • 0.7-1.0: Estructura fuerte")
            print("   • 0.5-0.7: Estructura razonable")
            print("   • 0.25-0.5: Estructura débil")
        
        return silhouette_scores, k_range
    
    
    def entrenar(self, X, verbose=True):
        """
        Entrena el modelo K-Means con los parámetros configurados.
        
        Parámetros:
        -----------
        X : numpy.ndarray
            Datos escalados
        verbose : bool
            Si True, imprime resultados
            
        Retorna:
        --------
        self : ModeloKMeans
            Retorna self para permitir method chaining
        """
        if verbose:
            print("\n" + "=" * 80)
            print(f"ENTRENANDO K-MEANS CON K = {self.n_clusters}")
            print("=" * 80)
        
        # Crear y entrenar modelo
        self.modelo = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=N_INIT,
            max_iter=MAX_ITER,
            tol=1e-4,
            algorithm='lloyd'
        )
        
        self.labels = self.modelo.fit_predict(X)
        self.centroides = self.modelo.cluster_centers_
        self.inercia = self.modelo.inertia_
        self.iteraciones = self.modelo.n_iter_
        
        if verbose:
            print(f"\n✅ Entrenamiento completado")
            print(f"   Iteraciones: {self.iteraciones}")
            print(f"   Inercia final: {self.inercia:.2f}")
            
            # Mostrar distribución de clusters
            unique, counts = np.unique(self.labels, return_counts=True)
            print(f"\n📊 Distribución de clientes por cluster:")
            print("─" * 40)
            for cluster_id, count in zip(unique, counts):
                porcentaje = (count / len(self.labels)) * 100
                print(f"   Cluster {cluster_id}: {count:3d} clientes ({porcentaje:5.1f}%)")
            print("─" * 40)
        
        return self
    
    
    def obtener_centroides_originales(self, scaler):
        """
        Revierte el escalado de los centroides para interpretarlos.
        
        Parámetros:
        -----------
        scaler : StandardScaler
            Scaler usado para escalar los datos
            
        Retorna:
        --------
        centroides_original : numpy.ndarray
            Centroides en escala original
        """
        if self.centroides is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        centroides_original = scaler.inverse_transform(self.centroides)
        return centroides_original
    
    
    def crear_dataframe_centroides(self, centroides_original, feature_names):
        """
        Crea un DataFrame con los centroides para mejor visualización.
        
        Parámetros:
        -----------
        centroides_original : numpy.ndarray
            Centroides en escala original
        feature_names : list
            Nombres de las features
            
        Retorna:
        --------
        df_centroides : pandas.DataFrame
            DataFrame con los centroides
        """
        df_centroides = pd.DataFrame(
            centroides_original,
            columns=feature_names
        )
        df_centroides['Cluster'] = range(self.n_clusters)
        
        return df_centroides
    
    
    def evaluar_modelo(self, X, verbose=True):
        """
        Evalúa el modelo con diferentes métricas.
        
        Parámetros:
        -----------
        X : numpy.ndarray
            Datos escalados
        verbose : bool
            Si True, imprime resultados
            
        Retorna:
        --------
        metricas : dict
            Diccionario con las métricas calculadas
        """
        if self.labels is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        # Calcular métricas
        silhouette = silhouette_score(X, self.labels)
        davies_bouldin = davies_bouldin_score(X, self.labels)
        
        metricas = {
            'inercia': self.inercia,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'n_iteraciones': self.iteraciones
        }
        
        if verbose:
            print("\n" + "=" * 80)
            print("EVALUACIÓN DEL MODELO")
            print("=" * 80)
            print(f"\n📊 Métricas de clustering:")
            print(f"   Inercia (WCSS): {metricas['inercia']:.2f}")
            print(f"   Silhouette Score: {metricas['silhouette_score']:.4f}")
            print(f"   Davies-Bouldin Index: {metricas['davies_bouldin_score']:.4f}")
            print(f"   Iteraciones: {metricas['n_iteraciones']}")
            
            print("\n💡 Interpretación:")
            print("   • Silhouette Score: Más alto = mejor (máx 1.0)")
            print("   • Davies-Bouldin: Más bajo = mejor (mín 0.0)")
        
        return metricas
    
    
    def predecir(self, X_nuevo):
        """
        Predice el cluster para nuevos datos.
        
        Parámetros:
        -----------
        X_nuevo : numpy.ndarray
            Nuevos datos (ya escalados)
            
        Retorna:
        --------
        cluster : int o numpy.ndarray
            ID del cluster asignado
        """
        if self.modelo is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        return self.modelo.predict(X_nuevo)
    
    
    def guardar_modelo(self, filepath=None):
        """
        Guarda el modelo entrenado en disco.
        
        Parámetros:
        -----------
        filepath : str o Path, opcional
            Ruta donde guardar el modelo
        """
        if filepath is None:
            filepath = MODEL_FILE
        
        if self.modelo is None:
            raise ValueError("Debe entrenar el modelo primero")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.modelo, f)
        
        print(f"\n✅ Modelo guardado en: {filepath}")
    
    
    def cargar_modelo(self, filepath=None):
        """
        Carga un modelo desde disco.
        
        Parámetros:
        -----------
        filepath : str o Path, opcional
            Ruta del modelo a cargar
        """
        if filepath is None:
            filepath = MODEL_FILE
        
        with open(filepath, 'rb') as f:
            self.modelo = pickle.load(f)
        
        self.n_clusters = self.modelo.n_clusters
        self.centroides = self.modelo.cluster_centers_
        self.inercia = self.modelo.inertia_
        self.iteraciones = self.modelo.n_iter_
        
        print(f"✅ Modelo cargado desde: {filepath}")
        print(f"   Número de clusters: {self.n_clusters}")


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def agregar_clusters_a_dataframe(df, labels):
    """
    Agrega las etiquetas de cluster al DataFrame original.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame original
    labels : numpy.ndarray
        Labels de clusters
        
    Retorna:
    --------
    df : pandas.DataFrame
        DataFrame con columna 'Cluster' añadida
    """
    df = df.copy()
    df[COL_CLUSTER] = labels
    
    # Agregar nombres de clusters
    nombre_map = {k: v['nombre'] for k, v in CLUSTER_PROFILES.items()}
    df[COL_CLUSTER_NAME] = df[COL_CLUSTER].map(nombre_map)
    
    return df


def analizar_clusters(df):
    """
    Realiza análisis estadístico de los clusters.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con columna 'Cluster'
        
    Retorna:
    --------
    resumen : pandas.DataFrame
        DataFrame con estadísticas por cluster
    """
    from config import COL_AGE, COL_INCOME, COL_SPENDING, COL_GENDER
    
    print("\n" + "=" * 80)
    print("ANÁLISIS ESTADÍSTICO POR CLUSTER")
    print("=" * 80)
    
    resumen = df.groupby(COL_CLUSTER).agg({
        COL_AGE: ['mean', 'std', 'min', 'max'],
        COL_INCOME: ['mean', 'std', 'min', 'max'],
        COL_SPENDING: ['mean', 'std', 'min', 'max'],
        COL_GENDER: lambda x: f"{(x=='Female').sum()}F / {(x=='Male').sum()}M",
        'CustomerID': 'count'
    }).round(2)
    
    # Renombrar columnas para mejor legibilidad
    resumen.columns = ['_'.join(col).strip() for col in resumen.columns.values]
    resumen.rename(columns={'CustomerID_count': 'Tamaño'}, inplace=True)
    
    print("\n" + resumen.to_string())
    
    return resumen


def mostrar_perfiles_clusters(df):
    """
    Muestra los perfiles descriptivos de cada cluster.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con columna 'Cluster'
    """
    print("\n" + "=" * 80)
    print("PERFILES DE CLUSTERS")
    print("=" * 80)
    
    for cluster_id in sorted(df[COL_CLUSTER].unique()):
        profile = CLUSTER_PROFILES.get(cluster_id, {})
        tamaño = (df[COL_CLUSTER] == cluster_id).sum()
        porcentaje = (tamaño / len(df)) * 100
        
        print(f"\n🏷️  CLUSTER {cluster_id}: {profile.get('nombre', f'Cluster {cluster_id}').upper()}")
        print(f"   Descripción: {profile.get('descripcion', 'Sin descripción')}")
        print(f"   Tamaño: {tamaño} clientes ({porcentaje:.1f}%)")
        print(f"   Características:")
        
        for caracteristica in profile.get('caracteristicas', []):
            print(f"      • {caracteristica}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal para probar el módulo."""
    from data_loader import cargar_dataset
    from preprocessing import PreprocessorClientes
    
    # Cargar y preprocesar datos
    df = cargar_dataset()
    prep = PreprocessorClientes()
    X_scaled, df_processed = prep.preprocesar_completo(df, tipo='simple')
    
    # Crear modelo
    modelo = ModeloKMeans(n_clusters=5)
    
    # Método del codo
    inertias, k_range = modelo.metodo_del_codo(X_scaled)
    
    # Silhouette Score
    silhouette_scores, k_range_sil = modelo.calcular_silhouette(X_scaled)
    
    # Entrenar modelo
    modelo.entrenar(X_scaled)
    
    # Evaluar
    metricas = modelo.evaluar_modelo(X_scaled)
    
    # Agregar clusters al DataFrame
    df_final = agregar_clusters_a_dataframe(df_processed, modelo.labels)
    
    # Analizar clusters
    resumen = analizar_clusters(df_final)
    
    # Mostrar perfiles
    mostrar_perfiles_clusters(df_final)
    
    # Guardar modelo
    modelo.guardar_modelo()
    
    print("\n✅ Clustering completado exitosamente")


if __name__ == '__main__':
    main()
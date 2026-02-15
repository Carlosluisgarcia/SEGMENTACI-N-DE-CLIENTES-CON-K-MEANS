"""
============================================================================
PROYECTO: SEGMENTACIÓN DE CLIENTES CON K-MEANS
Archivo: visualization.py  
Descripción: Funciones para visualización de datos y resultados
============================================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from config import (
    COL_AGE, COL_INCOME, COL_SPENDING, COL_GENDER, COL_CLUSTER,
    EXPLORACION_DIR, CLUSTERING_DIR, BUSINESS_DIR,
    DPI, COLOR_PALETTE, CLUSTER_PROFILES
)


# Configuración global de estilo
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 9)


class VisualizadorClustering:
    """Clase para crear todas las visualizaciones del proyecto."""
    
    def __init__(self, df, n_clusters=5):
        self.df = df
        self.n_clusters = n_clusters
        self.colors = sns.color_palette(COLOR_PALETTE, n_clusters)
    
    
    def grafico_exploracion_univariada(self, guardar=True):
        """Crea gráficos de distribución univariada."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ANÁLISIS UNIVARIADO', fontsize=16, fontweight='bold')
        
        # Edad
        axes[0, 0].hist(self.df[COL_AGE], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribución de Edades')
        axes[0, 0].set_xlabel('Edad')
        axes[0, 0].axvline(self.df[COL_AGE].mean(), color='red', linestyle='--', 
                          label=f'Media: {self.df[COL_AGE].mean():.1f}')
        axes[0, 0].legend()
        
        # Ingresos
        axes[0, 1].hist(self.df[COL_INCOME], bins=20, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribución de Ingresos')
        axes[0, 1].set_xlabel('Ingresos (k$)')
        axes[0, 1].axvline(self.df[COL_INCOME].mean(), color='red', linestyle='--',
                          label=f'Media: {self.df[COL_INCOME].mean():.1f}')
        axes[0, 1].legend()
        
        # Spending Score
        axes[1, 0].hist(self.df[COL_SPENDING], bins=20, color='salmon', edgecolor='black')
        axes[1, 0].set_title('Distribución de Spending Score')
        axes[1, 0].set_xlabel('Spending Score')
        axes[1, 0].axvline(self.df[COL_SPENDING].mean(), color='red', linestyle='--',
                          label=f'Media: {self.df[COL_SPENDING].mean():.1f}')
        axes[1, 0].legend()
        
        # Género
        gender_counts = self.df[COL_GENDER].value_counts()
        axes[1, 1].bar(gender_counts.index, gender_counts.values, 
                      color=['lightblue', 'pink'], edgecolor='black')
        axes[1, 1].set_title('Distribución de Género')
        for i, v in enumerate(gender_counts.values):
            axes[1, 1].text(i, v + 2, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        if guardar:
            EXPLORACION_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(EXPLORACION_DIR / 'univariado.png', dpi=DPI)
            print(f"✅ Gráfico guardado: {EXPLORACION_DIR / 'univariado.png'}")
        plt.show()
    
    
    def grafico_exploracion_bivariada(self, guardar=True):
        """Crea scatter plots bivariados."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('ANÁLISIS BIVARIADO', fontsize=16, fontweight='bold')
        
        colors_gender = self.df[COL_GENDER].map({'Male': 'blue', 'Female': 'red'})
        
        # Edad vs Spending
        axes[0].scatter(self.df[COL_AGE], self.df[COL_SPENDING], 
                       c=colors_gender, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('Edad')
        axes[0].set_ylabel('Spending Score')
        axes[0].set_title('Edad vs Spending Score')
        axes[0].grid(True, alpha=0.3)
        
        # Ingresos vs Spending (MÁS IMPORTANTE)
        axes[1].scatter(self.df[COL_INCOME], self.df[COL_SPENDING],
                       c=colors_gender, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel('Ingresos (k$)')
        axes[1].set_ylabel('Spending Score')
        axes[1].set_title('Ingresos vs Spending Score ⭐')
        axes[1].grid(True, alpha=0.3)
        
        # Edad vs Ingresos
        axes[2].scatter(self.df[COL_AGE], self.df[COL_INCOME],
                       c=colors_gender, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[2].set_xlabel('Edad')
        axes[2].set_ylabel('Ingresos (k$)')
        axes[2].set_title('Edad vs Ingresos')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if guardar:
            plt.savefig(EXPLORACION_DIR / 'bivariado.png', dpi=DPI)
            print(f"✅ Gráfico guardado: {EXPLORACION_DIR / 'bivariado.png'}")
        plt.show()
    
    
    def grafico_metodo_codo(self, inertias, k_range, guardar=True):
        """Visualiza el método del codo."""
        plt.figure(figsize=(10, 6))
        plt.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=10)
        plt.xlabel('Número de Clusters (K)', fontsize=12, fontweight='bold')
        plt.ylabel('Inercia (WCSS)', fontsize=12, fontweight='bold')
        plt.title('MÉTODO DEL CODO', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.xticks(k_range)
        
        # Marcar K óptimo
        if 5 in k_range:
            plt.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='K óptimo sugerido')
            plt.legend()
        
        plt.tight_layout()
        if guardar:
            CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(CLUSTERING_DIR / 'metodo_codo.png', dpi=DPI)
            print(f"✅ Gráfico guardado: {CLUSTERING_DIR / 'metodo_codo.png'}")
        plt.show()
    
    
    def grafico_segmentacion_2d(self, centroides_df=None, guardar=True):
        """Visualiza los clusters en 2D."""
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Plotear cada cluster
        for i in range(self.n_clusters):
            cluster_data = self.df[self.df[COL_CLUSTER] == i]
            ax.scatter(cluster_data[COL_INCOME], cluster_data[COL_SPENDING],
                      s=100, c=[self.colors[i]], label=f'Cluster {i}',
                      alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Plotear centroides si se proporcionan
        if centroides_df is not None:
            ax.scatter(centroides_df[COL_INCOME], centroides_df[COL_SPENDING],
                      s=300, c='red', marker='X', edgecolors='black',
                      linewidth=2, label='Centroides', zorder=10)
        
        ax.set_xlabel('Ingresos Anuales (k$)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Spending Score (1-100)', fontsize=14, fontweight='bold')
        ax.set_title('SEGMENTACIÓN DE CLIENTES - K-MEANS', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if guardar:
            plt.savefig(CLUSTERING_DIR / 'segmentacion_2d.png', dpi=DPI, bbox_inches='tight')
            print(f"✅ Gráfico guardado: {CLUSTERING_DIR / 'segmentacion_2d.png'}")
        plt.show()
    
    
    def grafico_segmentacion_3d(self, guardar=True):
        """Visualiza los clusters en 3D."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(self.n_clusters):
            cluster_data = self.df[self.df[COL_CLUSTER] == i]
            ax.scatter(cluster_data[COL_AGE], cluster_data[COL_INCOME],
                      cluster_data[COL_SPENDING], s=80, c=[self.colors[i]],
                      label=f'Cluster {i}', alpha=0.6, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Edad', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Ingresos (k$)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel('Spending Score', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title('SEGMENTACIÓN 3D', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best')
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        if guardar:
            plt.savefig(CLUSTERING_DIR / 'segmentacion_3d.png', dpi=DPI)
            print(f"✅ Gráfico guardado: {CLUSTERING_DIR / 'segmentacion_3d.png'}")
        plt.show()
    
    
    def grafico_con_nombres_negocio(self, guardar=True):
        """Visualiza clusters con nombres de negocio."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        for cluster_id in range(self.n_clusters):
            cluster_data = self.df[self.df[COL_CLUSTER] == cluster_id]
            profile = CLUSTER_PROFILES.get(cluster_id, {})
            nombre = profile.get('nombre', f'Cluster {cluster_id}')
            tamaño = len(cluster_data)
            
            ax.scatter(cluster_data[COL_INCOME], cluster_data[COL_SPENDING],
                      s=120, label=f"{nombre} ({tamaño})",
                      alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Ingresos Anuales (k$)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Spending Score (1-100)', fontsize=14, fontweight='bold')
        ax.set_title('SEGMENTACIÓN CON ETIQUETAS DE NEGOCIO', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best', framealpha=0.9, title='Segmentos')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if guardar:
            BUSINESS_DIR.mkdir(parents=True, exist_ok=True)
            plt.savefig(BUSINESS_DIR / 'segmentacion_nombres.png', dpi=DPI)
            print(f"✅ Gráfico guardado: {BUSINESS_DIR / 'segmentacion_nombres.png'}")
        plt.show()


def crear_todas_visualizaciones(df, df_con_clusters, inertias, k_range, centroides_df=None):
    """Crea todas las visualizaciones del proyecto."""
    print("\n" + "=" * 80)
    print("GENERANDO TODAS LAS VISUALIZACIONES")
    print("=" * 80)
    
    viz = VisualizadorClustering(df_con_clusters)
    
    # Exploración
    print("\n1. Análisis univariado...")
    VisualizadorClustering(df).grafico_exploracion_univariada()
    
    print("\n2. Análisis bivariado...")
    VisualizadorClustering(df).grafico_exploracion_bivariada()
    
    # Clustering
    print("\n3. Método del codo...")
    viz.grafico_metodo_codo(inertias, k_range)
    
    print("\n4. Segmentación 2D...")
    viz.grafico_segmentacion_2d(centroides_df)
    
    print("\n5. Segmentación 3D...")
    viz.grafico_segmentacion_3d()
    
    # Negocio
    print("\n6. Segmentación con nombres de negocio...")
    viz.grafico_con_nombres_negocio()
    
    print("\n✅ Todas las visualizaciones generadas")


if __name__ == '__main__':
    from data_loader import cargar_dataset
    from preprocessing import PreprocessorClientes
    from clustering import ModeloKMeans, agregar_clusters_a_dataframe
    
    df = cargar_dataset()
    prep = PreprocessorClientes()
    X_scaled, df_processed = prep.preprocesar_completo(df, tipo='simple')
    
    modelo = ModeloKMeans(n_clusters=5)
    inertias, k_range = modelo.metodo_del_codo(X_scaled, verbose=False)
    modelo.entrenar(X_scaled, verbose=False)
    
    df_final = agregar_clusters_a_dataframe(df_processed, modelo.labels)
    centroides_orig = modelo.obtener_centroides_originales(prep.scaler_simple)
    centroides_df = modelo.crear_dataframe_centroides(centroides_orig, [COL_INCOME, COL_SPENDING])
    
    crear_todas_visualizaciones(df, df_final, inertias, k_range, centroides_df)
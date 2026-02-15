"""
============================================================================
PROYECTO: SEGMENTACIÓN DE CLIENTES CON K-MEANS
Archivo: main.py
Descripción: Script principal que ejecuta todo el pipeline
Autor: Carlos - L'Luis
Fecha: Enero 2026
============================================================================
"""

import sys
from datetime import datetime
import pandas as pd

# Importar módulos del proyecto
from config import (
    crear_directorios,
    verificar_dataset,
    OUTPUT_CSV,
    STATS_CSV,
    REPORT_TXT,
    K_OPTIMO,
    COL_INCOME,
    COL_SPENDING,
    CLUSTER_PROFILES,
    MARKETING_STRATEGIES
)
from data_loader import cargar_dataset, validar_dataset, mostrar_info_dataset
from preprocessing import PreprocessorClientes
from clustering import ModeloKMeans, agregar_clusters_a_dataframe, analizar_clusters, mostrar_perfiles_clusters
from visualization import crear_todas_visualizaciones


def banner_inicio():
    """Muestra banner de inicio."""
    print("\n" + "=" * 80)
    print(" " * 20 + "PROYECTO DE SEGMENTACIÓN DE CLIENTES")
    print(" " * 25 + "Algoritmo: K-Means Clustering")
    print(" " * 28 + "Autor: Carlos - L'Luis")
    print("=" * 80)
    print(f"\nFecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def generar_reporte(df, modelo, centroides_df, metricas, resumen_clusters):
    """Genera reporte de texto con resultados."""
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("REPORTE DE SEGMENTACIÓN DE CLIENTES - K-MEANS CLUSTERING\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de clientes analizados: {len(df)}\n")
        f.write(f"Número de clusters: {modelo.n_clusters}\n")
        f.write(f"Features utilizadas: {COL_INCOME}, {COL_SPENDING}\n\n")
        
        f.write("─" * 100 + "\n")
        f.write("MÉTRICAS DEL MODELO\n")
        f.write("─" * 100 + "\n")
        f.write(f"Inercia final: {metricas['inercia']:.2f}\n")
        f.write(f"Silhouette Score: {metricas['silhouette_score']:.4f}\n")
        f.write(f"Davies-Bouldin Index: {metricas['davies_bouldin_score']:.4f}\n")
        f.write(f"Iteraciones: {metricas['n_iteraciones']}\n\n")
        
        f.write("─" * 100 + "\n")
        f.write("DISTRIBUCIÓN DE CLIENTES\n")
        f.write("─" * 100 + "\n")
        f.write(df['Cluster_Nombre'].value_counts().to_string())
        f.write("\n\n")
        
        f.write("─" * 100 + "\n")
        f.write("CENTROIDES POR CLUSTER\n")
        f.write("─" * 100 + "\n")
        f.write(centroides_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("─" * 100 + "\n")
        f.write("PERFILES DE SEGMENTOS\n")
        f.write("─" * 100 + "\n")
        for cluster_id, profile in CLUSTER_PROFILES.items():
            f.write(f"\nCLUSTER {cluster_id}: {profile['nombre']}\n")
            f.write(f"Descripción: {profile['descripcion']}\n")
            f.write(f"Tamaño: {(df['Cluster'] == cluster_id).sum()} clientes\n")
            f.write("Características:\n")
            for char in profile['caracteristicas']:
                f.write(f"  • {char}\n")
        
        f.write("\n" + "─" * 100 + "\n")
        f.write("ESTRATEGIAS RECOMENDADAS\n")
        f.write("─" * 100 + "\n")
        for segmento, estrategia in MARKETING_STRATEGIES.items():
            f.write(f"\n{segmento}:\n")
            f.write(estrategia + "\n")
    
    print(f"\n✅ Reporte generado: {REPORT_TXT}")


def main():
    """Función principal que ejecuta todo el proyecto."""
    
    # ========================================================================
    # 1. INICIO
    # ========================================================================
    banner_inicio()
    
    try:
        # ========================================================================
        # 2. PREPARACIÓN
        # ========================================================================
        print("\n" + "=" * 80)
        print("FASE 1: PREPARACIÓN DEL ENTORNO")
        print("=" * 80)
        
        # Crear directorios
        print("\n📁 Creando estructura de directorios...")
        crear_directorios()
        
        # Verificar dataset
        print("\n📊 Verificando dataset...")
        verificar_dataset()
        
        # ========================================================================
        # 3. CARGA Y EXPLORACIÓN DE DATOS
        # ========================================================================
        print("\n" + "=" * 80)
        print("FASE 2: CARGA Y EXPLORACIÓN DE DATOS")
        print("=" * 80)
        
        df = cargar_dataset()
        
        if not validar_dataset(df):
            print("\n❌ Error en validación del dataset")
            sys.exit(1)
        
        mostrar_info_dataset(df)
        
        # ========================================================================
        # 4. PREPROCESAMIENTO
        # ========================================================================
        print("\n" + "=" * 80)
        print("FASE 3: PREPROCESAMIENTO DE DATOS")
        print("=" * 80)
        
        prep = PreprocessorClientes()
        X_scaled, df_processed = prep.preprocesar_completo(df, tipo='simple')
        prep.guardar_scaler(tipo='simple')
        
        # ========================================================================
        # 5. DETERMINACIÓN DE K ÓPTIMO
        # ========================================================================
        print("\n" + "=" * 80)
        print("FASE 4: DETERMINACIÓN DE K ÓPTIMO")
        print("=" * 80)
        
        modelo = ModeloKMeans(n_clusters=K_OPTIMO)
        
        # Método del codo
        inertias, k_range = modelo.metodo_del_codo(X_scaled)
        
        # Silhouette Score
        silhouette_scores, k_range_sil = modelo.calcular_silhouette(X_scaled)
        
        # ========================================================================
        # 6. ENTRENAMIENTO DEL MODELO
        # ========================================================================
        print("\n" + "=" * 80)
        print("FASE 5: ENTRENAMIENTO DEL MODELO")
        print("=" * 80)
        
        modelo.entrenar(X_scaled)
        
        # Evaluar modelo
        metricas = modelo.evaluar_modelo(X_scaled)
        
        # Guardar modelo
        modelo.guardar_modelo()
        
        # ========================================================================
        # 7. ANÁLISIS DE RESULTADOS
        # ========================================================================
        print("\n" + "=" * 80)
        print("FASE 6: ANÁLISIS DE RESULTADOS")
        print("=" * 80)
        
        # Agregar clusters al DataFrame
        df_final = agregar_clusters_a_dataframe(df_processed, modelo.labels)
        
        # Obtener centroides en escala original
        centroides_original = modelo.obtener_centroides_originales(prep.scaler_simple)
        centroides_df = modelo.crear_dataframe_centroides(
            centroides_original, 
            [COL_INCOME, COL_SPENDING]
        )
        
        print("\n📊 CENTROIDES EN ESCALA ORIGINAL:")
        print(centroides_df.to_string(index=False))
        
        # Analizar clusters
        resumen_clusters = analizar_clusters(df_final)
        
        # Mostrar perfiles
        mostrar_perfiles_clusters(df_final)
        
        # ========================================================================
        # 8. VISUALIZACIONES
        # ========================================================================
        print("\n" + "=" * 80)
        print("FASE 7: GENERACIÓN DE VISUALIZACIONES")
        print("=" * 80)
        
        crear_todas_visualizaciones(df, df_final, inertias, k_range, centroides_df)
        
        # ========================================================================
        # 9. EXPORTACIÓN DE RESULTADOS
        # ========================================================================
        print("\n" + "=" * 80)
        print("FASE 8: EXPORTACIÓN DE RESULTADOS")
        print("=" * 80)
        
        # Exportar CSV con clusters
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Datos exportados: {OUTPUT_CSV}")
        print(f"   Total de clientes: {len(df_final)}")
        
        # Exportar estadísticas
        resumen_clusters.to_csv(STATS_CSV)
        print(f"✅ Estadísticas exportadas: {STATS_CSV}")
        
        # Generar reporte
        generar_reporte(df_final, modelo, centroides_df, metricas, resumen_clusters)
        
        # ========================================================================
        # 10. RESUMEN FINAL
        # ========================================================================
        print("\n" + "=" * 80)
        print("PROYECTO COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        
        print("\n📊 RESUMEN DE SEGMENTACIÓN:")
        print(f"   • Total de clientes analizados: {len(df_final)}")
        print(f"   • Número de segmentos: {modelo.n_clusters}")
        print(f"   • Silhouette Score: {metricas['silhouette_score']:.4f}")
        
        print("\n📈 DISTRIBUCIÓN POR SEGMENTO:")
        for cluster_id in sorted(df_final['Cluster'].unique()):
            nombre = CLUSTER_PROFILES[cluster_id]['nombre']
            count = (df_final['Cluster'] == cluster_id).sum()
            pct = (count / len(df_final)) * 100
            print(f"   • {nombre}: {count} clientes ({pct:.1f}%)")
        
        print("\n📁 ARCHIVOS GENERADOS:")
        print(f"   • Datos segmentados: {OUTPUT_CSV}")
        print(f"   • Estadísticas: {STATS_CSV}")
        print(f"   • Reporte: {REPORT_TXT}")
        print(f"   • Modelo guardado: {modelo.modelo}")
        print(f"   • Gráficos en: results/figures/")
        
        print("\n" + "=" * 80)
        print("¡Revisa la carpeta 'results/' para ver todos los outputs!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
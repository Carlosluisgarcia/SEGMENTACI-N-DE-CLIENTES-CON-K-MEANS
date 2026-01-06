# 📊 PROYECTO DE SEGMENTACIÓN DE CLIENTES CON K-MEANS

## Sistema de Clustering No Supervisado para Marketing Estratégico
 ## Nombre Estudiante : Carlos Luis Garcia Lopez 
 ## Curso : Ingenieria Informatica : 3ro 
---

## 📑 ÍNDICE

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Objetivos](#-objetivos)
3. [Tecnologías y Herramientas](#%EF%B8%8F-tecnologías-y-herramientas)
4. [Interpretación de Resultados](#-interpretación-de-resultados)
---

## 🎯 DESCRIPCIÓN DEL PROYECTO

Este proyecto implementa un **sistema de segmentación de clientes** utilizando el algoritmo de **Machine Learning no supervisado K-Means**. El objetivo es identificar grupos (clusters) de clientes con características similares para desarrollar estrategias de marketing personalizadas.

### ¿Qué es Machine Learning No Supervisado?

El aprendizaje no supervisado es una técnica donde el algoritmo aprende patrones en los datos **sin etiquetas previas**. A diferencia del aprendizaje supervisado (donde le decimos al algoritmo "esto es un perro, esto es un gato"), aquí le damos los datos y él encuentra grupos por sí mismo.

**Analogía práctica**: Es como organizar tu inventario de L'Luis sin saber de antemano qué categorías crear. El algoritmo mira las características de los productos (precio, tamaño, color, etc.) y automáticamente los agrupa en categorías lógicas.

### ¿Qué es K-Means?

**K-Means** es el algoritmo de clustering más popular. Funciona de esta manera:

1. **Eliges K** (el número de grupos que quieres crear)
2. El algoritmo coloca K "centroides" (puntos centrales) al azar
3. Asigna cada cliente al centroide más cercano
4. Recalcula la posición de cada centroide como el promedio de sus clientes
5. Repite los pasos 3-4 hasta que los centroides no se muevan

**Resultado**: Clientes agrupados por similitud en comportamiento, ingresos, gastos, etc.

---

## 🎯 OBJETIVOS

### Objetivos del Proyecto

- ✅ **Segmentar clientes** en grupos homogéneos basados en comportamiento de compra
- ✅ **Identificar perfiles** de clientes (VIP, conservadores, jóvenes gastadores, etc.)
- ✅ **Desarrollar estrategias** de marketing personalizadas por segmento
- ✅ **Visualizar resultados** con gráficos profesionales en 2D y 3D
- ✅ **Generar reportes** automatizados con insights accionables

### Objetivos de Aprendizaje

- 📚 Dominar el algoritmo K-Means desde cero
- 📚 Aprender exploración y visualización de datos (EDA)
- 📚 Implementar preprocesamiento: escalado y normalización
- 📚 Determinar el número óptimo de clusters (Método del Codo)
- 📚 Interpretar resultados desde perspectiva de negocio
- 📚 Crear un proyecto completo de ML end-to-end

---

## 🛠️ TECNOLOGÍAS Y HERRAMIENTAS

### Lenguaje de Programación
- **Python 3.8+** (recomendado: Python 3.10 o superior)

### Librerías Principales

| Librería | Versión | Propósito |
|----------|---------|-----------|
| **pandas** | 2.0+ | Manipulación y análisis de datos |
| **numpy** | 1.24+ | Operaciones numéricas y matrices |
| **matplotlib** | 3.7+ | Visualización de datos básica |
| **seaborn** | 0.12+ | Visualización estadística avanzada |
| **scikit-learn** | 1.3+ | Algoritmos de Machine Learning |
| **plotly** | 5.14+ | Gráficos interactivos (opcional) |

### Entorno de Desarrollo

**Opción 1: Jupyter Notebook** ⭐ RECOMENDADO PARA EMPEZAR
- Ideal para exploración interactiva
- Permite ejecutar código celda por celda
- Visualiza resultados inline

**Opción 2: Python Script (.py)**
- Ideal para producción y automatización
- Más fácil de versionar con Git
- Ejecutable desde terminal

**Opción 3: IDE (PyCharm, VSCode)**
- Entorno completo de desarrollo
- Debugging avanzado
- Autocompletado y sugerencias

### Herramientas Adicionales
- **Git** (control de versiones)
- **Anaconda** o **Miniconda** (gestión de entornos - opcional)

---

## 📈 INTERPRETACIÓN DE RESULTADOS

### Cómo Leer los Gráficos

#### Scatter Plot 2D
```
Spending Score
     100 │              ┌─────┐
         │              │  1  │ ← VIP High Spenders
         │              └─────┘
      75 │  ┌─────┐              
         │  │  3  │              ← Jóvenes Gastadores
      50 │  └─────┘    ┌─────┐
         │             │  0  │  ← Precavidos
      25 │             └─────┘
         │  ┌─────┐             ┌─────┐
       0 │  │  4  │             │  2  │
         └──────────────────────────────── Ingresos
            20    40    60    80   100k$
         
         Oportunidad         Conservadores
```

### Interpretación de Métricas

#### Inercia (WCSS)
- **Valor bajo**: Clusters compactos (bueno)
- **Valor alto**: Clusters dispersos (malo)
- **Comparación**: Solo tiene sentido comparar entre diferentes K

#### Silhouette Score
- **0.7 - 1.0**: Estructura fuerte y clara
- **0.5 - 0.7**: Estructura razonable
- **0.25 - 0.5**: Estructura débil
- **< 0.25**: No hay estructura natural

### Validación de Resultados

**Preguntas clave**:
1. ✅ ¿Los clusters tienen sentido de negocio?
2. ✅ ¿Son accionables las estrategias?
3. ✅ ¿Los tamaños de clusters son manejables?
4. ✅ ¿Las características son distintivas?

**Señales de alerta** 🚨:
- Cluster con 1-2 clientes (demasiado pequeño)
- Cluster con >80% de clientes (demasiado general)
- Clusters solapados en visualización
- Características casi idénticas entre clusters

---
## 🎓 CERTIFICADO DE COMPLETION

### Checklist del Proyecto

Marca cuando completes cada fase:

**FASE DE SETUP**
- [ ] Python 3.8+ instalado y verificado
- [ ] Entorno virtual creado y activado
- [ ] Todas las librerías instaladas correctamente
- [ ] Estructura de carpetas creada
- [ ] Dataset Mall_Customers.csv descargado

**FASE DE ANÁLISIS**
- [ ] EDA completado con visualizaciones
- [ ] Análisis univariado realizado
- [ ] Análisis bivariado completado
- [ ] Matriz de correlación generada
- [ ] Outliers identificados (si hay)

**FASE DE PREPROCESAMIENTO**
- [ ] Variables categóricas codificadas
- [ ] Features seleccionadas correctamente
- [ ] Datos escalados con StandardScaler
- [ ] Verificación de escalado realizada

**FASE DE MODELADO**
- [ ] Método del Codo implementado
- [ ] K óptimo determinado
- [ ] Silhouette Score calculado (opcional)
- [ ] Modelo K-Means entrenado exitosamente
- [ ] Centroides extraídos y analizados
- [ ] Modelo guardado en pickle

**FASE DE VISUALIZACIÓN**
- [ ] Gráfico 2D de clusters generado
- [ ] Gráfico 3D creado (si aplica)
- [ ] Heatmap de características producido
- [ ] Estadísticas por cluster calculadas
- [ ] Todos los gráficos guardados en alta resolución

**FASE DE NEGOCIO**
- [ ] Perfiles de clusters creados
- [ ] Nombres descriptivos asignados
- [ ] Estrategias de marketing definidas
- [ ] Visualización con nombres de negocio generada
- [ ] Plan de acción por segmento desarrollado

**FASE DE EXPORTACIÓN**
- [ ] CSV con clusters exportado
- [ ] Reporte de texto generado
- [ ] Todos los archivos organizados
- [ ] Documentación completada

**APLICACIÓN PRÁCTICA**
- [ ] Plan para aplicar a L'Luis desarrollado
- [ ] Estructura de datos de L'Luis definida
- [ ] Análisis RFM entendido
- [ ] Estrategias específicas para L'Luis creadas

---
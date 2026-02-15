# 📊 PROYECTO DE SEGMENTACIÓN DE CLIENTES CON K-MEANS

## Sistema de Clustering No Supervisado para Marketing Estratégico



# Nombre : Carlos Luis Garcia Lopez 
# Carrera : Ingieneria Informatica 
# Año : 3ro
---

## 📑 ÍNDICE

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Objetivos](#-objetivos)
3. [Tecnologías y Herramientas](#%EF%B8%8F-tecnologías-y-herramientas)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Requisitos Previos](#%EF%B8%8F-requisitos-previos)
6. [Instalación y Configuración](#-instalación-y-configuración)
7. [Dataset](#-dataset)
8. [Metodología](#-metodología)
9. [Guía de Ejecución Paso a Paso](#-guía-de-ejecución-paso-a-paso)
10. [Interpretación de Resultados](#-interpretación-de-resultados)

12. [Troubleshooting](#-troubleshooting)

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

## 📁 ESTRUCTURA DEL PROYECTO

```
proyecto_segmentacion_clientes/
│
├── README.md                          # Este archivo - documentación principal
├── requirements.txt                   # Dependencias del proyecto
├── .gitignore                         # Archivos a ignorar en Git
│
├── data/                              # 📊 DATOS
│   ├── raw/                          # Datos originales sin procesar
│   │   └── Mall_Customers.csv        # Dataset descargado de Kaggle
│   │
│   └── processed/                    # Datos procesados
│       └── clientes_segmentados.csv  # Output con clusters asignados
│
├── notebooks/                         # 📓 JUPYTER NOTEBOOKS
│   ├── 01_exploracion_datos.ipynb    # Análisis exploratorio (EDA)
│   ├── 02_preprocesamiento.ipynb     # Limpieza y transformación
│   ├── 03_modelado.ipynb             # Implementación K-Means
│   └── 04_visualizacion.ipynb        # Gráficos y reportes
│
├── src/                               # 💻 CÓDIGO FUENTE
│   ├── __init__.py
│   ├── config.py                     # Configuración del proyecto
│   ├── data_loader.py                # Carga de datos
│   ├── preprocessing.py              # Funciones de preprocesamiento
│   ├── clustering.py                 # Implementación K-Means
│   ├── visualization.py              # Funciones de visualización
│   └── main.py                       # Script principal ejecutable
│
├── results/                           # 📈 RESULTADOS
│   ├── figures/                      # Gráficos generados
│   │   ├── 01_exploracion/          # EDA
│   │   ├── 02_clustering/           # Visualizaciones de clusters
│   │   └── 03_business/             # Gráficos para reportes
│   │
│   ├── reports/                      # Reportes generados
│   │   ├── reporte_tecnico.txt      # Métricas y estadísticas
│   │   └── reporte_negocio.pdf      # Presentación ejecutiva
│   │
│   └── models/                       # Modelos guardados
│       └── kmeans_final.pkl         # Modelo K-Means entrenado
│
├── docs/                              # 📚 DOCUMENTACIÓN ADICIONAL
│   ├── teoria_kmeans.md             # Teoría del algoritmo
│   ├── interpretacion_negocio.md    # Guía de interpretación
│   └── guia_aplicacion_lluis.md     # Cómo aplicarlo a L'Luis
│
└── tests/                             # 🧪 PRUEBAS (opcional)
    └── test_clustering.py            # Tests unitarios
```

---

## ⚙️ REQUISITOS PREVIOS

### Conocimientos Necesarios

#### 🟢 Nivel Básico (IMPRESCINDIBLE)
- Python básico (variables, loops, funciones)
- Comprensión de lectura de CSV
- Uso básico de terminal/línea de comandos

#### 🟡 Nivel Intermedio (RECOMENDADO)
- Pandas básico (DataFrames)
- Conceptos de estadística (media, desviación estándar)
- Gráficos con Matplotlib

#### 🔴 Nivel Avanzado (OPCIONAL)
- Álgebra lineal (vectores, matrices)
- Optimización matemática
- Git y GitHub

### Software Necesario

1. **Python 3.8+**
   - Descargar: https://www.python.org/downloads/
   - Verificar instalación: `python --version`

2. **pip** (gestor de paquetes)
   - Incluido con Python
   - Verificar: `pip --version`

3. **Editor de código** (elige uno):
   - Jupyter Notebook (recomendado para empezar)
   - Visual Studio Code
   - PyCharm Community Edition

### Hardware Recomendado

- **RAM**: Mínimo 4GB (recomendado 8GB)
- **Espacio**: ~500MB para librerías + datos
- **Procesador**: Cualquier procesador moderno (2+ cores)

> ⚠️ **Nota**: Este proyecto NO requiere GPU. K-Means es computacionalmente ligero.

---

## 🔧 INSTALACIÓN Y CONFIGURACIÓN

### Paso 1: Clonar o Descargar el Proyecto

**Opción A: Con Git**
```bash
git clone https://github.com/tu-usuario/proyecto-segmentacion-clientes.git
cd proyecto-segmentacion-clientes
```

**Opción B: Descarga Manual**
1. Descarga el ZIP del proyecto
2. Extrae en la carpeta deseada
3. Abre terminal en esa carpeta

### Paso 2: Crear Entorno Virtual (RECOMENDADO)

**¿Por qué usar entorno virtual?**
- Aísla las dependencias del proyecto
- Evita conflictos con otras instalaciones
- Facilita la reproducibilidad

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verificar activación:**
Deberías ver `(venv)` al inicio de tu línea de comandos.

### Paso 3: Instalar Dependencias

**Opción A: Desde requirements.txt** ⭐ RECOMENDADO
```bash
pip install -r requirements.txt
```

**Contenido del requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
plotly>=5.14.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

**Opción B: Instalación Manual**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly jupyter
```

**Verificar instalación:**
```bash
python -c "import pandas, numpy, sklearn, matplotlib; print('✅ Todo instalado correctamente')"
```

### Paso 4: Configurar Jupyter Notebook (si lo usas)

```bash
# Instalar Jupyter
pip install jupyter

# Registrar el entorno virtual como kernel
python -m ipykernel install --user --name=venv --display-name "Python (Segmentación)"

# Iniciar Jupyter
jupyter notebook
```

### Paso 5: Crear Estructura de Carpetas

```bash
# En Windows
mkdir data\raw data\processed results\figures\01_exploracion results\figures\02_clustering results\figures\03_business results\reports results\models src notebooks docs tests

# En Mac/Linux
mkdir -p data/raw data/processed results/figures/01_exploracion results/figures/02_clustering results/figures/03_business results/reports results/models src notebooks docs tests
```

---

## 📊 DATASET

### Descripción del Dataset: "Mall Customer Segmentation"

**Fuente**: Kaggle
**Nombre**: Mall Customer Segmentation Data
**URL**: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

### Características del Dataset

| Atributo | Descripción | Tipo | Rango |
|----------|-------------|------|-------|
| **CustomerID** | Identificador único del cliente | Integer | 1-200 |
| **Gender** | Género del cliente | Categórica | Male / Female |
| **Age** | Edad en años | Integer | 18-70 |
| **Annual Income (k$)** | Ingresos anuales en miles de dólares | Integer | 15-137 |
| **Spending Score (1-100)** | Puntuación de gastos asignada por el mall | Integer | 1-99 |

**Total de registros**: 200 clientes
**Valores nulos**: 0 (dataset limpio)

### ¿Cómo Descargar el Dataset?

#### Método 1: Descarga Manual desde Kaggle

1. **Crear cuenta en Kaggle** (gratis)
   - Ve a https://www.kaggle.com
   - Click en "Register" (Registrarse)
   - Completa el formulario

2. **Buscar el dataset**
   - En la barra de búsqueda: "Mall Customer Segmentation"
   - O usa el link directo: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

3. **Descargar**
   - Click en el botón "Download" (azul, arriba a la derecha)
   - Se descargará `archive.zip`

4. **Extraer y colocar**
   - Descomprime el archivo ZIP
   - Copia `Mall_Customers.csv` a `data/raw/`

#### Método 2: Descarga con Kaggle API (Avanzado)

```bash
# Instalar Kaggle API
pip install kaggle

# Configurar credenciales (ver https://www.kaggle.com/docs/api)
# Descargar dataset
kaggle datasets download -d vjchoudhary7/customer-segmentation-tutorial-in-python

# Mover a carpeta correcta
unzip customer-segmentation-tutorial-in-python.zip -d data/raw/
```

#### Método 3: Dataset Sintético (Sin Kaggle)

Si no puedes usar Kaggle, puedes generar un dataset sintético similar:

```python
import pandas as pd
import numpy as np

np.random.seed(42)

# Generar datos sintéticos
n_customers = 200

data = {
    'CustomerID': range(1, n_customers + 1),
    'Gender': np.random.choice(['Male', 'Female'], n_customers),
    'Age': np.random.randint(18, 70, n_customers),
    'Annual Income (k$)': np.random.randint(15, 140, n_customers),
    'Spending Score (1-100)': np.random.randint(1, 100, n_customers)
}

df = pd.DataFrame(data)
df.to_csv('data/raw/Mall_Customers.csv', index=False)
print("✅ Dataset sintético creado")
```

### Verificar que el Dataset está Correcto

```python
import pandas as pd

df = pd.read_csv('data/raw/Mall_Customers.csv')

# Verificaciones
assert len(df) == 200, "Debe tener 200 filas"
assert list(df.columns) == ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'], "Columnas incorrectas"
assert df.isnull().sum().sum() == 0, "No debe tener valores nulos"

print("✅ Dataset verificado correctamente")
print(df.head())
```

---

## 🔬 METODOLOGÍA

### Proceso End-to-End del Proyecto

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESO DE CLUSTERING                     │
└─────────────────────────────────────────────────────────────┘

1. EXPLORACIÓN DE DATOS (EDA)
   ↓
   - Cargar dataset
   - Estadísticas descriptivas
   - Visualizaciones univariadas
   - Análisis bivariado
   - Detección de outliers
   - Análisis de correlaciones

2. PREPROCESAMIENTO
   ↓
   - Manejo de valores nulos (si hay)
   - Codificación de variables categóricas
   - Selección de features relevantes
   - Escalado/Normalización de datos
   - Creación de subsets

3. DETERMINACIÓN DE K ÓPTIMO
   ↓
   - Método del Codo (Elbow Method)
   - Silhouette Score (opcional)
   - Davies-Bouldin Index (opcional)

4. ENTRENAMIENTO DEL MODELO
   ↓
   - Inicializar K-Means con K óptimo
   - Ajustar modelo a datos escalados
   - Obtener labels de clusters
   - Extraer centroides

5. EVALUACIÓN
   ↓
   - Calcular métricas de clustering
   - Analizar distribución de clusters
   - Validar coherencia de grupos

6. VISUALIZACIÓN
   ↓
   - Scatter plots 2D
   - Gráficos 3D (opcional)
   - Heatmaps de características
   - Gráficos de barras por segmento

7. INTERPRETACIÓN DE NEGOCIO
   ↓
   - Perfilar cada cluster
   - Asignar nombres descriptivos
   - Definir estrategias de marketing
   - Generar reportes ejecutivos

8. EXPORTACIÓN Y DEPLOYMENT
   ↓
   - Guardar modelo entrenado
   - Exportar datos con clusters
   - Crear dashboards (opcional)
   - Documentar hallazgos
```

### Algoritmo K-Means en Detalle

#### Pseudocódigo Simplificado

```
FUNCIÓN K-Means(datos, K):
    1. Seleccionar K centroides iniciales al azar
    
    2. REPETIR hasta convergencia:
        a. Asignar cada punto al centroide más cercano
           (calcular distancia euclidiana)
        
        b. Recalcular cada centroide como el promedio
           de todos los puntos asignados a él
        
        c. SI los centroides no cambiaron:
              SALIR del loop (convergencia)
    
    3. DEVOLVER clusters y centroides finales
FIN FUNCIÓN
```

#### Fórmulas Matemáticas Clave

**1. Distancia Euclidiana** (para asignar puntos a clusters)

```
d(p, c) = √[(x₁ - c₁)² + (x₂ - c₂)² + ... + (xₙ - cₙ)²]
```

Donde:
- `p` = punto de datos
- `c` = centroide
- `n` = número de dimensiones

**2. Actualización de Centroides**

```
c_nuevo = (1/n) × Σ(puntos_en_cluster)
```

Donde:
- `n` = número de puntos en el cluster
- `Σ` = suma de todos los puntos

**3. Inercia (WCSS - Within-Cluster Sum of Squares)**

```
WCSS = Σ[k=1 to K] Σ[x en Cluster_k] ||x - μₖ||²
```

Donde:
- `K` = número de clusters
- `x` = punto de datos
- `μₖ` = centroide del cluster k
- `|| ||` = norma euclidiana

---

## 🚀 GUÍA DE EJECUCIÓN PASO A PASO

### FASE 1: EXPLORACIÓN DE DATOS (EDA)

**Objetivo**: Entender qué datos tenemos antes de aplicar el modelo.

**Archivo**: `notebooks/01_exploracion_datos.ipynb` o crear script Python.

#### Paso 1.1: Cargar y Visualizar Datos

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Cargar datos
df = pd.read_csv('data/raw/Mall_Customers.csv')

# Primeras filas
print("═" * 80)
print("PRIMERAS 5 FILAS DEL DATASET")
print("═" * 80)
print(df.head())

# Información del DataFrame
print("\n" + "═" * 80)
print("INFORMACIÓN DEL DATASET")
print("═" * 80)
print(df.info())

# Estadísticas descriptivas
print("\n" + "═" * 80)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("═" * 80)
print(df.describe())
```

**¿Qué buscar?**
- ✅ Número de filas y columnas
- ✅ Tipos de datos correctos
- ✅ Valores nulos (deberían ser 0)
- ✅ Rangos de valores lógicos

#### Paso 1.2: Análisis Univariado

```python
# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('ANÁLISIS UNIVARIADO', fontsize=16, fontweight='bold')

# Distribución de Edad
axes[0, 0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribución de Edades')
axes[0, 0].set_xlabel('Edad')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].axvline(df['Age'].mean(), color='red', linestyle='--', 
                    label=f'Media: {df["Age"].mean():.1f}')
axes[0, 0].legend()

# Distribución de Ingresos
axes[0, 1].hist(df['Annual Income (k$)'], bins=20, color='lightgreen', edgecolor='black')
axes[0, 1].set_title('Distribución de Ingresos')
axes[0, 1].set_xlabel('Ingresos Anuales (k$)')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].axvline(df['Annual Income (k$)'].mean(), color='red', linestyle='--',
                    label=f'Media: {df["Annual Income (k$)"].mean():.1f}')
axes[0, 1].legend()

# Distribución de Spending Score
axes[1, 0].hist(df['Spending Score (1-100)'], bins=20, color='salmon', edgecolor='black')
axes[1, 0].set_title('Distribución de Spending Score')
axes[1, 0].set_xlabel('Spending Score')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].axvline(df['Spending Score (1-100)'].mean(), color='red', linestyle='--',
                    label=f'Media: {df["Spending Score (1-100)"].mean():.1f}')
axes[1, 0].legend()

# Distribución de Género
gender_counts = df['Gender'].value_counts()
axes[1, 1].bar(gender_counts.index, gender_counts.values, 
               color=['lightblue', 'pink'], edgecolor='black')
axes[1, 1].set_title('Distribución de Género')
axes[1, 1].set_ylabel('Frecuencia')
for i, v in enumerate(gender_counts.values):
    axes[1, 1].text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/figures/01_exploracion/univariado.png', dpi=300)
plt.show()
```

**Interpretación esperada**:
- **Edad**: Distribución variada, posiblemente bimodal
- **Ingresos**: Rango amplio (15k-137k$)
- **Spending Score**: Distribución uniforme
- **Género**: Aproximadamente balanceado

#### Paso 1.3: Análisis Bivariado (MUY IMPORTANTE)

```python
# Scatter plots para identificar patrones
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('ANÁLISIS BIVARIADO - Búsqueda de Patrones', fontsize=16, fontweight='bold')

# Edad vs Spending Score
axes[0].scatter(df['Age'], df['Spending Score (1-100)'], 
                c=df['Gender'].map({'Male': 'blue', 'Female': 'red'}),
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[0].set_xlabel('Edad')
axes[0].set_ylabel('Spending Score')
axes[0].set_title('Edad vs Spending Score')
axes[0].legend(['Hombres', 'Mujeres'], loc='best')
axes[0].grid(True, alpha=0.3)

# Ingresos vs Spending Score (LA MÁS IMPORTANTE)
axes[1].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                c=df['Gender'].map({'Male': 'blue', 'Female': 'red'}),
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[1].set_xlabel('Ingresos Anuales (k$)')
axes[1].set_ylabel('Spending Score')
axes[1].set_title('Ingresos vs Spending Score ⭐')
axes[1].grid(True, alpha=0.3)

# Edad vs Ingresos
axes[2].scatter(df['Age'], df['Annual Income (k$)'],
                c=df['Gender'].map({'Male': 'blue', 'Female': 'red'}),
                alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
axes[2].set_xlabel('Edad')
axes[2].set_ylabel('Ingresos Anuales (k$)')
axes[2].set_title('Edad vs Ingresos')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/01_exploracion/bivariado.png', dpi=300)
plt.show()
```

**¿Qué buscamos?**
- 👀 **Clusters visibles**: ¿Se ven grupos naturales?
- 👀 **Patrones**: ¿Hay correlaciones?
- 👀 **Outliers**: ¿Puntos muy alejados?

**HALLAZGO CLAVE**: En el gráfico Ingresos vs Spending Score deberías ver ~5 grupos naturales.

#### Paso 1.4: Matriz de Correlación

```python
# Seleccionar solo columnas numéricas
numerical_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Calcular correlación
correlation_matrix = numerical_df.corr()

# Visualizar con heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
            center=0, square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            fmt='.2f', vmin=-1, vmax=1)
plt.title('MATRIZ DE CORRELACIÓN', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figures/01_exploracion/correlacion.png', dpi=300)
plt.show()

print("\nCOEFICIENTES DE CORRELACIÓN:")
print(correlation_matrix)
```

**Interpretación**:
- Valores cercanos a **+1**: Correlación positiva fuerte
- Valores cercanos a **-1**: Correlación negativa fuerte
- Valores cercanos a **0**: No hay correlación

---

### FASE 2: PREPROCESAMIENTO

**Objetivo**: Preparar los datos para el algoritmo K-Means.

**Archivo**: `src/preprocessing.py` o continuar en notebook.

#### Paso 2.1: Codificar Variables Categóricas

```python
from sklearn.preprocessing import LabelEncoder

# Crear copia para no modificar original
df_processed = df.copy()

# Codificar género
label_encoder = LabelEncoder()
df_processed['Gender_Encoded'] = label_encoder.fit_transform(df['Gender'])

print("CODIFICACIÓN DE GÉNERO:")
print(df_processed[['Gender', 'Gender_Encoded']].drop_duplicates())
```

**Resultado esperado**:
```
  Gender  Gender_Encoded
0 Female               0
1   Male               1
```

#### Paso 2.2: Selección de Features

Vamos a crear **DOS conjuntos de features**:

```python
# CONJUNTO 1: Solo Ingresos y Spending Score (2D - más fácil de visualizar)
X_simple = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
print(f"X_simple shape: {X_simple.shape}")  # Debe ser (200, 2)

# CONJUNTO 2: Todas las features (multidimensional - más completo)
X_completo = df_processed[['Age', 'Gender_Encoded', 
                            'Annual Income (k$)', 
                            'Spending Score (1-100)']].values
print(f"X_completo shape: {X_completo.shape}")  # Debe ser (200, 4)
```

**¿Cuál usar?**
- **X_simple**: Para aprender y visualizar fácilmente
- **X_completo**: Para análisis más robusto (recomendado en producción)

#### Paso 2.3: Escalado de Datos (CRÍTICO)

**¿Por qué escalar?**
K-Means usa distancia euclidiana. Si una variable va de 0-100 y otra de 0-150000, la segunda dominará el cálculo.

```python
from sklearn.preprocessing import StandardScaler

# Crear escaladores
scaler_simple = StandardScaler()
scaler_completo = StandardScaler()

# Escalar
X_simple_scaled = scaler_simple.fit_transform(X_simple)
X_completo_scaled = scaler_completo.fit_transform(X_completo)

# Comparar antes y después
print("═" * 80)
print("DATOS ORIGINALES (primeras 3 filas):")
print("═" * 80)
print(X_simple[:3])

print("\n" + "═" * 80)
print("DATOS ESCALADOS (primeras 3 filas):")
print("═" * 80)
print(X_simple_scaled[:3])

print("\n" + "═" * 80)
print("ESTADÍSTICAS DATOS ESCALADOS:")
print("═" * 80)
print(f"Media: {X_simple_scaled.mean(axis=0)}")  # Debe ser ~[0, 0]
print(f"Desviación estándar: {X_simple_scaled.std(axis=0)}")  # Debe ser ~[1, 1]
```

**Fórmula de StandardScaler**:
```
z = (x - μ) / σ
```
Donde:
- `x` = valor original
- `μ` = media de la columna
- `σ` = desviación estándar de la columna
- `z` = valor escalado

---

### FASE 3: DETERMINACIÓN DE K ÓPTIMO

**Objetivo**: Encontrar el número ideal de clusters usando el **Método del Codo**.

#### Paso 3.1: Método del Codo (Elbow Method)

```python
from sklearn.cluster import KMeans

# Probar diferentes valores de K
inertias = []
K_range = range(1, 11)

print("CALCULANDO INERCIA PARA DIFERENTES VALORES DE K...")
print("═" * 60)

for k in K_range:
    kmeans = KMeans(n_clusters=k, 
                    random_state=42,      # Para reproducibilidad
                    n_init=10,            # Número de inicializaciones
                    max_iter=300)         # Máximo de iteraciones
    
    kmeans.fit(X_simple_scaled)
    inertias.append(kmeans.inertia_)
    
    print(f"K = {k:2d} → Inercia = {kmeans.inertia_:8.2f}")

print("═" * 60)
```

#### Paso 3.2: Graficar el Codo

```python
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=10)
plt.xlabel('Número de Clusters (K)', fontsize=12, fontweight='bold')
plt.ylabel('Inercia (WCSS)', fontsize=12, fontweight='bold')
plt.title('MÉTODO DEL CODO - Determinación de K Óptimo', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.xticks(K_range)

# Marcar K=5 si es el óptimo
plt.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='K óptimo sugerido')
plt.legend()

plt.tight_layout()
plt.savefig('results/figures/02_clustering/metodo_codo.png', dpi=300)
plt.show()
```

**¿Cómo interpretar el gráfico?**

```
Inercia
   │
   │ *
   │   *
   │     *        ← CODO (punto óptimo)
   │       *___
   │           *___*___*___
   └─────────────────────────── K
     1  2  3  4  5  6  7  8  9
```

- El "codo" es donde la curva cambia de pendiente pronunciada a suave
- **Antes del codo**: Mucha mejora al añadir clusters
- **Después del codo**: Poca mejora (no vale la pena la complejidad)

**Para este dataset**, K=5 suele ser óptimo.

#### Paso 3.3: Silhouette Score (Métrica Complementaria - Opcional)

```python
from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):  # Silhouette necesita K >= 2
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_simple_scaled)
    score = silhouette_score(X_simple_scaled, labels)
    silhouette_scores.append(score)
    print(f"K = {k} → Silhouette Score = {score:.4f}")

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'go-', linewidth=2, markersize=10)
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score por K')
plt.grid(True, alpha=0.3)
plt.savefig('results/figures/02_clustering/silhouette.png', dpi=300)
plt.show()
```

**Interpretación Silhouette Score**:
- Rango: **-1 a +1**
- **Cerca de +1**: Clusters bien definidos
- **Cerca de 0**: Clusters solapados
- **Negativo**: Puntos mal asignados

El K con **mayor Silhouette Score** es otra buena opción.

---

### FASE 4: ENTRENAMIENTO DEL MODELO

**Objetivo**: Crear y entrenar el modelo K-Means con K óptimo.

#### Paso 4.1: Crear y Entrenar Modelo

```python
# Definir K óptimo (ajusta según tu análisis)
K_OPTIMO = 5

print("═" * 80)
print(f"ENTRENANDO K-MEANS CON K = {K_OPTIMO}")
print("═" * 80)

# Crear modelo
kmeans_final = KMeans(
    n_clusters=K_OPTIMO,
    random_state=42,        # Semilla para reproducibilidad
    n_init=10,              # Número de veces que se ejecuta con diferentes centroides
    max_iter=300,           # Máximo de iteraciones por ejecución
    tol=1e-4,               # Tolerancia para convergencia
    algorithm='lloyd'       # Algoritmo estándar
)

# Entrenar modelo
clusters = kmeans_final.fit_predict(X_simple_scaled)

print(f"✅ Modelo entrenado en {kmeans_final.n_iter_} iteraciones")
print(f"✅ Inercia final: {kmeans_final.inertia_:.2f}")

# Agregar clusters al DataFrame original
df['Cluster'] = clusters

print("\n" + "═" * 80)
print("DISTRIBUCIÓN DE CLIENTES POR CLUSTER")
print("═" * 80)
print(df['Cluster'].value_counts().sort_index())
```

#### Paso 4.2: Analizar Centroides

```python
# Obtener centroides en escala escalada
centroides_scaled = kmeans_final.cluster_centers_

# Revertir escalado para interpretar
centroides_original = scaler_simple.inverse_transform(centroides_scaled)

# Crear DataFrame para mejor visualización
centroides_df = pd.DataFrame(
    centroides_original,
    columns=['Ingresos (k$)', 'Spending Score']
)
centroides_df['Cluster'] = range(K_OPTIMO)

print("\n" + "═" * 80)
print("CENTROIDES DE CADA CLUSTER (Escala Original)")
print("═" * 80)
print(centroides_df.to_string(index=False))
```

**Salida esperada** (ejemplo):
```
 Ingresos (k$)  Spending Score  Cluster
         55.30           49.52        0
         86.54           82.13        1
         88.20           17.11        2
         25.73           79.36        3
         26.30           20.91        4
```

#### Paso 4.3: Guardar Modelo (Para Reutilizarlo)

```python
import pickle

# Guardar modelo
with open('results/models/kmeans_final.pkl', 'wb') as f:
    pickle.dump(kmeans_final, f)

# Guardar escalador (importante para predecir nuevos datos)
with open('results/models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_simple, f)

print("✅ Modelo y escalador guardados")
```

**Para cargar después**:
```python
# Cargar modelo
with open('results/models/kmeans_final.pkl', 'rb') as f:
    kmeans_loaded = pickle.load(f)

# Predecir nuevo cliente
nuevo_cliente = [[70, 85]]  # Ingresos 70k$, Spending 85
nuevo_cliente_scaled = scaler_simple.transform(nuevo_cliente)
cluster_asignado = kmeans_loaded.predict(nuevo_cliente_scaled)
print(f"Cliente asignado al Cluster {cluster_asignado[0]}")
```

---

### FASE 5: VISUALIZACIÓN DE RESULTADOS

**Objetivo**: Crear gráficos profesionales para presentar hallazgos.

#### Paso 5.1: Scatter Plot 2D con Clusters

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Paleta de colores profesional
colors = sns.color_palette('husl', K_OPTIMO)

# Crear figura
fig, ax = plt.subplots(figsize=(14, 9))

# Plotear cada cluster
for i in range(K_OPTIMO):
    cluster_data = df[df['Cluster'] == i]
    ax.scatter(
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        s=100,                      # Tamaño de puntos
        c=[colors[i]],              # Color del cluster
        label=f'Cluster {i}',
        alpha=0.6,                  # Transparencia
        edgecolors='black',
        linewidth=0.5
    )

# Plotear centroides
ax.scatter(
    centroides_df['Ingresos (k$)'],
    centroides_df['Spending Score'],
    s=300,                          # Más grandes que los puntos
    c='red',
    marker='X',                     # Forma de X
    edgecolors='black',
    linewidth=2,
    label='Centroides',
    zorder=10                       # Dibujarse encima
)

# Configuración del gráfico
ax.set_xlabel('Ingresos Anuales (k$)', fontsize=14, fontweight='bold')
ax.set_ylabel('Spending Score (1-100)', fontsize=14, fontweight='bold')
ax.set_title('SEGMENTACIÓN DE CLIENTES - K-MEANS CLUSTERING', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/02_clustering/segmentacion_2d.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Gráfico de segmentación guardado")
```

#### Paso 5.2: Gráfico 3D (Si usaste Edad también)

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plotear cada cluster
for i in range(K_OPTIMO):
    cluster_data = df[df['Cluster'] == i]
    ax.scatter(
        cluster_data['Age'],
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        s=80,
        c=[colors[i]],
        label=f'Cluster {i}',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

# Etiquetas
ax.set_xlabel('Edad', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Ingresos Anuales (k$)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_zlabel('Spending Score', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('SEGMENTACIÓN 3D DE CLIENTES', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best')

# Rotar para mejor vista
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('results/figures/02_clustering/segmentacion_3d.png', dpi=300)
plt.show()
```

#### Paso 5.3: Análisis Estadístico por Cluster

```python
# Resumen estadístico por cluster
cluster_stats = df.groupby('Cluster').agg({
    'Age': ['mean', 'std', 'min', 'max'],
    'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
    'Spending Score (1-100)': ['mean', 'std', 'min', 'max'],
    'Gender': lambda x: f"{(x=='Female').sum()} F / {(x=='Male').sum()} M",
    'CustomerID': 'count'
}).round(2)

cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
cluster_stats.rename(columns={'CustomerID_count': 'Tamaño'}, inplace=True)

print("\n" + "═" * 120)
print("RESUMEN ESTADÍSTICO POR CLUSTER")
print("═" * 120)
print(cluster_stats)

# Guardar en CSV
cluster_stats.to_csv('results/reports/cluster_statistics.csv')
```

#### Paso 5.4: Heatmap de Características por Cluster

```python
# Calcular medias por cluster
cluster_means = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()

# Normalizar para mejor visualización en heatmap
from sklearn.preprocessing import MinMaxScaler
scaler_heatmap = MinMaxScaler()
cluster_means_normalized = scaler_heatmap.fit_transform(cluster_means)
cluster_means_normalized_df = pd.DataFrame(
    cluster_means_normalized,
    index=cluster_means.index,
    columns=cluster_means.columns
)

# Crear heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means_normalized_df.T, 
            annot=cluster_means.T.values,  # Mostrar valores originales
            fmt='.1f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Valor Normalizado'},
            linewidths=1,
            linecolor='white')
plt.title('CARACTERÍSTICAS PROMEDIO POR CLUSTER', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Cluster', fontsize=12, fontweight='bold')
plt.ylabel('Característica', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/02_clustering/heatmap_clusters.png', dpi=300)
plt.show()
```

---

### FASE 6: INTERPRETACIÓN DE NEGOCIO

**Objetivo**: Convertir clusters matemáticos en segmentos de marketing accionables.

#### Paso 6.1: Perfilar Clusters

Basado en los centroides, asigna nombres descriptivos:

```python
# Análisis manual basado en centroides
# (Ajusta según TUS resultados)

cluster_profiles = {
    0: {
        'nombre': 'Precavidos de Ingresos Medios',
        'descripcion': 'Ingresos medios, spending moderado',
        'caracteristicas': ['Ingresos: 50-60k$', 'Spending: 40-50', 'Edad: Variada'],
        'tamaño': (df['Cluster'] == 0).sum()
    },
    1: {
        'nombre': 'VIP High Spenders',
        'descripcion': 'Altos ingresos, alto spending',
        'caracteristicas': ['Ingresos: 70-90k$', 'Spending: 75-90', 'Edad: 30-45'],
        'tamaño': (df['Cluster'] == 1).sum()
    },
    2: {
        'nombre': 'Conservadores de Alto Ingreso',
        'descripcion': 'Altos ingresos, bajo spending',
        'caracteristicas': ['Ingresos: 70-90k$', 'Spending: 10-25', 'Edad: 40-60'],
        'tamaño': (df['Cluster'] == 2).sum()
    },
    3: {
        'nombre': 'Jóvenes Gastadores',
        'descripcion': 'Bajos ingresos, alto spending',
        'caracteristicas': ['Ingresos: 20-40k$', 'Spending: 70-90', 'Edad: 18-30'],
        'tamaño': (df['Cluster'] == 3).sum()
    },
    4: {
        'nombre': 'Oportunidad de Crecimiento',
        'descripcion': 'Bajos ingresos, bajo spending',
        'caracteristicas': ['Ingresos: 20-40k$', 'Spending: 10-30', 'Edad: Variada'],
        'tamaño': (df['Cluster'] == 4).sum()
    }
}

# Agregar nombres al DataFrame
nombre_map = {k: v['nombre'] for k, v in cluster_profiles.items()}
df['Cluster_Nombre'] = df['Cluster'].map(nombre_map)

print("\n" + "═" * 100)
print("PERFILES DE SEGMENTOS")
print("═" * 100)
for cluster_id, profile in cluster_profiles.items():
    print(f"\n🏷️  CLUSTER {cluster_id}: {profile['nombre'].upper()}")
    print(f"   Descripción: {profile['descripcion']}")
    print(f"   Tamaño: {profile['tamaño']} clientes ({profile['tamaño']/len(df)*100:.1f}%)")
    print(f"   Características:")
    for char in profile['caracteristicas']:
        print(f"      • {char}")
```

#### Paso 6.2: Estrategias de Marketing por Segmento

```python
marketing_strategies = {
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

# Imprimir estrategias
print("\n" + "═" * 100)
print("ESTRATEGIAS DE MARKETING POR SEGMENTO")
print("═" * 100)
for segmento, estrategia in marketing_strategies.items():
    print(f"\n{segmento.upper()}")
    print("─" * 100)
    print(estrategia)
```

#### Paso 6.3: Visualización con Nombres de Negocio

```python
fig, ax = plt.subplots(figsize=(16, 10))

# Plotear con nombres descriptivos
for cluster_id, profile in cluster_profiles.items():
    cluster_data = df[df['Cluster'] == cluster_id]
    ax.scatter(
        cluster_data['Annual Income (k$)'],
        cluster_data['Spending Score (1-100)'],
        s=120,
        label=f"{profile['nombre']} ({profile['tamaño']})",
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

ax.set_xlabel('Ingresos Anuales (k$)', fontsize=14, fontweight='bold')
ax.set_ylabel('Spending Score (1-100)', fontsize=14, fontweight='bold')
ax.set_title('SEGMENTACIÓN DE CLIENTES CON ETIQUETAS DE NEGOCIO', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best', framealpha=0.9, title='Segmentos')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/03_business/segmentacion_con_nombres.png', dpi=300)
plt.show()
```

---

### FASE 7: EXPORTACIÓN Y REPORTES

#### Paso 7.1: Exportar Datos con Clusters

```python
# Crear DataFrame final
df_export = df[['CustomerID', 'Gender', 'Age', 
                'Annual Income (k$)', 'Spending Score (1-100)',
                'Cluster', 'Cluster_Nombre']]

# Guardar CSV
df_export.to_csv('data/processed/clientes_segmentados.csv', index=False)
print(f"✅ Datos exportados: {len(df_export)} clientes segmentados")
print(f"   Archivo: data/processed/clientes_segmentados.csv")
```

#### Paso 7.2: Generar Reporte de Texto

```python
from datetime import datetime

with open('results/reports/reporte_segmentacion.txt', 'w', encoding='utf-8') as f:
    f.write("═" * 100 + "\n")
    f.write("REPORTE DE SEGMENTACIÓN DE CLIENTES - K-MEANS CLUSTERING\n")
    f.write("═" * 100 + "\n\n")
    
    f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total de clientes analizados: {len(df)}\n")
    f.write(f"Número de clusters: {K_OPTIMO}\n")
    f.write(f"Features utilizadas: Ingresos Anuales, Spending Score\n\n")
    
    f.write("─" * 100 + "\n")
    f.write("MÉTRICAS DEL MODELO\n")
    f.write("─" * 100 + "\n")
    f.write(f"Inercia final: {kmeans_final.inertia_:.2f}\n")
    f.write(f"Iteraciones: {kmeans_final.n_iter_}\n\n")
    
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
    for cluster_id, profile in cluster_profiles.items():
        f.write(f"\nCLUSTER {cluster_id}: {profile['nombre']}\n")
        f.write(f"Descripción: {profile['descripcion']}\n")
        f.write(f"Tamaño: {profile['tamaño']} clientes\n")
        f.write("Características:\n")
        for char in profile['caracteristicas']:
            f.write(f"  • {char}\n")
    
    f.write("\n" + "─" * 100 + "\n")
    f.write("ESTRATEGIAS RECOMENDADAS\n")
    f.write("─" * 100 + "\n")
    for segmento, estrategia in marketing_strategies.items():
        f.write(f"\n{segmento}:\n")
        f.write(estrategia + "\n")

print("✅ Reporte generado: results/reports/reporte_segmentacion.txt")
```

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



## 🔧 TROUBLESHOOTING

### Problemas Comunes y Soluciones

#### Problema 1: "ModuleNotFoundError: No module named 'sklearn'"

**Solución**:
```bash
pip install scikit-learn
# o
conda install scikit-learn
```

#### Problema 2: El gráfico no muestra clusters claros

**Posibles causas**:
- K incorrecto → Revisar método del codo
- Features incorrectas → Probar con otras variables
- Datos no escalados → Aplicar StandardScaler

**Solución**:
```python
# Verificar escalado
print(X_scaled.mean(axis=0))  # Debe ser ~0
print(X_scaled.std(axis=0))   # Debe ser ~1

# Probar diferentes K
for k in [3, 4, 5, 6]:
    # Entrenar y visualizar
```

#### Problema 3: Todos los puntos en un solo cluster

**Causa**: Inercia muy alta, K=1 efectivo

**Solución**:
- Aumentar K
- Revisar outliers (eliminarlos con `df = df[df['columna'] < threshold]`)
- Probar con features diferentes

#### Problema 4: Jupyter Notebook no inicia

**Solución**:
```bash
# Reinstalar Jupyter
pip uninstall jupyter
pip install jupyter

# O usar JupyterLab
pip install jupyterlab
jupyter lab
```

#### Problema 5: Gráficos no se guardan

**Causa**: Carpeta no existe

**Solución**:
```python
import os
os.makedirs('results/figures', exist_ok=True)
```

#### Problema 6: Error "ConvergenceWarning"

**Causa**: El algoritmo no convergió en max_iter iteraciones

**Solución**:
```python
# Aumentar max_iter
kmeans = KMeans(n_clusters=5, max_iter=500, random_state=42)
```

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



## 📖 GLOSARIO DE TÉRMINOS

### Términos de Machine Learning

**Aprendizaje No Supervisado**: Técnica de ML donde el algoritmo encuentra patrones sin etiquetas previas.

**Clustering**: Agrupamiento de datos similares sin supervisión.

**Centroide**: Punto central de un cluster, calculado como promedio de todos los puntos.

**Inercia (WCSS)**: Suma de distancias al cuadrado de cada punto a su centroide.

**Convergencia**: Punto donde el algoritmo deja de mejorar significativamente.

**Feature**: Característica o variable del dataset (ej: edad, ingresos).

**Escalado**: Transformación de datos a una escala común.

**Overfitting**: Modelo demasiado ajustado a datos de entrenamiento.

**Underfitting**: Modelo demasiado simple que no captura patrones.

### Términos de Negocio

**Segmentación**: División de clientes en grupos homogéneos.

**RFM**: Recency (Recencia), Frequency (Frecuencia), Monetary (Valor monetario).

**Lifetime Value (LTV)**: Valor total que un cliente aporta durante su relación con la empresa.

**Churn**: Tasa de abandono de clientes.

**Customer Acquisition Cost (CAC)**: Costo de adquirir un nuevo cliente.

**Retention Rate**: Tasa de retención de clientes.

---



## 💡 TIPS Y MEJORES PRÁCTICAS

### Para Obtener Mejores Resultados

1. **Calidad de Datos**
   - ✨ Limpia datos antes de clustering
   - ✨ Elimina outliers extremos
   - ✨ Verifica consistencia

2. **Selección de Features**
   - ✨ Usa features relevantes al negocio
   - ✨ Evita multicolinealidad
   - ✨ Considera crear features derivadas

3. **Escalado**
   - ✨ SIEMPRE escala antes de K-Means
   - ✨ Usa StandardScaler como default
   - ✨ Considera RobustScaler si hay outliers

4. **Interpretación**
   - ✨ Valida clusters con expertos del negocio
   - ✨ Nombra clusters descriptivamente
   - ✨ Documenta suposiciones

5. **Mantenimiento**
   - ✨ Reentrena periódicamente
   - ✨ Monitorea tamaño de clusters
   - ✨ Ajusta K si es necesario

---


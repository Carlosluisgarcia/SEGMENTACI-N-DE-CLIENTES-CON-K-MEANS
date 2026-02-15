# Descripción Ampliada del Proyecto
## Segmentación de Clientes con K-Means Clustering
### Sistema de Inteligencia Artificial No Supervisada para Marketing Estratégico



# Nombre : Carlos Luis Garcia Lopez 
# Carrera : Ingieneria Informatica 
# Año : 3ro
---

## Introducción

En el contexto actual del comercio moderno, las empresas generan enormes volúmenes de datos sobre sus clientes a diario: registros de compras, frecuencia de visitas, nivel de ingresos, preferencias de productos y patrones de gasto. Sin embargo, almacenar esos datos no es suficiente. El verdadero valor reside en la capacidad de interpretar esa información y convertirla en decisiones estratégicas que impulsen el crecimiento del negocio. Es precisamente aquí donde la Inteligencia Artificial entra en juego.

El presente proyecto surge de una necesidad fundamental del mundo empresarial: comprender quiénes son realmente los clientes. No como una masa homogénea, sino como grupos de personas con características, comportamientos y necesidades distintas. Cuando una empresa lanza una campaña de marketing genérica dirigida a todos sus clientes por igual, inevitablemente desperdicia recursos: ofrece descuentos a quienes ya compran con frecuencia, ignora a quienes necesitan un incentivo para volver, y pierde oportunidades de fidelizar a sus clientes más rentables.

La segmentación de clientes resuelve este problema de raíz. Al dividir a la base de clientes en grupos homogéneos, es posible diseñar estrategias específicas para cada segmento, optimizar la inversión en marketing y mejorar significativamente la experiencia del cliente. Para lograr esta segmentación de forma automática, precisa y escalable, este proyecto implementa el algoritmo de aprendizaje automático conocido como **K-Means Clustering**, una de las técnicas más consolidadas dentro del campo del **aprendizaje no supervisado**.

El aprendizaje no supervisado es una rama de la Inteligencia Artificial que se distingue por su capacidad de encontrar patrones y estructuras ocultas en los datos sin necesidad de que un humano le indique de antemano qué buscar. A diferencia del aprendizaje supervisado, donde el algoritmo aprende a partir de ejemplos etiquetados, el aprendizaje no supervisado explora los datos de forma autónoma, identifica similitudes y agrupa los elementos que comparten características comunes. Esto lo convierte en la herramienta ideal cuando se desea descubrir perfiles de clientes que quizás ni siquiera el propio negocio sabía que existían.

El dataset utilizado en este proyecto es el conocido **"Mall Customer Segmentation Data"**, disponible en la plataforma Kaggle. Este conjunto de datos contiene información de 200 clientes de un centro comercial, incluyendo su edad, género, ingresos anuales expresados en miles de dólares, y una puntuación de gastos asignada por el propio establecimiento en una escala del 1 al 100. La combinación de estas variables ofrece una representación multidimensional del comportamiento del cliente que resulta ideal para aplicar técnicas de clustering.

Desde el punto de vista tecnológico, el proyecto se desarrolla íntegramente en **Python**, aprovechando el ecosistema de librerías científicas que lo convierten en el lenguaje de referencia para ciencia de datos e inteligencia artificial a nivel mundial. Se emplean herramientas como **pandas** y **numpy** para la manipulación y el análisis de datos, **scikit-learn** para la implementación del algoritmo de machine learning, y **matplotlib** junto con **seaborn** para la generación de visualizaciones profesionales que facilitan la interpretación de los resultados.

El proyecto está diseñado no solo como un ejercicio académico, sino como una solución aplicable a un negocio real. Toda la arquitectura, los módulos de código y la metodología empleada están pensados para que el sistema pueda adaptarse a datos reales de cualquier empresa comercial, con el objetivo final de transformar datos crudos en inteligencia de negocio accionable.

---

## Desarrollo

### 1. Exploración y Comprensión de los Datos

El primer paso del proyecto es la **Exploración de Datos**, conocida por sus siglas en inglés como EDA (Exploratory Data Analysis). Antes de aplicar cualquier algoritmo de inteligencia artificial, resulta indispensable comprender la naturaleza de los datos con los que se trabaja. Esta fase permite detectar posibles problemas como valores nulos, inconsistencias o valores extremos que podrían afectar la calidad del modelo.

En este proyecto, el análisis exploratorio revela que el dataset está perfectamente limpio: no contiene valores nulos y todas sus variables presentan rangos coherentes con la realidad. La edad de los clientes oscila entre 18 y 70 años, los ingresos anuales varían entre 15 y 137 mil dólares, y la puntuación de gastos distribuye de manera relativamente uniforme entre 1 y 99 puntos. La distribución de género muestra una ligera predominancia femenina, con aproximadamente el 56% de clientes mujeres frente al 44% de hombres.

El análisis bivariado resulta especialmente revelador. Al representar los ingresos anuales frente a la puntuación de gastos en un gráfico de dispersión, es posible apreciar a simple vista la existencia de aproximadamente cinco agrupaciones naturales de clientes, lo que justifica y anticipa la elección del número de clusters en fases posteriores. Esta visualización temprana es uno de los momentos más importantes del proceso, pues confirma que los datos contienen una estructura latente que el algoritmo podrá descubrir y formalizar.

### 2. Preprocesamiento de Datos

La segunda fase del proyecto se ocupa de preparar los datos para que el algoritmo K-Means pueda procesarlos correctamente. Esta etapa incluye dos transformaciones fundamentales.

La primera es la **codificación de variables categóricas**. El algoritmo K-Means trabaja exclusivamente con datos numéricos, por lo que la variable "Gender", que contiene los valores "Male" y "Female", debe convertirse a una representación numérica. Para ello se aplica un Label Encoder, que asigna el valor 0 a "Female" y 1 a "Male", preservando la información original en un formato que el algoritmo puede interpretar.

La segunda transformación, y quizás la más crítica de todo el proceso, es el **escalado de datos**. K-Means calcula distancias euclidianas entre puntos para determinar a qué cluster pertenece cada cliente. Si las variables tienen escalas muy diferentes entre sí, las de mayor magnitud dominarán el cálculo de distancias y distorsionarán los resultados. Por ejemplo, los ingresos anuales se expresan en valores entre 15 y 137, mientras que la puntuación de gastos va de 1 a 99. Aunque ambos rangos son similares en este caso, en proyectos reales las diferencias pueden ser mucho más pronunciadas. Para garantizar que todas las variables contribuyan equitativamente al cálculo, se aplica el **StandardScaler**, que transforma cada variable restando su media y dividiendo por su desviación estándar, de modo que todas queden centradas en cero con varianza unitaria.

Para este proyecto se definen dos conjuntos de variables o features: un modelo simple que utiliza únicamente los ingresos anuales y la puntuación de gastos, pensado para facilitar la visualización en dos dimensiones, y un modelo completo que incorpora también la edad y el género codificado, para un análisis más exhaustivo.

### 3. Determinación del Número Óptimo de Clusters

Uno de los aspectos más importantes al aplicar K-Means es decidir cuántos clusters crear, un parámetro que el algoritmo no puede determinar por sí solo y que debe ser definido por el analista. Para tomar esta decisión de forma objetiva y fundamentada, el proyecto implementa dos técnicas complementarias.

La primera es el **Método del Codo** (Elbow Method). Este método consiste en entrenar el algoritmo con diferentes valores de K, generalmente de 1 a 10, y registrar en cada caso la inercia del modelo, es decir, la suma de las distancias al cuadrado de cada punto respecto al centroide de su cluster. Al representar estas inercias en un gráfico, la curva resultante tiende a descender rápidamente al principio y luego a estabilizarse. El punto donde se produce esa inflexión, visualmente similar al "codo" de un brazo, indica el número óptimo de clusters, pues a partir de ese valor añadir más grupos no aporta una mejora significativa en la compacidad de los clusters.

La segunda técnica es el **Silhouette Score**, una métrica que evalúa qué tan bien definidos están los clusters. Para cada punto de datos, calcula qué tan similar es a los puntos de su propio cluster en comparación con los puntos del cluster más cercano. El resultado es un valor entre -1 y 1, donde valores próximos a 1 indican clusters bien separados y cohesionados, valores próximos a 0 sugieren solapamiento entre clusters, y valores negativos indican que los puntos podrían estar mal asignados.

Para el dataset de clientes del centro comercial, ambas técnicas convergen en indicar que **K=5** es el número óptimo de clusters, lo que confirma la observación visual realizada durante la exploración de datos.

### 4. Entrenamiento del Modelo K-Means

Con el número de clusters definido, el proceso de entrenamiento del modelo puede describirse en los siguientes pasos. En primer lugar, el algoritmo inicializa aleatoriamente cinco centroides en el espacio de datos escalados. A continuación, asigna cada uno de los 200 clientes al centroide más cercano, calculando la distancia euclidiana entre el punto que representa al cliente y cada uno de los cinco centroides. Una vez completada la asignación, recalcula la posición de cada centroide como el promedio de todos los clientes que le han sido asignados. Este proceso de asignación y recalculación se repite iterativamente hasta que los centroides dejan de moverse de forma significativa, lo que se conoce como convergencia del algoritmo.

El modelo se configura con parámetros específicos para garantizar robustez y reproducibilidad: se realizan diez inicializaciones independientes con diferentes posiciones aleatorias para los centroides, seleccionando al final la configuración que produce la menor inercia; se establece un máximo de 300 iteraciones por inicialización; y se fija una semilla aleatoria para asegurar que los resultados sean reproducibles cada vez que se ejecute el código.

Una vez entrenado, el modelo genera cinco etiquetas de cluster, una por cada cliente del dataset, indicando a qué grupo pertenece cada uno.

### 5. Análisis e Interpretación de los Clusters

Esta fase representa la transición del análisis técnico al análisis de negocio, y es posiblemente la más valiosa de todo el proyecto. Una vez que el algoritmo ha identificado los cinco grupos de clientes, es necesario examinar las características de cada uno para asignarles una interpretación significativa desde el punto de vista comercial.

El análisis de los centroides, expresados en escala original gracias a la inversión del proceso de escalado, revela cinco perfiles de cliente claramente diferenciados:

El primer segmento, denominado **"VIP High Spenders"**, agrupa a clientes con ingresos elevados y puntuación de gastos también alta. Son los clientes más rentables del negocio, dispuestos a gastar en proporción a sus ingresos. Representan el segmento premium que toda empresa desea retener y cultivar.

El segundo segmento, los **"Jóvenes Gastadores"**, reúne a clientes con ingresos relativamente bajos pero puntuación de gastos muy alta. Son clientes, generalmente más jóvenes, que gastan una proporción elevada de sus ingresos. Su alta predisposición al gasto los convierte en un segmento de gran potencial para crecer hacia la categoría VIP.

El tercer segmento, los **"Conservadores de Alto Ingreso"**, presenta la combinación opuesta: ingresos altos pero puntuación de gastos baja. Son clientes que podrían gastar mucho más pero no lo hacen, lo que representa una oportunidad de negocio enorme si se logra identificar las barreras que frenan su consumo.

El cuarto segmento, los **"Precavidos de Ingresos Medios"**, agrupa a clientes con ingresos y gastos moderados. Representan el segmento más estable y numeroso, y aunque su rentabilidad individual es menor, su volumen los convierte en un pilar importante de los ingresos totales.

El quinto segmento, denominado **"Oportunidad de Crecimiento"**, reúne a clientes con ingresos bajos y puntuación de gastos también baja. Son el segmento más sensible al precio y el que mayor esfuerzo requiere para convertir, pero su volumen potencial los hace relevantes en estrategias de penetración de mercado.

### 6. Estrategias de Marketing por Segmento

La utilidad práctica del proyecto se concreta en el diseño de estrategias de marketing diferenciadas para cada segmento. En lugar de lanzar una única campaña genérica, el negocio puede ahora comunicarse con cada grupo de clientes de forma personalizada, maximizando la efectividad de cada acción.

Para los clientes VIP, la estrategia se centra en la exclusividad y la experiencia: programas de lealtad premium, acceso anticipado a nuevos productos, atención personalizada y eventos exclusivos. Para los Jóvenes Gastadores, el enfoque pasa por las redes sociales, los programas de referidos y la gamificación. Para los Conservadores de Alto Ingreso, la clave es demostrar valor y calidad a través de contenido educativo, testimonios y garantías. Para los Precavidos, las promociones estacionales y los programas de puntos acumulables resultan más efectivos. Para el segmento de Oportunidad, las estrategias de entrada a precio accesible y los sistemas de financiamiento son el punto de partida.

### 7. Visualización de Resultados

El proyecto genera un conjunto completo de visualizaciones profesionales que permiten comunicar los hallazgos de forma clara y comprensible. Los gráficos de distribución univariada muestran cómo se distribuyen individualmente cada una de las variables del dataset. Los gráficos de dispersión bivariados permiten explorar las relaciones entre pares de variables e identificar visualmente los grupos antes de aplicar el algoritmo. La curva del método del codo documenta el proceso de selección del K óptimo. El gráfico principal de segmentación en dos dimensiones muestra los cinco clusters con sus respectivos centroides, codificados por colores para facilitar su identificación. La versión etiquetada del mismo gráfico reemplaza los números de cluster por sus nombres descriptivos de negocio, haciéndolo comprensible para cualquier persona independientemente de su formación técnica. Finalmente, la visualización tridimensional incorpora la variable edad como tercera dimensión, ofreciendo una perspectiva adicional sobre la distribución de los clusters.

### 8. Arquitectura del Código

El proyecto sigue una arquitectura modular y bien estructurada, dividiendo las responsabilidades en seis archivos Python independientes que se comunican entre sí. El archivo `config.py` centraliza toda la configuración del proyecto, incluyendo rutas, parámetros del modelo y definiciones de los perfiles de cluster. El módulo `data_loader.py` gestiona la carga y validación del dataset. El módulo `preprocessing.py` implementa todas las transformaciones necesarias para preparar los datos. El módulo `clustering.py` contiene la implementación completa del modelo K-Means, incluyendo el método del codo, el entrenamiento, la evaluación y las funciones de predicción para nuevos clientes. El módulo `visualization.py` agrupa todas las funciones de generación de gráficos. Finalmente, `main.py` actúa como orquestador del pipeline completo, ejecutando todas las fases en secuencia y generando los resultados finales.

Esta arquitectura facilita el mantenimiento del código, permite reutilizar módulos individuales en otros proyectos y hace que el sistema sea fácilmente escalable para incorporar nuevas funcionalidades en el futuro.

---

## Conclusiones

El desarrollo de este proyecto permite extraer conclusiones significativas tanto desde la perspectiva técnica como desde la perspectiva del impacto en el negocio.

Desde el punto de vista técnico, el proyecto demuestra que el algoritmo K-Means, a pesar de su aparente simplicidad conceptual, es una herramienta extraordinariamente poderosa cuando se aplica correctamente. La elección adecuada del número de clusters mediante el método del codo, combinada con un preprocesamiento riguroso que incluye el escalado de variables, produce resultados sólidos y reproducibles. Las métricas de evaluación, incluyendo el Silhouette Score y el índice de Davies-Bouldin, confirman que los cinco clusters identificados presentan buena separación y cohesión interna, lo que valida la calidad del modelo.

Desde la perspectiva del negocio, los resultados son igualmente contundentes. El análisis revela cinco perfiles de cliente con características suficientemente distintas como para justificar estrategias de marketing diferenciadas. La existencia de un segmento de clientes con altos ingresos y baja puntuación de gastos, los Conservadores de Alto Ingreso, es quizás el hallazgo más valioso del proyecto: identifica un grupo de clientes que posee el poder adquisitivo para gastar significativamente más, pero que por alguna razón no lo hace. Activar este segmento podría representar un incremento sustancial en los ingresos del negocio sin necesidad de captar nuevos clientes.

El proyecto confirma también que la segmentación no es un ejercicio estático sino dinámico. A medida que el negocio evoluciona, los clientes cambian de segmento: un Joven Gastador que aumenta sus ingresos puede convertirse en un VIP High Spender; un cliente Precavido que experimenta una mejora económica puede trasladarse al segmento Conservador. Por ello, se recomienda reentre nar el modelo periódicamente, al menos de forma trimestral, para que los perfiles de segmentación reflejen siempre la realidad actual de la base de clientes.

Una de las fortalezas más importantes del proyecto es su aplicabilidad directa a negocios reales. Todo el sistema está diseñado para poder adaptarse a los datos de cualquier empresa comercial. En el caso de L'Luis, la integración del análisis RFM (Recency, Frequency, Monetary) como base para la segmentación ofrecería un enfoque especialmente adecuado para el contexto de un negocio de comercio minorista, permitiendo identificar clientes frecuentes, clientes en riesgo de abandono, clientes de alto valor y clientes que acaban de incorporarse al negocio.

El proyecto sienta también las bases para evoluciones futuras de mayor complejidad. Una vez dominado K-Means, el siguiente paso natural es explorar algoritmos más avanzados como DBSCAN, que no requiere especificar el número de clusters de antemano y es capaz de detectar clusters de forma irregular, o los Modelos de Mezcla Gaussiana, que asignan probabilidades de pertenencia a los clusters en lugar de asignaciones binarias. A más largo plazo, la integración de los resultados de segmentación con un sistema de recomendación de productos o con un motor de personalización de contenidos representaría un salto cualitativo en la capacidad del negocio para ofrecer experiencias verdaderamente individualizadas.

En definitiva, este proyecto representa mucho más que un ejercicio de programación o de análisis estadístico. Es una demostración concreta de cómo la Inteligencia Artificial puede transformar datos brutos en conocimiento accionable, reducir la incertidumbre en la toma de decisiones de negocio y crear una ventaja competitiva sostenible para cualquier empresa que decida adoptar una cultura basada en datos. La segmentación de clientes mediante K-Means es el punto de partida de un camino más amplio hacia la personalización masiva, la predicción del comportamiento del cliente y la optimización continua de la experiencia de compra.

---

*Documento elaborado por Carlos — L'Luis*
*Pinar del Río, Cuba — Enero 2026*

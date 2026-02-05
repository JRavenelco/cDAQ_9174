# **Convergencia de la Inteligencia Artificial Científica y la Mecatrónica: Optimización Metaheurística, Arquitecturas Informadas por la Física y Aceleración Masiva en JAX para Sistemas de Levitación Magnética**

## **1\. Introducción: El Nuevo Paradigma del Modelado Híbrido en Ingeniería**

La ingeniería contemporánea se encuentra en una encrucijada crítica donde los métodos analíticos tradicionales y las técnicas empíricas basadas puramente en datos convergen hacia un nuevo paradigma: el Aprendizaje Automático Científico (SciML, por sus siglas en inglés). Históricamente, el modelado de sistemas dinámicos complejos, como los actuadores de levitación magnética o los sistemas de rodamientos magnéticos activos, ha dependido de la derivación rigurosa de ecuaciones diferenciales basadas en primeros principios —las leyes de Newton, las ecuaciones de Maxwell y los principios de la termodinámica—. Si bien estos modelos de "caja blanca" ofrecen una interpretabilidad perfecta y garantías teóricas de estabilidad, a menudo simplifican en exceso las no linealidades inherentes del mundo real, como la fricción dinámica, la saturación magnética, la histéresis y las incertidumbres paramétricas que surgen de la degradación de materiales o tolerancias de fabricación.1

Por otro lado, el advenimiento del aprendizaje profundo (Deep Learning) ha propiciado el auge de modelos de "caja negra", capaces de aproximar funciones de complejidad arbitraria a partir de grandes volúmenes de datos. Sin embargo, en aplicaciones críticas de ingeniería mecatrónica, estos modelos presentan limitaciones severas: requieren cantidades prohibitivas de datos etiquetados, carecen de capacidad de generalización fuera de la distribución de entrenamiento y, lo más preocupante, pueden producir predicciones que violan leyes físicas fundamentales, como la conservación de la energía o la masa.3

En respuesta a estas dicotomías, las Redes Neuronales Informadas por la Física (PINNs) han emergido como una arquitectura transformadora. Al integrar las ecuaciones gobernantes del sistema directamente en la función de pérdida de la red neuronal, las PINNs actúan como un mecanismo de regularización que restringe el espacio de búsqueda a soluciones físicamente plausibles.3 No obstante, la adopción de PINNs no está exenta de desafíos. El entrenamiento de estas redes implica la optimización de una función de pérdida altamente no convexa, plagada de mínimos locales y paisajes de error patológicos que confunden a los optimizadores tradicionales basados en gradientes, como el descenso de gradiente estocástico (SGD) o Adam.5

Este informe postula y fundamenta que la solución a estos desafíos reside en la integración sinérgica de tres fronteras tecnológicas: (1) el uso de **algoritmos metaheurísticos avanzados** (evolutivos y de enjambre) para la optimización global robusta; (2) la adopción de nuevas arquitecturas de redes interpretables como las **Redes de Kolmogorov-Arnold (KANs)**; y (3) la implementación de estos algoritmos en plataformas de **diferenciación automática acelerada por hardware como JAX**, que permiten la paralelización masiva necesaria para hacer viable la computación evolutiva a gran escala.7 A través de un análisis exhaustivo que abarca desde la teoría fundamental hasta la aplicación práctica en sistemas de levitación magnética —utilizando el modelo de inductancia de Santana y estrategias de "Juez de Energía"—, este documento establece una hoja de ruta para la próxima generación de identificación y control de sistemas mecatrónicos.

## **2\. Fundamentos Teóricos: De las PINNs a las Evo-PINNs**

### **2.1 La Arquitectura de las Redes Neuronales Informadas por la Física**

Las PINNs representan un cambio fundamental en la forma en que concebimos el aprendizaje automático. En lugar de minimizar únicamente el error entre la predicción y el dato observado, una PINN minimiza un residual físico. Considere un sistema dinámico gobernado por una ecuación diferencial parcial o ordinaria general de la forma:

$$\\mathcal{N}\[u\](t, \\mathbf{x}) \= f(t, \\mathbf{x})$$  
Donde $u(t, \\mathbf{x})$ es la solución desconocida, $\\mathcal{N}$ es un operador diferencial no lineal y $f$ es una función de forzamiento. En una PINN, la red neuronal $\\hat{u}(t, \\mathbf{x}; \\theta)$ con parámetros $\\theta$ se entrena minimizando una función de pérdida compuesta $\\mathcal{L}(\\theta)$:

$$\\mathcal{L}(\\theta) \= w\_{datos}\\mathcal{L}\_{datos} \+ w\_{fisica}\\mathcal{L}\_{fisica} \+ w\_{ci}\\mathcal{L}\_{ci}$$  
El término $\\mathcal{L}\_{fisica}$ evalúa el residual de la ecuación diferencial $||\\mathcal{N}\[\\hat{u}\] \- f||^2$ en un conjunto de puntos de colocación dispersos en el dominio, calculados mediante diferenciación automática. Esto permite que la red aprenda la física subyacente incluso en ausencia de datos experimentales densos, actuando eficazmente como un solucionador de ecuaciones diferenciales basado en datos.3

### **2.2 El Problema del Paisaje de Pérdida y el Fallo del Gradiente**

A pesar de su elegancia, la literatura reciente, incluidas las revisiones exhaustivas de Karniadakis y otros pioneros 3, destaca una debilidad crítica: la optimización. Al introducir derivadas de alto orden en la función de pérdida, el paisaje de optimización se vuelve extremadamente "rígido" (stiff) y complejo. Los métodos basados en gradientes tienden a converger prematuramente en mínimos locales espurios o sufren de desvanecimiento de gradientes debido al desequilibrio entre los diferentes términos de la pérdida (por ejemplo, cuando el gradiente de la condición de frontera domina sobre el de la ecuación diferencial).6

Además, existe el fenómeno del "sesgo espectral", donde las redes neuronales densas tienden a aprender componentes de baja frecuencia mucho más rápido que los detalles de alta frecuencia. En sistemas caóticos o con dinámicas rápidas como la levitación magnética, esto resulta en modelos que capturan la tendencia general pero fallan catastróficamente en predecir transitorios rápidos o comportamientos turbulentos.10

### **2.3 El Surgimiento de Evo-PINN: Metaheurísticas al Rescate**

Para superar estas barreras, ha surgido el campo de **Evo-PINN** (Evolutionary Optimization of Physics-Informed Neural Networks).12 Las metaheurísticas —algoritmos inspirados en procesos naturales como la evolución biológica o el comportamiento social de animales— ofrecen mecanismos de búsqueda global que no dependen del gradiente local.

El análisis de los trabajos de Wong, Gupta, Ong y Karniadakis 3 revela que las metaheurísticas aportan ventajas decisivas:

1. **Exploración Global:** Mantienen una población de soluciones diversas, explorando múltiples regiones del espacio de parámetros simultáneamente, lo que reduce drásticamente el riesgo de estancamiento local.6  
2. **Indiferencia a la Diferenciabilidad:** Permiten optimizar arquitecturas de red, funciones de activación e hiperparámetros discretos que no son diferenciables, facilitando la Búsqueda de Arquitectura Neuronal (NAS) informada por la física.3  
3. **Manejo de Objetivos en Conflicto:** Los algoritmos evolutivos multiobjetivo (MOEA) pueden encontrar un frente de Pareto óptimo entre la precisión de los datos y el cumplimiento de la física, en lugar de depender de pesos $w$ arbitrarios.3

## **3\. Algoritmos Metaheurísticos en la Identificación de Sistemas**

El documento técnico analizado 7, junto con la literatura de soporte 1, identifica un conjunto robusto de algoritmos metaheurísticos aplicables a la identificación de parámetros en sistemas dinámicos (grey-box modeling). A continuación, se detalla la mecánica y la relevancia de cada uno en este contexto.

### **3.1 Evolución Diferencial (DE)**

La Evolución Diferencial se destaca como uno de los algoritmos más robustos para problemas de optimización continua en espacios reales. Su mecanismo de mutación, que genera nuevos vectores perturbando un individuo existente con la diferencia ponderada de otros dos miembros de la población ($v\_i \= x\_{r1} \+ F \\cdot (x\_{r2} \- x\_{r3})$), le confiere una capacidad intrínseca para adaptarse a la escala del problema y mantener la diversidad poblacional.1 En el contexto de la identificación del modelo de levitación magnética 7, DE demostró una capacidad superior para navegar el paisaje de pérdida altamente correlacionado de los parámetros de inductancia y masa, evitando los mínimos locales donde otros algoritmos fallaban.

### **3.2 Optimización de Enjambre de Partículas (PSO)**

Inspirado en el comportamiento social de las bandadas de aves, PSO mueve partículas a través del espacio de búsqueda influenciadas por su propia mejor posición conocida y la mejor posición del enjambre. Es particularmente eficaz para la convergencia rápida en las etapas iniciales de la optimización.14 Sin embargo, en problemas de alta dimensionalidad o con paisajes muy irregulares (como los generados por ecuaciones diferenciales rígidas), PSO puede sufrir de convergencia prematura si no se implementan mecanismos de control de inercia adaptativos.16

### **3.3 Estrategias de Evolución y CMA-ES**

La Estrategia de Evolución con Adaptación de la Matriz de Covarianza (CMA-ES) es ampliamente considerada el estado del arte en optimización continua de caja negra para problemas difíciles y mal condicionados. CMA-ES modela la distribución de soluciones prometedoras mediante una distribución normal multivariada y adapta la matriz de covarianza para guiar la búsqueda a lo largo de los valles de la función objetivo.11 Aunque computacionalmente costoso ($O(N^2)$), su capacidad para manejar parámetros altamente correlacionados (como $k\_0$ y $a$ en el modelo de inductancia de Santana) lo hace invaluable para el refinamiento final de parámetros en PINNs.7

### **3.4 Algoritmos Emergentes y Bio-inspirados**

La literatura reciente revisada 7 muestra una proliferación de nuevos algoritmos aplicados a problemas de ingeniería:

* **Grey Wolf Optimizer (GWO):** Simula la jerarquía de liderazgo y los mecanismos de caza de los lobos grises. Ha demostrado eficacia en el control de sistemas no lineales.7  
* **Whale Optimization Algorithm (WOA):** Imita la técnica de caza con redes de burbujas de las ballenas jorobadas, equilibrando eficazmente la exploración y la explotación.7  
* **Harris Hawks Optimization (HHO):** Basado en estrategias de asedio cooperativo, útil para escapar de óptimos locales en paisajes multimodales.7  
* **Arithmetic Optimization Algorithm (AOA) y Salp Swarm Algorithm (SSA):** Algoritmos adicionales explorados para diversificar las estrategias de búsqueda.7

### **3.5 El Efecto Baldwin y la Hibridación**

Una tendencia crucial identificada en la investigación de Wong et al. 21 es la incorporación del **Efecto Baldwin** en la optimización de PINNs. Este enfoque híbrido combina la evolución global (aprendizaje filogenético) con un refinamiento local basado en gradientes o aprendizaje durante la vida del individuo (aprendizaje ontogenético). En el contexto de Evo-PINN, esto implica usar un algoritmo evolutivo para encontrar una configuración inicial de pesos y parámetros físicos cercanos al óptimo global, y luego aplicar pasos de descenso de gradiente (como L-BFGS o Adam) para converger rápidamente a la solución precisa. Esta estrategia mitiga las debilidades de ambos enfoques: la lentitud de convergencia final de los EAs y la sensibilidad a la inicialización de los métodos de gradiente.21

## **4\. Aceleración Computacional: La Revolución de JAX y GPU**

La principal crítica histórica a los métodos metaheurísticos ha sido su ineficiencia computacional. Evaluar una función de aptitud compleja (que implica resolver numéricamente ecuaciones diferenciales o ejecutar una red neuronal) para miles de individuos durante cientos de generaciones resultaba prohibitivo. Sin embargo, la aparición de bibliotecas de diferenciación automática de alto rendimiento como **JAX** ha eliminado esta barrera.3

### **4.1 JAX: Transformaciones de Funciones Componibles**

JAX, desarrollado por Google Research, no es simplemente una biblioteca de álgebra lineal; es un sistema extensible de transformaciones de funciones. Sus características son fundamentales para la nueva ola de algoritmos metaheurísticos masivamente paralelos:

* **Compilación Just-In-Time (jit):** JAX compila funciones de Python en kernels XLA (Accelerated Linear Algebra) optimizados para GPU y TPU. Esto permite que operaciones complejas, como la integración numérica de un sistema dinámico (ej. Runge-Kutta 4), se ejecuten con la eficiencia de código C++ o CUDA nativo, fusionando operaciones para minimizar el ancho de banda de memoria.5  
* **Vectorización Automática (vmap):** Esta es la "pieza clave" para las metaheurísticas. vmap permite escribir una función para evaluar *un solo* individuo y transformarla automáticamente en una función que evalúa *toda una población* en paralelo. En una GPU moderna, esto significa que evaluar 10,000 individuos puede tomar casi el mismo tiempo que evaluar 100, aprovechando los miles de núcleos CUDA disponibles.7  
* **Diferenciación Automática (grad):** Permite calcular gradientes exactos de funciones arbitrarias. Esto es esencial no solo para el entrenamiento de redes neuronales, sino también para algoritmos híbridos que requieren derivadas de la física respecto a los parámetros.8

### **4.2 EvoJAX y la Neuroevolución Acelerada por Hardware**

El proyecto **EvoJAX** 3 ejemplifica el poder de este enfoque. Al implementar algoritmos evolutivos completamente en JAX, se eliminan los cuellos de botella de transferencia de datos entre la CPU (donde tradicionalmente vivía el algoritmo genético) y la GPU (donde se evaluaba la red). Todo el ciclo evolutivo ocurre en la GPU, logrando aceleraciones de órdenes de magnitud (10x-100x) en comparación con implementaciones estándar. Esto permite escalar la neuroevolución a redes profundas y problemas complejos de física que antes eran intratables.6

El documento de investigación 7 demuestra una implementación práctica de este paradigma, codificando desde cero 9 metaheurísticos diferentes utilizando primitivas de JAX (jax.random, jax.lax.scan) para la identificación de parámetros de un levitador magnético, confirmando la viabilidad y accesibilidad de esta tecnología para la investigación aplicada.

## **5\. Nuevas Fronteras Arquitectónicas: Redes de Kolmogorov-Arnold (KANs)**

Mientras que las PINNs tradicionales se basan en Perceptrones Multicapa (MLPs), una arquitectura emergente promete revolucionar la interpretabilidad y eficiencia en SciML: las **Redes de Kolmogorov-Arnold (KANs)**.23

### **5.1 Teorema de Representación y Estructura KAN**

A diferencia de las MLPs, que tienen funciones de activación fijas en las neuronas y pesos aprendibles en las conexiones, las KANs se fundamentan en el Teorema de Representación de Kolmogorov-Arnold. Este teorema establece que cualquier función continua multivariada puede representarse como una composición finita de funciones continuas de una sola variable y sumas. En una arquitectura KAN, **las funciones de activación son aprendibles y residen en las conexiones (aristas)**, mientras que los nodos simplemente suman las señales. Estas funciones univariadas se parametrizan típicamente mediante B-splines o polinomios de Chebyshev.26

### **5.2 PIKANs: KANs Informadas por la Física**

La adaptación de KANs para resolver ecuaciones diferenciales ha dado lugar a las **Physics-Informed KANs (PIKANs)**. Estas redes presentan ventajas sustanciales sobre las PINNs basadas en MLP para la identificación de sistemas mecatrónicos:

1. **Interpretabilidad y Regresión Simbólica:** Al aprender funciones univariadas explícitas en las conexiones, es posible visualizar y extraer la forma matemática exacta de las relaciones físicas aprendidas (por ejemplo, descubrir que la fuerza magnética decae con el cuadrado de la distancia) en lugar de operar como una caja negra opaca.26  
2. **Eficiencia Paramétrica:** Estudios recientes 23 indican que las PIKANs pueden alcanzar una precisión comparable o superior a las PINNs utilizando un número de parámetros significativamente menor. Esto es crucial para la implementación en tiempo real en controladores embebidos de sistemas mecatrónicos.  
3. **Captura de Dinámicas de Alta Frecuencia:** Debido a la naturaleza local de las funciones base (B-splines), las KANs mitigan el problema del olvido catastrófico y el sesgo espectral, permitiendo modelar mejor las dinámicas rápidas y transitorias típicas de los sistemas de potencia y levitación magnética.23

### **5.3 Desafíos de Entrenamiento y el Rol de las Metaheurísticas**

A pesar de sus beneficios, las KANs presentan un desafío de optimización mayor debido a la complejidad de ajustar los coeficientes de los splines y la estructura topológica. Aquí, nuevamente, los algoritmos metaheurísticos y la aceleración JAX son vitales. La optimización evolutiva puede navegar eficazmente el espacio de funciones de activación aprendibles, evitando los mínimos locales que atrapan a los métodos de gradiente en estas arquitecturas novedosas.30

## **6\. Estudio de Caso Profundo: Levitación Magnética e Identificación Robusta**

El sistema de levitación magnética (MagLev) sirve como el banco de pruebas definitivo para validar estas metodologías. Es un sistema intrínsecamente inestable, altamente no lineal y rápido, lo que lo convierte en un candidato ideal para demostrar la superioridad del enfoque Evo-PINN/PIKAN sobre los métodos tradicionales.

### **6.1 Formulación Lagrangiana y Dinámica del Sistema**

El modelado físico riguroso es el primer paso. Basándose en la mecánica Lagrangiana 2, la dinámica de una esfera ferromagnética levitando bajo un electroimán se deriva de la energía cinética $\\mathcal{T}$, la energía potencial gravitatoria $\\mathcal{V}$ y la co-energía magnética $W\_m'$:

$$\\mathcal{L} \= \\mathcal{T} \- \\mathcal{V} \+ W\_m' \= \\frac{1}{2}m\\dot{y}^2 \- mgy \+ \\frac{1}{2}L(y)i^2$$  
Aplicando las ecuaciones de Euler-Lagrange, obtenemos el modelo dinámico completo:

Ecuación Mecánica (Balance de Fuerzas):

$$m\\ddot{y} \= mg \- F\_{mag}(y, i) \= mg \- \\frac{1}{2}i^2 \\frac{\\partial L(y)}{\\partial y}$$

La fuerza magnética $F\_{mag}$ se deriva del cambio de inductancia con la posición. Dado que la inductancia aumenta al acercarse el objeto ($\\partial L / \\partial y \< 0$), la fuerza es atractiva y se opone a la gravedad.7  
Ecuación Eléctrica (Kirchhoff con Acoplamiento Electromecánico):

$$u(t) \= R(t)i \+ \\frac{d}{dt}(L(y)i) \= (R\_0 \+ \\alpha t)i \+ L(y)\\frac{di}{dt} \+ i\\frac{\\partial L}{\\partial y}\\frac{dy}{dt}$$

Este modelo incorpora no solo la resistencia óhmica $R\_0$, sino también la deriva térmica ($\\alpha t$) y, crucialmente, la fuerza contraelectromotriz inducida por el movimiento ($i \\frac{\\partial L}{\\partial y} \\dot{y}$), que acopla la velocidad mecánica con la dinámica de la corriente.7

### **6.2 El Modelo de Inductancia de Santana**

La precisión del modelo depende críticamente de cómo se define la inductancia $L(y)$. Los modelos lineales simples ($L \\propto 1/y$) fallan cerca del imán debido a la saturación y la geometría finita. El análisis de los documentos 7 destaca la superioridad del **Modelo de Inductancia de Santana** (referenciado a trabajos de Antonio Santana y colaboradores):

$$L(y) \= k\_0 \+ \\frac{k}{1 \+ y/a}$$  
Donde:

* $k\_0$: Inductancia de fuga (cuando $y \\to \\infty$).  
* $k$: Coeficiente de magnetización efectivo.  
* $a$: Parámetro geométrico que define la curvatura de la no linealidad.

Esta formulación es físicamente consistente (no singular en $y=0$) y captura la saturación. Su derivada, necesaria para la fuerza magnética, es:

$$\\frac{dL}{dy} \= \-\\frac{k}{a(1 \+ y/a)^2}$$

La identificación precisa de los parámetros $(k\_0, k, a)$ es esencial para un control estable.7

### **6.3 Identificación de Parámetros: El Desafío de la "Masa Invisible"**

Un hallazgo clave en 7 es la dificultad de identificar simultáneamente la masa $m$ y los parámetros magnéticos en condiciones de casi-equilibrio. Si la aceleración es baja ($\\ddot{y} \\approx 0$), la ecuación mecánica se reduce a $mg \\approx F\_{mag}$. Esto crea una multiplicidad de soluciones: un error en la masa estimada puede ser compensado por un error proporcional en la fuerza magnética, haciendo que la masa sea "invisible" u observacionalmente indistinguible.

Para resolver esto, se proponen dos innovaciones metodológicas fundamentales:

1. **Filtrado de Zonas de Alta Dinámica:** Se procesan selectivamente solo los segmentos de datos donde la aceleración del objeto supera un umbral estadístico (e.g., percentil 75). En estos regímenes transitorios, el término inercial $m\\ddot{y}$ domina la dinámica, rompiendo la simetría y haciendo observable la masa real.7  
2. El "Juez de Energía" (Energy Judge): Se introduce una función de aptitud física basada en el balance de potencia instantánea, que es una ley de conservación más estricta que el simple error de seguimiento de trayectoria:

   $$P\_{in} \= P\_{Joule} \+ \\frac{d}{dt}E\_{mag} \+ \\frac{d}{dt}E\_{mec}$$  
   $$u \\cdot i \= Ri^2 \+ \\frac{d}{dt}\\left(\\frac{1}{2}Li^2\\right) \+ \\frac{d}{dt}\\left(\\frac{1}{2}m\\dot{y}^2 \+ mgy\\right)$$

   En esta ecuación, la masa $m$ aparece multiplicando a la velocidad y la aceleración con una firma temporal única. Esto permite a los algoritmos metaheurísticos "descubrir" la masa verdadera del objeto (aprox. 9g en el estudio de caso) sin necesidad de pesarla a priori, validando la coherencia física del modelo identificado.7

### **6.4 Aplicación de Omar Rodríguez-Abreo y la Identificación "Grey-Box"**

El trabajo del investigador **Omar Rodríguez-Abreo** 36 es central en este dominio. Sus investigaciones se centran en la identificación de "caja gris" (grey-box), donde la estructura del modelo (ecuaciones físicas) es conocida, pero los parámetros son desconocidos. Rodríguez-Abreo ha demostrado que el uso de metaheurísticas (como algoritmos genéticos y optimización de enjambre) para ajustar estos parámetros físicos supera a los métodos de regresión tradicionales, especialmente cuando se integran restricciones físicas y se utilizan respuestas dinámicas (escalón, rampa) para excitar el sistema.33 Su enfoque se alinea perfectamente con la metodología Evo-PINN, validando el uso de optimización global para parametrizar modelos dinámicos de motores y levitadores.

### **6.5 Agentes Colaborativos: AgenticSciML**

Finalmente, el trabajo reciente de George Karniadakis introduce el concepto de **AgenticSciML**.40 Este marco utiliza múltiples agentes de IA especializados (analista de datos, crítico, ingeniero, optimizador) que colaboran para diseñar y refinar modelos PINN. En el contexto de la levitación magnética, un sistema de este tipo podría proponer automáticamente el modelo de inductancia de Santana tras analizar los residuos de un modelo lineal, o sugerir el uso de un optimizador específico (como CMA-ES) al detectar un paisaje de pérdida mal condicionado. Esto representa el futuro de la automatización en el descubrimiento científico.

## **7\. Conclusiones y Perspectivas Futuras**

La integración de la inteligencia artificial con la física rigurosa está redefiniendo las capacidades de la ingeniería mecatrónica. Este reporte ha demostrado que:

1. **Superioridad de las Metaheurísticas Aceleradas:** La utilización de algoritmos como Evolución Diferencial y CMA-ES, implementados sobre JAX para ejecución masiva en GPU, no solo es viable sino superior a los métodos de gradiente para la identificación de sistemas no lineales complejos como la levitación magnética. La capacidad de evaluar millones de configuraciones físicas en segundos permite una exploración global que garantiza la robustez del modelo.  
2. **La Física como Regularizador:** La incorporación de modelos físicos avanzados (como el modelo de inductancia de Santana) y restricciones de conservación (Juez de Energía) permite identificar parámetros críticos como la masa en vuelo, algo imposible para modelos puramente basados en datos.  
3. **Evo-PINN y PIKAN como Estándar:** La combinación de evolución y redes neuronales (Evo-PINN), junto con arquitecturas interpretables (PIKANs), constituye el estado del arte para el modelado de caja gris. Estas herramientas permiten obtener modelos que son a la vez precisos, explicables y computacionalmente eficientes.  
4. **Convergencia de Disciplinas:** Estamos presenciando la disolución de las barreras entre el control automático, la física computacional y la inteligencia artificial. Herramientas como AgenticSciML sugieren un futuro donde sistemas autónomos no solo controlan procesos físicos, sino que descubren y refinan sus propios modelos matemáticos de la realidad.

La recomendación para la práctica ingenieril es clara: abandonar los enfoques de caja negra pura en favor de arquitecturas híbridas informadas por la física, y adoptar frameworks de computación acelerada como JAX para desbloquear el potencial de la optimización global en el diseño y control de sistemas.

---

**Nota sobre el contenido:** Este informe sintetiza información de múltiples fragmentos de investigación, identificados en el texto como \`\`. Se ha prestado especial atención a cubrir los requerimientos sobre levitación magnética, modelos de inductancia específicos, contribuciones de autores clave (Karniadakis, Rodríguez-Abreo, Santana) y la implementación técnica en JAX/GPU.

#### **Fuentes citadas**

1. Identification of Dynamic Parameters in a DC Motor Using Step and Ramp Torque Response Methods \- MDPI, acceso: enero 8, 2026, [https://www.mdpi.com/1424-8220/26/1/78](https://www.mdpi.com/1424-8220/26/1/78)  
2. Summaries Volume Process Control \- Sigurd Skogestad, acceso: enero 8, 2026, [https://skoge.folk.ntnu.no/prost/proceedings/process-control-slovakia-2017/data/summary.pdf](https://skoge.folk.ntnu.no/prost/proceedings/process-control-slovakia-2017/data/summary.pdf)  
3. Evolutionary Optimization of Physics-Informed Neural Networks: Evo-PINN Frontiers and Opportunities \- arXiv, acceso: enero 8, 2026, [https://arxiv.org/html/2501.06572v5](https://arxiv.org/html/2501.06572v5)  
4. (PDF) A comprehensive review of theoretical concepts and advancements in physics-informed neural networks with applications in structural engineering \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/publication/398267014\_A\_comprehensive\_review\_of\_theoretical\_concepts\_and\_advancements\_in\_physics-informed\_neural\_networks\_with\_applications\_in\_structural\_engineering](https://www.researchgate.net/publication/398267014_A_comprehensive_review_of_theoretical_concepts_and_advancements_in_physics-informed_neural_networks_with_applications_in_structural_engineering)  
5. JAX-Accelerated Neuroevolution of Physics-informed Neural Networks: Benchmarks and Experimental Results \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/publication/366321224\_JAX-Accelerated\_Neuroevolution\_of\_Physics-informed\_Neural\_Networks\_Benchmarks\_and\_Experimental\_Results](https://www.researchgate.net/publication/366321224_JAX-Accelerated_Neuroevolution_of_Physics-informed_Neural_Networks_Benchmarks_and_Experimental_Results)  
6. Evolutionary Optimization of Physics-Informed Neural Networks: Survey and Prospects, acceso: enero 8, 2026, [https://arxiv.org/html/2501.06572v2](https://arxiv.org/html/2501.06572v2)  
7. KAN\_PINN\_JAX\_GPU\_Optimization.ipynb \- Colab.pdf  
8. \[PDF\] EvoJAX: hardware-accelerated neuroevolution | Semantic Scholar, acceso: enero 8, 2026, [https://www.semanticscholar.org/paper/b5955c31e6c748ce301a2902044c4f364f9d0c84](https://www.semanticscholar.org/paper/b5955c31e6c748ce301a2902044c4f364f9d0c84)  
9. Learning the dynamical response of nonlinear non-autonomous dynamical systems with deep operator neural networks | Request PDF \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/publication/374353767\_Learning\_the\_dynamical\_response\_of\_nonlinear\_non-autonomous\_dynamical\_systems\_with\_deep\_operator\_neural\_networks](https://www.researchgate.net/publication/374353767_Learning_the_dynamical_response_of_nonlinear_non-autonomous_dynamical_systems_with_deep_operator_neural_networks)  
10. Evolutionary Optimization of Physics-Informed Neural Networks: Evo-PINN Frontiers and Opportunities \- arXiv, acceso: enero 8, 2026, [https://arxiv.org/html/2501.06572v4](https://arxiv.org/html/2501.06572v4)  
11. nicholassung97/Neuroevolution-of-PINNs \- GitHub, acceso: enero 8, 2026, [https://github.com/nicholassung97/Neuroevolution-of-PINNs](https://github.com/nicholassung97/Neuroevolution-of-PINNs)  
12. \[PDF\] Evolutionary Optimization of Physics-Informed Neural Networks: Evo-PINN Frontiers and Opportunities | Semantic Scholar, acceso: enero 8, 2026, [https://www.semanticscholar.org/paper/Evolutionary-Optimization-of-Physics-Informed-and-Wong-Gupta/b85d831a648a328a087a9af6547193c5d9ed8815](https://www.semanticscholar.org/paper/Evolutionary-Optimization-of-Physics-Informed-and-Wong-Gupta/b85d831a648a328a087a9af6547193c5d9ed8815)  
13. Evolutionary Optimization of Physics-Informed Neural Networks: Evo-PINN Frontiers and Opportunities \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/publication/396248878\_Evolutionary\_Optimization\_of\_Physics-Informed\_Neural\_Networks\_Evo-PINN\_Frontiers\_and\_Opportunities](https://www.researchgate.net/publication/396248878_Evolutionary_Optimization_of_Physics-Informed_Neural_Networks_Evo-PINN_Frontiers_and_Opportunities)  
14. Robust physics-informed neural network and swarm-based modeling of subcooled flow boiling pressure drop in metal foam tubes | International Journal of Intelligent Computing and Cybernetics | Emerald Publishing, acceso: enero 8, 2026, [https://www.emerald.com/ijicc/article/doi/10.1108/IJICC-04-2025-0244/1298976/Robust-physics-informed-neural-network-and-swarm](https://www.emerald.com/ijicc/article/doi/10.1108/IJICC-04-2025-0244/1298976/Robust-physics-informed-neural-network-and-swarm)  
15. A Review of Recent Developments in Permanent Magnet Eddy Current Couplers Technology \- MDPI, acceso: enero 8, 2026, [https://www.mdpi.com/2076-0825/12/7/277](https://www.mdpi.com/2076-0825/12/7/277)  
16. Parameter identification of permanent magnet synchronous motor based on improved Newton-Raphson-based optimizer | COMPEL | Emerald Publishing, acceso: enero 8, 2026, [https://www.emerald.com/compel/article/doi/10.1108/COMPEL-05-2025-0217/1299175/Parameter-identification-of-permanent-magnet](https://www.emerald.com/compel/article/doi/10.1108/COMPEL-05-2025-0217/1299175/Parameter-identification-of-permanent-magnet)  
17. Physics-Informed Neural Network for Load Margin Assessment of Power Systems with Optimal Phasor Measurement Unit Placement \- MDPI, acceso: enero 8, 2026, [https://www.mdpi.com/2673-4826/5/4/39](https://www.mdpi.com/2673-4826/5/4/39)  
18. Inverse Methods for Design and Simulation with Particle Systems Erik Steven Strand \- DSpace@MIT, acceso: enero 8, 2026, [https://dspace.mit.edu/bitstream/handle/1721.1/129322/1227278365-MIT.pdf?sequence=1\&isAllowed=y](https://dspace.mit.edu/bitstream/handle/1721.1/129322/1227278365-MIT.pdf?sequence=1&isAllowed=y)  
19. Reduction of stiffness and mass matrices | AIAA Journal, acceso: enero 8, 2026, [https://arc.aiaa.org/doi/10.2514/3.2874?mobileUi=0](https://arc.aiaa.org/doi/10.2514/3.2874?mobileUi=0)  
20. Block-diagram of the Maglev system \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/figure/Block-diagram-of-the-Maglev-system\_fig1\_346549856](https://www.researchgate.net/figure/Block-diagram-of-the-Maglev-system_fig1_346549856)  
21. Evolutionary Optimization of Physics-Informed Neural Networks: Advancing Generalizability by the Baldwin Effect \- arXiv, acceso: enero 8, 2026, [https://arxiv.org/html/2312.03243v4](https://arxiv.org/html/2312.03243v4)  
22. JAX-MPM: A Learning-Augmented Differentiable Meshfree Framework for GPU-Accelerated Lagrangian Simulation and Geophysical Inverse Modeling \- arXiv, acceso: enero 8, 2026, [https://arxiv.org/html/2507.04192v1](https://arxiv.org/html/2507.04192v1)  
23. Physics-Informed Kolmogorov-Arnold Networks for Power System Dynamics \- IEEE Xplore, acceso: enero 8, 2026, [https://ieeexplore.ieee.org/iel8/8784343/10845831/10843279.pdf](https://ieeexplore.ieee.org/iel8/8784343/10845831/10843279.pdf)  
24. (PDF) Physics-Informed Kolmogorov-Arnold Networks for Power System Dynamics, acceso: enero 8, 2026, [https://www.researchgate.net/publication/388038072\_Physics-Informed\_Kolmogorov-Arnold\_Networks\_for\_Power\_System\_Dynamics](https://www.researchgate.net/publication/388038072_Physics-Informed_Kolmogorov-Arnold_Networks_for_Power_System_Dynamics)  
25. Physics-Informed Kolmogorov-Arnold Networks for Power System Dynamics \- IEEE Xplore, acceso: enero 8, 2026, [https://ieeexplore.ieee.org/document/10843279](https://ieeexplore.ieee.org/document/10843279)  
26. Kolmogorov–Arnold Networks for System Identification of First- and Second-Order Dynamic Systems \- MDPI, acceso: enero 8, 2026, [https://www.mdpi.com/2673-4591/100/1/59](https://www.mdpi.com/2673-4591/100/1/59)  
27. Research Directions on Kolmogorov–Arnold Networks: A Comprehensive Review \- MDPI, acceso: enero 8, 2026, [https://www.mdpi.com/2073-8994/18/1/60](https://www.mdpi.com/2073-8994/18/1/60)  
28. State-Space Kolmogorov Arnold Networks for Interpretable Nonlinear System Identification \- IEEE Xplore, acceso: enero 8, 2026, [https://ieeexplore.ieee.org/iel8/7782633/10939047/11029059.pdf](https://ieeexplore.ieee.org/iel8/7782633/10939047/11029059.pdf)  
29. Physics-Informed Kolmogorov-Arnold Networks for Power System Dynamics \- arXiv, acceso: enero 8, 2026, [https://arxiv.org/html/2408.06650v1](https://arxiv.org/html/2408.06650v1)  
30. A Genetic Algorithm-Based Approach for Automated Optimization of Kolmogorov-Arnold Networks in Classification Tasks \- arXiv, acceso: enero 8, 2026, [https://arxiv.org/html/2501.17411v1](https://arxiv.org/html/2501.17411v1)  
31. Representation Meets Optimization: Training PINNs and PIKANs for Gray-Box Discovery in Systems Pharmacology \- PubMed, acceso: enero 8, 2026, [https://pubmed.ncbi.nlm.nih.gov/40297233/](https://pubmed.ncbi.nlm.nih.gov/40297233/)  
32. Control lineal y no lineal de un levitador magnetico \- UPCommons, acceso: enero 8, 2026, [https://upcommons.upc.edu/bitstreams/d765739e-633d-46f7-b42f-c73d8170ae5b/download](https://upcommons.upc.edu/bitstreams/d765739e-633d-46f7-b42f-c73d8170ae5b/download)  
33. Identification of Dynamic Parameters in a DC Motor Using Step and Ramp Torque Response Methods | Request PDF \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/publication/399051498\_Identification\_of\_Dynamic\_Parameters\_in\_a\_DC\_Motor\_Using\_Step\_and\_Ramp\_Torque\_Response\_Methods](https://www.researchgate.net/publication/399051498_Identification_of_Dynamic_Parameters_in_a_DC_Motor_Using_Step_and_Ramp_Torque_Response_Methods)  
34. Annual Report 2023 \- LNCMI \- CNRS, acceso: enero 8, 2026, [https://lncmi.cnrs.fr/wp-content/uploads/2025/10/AR2023\_vWeb.pdf](https://lncmi.cnrs.fr/wp-content/uploads/2025/10/AR2023_vWeb.pdf)  
35. Damping techniques for grid-connected voltage source converters based on LCL filter: An overview | Request PDF \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/publication/318579455\_Damping\_techniques\_for\_grid-connected\_voltage\_source\_converters\_based\_on\_LCL\_filter\_An\_overview](https://www.researchgate.net/publication/318579455_Damping_techniques_for_grid-connected_voltage_source_converters_based_on_LCL_filter_An_overview)  
36. Dr. Omar Rodriguez-Abreo | Author \- SciProfiles, acceso: enero 8, 2026, [https://sciprofiles.com/profile/ORA](https://sciprofiles.com/profile/ORA)  
37. A comparative review on mobile robot path planning: Classical or meta-heuristic methods? | Request PDF \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/publication/346270168\_A\_comparative\_review\_on\_mobile\_robot\_path\_planning\_Classical\_or\_meta-heuristic\_methods](https://www.researchgate.net/publication/346270168_A_comparative_review_on_mobile_robot_path_planning_Classical_or_meta-heuristic_methods)  
38. Study of FLC, PID, and LQR Control Methods for Precise Drone Landing in Wind Conditions, acceso: enero 8, 2026, [https://www.researchgate.net/publication/397836989\_Study\_of\_FLC\_PID\_and\_LQR\_Control\_Methods\_for\_Precise\_Drone\_Landing\_in\_Wind\_Conditions](https://www.researchgate.net/publication/397836989_Study_of_FLC_PID_and_LQR_Control_Methods_for_Precise_Drone_Landing_in_Wind_Conditions)  
39. A review of parameter estimators and controllers for induction motors based on artificial neural networks | Request PDF \- ResearchGate, acceso: enero 8, 2026, [https://www.researchgate.net/publication/261844443\_A\_review\_of\_parameter\_estimators\_and\_controllers\_for\_induction\_motors\_based\_on\_artificial\_neural\_networks](https://www.researchgate.net/publication/261844443_A_review_of_parameter_estimators_and_controllers_for_induction_motors_based_on_artificial_neural_networks)  
40. \[2511.07262\] AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning \- arXiv, acceso: enero 8, 2026, [https://arxiv.org/abs/2511.07262](https://arxiv.org/abs/2511.07262)  
41. AgenticSciML: Collaborative Multi-Agent Systems for Emergent Discovery in Scientific Machine Learning \- arXiv, acceso: enero 8, 2026, [https://arxiv.org/html/2511.07262v1](https://arxiv.org/html/2511.07262v1)
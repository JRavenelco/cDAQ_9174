# **Informe de Investigación Exhaustiva: Modelado de Gemelo Digital mediante Redes KAN-PINN Lagrangianas**

Proyecto: KAN-PINN Levitador Magnético  
Referencia Principal: Bitácora de Ingeniería \- José de Jesús Santana Ramírez (CEROC \- UAQ)  
Fecha del Informe: 2 de Diciembre, 2025  
Tema: Análisis Profundo de Arquitecturas Neuronales Informadas por Física y Estrategias de Estabilización Numérica

## ---

**1\. Introducción y Contexto Estratégico**

El desarrollo de Gemelos Digitales (Digital Twins) de alta fidelidad para sistemas electromecánicos representa una de las fronteras más desafiantes en la ingeniería de control moderna. El Proyecto KAN-PINN, liderado por el Centro de Investigación en Optimización y Control (CEROC) de la Universidad Autónoma de Querétaro (UAQ), se sitúa en la vanguardia de este esfuerzo al intentar construir un modelo matemático riguroso, transparente y físicamente consistente de un levitador magnético. Este informe analiza exhaustivamente los avances, obstáculos y soluciones implementadas en el proyecto, contextualizándolos dentro del estado del arte de la Inteligencia Artificial Científica (SciML).

### **1.1 La Evolución hacia la "Caja de Cristal"**

Históricamente, el modelado de sistemas dinámicos complejos como la levitación magnética ha oscilado entre dos paradigmas opuestos. Por un lado, los modelos analíticos basados en primeros principios (Leyes de Newton y Kirchhoff) ofrecen interpretabilidad pero sufren de simplificaciones excesivas, fallando a menudo en capturar no linealidades sutiles como la histéresis magnética, las corrientes de Foucault o la fricción dinámica compleja.1 Por otro lado, los modelos puramente basados en datos o "Cajas Negras" (como los Perceptrones Multicapa o MLPs tradicionales) poseen una capacidad de aproximación universal, pero carecen de garantías físicas, pudiendo predecir estados imposibles como la creación espontánea de energía o resistencias negativas.3

El objetivo del proyecto KAN-PINN no es simplemente ajustar una curva a los datos experimentales, sino descubrir la estructura matemática subyacente del sistema. La adopción de Redes de Kolmogorov-Arnold (KAN) en lugar de MLPs convencionales responde a la necesidad de crear una "Caja de Cristal" (Glass Box). Al utilizar funciones de activación aprendibles en las conexiones (aristas) de la red, basadas en B-splines, se habilita la capacidad de Regresión Simbólica.5 Esto permite, en teoría, que el modelo no solo prediga la posición del levitador, sino que revele explícitamente la fórmula matemática de la inductancia no lineal $L(y)$, transformando la red neuronal en una herramienta de descubrimiento científico.7

### **1.2 El Desafío de la Realidad Física y Computacional**

La bitácora de ingeniería del 2 de diciembre de 2025 destaca una tensión crítica entre la eficiencia computacional y la fidelidad física. El paso de un enfoque newtoniano directo a uno lagrangiano busca imponer la conservación de la energía como un sesgo inductivo fuerte.9 Sin embargo, este rigor matemático ha expuesto fragilidades en el proceso de entrenamiento, específicamente la tendencia del modelo a converger hacia "soluciones triviales" (masa nula o negativa) y la amplificación catastrófica del ruido de los sensores al calcular derivadas temporales.

Este documento desglosa cada componente del sistema, desde la fundamentación teórica lagrangiana hasta las técnicas de procesamiento de señales necesarias para mitigar el ruido "rosa" inherente a la instrumentación real 11, proporcionando una hoja de ruta detallada para superar el estancamiento actual en la convergencia de las ecuaciones diferenciales.

## ---

**2\. Fundamentación Teórica: El Paradigma Lagrangiano en Sistemas Electromecánicos**

La decisión de migrar de un modelado basado en fuerzas (Newtoniano) a uno basado en energías (Lagrangiano) es fundamental para garantizar la robustez del Gemelo Digital a largo plazo.

### **2.1 Definición y Conservación de la Energía**

En la mecánica clásica, el Lagrangiano $\\mathcal{L}$ se define como la diferencia entre la energía cinética ($T$) y la energía potencial ($V$). Para un sistema electromecánico acoplado, como un levitador magnético, la energía almacenada en el campo magnético del inductor actúa como un análogo de la energía cinética en el dominio eléctrico.

El Lagrangiano del sistema $\\mathcal{L}(y, \\dot{y}, q, \\dot{q})$ se formula como:

$$\\mathcal{L} \= T\_{mec} \+ T\_{mag} \- V\_{pot}$$

$$\\mathcal{L} \= \\frac{1}{2}m\\dot{y}^2 \+ \\frac{1}{2}L(y)\\dot{q}^2 \- mgy$$  
Donde:

* $y$: Posición vertical del objeto levitado (coordenada mecánica).  
* $\\dot{y} \= v$: Velocidad mecánica.  
* $q$: Carga eléctrica, tal que $\\dot{q} \= i$ (corriente).  
* $L(y)$: Inductancia de la bobina, la cual es una función altamente no lineal de la posición $y$.  
* $m$: Masa del objeto levitado.  
* $g$: Aceleración gravitatoria.

La ventaja crítica de este enfoque es que garantiza que las fuerzas mecánicas y los voltajes inducidos deriven de un único objeto matemático escalar ($\\mathcal{L}$). En un enfoque de caja negra no restringido, una red neuronal podría aprender una función de fuerza magnética y una función de contra-electromotriz que no sean consistentes entre sí, violando la ley de conservación de la energía.10 El enfoque lagrangiano fuerza esta coherencia interna: la fuerza magnética es la derivada parcial de la energía magnética respecto a la posición, y el voltaje inducido es la derivada respecto al tiempo del flujo magnético, ambos gobernados por la misma función $L(y)$.

### **2.2 Derivación de las Ecuaciones Gobernantes (Euler-Lagrange)**

El comportamiento dinámico del sistema se obtiene aplicando las ecuaciones de Euler-Lagrange para coordenadas generalizadas disipativas:

$$ \\frac{d}{dt}\\left(\\frac{\\partial \\mathcal{L}}{\\partial \\dot{x}}\\right) \- \\frac{\\partial \\mathcal{L}}{\\partial x} \= Q\_x $$

Donde $Q\_x$ representa las fuerzas generalizadas no conservativas (disipación y fuentes externas).

#### **2.2.1 Dinámica Eléctrica y el Descubrimiento de la Back-EMF**

Para la coordenada eléctrica $q$ (carga), la ecuación revela la interacción entre el movimiento y la electricidad:

1. **Momento Generalizado Eléctrico (Flujo):** $\\frac{\\partial \\mathcal{L}}{\\partial i} \= L(y)i$.  
2. Derivada Temporal del Momento:

   $$\\frac{d}{dt}(L(y)i) \= L(y)\\frac{di}{dt} \+ i\\frac{d}{dt}(L(y))$$

   Aplicando la regla de la cadena, $\\frac{d}{dt}(L(y)) \= \\frac{\\partial L}{\\partial y}\\dot{y}$. Por lo tanto:

   $$\\frac{d}{dt}(L(y)i) \= L(y)\\frac{di}{dt} \+ i \\frac{\\partial L}{\\partial y}\\dot{y}$$  
3. **Derivada Parcial respecto a la Coordenada:** $\\frac{\\partial \\mathcal{L}}{\\partial q} \= 0$ (el Lagrangiano no depende explícitamente de la carga acumulada, solo de la corriente).  
4. **Fuerzas No Conservativas:** $Q\_q \= V\_{in} \- Ri$ (Voltaje de entrada menos caída resistiva).

Reuniendo los términos, obtenemos la ecuación de voltaje de Kirchhoff extendida:

$$L(y)\\frac{di}{dt} \+ i \\frac{\\partial L}{\\partial y}\\dot{y} \+ Ri \= V\_{in}$$  
**Análisis de Insight:** Como señala la bitácora del autor, el término central $i \\frac{\\partial L}{\\partial y}\\dot{y}$ es el descubrimiento clave. Representa la **Fuerza Contra-Electromotriz de Movimiento (Motion Induced Voltage)**. Este término solo existe cuando el objeto se mueve ($\\dot{y} \\neq 0$) y hay corriente ($i \\neq 0$). Su magnitud depende de la pendiente del cambio de inductancia $\\frac{\\partial L}{\\partial y}$. Si la red KAN logra aprender este término correctamente, habrá "entendido" el mecanismo fundamental de transducción de energía del sistema, validando su uso como Gemelo Digital explicable.13

#### **2.2.2 Dinámica Mecánica y Fuerza Magnética**

Para la coordenada mecánica $y$:

1. **Momento Mecánico:** $\\frac{\\partial \\mathcal{L}}{\\partial \\dot{y}} \= m\\dot{y}$.  
2. **Derivada Temporal:** $\\frac{d}{dt}(m\\dot{y}) \= m\\ddot{y}$.  
3. Fuerza Generalizada Derivada del Potencial:  
   $$ \\frac{\\partial \\mathcal{L}}{\\partial y} \= \\frac{\\partial}{\\partial y}\\left( \\frac{1}{2}L(y)i^2 \- mgy \\right) \= \\frac{1}{2}i^2 \\frac{\\partial L}{\\partial y} \- mg $$  
4. **Fuerzas No Conservativas:** $Q\_y \= \-F\_{friccion}(\\dot{y})$.

La ecuación de movimiento resultante es:

$$m\\ddot{y} \- \\left( \\frac{1}{2}i^2 \\frac{\\partial L}{\\partial y} \- mg \\right) \= \-F\_{friccion}$$

$$m\\ddot{y} \+ mg \= \\frac{1}{2}i^2 \\frac{\\partial L}{\\partial y} \- F\_{friccion}$$  
Aquí se observa la simetría cruzada: el mismo término $\\frac{\\partial L}{\\partial y}$ que genera voltaje en la ecuación eléctrica es el responsable de generar fuerza de sustentación en la ecuación mecánica. Esta restricción estructural es lo que impide que el modelo "alucine" físicas inconsistentes, un problema común en las redes neuronales estándar.9

## ---

**3\. Superando el "Muro Computacional": Eficiencia en el Entrenamiento**

Uno de los primeros obstáculos documentados en la bitácora fue la ineficiencia extrema del entrenamiento inicial, con tiempos estimados superiores a las 30 horas. Este es un cuello de botella común en las PINNs, que requieren evaluar derivadas de alto orden en miles de puntos de colocación en cada iteración.14

### **3.1 El Problema de los Bucles Secuenciales**

En la implementación inicial, el cálculo de las pérdidas (residuos de las ecuaciones diferenciales) se realizaba probablemente mediante bucles estándar de Python o iteraciones simples sobre los puntos de datos. Dado que Python es un lenguaje interpretado, la sobrecarga de despachar operaciones individuales a la CPU es inmensa cuando se repite millones de veces. Además, el mecanismo de Diferenciación Automática (Autograd) construye grafos computacionales dinámicos que pueden volverse ineficientes si no se gestionan en lotes.

### **3.2 La Solución: Vectorización y Aceleración CUDA**

La solución implementada por el Ing. Santana Ramírez fue la **Vectorización**. Esto implica reestructurar los tensores de datos para que las dimensiones de lote (batch size) sean procesadas simultáneamente. En lugar de calcular el Lagrangiano para un estado $(y\_t, v\_t, i\_t)$ a la vez, se calcula para una matriz completa de estados $N \\times 3$ en una sola operación de kernel.

Esta reestructuración permite aprovechar la arquitectura SIMT (Single Instruction, Multiple Threads) de las GPUs modernas a través de CUDA. Las GPUs están diseñadas para ejecutar la misma instrucción matemática en miles de datos en paralelo.

* **Impacto Cuantitativo:** Reducción de \>30 horas a minutos.  
* **Impacto Cualitativo:** Habilita la búsqueda de hiperparámetros. En SciML, es imposible ajustar parámetros delicados como las tasas de aprendizaje o los pesos de regularización si cada ciclo de prueba toma un día. La velocidad se traduce directamente en capacidad de experimentación y calidad del modelo final.15

## ---

**4\. Arquitectura KAN: La "Caja de Cristal" y el Dilema Bias-Variance**

La elección de una Red de Kolmogorov-Arnold (KAN) sobre un Perceptrón Multicapa (MLP) es una decisión estratégica orientada a la interpretabilidad y la precisión en funciones de baja dimensionalidad pero alta complejidad.

### **4.1 Innovación Arquitectónica: Splines en las Aristas**

A diferencia de los MLPs, donde las no linealidades (ReLU, Tanh, Sigmoide) son fijas y están en las neuronas, las KANs colocan funciones de activación aprendibles en las conexiones (aristas) entre nodos. Según el teorema de representación de Kolmogorov-Arnold, cualquier función multivariada continua puede representarse como una superposición de funciones univariadas continuas.16

En la implementación LagrangianKAN, estas funciones univariadas se modelan mediante **B-splines** (Basis Splines). Un B-spline se define por un conjunto de puntos de control (knots) y un orden polinomial.

* **Ventaja Local:** Modificar un coeficiente de un B-spline solo afecta la función en un intervalo local, a diferencia de los pesos de un MLP que tienen un efecto global. Esto permite a la KAN aprender picos o discontinuidades locales (como la saturación magnética abrupta) sin perturbar la aproximación en otras regiones del espacio de estados.18

### **4.2 Ajuste de Hiperparámetros y el Bias-Variance Tradeoff**

La bitácora documenta un proceso empírico clásico de selección de modelos:

* **10 Knots (Underfitting):** Con pocos puntos de control, los splines son demasiado rígidos. No pueden doblarse lo suficiente para capturar la curva $1/y$ característica de la inductancia cerca del núcleo magnético. El error de sesgo (bias) es alto.  
* **50 Knots (Overfitting):** Con demasiados puntos, el modelo tiene demasiada libertad. Los splines comienzan a ondularse para pasar exactamente por cada punto de ruido de los sensores. El error de varianza es alto; el modelo "memoriza" el ruido en lugar de aprender la física.  
* **30 Knots (Punto Óptimo):** Este equilibrio permite capturar la curvatura suave de la inductancia física mientras se ignora el ruido de alta frecuencia, actuando efectivamente como un filtro de paso bajo estructural.19

### **4.3 Potencial de Regresión Simbólica**

Una característica única de las KANs, implementada en librerías como pykan, es la capacidad de extraer fórmulas explícitas. Una vez entrenada la red, se puede interrogar a la conexión que mapea la posición $y$ a la energía magnética. Usando técnicas de regresión simbólica, la red puede sugerir que la función aprendida se aproxima a $L(y) \\approx \\frac{k}{y \+ a} \+ b$.7  
Esto cumple el objetivo del "Gemelo Digital Explicable": no solo se tiene un modelo que funciona, sino una ecuación que el ingeniero puede validar contra la teoría electromagnética y usar en diseños futuros.

## ---

**5\. Diagnóstico de Fallas: El Conflicto Físico y las Soluciones Triviales**

A pesar de la robustez teórica, el entrenamiento de PINNs y LNNs (Lagrangian Neural Networks) es notoriamente inestable. La bitácora identifica un problema crítico: el modelo "hace trampa".

### **5.1 La Patología de la Solución Trivial**

Las redes neuronales son optimizadores perezosos; buscarán cualquier camino para minimizar la función de pérdida.

* **Masa Negativa ($m \= \-0.17$ kg):** En un sistema físico real, la masa es estrictamente positiva. Sin embargo, en el paisaje de optimización matemático, una masa negativa podría compensar un error de signo en otra parte de la ecuación (por ejemplo, si los sensores de corriente están invertidos o si el ruido sugiere una aceleración opuesta a la gravedad). Sin restricciones, la red explora regiones no físicas.21  
* **Inductancia Cero ($L \\approx 0$):** Si el modelo fija $L=0$ y $m=0$, la ecuación mecánica $m\\ddot{y} \+ mg \- F\_{mag} \= 0$ se reduce a $0 \+ 0 \- 0 \= 0$ (asumiendo que $g$ también se escala o ignora). El residuo de la ecuación diferencial es cero, lo cual es "perfecto" para el optimizador, pero inútil para el ingeniero. Esto se conoce como el colapso a la solución trivial o nula.23

### **5.2 Restricciones Rígidas (Hard Constraints)**

Para combatir esto, las "soluciones" sugeridas en la bitácora (Bounds) deben implementarse rigurosamente.

* **Parametrización Positiva:** En lugar de aprender $m$ directamente, se debe aprender un parámetro $p$ tal que $m \= \\text{Softplus}(p) \+ \\epsilon$ o $m \= e^p$. Esto garantiza matemáticamente que $m \> 0$ siempre, eliminando la posibilidad de masa negativa.21  
* **Límites de Inductancia:** Similarmente, la función de inductancia debe ser positiva definida. Esto es más difícil de imponer en una red neuronal general, pero en una KAN se puede restringir el rango de los coeficientes de los splines.

## ---

**6\. La Raíz del Estancamiento: Ruido y Diferenciación Numérica**

El hallazgo más crítico de la investigación es el impacto del ruido en las derivadas temporales. La bitácora señala: "El error de la ecuación eléctrica se estancó en \~29-30".

### **6.1 El Problema de "Derivar Ruido" (Garbage In, Garbage Out)**

Para entrenar la ecuación eléctrica $V \= L\\frac{di}{dt} \+ \\dots$, necesitamos conocer el valor verdadero de $\\frac{di}{dt}$.  
Los datos experimentales provienen de sensores discretos que inevitablemente contienen ruido.

* **Naturaleza del Ruido:** Los sensores de efecto Hall y las derivaciones de corriente en sistemas de levitación a menudo exhiben **ruido rosa** ($1/f$), que es más complejo que el ruido blanco gaussiano.11 Además, la conmutación PWM de los amplificadores de potencia introduce ruido de alta frecuencia.  
* **Amplificación por Diferenciación:** Si usamos diferencias finitas simples ($\\frac{i\_{t+1} \- i\_t}{\\Delta t}$) para estimar la derivada, el ruido se amplifica por un factor de $1/\\Delta t$. Cuanto más rápido muestreamos (menor $\\Delta t$), más domina el ruido sobre la señal física.  
* **Consecuencia en la PINN:** La red neuronal intenta aprender una función $L(y)$ que relacione el voltaje $V$ con una derivada $\\frac{di}{dt}$ que es básicamente aleatoria. No existe correlación matemática posible, por lo que el error se estanca y la red no puede converger.27

### **6.2 Solución Avanzada: Filtrado Savitzky-Golay**

La bitácora propone correctamente el uso de filtros Savitzky-Golay (SG).

* **Mecanismo:** A diferencia de un promedio móvil que simplemente aplana la señal (y destruye los picos de alta dinámica necesarios para aprender la Back-EMF), el filtro SG ajusta un polinomio local de grado $k$ a una ventana de puntos y calcula la derivada analítica de ese polinomio.29  
* **Ventaja Estratégica:** Los filtros SG son excelentes para preservar los momentos de orden superior de la señal (picos, cambios bruscos de pendiente) mientras eliminan el ruido estocástico. Esto es vital para un sistema de levitación donde las reacciones rápidas son la esencia de la estabilidad.31  
* **Comparativa:** Estudios demuestran que para datos de PINNs ruidosos, los filtros SG y los métodos de Variación Total (Total Variation) producen resultados cualitativamente superiores a la diferenciación directa o filtros gaussianos simples, permitiendo que la física subyacente emerja del ruido.27

## ---

**7\. Estrategias de Entrenamiento: Curriculum Learning y Balanceo de Pérdidas**

Para guiar al modelo fuera de las soluciones triviales y a través del paisaje de optimización complejo, se requiere una estrategia de "enseñanza" estructurada.

### **7.1 Curriculum Learning (Aprendizaje por Etapas)**

El concepto de Curriculum Learning se basa en enseñar a la red tareas fáciles antes que las difíciles.32

* **Fase 1: Ajuste de Datos (Interpolación):** Primero, se entrena la red solo para reproducir las trayectorias observadas de $y(t)$ e $i(t)$, con la pérdida de física desactivada. Esto ancla la red en la realidad observable y evita que "vuele" hacia estados imposibles.  
* **Fase 2: Física Eléctrica:** Se activa la pérdida de la ecuación de Kirchhoff. Como sugiere la bitácora, esto debe priorizarse. Si la red no entiende la inductancia (la relación V-I), la ecuación mecánica (que depende de $i^2$ y $L$) no tiene esperanza de ser correcta.  
* **Fase 3: Física Mecánica (Dinámica Completa):** Finalmente, se activa la ecuación de movimiento de Newton/Lagrange para ajustar la masa y la fricción.

### **7.2 Normalización y Balanceo de Pérdidas (Loss Balancing)**

La bitácora menciona la "Normalización de Pérdida". Esto es crucial porque las magnitudes físicas son dispares.

* Voltaje: \~12 V.  
* Fuerza: \~1-10 N.  
* Posición: \~0.01 m.  
* Inductancia: \~0.1 H.  
  Los gradientes derivados de estas magnitudes pueden variar en órdenes de magnitud. Si el gradiente de la pérdida de datos es 1000 veces mayor que el de la pérdida física, la red ignorará la física.  
* **Técnicas:** Algoritmos como **GradNorm**, **SoftAdapt** o **ReLoBRaLo** ajustan dinámicamente los pesos de cada término de la función de pérdida durante el entrenamiento para asegurar que todos contribuyan equitativamente al aprendizaje.34 El uso de GradNorm es particularmente recomendado para evitar que una tarea (ej. ajustar la posición) domine sobre otra (ej. satisfacer la ley de Kirchhoff).

## ---

**8\. Análisis Comparativo de Datos y Tablas Resumen**

A continuación, se presentan tablas que sintetizan las decisiones de diseño y los diagnósticos del proyecto, facilitando la visualización de las compensaciones técnicas.

### **Tabla 1: Comparativa de Arquitecturas de Modelado**

| Característica | Caja Negra (MLP Estándar) | Modelo Analítico (Física Pura) | Caja de Cristal (KAN Lagrangiana) |
| :---- | :---- | :---- | :---- |
| **Interpretabilidad** | Baja (Pesos opacos) | Alta (Ecuaciones explícitas) | **Alta (Regresión Simbólica)** |
| **Precisión** | Alta (Universal) | Media (Simplificaciones) | **Alta (Adaptable)** |
| **Consistencia Física** | No garantizada (Puede violar leyes) | Garantizada por diseño | **Forzada por Lagrangiano** |
| **Manejo de Ruido** | Propenso a Overfitting | Robusto (si los parámetros son correctos) | **Robusto (con Splines optimizados)** |
| **Descubrimiento** | Nulo | Nulo | **Posible (Forma de $L(y)$)** |

### **Tabla 2: Diagnóstico de Fallas y Soluciones PINN**

| Síntoma Observado | Causa Física/Matemática | Solución Técnica (Referencia) |
| :---- | :---- | :---- |
| **Entrenamiento \>30h** | Operaciones escalares en CPU | **Vectorización \+ CUDA** 15 |
| **Masa Negativa / $L=0$** | Minimización de pérdida vía solución trivial | **Restricciones Rígidas (Softplus) \+ Curriculum Learning** 21 |
| **Error Eléctrico Estancado** | Diferenciación de ruido rosa en sensores | **Filtros Savitzky-Golay** 29 |
| **Inestabilidad de Splines** | Trade-off Bias-Variance incorrecto | **Ajuste de Grid (30 Knots)** 19 |
| **Gradientes Desbalanceados** | Diferencias de escala en unidades físicas | **GradNorm / Normalización de Pérdida** 34 |

## ---

**9\. Hoja de Ruta y Recomendaciones Técnicas (Próximos Pasos)**

Basado en el análisis integral de la bitácora y la literatura científica, se propone el siguiente plan de acción inmediato para el equipo del CEROC-UAQ.

### **9.1 Fase Inmediata: Higiene de Datos (Pre-procesamiento)**

1. **Análisis Espectral:** Realizar una Transformada Rápida de Fourier (FFT) a los datos crudos de corriente y posición para identificar las frecuencias características del ruido (ej. frecuencia de PWM, ruido de red de 60Hz).  
2. **Sintonización de Savitzky-Golay:** Seleccionar la ventana del filtro SG basándose en el análisis espectral. La ventana debe ser lo suficientemente grande para atenuar el ruido identificado pero menor que la constante de tiempo eléctrica del sistema ($\\tau \= L/R$) para no filtrar la dinámica transitoria real.  
3. **Validación Visual:** Graficar $\\frac{di}{dt}$ filtrado superpuesto a los datos crudos para asegurar que la derivada no tiene retraso de fase significativo.

### **9.2 Fase de Entrenamiento: Restricciones y Currículo**

1. **Implementar "Hard Bounds":** Modificar la arquitectura de la red para que los parámetros físicos (masa, resistencia) sean la salida de una función softplus o exp, asegurando positividad estricta.  
2. **Inicialización Informada:** No iniciar los splines de inductancia aleatoriamente. Inicializarlos con una función decreciente simple ($1/y$) para dar al optimizador un punto de partida físicamente plausible ("Warm Start").37  
3. **Ejecución de Curriculum:** Entrenar primero solo con pérdida de datos (500 épocas), luego agregar la ecuación eléctrica (500 épocas) y finalmente la mecánica.

### **9.3 Fase de Validación: Descubrimiento Simbólico**

1. **Extracción de Fórmulas:** Una vez que el error converja, utilizar las herramientas de pykan para extraer la expresión simbólica de $L(y)$.  
2. **Verificación Cruzada:** Comparar la función extraída con medidas estáticas de inductancia (si están disponibles) o con modelos de elementos finitos (FEM) del electroimán. Si la forma funcional coincide (ej. hipérbola), se habrá validado el Gemelo Digital como una herramienta científica confiable.

## ---

**10\. Conclusión**

El proyecto KAN-PINN representa un avance significativo hacia la creación de Gemelos Digitales auto-adaptativos y explicables. Los desafíos encontrados (ruido, soluciones triviales) no son fallas del diseño, sino obstáculos característicos de la frontera del conocimiento en Aprendizaje Automático Científico. La transición a un enfoque Lagrangiano, combinada con la arquitectura KAN, ofrece una base teórica sólida para garantizar la conservación de la energía. Sin embargo, el éxito práctico depende ahora de la rigurosidad en el procesamiento de señales (filtrado SG) y en la ingeniería de la función de pérdida (restricciones y balanceo). Al superar el "conflicto físico" actual, el modelo resultante no solo controlará el levitador, sino que *comprenderá* su física interna, cumpliendo la promesa de la Inteligencia Artificial en la ingeniería moderna.

---

**Fin del Informe**

#### **Fuentes citadas**

1. \[2401.08667\] Data-Driven Physics-Informed Neural Networks: A Digital Twin Perspective, acceso: diciembre 2, 2025, [https://arxiv.org/abs/2401.08667](https://arxiv.org/abs/2401.08667)  
2. The Magnetic Levitation Weaving Needle Monitoring System and Predictive Analysis Based on Digital Twin \- MDPI, acceso: diciembre 2, 2025, [https://www.mdpi.com/2076-3417/14/14/6250](https://www.mdpi.com/2076-3417/14/14/6250)  
3. PHYSICS-INFORMED MACHINE LEARNING-DRIVEN STRUCTURAL DIGITAL TWIN FOR DAMAGE IDENTIFICATION THROUGH ANOMALY DETECTION \- Purdue University Graduate School \- Figshare, acceso: diciembre 2, 2025, [https://hammer.purdue.edu/articles/thesis/\_b\_PHYSICS-INFORMED\_MACHINE\_LEARNING-DRIVEN\_STRUCTURAL\_DIGITAL\_TWIN\_FOR\_DAMAGE\_IDENTIFICATION\_THROUGH\_ANOMALY\_DETECTION\_b\_/28826465](https://hammer.purdue.edu/articles/thesis/_b_PHYSICS-INFORMED_MACHINE_LEARNING-DRIVEN_STRUCTURAL_DIGITAL_TWIN_FOR_DAMAGE_IDENTIFICATION_THROUGH_ANOMALY_DETECTION_b_/28826465)  
4. Enforcing Analytic Constraints in Neural Networks Emulating Physical Systems \- Gentine Lab, acceso: diciembre 2, 2025, [https://gentinelab.eee.columbia.edu/sites/default/files/content/PhysRevLett.126.098302.pdf](https://gentinelab.eee.columbia.edu/sites/default/files/content/PhysRevLett.126.098302.pdf)  
5. (PDF) Data-driven model discovery with Kolmogorov-Arnold networks \- ResearchGate, acceso: diciembre 2, 2025, [https://www.researchgate.net/publication/390690615\_Data-driven\_model\_discovery\_with\_Kolmogorov-Arnold\_networks](https://www.researchgate.net/publication/390690615_Data-driven_model_discovery_with_Kolmogorov-Arnold_networks)  
6. KAN-SR: A Kolmogorov-Arnold Network Guided Symbolic Regression Framework \- arXiv, acceso: diciembre 2, 2025, [https://arxiv.org/html/2509.10089v1](https://arxiv.org/html/2509.10089v1)  
7. KAN 2.0: Kolmogorov-Arnold Networks Meet Science \- arXiv, acceso: diciembre 2, 2025, [https://arxiv.org/html/2408.10205v1](https://arxiv.org/html/2408.10205v1)  
8. KAN Regression \+ graduate-admissions \- Kaggle, acceso: diciembre 2, 2025, [https://www.kaggle.com/code/seyidcemkarakas/kan-regression-graduate-admissions](https://www.kaggle.com/code/seyidcemkarakas/kan-regression-graduate-admissions)  
9. Lagrangian Neural Networks \- Natural Intelligence, acceso: diciembre 2, 2025, [https://greydanus.github.io/2020/03/10/lagrangian-nns/](https://greydanus.github.io/2020/03/10/lagrangian-nns/)  
10. Lagrangian mechanics \- Wikipedia, acceso: diciembre 2, 2025, [https://en.wikipedia.org/wiki/Lagrangian\_mechanics](https://en.wikipedia.org/wiki/Lagrangian_mechanics)  
11. A nonlinear generalization of the Savitzky-Golay filter and the quantitative analysis of saccades | JOV | ARVO Journals, acceso: diciembre 2, 2025, [https://jov.arvojournals.org/article.aspx?articleid=2648985](https://jov.arvojournals.org/article.aspx?articleid=2648985)  
12. Is Langrangian Formalism Adequately Describing Energy Conservation? \- ScholarWorks@UTEP, acceso: diciembre 2, 2025, [https://scholarworks.utep.edu/cgi/viewcontent.cgi?article=1778\&context=cs\_techrep](https://scholarworks.utep.edu/cgi/viewcontent.cgi?article=1778&context=cs_techrep)  
13. Lagrangian Neural Network with Differential Symmetries and Relational Inductive Bias, acceso: diciembre 2, 2025, [https://www.researchgate.net/publication/355142323\_Lagrangian\_Neural\_Network\_with\_Differential\_Symmetries\_and\_Relational\_Inductive\_Bias](https://www.researchgate.net/publication/355142323_Lagrangian_Neural_Network_with_Differential_Symmetries_and_Relational_Inductive_Bias)  
14. Physics-Informed Neural Networks with Hard Constraints for Inverse Design | SIAM Journal on Scientific Computing \- DSpace@MIT, acceso: diciembre 2, 2025, [https://dspace.mit.edu/bitstream/handle/1721.1/138438/21m1397908.pdf?sequence=1](https://dspace.mit.edu/bitstream/handle/1721.1/138438/21m1397908.pdf?sequence=1)  
15. Using Hybrid Physics-Informed Neural Networks for Digital Twins in Prognosis and Health Management | NVIDIA Technical Blog, acceso: diciembre 2, 2025, [https://developer.nvidia.com/blog/using-hybrid-physics-informed-neural-networks-for-digital-twins-in-prognosis-and-health-management/](https://developer.nvidia.com/blog/using-hybrid-physics-informed-neural-networks-for-digital-twins-in-prognosis-and-health-management/)  
16. Scientific Machine Learning with Kolmogorov-Arnold Networks \- arXiv, acceso: diciembre 2, 2025, [https://arxiv.org/html/2507.22959v1](https://arxiv.org/html/2507.22959v1)  
17. Hello, KAN\! — Kolmogorov Arnold Network documentation \- Ziming Liu, acceso: diciembre 2, 2025, [https://kindxiaoming.github.io/pykan/intro.html](https://kindxiaoming.github.io/pykan/intro.html)  
18. Dissecting Kolmogorov-Arnold Network | by Fei Cheung \- Medium, acceso: diciembre 2, 2025, [https://feicheung2016.medium.com/dissecting-kolmogorov-arnold-network-f1bee719d949](https://feicheung2016.medium.com/dissecting-kolmogorov-arnold-network-f1bee719d949)  
19. Symbolic Regression Based on Kolmogorov–Arnold Networks for Gray-Box Simulation Program with Integrated Circuit Emphasis Model of Generic Transistors \- MDPI, acceso: diciembre 2, 2025, [https://www.mdpi.com/2079-9292/14/6/1161](https://www.mdpi.com/2079-9292/14/6/1161)  
20. kan package — Kolmogorov Arnold Network documentation \- Ziming Liu, acceso: diciembre 2, 2025, [https://kindxiaoming.github.io/pykan/kan.html](https://kindxiaoming.github.io/pykan/kan.html)  
21. Enforcing positive output values \- PyTorch Forums, acceso: diciembre 2, 2025, [https://discuss.pytorch.org/t/enforcing-positive-output-values/106851](https://discuss.pytorch.org/t/enforcing-positive-output-values/106851)  
22. Lagrangian neural networks for nonholonomic mechanics \- arXiv, acceso: diciembre 2, 2025, [https://arxiv.org/html/2411.00110v2](https://arxiv.org/html/2411.00110v2)  
23. Learning mechanical systems from real-world data using discrete forced Lagrangian dynamics \- arXiv, acceso: diciembre 2, 2025, [https://arxiv.org/html/2505.20370v1](https://arxiv.org/html/2505.20370v1)  
24. How to Avoid Trivial Solutions in Physics-Informed Neural Networks \- arXiv, acceso: diciembre 2, 2025, [https://arxiv.org/pdf/2112.05620](https://arxiv.org/pdf/2112.05620)  
25. How to avoid learning a trivial solution? · Issue \#356 · lululxvi/deepxde \- GitHub, acceso: diciembre 2, 2025, [https://github.com/lululxvi/deepxde/issues/356](https://github.com/lululxvi/deepxde/issues/356)  
26. Physics Informed Neural Networks (PINNs) \[Physics Informed Machine Learning\] \- YouTube, acceso: diciembre 2, 2025, [https://www.youtube.com/watch?v=-zrY7P2dVC4](https://www.youtube.com/watch?v=-zrY7P2dVC4)  
27. METHODS FOR NUMERICAL DIFFERENTIATION OF NOISY DATA 1\. Introduction The problem of approximating a derivative of a function defi, acceso: diciembre 2, 2025, [https://ejde.math.txstate.edu/conf-proc/21/k3/knowles.pdf](https://ejde.math.txstate.edu/conf-proc/21/k3/knowles.pdf)  
28. Physics-informed machine learning for robust inverse problem solving in maglev train levitation systems under noisy measurement | Request PDF \- ResearchGate, acceso: diciembre 2, 2025, [https://www.researchgate.net/publication/393571747\_Physics-informed\_machine\_learning\_for\_robust\_inverse\_problem\_solving\_in\_maglev\_train\_levitation\_systems\_under\_noisy\_measurement](https://www.researchgate.net/publication/393571747_Physics-informed_machine_learning_for_robust_inverse_problem_solving_in_maglev_train_levitation_systems_under_noisy_measurement)  
29. Numerical differentiation of noisy data: A unifying multi-objective optimization framework, acceso: diciembre 2, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7899139/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7899139/)  
30. Numerical Differentiation of Noisy Data (DoG and Savitzky–Golay Filters) \- YouTube, acceso: diciembre 2, 2025, [https://www.youtube.com/watch?v=h3WtjTuvhbU](https://www.youtube.com/watch?v=h3WtjTuvhbU)  
31. What is the best way to smooth and compute the derivatives of noisy data? \- MATLAB Answers \- MathWorks, acceso: diciembre 2, 2025, [https://www.mathworks.com/matlabcentral/answers/450562-what-is-the-best-way-to-smooth-and-compute-the-derivatives-of-noisy-data](https://www.mathworks.com/matlabcentral/answers/450562-what-is-the-best-way-to-smooth-and-compute-the-derivatives-of-noisy-data)  
32. \[2501.17281\] Stiff Transfer Learning for Physics-Informed Neural Networks \- arXiv, acceso: diciembre 2, 2025, [https://arxiv.org/abs/2501.17281](https://arxiv.org/abs/2501.17281)  
33. Characterizing possible failure modes in physics-informed neural networks, acceso: diciembre 2, 2025, [https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf)  
34. GradNorm-Based Dynamic Loss Balancing \- Emergent Mind, acceso: diciembre 2, 2025, [https://www.emergentmind.com/topics/gradnorm-based-dynamic-loss-balancing](https://www.emergentmind.com/topics/gradnorm-based-dynamic-loss-balancing)  
35. Multi-Objective Loss Balancing for Physics-Informed Deep Learning \- ETH Zurich Research Collection, acceso: diciembre 2, 2025, [https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/727362/1/1-s2.0-S0045782525001860-main.pdf](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/727362/1/1-s2.0-S0045782525001860-main.pdf)  
36. \[2110.09813\] Multi-Objective Loss Balancing for Physics-Informed Deep Learning \- arXiv, acceso: diciembre 2, 2025, [https://arxiv.org/abs/2110.09813](https://arxiv.org/abs/2110.09813)  
37. Physics informed neural network (PINN) for noise-robust phase-based magnetic resonance electrical properties tomography \- URSI, acceso: diciembre 2, 2025, [https://www.ursi.org/proceedings/procAT22/ATAPRASC2022-papers/YD0NQMGVDZ.pdf](https://www.ursi.org/proceedings/procAT22/ATAPRASC2022-papers/YD0NQMGVDZ.pdf)
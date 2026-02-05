# **Diseño Avanzado e Implementación de Control Sensorless de Alto Rendimiento para Sistemas de Levitación Magnética: Integración de Redes Neuronales Informadas por la Física (PINN) y Arquitecturas Kolmogorov-Arnold (KAN)**

## **1\. Resumen Ejecutivo y Arquitectura del Sistema**

El dominio de la suspensión electromagnética (EMS) y la levitación magnética (MagLev) representa un desafío por excelencia en la teoría de control no lineal, caracterizado por la inestabilidad de lazo abierto, dinámicas rápidas y una alta sensibilidad a las variaciones paramétricas. Mientras que las implementaciones tradicionales dependen de sensores de desplazamiento ópticos o de corrientes de Foucault de alta resolución, el paradigma del *control sensorless* —estimar la posición del objeto levitado únicamente a partir de mediciones eléctricas (voltaje y corriente)— ofrece ventajas convincentes en términos de reducción de costos, robustez mecánica y aplicabilidad en entornos hostiles donde los sensores pueden fallar o no pueden instalarse debido a restricciones de espacio o condiciones ambientales extremas.1

Este informe presenta una estrategia de diseño e implementación integral para un sistema de levitación magnética sensorless de alto rendimiento. La arquitectura propuesta aborda los dos desafíos fundamentales del MagLev sin sensores, que a menudo se tratan de forma aislada en la literatura pero que deben resolverse conjuntamente para una operación industrial viable: **(1) el "problema de arranque" (startup problem)**, donde la falta de conocimiento de la posición inicial impide la convergencia de los observadores de flujo magnético tradicionales, y **(2) el "problema de deriva y no linealidad"**, donde los observadores clásicos fallan al rastrear el estado a lo largo del tiempo debido a errores de integración numérica, variaciones térmicas en la resistencia de la bobina y dinámicas de corrientes parásitas (eddy currents) no modeladas.3

Para resolver estos problemas críticos, integramos tres tecnologías de vanguardia en un marco de control unificado:

1. **Inicialización Basada en Inductancia (Método de Pendiente de Corriente):** Una técnica de análisis transitorio que extrae la brecha de aire inicial (air gap) a partir de la respuesta de la pendiente de corriente ($di/dt$) a un escalón de voltaje, eliminando la dependencia de sensores físicos para la inicialización del observador y permitiendo un arranque desde cualquier posición arbitraria dentro del rango de captura.5  
2. **Red Híbrida Informada por la Física (Physics-Informed KAN):** Una arquitectura neural novedosa que combina un modelo físico determinista (la formulación de Santana 2023 derivada en el código proporcionado) con un corrector residual basado en Redes Kolmogorov-Arnold (KAN). A diferencia de los Perceptrones Multicapa (MLP) tradicionales que sufren de baja interpretabilidad y el problema del desvanecimiento del gradiente en funciones altamente no lineales, las KAN utilizan funciones de activación aprendibles en los bordes del grafo, ofreciendo una capacidad de aproximación de funciones superior para los mapas magnéticos altamente no lineales del levitador.7  
3. **Implementación de Computación de Alto Rendimiento (HPC) para Control en Tiempo Real:** Un motor de inferencia optimizado para latencia escrito en C++ que aprovecha las instrucciones SIMD (AVX2/AVX-512) y bibliotecas de álgebra lineal como Eigen o ONNX Runtime. Esta implementación está diseñada para ejecutar inferencias neurales complejas dentro de bucles de control estrictos ($T\_s \\le 100 \\mu s$), superando las limitaciones de latencia inherentes a las GPU para tamaños de lote unitarios (batch size \= 1\) típicos en sistemas de control de retroalimentación.10

El análisis se basa en una disección forense de los registros experimentales proporcionados (MONIT.txt), el código fuente del controlador (levitador.cpp, levitador\_kan\_sensorless.cpp), y una revisión exhaustiva de la literatura actual para sintetizar una solución robusta y lista para producción.

## ---

**2\. Fundamentos Teóricos de la Levitación Magnética Sensorless**

Para diseñar un observador capaz de reemplazar sensores físicos de alta precisión, es imperativo comprender profundamente la física subyacente que gobierna la interacción electromecánica. La levitación magnética no es simplemente un problema de control de posición; es un problema de gestión de energía magnética y co-energía en presencia de no linealidades geométricas y saturación del material ferromagnético.

### **2.1 Dinámica Electromecánica y el Problema de Observabilidad**

La dinámica de un levitador electromagnético de un solo grado de libertad se rige por la interacción acoplada entre el subsistema eléctrico (la bobina del electroimán) y el subsistema mecánico (la masa levitada). La ecuación de voltaje fundamental que describe el circuito eléctrico es una aplicación directa de la Ley de Faraday y la Ley de Ohm:

$$u(t) \= R(T) \\cdot i(t) \+ \\frac{d\\psi(y, i)}{dt}$$  
Donde $u(t)$ es el voltaje instantáneo aplicado a los terminales de la bobina, $R(T)$ es la resistencia de la bobina que es una función de la temperatura $T$, $i(t)$ es la corriente que fluye a través del devanado, y $\\psi$ es el enlace de flujo magnético (flux linkage). El enlace de flujo es una función no lineal tanto de la corriente como de la posición de la brecha de aire $y(t)$, y se define constitutivamente como:

$$\\psi(y, i) \= L(y, i) \\cdot i(t)$$  
Aquí, $L(y, i)$ representa la inductancia aparente de la bobina. En la mayoría de los modelos analíticos simplificados, se asume que la operación ocurre en la región lineal de la curva B-H del material magnético, lo que permite despreciar la dependencia de la corriente (saturación). Sin embargo, la dependencia de la posición es crítica y típicamente sigue una desintegración hiperbólica respecto a la brecha de aire. En el código proporcionado (levitador\_obs\_minimal.cpp), este comportamiento se modela explícitamente mediante la siguiente relación paramétrica 2:

$$L(y) \= K\_0 \+ \\frac{K}{1 \+ \\frac{y}{A}}$$  
Donde los parámetros tienen significados físicos específicos:

* $K\_0$: Inductancia de fuga (leakage inductance), que representa el valor asintótico de la inductancia cuando el objeto se retira al infinito ($y \\to \\infty$).  
* $K$: Parámetro de inductancia magnetizante, relacionado con la permeabilidad del núcleo y el número de vueltas.  
* $A$: Parámetro geométrico que caracteriza la escala de la brecha de aire efectiva y la dispersión del campo.

Las dinámicas mecánicas se rigen por la segunda ley de Newton, equilibrando la fuerza magnética de atracción contra la gravedad y cualquier perturbación externa:

$$m \\ddot{y} \= mg \- F\_{mag}(y, i) \+ F\_{dist}$$  
La fuerza magnética $F\_{mag}$ se deriva termodinámicamente de la co-energía del sistema electromagnético $W'\_{fld}(y, i) \= \\int\_0^i \\psi(y, \\xi) d\\xi$. Para el modelo de inductancia lineal en corriente dado anteriormente, la fuerza es:

$$F\_{mag} \= \\frac{\\partial W'\_{fld}}{\\partial y} \= \\frac{1}{2} i^2 \\frac{dL(y)}{dy} \= \-\\frac{1}{2} i^2 \\frac{K}{A \\left(1 \+ \\frac{y}{A}\\right)^2}$$  
Esta ecuación revela la inestabilidad inherente del sistema: a medida que la brecha de aire $y$ disminuye (el objeto se acerca al imán), la fuerza de atracción aumenta cuadráticamente con la inversa de la distancia, creando una retroalimentación positiva que tiende a colapsar el sistema si no se controla activamente.

El Desafío de la Estimación Sensorless:  
En una configuración puramente sensorless, el controlador tiene acceso solo a las variables de control $u(t)$ y a la medición de corriente $i(t)$. El objetivo es estimar la variable de estado no observada $y(t)$. Reorganizando la ecuación de voltaje, obtenemos una base teórica para esta estimación:

$$u \= Ri \+ L(y)\\frac{di}{dt} \+ i \\frac{dL(y)}{dy} \\frac{dy}{dt}$$  
El último término, $i \\frac{dL(y)}{dy} \\frac{dy}{dt}$, representa la "FEM de movimiento" (back-EMF). En los motores eléctricos convencionales que giran a alta velocidad, este término es dominante y proporciona una señal robusta para la estimación de la velocidad y la posición. Sin embargo, en la levitación magnética, surgen dos problemas fundamentales:

1. **Baja Velocidad:** Durante la regulación en un punto fijo (hovering), la velocidad $\\frac{dy}{dt}$ es cercana a cero, lo que hace que el término de EMF de movimiento desaparezca. Esto hace que la observabilidad basada en la velocidad sea extremadamente pobre o nula en el punto de operación nominal.14  
2. **Gradiente de Inductancia:** El gradiente $\\frac{dL}{dy}$ disminuye rápidamente con la distancia. A grandes brechas de aire, la sensibilidad de la inductancia a los cambios de posición es muy pequeña, reduciendo la relación señal-ruido (SNR) para cualquier estimador basado en back-EMF.

Debido a estas limitaciones, el enfoque moderno para MagLev sensorless no se basa en la back-EMF, sino en la **Estimación de Flujo Magnético**. Integrando la ecuación de voltaje, podemos recuperar el flujo instantáneo:

$$\\psi(t) \= \\int\_{0}^{t} (u(\\tau) \- R \\cdot i(\\tau)) d\\tau \+ \\psi(0)$$  
Si conocemos $\\psi(t)$ y medimos $i(t)$, podemos calcular la inductancia instantánea como $L\_{est} \= \\psi/i$. Dado que $L(y)$ es una función monótona de la posición, podemos invertir algebraicamente el modelo de inductancia para obtener $y$:

$$y \= A \\left( \\frac{K}{L\_{est} \- K\_0} \- 1 \\right)$$  
Este es el principio operativo detrás del "Observador Santana 2023" referenciado en los archivos del usuario. Sin embargo, este método introduce puntos críticos de falla que deben abordarse mediante técnicas avanzadas.

**Puntos Críticos de Falla en el Enfoque Clásico:**

1. **Condición Inicial Desconocida $\\psi(0)$:** La integral requiere un conocimiento perfecto del flujo en el tiempo $t=0$. Si el sistema arranca con corriente cero y el objeto en reposo lejos del imán, se puede asumir $\\psi(0) \\approx 0$. Sin embargo, si el sistema intenta atrapar un objeto en el aire o se reinicia tras una falla transitoria, $\\psi(0)$ es desconocida. El código proporcionado (levitador\_obs\_minimal.cpp) resalta explícitamente esta falla con el comentario: // ⚠️ NECESITA SENSOR PARA INICIALIZAR.13 Esto viola el requisito de ser 100% sensorless.  
2. **Deriva del Integrador (Integrator Drift):** Cualquier error en la estimación de la resistencia $R$ (debido al calentamiento óhmico) o pequeños offsets de CC en la medición de voltaje o corriente se acumulan en la integral $\\int (u \- Ri) dt$. Esto hace que el flujo estimado diverja de la realidad en cuestión de segundos, llevando a la inestabilidad del controlador.4  
3. **Mapeo No Lineal Incorrecto:** La fórmula analítica para $y(L)$ asume un modelo magnético ideal. Los solenoides reales exhiben histéresis magnética, saturación del núcleo y efectos de borde que distorsionan la relación $L(y)$, especialmente en los extremos del rango de operación.3

### **2.2 El Papel de las Corrientes de Foucault (Eddy Currents)**

Un factor frecuentemente subestimado en modelos teóricos simples, pero devastador en la práctica, son las corrientes de Foucault. Estas se inducen en cualquier material conductor cercano al campo magnético variable, como el propio núcleo del electroimán (si no está perfectamente laminado) o el objeto metálico que se levita.16

Desde una perspectiva de modelado de circuitos, las corrientes de Foucault actúan como una bobina secundaria en cortocircuito acoplada magnéticamente al devanado principal. Esto tiene dos efectos principales:

1. **Resistencia Fantasma:** Introduce una pérdida de energía adicional que aparece como un aumento en la resistencia efectiva de la bobina a altas frecuencias.  
2. **Retraso de Flujo (Shielding):** Las corrientes de Foucault crean un campo magnético opuesto que se opone al cambio de flujo principal (Ley de Lenz). Esto significa que cuando se aplica un escalón de voltaje, la corriente aumenta, pero el flujo magnético penetra en el material con un retraso temporal.

En los registros proporcionados (MONIT.txt), el comportamiento transitorio inicial muestra una respuesta que no es instantánea, lo que sugiere la presencia de dinámicas no ideales. La inversión algebraica simple de $L(y)$ falla porque asume que el flujo y la corriente están en fase y relacionados estáticamente. En la realidad dinámica, el "retraso" causado por las corrientes de Foucault hace que la inductancia aparente sea diferente durante los transitorios rápidos (como la conmutación PWM) en comparación con el estado estacionario.16 Esto requiere un estimador más robusto, como un Observador Neural o PINN, que pueda aprender y compensar estas dinámicas no modeladas.

## ---

**3\. Estimación de Posición Inicial: El Método de Gradiente de Inductancia**

La falla más inmediata y obvia en el observador base proporcionado en levitador\_obs\_minimal.cpp es su dependencia de un sensor físico para la inicialización. El código contiene la siguiente lógica:

C++

if (\!obs\_init && i \> 0.03f) {  
    obs\_init \= 1;  
    obs\_y\_est \= y\_sensor; // ⚠️ TRAMPA: Usa el sensor físico  
    float L\_init \= K0 \+ K / (1.0f \+ obs\_y\_est / A);  
    obs\_phi \= L\_init \* i;  
}

Para lograr un sistema verdaderamente sensorless, este bloque debe ser reemplazado por una rutina de estimación activa. El **Método de Pendiente de Corriente** (Current Slope Method o $di/dt$) es el estándar industrial para este propósito en motores de reluctancia conmutada y cojinetes magnéticos activos.6

### **3.1 Derivación Matemática del Método di/dt**

El principio se basa en la respuesta transitoria del inductor a un cambio abrupto de voltaje. En el instante preciso en que se aplica un pulso de voltaje (por ejemplo, el inicio de un ciclo PWM), la corriente $i$ es inicialmente cero (o constante en un valor previo). Si analizamos el instante $t=0^+$ de un pulso de voltaje $V\_{bus}$:

1. La caída de tensión resistiva $R \\cdot i$ es despreciable si $i \\approx 0$ o si consideramos la inductancia incremental.  
2. La velocidad mecánica $\\frac{dy}{dt}$ es cero debido a la inercia de la masa (el objeto no puede acelerar instantáneamente). Por lo tanto, el término de back-EMF es nulo.

Bajo estas condiciones, la ecuación de voltaje se simplifica dramáticamente a:

$$u(t) \\approx L(y\_0) \\frac{di}{dt} \\quad \\Rightarrow \\quad L(y\_0) \\approx \\frac{V\_{bus}}{\\left. \\frac{di}{dt} \\right|\_{t=0}}$$  
Donde $V\_{bus}$ es el voltaje del bus de CC (identificado como 16.0V en MONIT.txt). Esta relación establece que la inductancia inicial $L(y\_0)$ es inversamente proporcional a la pendiente inicial de la corriente. Una vez obtenido $L(y\_0)$, podemos invertir el modelo geométrico para encontrar $y\_0$.

### **3.2 Análisis Forense de MONIT.txt para Estimación de Inductancia**

El archivo de registro MONIT.txt 13 proporciona una ventana invaluable al comportamiento real del sistema. Aunque carece de encabezados, el análisis de patrones de datos permite identificar las columnas críticas:

* **Columna 1:** Tiempo ($t$), incrementando en pasos de 0.01s.  
* **Columna 5:** Voltaje de Bus ($V\_{bus} \\approx 16.0$ V, constante).  
* **Columna 8:** Corriente medida ($i$).

Examinemos la traza de datos alrededor del intervalo de tiempo $t=0.0400$ a $0.0500$, donde ocurre un evento de escalón claro:

* $t \= 0.0400, i \= 0.0000$ A  
* $t \= 0.0500, i \= 0.4693$ A

Podemos calcular la derivada discreta (pendiente) de la corriente:

$$\\frac{di}{dt} \\approx \\frac{\\Delta i}{\\Delta t} \= \\frac{0.4693 \- 0.0000}{0.0500 \- 0.0400} \= \\frac{0.4693}{0.0100} \= 46.93 \\, \\text{A/s}$$  
Utilizando la ecuación simplificada de voltaje con $V\_{bus} \= 16.0$ V, estimamos la inductancia aparente:

$$L\_{est} \\approx \\frac{V\_{bus}}{di/dt} \= \\frac{16.0}{46.93} \\approx 0.341 \\, \\text{H}$$  
A continuación, utilizamos los parámetros del modelo definidos en el código levitador\_obs\_minimal.cpp 13 para mapear esta inductancia a una posición:

* $K\_0 \= 0.0657$ H  
* $K \= 0.0393$ H  
* $A \= 0.00498$ m

Invertimos el modelo de inductancia $L \= K\_0 \+ \\frac{K}{1 \+ y/A}$:

$$L \- K\_0 \= \\frac{K}{1 \+ y/A} \\quad \\Rightarrow \\quad 1 \+ \\frac{y}{A} \= \\frac{K}{L \- K\_0}$$

$$y \= A \\left( \\frac{K}{L \- K\_0} \- 1 \\right)$$  
Sustituyendo los valores calculados:

$$y \= 0.00498 \\left( \\frac{0.0393}{0.341 \- 0.0657} \- 1 \\right)$$

$$y \= 0.00498 \\left( \\frac{0.0393}{0.2753} \- 1 \\right) \= 0.00498 (0.1427 \- 1\) \\approx \-0.0042 \\, \\text{m}$$  
El resultado es **\-4.2 mm**, lo cual es físicamente imposible para una brecha de aire (la distancia debe ser positiva). Esta discrepancia negativa es extremadamente reveladora e indica dos fenómenos físicos críticos que deben ser compensados en el diseño:

1. **Limitación de la Frecuencia de Muestreo:** El paso de tiempo de 0.01s (100 Hz) en el registro es órdenes de magnitud demasiado lento para capturar la verdadera pendiente transitoria $di/dt$. En un solenoide real, la corriente puede subir mucho más rápido en los primeros microsegundos. La medición "promediada" de 0.01s subestima drásticamente la pendiente instantánea en $t=0$, lo que lleva a una sobreestimación de la inductancia ($L \\approx 0.341$ H es muy alto comparado con los parámetros base $K\_0+K \\approx 0.1$ H).  
2. **Apantallamiento por Corrientes de Foucault:** Como se discutió en la sección 2.2, las corrientes de Foucault se oponen al cambio de flujo, lo que efectivamente reduce la inductancia vista por la fuente durante transitorios de alta frecuencia. Sin embargo, si el muestreo es lento, vemos el efecto contrario: la resistencia efectiva aumenta y la corriente tarda más en subir, lo que puede interpretarse erróneamente.

**Conclusión del Análisis de Datos:** Para implementar esto correctamente, el hardware no puede depender del bucle de control principal de 100 Hz. Se requiere una **rutina de interrupción de alta velocidad** o un periférico dedicado (como el disparador del ADC sincronizado con el PWM) para medir la corriente en una ventana de tiempo de $10-100 \\mu s$ después del encendido del pulso.21

### **3.3 Algoritmo Propuesto para Inicialización de Arranque (Startup)**

Para implementar una solución robusta en el controlador C++ (levitador.cpp), proponemos reemplazar la inicialización tramposa con una rutina de "inyección de señal" de pre-arranque.

**Algoritmo de Inicialización Sensorless:**

1. **Estado IDLE:** El sistema comienza con el PWM apagado y el control PID deshabilitado.  
2. **Inyección de Pulso de Prueba:** Se aplica un ciclo de trabajo de PWM fijo y corto (por ejemplo, un pulso de $100 \\mu s$ a $16$V). Esto es lo suficientemente corto para no mover el objeto significativamente (filtrado mecánico por inercia) pero suficiente para generar una rampa de corriente medible.  
3. **Muestreo de Alta Velocidad (Burst Sampling):** El ADC se configura para tomar una ráfaga de muestras (e.g., 10 muestras a 1 MHz) inmediatamente después del flanco de subida del pulso.  
4. **Cálculo de Pendiente (Regresión Lineal):** Se calcula $\\frac{di}{dt}$ ajustando una línea recta a las muestras de corriente para minimizar el ruido de cuantificación.  
5. Cálculo de Posición Inicial ($y\_0$): Se utiliza la fórmula invertida derivada anteriormente, pero calibrada con un factor de corrección $\\alpha\_{eddy}$ para compensar las corrientes de Foucault:

   $$L\_{eff} \= \\frac{V\_{bus} \- I\_{offset}R}{di/dt}$$  
   $$y\_{init} \= A \\left( \\frac{K}{\\alpha\_{eddy} \\cdot (L\_{eff} \- K\_0)} \- 1 \\right)$$  
6. Inicialización del Observador: Se establece el estado interno del observador de flujo:

   $$\\psi\_{est}(0) \= L(y\_{init}) \\cdot i\_{medido}(0)$$  
7. **Transición a Control Cerrado:** Se habilita el PID utilizando $y\_{est}$ del observador KAN como retroalimentación.

Esta secuencia elimina la dependencia de la variable y\_sensor y resuelve el problema del "huevo y la gallina" de la estimación de flujo.

## ---

**4\. Redes Neuronales Kolmogorov-Arnold (KAN): Arquitectura y Ventajas**

Mientras que el modelo analítico de inductancia es útil para la inicialización estática, es insuficiente para el seguimiento continuo de alta precisión debido a la histéresis magnética, la deriva térmica y las no linealidades complejas que surgen durante el movimiento. Para abordar esto, empleamos una **Red Kolmogorov-Arnold (KAN)** como un observador robusto basado en datos.

### **4.1 Teorema de Representación y Superioridad sobre MLP**

El Teorema de Representación de Kolmogorov-Arnold establece que cualquier función continua multivariada puede representarse como una superposición de funciones univariadas continuas y sumas:

$$f(\\mathbf{x}) \= \\sum\_{q=1}^{2n+1} \\Phi\_q \\left( \\sum\_{p=1}^{n} \\phi\_{q,p}(x\_p) \\right)$$  
Esta formulación difiere radicalmente de los Perceptrones Multicapa (MLP) tradicionales.

* **MLP:** Utiliza funciones de activación *fijas* (ReLU, Sigmoid, Tanh) en las neuronas y pesos lineales aprendibles en las conexiones.  
* **KAN:** Utiliza funciones de activación *aprendibles* (típicamente B-splines o funciones base ponderadas) en las conexiones (bordes), y nodos de suma simple.7

**Ventajas de KAN para Control de Levitación:**

1. **Interpretabilidad Física:** Las funciones univariadas aprendidas $\\phi\_{q,p}$ en una KAN a menudo convergen a formas que se asemejan a las leyes físicas subyacentes. Por ejemplo, una KAN entrenada en datos de levitación podría aprender internamente una función que se parece a la curva hiperbólica $1/(1+y)$ de la inductancia, haciendo que la "caja negra" sea transparente.  
2. **Eficiencia de Parámetros:** Las KANs han demostrado requerir órdenes de magnitud menos parámetros que los MLPs para lograr la misma precisión en la aproximación de funciones científicas y de ingeniería. Esto es crucial para la implementación en tiempo real, donde cada operación de punto flotante cuenta.8  
3. **Continuidad y Suavidad:** Al utilizar bases como B-splines o funciones trigonométricas suaves, las KANs garantizan que la salida sea diferenciable. Esto es crítico para el control PID, ya que el término derivativo $D$ amplificaría cualquier discontinuidad o ruido de cuantificación proveniente de un observador basado en tablas de búsqueda o redes ReLU simples.

### **4.2 Análisis de la Implementación KAN en C++ (levitador\_kan\_sensorless.cpp)**

El código proporcionado en levitador\_kan\_sensorless.cpp 13 implementa una variante optimizada de KAN diseñada específicamente para la velocidad de inferencia, que podemos clasificar como **Fourier-KAN** o **Trig-KAN**.

**Arquitectura de la Red:**

* **Capa de Entrada (2 nodos):** Recibe el Flujo Magnético Normalizado ($\\phi\_{norm}$) y la Corriente Normalizada ($i\_{norm}$).  
* **Capa Oculta 1 (Expansión de Características \- 64 nodos):** En lugar de B-splines (que son computacionalmente costosos de evaluar debido a la lógica de ramificación), esta implementación utiliza funciones base trigonométricas para capturar no linealidades periódicas y armónicas:  
  * Nodos 0-31: $\\text{ReLU}(\\sin(w \\cdot \\phi \\cdot \\pi) \\cdot i)$ — Modela la interacción flujo-corriente sinusoidal.  
  * Nodos 32-63: $\\text{ReLU}(\\cos(w \\cdot i \\cdot \\pi) \\cdot \\phi)$ — Modela la interacción complementaria.  
    Esta estructura realiza esencialmente un mapeo de características de Fourier, permitiendo a la red aprender correlaciones de alta frecuencia entre el flujo y la corriente.  
* **Capa Oculta 2 (Agregación \- 32 nodos):** Realiza una suma ponderada de las características expandidas seguida de una activación ReLU.  
* **Capa de Salida (1 nodo):** Suma final con una activación Sigmoid para restringir la estimación de posición al rango normalizado $$.

**Crítica Técnica de la Implementación:**

* **Pesos Procedurales:** El código utiliza una generación procedimental de pesos (w \= (i \+ j) % 10...) para demostración. En un despliegue de producción, estos pesos deben ser cargados desde un archivo de cabecera (.h) generado durante el entrenamiento offline en Python.  
* **Trigonometría vs. Splines:** El uso de sinf/cosf es extremadamente eficiente en CPUs modernas con instrucciones SIMD, a diferencia de la evaluación de B-splines de tercer orden que requiere múltiples comparaciones. Aunque las B-splines ofrecen control local (modificar un peso solo afecta una región de la función), las bases trigonométricas son globales (afectan todo el dominio). Para un problema de control con un rango de operación acotado, la aproximación espectral (trigonométrica) es una decisión de ingeniería sólida para maximizar la velocidad.

### **4.3 Estrategia de Entrenamiento Híbrido ("Aprendizaje de Residuos")**

El script de entrenamiento en Python (entrenar\_kan\_hibrido.py) 13 revela una estrategia sofisticada de **Modelado Híbrido Informado por la Física**.

El sistema no entrena a la KAN para predecir la posición $y$ desde cero ($y \= \\text{KAN}(\\phi, i)$). En su lugar, utiliza la fórmula física de Santana 2023 como base y entrena a la red para predecir el **error residual** de dicha fórmula:

$$y\_{target} \= y\_{sensor} \- y\_{formula}(\\phi, i)$$

$$y\_{final} \= y\_{formula}(\\phi, i) \+ \\text{KAN}(\\phi, i, y\_{formula})$$  
**Por qué esta estrategia es superior:**

1. **Corrección de Deriva:** El modelo analítico captura la física dominante (la ley inversa de la inductancia). La KAN solo necesita aprender las *desviaciones* de segundo orden causadas por la deriva de la resistencia, las corrientes de Foucault y el ruido del sensor. Esto simplifica enormemente la tarea de aprendizaje.  
2. **Seguridad y Robustez (Fail-Safe):** Si la red neuronal falla o produce una salida cercana a cero (por ejemplo, en una región del espacio de estados no vista durante el entrenamiento), el sistema degrada suavemente al modelo físico base, que es una aproximación segura y estable. Esto es fundamental para la seguridad en sistemas de levitación.  
3. **Convergencia Acelerada:** El rango dinámico del residuo es mucho menor que el rango completo de posición, lo que hace que el paisaje de optimización sea más suave y la convergencia del entrenamiento sea más rápida y precisa.

## ---

**5\. Mejoras con Redes Neuronales Informadas por la Física (PINN)**

Para endurecer aún más el observador contra la deriva del integrador —el modo de fallo principal identificado en la Sección 2.1— debemos transicionar de una KAN supervisada pura a una formulación completa de **Red Neuronal Informada por la Física (PINN)**.

### **5.1 Formulación de la Función de Pérdida Física (Physics Loss)**

El entrenamiento estándar minimiza el desajuste con los datos ($MSE\_{data}$). Una PINN añade un término de pérdida residual basado en la ecuación diferencial que gobierna el sistema, permitiendo que la red aprenda incluso en regiones donde no hay datos de sensores, simplemente obedeciendo las leyes de la física.24

Definimos el residuo físico $\\mathcal{R}\_{physics}$ basándonos en la conservación de la ecuación de voltaje:

$$\\mathcal{R}\_{physics} \= u(t) \- \\left( R \\cdot i(t) \+ \\frac{d\\hat{\\psi}}{dt} \\right)$$  
Dado que la KAN predice la posición $y$ y no el flujo directamente, debemos relacionar la salida de la red con el flujo. Invertimos la lógica: la red debe predecir una posición tal que, si calculamos el flujo correspondiente a esa posición, su derivada temporal coincida con el voltaje aplicado menos la caída resistiva.

Función de Pérdida PINN Propuesta:  
$$ \\mathcal{L}{total} \= \\underbrace{w\_1 ||y{pred} \- y\_{sensor}||^2}\_{\\text{Pérdida de Datos Supervisada}} \+ \\underbrace{w\_2 |  
| \\frac{d}{dt}(L(y\_{pred}) \\cdot i) \- (u \- Ri) ||^2}{\\text{Pérdida Física (Voltaje)}} \+ \\underbrace{w\_3 |  
| \\hat{y}{t=0} \- y\_{init\_slope} ||^2}\_{\\text{Pérdida de Frontera}} $$

* **Pérdida Supervisada:** Garantiza la precisión en los puntos donde se dispone de datos históricos del sensor (entrenamiento offline).  
* **Pérdida Física:** Asegura que la trayectoria estimada sea consistente con la Ley de Faraday. Esto actúa como un regularizador no supervisado, crucial para generalizar a puntos de operación no vistos en el conjunto de datos.26  
* **Pérdida de Frontera:** Ancla la solución a la estimación basada en inductancia ($di/dt$) derivada en la Sección 3, forzando a la red a respetar la condición inicial física.

### **5.2 Corrección de Deriva No Supervisada (Aprendizaje de Parámetros)**

Una de las aplicaciones más poderosas de las PINN en este contexto es la **adaptación de ganancia no supervisada**.28 La resistencia de la bobina $R$ cambia significativamente con la temperatura (el calentamiento óhmico puede aumentar $R$ en un 20-30%). Un valor fijo de $R$ en el observador provoca una deriva de integración catastrófica.

Podemos convertir a $R$ en un parámetro aprendible dentro de la PINN (o una salida secundaria de la red). Durante la operación online, la red ajusta $\\hat{R}$ para minimizar la Pérdida Física $\\mathcal{L}\_{physics}$ en tiempo real. Si la integral del flujo $\\int (u \- \\hat{R}i)$ comienza a divergir de la física esperada, el gradiente de la pérdida física forzará un ajuste en $\\hat{R}$ para restaurar la conservación de la energía, corrigiendo así la deriva sin necesidad de un sensor de temperatura externo.

## ---

**6\. Implementación de Computación de Alto Rendimiento (HPC)**

El control en tiempo real de sistemas de levitación magnética inestables requiere tiempos de ciclo extremadamente rápidos, típicamente $T\_s \\le 1$ ms, y a menudo tan bajos como $100 \\mu s$ para garantizar una rigidez adecuada. Ejecutar una inferencia de red neuronal dentro de esta ventana temporal en hardware embebido o un PC de control requiere una optimización agresiva.

### **6.1 Aceleración de Hardware: CPU (AVX2) vs. GPU (CUDA)**

Aunque la solicitud del usuario menciona "HPC/GPU", un análisis técnico riguroso de las cargas de trabajo de control revela que las GPU son a menudo subóptimas para este caso de uso específico.

**Análisis de Latencia (Benchmark):**

* **GPU (CUDA/TensorRT):** Las GPU están diseñadas para *throughput* masivo (procesar miles de imágenes en paralelo). Para un bucle de control, el "tamaño de lote" (batch size) es obligatoriamente **1** (una medición, una decisión). La sobrecarga de transferencia de datos a través del bus PCIe y el tiempo de lanzamiento del kernel (kernel launch overhead) introducen una latencia de **2 a 5 ms**, lo cual es inaceptable para un bucle de control de 1 ms.11  
* **CPU (AVX2/AVX-512):** Las CPU modernas pueden ejecutar multiplicaciones de matrices para redes pequeñas (como la KAN 64x32 propuesta) en cuestión de **microsegundos**. La latencia está limitada puramente por el cómputo, evitando los costos de transferencia de memoria externa.

**Recomendación Técnica:** Para la escala específica de esta KAN (64 nodos ocultos) y el requisito de tiempo real estricto, la **Vectorización en CPU (AVX2)** es la opción superior. La aceleración por GPU debe reservarse exclusivamente para el entrenamiento offline de la PINN o para simulaciones masivas de "gemelos digitales", pero no para el controlador en tiempo real.

### **6.2 Implementación con ONNX Runtime y Eigen**

La estrategia de implementación propuesta aprovecha **ONNX Runtime** para el despliegue del modelo, permitiendo que los modelos entrenados en Python (PyTorch/TensorFlow) se ejecuten en C++ con aceleración de hardware nativa.

**Pipeline de Inferencia en C++ (Optimizado):**

El siguiente pseudocódigo ilustra cómo integrar la inferencia KAN de alto rendimiento en el bucle de control C++, utilizando Ort::IoBinding para minimizar copias de memoria:

C++

// Pseudocódigo para Inferencia KAN de Alto Rendimiento  
\#**include** \<onnxruntime\_cxx\_api.h\>  
\#**include** \<Eigen/Dense\>

// Configuración de Sesión (Hacer UNA VEZ en la inicialización)  
// Usar el proveedor de ejecución CPU con instrucciones AVX2 habilitadas  
Ort::Env env(ORT\_LOGGING\_LEVEL\_WARNING, "LevitadorKAN");  
Ort::SessionOptions session\_options;  
session\_options.SetIntraOpNumThreads(1); // Single thread para latencia determinista  
session\_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT\_ENABLE\_ALL);  
Ort::Session session(env, "modelo\_kan\_hibrido.onnx", session\_options);

// Pre-asignación de tensores de entrada/salida para evitar malloc en el bucle  
std::vector\<float\> input\_data(3); // \[phi, i, y\_formula\]  
std::vector\<float\> output\_data(1);  
//... configuración de IoBinding...

// BUCLE DE CONTROL (tiempo real)  
void loop\_control() {  
    // 1\. Adquisición y Preprocesamiento (Normalización usando Eigen)  
    Eigen::Vector3f input\_vec(phi\_meas, i\_meas, y\_formula\_val);  
    input\_vec \= (input\_vec \- mean\_vec).cwiseQuotient(std\_dev\_vec);  
      
    // Copiar a buffer de entrada (o usar punteros directos si es posible)  
    memcpy(input\_data.data(), input\_vec.data(), 3 \* sizeof(float));

    // 2\. Ejecutar Inferencia (Batch Size \= 1\)  
    // RunWithBinding es más rápido que Run estándar  
    session.Run(Ort::RunOptions{nullptr}, io\_binding);

    // 3\. Post-procesamiento  
    float residuo\_pred \= output\_data; // Salida de la red  
    float y\_final \= y\_formula\_val \+ residuo\_pred; // Aplicar corrección híbrida  
}

**Técnicas de Optimización Clave:**

1. **Reutilización de Sesión:** Inicializar Ort::Session una sola vez. Recrearla en cada ciclo destruiría el rendimiento.  
2. **IoBinding:** Utilizar Ort::IoBinding para vincular buffers de memoria preasignados directamente a las entradas/salidas del modelo, evitando copias de datos innecesarias entre la aplicación y el motor de inferencia.30  
3. **Optimizaciones de Grafo:** Habilitar ORT\_ENABLE\_ALL para permitir la fusión de nodos (fusionar multiplicaciones y sumas en una sola instrucción FMA) y el plegado de constantes.  
4. **Activación Vectorizada:** Si se implementa la KAN manualmente en C++ (sin ONNX), se deben usar bibliotecas como FastOps o intrínsecos AVX crudos para aproximaciones rápidas de sin/cos/exp, que son 4-8 veces más rápidas que las implementaciones estándar std::sin.31

### **6.3 Optimización SIMD de las Funciones Base KAN**

Si se opta por una implementación manual de la KAN en C++ (como en levitador\_kan\_sensorless.cpp) para evitar la dependencia de ONNX, el cuello de botella será la evaluación de las 64 funciones trigonométricas. Esto puede optimizarse drásticamente usando instrucciones AVX2.

**Código Escalar (Lento):**

C++

for(int j=0; j\<32; j++) {  
    h1\[j\] \= relu(sinf(w \* phi \* PI) \* i);  
}

**Código Vectorizado AVX2 (Conceptual \- Rápido):**

C++

// Procesar 8 nodos simultáneamente (registros de 256 bits contienen 8 floats)  
\_\_m256 phi\_vec \= \_mm256\_set1\_ps(phi);  
\_\_m256 i\_vec \= \_mm256\_set1\_ps(i);  
\_\_m256 pi\_vec \= \_mm256\_set1\_ps(3.14159f);

for(int j=0; j\<32; j+=8) {  
    \_\_m256 w\_vec \= \_mm256\_load\_ps(weights \+ j); // Cargar 8 pesos  
    \_\_m256 arg \= \_mm256\_mul\_ps(w\_vec, phi\_vec); // w \* phi  
    arg \= \_mm256\_mul\_ps(arg, pi\_vec);           // w \* phi \* pi  
      
    // Aproximación rápida de seno usando polinomio o lookup  
    \_\_m256 sin\_val \= fast\_sin\_avx2(arg);   
      
    \_\_m256 res \= \_mm256\_mul\_ps(sin\_val, i\_vec); // sin(...) \* i  
      
    // ReLU Vectorizada: max(0, res)  
    res \= \_mm256\_max\_ps(res, \_mm256\_setzero\_ps());  
      
    \_mm256\_store\_ps(h1 \+ j, res); // Guardar 8 resultados  
}

Esta implementación reduce el conteo de ciclos para la capa de expansión base aproximadamente en un factor de 8, permitiendo que la red compleja se ejecute con una sobrecarga mínima en la CPU.34

## ---

**7\. Estrategia de Control Integrada y Validación Experimental**

### **7.1 El Bucle de Control Completo**

La arquitectura final integra el estimador de inductancia para el arranque, el observador híbrido PINN/KAN para el seguimiento, y el controlador PID existente en una máquina de estados unificada y robusta.

**Máquina de Estados del Controlador:**

1. **IDLE:** Espera comando de usuario. PWM deshabilitado.  
2. **INIT\_L (Arranque Sensorless):**  
   * Inyectar pulso de voltaje de $100 \\mu s$.  
   * Medir ráfaga de corriente.  
   * Calcular pendiente $di/dt$.  
   * Estimar $L\_{init}$ y $y\_0$ usando el modelo físico invertido.  
   * Inicializar integrador de flujo: $\\psi\_0 \= L\_{init} \\cdot i\_{medido}$.  
3. **CONTROL (Regulación Activa):**  
   * **Medición:** Leer $u\_{k}$ (ciclo de trabajo anterior) y $i\_{k}$ (ADC).  
   * **Adaptación:** Actualizar estimación de $R\_{k}$ (usando la salida de corrección de la PINN si está habilitada, o un modelo térmico simple).  
   * **Observación:**  
     * Integrar flujo: $\\psi\_{k} \= \\psi\_{k-1} \+ (u\_{k-1} \- R i\_{k-1})T\_s$.  
     * Predicción Base: $y\_{base} \= \\text{Formula2023}(\\psi\_k, i\_k)$.  
     * Corrección Neural: $y\_{residuo} \= \\text{KAN}\_{PINN}(\\psi\_k, i\_k, y\_{base})$.  
     * Posición Total: $y\_{est} \= y\_{base} \+ y\_{residuo}$.  
   * **Feedback:** Calcular error $e\_k \= y\_{ref} \- y\_{est}$.  
   * **PID:** Calcular comando de voltaje $u\_{cmd} \= K\_p e\_k \+ K\_d (e\_k \- e\_{k-1}) \+ K\_i \\sum e$.  
   * **Actuación:** Actualizar PWM.

### **7.2 Resultados Experimentales (Proyectados basados en Logs)**

Basado en el análisis del log MONIT.txt, el sistema actual (presumiblemente con el PID y sensor físico) muestra un tiempo de asentamiento de aproximadamente 0.4s con un sobreimpulso significativo y una oscilación amortiguada (sistema subamortiguado de segundo orden).

* **Rendimiento Sensorless Esperado:** Al reemplazar la inicialización "tramposa" con el método $di/dt$, el sistema será capaz de levitar desde un "arranque en frío" (masa apoyada en el tope) sin intervención manual. La precisión inicial dependerá de la calidad de la medición de la pendiente de corriente.  
* **Precisión en Estado Estacionario:** El enfoque KAN Híbrido reducirá drásticamente el error de posición en estado estacionario al compensar la no linealidad de la curva flujo-posición ($L(y)$), que típicamente se desvía un 5-10% de los modelos analíticos simples en los extremos del recorrido.2  
* **Robustez:** La regularización PINN asegura que incluso si la medición sensorless es ruidosa, el estado estimado no violará la conservación de la energía (dinámica del enlace de flujo), previniendo la "divergencia" común en observadores de IA pura.

## ---

**8\. Conclusión**

Este informe establece un marco riguroso para la implementación de levitación magnética sensorless de alto rendimiento. Al trascender los modelos analíticos simples hacia una **arquitectura Híbrida PINN/KAN**, abordamos las no linealidades e incertidumbres inherentes de la planta electromagnética con una precisión sin precedentes. Crucialmente, la incorporación de la **Estimación de Inductancia por Pendiente de Corriente** proporciona el eslabón perdido para un arranque sensorless verdadero, eliminando la necesidad de fijación mecánica inicial o sensores auxiliares.

La estrategia de implementación en **C++ con optimizaciones AVX2/ONNX** asegura que estos modelos avanzados de IA puedan ejecutarse dentro de las estrictas restricciones de tiempo real (\<1ms) de los sistemas magnéticos inestables, tendiendo un puente entre la teoría moderna del aprendizaje automático científico y el control embebido industrial.

### **Recomendaciones para Trabajos Futuros:**

1. **Recolección de Datos Especializada:** Generar un nuevo conjunto de datos que capture específicamente la pendiente $di/dt$ en varias posiciones fijas (clamped) para entrenar un mapa de inicialización $y \= f(di/dt)$ más preciso que la fórmula analítica.  
2. **Entrenamiento Refinado:** Reentrenar la KAN utilizando la función de pérdida híbrida completa (Datos \+ Pérdida Física de Voltaje) para mejorar la generalización fuera de la distribución de entrenamiento.  
3. **Despliegue:** Portar el código C++ optimizado al controlador objetivo, asegurando que la función observador\_paso utilice la nueva lógica de inicialización basada en $di/dt$ en lugar de la lectura del sensor.

#### **Fuentes citadas**

1. A novel robust position estimator for self-sensing magnetic levitation systems based on least squares identification | Request PDF \- ResearchGate, acceso: diciembre 23, 2025, [https://www.researchgate.net/publication/251629578\_A\_novel\_robust\_position\_estimator\_for\_self-sensing\_magnetic\_levitation\_systems\_based\_on\_least\_squares\_identification](https://www.researchgate.net/publication/251629578_A_novel_robust_position_estimator_for_self-sensing_magnetic_levitation_systems_based_on_least_squares_identification)  
2. A novel robust position estimator for self-sensing magnetic levitation systems based on least squares identification \- ACIN – TU Wien, acceso: diciembre 23, 2025, [https://www.acin.tuwien.ac.at/fileadmin/cds/pre\_post\_print/glueck2011.pdf](https://www.acin.tuwien.ac.at/fileadmin/cds/pre_post_print/glueck2011.pdf)  
3. A Solution to Ambiguities in Position Estimation for Solenoid Actuators by Exploiting Eddy Current Variations \- NIH, acceso: diciembre 23, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7349334/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7349334/)  
4. Integrator Drift Compensation of Magnetic Flux Transducers by Feed-Forward Correction, acceso: diciembre 23, 2025, [https://pubmed.ncbi.nlm.nih.gov/31835731/](https://pubmed.ncbi.nlm.nih.gov/31835731/)  
5. Sensorless control of switched reluctance motor based on inductance characteristic point under magnetic saturation \- PMC \- NIH, acceso: diciembre 23, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11696007/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11696007/)  
6. How to Calculate the Inductance of a Solenoid | Physics \- Study.com, acceso: diciembre 23, 2025, [https://study.com/skill/learn/how-to-calculate-the-inductance-of-a-solenoid-explanation.html](https://study.com/skill/learn/how-to-calculate-the-inductance-of-a-solenoid-explanation.html)  
7. KAN: Kolmogorov–Arnold Networks \- OpenReview, acceso: diciembre 23, 2025, [https://openreview.net/forum?id=Ozo7qJ5vZi](https://openreview.net/forum?id=Ozo7qJ5vZi)  
8. Integrating Kolmogorov–Arnold Networks with Time Series Prediction Framework in Electricity Demand Forecasting \- MDPI, acceso: diciembre 23, 2025, [https://www.mdpi.com/1996-1073/18/6/1365](https://www.mdpi.com/1996-1073/18/6/1365)  
9. A Comprehensive Survey on Kolmogorov Arnold Networks (KAN) \- arXiv, acceso: diciembre 23, 2025, [https://arxiv.org/html/2407.11075v4](https://arxiv.org/html/2407.11075v4)  
10. C++ Code Generation for Fast Inference of Deep Learning Models in ROOT/TMVA \- EPJ Web of Conferences, acceso: diciembre 23, 2025, [https://www.epj-conferences.org/articles/epjconf/pdf/2021/05/epjconf\_chep2021\_03040.pdf](https://www.epj-conferences.org/articles/epjconf/pdf/2021/05/epjconf_chep2021_03040.pdf)  
11. Scaling-up PyTorch inference: Serving billions of daily NLP inferences with ONNX Runtime \- Microsoft Open Source Blog, acceso: diciembre 23, 2025, [https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime](https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime)  
12. High-performance SAM2 inference framework with TensorRT | TIER IV, Inc., acceso: diciembre 23, 2025, [https://tier4.jp/en/media/detail/?sys\_id=1TVQeUacQGhCWaGN8D2cQD\&category=BLOG](https://tier4.jp/en/media/detail/?sys_id=1TVQeUacQGhCWaGN8D2cQD&category=BLOG)  
13. levitador\_obs\_minimal.cpp  
14. POSITION ESTIMATION IN SOLENOID ACTUATORS \- Dr. Norbert Cheung, acceso: diciembre 23, 2025, [https://www.norbert.idv.hk/Files\_Papers/C008.pdf](https://www.norbert.idv.hk/Files_Papers/C008.pdf)  
15. Drift-Correcting Multiphysics Informed Neural Network Coupled PDE Solver \- ResearchGate, acceso: diciembre 23, 2025, [https://www.researchgate.net/publication/383679932\_Drift-Correcting\_Multiphysics\_Informed\_Neural\_Network\_Coupled\_PDE\_Solver](https://www.researchgate.net/publication/383679932_Drift-Correcting_Multiphysics_Informed_Neural_Network_Coupled_PDE_Solver)  
16. Eddy current \- Wikipedia, acceso: diciembre 23, 2025, [https://en.wikipedia.org/wiki/Eddy\_current](https://en.wikipedia.org/wiki/Eddy_current)  
17. EFFECT OF EDDY CURRENT ON- THE MAGNETIC INDUCTANCE OF- SYNCHRONOUS MACHINE \- DiVA portal, acceso: diciembre 23, 2025, [https://www.diva-portal.org/smash/get/diva2:1330306/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1330306/FULLTEXT01.pdf)  
18. Eddy currents in a solenoid, acceso: diciembre 23, 2025, [https://osiris.df.unipi.it/\~macchi/FISICAB/PROBLEMI/B1/eddycurrents.pdf](https://osiris.df.unipi.it/~macchi/FISICAB/PROBLEMI/B1/eddycurrents.pdf)  
19. Phase Inductance and Rotor Position Estimation for Sensorless Permanent Magnet Synchronous Machine Drives at Standstill \- SciSpace, acceso: diciembre 23, 2025, [https://scispace.com/pdf/phase-inductance-and-rotor-position-estimation-for-1f66znkums.pdf](https://scispace.com/pdf/phase-inductance-and-rotor-position-estimation-for-1f66znkums.pdf)  
20. High-Frequency Voltage-Injection Methods and Observer Design for Initial Position Detection of Permanent Magnet Synchronous Machines \- Aalborg Universitets forskningsportal, acceso: diciembre 23, 2025, [https://vbn.aau.dk/ws/files/280258862/High\_Frequency\_Voltage\_Injection\_Methods\_and\_Observer\_Design\_for\_Initial\_Position\_Detection\_of\_Permanent\_Magnet\_Synchronous\_Machines.pdf](https://vbn.aau.dk/ws/files/280258862/High_Frequency_Voltage_Injection_Methods_and_Observer_Design_for_Initial_Position_Detection_of_Permanent_Magnet_Synchronous_Machines.pdf)  
21. Synchronous Sampling-Based Direct Current Estimation Method for Self-Sensing Active Magnetic Bearings \- MDPI, acceso: diciembre 23, 2025, [https://www.mdpi.com/1424-8220/20/12/3497](https://www.mdpi.com/1424-8220/20/12/3497)  
22. Non-linear observer based control of magnetic levitation systems \- White Rose eTheses Online, acceso: diciembre 23, 2025, [https://etheses.whiterose.ac.uk/id/eprint/20582/1/Non-linear%20observer%20based%20control%20of%20magnetic%20levitation%20systems%20-%20Benomair.pdf](https://etheses.whiterose.ac.uk/id/eprint/20582/1/Non-linear%20observer%20based%20control%20of%20magnetic%20levitation%20systems%20-%20Benomair.pdf)  
23. \[2405.08790\] Kolmogorov-Arnold Networks (KANs) for Time Series Analysis \- arXiv, acceso: diciembre 23, 2025, [https://arxiv.org/abs/2405.08790](https://arxiv.org/abs/2405.08790)  
24. PINN-Obs: Physics-Informed Neural Network-Based Observer for Nonlinear Dynamical Systems \- arXiv, acceso: diciembre 23, 2025, [https://arxiv.org/html/2507.06712v1](https://arxiv.org/html/2507.06712v1)  
25. What Are Physics-Informed Neural Networks (PINNs)? \- MATLAB & Simulink \- MathWorks, acceso: diciembre 23, 2025, [https://www.mathworks.com/discovery/physics-informed-neural-networks.html](https://www.mathworks.com/discovery/physics-informed-neural-networks.html)  
26. Recipes for when physics fails: recovering robust learning of physics informed neural networks \- PMC \- NIH, acceso: diciembre 23, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10481851/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10481851/)  
27. \[R\] Equation requirements for PINNs (Physics-inforemd Neural networks) \- Reddit, acceso: diciembre 23, 2025, [https://www.reddit.com/r/MachineLearning/comments/1e9belt/r\_equation\_requirements\_for\_pinns\_physicsinforemd/](https://www.reddit.com/r/MachineLearning/comments/1e9belt/r_equation_requirements_for_pinns_physicsinforemd/)  
28. (PDF) Unsupervised Physics-Informed Neural Network-based Nonlinear Observer design for autonomous systems using contraction analysis \- ResearchGate, acceso: diciembre 23, 2025, [https://www.researchgate.net/publication/385822967\_Unsupervised\_Physics-Informed\_Neural\_Network-based\_Nonlinear\_Observer\_design\_for\_autonomous\_systems\_using\_contraction\_analysis](https://www.researchgate.net/publication/385822967_Unsupervised_Physics-Informed_Neural_Network-based_Nonlinear_Observer_design_for_autonomous_systems_using_contraction_analysis)  
29. onnxruntime gpu performance 5x worse than pytorch gpu performance \#8166 \- GitHub, acceso: diciembre 23, 2025, [https://github.com/microsoft/onnxruntime/issues/8166](https://github.com/microsoft/onnxruntime/issues/8166)  
30. onnxruntime inference is way slower than pytorch on GPU \- Stack Overflow, acceso: diciembre 23, 2025, [https://stackoverflow.com/questions/70740287/onnxruntime-inference-is-way-slower-than-pytorch-on-gpu](https://stackoverflow.com/questions/70740287/onnxruntime-inference-is-way-slower-than-pytorch-on-gpu)  
31. Fast sigmoid algorithm \- neural network \- Stack Overflow, acceso: diciembre 23, 2025, [https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm](https://stackoverflow.com/questions/10732027/fast-sigmoid-algorithm)  
32. yandex/fastops: This small library enables acceleration of bulk calls of certain math functions on AVX and AVX2 hardware. Currently supported operations are exp, log, sigmoid and tanh. The library is designed with extensibility in mind. \- GitHub, acceso: diciembre 23, 2025, [https://github.com/yandex/fastops](https://github.com/yandex/fastops)  
33. FABE13-HX: High-Performance SIMD Trigonometric Library for Scientific Computing \- GitHub, acceso: diciembre 23, 2025, [https://github.com/farukalpay/FABE](https://github.com/farukalpay/FABE)  
34. Harnessing the Power of SIMD Programming Using AVX | by Nevin Baiju \- Medium, acceso: diciembre 23, 2025, [https://medium.com/@nevinbaiju\_77488/harnessing-the-power-of-simd-programming-using-avx-e57cb14058e7](https://medium.com/@nevinbaiju_77488/harnessing-the-power-of-simd-programming-using-avx-e57cb14058e7)  
35. Improving performance with SIMD intrinsics in three use cases \- The Stack Overflow Blog, acceso: diciembre 23, 2025, [https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/](https://stackoverflow.blog/2020/07/08/improving-performance-with-simd-intrinsics-in-three-use-cases/)
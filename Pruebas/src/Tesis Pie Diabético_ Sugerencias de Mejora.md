# **Auditoría Técnica y Plan de Expansión Doctoral: Sistema de Monitoreo Térmico Plantar Basado en Textiles Inteligentes e IoT**

## **1\. Introducción y Alcance de la Revisión Biomédica**

El presente documento constituye un análisis exhaustivo, crítico y propositivo de la tesis de ingeniería titulada *"Prototipo de calcetín fabricado con textil inteligente para monitoreo de temperatura por medio de IoT para el cuidado del pie diabético"*, presentada por Frida Sofía Andrade Sierra.1 Como experto en ingeniería biomédica y desarrollo de dispositivos médicos, el objetivo de este reporte es proporcionar una hoja de ruta detallada para elevar el nivel académico y técnico de la investigación, transformándola de un prototipo académico funcional a una propuesta de dispositivo médico viable, alineada con el estado del arte científico comprendido entre los años 2020 y 2025\.

La tesis original 1 establece una base sólida al integrar textiles conductores (EeonTex NW170-PI), instrumentación electrónica basada en impedancia (AD5934) y conectividad inalámbrica (ESP32). Sin embargo, la transición hacia una aplicación clínica real exige una comprensión mucho más profunda de la fisiopatología de la diabetes, la física de los materiales conductores heterogéneos y los rigurosos estándares de validación clínica. A lo largo de este reporte, diseccionaremos cada componente del sistema propuesto, identificando brechas críticas en la metodología y ofreciendo soluciones técnicas avanzadas para rehacer o extender los capítulos correspondientes.

La premisa central de esta revisión es que el monitoreo de temperatura plantar no es simplemente un problema de "medición de calor", sino un desafío complejo de **biofísica de interfases**, donde la presión mecánica, la humedad, la termodinámica vascular y la biocompatibilidad juegan roles entrelazados que deben ser desacoplados mediante ingeniería de precisión.

## ---

**2\. Marco Fisiopatológico: Redefiniendo la Termodinámica del Pie Diabético**

### **2.1. Más allá de la "Infección": La Neuropatía Autonómica y la Hemodinámica**

El documento original plantea correctamente que la temperatura es un indicador de inflamación.1 Sin embargo, para una tesis de alto nivel, la justificación médica debe ser mucho más granular. Es imperativo reescribir la sección de antecedentes médicos para incorporar los mecanismos hemodinámicos exactos que justifican la medición.

La elevación de temperatura en el pie diabético antes de la ulceración no se debe únicamente a la inflamación tisular por trauma (teoría del estrés tisular), sino fundamentalmente a la **neuropatía autonómica**. La denervación simpática provoca la pérdida del tono vasomotor, lo que resulta en la apertura de comunicaciones arteriovenosas (shunts). Esto causa un flujo sanguíneo aumentado a la piel (hiperemia) que no es nutricional, sino termogénico. El "pie caliente" neuropático es un predictor temprano de riesgo, mucho antes de que ocurra el trauma mecánico visible.

**Sugerencia de Extensión:** Se debe incorporar una sección titulada "Fisiopatología de la Termorregulación en el Pie Neuropático", donde se discuta:

1. **La Tríada de la Termogénesis Patológica:** Inflamación subclínica (liberación de citocinas), hiperemia neurogénica (shunts AV) y fallo en la disipación de calor por anhidrosis (falta de sudoración).  
2. **Umbrales Clínicos Actualizados (2020-2025):** Mientras que la literatura clásica cita una diferencia de 2.2 °C (4 °F) como umbral de riesgo, estudios recientes (2022-2024) sugieren el uso de algoritmos personalizados que analizan la *variabilidad* longitudinal de la temperatura de un mismo paciente, más que un umbral estático universal. La asimetría térmica contralateral sigue siendo el estándar de oro, pero la tesis debe discutir sus limitaciones en pacientes con amputaciones previas o enfermedad vascular bilateral.

### **2.2. Selección de Sitios Anatómicos y Angiosomas**

La tesis menciona la ubicación de sensores en "puntos estratégicos" como el dedo gordo, talón y cabezas metatarsales.1 Aunque estas son zonas de alta presión, la justificación debe ser anatómica vascular.

**Recomendación de Rehacer:** La justificación de la ubicación de los sensores debe basarse en el concepto de **Angiosomas**. El pie está dividido en regiones tridimensionales de tejido irrigadas por arterias fuente específicas (arteria tibial posterior, arteria peronea, arteria tibial anterior).

* El talón y la planta medial corresponden a la arteria tibial posterior.  
* El borde lateral corresponde a la arteria peronea.  
* El dorso y el primer espacio interdigital corresponden a la arteria tibial anterior.

Al alinear los sensores con estos angiosomas, el dispositivo no solo detecta úlceras por presión, sino que potencialmente puede monitorear la perfusión vascular periférica. Si un sensor en el angiosoma de la tibial posterior muestra una caída brusca de temperatura (pie frío) mientras el resto se mantiene estable, el sistema podría alertar sobre una isquemia crítica, no una úlcera caliente. Esta distinción eleva la propuesta de un "monitor de úlceras" a una "herramienta de vigilancia vascular integral".

| Ubicación del Sensor | Estructura Anatómica | Angiosoma Correspondiente | Justificación Patológica Avanzada |
| :---- | :---- | :---- | :---- |
| Sensor 1 (Dedo Gordo) | Hallux / Falange Distal | A. Tibial Anterior / Plantar Medial | Zona de alto cizallamiento en el despegue (Toe-off). Riesgo isquémico distal. |
| Sensores 2, 3, 4 (Metatarsos) | Cabezas Metatarsales (1ª, 3ª, 5ª) | A. Plantar Medial y Lateral | Puntos de máxima presión vertical y cizalla horizontal. Frecuentes en deformidad de Charcot. |
| Sensor 5 (Arco/Medio pie) | Articulación de Lisfranc | A. Plantar Medial | Zona crítica para colapso del arco (Neuroartropatía de Charcot fase activa). |
| Sensor 6 (Talón) | Calcáneo | A. Calcaneal (Rama Tibial Post.) | Riesgo de úlceras por decúbito (pacientes encamados) y fisuras por xerosis. |

## ---

**3\. Ingeniería de Materiales: Crítica al Sensor Textil y el Problema de la Sensibilidad Cruzada**

### **3.1. La Dualidad Piezorresistiva vs. Termorresistiva**

El documento identifica el uso del textil **EeonTex NW170-PI** 1, un fieltro no tejido recubierto de polímero conductor (probablemente polipirrol o polianilina dopada). La caracterización presentada en la tesis se centra exclusivamente en su respuesta térmica (Coeficiente de Temperatura Negativo o NTC), mostrando que la impedancia baja al aumentar la temperatura.

Identificación de Fallo Crítico (A Rehacer):  
Existe una omisión fundamental en la caracterización: El EeonTex es inherentemente piezorresistivo.  
En la física de polímeros conductores, la resistencia eléctrica depende de las rutas de percolación (caminos conductores) entre las fibras.

1. **Efecto de la Temperatura (NTC):** El calor aporta energía a los electrones para saltar entre las barreras de potencial de las cadenas poliméricas (hopping térmico) y expande el polímero, alterando las conexiones. En este material, el efecto neto reportado es una bajada de resistencia.1  
2. **Efecto de la Presión (Piezorresistencia):** Al pisar el calcetín, las fibras se comprimen. Esto aumenta el número de contactos fibra-fibra, creando más caminos de percolación y **disminuyendo drásticamente la resistencia**.

**El Conflicto:** Tanto el aumento de temperatura (inflamación) como el aumento de presión (caminar) provocan el mismo efecto eléctrico: una caída de la resistencia.

* ¿Cómo sabe el microcontrolador si la resistencia bajó porque el pie está a 38°C (riesgo de úlcera) o porque el paciente pesa 90 kg y está de pie?

Estrategia de Solución Obligatoria:  
La tesis debe extenderse para incluir un mecanismo de Desacoplamiento de Variables.

* *Opción A (Hardware):* Integrar un sensor de presión comercial (FSR) o capacitivo en una capa separada para "gating". El sistema solo mide temperatura cuando la presión es cero (fase de vuelo en la marcha o reposo).  
* *Opción B (Caracterización Diferencial):* Realizar pruebas de laboratorio donde se aplique presión controlada a diferentes temperaturas constantes. Generar una superficie de calibración 3D (Presión, Temperatura, Impedancia) en lugar de una curva 2D.  
* *Opción C (Referencia Pasiva):* Colocar un sensor de temperatura idéntico en una zona del calcetín que no soporte peso (ej. el tobillo o el empeine superior) para medir la temperatura basal y restar el ruido sistémico, aunque esto no corrige el efecto local de la presión plantar.

### **3.2. Geometría del Sensor y Campos Eléctricos**

La tesis reporta un hallazgo empírico valioso: los sensores cuadrados mostraban variabilidad según su orientación, mientras que los circulares eran más estables.1  
Extensión Teórica: Se debe formalizar este hallazgo mediante la teoría electromagnética. En un material anisotrópico (como un textil no tejido donde las fibras tienen una dirección preferencial de cardado), la densidad de corriente no es uniforme. En las esquinas de un cuadrado, se produce una acumulación de líneas de campo eléctrico (efecto de borde), lo que hace que la medición sea muy sensible a la alineación de los hilos conductores. La geometría circular homogeneiza el campo radialmente, promediando las anisotropías del material. Explicar esto con diagramas de líneas equipotenciales elevará el nivel científico del reporte.

## ---

**4\. Instrumentación Electrónica y Espectroscopía de Impedancia**

### **4.1. Análisis de Frecuencia: La Física detrás de los 90 kHz**

El reporte indica que se probó medir a 10 Hz, 50 kHz y 90 kHz, seleccionando 90 kHz por tener el menor error.1  
Necesidad de Profundización: Es insuficiente decir "dio menos error". Se debe explicar por qué.  
La interfaz entre un conductor iónico (piel/sudor) y un conductor electrónico (textil/metal) o dentro del propio material heterogéneo genera una Doble Capa Eléctrica que actúa como un capacitor ($C\_{dl}$). La impedancia total ($Z$) es la suma de la resistencia del material ($R\_{sensor}$) y la reactancia de este capacitor ($X\_c \= \\frac{1}{2\\pi f C}$).

* **A 10 Hz:** La frecuencia ($f$) es baja, por lo que la reactancia capacitiva ($X\_c$) es enorme. La medición está dominada por la capacitancia de la interfaz, no por la resistencia del sensor. Cualquier pequeño cambio en el contacto (movimiento) altera drásticamente la lectura. Esto explica el alto error reportado.1  
* A 90 kHz: La frecuencia es alta, lo que hace que la reactancia capacitiva sea muy pequeña (el capacitor se comporta casi como un cortocircuito para la señal AC). Lo que queda dominante es la parte real (resistiva) de la impedancia, que es la que varía con la temperatura.  
  Sugerencia: Incluir un diagrama de circuito equivalente (Modelo de Randles modificado) que represente el sensor textil, explicando matemáticamente cómo la alta frecuencia permite "bypass" a los capacitores parásitos.

### **4.2. Calibración y Deriva Temporal (Drift)**

Los polímeros conductores sufren de histéresis y envejecimiento. Después de múltiples ciclos de calentamiento/enfriamiento, la resistencia base puede cambiar (drift).  
Recomendación de Extensión: Proponer un protocolo de "Autocalibración".

* El sistema podría usar la temperatura ambiente (medida por el termistor interno del microcontrolador al encenderse, asumiendo que el calcetín no está puesto aún) para ajustar la línea base del modelo matemático cada día.

## ---

**5\. Diseño del Sistema IoT y Gestión Energética**

### **5.1. Microcontrolador y Eficiencia**

El uso del ESP32 es adecuado para prototipado rápido debido a su WiFi/BLE nativo.1 Sin embargo, su consumo energético es excesivo para un wearable comercial (aprox. 80-240 mA en transmisión).  
Sugerencia de Revisión (2020-2025): En el estado del arte actual, se debe discutir la migración hacia SoCs (System on Chip) específicos para wearables, como la serie nRF52 o nRF53 de Nordic Semiconductor, o el ESP32-C3/S3 (RISC-V) que tienen mejor gestión de energía.  
Si se mantiene el ESP32, se debe detallar la estrategia de Duty Cycling:

* El pie diabético es un fenómeno lento. No es necesario medir cada 10 segundos 1 y transmitir.  
* *Estrategia Propuesta:* Despertar cada 5 minutos, tomar medición rápida (\< 1 seg), almacenar en buffer. Transmitir por BLE al teléfono solo una vez cada hora o inmediatamente si se detecta una anomalía crítica. Esto extendería la batería de horas a semanas.

### **5.2. Seguridad y Privacidad de Datos (IoMT)**

La tesis menciona una aplicación móvil y envío de datos.1 Al tratar datos médicos, se entra en el ámbito del Internet of Medical Things (IoMT).  
Requisito Faltante: Es obligatorio mencionar la seguridad. El protocolo BLE estándar transmite en texto plano si no se configura. Se debe proponer:

1. **Emparejamiento Seguro (LE Secure Connections).**  
2. **Encriptación de la carga útil (AES-128).**  
3. **Anonimización:** El calcetín no debe transmitir el nombre del paciente, sino un UUID que solo la App vinculada puede resolver.

## ---

**6\. Manufacturabilidad, Lavabilidad y Experiencia de Usuario (UX)**

### **6.1. El Desafío de la Lavabilidad**

La conclusión de la tesis menciona "Hacer pruebas de impedancia lavable" como trabajo futuro.1 Esto es inaceptable para una tesis de grado avanzado; debe ser parte del diseño.  
Extensión Técnica: Discutir técnicas de protección.

* **Encapsulamiento:** Uso de películas de TPU (Poliuretano Termoplástico) laminadas sobre el sensor EeonTex. El TPU es impermeable pero permite la transferencia térmica.  
* **Conectores Removibles:** La electrónica (ESP32 \+ batería) *no* puede lavarse. Se debe diseñar un módulo "pod" o "pastilla" que se desconecte del calcetín mediante broches magnéticos o de presión conductivos antes de meter la prenda a la lavadora.

### **6.2. Diseño Centrado en el Paciente Geriátrico**

El usuario final típico es un adulto mayor, posiblemente con retinopatía (visión limitada) y movilidad reducida.  
Crítica a la UI: Las gráficas de impedancia mostradas en la tesis 1 son útiles para el ingeniero, pero inútiles para el paciente.  
Rehacer Sección de App: Diseñar una interfaz basada en Semáforos.

* Verde: "Pies sanos".  
* Amarillo: "Diferencia térmica detectada. Revise su calzado".  
* Rojo: "Alerta de riesgo. Contacte a su podólogo".  
* Incluir notificaciones por voz (Text-to-Speech) para pacientes con discapacidad visual.

## ---

**7\. Plan de Trabajo Experimental Extendido (Protocolo de Validación)**

Para validar las mejoras propuestas, se sugiere reestructurar la metodología experimental en tres fases claras:

### **Fase 1: Caracterización Electromecánica (Laboratorio)**

* **Objetivo:** Desacoplar Presión y Temperatura.  
* **Setup:** Montar el sensor en una máquina universal de ensayos (Instron) dentro de una cámara climática.  
* **Protocolo:** Aplicar barridos de presión (0 a 200 kPa) manteniendo temperaturas fijas (25, 30, 35, 40°C).  
* **Resultado Esperado:** Una matriz de calibración $Z \= f(T, P)$ que permita al algoritmo compensar el efecto del peso del paciente.

### **Fase 2: Pruebas de Lavabilidad (Estándar AATCC)**

* **Objetivo:** Determinar la vida útil del sensor.  
* **Protocolo:** Realizar 10, 20, 30 ciclos de lavado según norma AATCC 135\. Medir la resistencia base ($R\_0$) después de cada ciclo.  
* **Criterio de Éxito:** Variación de $R\_0 \< 10\\%$ tras 20 lavados.

### **Fase 3: Estudio Clínico Piloto (Sujetos Sanos \+ Simulación)**

* **Objetivo:** Validar la usabilidad y la estabilidad de la señal en movimiento.  
* **Protocolo:**  
  1. Sujeto en reposo (sentado) vs. caminando (marcha).  
  2. Inducción de asimetría térmica artificial: Aplicar un parche caliente (o bolsa de gel) en un pie y verificar si el calcetín detecta el diferencial $\> 2.2^\\circ C$ respecto al pie contralateral.  
  3. Evaluación de confort mediante encuesta (Escala Likert) enfocada en la sensación de los hilos y costuras (riesgo de causar ampollas).

## ---

**8\. Análisis Comparativo del Estado del Arte (2020-2025)**

A continuación, se presenta una tabla que contrasta la aproximación de la tesis original con las tendencias actuales de la investigación global, proporcionando referencias conceptuales para la actualización bibliográfica.

| Característica | Enfoque Tesis Original (2018-2019) | Enfoque Estado del Arte (2020-2025) | Referencias Conceptuales / Tendencias |
| :---- | :---- | :---- | :---- |
| **Material del Sensor** | Textil Conductivo Comercial (EeonTex) | Compuestos Nano-Híbridos (Grafeno, MXenos, Nanotubos de Carbono sobre TPU) | Mayor sensibilidad, flexibilidad extrema y mejor resistencia al lavado. (Investigaciones de *Donghua University*, *Georgia Tech*). |
| **Integración Electrónica** | Módulo Rígido (ESP32 dev board) \+ Cables | Electrónica Flexible Híbrida (FHE), Circuitos Estirables, Comunicación NFC pasiva | Eliminación de baterías voluminosas usando *Energy Harvesting* (piezoeléctrico/triboeléctrico) o lectura pasiva vía NFC. |
| **Algoritmo de Detección** | Umbral simple de temperatura absoluta o relativa | Machine Learning (TinyML) en el borde | Redes neuronales ligeras en el microcontrolador que aprenden el patrón térmico *normal* del usuario y detectan anomalías sutiles (Detección de Anomalías). |
| **Validación Clínica** | Temperatura Absoluta | Termografía Infrarroja y Asimetría Contralateral | Integración de datos de actividad física para correlacionar "Dosis de Pasos" con "Dosis Térmica". |
| **Fabricación** | Costura manual de sensores | Tejido 3D (Knitting) sin costuras | Sensores integrados en la estructura misma del tejido (Intarsia knitting) para eliminar puntos de fricción que causan úlceras. |

## ---

**9\. Resumen de Recomendaciones Críticas**

Para finalizar la revisión de experto, se condensan las acciones requeridas en una lista priorizada de "To-Do" para la reescritura de la tesis:

1. **Fundamental:** Diseñar e incluir el experimento de **Presión vs. Temperatura**. Sin esto, el dispositivo no es confiable para uso en bipedestación.  
2. **Teórico:** Reescribir el marco teórico de la medición de impedancia explicando la física de la selección de frecuencia (90 kHz) basada en la reactancia capacitiva.  
3. **Clínico:** Adoptar el modelo de **Angiosomas** para la ubicación de sensores y discutir la fisiología de los shunts AV.  
4. **Ingeniería:** Desarrollar un apartado sobre la **mecánica de contacto** entre el hilo de acero y el textil suave (resistencia de contacto inestable).  
5. **Software:** Proponer un algoritmo de **compensación** que use datos de un acelerómetro (ya presente en muchos ESP32 o módulos externos) para invalidar lecturas de temperatura tomadas durante el impacto del talón (fase de carga).

La implementación de estas sugerencias transformará el trabajo de un ejercicio de integración de componentes a una investigación de ingeniería biomédica robusta, con potencial de publicación en revistas de alto impacto (Q1/Q2) y transferencia tecnológica real.

---

*(Fin del Resumen Ejecutivo de la Revisión. A continuación, se desarrolla el cuerpo completo del reporte técnico con la extensión solicitada.)*

# **Capítulo 1: Fundamentos Fisiológicos Avanzados del Pie Diabético**

## **1.1 La Cascada de Eventos en la Ulceración**

El pie diabético no es una entidad estática, sino la consecuencia final de una cascada de eventos fisiopatológicos. La tesis debe describir este proceso para justificar la ventana de oportunidad de intervención.

1. **Fase Hiperglucémica:** El exceso de glucosa genera productos finales de glicación avanzada (AGEs) que dañan el endotelio vascular y las vainas de mielina nerviosa.  
2. **Fase Neuropática:**  
   * *Sensitiva:* Pérdida de la sensación protectora (LOPS). El paciente no siente el trauma repetitivo de caminar.  
   * *Motora:* Atrofia de músculos intrínsecos, causando deformidades (dedos en garra) que crean nuevos puntos de presión.  
   * *Autonómica:* La clave del monitoreo térmico. La pérdida de sudoración (piel seca/grietas) y la vasodilatación no regulada.  
3. **Fase Pre-Ulcerativa (Inflamatoria):** Aquí es donde actúa el calcetín inteligente. El tejido sometido a estrés mecánico excesivo sufre micro-traumas. El cuerpo responde con inflamación aguda (rubor, calor, tumor). En un pie normal, el dolor limitaría la actividad. En el pie diabético, la actividad continúa, perpetuando el ciclo de inflamación \-\> necrosis \-\> úlcera.

La detección de un diferencial térmico persistente permite intervenir en esta fase pre-ulcerativa, recomendando reposo ("off-loading") para permitir que la inflamación subyacente sane antes de que la piel se rompa.

## **1.2 Algoritmos de Predicción de Riesgo**

La literatura moderna (Armstrong et al., Lavery et al.) ha refinado el criterio de "High Risk Foot". La tesis debe discutir cómo el sistema IoT puede categorizar al paciente en tiempo real:

* **Riesgo Bajo:** Diferencial \< 1.0°C. Monitoreo diario.  
* **Riesgo Moderado:** Diferencial 1.0°C \- 2.2°C. Alerta amarilla. Reducir pasos diarios en un 50%.  
* **Riesgo Alto:** Diferencial \> 2.2°C en dos días consecutivos. Alerta roja. Cese de actividad y visita médica.

# ---

**Capítulo 2: Actualización Tecnológica y Estado del Arte (2020-2025)**

## **2.1 Evolución de los E-Textiles**

Mientras que el EeonTex es un material excelente para prototipado, la investigación actual explora límites más allá:

* **Tintas de Grafeno Elásticas:** Investigadores han desarrollado tintas conductoras basadas en nanoplaquetas de grafeno que pueden ser estiradas hasta un 100% sin perder conductividad. Esto resuelve el problema de la rigidez de los sensores metálicos.  
* **Fibras Líquidas:** Uso de canales microfluídicos llenos de metal líquido (Galinstan) dentro de fibras de silicona, permitiendo una durabilidad mecánica infinita frente al lavado.

## **2.2 Integración de IoT Moderno**

El protocolo **Matter** y **Thread** está revolucionando el IoT doméstico. Un dispositivo médico moderno debería ser capaz de integrarse en el ecosistema de "Smart Home" del paciente. Imaginar que el calcetín no solo envíe datos al celular, sino que pueda encender una luz roja en el espejo del baño (vía hub domótico) para alertar visualmente al paciente anciano, es una visión de ingeniería de sistemas de 2025 que debe ser mencionada.

# ---

**Capítulo 3: Profundización en la Caracterización del Sensor**

## **3.1 Fenomenología del Transporte de Carga en Polímeros Conductores**

Para elevar el nivel académico, se debe explicar el mecanismo de conducción en el EeonTex. Se basa en el Hopping de Rango Variable (VRH). Los electrones no fluyen libremente como en un metal; "saltan" entre islas conductoras en la matriz polimérica aislante.  
La ecuación que rige la resistencia ($R$) en función de la temperatura ($T$) en semiconductores intrínsecos y polímeros suele seguir una forma tipo Arrhenius o modelos de Mott:

$$R(T) \= R\_0 \\exp\\left$$

Donde $\\gamma$ depende de la dimensionalidad del sistema. Ajustar los datos experimentales de la tesis a este modelo físico (en lugar de una simple regresión polinómica) demostraría un entendimiento profundo de la ciencia de materiales.

## **3.2 El Experimento Crucial: Histéresis Térmica**

Un sensor ideal sigue el mismo camino de resistencia al calentarse que al enfriarse. Los sensores textiles reales tienen histéresis.

* **Sugerencia de Gráfica:** Se debe generar una gráfica de "Lazo de Histéresis": Eje X (Temperatura), Eje Y (Impedancia). Trazar la curva de subida (25-\>40°C) y la de bajada (40-\>25°C). El área dentro del lazo representa el error del sensor debido a la memoria térmica del material. Cuantificar este error es vital para definir la precisión real del dispositivo (±0.5°C o ±2.0°C?).

# ---

**Capítulo 4: Electrónica y Procesamiento de Señales Avanzado**

## **4.1 Diseño de PCB para Wearables**

La tesis menciona una tarjeta de desarrollo y protoboard.1 Para la versión final ("Qué rehacer"), se debe diseñar una PCB (Printed Circuit Board).

* **Requisitos de la PCB:**  
  * Sustrato flexible (Flex-PCB) en poliimida (Kapton) para adaptarse a la curvatura del tobillo.  
  * Uso de componentes SMD (Surface Mount Device) de tamaño 0402 o 0201 para minimizar peso.  
  * Planos de tierra mallados (hatched ground planes) para permitir flexibilidad sin fractura del cobre.

## **4.2 Filtrado Digital**

La señal de un sensor en un pie en movimiento es extremadamente ruidosa.

* **Sugerencia:** Implementar un filtro digital en el firmware del ESP32. Un simple promedio móvil es lento. Se recomienda un **Filtro de Kalman Unidimensional**. El filtro de Kalman predice el estado futuro (temperatura) basándose en el estado previo y corrige la predicción con la nueva medición, ponderando la incertidumbre del sensor. Esto es muy efectivo para suavizar las lecturas ruidosas de sensores textiles sin introducir el retardo (lag) excesivo que tienen los filtros paso-bajo convencionales.

# ---

**Capítulo 5: Conclusiones y Hoja de Ruta de Desarrollo**

La revisión concluye que el trabajo de Andrade Sierra 1 es una prueba de concepto válida pero incompleta para su despliegue clínico. El camino a seguir requiere una colaboración multidisciplinaria:

1. **Con Expertos en Materiales:** Para estabilizar el sensor frente a la presión y el lavado.  
2. **Con Clínicos (Podólogos/Endocrinólogos):** Para validar la relevancia de los puntos de medición y los algoritmos de alerta.  
3. **Con Ingenieros de Software:** Para asegurar los datos y crear interfaces accesibles.

El potencial de impacto es alto, dada la prevalencia de la diabetes en México. Si se resuelven los problemas de ingeniería fundamental detallados en este reporte (especialmente la sensibilidad cruzada a la presión y la lavabilidad), este prototipo podría convertirse en una herramienta estándar en la prevención de amputaciones.

#### **Fuentes citadas**

1. t\_word\_Frida\_Andrade.docx
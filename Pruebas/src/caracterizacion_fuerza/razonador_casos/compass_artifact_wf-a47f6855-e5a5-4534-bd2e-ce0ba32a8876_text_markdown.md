# Defensa metodológica doctoral de un razonador CBR para predicción de histéresis en fresado

## TL;DR
- El paradigma CBR (recuperación por vecino más cercano), la normalización robusta mediana/IQR, la distancia euclídea ponderada y el rechazo por novedad SÍ tienen respaldo peer-reviewed verificable (Aamodt & Plaza 1994; López de Mántaras et al. 2005; Wettschereck, Aha & Mohri 1997; Perner 2008), y la física del área del lazo como energía disipada por ciclo está sólidamente fundamentada (Charalampakis & Koumousis 2008).
- Los VALORES NUMÉRICOS concretos de los pesos (1.0/0.8/0.7/1.4/0.3) y de los umbrales (0.3/1.0 y el umbral de rechazo) NO tienen respaldo directo en la literatura: deben defenderse mediante análisis de sensibilidad, validación cruzada y criterio físico, no forzando una cita.
- La restricción de pocos casos (~19) favorece a CBR sobre clasificadores entrenados, pero exige declarar honestamente el riesgo de sobreajuste de pesos/umbrales y validar con LOOCV; el despliegue embebido (kNN) es factible y tiene antecedentes (Behera & Prathuri 2024; CBR online en diagnóstico).

## Key Findings
1. **CBR es defendible con n≈19**: la literatura fundacional y aplicada respalda CBR cuando hay pocos datos, se requiere interpretabilidad y aprendizaje incremental.
2. **El área del lazo merece el mayor peso por física, no por capricho**: el área de la curva fuerza–desplazamiento ES la energía disipada por ciclo; en Bouc–Wen esto se demuestra analíticamente (Charalampakis & Koumousis 2008).
3. **Mediana/IQR vs z-score está bien justificado** para señales ruidosas con outliers y muestras pequeñas, aunque el respaldo es por documentación canónica más que por un artículo peer-reviewed con DOI específico.
4. **Los números concretos son el flanco débil**: hay que blindarlos con análisis de sensibilidad y validación cruzada (LOOCV), no con citas inexistentes.

## Details

### Tabla resumen

| Decisión | Justificación (1 línea) | Fuentes (citas cortas) | Nivel de evidencia |
|---|---|---|---|
| 1. CBR (vecino más cercano) vs clasificador entrenado | Pocos datos, interpretabilidad y aprendizaje incremental favorecen razonamiento por casos | Aamodt & Plaza (1994); López de Mántaras et al. (2005); Kolodner (1993); Dahmoune et al. (2025) | Directa (CBR-TCM) + Analógica |
| 2. Vector de 7 features con pesos físicos | RMS/pico de fuerza, acoplamiento y área del lazo son descriptores físicamente válidos de desgaste/histéresis | Salgado & Alonso (2007); Kaya et al. (2011); Charalampakis & Koumousis (2008) | Directa (features) / Débil (valores de peso) |
| 3. Normalización robusta (x−mediana)/IQR | Resistente a outliers en señales ruidosas y muestras pequeñas | scikit-learn RobustScaler (doc. canónica); estadística robusta | Analógica |
| 4. Distancia euclídea ponderada | Métrica estándar y sensible al escalado/peso en aprendizaje perezoso | Wettschereck, Aha & Mohri (1997) | Directa (CBR/kNN) |
| 5. Similitud score = 1/(1+dist) | Transformación monótona estándar distancia→similitud | López de Mántaras et al. (2005) | Analógica / Débil |
| 6. Etiqueta por umbral de área del lazo | El área crece con la energía disipada/no linealidad | Charalampakis & Koumousis (2008); Ismail et al. (2009) | Directa (física) / Débil (cortes 0.3/1.0) |
| 7. Umbral de rechazo (novedad/OOD) | Rechazar casos fuera de distribución preserva competencia del CBR | Perner (2008); Szczepaniak (2018) | Analógica |

---

### Fundamentos de CBR: el ciclo 4R y su ajuste al problema

El razonamiento basado en casos resuelve un problema nuevo recuperando y adaptando soluciones de problemas previos similares. El artículo seminal de **Aamodt & Plaza (1994)** define el ciclo de las "4R": **Retrieve** (recuperar el caso más similar), **Reuse** (reutilizar su solución), **Revise** (revisarla/validarla) y **Retain** (retener el caso revisado en memoria). Este modelo es el marco canónico citado por toda la comunidad y verificado: *AI Communications*, vol. 7(1), pp. 39–59, DOI 10.3233/AIC-1994-7104.

La revisión de consenso de **López de Mántaras et al. (2005)** (*Knowledge Engineering Review* 20(3):215–240, DOI 10.1017/S0269888906000646), firmada por trece líderes del campo (incluido Aamodt), profundiza en recuperación, reutilización, revisión y retención y confirma explícitamente que el ciclo 4R de Aamodt & Plaza sigue siendo el esqueleto de la metodología.

**Ajuste al problema doctoral**:
- **Pocos datos (~19 cortes)**: CBR es un aprendiz "perezoso" (lazy) que no requiere entrenar un modelo paramétrico; cada caso es directamente utilizable. La memoria de cortes ES el modelo.
- **Interpretabilidad**: el diagnóstico es explicable citando el caso recuperado y sus descriptores; ventaja reconocida de CBR frente a modelos de caja negra.
- **Aprendizaje incremental**: la fase Retain permite añadir cada nuevo corte validado sin reentrenar, alineándose con HIL y operación en línea.

Fuente fundacional de libro: **Kolodner (1993)**, *Case-Based Reasoning*, Morgan Kaufmann, ISBN 1-55860-237-2 (texto de referencia; sin DOI, es libro).

---

### Decisión 1 — Paradigma CBR en lugar de clasificador entrenado

**(a) Justificación técnica.** Con ~19 cortes, sin etiqueta directa de desgaste y con señales ruidosas, un clasificador entrenado (red neuronal, SVM, random forest) tiende a sobreajustar y es difícil de validar estadísticamente. CBR no estima parámetros globales: almacena casos y razona por similitud local, lo que es robusto con muestras pequeñas y permite interpretabilidad y aprendizaje incremental.

**(b) Referencias.**
- Aamodt & Plaza (1994), *AI Communications* 7(1):39–59, DOI 10.3233/AIC-1994-7104.
- López de Mántaras et al. (2005), *Knowledge Engineering Review* 20(3):215–240, DOI 10.1017/S0269888906000646.
- Dahmoune, Meddour, Elbah, Yallese & Belhadi (2025), "Development of an adaptive tool condition monitoring system: integration of case-based reasoning with CNN", *Journal of Intelligent Manufacturing*, DOI 10.1007/s10845-025-02566-9.
- "An improved case based reasoning method… toward intelligent machining" (2020), *Journal of Intelligent Manufacturing*, DOI 10.1007/s10845-020-01573-2.

**(c) Nivel de evidencia: Directa** (CBR aplicado a monitorización de condición de herramienta / mecanizado) reforzada con la fundacional.

**(d) Alternativas y por qué CBR es defendible.** Clasificadores entrenados (NN/SVM/RF) requieren muchos datos etiquetados y son cajas negras; con n≈19 su varianza es alta. CBR es defendible porque: (i) es naturalmente "few-shot"; (ii) es transparente; (iii) Dahmoune et al. (2025) muestran que CBR es competitivo (incluso combinado con CNN) en TCM real.

**(e) Riesgos y respuesta al tribunal.** Riesgo: CBR no generaliza si la memoria no cubre el espacio. Respuesta: cuantificar competencia/cobertura, usar validación leave-one-out (LOOCV) sobre los 19 casos, y declarar el rechazo por novedad (Decisión 7) como salvaguarda explícita.

---

### Decisión 2 — Vector de 7 features con pesos físicos

**(a) Justificación técnica.** Las features elegidas tienen significado físico en mecanizado: el **RMS de fuerza** es un indicador clásico de nivel de corte/desgaste; el **pico de fuerza** captura el impacto por diente; el **RMS/pico de la señal de entrada** caracteriza la excitación; la **correlación fuerza–entrada** captura el desfase (lag) que es firma de histéresis; el **área del lazo normalizada** mide energía disipada por ciclo (descriptor central, ver sección de física); la **duración** contextualiza la ventana.

**(b) Referencias.**
- Salgado & Alonso (2007), "An approach based on current and sound signals for in-process tool wear monitoring", *International Journal of Machine Tools and Manufacture* 47:2140–2152.
- Kaya, Oysu & Ertunc (2011), "Force–torque based on-line tool wear estimation system for CNC milling of Inconel 718 using neural networks", *Advances in Engineering Software* 42:76–84.
- Para el área del lazo: Charalampakis & Koumousis (2008), *Journal of Sound and Vibration* 309:887–895, DOI 10.1016/j.jsv.2007.07.080.

**(c) Nivel de evidencia: Directa** para la elección de descriptores de fuerza (RMS/pico) en TCM; **Débil** para los valores concretos de los pesos.

**(d) Alternativas.** Features en frecuencia/tiempo-frecuencia (FFT, wavelet, Hilbert–Huang) o coeficientes de fuerza de corte. Defensa: el conjunto tiempo-dominio es interpretable, barato de computar en embebido (Hailo-8L) y físicamente trazable; las alternativas espectrales pueden añadirse después sin romper el marco CBR.

**(e) Riesgos y respuesta.** Riesgo principal: **los valores de peso (1.0/0.8/0.7/1.4/0.3) carecen de respaldo bibliográfico directo.** Respuesta honesta al tribunal: declararlos como hiperparámetros de diseño justificados por (i) criterio físico (mayor peso al descriptor energético) y (ii) un análisis de sensibilidad + validación cruzada que demuestre que el ranking de recuperación es estable frente a perturbaciones de los pesos. NO se debe forzar una cita inexistente.

---

### Decisión 3 — Normalización robusta (x − mediana)/IQR

**(a) Justificación técnica.** Con señales ruidosas, outliers y n pequeño, la media y la desviación estándar se contaminan (un solo outlier infla σ y comprime el resto). La mediana y el IQR son estimadores robustos: no se ven arrastrados por valores extremos, preservando la separación relativa de los inliers. Esto es crítico porque la distancia euclídea (Decisión 4) es muy sensible al escalado.

**(b) Referencias.**
- Documentación de `sklearn.preprocessing.RobustScaler` (scikit-learn), que implementa exactamente (x − mediana)/IQR y motiva su uso "porque los outliers influyen negativamente en media/varianza": https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
- Base conceptual en estadística robusta (mediana/IQR como estimadores resistentes). **Nota de honestidad: una referencia peer-reviewed específica y verificada con DOI para "robust scaling con mediana/IQR" no fue localizada en esta búsqueda — sin respaldo localizado con DOI primario; se apoya en documentación canónica de scikit-learn.**

**(c) Nivel de evidencia: Analógica** (preprocesamiento ML general, no específico de fresado).

**(d) Alternativas.** Z-score (media/σ) y min–max. Defensa: ambas son frágiles ante outliers; min–max además fija el rango al extremo. Mediana/IQR es la opción estándar recomendada cuando hay outliers.

**(e) Riesgos y respuesta.** Riesgo: con n≈19, los cuartiles se estiman con incertidumbre. Respuesta: reportar IC de los cuartiles, considerar IQR con rango de cuantiles ajustado, y validar que el escalado no degrada el LOOCV.

---

### Decisión 4 — Distancia euclídea ponderada

**(a) Justificación técnica.** dist = ‖(q − c)·w‖ / √D normaliza por la dimensión (D=7) y pondera por relevancia física. La euclídea es la métrica más común en kNN/CBR y su rendimiento depende fuertemente de la definición de la función de distancia y de los pesos.

**(b) Referencias.**
- Wettschereck, Aha & Mohri (1997), "A Review and Empirical Evaluation of Feature Weighting Methods for a Class of Lazy Learning Algorithms", *Artificial Intelligence Review* 11:273–314, DOI 10.1023/A:1006593614256.
- López de Mántaras et al. (2005), DOI 10.1017/S0269888906000646 (recuperación en CBR).

**(c) Nivel de evidencia: Directa** (ponderación de features en aprendizaje perezoso/CBR).

**(d) Alternativas.** Distancia de Mahalanobis (capturaría correlaciones entre features) o métricas aprendidas. Defensa: Mahalanobis requiere estimar una matriz de covarianza 7×7 fiable, inviable con n≈19; la euclídea ponderada es parsimoniosa y estable.

**(e) Riesgos y respuesta.** Wettschereck et al. (1997) muestran que los métodos que fijan pesos por *feedback* de rendimiento suelen superar a los pesos fijos. Respuesta: presentar los pesos físicos como prior, y mostrar (sensibilidad/CV) que un esquema aprendido no mejora significativamente con tan pocos datos — o adoptarlo si lo hace.

---

### Decisión 5 — Similitud score = 1/(1+dist)

**(a) Justificación técnica.** Es una transformación monótona decreciente estándar que mapea dist∈[0,∞) a score∈(0,1], con score=1 para distancia nula. Conserva el orden de recuperación.

**(b) Referencias.** López de Mántaras et al. (2005), DOI 10.1017/S0269888906000646 (medidas de similitud en recuperación CBR).

**(c) Nivel de evidencia: Débil/Analógica** — es una convención de ingeniería; la forma exacta 1/(1+dist) no es un resultado teórico con cita única.

**(d) Alternativas.** exp(−dist), 1−dist/dist_max, kernels gaussianos. Defensa: 1/(1+dist) está acotada, no requiere dist_max global (robusto en incremental) y es barata en embebido.

**(e) Riesgos y respuesta.** Riesgo: la elección afecta calibración de confianza. Respuesta: como la decisión final usa orden (vecino más cercano) y un umbral, la forma de similitud no cambia el ranking; declararlo así desactiva la objeción.

---

### Decisión 6 — Etiqueta de histéresis por umbral del área del lazo

**(a) Justificación técnica.** El área del lazo normalizada crece monótonamente con la energía disipada y la no linealidad histerética; por ello segmentar en lineal/moderado/marcado por el área tiene sentido físico.

**(b) Referencias.**
- Charalampakis & Koumousis (2008), *Journal of Sound and Vibration* 309:887–895, DOI 10.1016/j.jsv.2007.07.080.
- Ismail, Ikhouane & Rodellar (2009), "The Hysteresis Bouc-Wen Model, a Survey", *Archives of Computational Methods in Engineering* 16(2):161–188, DOI 10.1007/s11831-009-9031-8.

**(c) Nivel de evidencia: Directa** para la física (área↔energía); **Débil** para los cortes numéricos 0.3 y 1.0.

**(d) Alternativas.** Umbrales por cuantiles de los datos, clustering no supervisado (k-means/GMM) sobre el área, o ajuste a un modelo (Bouc–Wen) y umbral sobre un parámetro. Defensa: con n≈19 los umbrales fijos interpretables son preferibles a clustering inestable, pero deben calibrarse físicamente.

**(e) Riesgos y respuesta.** **Los cortes 0.3/1.0 no tienen respaldo bibliográfico.** Respuesta: derivarlos de los datos (p. ej. terciles del área en la memoria) o de un criterio físico (fracción de energía elástica vs disipada), y mostrar robustez de la clasificación ante variaciones ±20% de los cortes.

---

### Decisión 7 — Umbral de rechazo (novedad / OOD)

**(a) Justificación técnica.** Si la mejor distancia ≥ umbral, el caso se rechaza por estar fuera de la distribución cubierta por la memoria. Esto evita diagnósticos no fiables por extrapolación y es la salvaguarda natural de un CBR con pocos casos.

**(b) Referencias.**
- Perner (2008), "Concepts for novelty detection and handling based on a case-based reasoning process scheme", *Engineering Applications of Artificial Intelligence* (versión journal de ScienceDirect S095219760800105X); versión LNCS DOI 10.1007/978-3-540-73435-2_3.
- Szczepaniak (2018), "Case-Based Reasoning: The Search for Similar Solutions and Identification of Outliers", *Complexity*, DOI 10.1155/2018/9280787.

**(c) Nivel de evidencia: Analógica** (novedad/outlier en CBR, no en fresado).

**(d) Alternativas.** One-class SVM, Local Outlier Factor, umbral sobre densidad. Defensa: estos requieren más datos; un umbral de distancia dentro del propio CBR es coherente, barato y explicable.

**(e) Riesgos y respuesta.** **El valor del umbral no tiene cita.** Respuesta: fijarlo por la distribución de distancias intra-memoria (p. ej. percentil de las distancias LOOCV de los aciertos) y reportar curva de rechazo vs error.

---

### Física del descriptor central: por qué el área del lazo merece el peso más alto (1.4)

En un sistema mecánico sometido a carga cíclica, el **área encerrada por el lazo de histéresis en el plano fuerza (restauradora) – desplazamiento es exactamente la energía disipada por ciclo** debido a la fricción interna/amortiguamiento. Este es un resultado físico estándar, no una heurística:

- **Charalampakis & Koumousis (2008)** derivan analíticamente la energía disipada por el modelo Bouc–Wen bajo excitación cíclica, identificándola con el área del lazo (*Journal of Sound and Vibration* 309:887–895, DOI 10.1016/j.jsv.2007.07.080).
- **Ruderman (2023)**, "Energy dissipation and hysteresis cycles in pre-sliding transients of kinetic friction", *Applications of Mathematics* 68(6):845–860, DOI 10.21136/AM.2023.0283-22, muestra cómo el mapa fuerza–desplazamiento entra en el balance de energía, tratando el área del lazo como energía disipada por ciclo.
- **Ismail, Ikhouane & Rodellar (2009)** (survey Bouc–Wen, DOI 10.1007/s11831-009-9031-8) confirman que la energía disipada/área es la medida de daño acumulado, base de la identificación de modelos histeréticos.

**Conexión con Bouc–Wen / Duhem / LuGre.** El área del lazo es el observable macroscópico que estos modelos reproducen: en Bouc–Wen la energía disipada acumulada gobierna la degradación (Baber–Wen–Noori); los modelos de fricción Dahl, LuGre y Maxwell-slip se reformulan como modelos de Duhem y su lazo fuerza–desplazamiento codifica la disipación: **Oh & Bernstein (2005)**, "Semilinear Duhem model for rate-independent and rate-dependent hysteresis", *IEEE Transactions on Automatic Control* 50(5):631–645, DOI 10.1109/TAC.2005.847035; y **Padthe, Drincic, Oh, Rizos, Fassois & Bernstein (2008)**, "Duhem modeling of friction-induced hysteresis", *IEEE Control Systems Magazine* 28(5):90–107, DOI 10.1109/MCS.2008.927331. En mecanizado, la fricción en el flanco/cara de la herramienta es fuente de histéresis y desfase fuerza–desplazamiento: **Huerta et al. (2017)**, *Shock and Vibration*, DOI 10.1155/2017/5956425.

**Por qué el peso más alto.** Dado que el área es el descriptor más directamente ligado al fenómeno objetivo (histéresis = disipación), asignarle el mayor peso (1.4) es físicamente coherente. **Pero el valor 1.4 en sí es de diseño**: debe defenderse mostrando que dominar la recuperación con el descriptor energético mejora (o no degrada) el LOOCV frente a pesos uniformes.

---

### Normalización robusta: mediana/IQR vs media/desviación

Con datos experimentales ruidosos, outliers y n pequeño, la media y σ se distorsionan por valores extremos (un outlier los infla, comprimiendo los inliers y sesgando las distancias euclídeas). La mediana y el IQR son resistentes: no dependen de cada valor, sólo del centro de la distribución. La transformación (x − mediana)/IQR centra en 0 y escala por la dispersión robusta, preservando la geometría relativa de los inliers. Implementación canónica: `RobustScaler` de scikit-learn. (Respaldo peer-reviewed específico con DOI: sin respaldo localizado; se apoya en documentación canónica y en la teoría de estadística robusta.)

---

### CBR en tiempo real / embebido: factibilidad

La recuperación por vecino más cercano sobre una memoria de ~19 casos y 7 features es trivial computacionalmente (O(n·D) por consulta), perfectamente compatible con HIL y con el acelerador Hailo-8L. Existen antecedentes de CBR en diagnóstico en línea y de kNN acelerado en hardware:

- **Behera & Prathuri (2024)**, "FPGA-Based Acceleration of K-Nearest Neighbor Algorithm on Fully Homomorphic Encrypted Data", *Cryptography* 8(1):8, DOI 10.3390/cryptography8010008 — implementa kNN sobre datos **cifrados** (esquema homomórfico CKKS) en una FPGA Intel Agilex7, llevando el tiempo de cómputo de kNN sobre texto cifrado a un valor realista del orden del kNN sobre texto plano; demuestra la **factibilidad de acelerar kNN en hardware embebido** (no es una cifra de latencia medida en fresado, sino evidencia de viabilidad hardware).
- CBR para diagnóstico de fallos embebido: "Optimized fault diagnosis based on FMEA-style CBR and BN for embedded software system", *International Journal of Advanced Manufacturing Technology*, DOI 10.1007/s00170-017-0110-y.
- CBR para diagnóstico de fallos en aero-motores (similitud por kNN, validación 5-fold), *Expert Systems with Applications* (S0957417422007047).

La traducción Python→MATLAB/Simulink (MATLAB Function block / System Object) es directa porque el razonador es aritmética simple (normalización, distancia ponderada, umbral), sin dependencias de entrenamiento.

---

### Referencias (IEEE con DOI/URL)

1. A. Aamodt y E. Plaza, "Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches," *AI Communications*, vol. 7, no. 1, pp. 39–59, 1994. DOI: 10.3233/AIC-1994-7104.
2. R. López de Mántaras et al., "Retrieval, reuse, revision and retention in case-based reasoning," *The Knowledge Engineering Review*, vol. 20, no. 3, pp. 215–240, 2005. DOI: 10.1017/S0269888906000646.
3. J. Kolodner, *Case-Based Reasoning*. San Mateo, CA: Morgan Kaufmann, 1993. ISBN 1-55860-237-2.
4. D. Wettschereck, D. W. Aha y T. Mohri, "A Review and Empirical Evaluation of Feature Weighting Methods for a Class of Lazy Learning Algorithms," *Artificial Intelligence Review*, vol. 11, pp. 273–314, 1997. DOI: 10.1023/A:1006593614256.
5. O. Dahmoune, I. Meddour, M. Elbah, M. A. Yallese y S. Belhadi, "Development of an adaptive tool condition monitoring system: integration of case-based reasoning with CNN," *Journal of Intelligent Manufacturing*, 2025. DOI: 10.1007/s10845-025-02566-9.
6. "An improved case based reasoning method and its application in estimation of surface quality toward intelligent machining," *Journal of Intelligent Manufacturing*, 2020. DOI: 10.1007/s10845-020-01573-2.
7. D. R. Salgado y F. J. Alonso, "An approach based on current and sound signals for in-process tool wear monitoring," *International Journal of Machine Tools and Manufacture*, vol. 47, pp. 2140–2152, 2007.
8. B. Kaya, C. Oysu y H. M. Ertunc, "Force–torque based on-line tool wear estimation system for CNC milling of Inconel 718 using neural networks," *Advances in Engineering Software*, vol. 42, pp. 76–84, 2011.
9. A. E. Charalampakis y V. K. Koumousis, "On the response and dissipated energy of Bouc–Wen hysteretic model," *Journal of Sound and Vibration*, vol. 309, pp. 887–895, 2008. DOI: 10.1016/j.jsv.2007.07.080.
10. M. Ismail, F. Ikhouane y J. Rodellar, "The Hysteresis Bouc-Wen Model, a Survey," *Archives of Computational Methods in Engineering*, vol. 16, no. 2, pp. 161–188, 2009. DOI: 10.1007/s11831-009-9031-8.
11. M. Ruderman, "Energy dissipation and hysteresis cycles in pre-sliding transients of kinetic friction," *Applications of Mathematics*, vol. 68, no. 6, pp. 845–860, 2023. DOI: 10.21136/AM.2023.0283-22.
12. J. Oh y D. S. Bernstein, "Semilinear Duhem model for rate-independent and rate-dependent hysteresis," *IEEE Transactions on Automatic Control*, vol. 50, no. 5, pp. 631–645, 2005. DOI: 10.1109/TAC.2005.847035.
13. A. K. Padthe, B. Drincic, J. Oh, D. D. Rizos, S. D. Fassois y D. S. Bernstein, "Duhem modeling of friction-induced hysteresis," *IEEE Control Systems Magazine*, vol. 28, no. 5, pp. 90–107, 2008. DOI: 10.1109/MCS.2008.927331.
14. P. Perner, "Concepts for novelty detection and handling based on a case-based reasoning process scheme," *Engineering Applications of Artificial Intelligence* (versión LNCS: DOI 10.1007/978-3-540-73435-2_3).
15. P. A. Szczepaniak (y col.), "Case-Based Reasoning: The Search for Similar Solutions and Identification of Outliers," *Complexity*, 2018. DOI: 10.1155/2018/9280787.
16. M. Huerta et al., "Method for Friction Force Estimation on the Flank of Cutting Tools," *Shock and Vibration*, 2017. DOI: 10.1155/2017/5956425.
17. "Optimized fault diagnosis based on FMEA-style CBR and BN for embedded software system," *International Journal of Advanced Manufacturing Technology*, 2017. DOI: 10.1007/s00170-017-0110-y.
18. S. Behera y J. R. Prathuri, "FPGA-Based Acceleration of K-Nearest Neighbor Algorithm on Fully Homomorphic Encrypted Data," *Cryptography*, vol. 8, no. 1, art. 8, 2024. DOI: 10.3390/cryptography8010008.
19. scikit-learn, "RobustScaler," documentación. URL: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

*Nota: las entradas 7 y 8 (Salgado & Alonso 2007; Kaya et al. 2011) y las versiones journal de Perner (2008) y Szczepaniak (2018) se citan con venue y, donde corresponde, DOI/PII; conviene confirmar el DOI exacto en la base del editor antes de la versión final. Las entradas con DOI completo (1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 14-LNCS, 15, 16, 17, 18) fueron verificadas.*

---

### Puntos débiles y defensa (los 3 ataques más probables del tribunal)

**Ataque 1: "Los valores de los pesos (1.4/1.0/0.8/0.7/0.3) y los umbrales (0.3/1.0 y rechazo) son arbitrarios."**
Defensa: reconocer que son hiperparámetros de diseño sin cita directa; justificarlos por criterio físico (mayor peso al descriptor energético, el área del lazo, que ES la energía disipada según Charalampakis & Koumousis 2008) y demostrar con análisis de sensibilidad + LOOCV que el ranking de recuperación y la clasificación son estables frente a perturbaciones de ±20%. Citar Wettschereck et al. (1997) para encuadrar la elección de pesos como problema metodológico conocido y abierto.

**Ataque 2: "Con 19 casos no se puede validar ni generalizar."**
Defensa: CBR es un paradigma few-shot por diseño (Aamodt & Plaza 1994; Dahmoune et al. 2025); usar LOOCV (máximo aprovechamiento de n pequeño), reportar cobertura/competencia de la memoria y activar el rechazo por novedad (Perner 2008) como salvaguarda explícita frente a extrapolación.

**Ataque 3: "Un modelo entrenado (Bouc–Wen/PINN) sería más riguroso que recuperar casos."**
Defensa: el CBR no compite con sino que alimenta esos modelos: cada caso enlaza a ajustes Bouc–Wen/Duhem/LuGre/PINN/KAN; el área del lazo (físicamente la energía disipada, Charalampakis & Koumousis 2008) es el puente entre el descriptor empírico y la identificación paramétrica. El CBR aporta interpretabilidad, operación en línea barata (factibilidad de aceleración hardware, Behera & Prathuri 2024) y aprendizaje incremental que un PINN entrenado off-line no ofrece con sólo 19 cortes.

## Recommendations
1. **Inmediato**: ejecutar LOOCV sobre los 19 cortes y un análisis de sensibilidad de pesos/umbrales (±20%); documentar que el ranking es estable. Umbral de cambio: si el LOOCV se degrada >10% al perturbar un peso, ese peso debe re-derivarse de los datos en vez de fijarse a priori.
2. **Derivar umbrales de los datos**: fijar 0.3/1.0 como terciles del área observada o por criterio físico (energía elástica vs disipada), no como números fijos sin justificación.
3. **Fijar el umbral de rechazo** por el percentil 90–95 de las distancias LOOCV de aciertos; reportar curva rechazo-vs-error como evidencia de calibración.
4. **Antes de imprenta**: confirmar los DOI exactos de las entradas sin DOI completo (Salgado & Alonso 2007; Kaya et al. 2011; versiones journal de Perner y Szczepaniak) y localizar (o declarar formalmente ausente, "sin respaldo localizado") una referencia peer-reviewed con DOI para robust scaling mediana/IQR; mientras tanto, apoyarse en la documentación canónica de scikit-learn y la teoría de estadística robusta.
5. **Reforzar el puente físico**: presentar explícitamente al tribunal la cadena área del lazo → energía disipada (Charalampakis & Koumousis 2008; Ruderman 2023) → identificación Bouc–Wen/Duhem (Ismail et al. 2009; Oh & Bernstein 2005; Padthe et al. 2008) como la justificación del peso 1.4.

## Caveats
- Los valores numéricos concretos de pesos (1.0/0.8/0.7/1.4/0.3) y umbrales (0.3/1.0 y rechazo) NO tienen respaldo en literatura; deben defenderse empíricamente (sensibilidad + LOOCV + criterio físico), no por cita. Forzar una cita para estos números invalidaría el rigor del trabajo.
- Varias evidencias son "analógicas" (otro dominio: preprocesamiento ML general, novedad en CBR, kNN en hardware para datos cifrados) y deben presentarse como tales, no como evidencia directa en fresado.
- La referencia específica de robust scaling con mediana/IQR queda como "sin respaldo localizado" con DOI primario peer-reviewed; se sostiene con documentación canónica de scikit-learn y teoría de estadística robusta.
- Behera & Prathuri (2024) acelera kNN sobre datos cifrados en FPGA; sustenta la factibilidad de acelerar kNN en hardware embebido, pero no aporta una cifra de latencia medida en una aplicación de fresado ni en el Hailo-8L específicamente.
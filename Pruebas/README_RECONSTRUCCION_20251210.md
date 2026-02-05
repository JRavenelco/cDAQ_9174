# Reconstrucción de Actividades - 10 de Diciembre de 2025

Este documento resume la actividad de investigación y desarrollo realizada el 10 de diciembre de 2025, basada en la evidencia forense de los archivos del proyecto. El foco principal fue el **modelado matemático de la fuerza de corte** utilizando la **envolvente de la aceleración** como característica clave.

## 📂 Archivos Clave y Rutas Relativas

Todas las rutas son relativas a la raíz del proyecto (`Pruebas/`).

| Archivo | Ruta Relativa | Descripción |
|---------|---------------|-------------|
| **Generador de Datos** | `src/caracterizacion_sensor_fuerza.py` | Script de adquisición que generó los datos crudos. Contiene la lógica `_save_raw_recording` que crea los CSVs. |
| **Datos Crudos** | `src/caracterizacion_fuerza/corte_20251209_144635.csv` | Archivo de datos capturado el 9 de dic. Contiene: `tiempo_s`, `fuerza_V`, `aceleracion_prensa_g`, `aceleracion_pieza_g`. |
| **Comparativa Modelos** | `src/caracterizacion_fuerza/test_modelos_envolvente.py` | Script principal del 10 de dic. Compara 5 modelos (BW, Duhem, KAN-PINN) usando la envolvente de aceleración. |
| **Prueba KAN-PINN** | `src/caracterizacion_fuerza/test_kan_pinn_simple.py` | Implementación específica del modelo híbrido `F = m·a + k·x + c·v + α·k·E + (1-α)·k·z`. |
| **Visualización** | `src/manim_videos/presentacion_semestral_v3.py` | Script de animación (Manim) modificado para incluir visualizaciones de estos hallazgos para la presentación semestral. |

## 🕵️‍♂️ Reconstrucción de los Hechos

### 1. Origen de los Datos
El archivo `corte_20251209_144635.csv` no es un archivo externo. Fue generado por tu propia herramienta `caracterizacion_sensor_fuerza.py` durante una sesión de adquisición. Es la materia prima que usaste para los análisis del día siguiente.

### 2. Hipótesis Investigada
La actividad del 10 de diciembre giró en torno a una hipótesis central:
> *La aceleración cruda es demasiado ruidosa para predecir la fuerza de corte directamente, pero su **envolvente (energía)** tiene una alta correlación con la fuerza y puede usarse como entrada para modelos híbridos.*

### 3. Flujo de Trabajo (10 Dic 2025)

#### A. Modelado Híbrido (`test_modelos_envolvente.py`)
Creaste un entorno de pruebas ("benchmark") para comparar diferentes ecuaciones constitutivas para la fuerza de corte:
*   **BW Simple & Viscoso:** Modelos de Bouc-Wen clásicos.
*   **Duhem:** Modelo de histéresis alternativo.
*   **BW-ENV:** Variante donde la histéresis es impulsada por la envolvente.
*   **KAN-PINN:** Tu arquitectura propuesta que combina física lineal (inercia, rigidez) con aprendizaje de envolvente y componentes no lineales.

#### B. Refinamiento KAN-PINN (`test_kan_pinn_simple.py`)
Profundizaste en la arquitectura **KAN-PINN**, definiendo una ecuación donde el parámetro $\alpha$ controla el peso entre el comportamiento lineal (basado en envolvente) y el comportamiento histerético:
$$ F = m \cdot a + k \cdot x + c \cdot v + \alpha \cdot k \cdot E + (1-\alpha) \cdot k \cdot z $$
El script utiliza algoritmos de evolución diferencial para encontrar los parámetros óptimos que ajustan esta ecuación a los datos experimentales del 9 de diciembre.

#### C. Comunicación (`presentacion_semestral_v3.py`)
Paralelamente, trabajaste en la comunicación de estos resultados, integrando los nuevos conceptos en las animaciones de tu presentación semestral usando la librería Manim.

## ✅ Estado Actual
El proyecto se encuentra en una etapa donde la **inclusión de la envolvente de aceleración** ha sido validada preliminarmente como una mejora significativa para los modelos de estimación de fuerza. El siguiente paso lógico sería consolidar el modelo KAN-PINN como el candidato principal.

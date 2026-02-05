# GUÍA DE CONTEXTO Y PROYECTOS - LABORATORIO DAQ & LEVITADOR

Esta guía tiene como objetivo orientar a un agente de IA sobre la estructura, estado y objetivos de los proyectos alojados en este directorio.

## 1. Resumen General
Este espacio de trabajo contiene experimentos de doctorado enfocados en:
1.  **Control de Levitador Magnético**: Implementación de controladores PID, Observadores de Flujo (Santana 2023) y redes neuronales Physics-Informed (HiPPO-KAN) para levitación sensorless.
2.  **Análisis de Dinámica de Corte (Fresado)**: Identificación de fricción, histéresis (Bouc-Wen) y análisis de señales de vibración/fuerza.
3.  **Adquisición de Datos (DAQ)**: Interfaces personalizadas en Python (PyQt) para hardware National Instruments (cDAQ-9174).

## 2. Estructura de Directorios Clave

### `levitador valentin/` (C++)
-   **Propósito**: Control en tiempo real de bajo nivel y generación de datos físicos.
-   **Archivo Principal**: `levitador.exe` (compilado de `levitador.cpp`).
-   **Uso**: Generar datasets físicos de alta frecuencia.
-   **Ejecución**: Usar `ejecutar_levitador.bat`.

### `levitador-python/` (Python)
-   **Propósito**: Control de alto nivel, prototipado de observadores y entrenamiento de redes.
-   **Archivos Clave**:
    -   `control.py`: Script de control PID principal.
    -   `control_observador.py`: Implementación de observadores (Luenberger, KAN).
    -   `entrenar_kan_pinn_v2.py`: Entrenamiento de redes KAN.

### `Pruebas/` (General)
-   **Propósito**: Scripts de análisis, interfaces DAQ y experimentos varios.
-   **Subdirectorios importantes**:
    -   `src/projectAD/`: Interfaz principal de adquisición DAQ.
    -   `src/levitador_gui.py`: Digital Twin y visualización 3D.
    -   `src/caracterizacion_fuerza/`: Datos y scripts de análisis FRF.

### `levitador-benchmark/`
-   **Propósito**: Repositorio para benchmarking de algoritmos de control.

## 3. Estado de los Proyectos (Enero 2026)

### A. Levitador Sensorless
-   **Estado**: Funcional en modo Híbrido (80% Física / 20% IA).
-   **Hito Reciente**: Integración de HiPPO-KAN para corrección de residuos del modelo físico.
-   **Archivos de Datos**: `MONIT.txt` (logs de telemetría).

### B. Análisis de Corte y Fricción
-   **Estado**: Validación de modelos de fricción (CNN vs PINN).
-   **Hallazgo**: Las CNN superan a las PINN clásicas en datos de alta frecuencia.
-   **Pendiente**: Validación de histéresis en corte agresivo (Fase 2).

## 4. Notas Técnicas para el Agente
-   **Entorno**: Python 3.14 en Windows.
-   **Hardware**: NI cDAQ-9174, Módulos NI-9205, NI-9234, NI-9219.
-   **Comunicación**: Puerto Serie (COM1/COM4) para el levitador, UDP para telemetría GUI.
-   **Memorias**: Consultar base de datos de memorias para detalles de parámetros PID (kp=100, ki=50, kd=1.5) y calibración de sensores.

## 5. Instrucciones de Seguridad
-   **NO** borrar archivos `.txt` de datos crudos sin respaldo.
-   **NO** ejecutar comandos de movimiento de máquina (CNC) sin confirmación explícita.
-   Verificar puertos COM antes de iniciar scripts de control.

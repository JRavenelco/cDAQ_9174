#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera una presentación PowerPoint con el resumen del experimento de fricción.

Requisitos:
    pip install python-pptx

Salida:
    caracterizacion_fuerza/presentacion_friccion_kanpinn.pptx
"""

import os
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

# =============================================================================
# RUTAS BÁSICAS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "caracterizacion_fuerza")

OUTPUT_PPTX = os.path.join(
    FIG_DIR,
    "presentacion_friccion_kanpinn_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pptx",
)

# Figuras generadas previamente por los otros scripts
FIGURAS = {
    "fft_thd_comparativo": os.path.join(FIG_DIR, "fft_thd_comparativo.png"),
    "fft_espectro_lc302_25": os.path.join(FIG_DIR, "fft_espectros_LC3021K_25Hz.png"),
    "fft_espectro_dymh_20": os.path.join(FIG_DIR, "fft_espectros_DYMH105_20Hz.png"),
    "analisis_hoy_temporal": os.path.join(FIG_DIR, "analisis_hoy_temporal.png"),
    "analisis_hoy_fft": os.path.join(FIG_DIR, "analisis_hoy_fft.png"),
    "analisis_hoy_histeresis": os.path.join(FIG_DIR, "analisis_hoy_histeresis.png"),
    "kan_identificacion_hoy": os.path.join(FIG_DIR, "kan_identificacion_mkc_hoy.png"),
    "kan_comparacion_sensores": os.path.join(FIG_DIR, "kan_comparacion_sensores.png"),
    "kan_identificacion_todos": os.path.join(FIG_DIR, "kan_identificacion_todos.png"),
}


# =============================================================================
# UTILIDADES
# =============================================================================


def add_title_slide(prs, title, subtitle=""):
    layout = prs.slide_layouts[0]  # Title slide
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title
    if subtitle:
        slide.placeholders[1].text = subtitle
    return slide


def add_text_slide(prs, title, bullet_lines):
    """Crea una diapositiva con título y viñetas."""
    layout = prs.slide_layouts[1]  # Title and Content
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title

    body = slide.placeholders[1]
    tf = body.text_frame
    tf.clear()

    first = True
    for line in bullet_lines:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(20)
    return slide


def add_image_slide(prs, title, image_path, notes=None, max_height_inches=5.0):
    layout = prs.slide_layouts[5]  # Title Only
    slide = prs.slides.add_slide(layout)
    slide.shapes.title.text = title

    if os.path.exists(image_path):
        left = Inches(0.5)
        top = Inches(1.5)
        pic = slide.shapes.add_picture(image_path, left, top)

        # Redimensionar si es demasiado alto
        max_h = Inches(max_height_inches)
        if pic.height > max_h:
            scale = max_h / float(pic.height)
            pic.height = max_h
            pic.width = int(pic.width * scale)
    else:
        body = slide.placeholders[0] if slide.placeholders else None
        if body is not None:
            tf = body.text_frame
            tf.text = f"Figura no encontrada:\n{image_path}"

    if notes:
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = notes

    return slide


# =============================================================================
# CONTENIDO DE LA PRESENTACIÓN
# =============================================================================


def build_presentation():
    prs = Presentation()

    # --- Portada ---
    add_title_slide(
        prs,
        title="Identificación de Fricción con KAN-PINN",
        subtitle=(
            "Análisis experimental de fricción en levitador mecánico\n"
            "FFT, THD y modelo masa-resorte-amortiguador lineal"
        ),
    )

    # --- Objetivo general ---
    add_text_slide(
        prs,
        title="Objetivo del experimento",
        bullet_lines=[
            "Caracterizar la fricción de un sistema mecánico bajo excitación armónica (5–85 Hz)",
            "Comparar sensores DYMH-105 (bancada) y LC302-1K (punto de interés)",
            "Aplicar un KAN-PINN para identificar parámetros m, k, c y componente de fricción",
            "Validar linealidad/no-linealidad mediante FFT y Distorsión Armónica Total (THD)",
        ],
    )

    # --- Pipeline de procesamiento ---
    add_text_slide(
        prs,
        title="Pipeline de procesamiento de señales",
        bullet_lines=[
            "1. Adquisición: fuerza_V, aceleración_g, tiempo_s a 2500 Hz",
            "2. Conversión: g → m/s² y centrado (quita DC)",
            "3. Integración numérica (trapezoidal) para obtener v_raw y x_raw",
            "4. Filtro Butterworth pasa-alto tras cada integración para eliminar drift",
            "   a(t) → ∫ → v_raw → HP → v(t) → ∫ → x_raw → HP → x(t)",
            "5. Cálculo de fricción F_fric = F_total - m·a (más adelante, con m identificado)",
        ],
    )

    # --- Señales temporales de hoy ---
    add_image_slide(
        prs,
        title="Señales temporales 10–40 Hz (LC302-1K)",
        image_path=FIGURAS["analisis_hoy_temporal"],
        notes="Fuerza y aceleración en el tiempo para 10, 20, 30 y 40 Hz.",
    )

    # --- FFT hoy ---
    add_image_slide(
        prs,
        title="Espectros FFT y fricción (datos de hoy)",
        image_path=FIGURAS["analisis_hoy_fft"],
        notes=(
            "Espectros de aceleración, fuerza y fricción para el barrido 10–40 Hz. "
            "Permiten ver armónicos impares característicos de fricción no lineal."
        ),
    )

    # --- Histéresis ---
    add_image_slide(
        prs,
        title="Curvas de histéresis F vs x (4 Dic)",
        image_path=FIGURAS["analisis_hoy_histeresis"],
        notes="Histéresis moderada, consistente con fricción principalmente viscosa con pequeñas no linealidades.",
    )

    # --- THD comparativo ---
    add_image_slide(
        prs,
        title="THD de fricción vs frecuencia de excitación",
        image_path=FIGURAS["fft_thd_comparativo"],
        notes=(
            "DYMH-105 presenta THD muy alto (no linealidad fuerte) en varias frecuencias.\n"
            "LC302-1K muestra THD moderado/bajo → fricción mayormente viscosa en el punto de interés."
        ),
    )

    # --- Caso casi lineal ---
    add_image_slide(
        prs,
        title="Ejemplo casi lineal: LC302-1K @ 25 Hz (THD ≈ 10.9%)",
        image_path=FIGURAS["fft_espectro_lc302_25"],
        notes=(
            "Fundamental limpia a 25 Hz y armónicos 3f, 5f pequeños.\n"
            "Confirma que la fricción seca (Fc) es pequeña en esta configuración."
        ),
    )

    # --- Caso fuertemente no lineal ---
    add_image_slide(
        prs,
        title="Ejemplo NO lineal: DYMH-105 @ 20 Hz (THD ≈ 109%)",
        image_path=FIGURAS["fft_espectro_dymh_20"],
        notes=(
            "Armónicos impares muy fuertes (3f, 5f, ...) → fricción de Coulomb/histéresis pronunciada \n"
            "en el montaje de bancada con DYMH-105."
        ),
    )

    # --- KAN-PINN: datos de hoy ---
    add_image_slide(
        prs,
        title="KAN-PINN: identificación m, k, c con datos de hoy",
        image_path=FIGURAS["kan_identificacion_hoy"],
        notes=(
            "Entrenamiento del KAN-PINN con datos 10–40 Hz (4 Dic).\n"
            "El modelo converge a una frecuencia natural alrededor de 26–27 Hz y Fc ≈ 0."
        ),
    )

    # --- Comparación sensores ---
    add_image_slide(
        prs,
        title="Comparación KAN-PINN: DYMH-105 vs LC302-1K",
        image_path=FIGURAS["kan_comparacion_sensores"],
        notes=(
            "DYMH-105 y LC302-1K identifican un oscilador masa-resorte-amortiguador.\n"
            "LC302-1K muestra mayor rigidez efectiva y menor amortiguamiento relativo."
        ),
    )

    # --- KAN-PINN: todos los datos ---
    add_image_slide(
        prs,
        title="KAN-PINN global: todos los datos (DYMH-105 + LC302-1K + 4 Dic)",
        image_path=FIGURAS["kan_identificacion_todos"],
        notes=(
            "Modelo entrenado con 27 archivos y ≈166k puntos.\n"
            "Resultado global: fn ≈ 23.8 Hz, ζ ≈ 6%, Fc ≈ 0 → sistema globalmente lineal/viscoso."
        ),
    )

    # --- Diapositiva de conclusiones ---
    add_text_slide(
        prs,
        title="Conclusiones principales",
        bullet_lines=[
            "1. El sistema se comporta como un oscilador masa-resorte-amortiguador con fn ≈ 24 Hz",
            "2. El factor de amortiguamiento global es ζ ≈ 6% (amortiguamiento bajo)",
            "3. El sensor LC302-1K ve una fricción mayormente viscosa (THD bajo, Fc ≈ 0)",
            "4. Las no linealidades fuertes (THD alto) se concentran en la bancada (DYMH-105)",
            "5. El KAN-PINN permite identificar m_v, k_v, c_v coherentes con la evidencia espectral",
        ],
    )

    # --- Checklist metodológico ---
    add_text_slide(
        prs,
        title="Checklist metodológico (pasos para replicar)",
        bullet_lines=[
            "1. Adquirir fuerza_V y aceleración_g a 2500 Hz para un barrido de frecuencias",
            "2. Convertir a(t) a m/s², integrar → v(t), x(t) con filtros pasa-alto tras cada integración",
            "3. Calcular FFT y THD de la fricción para evaluar linealidad vs. no linealidad",
            "4. Entrenar KAN-PINN con ecuación F_V = m_v a + c_v v + k_v x + Fc sign(v) + f_residual",
            "5. Obtener fn y ζ a partir de m_v y k_v, verificar que Fc ≈ 0 y residual pequeño",
            "6. Comparar resultados entre sensores/barridos para validar el modelo global",
        ],
    )

    return prs


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    prs = build_presentation()
    os.makedirs(FIG_DIR, exist_ok=True)
    prs.save(OUTPUT_PPTX)
    print(f"Presentación guardada en: {OUTPUT_PPTX}")

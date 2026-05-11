"""
Analizador de tesis doctoral usando Gemini API.
Replica el sistema de 4 niveles del video para la tesis de histéresis.
"""

import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY or API_KEY == "tu_clave_aqui":
    print("ERROR: Agrega tu clave en el archivo .env")
    print(f"  Archivo: {BASE_DIR / '.env'}")
    print("  Línea:   GEMINI_API_KEY=AIza...")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
MODELOS = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro"]


def leer_docx(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
    root = tree.getroot()
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    parrafos = []
    for p in root.iter(f"{{{W}}}p"):
        texto = "".join(t.text for t in p.iter(f"{{{W}}}t") if t.text).strip()
        if texto:
            parrafos.append(texto)
    return "\n".join(parrafos)


def leer_texto(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def analizar(pregunta: str, contexto: str, titulo: str = "") -> str:
    if titulo:
        print(f"\n{'='*60}")
        print(f"  {titulo}")
        print('='*60)

    prompt = f"""Eres un experto en mecatrónica, control no lineal y machine learning científico.
Analiza el siguiente material de tesis doctoral sobre predicción de histéresis.

CONTEXTO DE LA TESIS:
- Autor: José de Jesús Santana Ramírez (UAQ, Doctor en Ciencias de la Computación)
- Tema: Modelo predictivo de histéresis usando CBR + aritmética modular + análisis de entropía
- Modelos físicos: LuGre, Bouc-Wen, Dahl, Preisach
- Técnicas IA: PINNs, DeepONet, KAN, HiPPO-KAN
- Hardware: NI cDAQ-9174 (módulos 9205/9234/9219)

MATERIAL:
{contexto[:30000]}

TAREA:
{pregunta}

Responde en español, de forma estructurada, con observaciones concretas y accionables.
"""
    for modelo in MODELOS:
        for intento in range(3):
            try:
                print(f"  [usando {modelo}...]")
                response = client.models.generate_content(model=modelo, contents=prompt)
                resultado = response.text
                print(resultado)
                return resultado
            except genai_errors.ServerError as e:
                espera = 20 * (intento + 1)
                print(f"  Servidor ocupado, reintentando en {espera}s... ({intento+1}/3)")
                time.sleep(espera)
            except genai_errors.ClientError as e:
                if "429" in str(e):
                    print(f"  Cuota agotada en {modelo}, probando siguiente modelo...")
                    break
                raise
        else:
            continue
        break
    print("ERROR: Todos los modelos fallaron. Intenta más tarde.")
    return ""


def menu():
    print("\n" + "="*60)
    print("  ANALIZADOR DE TESIS - Sistema 4 Niveles Gemini")
    print("="*60)
    print("\n[1] Análisis completo de la tesis (brechas y fortalezas)")
    print("[2] Revisar coherencia del Capítulo 3 (metodología)")
    print("[3] Evaluar resultados del Capítulo 4")
    print("[4] Sugerir mejoras para el artículo IFToMM")
    print("[5] Generar preguntas de defensa doctoral")
    print("[6] Pregunta libre")
    print("[0] Salir")
    return input("\nOpción: ").strip()


def main():
    print("\nCargando documentos...")

    tesis_path = BASE_DIR / "TESIS DOCTORAL 2025 SANTANA-RAMIREZ.docx"
    cap4_path = BASE_DIR / "Capitulo4_Resultados_Extendido.md"
    guia_path = BASE_DIR / "GUIA_PROYECTO_IA.md"

    tesis = leer_docx(str(tesis_path))
    cap4 = leer_texto(str(cap4_path))
    guia = leer_texto(str(guia_path))

    contexto_completo = f"""=== TESIS DOCTORAL ===
{tesis}

=== CAPÍTULO 4 EXTENDIDO ===
{cap4}

=== GUÍA DEL PROYECTO ===
{guia}
"""
    print(f"Documentos cargados: {len(contexto_completo):,} caracteres")

    preguntas = {
        "1": (
            "Identifica las principales fortalezas y brechas metodológicas de esta tesis. "
            "¿Qué falta para que sea publicable en una revista Q1? Lista máximo 5 puntos críticos.",
            "ANÁLISIS COMPLETO — BRECHAS Y FORTALEZAS"
        ),
        "2": (
            "Evalúa la coherencia interna del Capítulo 3 (metodología). "
            "¿Los objetivos específicos están alineados con el diseño experimental? "
            "¿Hay pasos metodológicos sin respaldo o mal definidos?",
            "COHERENCIA DEL CAPÍTULO 3 — METODOLOGÍA"
        ),
        "3": (
            "Analiza los resultados del Capítulo 4. "
            "¿Son suficientes para sustentar las proposiciones del Capítulo 3? "
            "¿Qué experimentos adicionales se necesitan?",
            "EVALUACIÓN DEL CAPÍTULO 4 — RESULTADOS"
        ),
        "4": (
            "El artículo 'Detección de histéresis F-v a partir de vibraciones' recibió retroalimentación. "
            "Basándote en la metodología de la tesis, sugiere las 3 mejoras más importantes "
            "para fortalecerlo antes de reenvío.",
            "MEJORAS PARA EL ARTÍCULO IFTOMM"
        ),
        "5": (
            "Genera 10 preguntas difíciles que el comité doctoral podría hacer en la defensa. "
            "Incluye preguntas sobre: fundamentos teóricos, validación experimental, "
            "comparación con el estado del arte y limitaciones del modelo.",
            "PREGUNTAS DE DEFENSA DOCTORAL"
        ),
    }

    while True:
        opcion = menu()

        if opcion == "0":
            print("\nHasta luego.")
            break
        elif opcion in preguntas:
            pregunta, titulo = preguntas[opcion]
            resultado = analizar(pregunta, contexto_completo, titulo)
            guardar = input("\n¿Guardar resultado? (s/n): ").strip().lower()
            if guardar == "s":
                nombre = f"resultado_{titulo[:30].replace(' ', '_').replace('—', '')}.txt"
                with open(BASE_DIR / nombre, "w", encoding="utf-8") as f:
                    f.write(f"{titulo}\n{'='*60}\n\n{resultado}")
                print(f"Guardado: {nombre}")
        elif opcion == "6":
            pregunta = input("\nEscribe tu pregunta: ").strip()
            if pregunta:
                analizar(pregunta, contexto_completo, "PREGUNTA LIBRE")
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()

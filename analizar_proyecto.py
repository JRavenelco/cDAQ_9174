"""
Analizador genérico de proyectos usando Gemini API.
Replica el sistema de 4 niveles (NotebookLM → Gemini → GEMs → Workspace).
Uso: python analizar_proyecto.py
"""

import os, sys, time, shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ── Fix lxml en Linux/Jetson ──────────────────────────────────────────────────
_user_libs = Path.home() / ".local/lib/python3.10/site-packages"
if _user_libs.exists():
    sys.path.insert(0, str(_user_libs))

from google import genai
from google.genai import errors as genai_errors

# ── Configuración ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY or API_KEY == "tu_clave_aqui":
    print("\nERROR: Crea el archivo .env con tu clave:")
    print(f"  {BASE_DIR / '.env'}")
    print("  Contenido: GEMINI_API_KEY=AIzaSy...")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
MODELOS = ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro"]


# ── Carga de documentos ───────────────────────────────────────────────────────
def cargar_archivos(rutas: list[str]) -> str:
    """Lee todos los archivos de texto/markdown/docx y los concatena."""
    import zipfile, xml.etree.ElementTree as ET

    partes = []
    for ruta in rutas:
        p = Path(ruta)
        if not p.exists():
            print(f"  [!] No encontrado: {ruta}")
            continue
        try:
            if p.suffix.lower() == ".docx":
                with zipfile.ZipFile(p) as z:
                    with z.open("word/document.xml") as f:
                        tree = ET.parse(f)
                W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                texto = "\n".join(
                    "".join(t.text for t in para.iter(f"{{{W}}}t") if t.text).strip()
                    for para in tree.getroot().iter(f"{{{W}}}p")
                )
            else:
                texto = p.read_text(encoding="utf-8", errors="ignore")
            partes.append(f"\n=== {p.name} ===\n{texto[:15000]}")
            print(f"  ✓ {p.name} ({len(texto):,} chars)")
        except Exception as e:
            print(f"  [!] Error leyendo {p.name}: {e}")

    return "\n".join(partes)


# ── Motor Gemini ──────────────────────────────────────────────────────────────
def gemini(pregunta: str, contexto: str, titulo: str = "") -> str:
    if titulo:
        print(f"\n{'='*60}\n  {titulo}\n{'='*60}")

    prompt = f"""Eres un experto analista de proyectos de investigación y desarrollo.

CONTEXTO DEL PROYECTO:
{contexto[:32000]}

TAREA:
{pregunta}

Responde en español. Sé estructurado, concreto y accionable.
"""
    for modelo in MODELOS:
        for intento in range(3):
            try:
                print(f"  [modelo: {modelo}]")
                r = client.models.generate_content(model=modelo, contents=prompt)
                print(r.text)
                return r.text
            except genai_errors.ServerError:
                espera = 20 * (intento + 1)
                print(f"  Servidor ocupado, reintentando en {espera}s...")
                time.sleep(espera)
            except genai_errors.ClientError as e:
                if "429" in str(e):
                    print(f"  Cuota agotada en {modelo}, probando siguiente...")
                    break
                raise
    return ""


def guardar(resultado: str, nombre: str = ""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = nombre or f"resultado_{ts}.txt"
    path = BASE_DIR / nombre
    path.write_text(resultado, encoding="utf-8")
    print(f"\n  ✓ Guardado: {nombre}")
    return path


# ── Análisis predefinidos ─────────────────────────────────────────────────────
ANALISIS = {
    "1": (
        "Haz un análisis completo del proyecto: fortalezas, brechas, riesgos y "
        "oportunidades. Lista los 5 puntos más críticos con acciones concretas.",
        "ANÁLISIS COMPLETO DEL PROYECTO"
    ),
    "2": (
        "Evalúa la coherencia metodológica. ¿Los objetivos están alineados con "
        "la hipótesis y el diseño experimental? ¿Qué pasos están incompletos?",
        "COHERENCIA METODOLÓGICA"
    ),
    "3": (
        "Analiza los resultados disponibles. ¿Son suficientes para sustentar "
        "las conclusiones? ¿Qué experimentos o datos adicionales se necesitan?",
        "EVALUACIÓN DE RESULTADOS"
    ),
    "4": (
        "Diseña un DOE (Design of Experiments) completo y ejecutable para "
        "validar la hipótesis principal del proyecto. Incluye: factores, niveles, "
        "corridas, protocolo y análisis estadístico.",
        "DISEÑO DOE"
    ),
    "5": (
        "Genera 10 preguntas difíciles que un evaluador experto haría sobre este "
        "proyecto. Para cada una, sugiere la respuesta clave.",
        "PREGUNTAS DE EVALUACIÓN"
    ),
    "6": (
        "Redacta un RESUMEN EJECUTIVO (300 palabras) y un ABSTRACT en inglés "
        "(300 palabras) que capturen la esencia del proyecto: problema, propuesta, "
        "metodología, resultados y contribución.",
        "RESUMEN EJECUTIVO + ABSTRACT"
    ),
}


# ── Configuración del proyecto ────────────────────────────────────────────────
def configurar_proyecto() -> tuple[str, str]:
    """Pide al usuario los datos del proyecto y los archivos a cargar."""
    print("\n" + "="*60)
    print("  CONFIGURACIÓN DEL PROYECTO")
    print("="*60)
    print("\nDescribe el proyecto en 2-3 líneas:")
    descripcion = input("> ").strip()

    print("\nRutas de archivos a analizar (Enter para terminar):")
    print("  Soporta: .txt  .md  .pdf (texto)  .docx  .py  .csv")
    rutas = []
    while True:
        r = input(f"  Archivo {len(rutas)+1}: ").strip().strip('"')
        if not r:
            break
        rutas.append(r)

    return descripcion, rutas


# ── Menú principal ────────────────────────────────────────────────────────────
def menu(nombre_proyecto: str):
    print("\n" + "="*60)
    print(f"  ANALIZADOR — {nombre_proyecto[:40]}")
    print("="*60)
    print("[1] Análisis completo del proyecto")
    print("[2] Coherencia metodológica")
    print("[3] Evaluación de resultados")
    print("[4] Diseñar DOE")
    print("[5] Preguntas de evaluación")
    print("[6] Redactar Resumen + Abstract")
    print("[7] Pregunta libre")
    print("[8] Cambiar proyecto / cargar nuevos archivos")
    print("[0] Salir")
    return input("\nOpción: ").strip()


def main():
    print("\n" + "="*60)
    print("  SISTEMA 4 NIVELES — ANALIZADOR DE PROYECTOS")
    print("  Powered by Gemini API")
    print("="*60)

    descripcion, rutas = configurar_proyecto()

    if not rutas:
        print("\nSin archivos cargados. Continuando con descripción solamente.")
        contexto = f"DESCRIPCIÓN DEL PROYECTO:\n{descripcion}"
    else:
        print("\nCargando archivos...")
        contenido = cargar_archivos(rutas)
        contexto = f"DESCRIPCIÓN DEL PROYECTO:\n{descripcion}\n\n{contenido}"
        print(f"\nContexto total: {len(contexto):,} caracteres")

    nombre_proyecto = descripcion[:50] if descripcion else "Proyecto"

    while True:
        opcion = menu(nombre_proyecto)

        if opcion == "0":
            print("Hasta luego.")
            break

        elif opcion in ANALISIS:
            pregunta, titulo = ANALISIS[opcion]
            resultado = gemini(pregunta, contexto, titulo)
            if resultado:
                guardar_r = input("\n¿Guardar resultado? (s/n): ").strip().lower()
                if guardar_r == "s":
                    nombre = input("Nombre del archivo (Enter = automático): ").strip()
                    guardar(resultado, nombre or f"resultado_{opcion}_{datetime.now().strftime('%H%M%S')}.txt")

        elif opcion == "7":
            pregunta = input("\nEscribe tu pregunta o instrucción: ").strip()
            if pregunta:
                resultado = gemini(pregunta, contexto, "CONSULTA LIBRE")
                if resultado:
                    guardar_r = input("\n¿Guardar? (s/n): ").strip().lower()
                    if guardar_r == "s":
                        guardar(resultado)

        elif opcion == "8":
            descripcion, rutas = configurar_proyecto()
            if rutas:
                print("\nCargando archivos...")
                contenido = cargar_archivos(rutas)
                contexto = f"DESCRIPCIÓN DEL PROYECTO:\n{descripcion}\n\n{contenido}"
                print(f"Contexto actualizado: {len(contexto):,} caracteres")
            nombre_proyecto = descripcion[:50]

        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()

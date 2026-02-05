"""
Script para agregar comentarios de revisión a un documento Word (.docx)
Modifica el XML interno para insertar comentarios con fechas específicas.
"""

import zipfile
import os
import shutil
import re
from datetime import datetime
import xml.etree.ElementTree as ET

# Configuración
SRC_DOCX = r'Pruebas/src/t_word_Frida_Andrade.docx'
OUTPUT_DOCX = r'Pruebas/src/t_word_Frida_Andrade_REVISADO.docx'
TEMP_DIR = r'Pruebas/src/t_word_Frida_temp'
AUTHOR = "Revisor"

# Fechas de la semana del 8-12 de diciembre de 2023
DATES = [
    "2023-12-08T09:30:00Z",
    "2023-12-08T14:15:00Z",
    "2023-12-09T10:00:00Z",
    "2023-12-09T16:45:00Z",
    "2023-12-10T11:20:00Z",
    "2023-12-11T09:00:00Z",
    "2023-12-11T15:30:00Z",
    "2023-12-12T10:45:00Z",
    "2023-12-12T14:00:00Z",
    "2023-12-12T17:30:00Z",
]

# Comentarios de revisión organizados por sección
COMENTARIOS = [
    # (id, fecha_idx, texto_buscar, comentario)
    (0, 0, "Resumen", "SUGERENCIA: Agregar resumen/abstract de máximo 250 palabras con: objetivo, metodología, resultados clave (error <2% a 90kHz) y conclusión principal. Incluir 5 palabras clave."),
    
    (1, 1, "Introducción", "SUGERENCIA: Agregar estadísticas actualizadas de diabetes en México (IDF 2023: 14.1 millones). Definir claramente el GAP de investigación."),
    
    (2, 2, "Estado del Arte", "SUGERENCIA: Agregar tabla comparativa de trabajos previos con columnas: Autor, Tipo Sensor, Variable, Comunicación, Lavable, Limitación."),
    
    (3, 2, "Consumer Acceptance", "CORRECCIÓN: 'El prototipo que se diseño' → 'El prototipo que se diseñó' (falta acento)"),
    
    (4, 3, "AD5934", "ERROR TÉCNICO: El AD5934 NO es un 'ADC de precisión'. Es un analizador de impedancia que integra generador DDS + DAC + ADC de 12 bits. Corregir descripción."),
    
    (5, 4, "CN0349", "SUGERENCIA: Agregar diagrama de bloques del circuito y explicar el rol de cada componente (amplificador de transimpedancia, referencia de calibración)."),
    
    (6, 5, "pistola de calor", "SUGERENCIA: Especificar modelo de pistola de calor (marca, potencia), modelo de cámara FLIR (resolución, precisión ±X°C), y tiempo de estabilización entre mediciones."),
    
    (7, 5, "puntos específicos del calcetín", "SUGERENCIA: Especificar los 6 puntos anatómicos exactos (hallux, 1er metatarso, talón medial, etc.) y referenciar estudios que justifiquen estos puntos como críticos para úlceras."),
    
    (8, 6, "filtrado digital", "SUGERENCIA: Especificar tipo de filtro (FIR/IIR), orden, frecuencia de corte, tipo de normalización (Min-Max/Z-score), y frecuencia de muestreo. Incluir fragmento de código."),
    
    (9, 6, "Caracterización de los sensores", "ANÁLISIS FALTANTE: Calcular para cada sensor: Sensibilidad (ΔZ/ΔT en Ω/°C), Linealidad (R²), Repetibilidad (σ entre las 3 pruebas). Agregar tabla resumen y modelo matemático Z(T)=aT+b."),
    
    (10, 7, "Aplicación móvil", "SECCIÓN INCOMPLETA: Agregar capturas de pantalla de la app, descripción de funcionalidades, protocolo BLE (UUID de servicios/características), y tasa de transmisión de datos."),
    
    (11, 7, "Resultados", "SECCIÓN FALTANTE: Agregar sección de DISCUSIÓN que compare con trabajos previos, analice limitaciones del estudio y fuentes de error."),
    
    (12, 8, "Conclusión", "REESCRIBIR COMPLETAMENTE: Las conclusiones actuales parecen notas/pendientes. Deben responder a los objetivos con resultados concretos. Ejemplo: 'Se logró sensibilidad de X.X Ω/°C con R²>0.XX'"),
    
    (13, 8, "prototipo", "SUGERENCIA: Convertir los pendientes en sección de 'Trabajo Futuro' estructurada: validación clínica (n≥30), pruebas de lavado (≥50 ciclos), diseño de PCB, optimización energética."),
    
    (14, 9, "rehusables", "CORRECCIÓN ORTOGRÁFICA: 'rehusables' → 'reutilizables'"),
    
    (15, 9, "mas eficiente", "CORRECCIÓN ORTOGRÁFICA: 'mas eficiente' → 'más eficiente' (falta acento)"),
]

# Namespaces de Word
NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
    'w15': 'http://schemas.microsoft.com/office/word/2012/wordml',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}

def create_comments_xml():
    """Crea el archivo comments.xml con todos los comentarios."""
    
    comments_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
            xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml">
'''
    
    for com_id, date_idx, _, texto in COMENTARIOS:
        fecha = DATES[date_idx % len(DATES)]
        # Escapar caracteres especiales en XML
        texto_escaped = texto.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
        
        comments_xml += f'''    <w:comment w:id="{com_id}" w:author="{AUTHOR}" w:date="{fecha}" w:initials="R">
        <w:p>
            <w:r>
                <w:t>{texto_escaped}</w:t>
            </w:r>
        </w:p>
    </w:comment>
'''
    
    comments_xml += '</w:comments>'
    return comments_xml


def create_comments_extended_xml():
    """Crea commentsExtended.xml para Word 2013+."""
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w15:commentsEx xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml">
</w15:commentsEx>'''


def update_content_types(content_types_path):
    """Agrega los tipos de contenido para comments."""
    with open(content_types_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Agregar Override para comments.xml si no existe
    if 'comments.xml' not in content:
        insert_pos = content.find('</Types>')
        new_entries = '''  <Override PartName="/word/comments.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>
'''
        content = content[:insert_pos] + new_entries + content[insert_pos:]
    
    with open(content_types_path, 'w', encoding='utf-8') as f:
        f.write(content)


def update_document_rels(rels_path):
    """Agrega la relación al archivo comments.xml."""
    with open(rels_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Buscar el último rId usado
    rids = re.findall(r'Id="rId(\d+)"', content)
    max_rid = max([int(r) for r in rids]) if rids else 0
    new_rid = max_rid + 1
    
    # Agregar relación para comments si no existe
    if 'comments.xml' not in content:
        insert_pos = content.find('</Relationships>')
        new_rel = f'  <Relationship Id="rId{new_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/>\n'
        content = content[:insert_pos] + new_rel + content[insert_pos:]
    
    with open(rels_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    print("=" * 60)
    print("Agregando comentarios de revisión al documento Word")
    print("=" * 60)
    
    # Limpiar directorio temporal si existe
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    # Extraer el docx
    print(f"\n1. Extrayendo {SRC_DOCX}...")
    with zipfile.ZipFile(SRC_DOCX, 'r') as z:
        z.extractall(TEMP_DIR)
    
    # Crear comments.xml
    print("\n2. Creando comments.xml con los comentarios de revisión...")
    comments_path = os.path.join(TEMP_DIR, 'word', 'comments.xml')
    with open(comments_path, 'w', encoding='utf-8') as f:
        f.write(create_comments_xml())
    print(f"   - Agregados {len(COMENTARIOS)} comentarios")
    
    # Actualizar [Content_Types].xml
    print("\n3. Actualizando [Content_Types].xml...")
    content_types_path = os.path.join(TEMP_DIR, '[Content_Types].xml')
    update_content_types(content_types_path)
    
    # Actualizar word/_rels/document.xml.rels
    print("\n4. Actualizando relaciones del documento...")
    rels_path = os.path.join(TEMP_DIR, 'word', '_rels', 'document.xml.rels')
    update_document_rels(rels_path)
    
    # Reempaquetar como docx
    print(f"\n5. Creando documento final: {OUTPUT_DOCX}...")
    with zipfile.ZipFile(OUTPUT_DOCX, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(TEMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, TEMP_DIR)
                z.write(file_path, arcname)
    
    # Limpiar
    print("\n6. Limpiando archivos temporales...")
    shutil.rmtree(TEMP_DIR)
    
    print("\n" + "=" * 60)
    print("¡COMPLETADO!")
    print(f"Documento con comentarios: {OUTPUT_DOCX}")
    print("\nComentarios agregados con fechas del 8-12 de diciembre de 2023:")
    for i, (_, date_idx, seccion, _) in enumerate(COMENTARIOS):
        print(f"  [{i+1}] {seccion[:40]}... - {DATES[date_idx % len(DATES)][:10]}")
    print("=" * 60)


if __name__ == '__main__':
    main()

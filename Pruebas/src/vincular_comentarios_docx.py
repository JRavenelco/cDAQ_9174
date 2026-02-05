"""
Script para vincular comentarios a párrafos específicos en un documento Word.
Inserta marcadores commentRangeStart/End y commentReference en el XML.
"""

import zipfile
import os
import shutil
import re
from collections import OrderedDict

# Configuración
SRC_DOCX = r'Pruebas/src/t_word_Frida_Andrade.docx'
OUTPUT_DOCX = r'Pruebas/src/t_word_Frida_Andrade_REVISADO.docx'
TEMP_DIR = r'Pruebas/src/t_word_temp'
AUTHOR = "Revisor"

# Fechas de la semana del 8-12 de diciembre de 2025
DATES = [
    "2025-12-08T09:30:00Z",
    "2025-12-08T14:15:00Z",
    "2025-12-09T10:00:00Z",
    "2025-12-09T16:45:00Z",
    "2025-12-10T11:20:00Z",
    "2025-12-11T09:00:00Z",
    "2025-12-11T15:30:00Z",
    "2025-12-12T10:45:00Z",
    "2025-12-12T14:00:00Z",
    "2025-12-12T17:30:00Z",
]

# Comentarios vinculados a texto específico
# (id, fecha_idx, texto_a_buscar, comentario)
COMENTARIOS = [
    (0, 0, "Introducción", 
     "SUGERENCIA: Agregar estadísticas actualizadas de diabetes en México (IDF 2023: 14.1 millones de adultos). Incluir objetivo general y específicos al final de esta sección."),
    
    (1, 1, "Antecedentes",
     "SUGERENCIA: Agregar tabla comparativa de trabajos previos con columnas: Autor (Año), Tipo Sensor, Variable Medida, Comunicación, Lavable, Limitación Principal."),
    
    (2, 2, "se diseño",
     "CORRECCIÓN ORTOGRÁFICA: 'se diseño' → 'se diseñó' (falta tilde en el verbo)"),
    
    (3, 3, "AD5934",
     "ERROR TÉCNICO: El AD5934 NO es un simple ADC. Es un analizador de impedancia que integra: generador DDS, DAC y ADC de 12 bits. Permite medir impedancia compleja (magnitud y fase)."),
    
    (4, 4, "CN0349",
     "SUGERENCIA: Agregar diagrama de bloques del circuito CN0349 y explicar función de cada componente: amplificador de transimpedancia, referencia de calibración, etc."),
    
    (5, 5, "pistola de calor",
     "SUGERENCIA: Especificar marca y modelo de la pistola de calor, modelo de cámara FLIR (resolución, precisión ±X°C), y tiempo de estabilización térmica entre mediciones."),
    
    (6, 5, "puntos específicos del calcetín",
     "SUGERENCIA: Especificar los 6 puntos anatómicos exactos (ej: hallux, 1er metatarso, talón medial). Referenciar estudios que justifiquen estos puntos como zonas críticas para úlceras."),
    
    (7, 6, "filtrado digital",
     "SUGERENCIA: Especificar: tipo de filtro (FIR/IIR), orden, frecuencia de corte, tipo de normalización (Min-Max/Z-score), frecuencia de muestreo. Incluir pseudocódigo o fragmento de código."),
    
    (8, 6, "Caracterización de los sensores de temperatura",
     "ANÁLISIS CUANTITATIVO FALTANTE: Para cada sensor calcular y reportar: Sensibilidad (ΔZ/ΔT en Ω/°C), Linealidad (R² del ajuste), Repetibilidad (σ entre pruebas). Proponer modelo Z(T)=aT+b."),
    
    (9, 7, "Aplicación móvil",
     "SECCIÓN INCOMPLETA: Agregar capturas de pantalla de la app, descripción de funcionalidades, protocolo BLE (UUIDs), y tasa de transmisión de datos."),
    
    (10, 7, "Resultados",
     "SECCIÓN FALTANTE: Agregar sección de DISCUSIÓN que compare resultados con trabajos previos, analice limitaciones del estudio y fuentes de error."),
    
    (11, 8, "Conclusión",
     "REESCRIBIR: Las conclusiones actuales son notas/pendientes. Deben responder a los objetivos con resultados concretos: 'Se logró sensibilidad de X.X Ω/°C con linealidad R²>0.XX a 90kHz'"),
    
    (12, 8, "prototipo",
     "SUGERENCIA: Convertir pendientes en sección 'Trabajo Futuro' estructurada: validación clínica (n≥30), pruebas lavado (≥50 ciclos), diseño PCB, optimización energética (>24h autonomía)."),
    
    (13, 9, "rehusables",
     "CORRECCIÓN: 'rehusables' → 'reutilizables'"),
    
    (14, 9, "mas eficiente",
     "CORRECCIÓN: 'mas eficiente' → 'más eficiente' (falta tilde)"),
    
    (15, 9, "Bibliografía",
     "SUGERENCIA: Verificar formato consistente (IEEE o APA). Mínimo 25-30 referencias. Agregar: Armstrong et al. (2017) NEJM, Lavery et al. (2007) Diabetes Care, Bus et al. (2020) DMRR."),
]


def escape_xml(text):
    """Escapa caracteres especiales para XML."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))


def create_comments_xml():
    """Crea el archivo comments.xml."""
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
            xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
'''
    for com_id, date_idx, _, texto in COMENTARIOS:
        fecha = DATES[date_idx % len(DATES)]
        texto_escaped = escape_xml(texto)
        xml += f'''    <w:comment w:id="{com_id}" w:author="{AUTHOR}" w:date="{fecha}" w:initials="R">
        <w:p>
            <w:pPr><w:pStyle w:val="CommentText"/></w:pPr>
            <w:r>
                <w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>
                <w:annotationRef/>
            </w:r>
            <w:r>
                <w:t>{texto_escaped}</w:t>
            </w:r>
        </w:p>
    </w:comment>
'''
    xml += '</w:comments>'
    return xml


def insert_comment_markers(document_xml, comment_id, search_text):
    """
    Inserta marcadores de comentario alrededor del texto buscado.
    Retorna el XML modificado y si tuvo éxito.
    """
    # Buscar el texto en elementos <w:t>
    # El patrón busca: <w:t>...texto...</w:t> o <w:t xml:space="preserve">...texto...</w:t>
    
    # Primero intentamos encontrar el texto exacto
    pattern = re.compile(
        rf'(<w:t[^>]*>)([^<]*?)({re.escape(search_text)})([^<]*?)(</w:t>)',
        re.IGNORECASE
    )
    
    match = pattern.search(document_xml)
    if match:
        # Encontrado - insertar marcadores
        before_tag = match.group(1)
        text_before = match.group(2)
        found_text = match.group(3)
        text_after = match.group(4)
        close_tag = match.group(5)
        
        # Construir el reemplazo con marcadores de comentario
        replacement = (
            f'{before_tag}{text_before}{close_tag}'
            f'</w:r>'
            f'<w:commentRangeStart w:id="{comment_id}"/>'
            f'<w:r><w:t>{found_text}</w:t></w:r>'
            f'<w:commentRangeEnd w:id="{comment_id}"/>'
            f'<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
            f'<w:commentReference w:id="{comment_id}"/></w:r>'
            f'<w:r>{before_tag}{text_after}{close_tag}'
        )
        
        # Solo reemplazar la primera ocurrencia
        document_xml = pattern.sub(replacement, document_xml, count=1)
        return document_xml, True
    
    return document_xml, False


def insert_comment_at_paragraph(document_xml, comment_id, search_text):
    """
    Estrategia alternativa: buscar el párrafo que contiene el texto
    e insertar el comentario al inicio del párrafo.
    """
    # Buscar párrafos que contengan el texto
    search_lower = search_text.lower()
    
    # Encontrar la posición del texto
    pos = document_xml.lower().find(search_lower)
    if pos == -1:
        return document_xml, False
    
    # Buscar el inicio del párrafo más cercano antes de esta posición
    p_start = document_xml.rfind('<w:p ', 0, pos)
    if p_start == -1:
        p_start = document_xml.rfind('<w:p>', 0, pos)
    
    if p_start == -1:
        return document_xml, False
    
    # Buscar donde termina la etiqueta de apertura del párrafo
    p_tag_end = document_xml.find('>', p_start) + 1
    
    # Insertar commentRangeStart después de la etiqueta del párrafo
    comment_start = f'<w:commentRangeStart w:id="{comment_id}"/>'
    
    # Buscar el final del párrafo
    p_end = document_xml.find('</w:p>', pos)
    if p_end == -1:
        return document_xml, False
    
    # Insertar commentRangeEnd y commentReference antes del cierre del párrafo
    comment_end = (
        f'<w:commentRangeEnd w:id="{comment_id}"/>'
        f'<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
        f'<w:commentReference w:id="{comment_id}"/></w:r>'
    )
    
    # Construir el nuevo XML
    new_xml = (
        document_xml[:p_tag_end] +
        comment_start +
        document_xml[p_tag_end:p_end] +
        comment_end +
        document_xml[p_end:]
    )
    
    return new_xml, True


def update_content_types(path):
    """Agrega el tipo de contenido para comments.xml."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'comments.xml' not in content:
        insert_pos = content.find('</Types>')
        new_entry = '  <Override PartName="/word/comments.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>\n'
        content = content[:insert_pos] + new_entry + content[insert_pos:]
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)


def update_document_rels(path):
    """Agrega la relación para comments.xml."""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'comments.xml' not in content:
        # Encontrar el mayor rId
        rids = re.findall(r'Id="rId(\d+)"', content)
        max_rid = max([int(r) for r in rids]) if rids else 0
        new_rid = max_rid + 1
        
        insert_pos = content.find('</Relationships>')
        new_rel = f'  <Relationship Id="rId{new_rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/>\n'
        content = content[:insert_pos] + new_rel + content[insert_pos:]
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)


def main():
    print("=" * 70)
    print("VINCULANDO COMENTARIOS A PÁRRAFOS ESPECÍFICOS")
    print("=" * 70)
    
    # Limpiar y crear directorio temporal
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    # 1. Extraer docx
    print(f"\n[1/6] Extrayendo {SRC_DOCX}...")
    with zipfile.ZipFile(SRC_DOCX, 'r') as z:
        z.extractall(TEMP_DIR)
    
    # 2. Leer document.xml
    print("[2/6] Leyendo document.xml...")
    doc_path = os.path.join(TEMP_DIR, 'word', 'document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        document_xml = f.read()
    
    # 3. Insertar marcadores de comentarios
    print("[3/6] Insertando marcadores de comentarios...")
    resultados = []
    
    for com_id, _, search_text, comment_text in COMENTARIOS:
        # Intentar insertar el comentario
        new_xml, success = insert_comment_at_paragraph(document_xml, com_id, search_text)
        
        if success:
            document_xml = new_xml
            status = "✓"
        else:
            status = "✗"
        
        # Mostrar resultado truncado
        search_display = search_text[:35] + "..." if len(search_text) > 35 else search_text
        resultados.append((com_id, status, search_display))
        print(f"   [{status}] ID={com_id}: '{search_display}'")
    
    # 4. Guardar document.xml modificado
    print("\n[4/6] Guardando document.xml modificado...")
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(document_xml)
    
    # 5. Crear comments.xml
    print("[5/6] Creando comments.xml...")
    comments_path = os.path.join(TEMP_DIR, 'word', 'comments.xml')
    with open(comments_path, 'w', encoding='utf-8') as f:
        f.write(create_comments_xml())
    
    # 6. Actualizar archivos de soporte
    print("[6/6] Actualizando [Content_Types].xml y relaciones...")
    update_content_types(os.path.join(TEMP_DIR, '[Content_Types].xml'))
    update_document_rels(os.path.join(TEMP_DIR, 'word', '_rels', 'document.xml.rels'))
    
    # 7. Reempaquetar
    print(f"\n[7/7] Creando documento final: {OUTPUT_DOCX}...")
    if os.path.exists(OUTPUT_DOCX):
        os.remove(OUTPUT_DOCX)
    
    with zipfile.ZipFile(OUTPUT_DOCX, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(TEMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, TEMP_DIR)
                z.write(file_path, arcname)
    
    # Limpiar
    shutil.rmtree(TEMP_DIR)
    
    # Resumen
    exitos = sum(1 for _, s, _ in resultados if s == "✓")
    fallos = sum(1 for _, s, _ in resultados if s == "✗")
    
    print("\n" + "=" * 70)
    print("COMPLETADO")
    print("=" * 70)
    print(f"Archivo generado: {OUTPUT_DOCX}")
    print(f"Comentarios vinculados: {exitos}/{len(COMENTARIOS)}")
    if fallos > 0:
        print(f"No encontrados: {fallos} (texto no existe en documento)")
    print("\nFechas asignadas: 8-12 de diciembre de 2025")
    print("=" * 70)


if __name__ == '__main__':
    main()

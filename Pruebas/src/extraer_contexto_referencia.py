import os
import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(path):
    text = []
    try:
        # Intento básico con zipfile (funciona sin librerías externas)
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            # Namespace usual de Word
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Buscar cuerpos de texto
            for node in tree.iter():
                if node.tag.endswith('}t'): # Text nodes usually w:t
                    if node.text:
                        text.append(node.text)
                elif node.tag.endswith('}p'): # Paragraph breaks
                    text.append('\n')
    except Exception as e:
        return f"Error leyendo DOCX: {e}"
    return "".join(text)

def extract_text_from_pptx(path):
    text = []
    try:
        with zipfile.ZipFile(path) as z:
            # Listar slides
            slides = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
            # Ordenar slides numéricamente si es posible (slide1, slide2...)
            slides.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            
            for slide in slides:
                text.append(f"\n--- DIAPOSITIVA {slide} ---\n")
                xml_content = z.read(slide)
                tree = ET.fromstring(xml_content)
                for node in tree.iter():
                    if node.tag.endswith('}t'): # Text nodes usually a:t
                        if node.text:
                            text.append(node.text + "\n")
    except Exception as e:
        return f"Error leyendo PPTX: {e}"
    return "".join(text)

base_dir = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\Archivos_referencia"
docx_file = os.path.join(base_dir, "TESIS DOCTORAL 2025 SANTANA-RAMIREZ.docx")
pptx_file = os.path.join(base_dir, "Presentacion_avances_semestre_3.pptx")

print("=== CONTENIDO TESIS (EXTRACTO ESTRUCTURA) ===")
if os.path.exists(docx_file):
    full_text = extract_text_from_docx(docx_file)
    # Filtrar lineas vacias y mostrar las primeras lineas y estructura probable
    lines = [l.strip() for l in full_text.split('\n') if l.strip()]
    
    # Intentar detectar capitulos o secciones importantes
    print("\n".join(lines[:50])) # Primeras 50 lineas
    print("\n... [Buscando Capítulos] ...\n")
    for line in lines:
        if "CAPÍTULO" in line.upper() or "CHAPTER" in line.upper() or line.startswith("1.") or line.startswith("2."):
            print(line)
else:
    print("Archivo DOCX no encontrado")

print("\n\n=== CONTENIDO PRESENTACION ===")
if os.path.exists(pptx_file):
    pptx_text = extract_text_from_pptx(pptx_file)
    print(pptx_text[:2000]) # Primeros 2000 caracteres
else:
    print("Archivo PPTX no encontrado")

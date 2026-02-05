"""
Script mejorado para vincular comentarios detallados a párrafos específicos en Word.
Autor: José de Jesús Santana Ramírez
Basado en documento de revisión experta del sistema de monitoreo térmico plantar.
"""

import zipfile
import os
import shutil
import re

# Configuración
SRC_DOCX = r'Pruebas/src/t_word_Frida_Andrade.docx'
OUTPUT_DOCX = r'Pruebas/src/t_word_Frida_Andrade_REVISADO.docx'
TEMP_DIR = r'Pruebas/src/t_word_temp'
AUTHOR = "José de Jesús Santana Ramírez"
AUTHOR_EMAIL = "jesus.santana@uaq.mx"

# Fechas de la semana del 8-12 de diciembre de 2025
DATES = [
    "2025-12-08T09:30:00Z",
    "2025-12-08T11:45:00Z",
    "2025-12-08T14:15:00Z",
    "2025-12-09T10:00:00Z",
    "2025-12-09T14:30:00Z",
    "2025-12-09T16:45:00Z",
    "2025-12-10T09:15:00Z",
    "2025-12-10T11:20:00Z",
    "2025-12-10T15:00:00Z",
    "2025-12-11T09:00:00Z",
    "2025-12-11T11:30:00Z",
    "2025-12-11T15:30:00Z",
    "2025-12-12T09:45:00Z",
    "2025-12-12T11:00:00Z",
    "2025-12-12T14:00:00Z",
    "2025-12-12T16:30:00Z",
    "2025-12-12T17:30:00Z",
]

# Comentarios expandidos basados en la revisión de experto
# (id, fecha_idx, texto_a_buscar, comentario)
COMENTARIOS = [
    # === INTRODUCCIÓN Y MARCO TEÓRICO ===
    (0, 0, "Introducción", 
     "EXTENSIÓN REQUERIDA: La justificación médica debe ser más granular. Incorporar mecanismos hemodinámicos exactos: "
     "la elevación de temperatura NO se debe únicamente a inflamación tisular, sino fundamentalmente a la NEUROPATÍA AUTONÓMICA. "
     "La denervación simpática provoca pérdida del tono vasomotor, abriendo shunts arteriovenosos que causan hiperemia termogénica. "
     "El 'pie caliente' neuropático es un predictor temprano de riesgo. "
     "Ref: Armstrong DG et al. (2017) NEJM; Lavery LA et al. (2007) Diabetes Care."),
    
    (1, 1, "Antecedentes",
     "AGREGAR TABLA COMPARATIVA del estado del arte (2020-2025) con columnas: Material del Sensor, Integración Electrónica, "
     "Algoritmo de Detección, Validación Clínica, Fabricación. Contrastar EeonTex vs compuestos nano-híbridos (Grafeno, MXenos). "
     "Mencionar tendencias: TinyML en el borde, tejido 3D sin costuras (Intarsia knitting), protocolos Matter/Thread para IoT doméstico. "
     "Ref: Investigaciones de Donghua University, Georgia Tech (2022-2024)."),
    
    (2, 2, "diabetes",
     "AGREGAR SECCIÓN: 'Fisiopatología de la Termorregulación en el Pie Neuropático' que incluya: "
     "(1) La Tríada de la Termogénesis Patológica: inflamación subclínica, hiperemia neurogénica, anhidrosis. "
     "(2) Umbrales Clínicos Actualizados: estudios 2022-2024 sugieren algoritmos personalizados que analizan "
     "la VARIABILIDAD longitudinal, no solo el umbral estático de 2.2°C. "
     "(3) Limitaciones de la asimetría contralateral en amputados o enfermedad vascular bilateral."),

    # === UBICACIÓN DE SENSORES ===
    (3, 3, "puntos estratégicos",
     "REHACER con base en ANGIOSOMAS: El pie está dividido en regiones irrigadas por arterias fuente específicas. "
     "Sensor 1 (Hallux): A. Tibial Anterior - zona de cizallamiento en toe-off. "
     "Sensores 2-4 (Metatarsos): A. Plantar Medial/Lateral - máxima presión vertical. "
     "Sensor 5 (Arco): A. Plantar Medial - crítico para Charcot activo. "
     "Sensor 6 (Talón): A. Calcaneal - úlceras por decúbito. "
     "Esto eleva el dispositivo de 'monitor de úlceras' a 'herramienta de vigilancia vascular integral'. "
     "Ref: Bus SA et al. (2020) Diabetes Metab Res Rev."),

    # === MATERIAL DEL SENSOR - CRÍTICA FUNDAMENTAL ===
    (4, 4, "NW170-PI",
     "⚠️ FALLO CRÍTICO IDENTIFICADO: El EeonTex es INHERENTEMENTE PIEZORRESISTIVO. "
     "La resistencia depende de rutas de percolación entre fibras. Tanto el aumento de temperatura (inflamación) "
     "como el aumento de presión (caminar) provocan el MISMO EFECTO: caída de resistencia. "
     "PREGUNTA SIN RESOLVER: ¿Cómo sabe el MCU si R bajó porque el pie está a 38°C o porque el paciente de 90kg está de pie? "
     "SOLUCIONES OBLIGATORIAS: (A) Integrar FSR para 'gating' - medir solo cuando presión=0. "
     "(B) Caracterización diferencial: superficie de calibración 3D Z=f(T,P). "
     "(C) Sensor de referencia en zona sin carga (tobillo)."),

    (5, 5, "cuadrados",
     "FORMALIZAR HALLAZGO con teoría electromagnética: En material anisotrópico (fibras con dirección preferencial de cardado), "
     "la densidad de corriente no es uniforme. En esquinas del cuadrado hay acumulación de líneas de campo (efecto de borde). "
     "La geometría circular homogeneiza el campo radialmente, promediando anisotropías. "
     "AGREGAR: Diagramas de líneas equipotenciales para elevar nivel científico."),

    # === AD5934 Y FRECUENCIA ===
    (6, 6, "AD5934",
     "ERROR TÉCNICO: El AD5934 NO es un simple ADC. Es un ANALIZADOR DE IMPEDANCIA que integra: "
     "generador DDS, DAC y ADC de 12 bits para medir impedancia compleja (magnitud y fase). "
     "PROFUNDIZAR la selección de 90kHz: La interfaz conductor-piel genera una Doble Capa Eléctrica (capacitor Cdl). "
     "A 10Hz: Xc enorme → medición dominada por capacitancia de interfaz → alto error. "
     "A 90kHz: Xc pequeña → capacitor es cortocircuito AC → domina parte resistiva (que varía con T). "
     "AGREGAR: Diagrama de circuito equivalente (Modelo de Randles modificado)."),

    (7, 7, "CN0349",
     "EXTENSIÓN TÉCNICA: Explicar arquitectura de acondicionamiento que minimiza errores por polarización y por "
     "impedancia de salida del generador. Discutir RFB/RCAL conmutables. "
     "AGREGAR: Protocolo de calibración de dos o tres puntos documentado para integración en firmware."),

    (8, 8, "frecuencia",
     "INSUFICIENTE decir 'dio menos error'. Explicar la FÍSICA: "
     "Impedancia total Z = R_sensor + Xc, donde Xc = 1/(2πfC). "
     "A baja f: Xc domina (cambios de contacto alteran lectura). "
     "A alta f: Xc→0, queda R que varía con temperatura. "
     "Proporcionar ecuaciones y modelo de Randles."),

    # === CARACTERIZACIÓN ===
    (9, 9, "pistola de calor",
     "ESPECIFICAR: Marca/modelo de pistola de calor, modelo de cámara FLIR (resolución, precisión ±X°C), "
     "tiempo de estabilización térmica entre mediciones. "
     "AGREGAR EXPERIMENTO CRUCIAL: Histéresis térmica - graficar 'Lazo de Histéresis' (curva subida 25→40°C vs bajada 40→25°C). "
     "El área del lazo = error por memoria térmica del material. Cuantificar precisión real (±0.5°C o ±2.0°C?)."),

    (10, 10, "Caracterización",
     "ANÁLISIS CUANTITATIVO FALTANTE - Para cada sensor calcular: "
     "(1) Sensibilidad ΔZ/ΔT [Ω/°C], (2) Linealidad R², (3) Repetibilidad σ. "
     "MODELO FÍSICO: Ajustar datos a ecuación tipo Arrhenius/Mott para polímeros conductores: "
     "R(T) = R₀·exp[(T₀/T)^γ] donde γ depende de dimensionalidad. "
     "Esto demuestra entendimiento de ciencia de materiales vs regresión polinómica empírica."),

    # === SISTEMA EMBEBIDO ===
    (11, 11, "ESP32",
     "CONSUMO ENERGÉTICO EXCESIVO para wearable comercial (80-240mA en TX). "
     "Estado del arte 2020-2025: Migrar a nRF52/nRF53 (Nordic) o ESP32-C3/S3 (RISC-V) con mejor gestión energética. "
     "Si se mantiene ESP32, DETALLAR estrategia de Duty Cycling: "
     "- Despertar cada 5 min (no 10 seg), medir <1s, buffer local. "
     "- TX por BLE solo 1x/hora o si anomalía crítica. "
     "- Esto extiende batería de horas a SEMANAS."),

    (12, 11, "filtrado digital",
     "MUY VAGO - Especificar: Tipo de filtro (FIR/IIR), orden, frecuencia de corte. "
     "RECOMENDACIÓN AVANZADA: Implementar FILTRO DE KALMAN unidimensional. "
     "Predice estado futuro basándose en estado previo, corrige con nueva medición ponderando incertidumbre. "
     "Más efectivo que paso-bajo convencional para suavizar señales textiles ruidosas sin lag excesivo."),

    (13, 12, "BLE",
     "REQUISITO FALTANTE - SEGURIDAD IoMT (Internet of Medical Things): "
     "(1) Emparejamiento Seguro (LE Secure Connections). "
     "(2) Encriptación de carga útil (AES-128). "
     "(3) Anonimización: el calcetín NO debe transmitir nombre del paciente, solo UUID que la App resuelve. "
     "Mencionar protocolo Matter/Thread para integración en ecosistema Smart Home."),

    # === APLICACIÓN MÓVIL ===
    (14, 13, "Aplicación móvil",
     "SECCIÓN INCOMPLETA - Agregar: capturas de pantalla, descripción de funcionalidades, protocolo BLE (UUIDs), tasa de TX. "
     "CRÍTICA UX: Las gráficas de impedancia son útiles para el ingeniero pero INÚTILES para el paciente (adulto mayor con retinopatía). "
     "REDISEÑAR con interfaz de SEMÁFOROS: "
     "🟢 Verde: 'Pies sanos'. "
     "🟡 Amarillo: 'Diferencia detectada. Revise calzado'. "
     "🔴 Rojo: 'Alerta. Contacte podólogo'. "
     "Incluir notificaciones por voz (TTS) para discapacidad visual."),

    # === RESULTADOS ===
    (15, 14, "Resultados",
     "SECCIÓN FALTANTE: Agregar DISCUSIÓN que compare con trabajos previos, analice limitaciones y fuentes de error. "
     "Discutir: (1) Sensibilidad cruzada presión-temperatura no resuelta. "
     "(2) Falta de validación de lavabilidad. "
     "(3) Muestra pequeña (6 sensores, 0 pacientes reales)."),

    # === CONCLUSIONES ===
    (16, 15, "Conclusión",
     "⚠️ REESCRIBIR COMPLETAMENTE - Las conclusiones actuales son notas/pendientes, no conclusiones formales. "
     "Deben responder a objetivos con resultados concretos: "
     "'Se logró sensibilidad de X.X Ω/°C con linealidad R²>0.XX a 90kHz'. "
     "'El sistema detecta diferenciales térmicos >2.2°C con precisión de ±Y°C'. "
     "Estructura: (1) Logros vs objetivos, (2) Contribución científica, (3) Limitaciones, (4) Trabajo futuro."),

    (17, 15, "prototipo",
     "CONVERTIR pendientes en sección 'TRABAJO FUTURO' estructurada con 3 fases: "
     "FASE 1 (Lab): Caracterización electromecánica - matriz Z=f(T,P) con Instron + cámara climática. "
     "FASE 2 (Estándar): Lavabilidad AATCC 135 - 10/20/30 ciclos, criterio ΔR₀<10% tras 20 lavados. "
     "FASE 3 (Clínico): Piloto n≥30 pacientes - reposo vs marcha, inducción térmica artificial, encuesta Likert de confort."),

    (18, 16, "lavable",
     "INACEPTABLE como 'trabajo futuro' - debe ser parte del diseño. "
     "TÉCNICAS DE PROTECCIÓN: "
     "(1) Encapsulamiento con TPU (impermeable pero permite transferencia térmica). "
     "(2) Conectores removibles: módulo 'pod' con broches magnéticos para separar electrónica antes de lavar. "
     "Ref: Norma AATCC 135 para textiles."),

    # === ORTOGRAFÍA ===
    (19, 16, "se diseño",
     "CORRECCIÓN ORTOGRÁFICA: 'se diseño' → 'se diseñó' (falta tilde en pretérito)."),

    (20, 16, "rehusables",
     "CORRECCIÓN: 'rehusables' → 'reutilizables'."),

    (21, 16, "mas eficiente",
     "CORRECCIÓN: 'mas eficiente' → 'más eficiente' (falta tilde)."),

    # === BIBLIOGRAFÍA ===
    (22, 16, "Bibliografía",
     "ACTUALIZAR con referencias 2020-2025. Mínimo 25-30 referencias. Agregar: "
     "- Armstrong DG et al. (2017) Diabetic foot ulcers. NEJM. "
     "- Lavery LA et al. (2007) Preventing recurrence. Diabetes Care. "
     "- Bus SA et al. (2020) Guidelines offloading. DMRR. "
     "- Investigaciones Donghua University, Georgia Tech (2022-2024) sobre e-textiles con grafeno/MXenos. "
     "Verificar formato consistente (IEEE o APA)."),
    
    # === FIGURAS DE LA APLICACIÓN MÓVIL (Páginas 38-41) ===
    (23, 13, "Error de medición",
     "FIGURA - Tabla y Gráfica de Error: Agregar título descriptivo. "
     "Sugerencia: 'Tabla X. Comparación del error relativo de medición de impedancia a tres frecuencias de excitación "
     "(10 Hz, 50 kHz, 90 kHz) con resistencia de calibración de 1 kΩ.' "
     "Para la gráfica: 'Figura X. Error relativo vs resistencia (100 Ω - 3.3 kΩ): validación de la selección de 90 kHz "
     "como frecuencia óptima de operación para el rango de impedancia de los sensores textiles.'"),

    (24, 13, "Información personal",
     "FIGURA - Pantalla de Información Personal: Agregar título y descripción. "
     "Sugerencia: 'Figura X. Pantalla de registro de usuario en la aplicación Smart Socks. "
     "Permite capturar datos demográficos del paciente (nombre, edad, teléfono, correo) y contactos de emergencia "
     "para notificación automática en caso de alerta térmica crítica.' "
     "OBSERVACIÓN: Verificar cumplimiento de LFPDPPP (Ley Federal de Protección de Datos Personales) "
     "y considerar encriptación local de datos sensibles."),

    (25, 14, "Smart Socks",
     "FIGURAS - Pantallas de Visualización Plantar: Requieren títulos descriptivos. "
     "Sugerencia izquierda: 'Figura X. Interfaz de mapeo plantar en estado de espera mostrando la distribución "
     "de los 12 sensores (S1-S12) sobre la representación anatómica de ambos pies.' "
     "Sugerencia derecha: 'Figura X. Visualización en tiempo real de temperaturas plantares con código de colores: "
     "verde (23-25°C, normal), amarillo (25-26°C, precaución), rojo (>35°C, alerta de riesgo). "
     "Se observa anomalía térmica en hallux izquierdo (38°C) indicando posible proceso inflamatorio pre-ulcerativo.'"),

    (26, 14, "Editar mi información",
     "FIGURA - Formulario de Edición: Agregar título. "
     "Sugerencia: 'Figura X. Interfaz de edición de datos personales con validación de campos obligatorios "
     "y formato de entrada para información de contacto de emergencia.' "
     "MEJORA UX: Considerar campos de autocompletado y validación en tiempo real para números telefónicos."),

    (27, 14, "Contactos de Emergencia",
     "FIGURA - Registro de Contactos: Agregar título descriptivo. "
     "Sugerencia: 'Figura X. Configuración de contactos de emergencia (hasta 2) para notificación automática "
     "vía SMS/llamada cuando se detecte alerta de riesgo térmico persistente (>2.2°C por >24 horas).' "
     "EXTENSIÓN: Describir el flujo de notificación y los criterios de escalamiento de alertas."),

    (28, 15, "Bluetooth apagado",
     "FIGURA - Pantalla de Conexión BLE: Agregar título. "
     "Sugerencia: 'Figura X. Gestión de conectividad Bluetooth Low Energy (BLE) entre la aplicación móvil "
     "y el módulo ESP32 del calcetín inteligente. La conexión se establece mediante emparejamiento seguro "
     "con UUID específico del dispositivo.' "
     "SEGURIDAD: Documentar el protocolo de emparejamiento y encriptación utilizado (LE Secure Connections, AES-128)."),

    (29, 15, "Anomalía",
     "FIGURA - Historial de Anomalías: Agregar título y explicación del formato. "
     "Sugerencia: 'Figura X. Registro histórico de eventos de anomalía térmica detectados por el sistema. "
     "Cada entrada incluye: número de evento, fecha/hora de detección, y temperatura diferencial registrada. "
     "Este historial permite al profesional de salud evaluar patrones longitudinales de riesgo.' "
     "EXTENSIÓN: Agregar funcionalidad de exportación a CSV/PDF para compartir con el médico tratante. "
     "Incluir gráfica de tendencia temporal y correlación con actividad física (pasos/día)."),

    (30, 15, "Historial",
     "FIGURA - Vista de Historial: Agregar descripción de la información mostrada. "
     "Sugerencia: 'Figura X. Interfaz de consulta de historial con lista desplegable de anomalías ordenadas "
     "cronológicamente. El usuario puede expandir cada evento para ver detalles: zona afectada, "
     "temperatura máxima alcanzada, duración del evento, y acción recomendada.' "
     "MEJORA: Implementar filtros por rango de fechas, severidad, y zona anatómica afectada."),
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
    """Crea el archivo comments.xml con comentarios detallados."""
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"
            xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml"
            xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
'''
    for com_id, date_idx, _, texto in COMENTARIOS:
        fecha = DATES[date_idx % len(DATES)]
        texto_escaped = escape_xml(texto)
        xml += f'''    <w:comment w:id="{com_id}" w:author="{AUTHOR}" w:date="{fecha}" w:initials="JJSR">
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


def insert_comment_at_paragraph(document_xml, comment_id, search_text):
    """Inserta marcadores de comentario en el párrafo que contiene el texto."""
    search_lower = search_text.lower()
    pos = document_xml.lower().find(search_lower)
    
    if pos == -1:
        return document_xml, False
    
    p_start = document_xml.rfind('<w:p ', 0, pos)
    if p_start == -1:
        p_start = document_xml.rfind('<w:p>', 0, pos)
    
    if p_start == -1:
        return document_xml, False
    
    p_tag_end = document_xml.find('>', p_start) + 1
    comment_start = f'<w:commentRangeStart w:id="{comment_id}"/>'
    
    p_end = document_xml.find('</w:p>', pos)
    if p_end == -1:
        return document_xml, False
    
    comment_end = (
        f'<w:commentRangeEnd w:id="{comment_id}"/>'
        f'<w:r><w:rPr><w:rStyle w:val="CommentReference"/></w:rPr>'
        f'<w:commentReference w:id="{comment_id}"/></w:r>'
    )
    
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
    print("REVISIÓN TÉCNICA DE TESIS - SISTEMA MONITOREO TÉRMICO PLANTAR")
    print(f"Revisor: {AUTHOR}")
    print("=" * 70)
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    print(f"\n[1/6] Extrayendo {SRC_DOCX}...")
    with zipfile.ZipFile(SRC_DOCX, 'r') as z:
        z.extractall(TEMP_DIR)
    
    print("[2/6] Leyendo document.xml...")
    doc_path = os.path.join(TEMP_DIR, 'word', 'document.xml')
    with open(doc_path, 'r', encoding='utf-8') as f:
        document_xml = f.read()
    
    print(f"[3/6] Insertando {len(COMENTARIOS)} comentarios de revisión técnica...")
    resultados = []
    
    for com_id, _, search_text, _ in COMENTARIOS:
        new_xml, success = insert_comment_at_paragraph(document_xml, com_id, search_text)
        
        if success:
            document_xml = new_xml
            status = "✓"
        else:
            status = "✗"
        
        search_display = search_text[:30] + "..." if len(search_text) > 30 else search_text
        resultados.append((com_id, status, search_display))
        print(f"   [{status}] ID={com_id:2d}: '{search_display}'")
    
    print("\n[4/6] Guardando document.xml modificado...")
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(document_xml)
    
    print("[5/6] Creando comments.xml...")
    comments_path = os.path.join(TEMP_DIR, 'word', 'comments.xml')
    with open(comments_path, 'w', encoding='utf-8') as f:
        f.write(create_comments_xml())
    
    print("[6/6] Actualizando metadatos...")
    update_content_types(os.path.join(TEMP_DIR, '[Content_Types].xml'))
    update_document_rels(os.path.join(TEMP_DIR, 'word', '_rels', 'document.xml.rels'))
    
    print(f"\n[7/7] Creando documento final...")
    if os.path.exists(OUTPUT_DOCX):
        os.remove(OUTPUT_DOCX)
    
    with zipfile.ZipFile(OUTPUT_DOCX, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(TEMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, TEMP_DIR)
                z.write(file_path, arcname)
    
    shutil.rmtree(TEMP_DIR)
    
    exitos = sum(1 for _, s, _ in resultados if s == "✓")
    fallos = sum(1 for _, s, _ in resultados if s == "✗")
    
    print("\n" + "=" * 70)
    print("REVISIÓN COMPLETADA")
    print("=" * 70)
    print(f"Archivo: {OUTPUT_DOCX}")
    print(f"Autor: {AUTHOR}")
    print(f"Comentarios vinculados: {exitos}/{len(COMENTARIOS)}")
    if fallos > 0:
        print(f"No encontrados: {fallos}")
    print(f"Fechas: 8-12 diciembre 2025")
    print("=" * 70)


if __name__ == '__main__':
    main()

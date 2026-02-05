"""
Script para extraer datos de levitación controlada de TODOS los experimentos.
Criterios de control:
  - Posición y dentro de rango válido (1-20mm)
  - Corriente i > 0 (control activo)
  - Error |y - sp| < umbral (siguiendo setpoint)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Directorios a buscar
BASE_DIR = Path(r"c:/Users/jesus/Documents/Doctorado/Experimentos/CRio DAQ/cDAQ_9174")
OUTPUT_DIR = BASE_DIR / "levitador-benchmark" / "data"

# Archivos de datos conocidos
DATA_SOURCES = {
    # levitador-benchmark/data/
    'MONIT_MAESTRO': 'levitador-benchmark/data/MONIT_MAESTRO.txt',
    'MONIT_SWEEP': 'levitador-benchmark/data/MONIT_SWEEP.txt',
    'datos_levitador': 'levitador-benchmark/data/datos_levitador.txt',
    'datos_zonas_faltantes': 'levitador-benchmark/data/datos_zonas_faltantes.txt',
    # sesiones_kan_pinn
    'chirp': 'levitador-benchmark/data/sesiones_kan_pinn/dataset_chirp_20251217_210058.txt',
    'constante': 'levitador-benchmark/data/sesiones_kan_pinn/dataset_constante_20251217_205611.txt',
    'escalon': 'levitador-benchmark/data/sesiones_kan_pinn/dataset_escalon_20251217_205858.txt',
    'multiescalon': 'levitador-benchmark/data/sesiones_kan_pinn/dataset_multiescalon_20251217_210203.txt',
    'senoidal': 'levitador-benchmark/data/sesiones_kan_pinn/dataset_senoidal_20251217_205952.txt',
    # levitador valentin
    'valentin_sweep': 'levitador valentin/MONIT_SWEEP.txt',
    'valentin_sine': 'levitador valentin/MONIT_SINE.txt',
    'valentin_v3': 'levitador valentin/MONIT_V3.txt',
    # Root
    'MONIT_root': 'MONIT.txt',
    'MONIT_pid_obs': 'MONIT_pid_obs.txt',
    'MONIT_observador': 'MONIT_observador.txt',
    # levitador-python
    'datos_levitacion_py': 'levitador-python/datos_levitacion.csv',
    'datos_control_C': 'levitador-python/datos_control_C.csv',
}

def load_data_flexible(filepath):
    """Carga datos con detección automática de formato"""
    try:
        # Intentar diferentes separadores y encodings
        for sep in ['\t', r'\s+', ',', ';']:
            for enc in ['utf-8', 'latin1', 'cp1252']:
                try:
                    # Saltar líneas de comentario
                    df = pd.read_csv(filepath, sep=sep, comment='#', header=None, 
                                    encoding=enc, on_bad_lines='skip')
                    if len(df.columns) >= 5 and len(df) > 10:
                        # Detectar formato basado en valores
                        # Formato sesiones_kan_pinn: [t, y, sp, ?, i, u, ?] - col1 varía, col2 constante
                        # Formato estándar: [t, sp, y, i, u] - col1 constante (sp), col2 varía (y)
                        
                        col1_std = df[1].std()
                        col2_std = df[2].std()
                        
                        if col1_std > col2_std * 2:  # col1 varía más -> formato kan_pinn
                            # [t, y, sp, ?, i, u, ?]
                            result = pd.DataFrame({
                                't': df[0],
                                'y': df[1],
                                'sp': df[2],
                                'i': df[4],
                                'u': df[5]
                            })
                        else:  # formato estándar
                            # [t, sp, y, i, u]
                            result = pd.DataFrame({
                                't': df[0],
                                'sp': df[1],
                                'y': df[2],
                                'i': df[3],
                                'u': df[4]
                            })
                        
                        # Convertir a numérico
                        for col in ['t', 'sp', 'y', 'i', 'u']:
                            result[col] = pd.to_numeric(result[col], errors='coerce')
                        result = result.dropna(subset=['t', 'sp', 'y', 'i', 'u'])
                        
                        if len(result) > 0:
                            return result
                except:
                    continue
    except Exception as e:
        print(f"  Error: {e}")
    return None

def is_controlled(df, y_min=0.001, y_max=0.025, i_min=0.01, error_max=0.005):
    """
    Determina qué muestras están bajo control activo.
    Retorna máscara booleana.
    """
    y = df['y'].values
    i = df['i'].values
    sp = df['sp'].values
    
    # Criterios:
    # 1. Posición dentro de rango físico (1-25mm)
    valid_y = (y >= y_min) & (y <= y_max)
    
    # 2. Corriente activa (control encendido)
    valid_i = np.abs(i) > i_min
    
    # 3. Error de seguimiento aceptable
    error = np.abs(y - sp)
    valid_error = error < error_max
    
    # 4. No hay saltos bruscos (derivada de posición razonable)
    dy = np.abs(np.diff(y, prepend=y[0]))
    valid_dy = dy < 0.005  # < 5mm entre muestras
    
    mask = valid_y & valid_i & valid_error & valid_dy
    return mask

def extract_controlled_segments(df, min_segment_length=50):
    """Extrae segmentos continuos de datos controlados"""
    mask = is_controlled(df)
    
    # Encontrar segmentos continuos
    segments = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif not val and start is not None:
            if i - start >= min_segment_length:
                segments.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_segment_length:
        segments.append((start, len(mask)))
    
    return segments, mask

def main():
    print("=" * 70)
    print("EXTRACCIÓN DE DATOS DE LEVITACIÓN CONTROLADA")
    print("=" * 70)
    
    all_controlled = []
    stats = []
    
    for name, rel_path in DATA_SOURCES.items():
        filepath = BASE_DIR / rel_path
        print(f"\n📂 {name}: ", end="")
        
        if not filepath.exists():
            print("❌ No encontrado")
            continue
        
        df = load_data_flexible(filepath)
        if df is None or len(df) < 10:
            print("⚠️ No se pudo cargar")
            continue
        
        segments, mask = extract_controlled_segments(df)
        controlled_pct = 100 * mask.sum() / len(mask)
        
        print(f"✅ {len(df)} muestras, {mask.sum()} controladas ({controlled_pct:.1f}%)")
        
        # Extraer datos controlados
        for seg_start, seg_end in segments:
            seg_df = df.iloc[seg_start:seg_end].copy()
            seg_df['source'] = name
            all_controlled.append(seg_df)
            print(f"   └─ Segmento: {seg_end - seg_start} muestras ({seg_start}-{seg_end})")
        
        stats.append({
            'name': name,
            'total': len(df),
            'controlled': mask.sum(),
            'pct': controlled_pct,
            'segments': len(segments)
        })
    
    # Combinar todos los datos controlados
    if all_controlled:
        mega_df = pd.concat(all_controlled, ignore_index=True)
        
        # Resetear tiempo
        mega_df['t'] = np.arange(len(mega_df)) * 0.01  # Asumir Ts=10ms
        
        print("\n" + "=" * 70)
        print("RESUMEN")
        print("=" * 70)
        print(f"Total muestras controladas: {len(mega_df)}")
        print(f"Duración total: {len(mega_df) * 0.01:.1f} segundos")
        print(f"Fuentes utilizadas: {mega_df['source'].nunique()}")
        
        # Estadísticas por fuente
        print("\n📊 Por fuente:")
        for src in mega_df['source'].unique():
            n = (mega_df['source'] == src).sum()
            print(f"   {src}: {n} muestras ({100*n/len(mega_df):.1f}%)")
        
        # Guardar dataset
        output_file = OUTPUT_DIR / "mega_controlled_dataset.txt"
        mega_df[['t', 'sp', 'y', 'i', 'u']].to_csv(
            output_file, sep='\t', index=False, header=False, float_format='%.6f'
        )
        print(f"\n💾 Dataset guardado: {output_file}")
        
        # Visualización
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        
        axes[0, 0].plot(mega_df['t'], mega_df['y'] * 1000, 'b-', lw=0.5, alpha=0.7)
        axes[0, 0].plot(mega_df['t'], mega_df['sp'] * 1000, 'r--', lw=1, label='Setpoint')
        axes[0, 0].set_xlabel('t [s]')
        axes[0, 0].set_ylabel('Posición [mm]')
        axes[0, 0].set_title('Posición vs Setpoint (datos controlados)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(mega_df['t'], mega_df['i'], 'g-', lw=0.5)
        axes[0, 1].set_xlabel('t [s]')
        axes[0, 1].set_ylabel('Corriente [A]')
        axes[0, 1].set_title('Corriente de control')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(mega_df['t'], mega_df['u'], 'm-', lw=0.5)
        axes[1, 0].set_xlabel('t [s]')
        axes[1, 0].set_ylabel('Voltaje [V]')
        axes[1, 0].set_title('Voltaje aplicado')
        axes[1, 0].grid(True)
        
        # Histograma de posiciones
        axes[1, 1].hist(mega_df['y'] * 1000, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Posición [mm]')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de posiciones')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'mega_controlled_analysis.png', dpi=150)
        plt.show()
        
        return mega_df
    else:
        print("\n⚠️ No se encontraron datos controlados!")
        return None

if __name__ == "__main__":
    df = main()

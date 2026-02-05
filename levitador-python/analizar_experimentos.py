"""
Análisis de Experimentos del Levitador Magnético
=================================================

Genera gráficas y métricas de los experimentos realizados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.optimize import curve_fit

# Directorio de experimentos
EXP_DIR = Path(__file__).parent / 'experimentos'

def cargar_experimento(filepath):
    """Carga datos de un experimento."""
    df = pd.read_csv(filepath, sep='\t', comment='#', header=None,
                     names=['t', 'yd', 'y', 'id', 'ie', 'u'])
    return df

def analizar_escalon(df):
    """Analiza respuesta al escalón."""
    t = df['t'].values
    y = df['y'].values * 1000  # mm
    yd = df['yd'].values * 1000
    
    # Detectar cambio de referencia
    dyd = np.diff(yd)
    idx_cambio = np.where(np.abs(dyd) > 0.01)[0]
    
    if len(idx_cambio) == 0:
        return None
    
    idx_start = idx_cambio[0]
    y_inicial = y[idx_start]
    y_final = yd[-1]
    delta_y = y_final - y_inicial
    
    metricas = {}
    
    if abs(delta_y) > 0.01:
        # Tiempo de subida (10% a 90%)
        y_10 = y_inicial + 0.1 * delta_y
        y_90 = y_inicial + 0.9 * delta_y
        
        try:
            if delta_y > 0:
                idx_10 = np.where(y[idx_start:] >= y_10)[0][0] + idx_start
                idx_90 = np.where(y[idx_start:] >= y_90)[0][0] + idx_start
            else:
                idx_10 = np.where(y[idx_start:] <= y_10)[0][0] + idx_start
                idx_90 = np.where(y[idx_start:] <= y_90)[0][0] + idx_start
            metricas['tr'] = t[idx_90] - t[idx_10]
        except:
            metricas['tr'] = np.nan
        
        # Sobreimpulso
        y_post = y[idx_start:]
        if delta_y > 0:
            metricas['Mp'] = max(0, (np.max(y_post) - y_final) / abs(delta_y) * 100)
        else:
            metricas['Mp'] = max(0, (y_final - np.min(y_post)) / abs(delta_y) * 100)
        
        # Tiempo de asentamiento (±2%)
        banda = 0.02 * abs(delta_y)
        dentro_banda = np.abs(y[idx_start:] - y_final) < banda
        if np.any(dentro_banda):
            for i in range(len(dentro_banda) - 1, -1, -1):
                if np.all(dentro_banda[i:]):
                    metricas['ts'] = t[idx_start + i] - t[idx_start]
                    break
        
        # Error en estado estable
        n_ss = int(0.1 * len(y))
        metricas['ess'] = np.mean(np.abs(yd[-n_ss:] - y[-n_ss:]))
    
    return metricas

def analizar_senoidal(df):
    """Analiza respuesta senoidal - ganancia y fase."""
    t = df['t'].values
    y = df['y'].values * 1000
    yd = df['yd'].values * 1000
    
    # Solo parte senoidal (después de t=3s)
    mask = t > 5
    t_sin = t[mask]
    y_sin = y[mask]
    yd_sin = yd[mask]
    
    if len(t_sin) < 100:
        return None
    
    # Amplitudes
    amp_ref = (np.max(yd_sin) - np.min(yd_sin)) / 2
    amp_y = (np.max(y_sin) - np.min(y_sin)) / 2
    
    metricas = {
        'ganancia': amp_y / amp_ref if amp_ref > 0 else np.nan,
        'amp_referencia_mm': amp_ref,
        'amp_respuesta_mm': amp_y,
    }
    
    # Estimar fase usando correlación cruzada
    yd_norm = (yd_sin - np.mean(yd_sin)) / np.std(yd_sin)
    y_norm = (y_sin - np.mean(y_sin)) / np.std(y_sin)
    corr = np.correlate(yd_norm, y_norm, mode='full')
    lag = np.argmax(corr) - len(y_norm) + 1
    Ts = t_sin[1] - t_sin[0]
    metricas['desfase_ms'] = lag * Ts * 1000
    
    return metricas

def analizar_approach(df):
    """Analiza experimento de aproximación."""
    t = df['t'].values
    y = df['y'].values * 1000
    yd = df['yd'].values * 1000
    u = df['u'].values
    
    metricas = {
        'y_max_mm': np.max(y),
        'y_min_mm': np.min(y),
        'y_final_mm': y[-1],
        'u_max_V': np.max(u),
        'u_final_V': u[-1],
    }
    
    # Detectar si hubo pegado (y muy cerca de 0 o u saturado)
    if np.min(y) < 1.5:  # Menos de 1.5mm
        metricas['alerta'] = "⚠️ Posible zona de pegado"
    
    # Error de seguimiento promedio
    error = np.abs(yd - y)
    metricas['error_medio_mm'] = np.mean(error)
    metricas['error_max_mm'] = np.max(error)
    
    return metricas

def graficar_todos(archivos):
    """Genera gráficas de todos los experimentos."""
    fig, axes = plt.subplots(len(archivos), 2, figsize=(14, 4*len(archivos)))
    
    if len(archivos) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, archivo in enumerate(archivos):
        df = cargar_experimento(archivo)
        nombre = archivo.stem.replace('exp_', '').split('_')[0].upper()
        
        t = df['t'].values
        y = df['y'].values * 1000
        yd = df['yd'].values * 1000
        u = df['u'].values
        
        # Gráfica de posición
        axes[idx, 0].plot(t, y, 'b-', label='y medida', linewidth=0.8)
        axes[idx, 0].plot(t, yd, 'r--', label='yd referencia', linewidth=0.8)
        axes[idx, 0].axhline(y=2, color='orange', linestyle=':', alpha=0.7, label='Límite seguro')
        axes[idx, 0].set_ylabel('Posición [mm]')
        axes[idx, 0].set_title(f'{nombre} - Posición')
        axes[idx, 0].legend(loc='upper right', fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Gráfica de control
        axes[idx, 1].plot(t, u, 'g-', linewidth=0.8)
        axes[idx, 1].set_ylabel('Voltaje [V]')
        axes[idx, 1].set_title(f'{nombre} - Control')
        axes[idx, 1].grid(True, alpha=0.3)
        
        if idx == len(archivos) - 1:
            axes[idx, 0].set_xlabel('Tiempo [s]')
            axes[idx, 1].set_xlabel('Tiempo [s]')
    
    plt.tight_layout()
    plt.savefig(EXP_DIR / 'resumen_experimentos.png', dpi=150)
    plt.show()
    
    return fig

def main():
    print("="*60)
    print("📊 ANÁLISIS DE EXPERIMENTOS DEL LEVITADOR")
    print("="*60)
    
    archivos = sorted(EXP_DIR.glob('exp_*.txt'))
    
    if not archivos:
        print("❌ No se encontraron experimentos en", EXP_DIR)
        return
    
    print(f"\n📁 Encontrados {len(archivos)} experimentos:\n")
    
    for archivo in archivos:
        df = cargar_experimento(archivo)
        nombre = archivo.stem.replace('exp_', '').split('_')[0]
        
        print(f"\n{'─'*50}")
        print(f"🔬 {nombre.upper()}")
        print(f"{'─'*50}")
        print(f"   Archivo: {archivo.name}")
        print(f"   Muestras: {len(df)}")
        print(f"   Duración: {df['t'].max():.1f}s")
        
        # Análisis específico
        if 'escalon' in nombre:
            metricas = analizar_escalon(df)
            if metricas:
                print(f"\n   📈 Métricas de respuesta transitoria:")
                if 'tr' in metricas and not np.isnan(metricas['tr']):
                    print(f"      Tiempo de subida (tr): {metricas['tr']*1000:.1f} ms")
                if 'ts' in metricas:
                    print(f"      Tiempo de asentamiento (ts): {metricas['ts']*1000:.1f} ms")
                if 'Mp' in metricas:
                    print(f"      Sobreimpulso (Mp): {metricas['Mp']:.1f}%")
                if 'ess' in metricas:
                    print(f"      Error estado estable: {metricas['ess']:.3f} mm")
        
        elif 'senoidal' in nombre:
            metricas = analizar_senoidal(df)
            if metricas:
                print(f"\n   📈 Métricas de respuesta frecuencial:")
                print(f"      Ganancia: {metricas['ganancia']:.3f}")
                print(f"      Amplitud referencia: {metricas['amp_referencia_mm']:.2f} mm")
                print(f"      Amplitud respuesta: {metricas['amp_respuesta_mm']:.2f} mm")
                print(f"      Desfase: {metricas['desfase_ms']:.1f} ms")
        
        elif 'approach' in nombre:
            metricas = analizar_approach(df)
            if metricas:
                print(f"\n   📈 Métricas de aproximación:")
                print(f"      Posición máxima: {metricas['y_max_mm']:.2f} mm")
                print(f"      Posición mínima: {metricas['y_min_mm']:.2f} mm")
                print(f"      Posición final: {metricas['y_final_mm']:.2f} mm")
                print(f"      Error medio: {metricas['error_medio_mm']:.3f} mm")
                print(f"      Voltaje máximo: {metricas['u_max_V']:.2f} V")
                if 'alerta' in metricas:
                    print(f"      {metricas['alerta']}")
        
        elif 'multiescalon' in nombre:
            # Analizar cada escalón
            y = df['y'].values * 1000
            yd = df['yd'].values * 1000
            niveles_unicos = np.unique(yd)
            print(f"\n   📈 Niveles de operación explorados:")
            for nivel in niveles_unicos:
                mask = yd == nivel
                y_nivel = y[mask]
                print(f"      Ref={nivel:.1f}mm → y_medio={np.mean(y_nivel):.2f}mm (σ={np.std(y_nivel):.3f}mm)")
        
        elif 'chirp' in nombre:
            print(f"\n   📈 Barrido de frecuencia:")
            print(f"      Rango: 0.1 Hz → 2 Hz")
            y = df['y'].values * 1000
            print(f"      Amplitud respuesta: {(np.max(y) - np.min(y))/2:.2f} mm")
    
    # Generar gráficas
    print(f"\n{'='*60}")
    print("📊 Generando gráficas...")
    graficar_todos(archivos)
    print(f"💾 Guardado: {EXP_DIR / 'resumen_experimentos.png'}")
    
    # Interpretación física
    print(f"\n{'='*60}")
    print("🔍 INTERPRETACIÓN FÍSICA")
    print("="*60)
    print("""
Los experimentos permiten caracterizar el sistema:

1. ESCALÓN: Revela la dinámica del lazo cerrado
   - Tiempo de respuesta del controlador PID
   - Estabilidad y amortiguamiento

2. SENOIDAL: Respuesta en frecuencia
   - Ganancia y fase a una frecuencia específica
   - Útil para ajuste de controlador

3. APPROACH: Exploración de zona no lineal
   - Comportamiento cerca del electroimán
   - Identificación de límites de operación

4. CHIRP: Función de transferencia completa
   - Barrido de frecuencias para FRF
   - Identificación de modos y resonancias

5. MULTIESCALON: Puntos de operación
   - Linealidad del sistema en diferentes posiciones
   - Variación de parámetros con la posición
""")

if __name__ == '__main__':
    main()

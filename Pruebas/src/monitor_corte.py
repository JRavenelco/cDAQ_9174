#!/usr/bin/env python3
"""
Monitor en tiempo real de eventos de corte.
Vigila la carpeta y analiza automáticamente cada archivo nuevo.
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.integrate import cumulative_trapezoid

# Carpetas a vigilar
CARPETAS_VIGILAR = [
    Path(r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\caracterizacion_fuerza"),
    Path(r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\experimentos_mesa"),
]

# Archivos ya procesados
archivos_procesados = set()

def analizar_archivo(filepath):
    """Analiza un archivo de datos de corte."""
    print(f"\n{'='*60}")
    print(f"📊 ANALIZANDO: {filepath.name}")
    print(f"   {datetime.now().strftime('%H:%M:%S')}")
    print('='*60)
    
    try:
        # Detectar separador
        with open(filepath, 'r') as f:
            primera_linea = f.readline()
        
        if '\t' in primera_linea:
            sep = '\t'
        elif ',' in primera_linea:
            sep = ','
        else:
            sep = None
        
        df = pd.read_csv(filepath, sep=sep)
        
        # Detectar columnas
        cols = df.columns.tolist()
        print(f"   Columnas: {cols}")
        print(f"   Puntos: {len(df)}")
        
        # Buscar columnas de tiempo, fuerza, aceleración
        tiempo_col = None
        fuerza_col = None
        acel_col = None
        
        for c in cols:
            cl = c.lower()
            if 'tiempo' in cl or 'time' in cl or cl == 't':
                tiempo_col = c
            elif 'fuerza' in cl or 'force' in cl or cl == 'f':
                fuerza_col = c
            elif 'acel' in cl or 'accel' in cl or cl == 'a':
                acel_col = c
        
        # Si no encuentra, usar posición
        if tiempo_col is None and len(cols) >= 1:
            tiempo_col = cols[0]
        if fuerza_col is None and len(cols) >= 2:
            fuerza_col = cols[1]
        if acel_col is None and len(cols) >= 3:
            acel_col = cols[2]
        
        tiempo = df[tiempo_col].values if tiempo_col else np.arange(len(df))
        
        # Calcular fs
        if len(tiempo) > 1:
            dt = tiempo[1] - tiempo[0]
            if dt > 0:
                fs = 1 / dt
            else:
                fs = 2500  # default
        else:
            fs = 2500
        
        duracion = tiempo[-1] - tiempo[0] if len(tiempo) > 1 else 0
        
        print(f"\n📈 ESTADÍSTICAS:")
        print(f"   Duración: {duracion*1000:.1f} ms")
        print(f"   Fs: {fs:.0f} Hz")
        
        # Fuerza
        if fuerza_col and fuerza_col in df.columns:
            fuerza = df[fuerza_col].values
            print(f"\n   FUERZA ({fuerza_col}):")
            print(f"     Media: {np.mean(fuerza):.4f}")
            print(f"     Std: {np.std(fuerza):.4f}")
            print(f"     Rango: [{np.min(fuerza):.4f}, {np.max(fuerza):.4f}]")
            
            # FFT
            fuerza_centered = fuerza - np.mean(fuerza)
            n = len(fuerza_centered)
            fft_mag = 2 * np.abs(rfft(fuerza_centered * np.hanning(n))) / n
            fft_freq = rfftfreq(n, 1/fs)
            
            # Top frecuencias
            idx_sorted = np.argsort(fft_mag)[::-1]
            print(f"     Frecuencias dominantes:")
            count = 0
            for idx in idx_sorted:
                if fft_freq[idx] > 5 and count < 3:  # Ignorar DC
                    print(f"       {fft_freq[idx]:.1f} Hz (mag: {fft_mag[idx]:.4f})")
                    count += 1
        
        # Aceleración
        if acel_col and acel_col in df.columns:
            acel = df[acel_col].values
            print(f"\n   ACELERACIÓN ({acel_col}):")
            print(f"     Media: {np.mean(acel):.4f}")
            print(f"     Std: {np.std(acel):.4f}")
            print(f"     Rango: [{np.min(acel):.4f}, {np.max(acel):.4f}]")
            
            # FFT aceleración
            acel_centered = acel - np.mean(acel)
            n = len(acel_centered)
            fft_mag_a = 2 * np.abs(rfft(acel_centered * np.hanning(n))) / n
            fft_freq_a = rfftfreq(n, 1/fs)
            
            idx_sorted_a = np.argsort(fft_mag_a)[::-1]
            print(f"     Frecuencias dominantes:")
            count = 0
            for idx in idx_sorted_a:
                if fft_freq_a[idx] > 5 and count < 3:
                    print(f"       {fft_freq_a[idx]:.1f} Hz (mag: {fft_mag_a[idx]:.4f})")
                    count += 1
            
            # Integrar para velocidad
            if fs > 0:
                try:
                    # Filtro pasa-altas
                    fc = 5  # Hz
                    b, a = signal.butter(2, fc / (fs/2), btype='high')
                    acel_filt = signal.filtfilt(b, a, acel * 9.81)
                    vel = cumulative_trapezoid(acel_filt, tiempo, initial=0)
                    vel_filt = signal.filtfilt(b, a, vel)
                    
                    print(f"\n   VELOCIDAD (integrada):")
                    print(f"     Rango: [{vel_filt.min()*1000:.2f}, {vel_filt.max()*1000:.2f}] mm/s")
                except:
                    pass
        
        # Correlación F-a
        if fuerza_col and acel_col and fuerza_col in df.columns and acel_col in df.columns:
            corr = np.corrcoef(fuerza, acel)[0, 1]
            print(f"\n   CORRELACIÓN F-a: {corr:.3f}")
            if abs(corr) > 0.7:
                print(f"     ✅ Buena correlación - potencial para modelo")
            elif abs(corr) > 0.3:
                print(f"     ⚠️ Correlación moderada")
            else:
                print(f"     ❌ Baja correlación - señales desacopladas")
        
        print(f"\n{'='*60}\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def buscar_archivos_nuevos():
    """Busca archivos nuevos en las carpetas vigiladas."""
    nuevos = []
    
    for carpeta in CARPETAS_VIGILAR:
        if not carpeta.exists():
            continue
        
        # Buscar archivos CSV y TXT recientes
        for ext in ['*.csv', '*.txt']:
            for archivo in carpeta.glob(ext):
                # Ignorar archivos ya procesados
                if archivo in archivos_procesados:
                    continue
                
                # Verificar que sea reciente (últimos 30 minutos)
                mtime = archivo.stat().st_mtime
                edad_segundos = time.time() - mtime
                
                if edad_segundos < 1800:  # 30 minutos
                    # Verificar que tenga datos (>1KB)
                    if archivo.stat().st_size > 1000:
                        nuevos.append(archivo)
    
    return nuevos


def main():
    print("="*60)
    print("🔍 MONITOR DE EVENTOS DE CORTE EN TIEMPO REAL")
    print("="*60)
    print(f"Vigilando carpetas:")
    for c in CARPETAS_VIGILAR:
        print(f"  📁 {c}")
    print(f"\nEsperando archivos nuevos... (Ctrl+C para salir)")
    print("="*60 + "\n")
    
    # Marcar archivos existentes como ya procesados
    for carpeta in CARPETAS_VIGILAR:
        if carpeta.exists():
            for ext in ['*.csv', '*.txt']:
                for archivo in carpeta.glob(ext):
                    archivos_procesados.add(archivo)
    
    print(f"📋 {len(archivos_procesados)} archivos existentes ignorados\n")
    
    try:
        while True:
            nuevos = buscar_archivos_nuevos()
            
            for archivo in nuevos:
                # Esperar un momento para que termine de escribirse
                time.sleep(1)
                
                if analizar_archivo(archivo):
                    archivos_procesados.add(archivo)
            
            time.sleep(2)  # Revisar cada 2 segundos
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitor detenido.")
        print(f"   Archivos analizados: {len(archivos_procesados)}")


if __name__ == "__main__":
    main()

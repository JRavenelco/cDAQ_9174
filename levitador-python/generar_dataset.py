"""
Generador de Datasets para Metaheurísticos, PINN y KAN
=======================================================

Este script genera datos del levitador magnético para:
1. Benchmark de metaheurísticos (identificación de k0, k, a)
2. Entrenamiento de PINN (Physics-Informed Neural Networks)
3. Entrenamiento de KAN (Kolmogorov-Arnold Networks)

Datos generados:
- t: tiempo [s]
- y: posición medida (sensor) [m]
- y_obs: posición estimada (observador) [m]
- i: corriente [A]
- u: voltaje de control [V]
- yd: referencia [m]
- dy_obs: velocidad estimada [m/s]

Autor: Jesús (Doctorado UAQ)
"""

import time
import numpy as np
from serial_win32 import Win32Serial
import threading
import queue
from datetime import datetime
from pathlib import Path
import argparse

# Parámetros del modelo (calibrados)
K0 = 0.0704   # H
K = 0.0327    # H
A = 0.0052    # m
R = 2.72      # Ω
M = 0.018     # kg
G = 9.81      # m/s²

# Parámetros de control PID
Ts = 0.01
kp = 100
ki = 50
kd = 1.5
kpi = 12.0
kii = 3000.0
Vref = 9.86
Iref = 0.827
Rs = 2.2

# Importar observador
from control_observador import ObservadorPosicion

# =============================================================================
# PERFILES DE REFERENCIA PARA GENERAR DATOS RICOS
# =============================================================================

def perfil_constante(t, y0=0.005):
    """Referencia constante."""
    return y0

def perfil_escalon(t, y0=0.005, y1=0.004, t_cambio=10.0):
    """Escalón simple."""
    return y0 if t < t_cambio else y1

def perfil_senoidal(t, y0=0.005, amp=0.001, freq=0.3, t_inicio=5.0):
    """Senoidal suave."""
    if t < t_inicio:
        return y0
    return y0 + amp * np.sin(2 * np.pi * freq * (t - t_inicio))

def perfil_chirp(t, y0=0.005, amp=0.0008, f0=0.1, f1=1.0, dur=30.0, t_inicio=5.0):
    """Barrido de frecuencia."""
    if t < t_inicio:
        return y0
    t_rel = t - t_inicio
    if t_rel > dur:
        return y0
    f = f0 + (f1 - f0) * t_rel / dur
    return y0 + amp * np.sin(2 * np.pi * f * t_rel)

def perfil_multiescalon(t, niveles=[0.006, 0.005, 0.004, 0.0045], t_nivel=8.0, t_inicio=3.0):
    """Múltiples escalones."""
    if t < t_inicio:
        return niveles[0]
    t_rel = t - t_inicio
    idx = min(int(t_rel / t_nivel), len(niveles) - 1)
    return niveles[idx]

def perfil_prbs(t, y_bajo=0.004, y_alto=0.006, periodo=0.8, seed=42, t_inicio=3.0):
    """Señal pseudo-aleatoria."""
    if t < t_inicio:
        return (y_bajo + y_alto) / 2
    np.random.seed(seed + int(t / periodo))
    return y_alto if np.random.rand() > 0.5 else y_bajo

PERFILES = {
    'constante': perfil_constante,
    'escalon': perfil_escalon,
    'senoidal': perfil_senoidal,
    'chirp': perfil_chirp,
    'multiescalon': perfil_multiescalon,
    'prbs': perfil_prbs
}

# =============================================================================
# GENERADOR DE DATOS
# =============================================================================

class GeneradorDataset:
    def __init__(self, port='COM1', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.data = []
        self.running = False
        self.observador = ObservadorPosicion()
        
    def generar(self, duracion=30.0, perfil='constante', output_dir='datasets'):
        """Genera un dataset con el perfil especificado."""
        
        # Crear directorio de salida
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Nombre del archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"dataset_{perfil}_{timestamp}.txt"
        filepath = output_path / filename
        
        print(f"\n{'='*60}")
        print(f"🔬 GENERANDO DATASET: {perfil.upper()}")
        print(f"{'='*60}")
        print(f"Duración: {duracion}s")
        print(f"Archivo: {filepath}")
        print(f"{'='*60}\n")
        
        # Obtener función de perfil
        perfil_func = PERFILES.get(perfil, perfil_constante)
        
        # Variables de control
        pv, i, y, ef, ef_1, u, t = 0, 0, 0, 0, 0, 0, 0
        esc = 0.05 / 1023.0
        esci = 5.0 / (Rs * 1023.0)
        escs = 254.0 / Vref
        iTs = 1 / Ts
        pwmf, yd, proporcional, derivativa, ie, ied, id_val, ei, propi, intei, integral = 0, 0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
        self.data = []
        self.observador.reset()
        
        try:
            ser = Win32Serial(self.port, self.baudrate)
            print(f"✅ Puerto {self.port} abierto")
            
            # Esperar switch
            print("⏳ Esperando switch (byte 0xAA)...")
            while True:
                b = ser.read(1)
                if b == b'\xAA':
                    print("✅ ¡Switch activado!")
                    break
            
            # Bucle principal
            flagcom = 0
            pv = 0
            i = 0
            t_inicio = time.time()
            
            while (time.time() - t_inicio) < duracion:
                b = ser.read(1)
                if not b:
                    continue
                
                recibido = b[0]
                
                if flagcom != 0:
                    flagcom += 1
                
                if (recibido == 0xAA) and (flagcom == 0):
                    pv = 0
                    i = 0
                    flagcom = 1
                    continue
                
                if flagcom == 2:
                    pv = recibido << 8
                    continue
                
                if flagcom == 3:
                    pv = pv + recibido
                    continue
                
                if flagcom == 4:
                    i = recibido << 8
                    continue
                
                if flagcom == 5:
                    i = i + recibido
                    
                    if pv > 1023 or i > 1023:
                        flagcom = 0
                        continue
                    
                    # Referencia del perfil
                    yd = perfil_func(t)
                    
                    # Mediciones
                    y_sensor = esc * pv
                    ie = esci * i
                    
                    # Observador
                    y_obs, dy_obs = self.observador.estimar(ie, u)
                    
                    # Control PID (usando sensor)
                    y = y_sensor
                    ef_1 = ef
                    ef = yd - y
                    proporcional = kp * ef
                    derivativa = kd * (ef - ef_1) * iTs
                    
                    if -Iref < integral < Iref:
                        integral += ki * Ts * ef
                    else:
                        integral = 0.95 * Iref if integral >= Iref else -0.95 * Iref
                    
                    id_val = proporcional + integral + derivativa
                    if id_val > 0:
                        id_val = 0
                    if id_val <= -Iref:
                        id_val = -Iref
                    
                    ied = -id_val
                    ei = ied - ie
                    propi = kpi * ei
                    
                    if -Vref < intei < Vref:
                        intei += kii * Ts * ei
                    else:
                        intei = 0.95 * Vref if intei >= Vref else -0.95 * Vref
                    
                    u = propi + intei
                    u = max(0, min(u, Vref))
                    
                    pwmf = escs * u
                    pwm = int(abs(pwmf))
                    ser.write(bytes([pwm]))
                    
                    # Guardar datos
                    self.data.append({
                        't': t,
                        'y': y_sensor,
                        'y_obs': y_obs,
                        'dy_obs': dy_obs,
                        'i': ie,
                        'u': u,
                        'yd': yd,
                        'pv_raw': pv,
                        'i_raw': i
                    })
                    
                    flagcom = 0
                    t += Ts
            
            ser.close()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
        
        # Guardar archivo
        print(f"\n💾 Guardando {len(self.data)} muestras...")
        
        with open(filepath, 'w') as f:
            # Header con metadata
            f.write(f"# Dataset Levitador Magnético\n")
            f.write(f"# Fecha: {datetime.now().isoformat()}\n")
            f.write(f"# Perfil: {perfil}\n")
            f.write(f"# Duración: {duracion}s\n")
            f.write(f"# Muestras: {len(self.data)}\n")
            f.write(f"# Ts: {Ts}s\n")
            f.write(f"# Parámetros modelo: k0={K0}, k={K}, a={A}, R={R}\n")
            f.write(f"# Columnas: t y y_obs dy_obs i u yd\n")
            f.write(f"#\n")
            
            for d in self.data:
                f.write(f"{d['t']:.4f}\t{d['y']:.6f}\t{d['y_obs']:.6f}\t{d['dy_obs']:.6f}\t{d['i']:.4f}\t{d['u']:.4f}\t{d['yd']:.6f}\n")
        
        print(f"✅ Dataset guardado: {filepath}")
        
        # Diagnóstico del observador
        diag = self.observador.get_diagnostics()
        print(f"\n📊 Diagnóstico del observador:")
        print(f"   Muestras válidas: {diag['valid_samples']}/{diag['total_samples']} ({100*diag['valid_samples']/max(1,diag['total_samples']):.1f}%)")
        print(f"   Resets: {diag['resets']}")
        
        return filepath


def generar_dataset_completo(duracion_por_perfil=30):
    """Genera datasets con todos los perfiles."""
    gen = GeneradorDataset()
    archivos = []
    
    perfiles = ['constante', 'escalon', 'senoidal', 'chirp', 'multiescalon']
    
    for perfil in perfiles:
        print(f"\n{'#'*60}")
        print(f"# PERFIL: {perfil.upper()}")
        print(f"{'#'*60}")
        
        filepath = gen.generar(duracion=duracion_por_perfil, perfil=perfil)
        if filepath:
            archivos.append(filepath)
        
        # Pausa entre experimentos
        print("\n⏳ Pausa de 5 segundos antes del siguiente perfil...")
        time.sleep(5)
    
    print(f"\n{'='*60}")
    print(f"✅ GENERACIÓN COMPLETA")
    print(f"{'='*60}")
    print(f"Archivos generados: {len(archivos)}")
    for f in archivos:
        print(f"  - {f}")
    
    return archivos


def main():
    parser = argparse.ArgumentParser(description='Generador de datasets para el levitador')
    parser.add_argument('--perfil', '-p', default='constante', 
                        choices=list(PERFILES.keys()),
                        help='Perfil de referencia')
    parser.add_argument('--duracion', '-d', type=float, default=30.0,
                        help='Duración en segundos')
    parser.add_argument('--completo', '-c', action='store_true',
                        help='Generar todos los perfiles')
    parser.add_argument('--output', '-o', default='datasets',
                        help='Directorio de salida')
    args = parser.parse_args()
    
    if args.completo:
        generar_dataset_completo(args.duracion)
    else:
        gen = GeneradorDataset()
        gen.generar(duracion=args.duracion, perfil=args.perfil, output_dir=args.output)


if __name__ == '__main__':
    main()

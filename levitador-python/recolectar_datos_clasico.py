"""
Script de Recolección de Datos de Levitación (Control Clásico)
==============================================================

Este script ejecuta el control PID clásico para mantener la esfera levitando
de forma estable y guarda los datos (t, y, u, i) para sintonización offline.

Duración: 20 segundos
Salida: datos_levitacion.csv

Autor: Cascade
"""

import time
import queue
import threading
import pandas as pd
from serial_win32 import Win32Serial

# Configuración
PORT = 'COM1'
BAUDRATE = 115200
DURATION = 20  # Segundos
OUTPUT_FILE = 'datos_levitacion.csv'

# Parámetros PID (Copiados de control.py)
Ts = 0.01
kp = 100
ki = 50
kd = 1.5
kpi = 12.0
kii = 3000.0
Vref = 9.86
Iref = 0.827
Rs = 2.2  # Resistencia de sensado original del modelo
BETA_P = 0.8
KAW_I = 200.0
KAW_U = 2000.0

exit_event = threading.Event()

def control_loop():
    # Variables de estado
    integral = 0
    intei = 0
    y_prev = 0.005
    ef = 0
    ef_1 = 0
    t = 0
    
    # Escalas
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    iTs = 1 / Ts
    
    data_buffer = []
    
    try:
        ser = Win32Serial(PORT, BAUDRATE)
        print(f"✅ Puerto {PORT} abierto.")
        
        # Sincronización
        print("⏳ Esperando switch (0xAA)...")
        while not exit_event.is_set():
            b = ser.read(1)
            if b and b == b'\xAA':
                print("🚀 Iniciando control...")
                break
        
        start_time = time.time()
        flagcom = 0
        pv = 0
        i_raw = 0
        
        while not exit_event.is_set():
            if time.time() - start_time > DURATION:
                print("⏱️ Tiempo completado.")
                break
                
            b = ser.read(1)
            if not b: continue
            
            recibido = b[0]
            
            # Máquina de estados protocolo (igual a control.py)
            if flagcom != 0: flagcom += 1
            if (recibido == 0xAA) and (flagcom == 0):
                pv = 0; i_raw = 0; flagcom = 1; continue
            if flagcom == 2: pv = recibido << 8; continue
            if flagcom == 3: pv += recibido; continue
            if flagcom == 4: i_raw = recibido << 8; continue
            if flagcom == 5:
                i_raw += recibido
                
                if pv > 1023 or i_raw > 1023:
                    flagcom = 0; continue
                
                # --- CONTROL PID ---
                yd = 0.005 # Referencia fija 5mm
                
                # Lecturas
                y = esc * pv
                ie = esci * i_raw
                
                # PID Posición
                e_i = yd - y
                e_p = (BETA_P * yd) - y
                ef = e_i
                d_y = (y - y_prev) * iTs
                derivativa = -kd * d_y
                
                id_unsat = (kp * e_p) + integral + derivativa
                id_val = max(-Iref, min(0.0, id_unsat))
                
                integral += (ki * Ts * e_i) + (KAW_I * (id_val - id_unsat) * Ts)
                integral = max(-Iref, min(Iref, integral))
                
                y_prev = y
                
                # PID Corriente
                ied = -id_val
                error_i = ied - ie
                propi = kpi * error_i
                
                u_unsat = propi + intei
                u = max(0.0, min(Vref, u_unsat))
                
                intei += (kii * Ts * error_i) + (KAW_U * (u - u_unsat) * Ts)
                intei = max(-Vref, min(Vref, intei))
                
                # Actuación
                pwm = int(abs(escs * u))
                ser.write(bytes([pwm]))
                
                # Guardar datos
                data_buffer.append({
                    't': t,
                    'y': y,
                    'u': u,
                    'i': ie,
                    'yd': yd
                })
                
                t += Ts
                flagcom = 0
                
                if len(data_buffer) % 100 == 0:
                    print(f"t={t:.1f}s | y={y*1000:.1f}mm | i={ie:.3f}A")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open: ser.close()
        
        # Guardar archivo
        df = pd.DataFrame(data_buffer)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Datos guardados en {OUTPUT_FILE}")

if __name__ == '__main__':
    control_loop()

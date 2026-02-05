"""
Script de Identificación de Parámetros Térmicos (alpha, beta)
=============================================================

Este script tiene dos funciones:
1. RECOLECTAR: Ejecuta un protocolo de prueba en el levitador (sin esfera)
   para generar una curva de calentamiento y enfriamiento.
2. ANALIZAR: Ajusta los parámetros del modelo térmico a los datos reales
   usando optimización no lineal.

Uso:
    python identificar_parametros_termicos.py --mode recolectar --port COM1
    python identificar_parametros_termicos.py --mode analizar --file datos_termicos.csv

Requisitos:
    - Esfera FUERA del levitador (para L constante).
    - Puerto serie disponible.
"""

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from serial_win32 import Win32Serial
from modelo_resistencia import EstimadorResistencia

# Parámetros de Configuración
PWM_CALENTAMIENTO = 200  # ~78% de ciclo de trabajo (calentamiento fuerte)
PWM_MEDICION = 30        # ~12% (mínimo para medir corriente sin calentar mucho)
DURACION_HEAT = 60       # Segundos
DURACION_COOL = 60       # Segundos
V_REF = 9.86             # Voltaje de fuente
R_SENSE = 2.2            # Resistencia de sensado

def recolectar_datos(port='COM1'):
    print("="*60)
    print("🧪 PROTOCOLO DE IDENTIFICACIÓN TÉRMICA")
    print("="*60)
    print("⚠️  IMPORTANTE: RETIRA LA ESFERA DEL LEVITADOR")
    print("   (Necesitamos inductancia constante para medir R = V/I)")
    if not args.no_wait:
        input("👉 Presiona ENTER cuando la esfera esté fuera...")
    
    try:
        ser = Win32Serial(port, 115200)
        print(f"✅ Puerto {port} abierto.")
    except Exception as e:
        print(f"❌ Error abriendo puerto: {e}")
        return

    datos = []
    start_time = time.time()
    
    try:
        # Esperar handshake
        print("⏳ Esperando microcontrolador...")
        while True:
            b = ser.read(1)
            if b and b == b'\xAA':
                break
        print("✅ Conexión establecida.")

        # --- FASE 1: CALENTAMIENTO ---
        print(f"\n🔥 FASE 1: CALENTAMIENTO ({DURACION_HEAT}s)")
        print(f"   Aplicando PWM={PWM_CALENTAMIENTO}...")
        
        t_fase = time.time()
        while time.time() - t_fase < DURACION_HEAT:
            # Enviar PWM alto
            ser.write(bytes([PWM_CALENTAMIENTO]))
            
            # Leer datos (protocolo binario igual que control.py)
            # Esperar 6 bytes
            buffer = []
            while len(buffer) < 6:
                b = ser.read(1)
                if b: buffer.append(b[0])
            
            # Decodificar (simplificado, asumiendo sincronía por simplicidad en test)
            # En producción idealmente usaríamos la misma lógica robusta de control.py
            # Aquí confiamos en que al enviar 1 byte recibimos respuesta
            
            # Re-implementando lectura robusta básica
            # El micro envía: 0xAA, H_pv, L_pv, H_i, L_i
            # Pero el Win32Serial lee byte a byte.
            # Vamos a usar una lógica simplificada de lectura bloqueante
            pass 
            
            # MEJOR: Usar la misma lógica de lectura que control.py
            # Pero para no complicar, vamos a leer y parsear "al vuelo"
            # Ojo: Win32Serial read(1) retorna bytes
            
            # Leer hasta encontrar 0xAA
            while True:
                b = ser.read(1)
                if b == b'\xAA': break
            
            # Leer 4 bytes de datos
            raw = ser.read(4)
            if len(raw) == 4:
                pv = (raw[0] << 8) | raw[1]
                i_raw = (raw[2] << 8) | raw[3]
                
                # Conversión
                u = (PWM_CALENTAMIENTO / 254.0) * V_REF
                i = (i_raw * 5.0) / (1023.0 * R_SENSE)
                t = time.time() - start_time
                
                # Calcular R instantánea (Ley de Ohm)
                # Como L es constante y I casi constante, V = R*I
                r_inst = u / i if i > 0.05 else 0
                
                datos.append({'t': t, 'u': u, 'i': i, 'r_med': r_inst, 'fase': 'heat'})
                
                if len(datos) % 10 == 0:
                    print(f"   t={t:.1f}s | I={i:.3f}A | R={r_inst:.3f}Ω")

        # --- FASE 2: ENFRIAMIENTO ---
        print(f"\n❄️  FASE 2: ENFRIAMIENTO ({DURACION_COOL}s)")
        print(f"   Aplicando PWM={PWM_MEDICION} (solo medición)...")
        
        t_fase = time.time()
        while time.time() - t_fase < DURACION_COOL:
            ser.write(bytes([PWM_MEDICION]))
            
            # Leer (misma lógica)
            while True:
                b = ser.read(1)
                if b == b'\xAA': break
            raw = ser.read(4)
            
            if len(raw) == 4:
                i_raw = (raw[2] << 8) | raw[3]
                u = (PWM_MEDICION / 254.0) * V_REF
                i = (i_raw * 5.0) / (1023.0 * R_SENSE)
                t = time.time() - start_time
                r_inst = u / i if i > 0.05 else 0
                
                datos.append({'t': t, 'u': u, 'i': i, 'r_med': r_inst, 'fase': 'cool'})
                
                if len(datos) % 10 == 0:
                    print(f"   t={t:.1f}s | I={i:.3f}A | R={r_inst:.3f}Ω")
                    
    except KeyboardInterrupt:
        print("\n⏹️ Cancelado por usuario.")
    finally:
        ser.close()
        
    # Guardar
    df = pd.DataFrame(datos)
    filename = f"datos_termicos_{int(time.time())}.csv"
    df.to_csv(filename, index=False)
    print(f"\n💾 Datos guardados en: {filename}")
    
    # Graficar preliminar
    plt.figure(figsize=(10, 6))
    plt.plot(df['t'], df['r_med'], label='R Medida (V/I)')
    plt.axvline(DURACION_HEAT, color='r', linestyle='--', label='Fin Calentamiento')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Resistencia [Ω]')
    plt.title('Curva de Resistencia Experimental')
    plt.legend()
    plt.grid(True)
    plt.show()

def simular_modelo(params, t, u, i, r0):
    """Simula el modelo térmico con parámetros dados."""
    alpha, beta = params
    
    # Crear estimador
    est = EstimadorResistencia(R0=r0, alpha=alpha, beta=beta, dt=0.0, gain_fusion=0.0)
    
    r_sim = []
    r_curr = r0
    
    # Simulación vectorizada manual (o bucle simple)
    # Como dt no es constante (depende de la velocidad de serial), calculamos dt paso a paso
    est.R = r0
    
    for k in range(len(t)):
        dt = t[k] - t[k-1] if k > 0 else 0.01
        if dt > 0.5: dt = 0.01 # Corrección por gaps grandes
        est.dt = dt
        
        # Update
        est.update(u[k], i[k])
        r_sim.append(est.R)
        
    return np.array(r_sim)

def funcion_costo(params, t, u, i, r_real):
    """Calcula el error cuadrático medio entre modelo y datos."""
    alpha, beta = params
    if alpha < 0 or beta < 0: return 1e6 # Penalización
    
    r0 = r_real[0] # Asumimos que empieza en ambiente
    r_sim = simular_modelo(params, t, u, i, r0)
    
    # Error RMS
    error = np.sqrt(np.mean((r_real - r_sim)**2))
    return error

def analizar_datos(archivo):
    print(f"\n🔍 Analizando: {archivo}")
    df = pd.read_csv(archivo)
    
    # Filtrar datos ruidosos (R muy bajo o muy alto)
    # Datos muestran ~13-17 Ohms. Ampliamos rango.
    df = df[(df['r_med'] > 2.0) & (df['r_med'] < 25.0)]
    
    t = df['t'].values
    u = df['u'].values
    i = df['i'].values
    r_real = df['r_med'].values
    
    # Estimación inicial
    # alpha ~ 0.05, beta ~ 0.01
    x0 = [0.05, 0.01]
    
    print("🚀 Optimizando parámetros...")
    res = minimize(funcion_costo, x0, args=(t, u, i, r_real), 
                   method='Nelder-Mead', tol=1e-4)
    
    alpha_opt, beta_opt = res.x
    err = res.fun
    
    print("\n" + "="*40)
    print("RESULTADOS DE IDENTIFICACIÓN")
    print("="*40)
    print(f"✅ Alpha (Calentamiento): {alpha_opt:.6f} Ω/(W·s)")
    print(f"✅ Beta  (Enfriamiento):  {beta_opt:.6f} 1/s")
    print(f"   Tau Térmica:           {1/beta_opt:.2f} s")
    print(f"   Error RMS Ajuste:      {err:.4f} Ω")
    print("="*40)
    
    # Graficar comparación
    r_sim = simular_modelo(res.x, t, u, i, r_real[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, r_real, 'b.', alpha=0.3, label='Datos Reales')
    plt.plot(t, r_sim, 'r-', linewidth=2, label=f'Modelo (α={alpha_opt:.3f}, β={beta_opt:.3f})')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Resistencia [Ω]')
    plt.title('Validación del Modelo Térmico')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['recolectar', 'analizar'], required=True)
    parser.add_argument('--port', default='COM1', help='Puerto serie (solo recolección)')
    parser.add_argument('--file', help='Archivo CSV (solo análisis)')
    parser.add_argument('--no-wait', action='store_true', help='No esperar confirmación de usuario')
    
    args = parser.parse_args()
    
    if args.mode == 'recolectar':
        recolectar_datos(args.port)
    elif args.mode == 'analizar':
        if not args.file:
            print("❌ Error: Debes especificar --file para analizar.")
        else:
            analizar_datos(args.file)

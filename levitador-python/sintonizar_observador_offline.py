"""
Sintonización Offline del Observador
====================================

Este script carga los datos de levitación recolectados (datos_levitacion.csv)
y optimiza los parámetros del modelo físico (R, K0, K, A) para minimizar
el error de estimación de posición del observador.

Método:
1. Replay de señales u, i registradas.
2. Simulación del observador con parámetros de prueba.
3. Optimización con scipy.optimize.minimize.

Autor: Cascade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from control_observador import ObservadorPosicion
import argparse

def cargar_datos(file_path):
    try:
        df = pd.read_csv(file_path)
        # Filtrar datos donde y=0 (posible error de sensor o fuera de rango)
        df = df[df['y'] > 0.001]
        return df
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return None

def simular_observador(params, t, u, i, y_real):
    # Desempaquetar parámetros a optimizar
    # R_eff, K0, K, A
    R_eff, K0, K, A = params
    
    # Crear observador con estos parámetros
    # alpha=0 para R constante (o R efectiva promedio) durante la prueba corta
    obs = ObservadorPosicion(k0=K0, k=K, a=A, r=R_eff, dt=0.01)
    # Forzar parámetros internos que podrían haber sido sobreescritos por __init__
    obs.estimador_R.R = R_eff
    obs.estimador_R.alpha = 0.0
    obs.estimador_R.gain = 0.0 # Desactivar corrección Ohm para evaluar modelo puro
    
    y_est = []
    
    # Simulación
    # Necesitamos dt variable si hay jitter, o asumir fijo
    t_prev = t[0]
    
    for k in range(len(t)):
        dt = t[k] - t_prev if k > 0 else 0.01
        if dt <= 0: dt = 0.01
        if dt > 0.1: dt = 0.01 # Gap grande
        
        obs.dt = dt
        obs.estimador_R.dt = dt
        
        # Estimar
        y_val, _ = obs.estimar(i[k], u[k])
        y_est.append(y_val)
        
        t_prev = t[k]
        
    return np.array(y_est)

def funcion_costo(params, t, u, i, y_real):
    # Penalizaciones por valores físicos imposibles
    R_eff, K0, K, A = params
    
    if R_eff < 10.0 or R_eff > 25.0: return 1e6
    if K0 < 0.03 or K0 > 0.15: return 1e6
    if K < 0.01 or K > 0.08: return 1e6
    if A < 0.003 or A > 0.015: return 1e6
    
    y_sim = simular_observador(params, t, u, i, y_real)
    
    # Error RMSE
    # Dar más peso a errores grandes o filtrar transitorios?
    # RMSE simple
    error = np.sqrt(np.mean((y_real - y_sim)**2))
    return error

def main():
    parser = argparse.ArgumentParser(description='Sintonización offline del observador')
    parser.add_argument('file', type=str, default='datos_levitacion.csv', nargs='?',
                        help='Archivo CSV con datos (t, y, u, i)')
    parser.add_argument('--duracion', type=float, default=None,
                        help='Limitar datos a primeros N segundos')
    args = parser.parse_args()
    
    df = cargar_datos(args.file)
    if df is None: return
    
    # Limitar duración si se especifica
    if args.duracion is not None:
        df = df[df['t'] <= args.duracion]
    
    t = df['t'].values
    u = df['u'].values
    i = df['i'].values
    y = df['y'].values
    
    print(f"Datos cargados: {len(t)} muestras (t={t[0]:.1f}-{t[-1]:.1f}s).")
    print(f"Rango posición: {y.min()*1000:.2f} - {y.max()*1000:.2f} mm")
    print(f"Rango corriente: {i.min():.3f} - {i.max():.3f} A")
    
    # Valores iniciales basados en análisis del código C
    # R ~ 16.0 (Escala efectiva del C), K0, K, A del modelo actual
    x0 = [16.0, 0.0704, 0.0327, 0.0052]
    
    print("\n🚀 Optimizando parámetros del observador...")
    print(f"Inicial: R={x0[0]:.2f}Ω, K0={x0[1]:.4f}H, K={x0[2]:.4f}H, A={x0[3]:.4f}m")
    
    res = minimize(funcion_costo, x0, args=(t, u, i, y), 
                   method='Nelder-Mead', tol=1e-4,
                   options={'maxiter': 500, 'disp': True})
    
    opt_params = res.x
    print("\n✅ Optimización completada.")
    print("="*40)
    print(f"R efectiva: {opt_params[0]:.4f} Ω")
    print(f"K0:         {opt_params[1]:.6f} H")
    print(f"K:          {opt_params[2]:.6f} H")
    print(f"A:          {opt_params[3]:.6f} m")
    print(f"Error RMSE: {res.fun*1000:.2f} mm")
    print("="*40)
    
    # Graficar resultado
    y_opt = simular_observador(opt_params, t, u, i, y)
    y_init = simular_observador(x0, t, u, i, y)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, y*1000, 'k-', alpha=0.6, label='Sensor Real')
    plt.plot(t, y_init*1000, 'b--', label='Modelo Inicial')
    plt.plot(t, y_opt*1000, 'r-', linewidth=2, label='Modelo Optimizado')
    plt.ylabel('Posición [mm]')
    plt.xlabel('Tiempo [s]')
    plt.title('Sintonización Offline del Observador')
    plt.legend()
    plt.grid(True)
    plt.savefig('sintonizacion_observador.png')
    print("Gráfica guardada en sintonizacion_observador.png")

if __name__ == '__main__':
    main()

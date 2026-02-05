#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajuste del Modelo de Bouc-Wen a datos experimentales de Fuerza y Aceleración

Basado en el código MATLAB del usuario:
- Modelo Bouc-Wen para histéresis
- Ajuste con lsqcurvefit (scipy.optimize.curve_fit / least_squares)
- Normalización Min-Max a [-1, 1]
- Suavizado Savitzky-Golay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, curve_fit
from scipy.signal import savgol_filter
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. CARGAR DATOS
# ============================================
archivo = "experimentos_mesa/sesion_20251126_170150.csv"
print(f"📂 Cargando: {archivo}")

df = pd.read_csv(archivo, comment='#', on_bad_lines='skip')
print(f"📊 Columnas: {df.columns.tolist()}")
print(f"📏 Muestras: {len(df)}")

# Extraer señales
tiempo = df['Tiempo(s)'].values
aceleracion = df['Acel_ai0(g)'].values  # Salida (como Fuerza en MATLAB)
fuerza_volt = df['Volt_ai0(V)'].values  # Entrada (como Corriente en MATLAB)

dt = tiempo[1] - tiempo[0]
fs = 1 / dt
print(f"⏱️ Sample Rate: {fs:.1f} Hz")

# ============================================
# 2. NORMALIZACIÓN MIN-MAX [-1, 1]
# ============================================
min_entrada = np.min(fuerza_volt)
max_entrada = np.max(fuerza_volt)
min_salida = np.min(aceleracion)
max_salida = np.max(aceleracion)

# Normalizar como en MATLAB: 2 * ((x - min) / (max - min)) - 1
entrada_norm = 2 * ((fuerza_volt - min_entrada) / (max_entrada - min_entrada)) - 1
salida_norm = 2 * ((aceleracion - min_salida) / (max_salida - min_salida)) - 1

print(f"\n📊 Normalización Min-Max a [-1, 1]:")
print(f"   Entrada (Fuerza V): min={min_entrada:.6f}, max={max_entrada:.6f}")
print(f"   Salida (Acel g):    min={min_salida:.6f}, max={max_salida:.6f}")

# ============================================
# 3. SUAVIZADO SAVITZKY-GOLAY
# ============================================
window_length = 51
polyorder = 3

entrada_smooth = savgol_filter(entrada_norm, window_length, polyorder)
salida_smooth = savgol_filter(salida_norm, window_length, polyorder)
print(f"   Suavizado: ventana={window_length}, orden={polyorder}")

# ============================================
# 4. MODELO DE BOUC-WEN
# ============================================
def bouc_wen_model(params, t, u):
    """
    Modelo de Bouc-Wen para histéresis
    
    Parámetros:
        A, B, C, n, k = params
        
    Ecuación diferencial:
        dz/dt = A*du/dt - B*|du/dt|*z - C*du/dt*|z|^n
        F = k * z^2  (o k*z para modelo lineal)
    
    Args:
        params: [A, B, C, n, k]
        t: vector de tiempo
        u: señal de entrada (corriente/fuerza)
    
    Returns:
        F: fuerza modelada
        z: variable de histéresis
    """
    A, B, C, n, k = params
    
    # Calcular derivada de entrada
    du_dt = np.gradient(u, t)
    
    # Integrar ecuación de Bouc-Wen
    z = np.zeros_like(t)
    
    for i in range(1, len(t)):
        dt_i = t[i] - t[i-1]
        du = du_dt[i]
        
        # dz/dt = A*du - B*|du|*z - C*du*|z|^n
        dz_dt = A * du - B * np.abs(du) * z[i-1] - C * du * np.abs(z[i-1])**n
        z[i] = z[i-1] + dz_dt * dt_i
    
    # Fuerza = k * z^2
    F = k * z**2
    
    return F, z


def bouc_wen_residuals(params, t, u, y_target):
    """Residuos para optimización"""
    try:
        F, _ = bouc_wen_model(params, t, u)
        # Normalizar salida del modelo
        if np.max(F) - np.min(F) > 1e-10:
            F_norm = 2 * ((F - np.min(F)) / (np.max(F) - np.min(F))) - 1
        else:
            F_norm = np.zeros_like(F)
        return F_norm - y_target
    except:
        return np.ones_like(y_target) * 1e6


# ============================================
# 5. AJUSTE DE PARÁMETROS
# ============================================
print("\n" + "="*60)
print("🔧 AJUSTE DEL MODELO DE BOUC-WEN")
print("="*60)

# Usar subset de datos para ajuste más rápido
n_samples = min(10000, len(tiempo))
step = max(1, len(tiempo) // n_samples)

t_fit = tiempo[::step]
u_fit = entrada_smooth[::step]
y_fit = salida_smooth[::step]

print(f"   Usando {len(t_fit)} muestras para ajuste")

# Condiciones iniciales [A, B, C, n, k]
# Similar a MATLAB: x0 = [1, 0.5, 0.5, 2, 1]
x0_list = [
    [1.0, 0.5, 0.5, 2.0, 1.0],
    [0.4351, 0.7, 0.06, 0.58, 0.35],
    [0.2351, 0.7, 0.06, 0.58, 0.05],
]

# Límites para parámetros
bounds_lower = [0.001, 0.001, 0.001, 0.1, 0.001]
bounds_upper = [10.0, 10.0, 10.0, 5.0, 10.0]

best_result = None
best_cost = np.inf

for i, x0 in enumerate(x0_list):
    print(f"\n   Probando x0_{i+1}: {x0}")
    
    try:
        result = least_squares(
            bouc_wen_residuals,
            x0,
            args=(t_fit, u_fit, y_fit),
            bounds=(bounds_lower, bounds_upper),
            method='trf',
            verbose=0,
            max_nfev=500
        )
        
        cost = np.sum(result.fun**2)
        print(f"      Costo: {cost:.6f}, Éxito: {result.success}")
        
        if cost < best_cost:
            best_cost = cost
            best_result = result
            
    except Exception as e:
        print(f"      Error: {e}")

if best_result is not None:
    x_estimado = best_result.x
    print(f"\n✅ Parámetros estimados del modelo de Bouc-Wen:")
    print(f"   A = {x_estimado[0]:.6f}")
    print(f"   B = {x_estimado[1]:.6f}")
    print(f"   C = {x_estimado[2]:.6f}")
    print(f"   n = {x_estimado[3]:.6f}")
    print(f"   k = {x_estimado[4]:.6f}")
    
    K_m = x_estimado[4]
else:
    print("❌ No se pudo ajustar el modelo")
    x_estimado = x0_list[0]
    K_m = 1.0

# ============================================
# 6. SIMULAR MODELO CON PARÁMETROS ESTIMADOS
# ============================================
print("\n📈 Simulando modelo con parámetros estimados...")

# Usar datos completos suavizados
F_modelada, Z = bouc_wen_model(x_estimado, tiempo, entrada_smooth)

# Normalizar salida del modelo
min_F = np.min(F_modelada)
max_F = np.max(F_modelada)
if max_F - min_F > 1e-10:
    F_modelada_norm = 2 * ((F_modelada - min_F) / (max_F - min_F)) - 1
else:
    F_modelada_norm = np.zeros_like(F_modelada)

# Suavizar histéresis
Z_smooth = savgol_filter(Z, window_length, polyorder)

# ============================================
# 7. MODELO DE DUHEM (opcional)
# ============================================
print("\n🔧 Ajustando modelo de Duhem...")

# Calcular derivada de Z respecto a t
dZ_dt = np.gradient(Z, tiempo)

def duhem_model(H, dH_dt, p):
    """Modelo de Duhem: p[0]*H + p[1]*H^2 + p[2]*H^3 + p[3]*dH/dt"""
    return p[0]*H + p[1]*H**2 + p[2]*H**3 + p[3]*dH_dt

# Parámetros iniciales
p0_duhem = [0.033, -0.2, 0.01, 0.1]

try:
    from scipy.optimize import minimize
    
    def duhem_cost(p):
        B_pred = duhem_model(Z_smooth, dZ_dt, p)
        # Normalizar
        if np.max(B_pred) - np.min(B_pred) > 1e-10:
            B_norm = 2 * ((B_pred - np.min(B_pred)) / (np.max(B_pred) - np.min(B_pred))) - 1
        else:
            B_norm = np.zeros_like(B_pred)
        return np.sum((B_norm - salida_smooth)**2)
    
    result_duhem = minimize(duhem_cost, p0_duhem, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'disp': False})
    p_fit = result_duhem.x
    print(f"   Parámetros Duhem: {p_fit}")
    
    B_fit = duhem_model(Z_smooth, dZ_dt, p_fit)
    
except Exception as e:
    print(f"   Error en Duhem: {e}")
    p_fit = p0_duhem
    B_fit = Z_smooth

# ============================================
# 8. GRÁFICAS
# ============================================
print("\n📊 Generando gráficas...")

# --- FIGURA 1: Comparación datos vs modelo ---
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Ajuste del Modelo de Bouc-Wen', fontsize=14, fontweight='bold')

# Zoom para visualización
n_ciclos = 5
freq_est = 40  # Hz aproximado
idx_zoom = int(n_ciclos / freq_est * fs)
t_zoom = tiempo[:idx_zoom] * 1000  # ms

# 1. Comparación temporal
ax1 = axes1[0, 0]
ax1.plot(t_zoom, salida_smooth[:idx_zoom], 'b-', linewidth=1.5, label='Datos Experimentales')
ax1.plot(t_zoom, F_modelada_norm[:idx_zoom], 'r--', linewidth=1.5, label='Modelo Bouc-Wen')
ax1.set_xlabel('Tiempo (ms)')
ax1.set_ylabel('Amplitud (normalizada)')
ax1.set_title('Comparación: Datos vs Modelo Bouc-Wen')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Variable de histéresis Z
ax2 = axes1[0, 1]
ax2.plot(t_zoom, Z_smooth[:idx_zoom], 'g-', linewidth=1.5)
ax2.set_xlabel('Tiempo (ms)')
ax2.set_ylabel('Z (variable de histéresis)')
ax2.set_title('Variable de Histéresis Z')
ax2.grid(True, alpha=0.3)

# 3. Curva de histéresis (Entrada vs Salida)
ax3 = axes1[1, 0]
ax3.plot(entrada_smooth[:idx_zoom], salida_smooth[:idx_zoom], 'b-', linewidth=0.8, alpha=0.7, label='Datos reales')
ax3.plot(entrada_smooth[:idx_zoom], F_modelada_norm[:idx_zoom], 'r--', linewidth=1.5, label='Modelo Bouc-Wen')
ax3.set_xlabel('Entrada (Fuerza normalizada)')
ax3.set_ylabel('Salida (Aceleración normalizada)')
ax3.set_title('Curva de Histéresis')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Histéresis Z vs Entrada
ax4 = axes1[1, 1]
ax4.plot(entrada_smooth[:idx_zoom], Z_smooth[:idx_zoom], 'g-', linewidth=1)
ax4.set_xlabel('Entrada (normalizada)')
ax4.set_ylabel('Z (histéresis)')
ax4.set_title('Fenómeno de Histéresis: Entrada vs Z')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experimentos_mesa/ajuste_bouc_wen.png', dpi=150)
print(f"💾 Guardado: experimentos_mesa/ajuste_bouc_wen.png")

# --- FIGURA 2: Gráficas 3D ---
fig2 = plt.figure(figsize=(16, 6))

# 3D: Tiempo x Entrada x Salida
ax3d1 = fig2.add_subplot(1, 3, 1, projection='3d')
step_3d = max(1, idx_zoom // 1000)
ax3d1.plot(t_zoom[::step_3d], entrada_smooth[:idx_zoom:step_3d], salida_smooth[:idx_zoom:step_3d], 
           'b-', linewidth=0.8, label='Datos')
ax3d1.plot(t_zoom[::step_3d], entrada_smooth[:idx_zoom:step_3d], F_modelada_norm[:idx_zoom:step_3d], 
           'r--', linewidth=0.8, label='Modelo')
ax3d1.set_xlabel('Tiempo (ms)')
ax3d1.set_ylabel('Entrada')
ax3d1.set_zlabel('Salida')
ax3d1.set_title('3D: Datos vs Modelo')
ax3d1.legend()

# 3D: Tiempo x Entrada x Z
ax3d2 = fig2.add_subplot(1, 3, 2, projection='3d')
ax3d2.plot(t_zoom[::step_3d], entrada_smooth[:idx_zoom:step_3d], Z_smooth[:idx_zoom:step_3d], 
           'g-', linewidth=1)
ax3d2.set_xlabel('Tiempo (ms)')
ax3d2.set_ylabel('Entrada')
ax3d2.set_zlabel('Z (histéresis)')
ax3d2.set_title('3D: Histéresis')

# Lissajous con color por tiempo
ax3d3 = fig2.add_subplot(1, 3, 3, projection='3d')
colors = t_zoom[::step_3d]
scatter = ax3d3.scatter(entrada_smooth[:idx_zoom:step_3d], salida_smooth[:idx_zoom:step_3d], 
                        Z_smooth[:idx_zoom:step_3d], c=colors, cmap='viridis', s=2)
ax3d3.set_xlabel('Entrada')
ax3d3.set_ylabel('Salida')
ax3d3.set_zlabel('Z')
ax3d3.set_title('3D: Entrada × Salida × Z')

plt.tight_layout()
plt.savefig('experimentos_mesa/ajuste_bouc_wen_3d.png', dpi=150)
print(f"💾 Guardado: experimentos_mesa/ajuste_bouc_wen_3d.png")

# --- FIGURA 3: Múltiples condiciones iniciales ---
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Curvas de Histéresis para diferentes x0', fontsize=14, fontweight='bold')

colors_list = ['b', 'r', 'g', 'k', 'm']

ax_hist = axes3[0]
ax_comp = axes3[1]

for i, x0 in enumerate(x0_list):
    try:
        F_sim, Z_sim = bouc_wen_model(x0, tiempo, entrada_smooth)
        if np.max(F_sim) - np.min(F_sim) > 1e-10:
            F_sim_norm = 2 * ((F_sim - np.min(F_sim)) / (np.max(F_sim) - np.min(F_sim))) - 1
        else:
            F_sim_norm = np.zeros_like(F_sim)
        
        ax_hist.plot(entrada_smooth[:idx_zoom], F_sim_norm[:idx_zoom], 
                    colors_list[i % len(colors_list)], linewidth=1, 
                    label=f'x0_{i+1}', alpha=0.7)
        
        ax_comp.plot(t_zoom, F_sim_norm[:idx_zoom], 
                    colors_list[i % len(colors_list)], linewidth=1, 
                    label=f'x0_{i+1}', alpha=0.7)
    except:
        pass

# Datos reales
ax_hist.plot(entrada_smooth[:idx_zoom], salida_smooth[:idx_zoom], 
            'k--', linewidth=2, label='Datos reales', alpha=0.5)
ax_comp.plot(t_zoom, salida_smooth[:idx_zoom], 
            'k--', linewidth=2, label='Datos reales', alpha=0.5)

ax_hist.set_xlabel('Entrada (normalizada)')
ax_hist.set_ylabel('Salida (normalizada)')
ax_hist.set_title('Curvas de Histéresis')
ax_hist.legend()
ax_hist.grid(True, alpha=0.3)

ax_comp.set_xlabel('Tiempo (ms)')
ax_comp.set_ylabel('Salida (normalizada)')
ax_comp.set_title('Comparación Temporal')
ax_comp.legend()
ax_comp.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experimentos_mesa/ajuste_bouc_wen_comparacion.png', dpi=150)
print(f"💾 Guardado: experimentos_mesa/ajuste_bouc_wen_comparacion.png")

plt.show()

# ============================================
# RESUMEN
# ============================================
print("\n" + "="*60)
print("📋 RESUMEN DEL AJUSTE")
print("="*60)
print(f"  Archivo:        {archivo}")
print(f"  Muestras:       {len(tiempo)}")
print(f"  Sample Rate:    {fs:.0f} Hz")
print(f"\n  Parámetros Bouc-Wen estimados:")
print(f"    A = {x_estimado[0]:.6f}")
print(f"    B = {x_estimado[1]:.6f}")
print(f"    C = {x_estimado[2]:.6f}")
print(f"    n = {x_estimado[3]:.6f}")
print(f"    k = {x_estimado[4]:.6f}")
print(f"\n  Parámetros Duhem:")
print(f"    p = {p_fit}")
print("="*60)

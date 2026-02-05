"""
Analizar datos del control C exitoso (MONIT.txt) para extraer parámetros efectivos.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ruta al archivo de datos del control C
data_file = Path(__file__).parent.parent / "levitador valentin" / "MONIT.txt"

print(f"Leyendo datos desde: {data_file}")

# Cargar datos: t, yd, y_sensor, ie (corriente), col5, u (voltaje)
data = np.loadtxt(data_file)

t = data[:, 0]
yd = data[:, 1]
y_sensor = data[:, 2]
i = data[:, 3]  # Corriente medida
col5 = data[:, 4]  # Columna desconocida
u = data[:, 5]  # Voltaje aplicado

print(f"\n📊 Estadísticas de los datos (N={len(t)} muestras):")
print(f"  Tiempo: {t[0]:.2f} - {t[-1]:.2f} s")
print(f"  Posición sensor: {y_sensor.min()*1000:.2f} - {y_sensor.max()*1000:.2f} mm (media: {y_sensor.mean()*1000:.2f} mm)")
print(f"  Corriente: {i.min():.3f} - {i.max():.3f} A (media: {i.mean():.3f} A)")
print(f"  Voltaje: {u.min():.2f} - {u.max():.2f} V (media: {u.mean():.2f} V)")

# Calcular resistencia aparente R = U/I (solo cuando i > 0.1A para evitar ruido)
mask_valid = i > 0.1
R_app = np.zeros_like(i)
R_app[mask_valid] = u[mask_valid] / i[mask_valid]

R_mean = R_app[mask_valid].mean()
R_std = R_app[mask_valid].std()

print(f"\n🔌 Resistencia Aparente (U/I con i > 0.1A):")
print(f"  R_efectiva = {R_mean:.2f} ± {R_std:.2f} Ω")
print(f"  Rango: {R_app[mask_valid].min():.2f} - {R_app[mask_valid].max():.2f} Ω")

# Comparación con valor nominal
R_nominal = 2.72
print(f"  Factor de escala vs nominal (2.72Ω): {R_mean/R_nominal:.2f}x")

# Analizar relación V_ref vs voltaje
V_REF_nominal = 9.86
u_max = u.max()
print(f"\n⚡ Voltaje:")
print(f"  V_REF nominal: {V_REF_nominal:.2f} V")
print(f"  Voltaje máximo observado: {u_max:.2f} V")
print(f"  Factor: {u_max/V_REF_nominal:.2f}")

# Estimar dt (periodo de muestreo)
dt_samples = np.diff(t)
dt_mean = dt_samples.mean()
dt_std = dt_samples.std()
print(f"\n⏱️  Periodo de muestreo:")
print(f"  dt = {dt_mean:.4f} ± {dt_std:.6f} s")
print(f"  Frecuencia: {1/dt_mean:.1f} Hz")

# Graficar
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Posición
axes[0].plot(t, y_sensor * 1000, 'b-', linewidth=0.8, label='y_sensor')
axes[0].plot(t, yd * 1000, 'r--', linewidth=0.8, alpha=0.6, label='yd (setpoint)')
axes[0].set_ylabel('Posición [mm]')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Datos del Control C (MONIT.txt)')

# Corriente
axes[1].plot(t, i, 'g-', linewidth=0.8, label='i (medida)')
axes[1].plot(t, col5, 'm--', linewidth=0.8, alpha=0.6, label='col5')
axes[1].axhline(0.1, color='r', linestyle=':', alpha=0.5, label='i_min (umbral)')
axes[1].set_ylabel('Corriente [A]')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Voltaje
axes[2].plot(t, u, 'orange', linewidth=0.8)
axes[2].axhline(V_REF_nominal, color='r', linestyle='--', alpha=0.5, label=f'V_REF={V_REF_nominal}V')
axes[2].set_ylabel('Voltaje [V]')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

# Resistencia aparente
axes[3].plot(t[mask_valid], R_app[mask_valid], 'purple', linewidth=0.8, alpha=0.7)
axes[3].axhline(R_mean, color='r', linestyle='--', alpha=0.7, label=f'R_media={R_mean:.2f}Ω')
axes[3].axhline(R_nominal, color='b', linestyle=':', alpha=0.5, label=f'R_nominal={R_nominal}Ω')
axes[3].set_ylabel('R aparente [Ω]')
axes[3].set_xlabel('Tiempo [s]')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_control_C.png', dpi=150, bbox_inches='tight')
print(f"\n📈 Gráfica guardada: analisis_control_C.png")
plt.show()

# Guardar datos procesados en formato compatible con sintonizar_observador_offline.py
output_file = Path(__file__).parent / "datos_control_C.csv"
np.savetxt(output_file, 
           np.column_stack([t, y_sensor, u, i]),
           header='t,y,u,i',
           delimiter=',',
           comments='')
print(f"💾 Datos procesados guardados: {output_file}")

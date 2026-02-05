"""
Evaluación Automática del Observador
======================================
Analiza MONIT_observador.txt y calcula métricas de desempeño
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Cargar datos
data_file = Path(__file__).parent / "MONIT_observador.txt"

if not data_file.exists():
    print(f"❌ No se encontró {data_file}")
    exit(1)

# Leer: t, yd, y_sensor, y_obs, dy_obs, ie, u, R_est
data = np.loadtxt(data_file, skiprows=1)

t = data[:, 0]
yd = data[:, 1]
y_sensor = data[:, 2]
y_obs = data[:, 3]
dy_obs = data[:, 4]
ie = data[:, 5]
u = data[:, 6]
R_est = data[:, 7]

# Métricas del observador
error = np.abs(y_sensor - y_obs) * 1000  # mm
mae = np.mean(error)
rmse = np.sqrt(np.mean((y_sensor - y_obs)**2)) * 1000
max_error = np.max(error)

# Métricas del control
error_control = np.abs(y_sensor - yd) * 1000
mae_control = np.mean(error_control)

# Drift de resistencia
R_inicial = R_est[10] if len(R_est) > 10 else R_est[0]
R_final = R_est[-1]
drift_R = R_final - R_inicial

# Estadísticas de R
R_mean = np.mean(R_est)
R_std = np.std(R_est)
R_min = np.min(R_est)
R_max = np.max(R_est)

print("="*60)
print("📊 EVALUACIÓN DEL OBSERVADOR")
print("="*60)
print(f"\n⏱️  Duración: {t[-1]:.1f}s ({len(t)} muestras)")

print(f"\n🎯 Control (Sensor):")
print(f"  MAE:  {mae_control:.2f} mm")
print(f"  Rango: {y_sensor.min()*1000:.1f} - {y_sensor.max()*1000:.1f} mm")

print(f"\n👁️  Observador (Estimación):")
print(f"  MAE:  {mae:.2f} mm")
print(f"  RMSE: {rmse:.2f} mm")
print(f"  Max Error: {max_error:.2f} mm")
print(f"  Rango: {y_obs.min()*1000:.1f} - {y_obs.max()*1000:.1f} mm")

print(f"\n🔌 Resistencia Estimada:")
print(f"  R_inicial: {R_inicial:.2f} Ω")
print(f"  R_final:   {R_final:.2f} Ω")
print(f"  Drift:     {drift_R:+.2f} Ω ({100*drift_R/R_inicial:+.1f}%)")
print(f"  R_mean:    {R_mean:.2f} ± {R_std:.2f} Ω")
print(f"  Rango:     {R_min:.2f} - {R_max:.2f} Ω")

# Evaluación cualitativa
print(f"\n📈 Evaluación:")
if mae < 1.0:
    print(f"  ✅ EXCELENTE - MAE < 1mm")
elif mae < 2.0:
    print(f"  ✅ BUENO - MAE < 2mm")
elif mae < 5.0:
    print(f"  ⚠️  REGULAR - MAE < 5mm")
else:
    print(f"  ❌ MALO - MAE > 5mm (no usable)")

if abs(drift_R) < 0.5:
    print(f"  ✅ Sin drift de resistencia")
elif abs(drift_R) < 1.0:
    print(f"  ⚠️  Drift moderado de resistencia")
else:
    print(f"  ❌ Drift significativo de resistencia")

# Detectar fases
print(f"\n🔍 Análisis temporal:")
# Primera mitad vs segunda mitad
mid = len(t) // 2
mae_1h = np.mean(error[:mid])
mae_2h = np.mean(error[mid:])
print(f"  MAE (0-{t[mid]:.1f}s):  {mae_1h:.2f} mm")
print(f"  MAE ({t[mid]:.1f}-{t[-1]:.1f}s): {mae_2h:.2f} mm")

if mae_2h < mae_1h * 0.8:
    print(f"  → Observador converge ✅")
elif mae_2h > mae_1h * 1.2:
    print(f"  → Observador diverge ❌")
else:
    print(f"  → Observador estable")

# Gráfica
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Posición
axes[0].plot(t, y_sensor*1000, 'b-', linewidth=1, alpha=0.7, label='Sensor')
axes[0].plot(t, y_obs*1000, 'r-', linewidth=1, alpha=0.7, label='Observador')
axes[0].plot(t, yd*1000, 'g--', linewidth=1, alpha=0.5, label='Referencia')
axes[0].set_ylabel('Posición [mm]')
axes[0].set_title(f'Observador - MAE={mae:.2f}mm, RMSE={rmse:.2f}mm')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Error del observador
axes[1].plot(t, error, 'purple', linewidth=0.8)
axes[1].axhline(mae, color='r', linestyle='--', linewidth=1, label=f'MAE={mae:.2f}mm')
axes[1].fill_between(t, 0, error, alpha=0.3, color='purple')
axes[1].set_ylabel('Error Absoluto [mm]')
axes[1].set_title('Error del Observador |Sensor - Obs|')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Resistencia estimada
axes[2].plot(t, R_est, 'orange', linewidth=1)
axes[2].axhline(R_inicial, color='g', linestyle=':', alpha=0.5, label=f'R_inicial={R_inicial:.2f}Ω')
axes[2].axhline(R_final, color='r', linestyle=':', alpha=0.5, label=f'R_final={R_final:.2f}Ω')
axes[2].set_ylabel('Resistencia [Ω]')
axes[2].set_xlabel('Tiempo [s]')
axes[2].set_title(f'Resistencia Estimada (Drift={drift_R:+.2f}Ω)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluacion_observador.png', dpi=150, bbox_inches='tight')
print(f"\n💾 Gráfica guardada: evaluacion_observador.png")
plt.show()

# Guardar reporte
with open('reporte_observador.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("REPORTE DE EVALUACIÓN DEL OBSERVADOR\n")
    f.write("="*60 + "\n\n")
    f.write(f"Duración: {t[-1]:.1f}s\n")
    f.write(f"Muestras: {len(t)}\n\n")
    
    f.write("Control (Sensor):\n")
    f.write(f"  MAE: {mae_control:.2f} mm\n\n")
    
    f.write("Observador:\n")
    f.write(f"  MAE:  {mae:.2f} mm\n")
    f.write(f"  RMSE: {rmse:.2f} mm\n")
    f.write(f"  Max:  {max_error:.2f} mm\n\n")
    
    f.write("Resistencia:\n")
    f.write(f"  Inicial: {R_inicial:.2f} Ω\n")
    f.write(f"  Final:   {R_final:.2f} Ω\n")
    f.write(f"  Drift:   {drift_R:+.2f} Ω\n")
    f.write(f"  Media:   {R_mean:.2f} ± {R_std:.2f} Ω\n\n")
    
    f.write("Conclusión:\n")
    if mae < 2.0 and abs(drift_R) < 0.5:
        f.write("  ✅ Observador funcional - listo para modo sensorless\n")
    elif mae < 5.0:
        f.write("  ⚠️  Observador requiere ajuste de parámetros\n")
    else:
        f.write("  ❌ Observador no funcional - revisar modelo\n")

print(f"💾 Reporte guardado: reporte_observador.txt")

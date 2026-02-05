"""
Caracterización de la Resistencia desde datos del control C
============================================================

Objetivo: Extraer la dinámica térmica de la resistencia R=U/I
para identificar parámetros alpha y beta del modelo:

    dR/dt = alpha * P - beta * (R - R_amb)
    donde P = R * i^2

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# Cargar datos del control C
data_file = Path(__file__).parent.parent / "levitador valentin" / "MONIT.txt"
data = np.loadtxt(data_file)

t = data[:, 0]
y = data[:, 2]  # posición
i = data[:, 3]  # corriente
u = data[:, 5]  # voltaje

# Calcular resistencia R = U/I (solo cuando i > 0.1A)
mask = i > 0.1
t_valid = t[mask]
i_valid = i[mask]
u_valid = u[mask]
R = u_valid / i_valid

# Calcular potencia P = R*i^2
P = R * i_valid**2

print(f"📊 Análisis de Resistencia - Datos del Control C")
print(f"="*60)
print(f"Duración: {t[-1]:.1f}s")
print(f"Muestras válidas (i>0.1A): {len(R)}/{len(t)} ({100*len(R)/len(t):.1f}%)")
print(f"\n📈 Estadísticas de Resistencia:")
print(f"  R_min:  {R.min():.2f} Ω")
print(f"  R_max:  {R.max():.2f} Ω")
print(f"  R_mean: {R.mean():.2f} Ω")
print(f"  R_std:  {R.std():.2f} Ω")
print(f"\n⚡ Estadísticas de Potencia:")
print(f"  P_min:  {P.min():.2f} W")
print(f"  P_max:  {P.max():.2f} W")
print(f"  P_mean: {P.mean():.2f} W")

# Calcular dR/dt (derivada numérica)
dt_samples = np.diff(t_valid)
dR_dt = np.diff(R) / dt_samples

# Tomar valores centrales para alinear arrays
t_deriv = t_valid[:-1]
R_deriv = R[:-1]
P_deriv = P[:-1]

# Identificar si hay tendencia térmica
# Modelo: dR/dt = alpha*P - beta*(R - R_amb)
# Simplificado: dR/dt ≈ alpha*P  (si beta es pequeño)

# Filtrar outliers en dR/dt
dR_dt_median = np.median(dR_dt)
dR_dt_std = np.std(dR_dt)
mask_inliers = np.abs(dR_dt - dR_dt_median) < 3*dR_dt_std

t_fit = t_deriv[mask_inliers]
dR_dt_fit = dR_dt[mask_inliers]
P_fit = P_deriv[mask_inliers]
R_fit = R_deriv[mask_inliers]

print(f"\n🔍 Análisis de dR/dt:")
print(f"  dR/dt_mean: {dR_dt.mean():.4f} Ω/s")
print(f"  dR/dt_std:  {dR_dt.std():.4f} Ω/s")
print(f"  Inliers: {len(dR_dt_fit)}/{len(dR_dt)} ({100*len(dR_dt_fit)/len(dR_dt):.1f}%)")

# Ajustar modelo térmico completo
def modelo_termico(X, alpha, beta, R_amb):
    """dR/dt = alpha*P - beta*(R - R_amb)"""
    P, R = X
    return alpha * P - beta * (R - R_amb)

try:
    # Intentar ajuste completo
    popt, pcov = curve_fit(
        modelo_termico, 
        (P_fit, R_fit), 
        dR_dt_fit,
        p0=[0.01, 0.02, 16.0],  # Valores iniciales razonables
        maxfev=5000
    )
    
    alpha_opt, beta_opt, R_amb_opt = popt
    perr = np.sqrt(np.diag(pcov))
    
    print(f"\n✅ Modelo Térmico Identificado:")
    print(f"  alpha (calentamiento): {alpha_opt:.6f} ± {perr[0]:.6f}")
    print(f"  beta  (enfriamiento):  {beta_opt:.6f} ± {perr[1]:.6f}")
    print(f"  R_amb (ambiente):      {R_amb_opt:.2f} ± {perr[2]:.2f} Ω")
    
    # Calcular bondad de ajuste
    dR_dt_pred = modelo_termico((P_fit, R_fit), *popt)
    residuos = dR_dt_fit - dR_dt_pred
    rmse = np.sqrt(np.mean(residuos**2))
    r2 = 1 - (np.sum(residuos**2) / np.sum((dR_dt_fit - dR_dt_fit.mean())**2))
    
    print(f"\n📊 Bondad de Ajuste:")
    print(f"  RMSE: {rmse:.6f} Ω/s")
    print(f"  R²:   {r2:.4f}")
    
    model_fitted = True
    
except Exception as e:
    print(f"\n❌ No se pudo ajustar modelo completo: {e}")
    alpha_opt = beta_opt = R_amb_opt = None
    model_fitted = False

# Graficar
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# 1. Resistencia vs Tiempo
axes[0, 0].plot(t_valid, R, 'b-', alpha=0.6, linewidth=0.5)
axes[0, 0].axhline(R.mean(), color='r', linestyle='--', label=f'Media={R.mean():.2f}Ω')
axes[0, 0].set_xlabel('Tiempo [s]')
axes[0, 0].set_ylabel('Resistencia [Ω]')
axes[0, 0].set_title('Resistencia R = U/I')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Potencia vs Tiempo
axes[0, 1].plot(t_valid, P, 'orange', alpha=0.6, linewidth=0.5)
axes[0, 1].axhline(P.mean(), color='r', linestyle='--', label=f'Media={P.mean():.2f}W')
axes[0, 1].set_xlabel('Tiempo [s]')
axes[0, 1].set_ylabel('Potencia [W]')
axes[0, 1].set_title('Potencia Disipada P = R·i²')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. dR/dt vs Tiempo
axes[1, 0].plot(t_deriv, dR_dt, 'g.', alpha=0.3, markersize=2, label='Todos')
axes[1, 0].plot(t_fit, dR_dt_fit, 'b.', markersize=3, label='Inliers')
axes[1, 0].axhline(0, color='k', linestyle=':', alpha=0.5)
axes[1, 0].set_xlabel('Tiempo [s]')
axes[1, 0].set_ylabel('dR/dt [Ω/s]')
axes[1, 0].set_title('Derivada de Resistencia')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. dR/dt vs Potencia
axes[1, 1].scatter(P_fit, dR_dt_fit, c=t_fit, cmap='viridis', s=5, alpha=0.5)
if model_fitted:
    P_range = np.linspace(P_fit.min(), P_fit.max(), 50)
    R_range = np.full_like(P_range, R_amb_opt)
    dR_dt_model = modelo_termico((P_range, R_range), alpha_opt, beta_opt, R_amb_opt)
    axes[1, 1].plot(P_range, dR_dt_model, 'r-', linewidth=2, label='Modelo')
axes[1, 1].set_xlabel('Potencia [W]')
axes[1, 1].set_ylabel('dR/dt [Ω/s]')
axes[1, 1].set_title('dR/dt vs Potencia')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Tiempo [s]')

# 5. Corriente vs Voltaje (coloreado por R)
sc = axes[2, 0].scatter(i_valid, u_valid, c=R, cmap='coolwarm', s=3, alpha=0.6)
axes[2, 0].set_xlabel('Corriente [A]')
axes[2, 0].set_ylabel('Voltaje [V]')
axes[2, 0].set_title('U vs I (color = R)')
axes[2, 0].grid(True, alpha=0.3)
plt.colorbar(sc, ax=axes[2, 0], label='R [Ω]')

# 6. Histograma de R
axes[2, 1].hist(R, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[2, 1].axvline(R.mean(), color='r', linestyle='--', linewidth=2, label=f'Media={R.mean():.2f}Ω')
axes[2, 1].axvline(R.mean()-R.std(), color='orange', linestyle=':', label=f'±σ')
axes[2, 1].axvline(R.mean()+R.std(), color='orange', linestyle=':')
axes[2, 1].set_xlabel('Resistencia [Ω]')
axes[2, 1].set_ylabel('Frecuencia')
axes[2, 1].set_title('Distribución de Resistencia')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('caracterizacion_resistencia_C.png', dpi=150, bbox_inches='tight')
print(f"\n💾 Gráfica guardada: caracterizacion_resistencia_C.png")
plt.show()

# Guardar resumen
with open('resumen_resistencia_C.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("CARACTERIZACIÓN DE RESISTENCIA - Control C\n")
    f.write("="*60 + "\n\n")
    f.write(f"Datos: {len(R)}/{len(t)} muestras válidas\n")
    f.write(f"Duración: {t[-1]:.1f}s\n\n")
    f.write("Estadísticas R:\n")
    f.write(f"  Min:  {R.min():.2f} Ω\n")
    f.write(f"  Max:  {R.max():.2f} Ω\n")
    f.write(f"  Mean: {R.mean():.2f} Ω\n")
    f.write(f"  Std:  {R.std():.2f} Ω\n\n")
    
    if model_fitted:
        f.write("Modelo Térmico (dR/dt = alpha*P - beta*(R-R_amb)):\n")
        f.write(f"  alpha: {alpha_opt:.6f} ± {perr[0]:.6f}\n")
        f.write(f"  beta:  {beta_opt:.6f} ± {perr[1]:.6f}\n")
        f.write(f"  R_amb: {R_amb_opt:.2f} ± {perr[2]:.2f} Ω\n")
        f.write(f"  RMSE:  {rmse:.6f} Ω/s\n")
        f.write(f"  R²:    {r2:.4f}\n\n")
        f.write("RECOMENDACIÓN PARA EstimadorResistencia:\n")
        f.write(f"  R0 = {R_amb_opt:.2f}  # Resistencia ambiente\n")
        f.write(f"  alpha = {alpha_opt:.6f}  # Coef. calentamiento\n")
        f.write(f"  beta = {beta_opt:.6f}   # Coef. enfriamiento\n")
    else:
        f.write("No se pudo ajustar modelo térmico.\n")
        f.write("Recomendación: usar modo adaptativo puro (alpha=0, beta pequeño)\n")

print(f"💾 Resumen guardado: resumen_resistencia_C.txt")

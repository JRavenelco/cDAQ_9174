import numpy as np

# Cargar datos
data = np.loadtxt('MONIT_obs2023_mon.txt')
t, yd, y_s, y_e, v_e, ie, u, phi, R_est = data.T

print('='*60)
print('=== OBSERVADOR 2023 (José Santana - Sensorless) ===')
print('='*60)

# Filtrar período estable (después de transitorios, antes de caída)
mask = (t > 5) & (t < 18)
y_s_est = y_s[mask]
y_e_est = y_e[mask]

e_obs = np.abs(y_s_est - y_e_est)
corr = np.corrcoef(y_s_est, y_e_est)[0,1]

print(f'\n📊 Métricas (período estable 5-18s):')
print(f'  MAE:  {e_obs.mean()*1000:.2f} mm')
print(f'  RMSE: {np.sqrt((e_obs**2).mean())*1000:.2f} mm')
print(f'  Max Error: {e_obs.max()*1000:.2f} mm')
print(f'  Correlación: {corr:.4f}')

# Estadísticas de flujo
print(f'\n🔋 Flujo magnético φ:')
print(f'  φ_inicial: {phi[0]:.4f} Wb')
print(f'  φ_final:   {phi[-1]:.4f} Wb')
print(f'  φ_mean:    {phi[mask].mean():.4f} Wb')
print(f'  φ_std:     {phi[mask].std():.4f} Wb')

# Estadísticas de resistencia
print(f'\n🔌 Resistencia estimada R:')
print(f'  R_inicial: {R_est[int(len(R_est)*0.1)]:.2f} Ω')
print(f'  R_final:   {R_est[-1]:.2f} Ω')
print(f'  R_drift:   {R_est[-1] - R_est[int(len(R_est)*0.1)]:+.2f} Ω')
print(f'  R_mean:    {R_est[mask].mean():.2f} ± {R_est[mask].std():.2f} Ω')

# Comparación con benchmarks
print(f'\n--- vs Observadores Probados ---')
print(f'ObservadorLuenberger:  MAE = 0.27 mm, Corr = 0.997 ✅')
print(f'ObservadorClasico:     MAE = 0.50 mm, Corr = 0.990 ✅')
print(f'Observador2023:        MAE = {e_obs.mean()*1000:.2f} mm, Corr = {corr:.4f}')

diff_luenberger = e_obs.mean()*1000 - 0.27
diff_clasico = e_obs.mean()*1000 - 0.50

if e_obs.mean()*1000 < 0.5:
    print(f'\n🏆 Obs2023 EXCELENTE - Mejor que ObservadorClasico!')
    print(f'   Δ vs Luenberger = {diff_luenberger:+.2f} mm')
elif e_obs.mean()*1000 < 1.0:
    print(f'\n✅ Obs2023 BUENO - Comparable a ObservadorClasico')
    print(f'   Δ vs Clasico = {diff_clasico:+.2f} mm')
else:
    print(f'\n⚠️ Obs2023 necesita ajuste')
    print(f'   Factor de mejora vs Luenberger: {e_obs.mean()*1000/0.27:.1f}x peor')

# Drift check
drift_phi = phi[-1] - phi[int(len(phi)*0.3)]
print(f'\n🔍 Drift del flujo:')
print(f'   Δφ (últimos 70%): {drift_phi:+.4f} Wb')
if abs(drift_phi) < 0.01:
    print(f'   ✅ Drift controlado (<0.01)')
else:
    print(f'   ⚠️ Drift significativo (>{drift_phi:.4f})')

print(f'\n💾 Datos guardados en: MONIT_obs2023_mon.txt')
print('='*60)

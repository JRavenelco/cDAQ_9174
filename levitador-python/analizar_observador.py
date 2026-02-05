import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analizar_resultados(archivo="MONIT_observador.txt"):
    try:
        # Cargar datos
        # El archivo puede tener líneas comentadas con #
        df = pd.read_csv(archivo, sep='\t', comment='#', header=None)
        
        # Asignar nombres de columnas si no están en el header
        # Según control_observador.py: t, yd, y_sensor, y_obs, dy_obs, ie, u, R_est
        if df.shape[1] == 8:
            df.columns = ['t', 'yd', 'y_sensor', 'y_obs', 'dy_obs', 'ie', 'u', 'R_est']
        elif df.shape[1] == 7:
             df.columns = ['t', 'yd', 'y_sensor', 'y_obs', 'dy_obs', 'ie', 'u']
             df['R_est'] = np.nan
        else:
            print(f"Formato desconocido con {df.shape[1]} columnas")
            return

        print(f"Cargados {len(df)} puntos.")

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        # 1. Posición
        ax1 = axes[0]
        ax1.plot(df['t'], df['yd']*1000, 'k--', label='Ref (yd)', alpha=0.5)
        ax1.plot(df['t'], df['y_sensor']*1000, 'b-', label='Sensor', linewidth=1)
        ax1.plot(df['t'], df['y_obs']*1000, 'r-', label='Observador', linewidth=1)
        ax1.set_ylabel('Posición [mm]')
        ax1.set_title('Comparación: Sensor vs Observador Adaptativo')
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim(-1, 25) # Rango típico

        # 2. Resistencia Estimada
        ax2 = axes[1]
        ax2.plot(df['t'], df['R_est'], 'g-', label='R Estimada')
        ax2.set_ylabel('Resistencia [Ω]')
        ax2.set_title('Estimación Dinámica de Resistencia')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Corriente y Voltaje
        ax3 = axes[2]
        ax3.plot(df['t'], df['ie'], 'm-', label='Corriente [A]')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['t'], df['u'], 'c-', label='Voltaje [V]', alpha=0.5)
        ax3.set_ylabel('Corriente [A]', color='m')
        ax3_twin.set_ylabel('Voltaje [V]', color='c')
        ax3.set_xlabel('Tiempo [s]')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True)

        plt.tight_layout()
        output_file = archivo.replace('.txt', '.png')
        plt.savefig(output_file)
        print(f"Gráfica guardada en {output_file}")
        
        # Estadísticas
        error = (df['y_sensor'] - df['y_obs']) * 1000
        mae = np.mean(np.abs(error))
        print(f"\nEstadísticas:")
        print(f"MAE (Sensor - Obs): {mae:.2f} mm")
        print(f"R final: {df['R_est'].iloc[-1]:.2f} Ω")
        print(f"R promedio: {df['R_est'].mean():.2f} Ω")

    except Exception as e:
        print(f"Error analizando: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analizar_resultados()


import pandas as pd
import matplotlib.pyplot as plt
import sys

# Cargar archivo
file = "datos_termicos_1766190080.csv"
try:
    df = pd.read_csv(file)
    
    # Filtrar valores locos
    df = df[(df['r_med'] > 0) & (df['r_med'] < 30)]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(df['t'], df['r_med'], 'b.', label='R Measured')
    plt.ylabel('Resistance [Ohm]')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df['t'], df['u'], 'r-', label='Voltage [V]')
    plt.ylabel('Voltage [V]')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(df['t'], df['i'], 'g-', label='Current [A]')
    plt.ylabel('Current [A]')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('debug_termico.png')
    print("Gráfica guardada en debug_termico.png")
    
except Exception as e:
    print(f"Error: {e}")


import pandas as pd
import numpy as np

file = "datos_termicos_1766190080.csv"
df = pd.read_csv(file)

# Filtrar fase heat
heat = df[df['fase'] == 'heat']
cool = df[df['fase'] == 'cool']

print(f"Datos Heat: {len(heat)}")
print(f"Datos Cool: {len(cool)}")

if len(heat) > 0:
    r_start = heat['r_med'].iloc[:50].mean() # Promedio primeros 50 ptos
    r_end = heat['r_med'].iloc[-50:].mean()   # Promedio últimos 50 ptos
    
    print(f"Resistencia Inicio Heat: {r_start:.4f} Ohm")
    print(f"Resistencia Final Heat:  {r_end:.4f} Ohm")
    print(f"Delta R: {r_end - r_start:.4f} Ohm")
    
    if r_end > r_start:
        print("✅ Se detecta calentamiento.")
    else:
        print("⚠️ NO se detecta aumento de resistencia claro.")

if len(cool) > 0:
    r_c_start = cool['r_med'].iloc[:50].mean()
    r_c_end = cool['r_med'].iloc[-50:].mean()
    print(f"Resistencia Inicio Cool: {r_c_start:.4f} Ohm")
    print(f"Resistencia Final Cool:  {r_c_end:.4f} Ohm")

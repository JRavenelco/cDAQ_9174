import pandas as pd
import matplotlib.pyplot as plt

# Nombre del archivo CSV que generaste
CSV_FILE = "datos_osciloscopio_20250309_135730.csv"  # Ajusta con tu nombre real

def main():
    # 1. Leer el CSV con pandas
    #    - header=0 indica que la primera fila es la cabecera con nombres de columnas
    #    - delimiter="," es el separador por defecto, pero se puede especificar si fuera necesario
    df = pd.read_csv(CSV_FILE, header=0)
    
    # 2. Mostrar información básica
    print("\n=== Primeras filas del CSV ===")
    print(df.head())  # Muestra las 5 primeras filas
    
    print("\n=== Resumen estadístico ===")
    print(df.describe())  # Estadísticas (mínimo, máximo, media, etc.)
    
    # 3. Graficar los datos
    plt.figure(figsize=(10, 6))
    
    # Asumiendo que las columnas se llaman "Canal_ai0", "Canal_ai1", etc.
    # Si tus columnas tienen otros nombres, ajusta aquí
    for col in df.columns:
        plt.plot(df[col], label=col)
    
    plt.title("Datos de Registro - Archivo CSV")
    plt.xlabel("Muestra (índice)")
    plt.ylabel("Voltaje (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 4. Mostrar la gráfica
    plt.show()

if __name__ == "__main__":
    main()

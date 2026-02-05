#!/usr/bin/env python3
"""
Herramienta para visualizar datos CSV capturados por interfaz_DAQ_V2.py
Permite cargar y analizar datos sin ejecutar el sistema de adquisición.

Uso:
    python visualizar_datos_capturados.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import os
import sys
import glob
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.fft import fft, fftfreq
from scipy import signal

class DataViewer:
    def __init__(self):
        self.fig = None
        self.axes = []
        self.data_fuerza = None
        self.data_vibracion = None
        self.current_files = {"fuerza": None, "vibracion": None}
        
    def select_files(self):
        """Seleccionar archivos CSV para cargar"""
        root = tk.Tk()
        root.withdraw()
        
        # Directorio por defecto donde se guardan los datos
        default_dir = os.path.join(os.path.dirname(__file__), "datos_automaticos")
        if not os.path.exists(default_dir):
            default_dir = os.path.dirname(__file__)
        
        print("Seleccione el archivo de datos de fuerza (o cancele si no tiene)...")
        file_fuerza = filedialog.askopenfilename(
            title="Seleccionar archivo de fuerza",
            initialdir=default_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv"
        )
        
        print("Seleccione el archivo de datos de vibración (o cancele si no tiene)...")
        file_vibracion = filedialog.askopenfilename(
            title="Seleccionar archivo de vibración", 
            initialdir=default_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv"
        )
        
        root.destroy()
        
        if not file_fuerza and not file_vibracion:
            print("No se seleccionaron archivos. Saliendo...")
            return False
            
        return self.load_files(file_fuerza, file_vibracion)
    
    def load_files(self, file_fuerza=None, file_vibracion=None):
        """Cargar archivos CSV seleccionados"""
        success = False
        
        if file_fuerza and os.path.exists(file_fuerza):
            try:
                print(f"Cargando datos de fuerza: {os.path.basename(file_fuerza)}")
                self.data_fuerza = pd.read_csv(file_fuerza)
                self.current_files["fuerza"] = file_fuerza
                print(f"  ✓ {len(self.data_fuerza)} muestras de fuerza cargadas")
                success = True
            except Exception as e:
                print(f"Error cargando archivo de fuerza: {e}")
        
        if file_vibracion and os.path.exists(file_vibracion):
            try:
                print(f"Cargando datos de vibración: {os.path.basename(file_vibracion)}")
                self.data_vibracion = pd.read_csv(file_vibracion)
                self.current_files["vibracion"] = file_vibracion
                print(f"  ✓ {len(self.data_vibracion)} muestras de vibración cargadas")
                success = True
            except Exception as e:
                print(f"Error cargando archivo de vibración: {e}")
        
        return success
    
    def calculate_sample_rate(self, data):
        """Calcular frecuencia de muestreo basada en la columna de tiempo"""
        if len(data) < 2:
            return 1000  # Default fallback
        
        time_col = data.columns[0]  # Primera columna debería ser tiempo
        dt = data[time_col].iloc[1] - data[time_col].iloc[0]
        return 1.0 / dt if dt > 0 else 1000
    
    def plot_time_series(self):
        """Visualizar series de tiempo"""
        if self.data_fuerza is None and self.data_vibracion is None:
            print("No hay datos cargados para visualizar")
            return
        
        # Configurar subplots
        n_plots = 0
        if self.data_fuerza is not None:
            n_plots += 1
        if self.data_vibracion is not None:
            n_plots += 1
        
        self.fig, self.axes = plt.subplots(n_plots, 2, figsize=(15, 4*n_plots))
        if n_plots == 1:
            self.axes = self.axes.reshape(1, -1)
        
        plot_idx = 0
        
        # Plot datos de fuerza
        if self.data_fuerza is not None:
            self._plot_dataset(self.data_fuerza, "Fuerza", plot_idx, "N")
            plot_idx += 1
        
        # Plot datos de vibración  
        if self.data_vibracion is not None:
            self._plot_dataset(self.data_vibracion, "Vibración", plot_idx, "g")
            plot_idx += 1
        
        plt.tight_layout()
        plt.show()
    
    def _plot_dataset(self, data, title, row_idx, unit):
        """Plotear un dataset específico (fuerza o vibración)"""
        time_col = data.columns[0]
        data_cols = data.columns[1:]
        
        # Calcular frecuencia de muestreo
        fs = self.calculate_sample_rate(data)
        
        # Plot series de tiempo
        ax_time = self.axes[row_idx, 0]
        for col in data_cols:
            ax_time.plot(data[time_col], data[col], label=col, alpha=0.7)
        
        ax_time.set_xlabel('Tiempo (s)')
        ax_time.set_ylabel(f'{title} ({unit})')
        ax_time.set_title(f'{title} - Serie de Tiempo\nFs: {fs:.0f} Hz, Muestras: {len(data):,}')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        # Plot FFT
        ax_fft = self.axes[row_idx, 1]
        for col in data_cols:
            y = data[col].values
            # Calcular FFT
            Y = np.abs(fft(y - np.mean(y)))
            freqs = fftfreq(len(y), 1/fs)
            
            # Solo frecuencias positivas
            pos_mask = freqs > 0
            ax_fft.semilogy(freqs[pos_mask], Y[pos_mask], label=col, alpha=0.7)
        
        ax_fft.set_xlabel('Frecuencia (Hz)')
        ax_fft.set_ylabel('Magnitud FFT')
        ax_fft.set_title(f'{title} - Espectro FFT')
        ax_fft.legend()
        ax_fft.grid(True, alpha=0.3)
    
    def show_statistics(self):
        """Mostrar estadísticas básicas de los datos"""
        print("\n" + "="*50)
        print("ESTADÍSTICAS DE DATOS CARGADOS")
        print("="*50)
        
        if self.data_fuerza is not None:
            print(f"\n📊 DATOS DE FUERZA:")
            print(f"   Archivo: {os.path.basename(self.current_files['fuerza'])}")
            print(f"   Muestras: {len(self.data_fuerza):,}")
            fs = self.calculate_sample_rate(self.data_fuerza)
            print(f"   Frecuencia muestreo: {fs:.0f} Hz")
            duration = len(self.data_fuerza) / fs
            print(f"   Duración: {duration:.1f} segundos")
            
            print("\n   Estadísticas por canal:")
            for col in self.data_fuerza.columns[1:]:
                data_col = self.data_fuerza[col]
                print(f"     {col}: μ={data_col.mean():.3f}, σ={data_col.std():.3f}, "
                      f"min={data_col.min():.3f}, max={data_col.max():.3f}")
        
        if self.data_vibracion is not None:
            print(f"\n📈 DATOS DE VIBRACIÓN:")
            print(f"   Archivo: {os.path.basename(self.current_files['vibracion'])}")
            print(f"   Muestras: {len(self.data_vibracion):,}")
            fs = self.calculate_sample_rate(self.data_vibracion)
            print(f"   Frecuencia muestreo: {fs:.0f} Hz")
            duration = len(self.data_vibracion) / fs
            print(f"   Duración: {duration:.1f} segundos")
            
            print("\n   Estadísticas por canal:")
            for col in self.data_vibracion.columns[1:]:
                data_col = self.data_vibracion[col]
                print(f"     {col}: μ={data_col.mean():.3f}, σ={data_col.std():.3f}, "
                      f"min={data_col.min():.3f}, max={data_col.max():.3f}")
    
    def export_analysis(self):
        """Exportar análisis básico a archivo de texto"""
        if self.data_fuerza is None and self.data_vibracion is None:
            print("No hay datos para exportar")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analisis_datos_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("ANÁLISIS DE DATOS CAPTURADOS\n")
            f.write("="*50 + "\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if self.data_fuerza is not None:
                f.write("DATOS DE FUERZA:\n")
                f.write(f"Archivo: {self.current_files['fuerza']}\n")
                f.write(f"Muestras: {len(self.data_fuerza):,}\n")
                fs = self.calculate_sample_rate(self.data_fuerza)
                f.write(f"Frecuencia: {fs:.0f} Hz\n")
                f.write(f"Duración: {len(self.data_fuerza)/fs:.1f} segundos\n\n")
                f.write(self.data_fuerza.describe().to_string())
                f.write("\n\n")
            
            if self.data_vibracion is not None:
                f.write("DATOS DE VIBRACIÓN:\n")
                f.write(f"Archivo: {self.current_files['vibracion']}\n")
                f.write(f"Muestras: {len(self.data_vibracion):,}\n")
                fs = self.calculate_sample_rate(self.data_vibracion)
                f.write(f"Frecuencia: {fs:.0f} Hz\n")
                f.write(f"Duración: {len(self.data_vibracion)/fs:.1f} segundos\n\n")
                f.write(self.data_vibracion.describe().to_string())
        
        print(f"Análisis exportado a: {filename}")

def main():
    """Función principal"""
    print("🔍 VISUALIZADOR DE DATOS CAPTURADOS")
    print("="*40)
    
    viewer = DataViewer()
    
    # Cargar archivos
    if not viewer.select_files():
        return
    
    # Mostrar estadísticas
    viewer.show_statistics()
    
    # Menú interactivo
    while True:
        print("\n" + "-"*40)
        print("OPCIONES:")
        print("1. Visualizar gráficas")
        print("2. Mostrar estadísticas")
        print("3. Exportar análisis")
        print("4. Cargar otros archivos") 
        print("5. Salir")
        
        try:
            opcion = input("\nSeleccione opción (1-5): ").strip()
            
            if opcion == "1":
                viewer.plot_time_series()
            elif opcion == "2":
                viewer.show_statistics()
            elif opcion == "3":
                viewer.export_analysis()
            elif opcion == "4":
                if viewer.select_files():
                    viewer.show_statistics()
            elif opcion == "5":
                break
            else:
                print("Opción inválida. Intente de nuevo.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n¡Hasta luego! 👋")

if __name__ == "__main__":
    main()

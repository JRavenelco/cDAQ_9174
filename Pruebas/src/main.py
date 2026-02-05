#%% [code]
import nidaqmx
from nidaqmx.system import System
import numpy as np
import time
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import tensorflow as tf
#%% [code]
# Crea una instancia del sistema local de NI-DAQmx
system = System.local()
print(tf.__version__) 
print(tf.config.list_physical_devices('GPU'))
#%% [code]

print("Dispositivos detectados:")
for device in system.devices:
    print(f"Nombre del dispositivo: {device.name}")
    print(f"Tipo de producto:       {device.product_type}")
    print(f"Número de serie:        {device.serial_num}")
    print("--------------------")
# %% [code]
# Configuración de la adquisición de datos analógicos
DISPOSITIVO = "cDAQ1Mod1"  # Ajusta según el nombre de tu dispositivo detectado
CANALES = ["ai0", "ai1"]  # Dos canales analógicos
TASA_MUESTREO = 1000  # Tasa de muestreo en Hz
TIEMPO_ADQUISICION = 60  # Tiempo de adquisición en segundos
NUM_MUESTRAS = TASA_MUESTREO * TIEMPO_ADQUISICION
VOLTAJE_MIN = -10.0
VOLTAJE_MAX = 10.0


# %% [code]
try:
    # Crear tarea para adquisición analógica
    with nidaqmx.Task() as tarea:
        # Añadir los dos canales analógicos
        for canal in CANALES:
            nombre_canal_completo = f"{DISPOSITIVO}/{canal}"
            print(f"Configurando canal: {nombre_canal_completo}")
            tarea.ai_channels.add_ai_voltage_chan(
                nombre_canal_completo,
                terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,  # Referencia a tierra (single-ended)
                min_val=VOLTAJE_MIN,
                max_val=VOLTAJE_MAX
            )
        
        # Configurar el timing para la adquisición
        tarea.timing.cfg_samp_clk_timing(
            rate=TASA_MUESTREO,
            sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
            samps_per_chan=NUM_MUESTRAS
        )
        
        print(f"\nAdquiriendo {NUM_MUESTRAS} muestras a {TASA_MUESTREO} Hz de {len(CANALES)} canales...")
        
        # Iniciar la adquisición y leer los datos
        datos = tarea.read(number_of_samples_per_channel=NUM_MUESTRAS)
        
        print("Adquisición completada!")
        
        # Convertir a array de NumPy para procesamiento
        datos_np = np.array(datos)
        
        # Crear vector de tiempo
        tiempo = np.linspace(0, TIEMPO_ADQUISICION, NUM_MUESTRAS)
        
        # Graficar los resultados
        plt.figure(figsize=(12, 8))
        for i, canal in enumerate(CANALES):
            plt.subplot(len(CANALES), 1, i+1)
            plt.plot(tiempo, datos_np[i], label=f"Canal {canal}")
            plt.title(f"Señal del canal {canal}")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Voltaje (V)")
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig("datos_adquiridos.png")
        plt.show()
        
        # Guardar los datos en un archivo CSV
        datos_transposed = np.transpose(datos_np)
        np.savetxt(
            "datos_adquiridos.csv", 
            datos_transposed, 
            delimiter=",", 
            header=",".join([f"Canal_{canal}" for canal in CANALES])
        )
        
        print("Datos guardados en 'datos_adquiridos.csv' y gráfica en 'datos_adquiridos.png'")
        
        # Análisis básico de los datos
        for i, canal in enumerate(CANALES):
            print(f"\nEstadísticas del Canal {canal}:")
            print(f"  Valor medio: {np.mean(datos_np[i]):.4f} V")
            print(f"  Valor máximo: {np.max(datos_np[i]):.4f} V")
            print(f"  Valor mínimo: {np.min(datos_np[i]):.4f} V")
            print(f"  Desviación estándar: {np.std(datos_np[i]):.4f} V")

except nidaqmx.errors.DaqError as e:
    print(f"Error de DAQmx: {e}")
    
    # Mostrar información más detallada para ayudar a solucionar el problema
    if "invalid physical channel" in str(e).lower():
        print("\nPosibles soluciones:")
        print("1. Verifica el nombre del dispositivo y los canales")
        print("2. Asegúrate de que el dispositivo esté correctamente conectado")
        print("3. Prueba con otro nombre de canal disponible")
    
    print("\nDispositivos detectados y sus canales:")
    for device in system.devices:
        print(f"Dispositivo: {device.name}")
        try:
            for channel in device.ai_physical_chans:
                print(f"  Canal AI: {channel.name}")
        except:
            print("  No se pudieron leer los canales analógicos de este dispositivo")
except Exception as e:
    print(f"Error general: {e}")

# %%

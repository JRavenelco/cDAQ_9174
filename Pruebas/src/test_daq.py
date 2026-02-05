import nidaqmx
import time
import numpy as np
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# Definir constantes
DISPOSITIVO = "cDAQ1Mod1"
CANALES = ["ai0", "ai1", "ai2", "ai3"]
SAMPLE_RATE = 1000
MUESTRAS_POR_BLOQUE = 100
FORCE_MIN_VOLTAGE = -1.5
FORCE_MAX_VOLTAGE = 1.5

print(f"Probando la lectura de {len(CANALES)} canales desde {DISPOSITIVO}")

try:
    # Crear y configurar la tarea
    task = nidaqmx.Task()
    
    # Añadir canales
    for canal in CANALES:
        nombre_canal = f"{DISPOSITIVO}/{canal}"
        print(f"Añadiendo canal: {nombre_canal}")
        task.ai_channels.add_ai_voltage_chan(
            nombre_canal,
            terminal_config=TerminalConfiguration.DIFF,
            min_val=FORCE_MIN_VOLTAGE,
            max_val=FORCE_MAX_VOLTAGE
        )
    
    # Configurar timing
    print("Configurando timing...")
    task.timing.cfg_samp_clk_timing(
        rate=SAMPLE_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=MUESTRAS_POR_BLOQUE * 10
    )
    
    # Iniciar tarea
    print("Iniciando tarea...")
    task.start()
    
    # Leer datos 3 veces
    for i in range(3):
        print(f"\nLectura #{i+1}:")
        try:
            print("Intentando leer datos...")
            datos = task.read(
                number_of_samples_per_channel=MUESTRAS_POR_BLOQUE,
                timeout=2.0
            )
            datos_np = np.array(datos)
            print(f"Datos leídos. Shape: {datos_np.shape}")
            
            # Mostrar muestras
            for j, canal in enumerate(CANALES):
                print(f"Canal {canal} - Primeras 3 muestras: {datos_np[j][:3]}")
                print(f"Canal {canal} - Estadísticas: min={np.min(datos_np[j]):.4f}, max={np.max(datos_np[j]):.4f}, mean={np.mean(datos_np[j]):.4f}")
            
        except Exception as read_ex:
            print(f"Error durante la lectura: {read_ex}")
        
        # Esperar un poco entre lecturas
        time.sleep(0.5)
    
except Exception as ex:
    print(f"Error general: {ex}")

finally:
    # Cerrar tarea si existe
    if 'task' in locals() and task is not None:
        try:
            task.stop()
            task.close()
            print("\nTarea detenida y cerrada correctamente.")
        except Exception as close_ex:
            print(f"\nError al cerrar la tarea: {close_ex}")

print("Prueba de DAQ finalizada.")

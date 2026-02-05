import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import time
import logging
import os
from datetime import datetime

# Configuración del logger
def setup_logger():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f'usb_capture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('usb_logger')

# Configuración de adquisición
DISPOSITIVO = "cDAQ1Mod1"
CANALES = ["ai0", "ai1", "ai2", "ai3"]
SAMPLE_RATE = 5000  # Reducir la tasa de muestreo para disminuir la carga
MUESTRAS_POR_BLOQUE = 500  # Reducir el tamaño del bloque para lecturas más frecuentes
VOLTAJE_MIN = -5.0
VOLTAJE_MAX = 5.0
# Aumentar el tamaño del buffer para evitar desbordamientos
BUFFER_SIZE_FACTOR = 50  # Buffer aún más grande

class AdquisicionThread:
    def __init__(self):
        self.task = None
        self.logger = setup_logger()
        self.logger.info("Iniciando AdquisicionThread para pruebas de captura USB")
        self.logger.info(f"Configuración: Dispositivo={DISPOSITIVO}, Canales={CANALES}")
        self.logger.info(f"Parámetros: SampleRate={SAMPLE_RATE}, MuestrasPorBloque={MUESTRAS_POR_BLOQUE}")
        self.logger.info(f"Rango: Voltaje Min={VOLTAJE_MIN}, Voltaje Max={VOLTAJE_MAX}")

    def run(self):
        try:
            self.logger.info("1. Creando tarea...")
            self.task = nidaqmx.Task()
            
            self.logger.info("2. Configurando canales...")
            for canal in CANALES:
                nombre_canal = f"{DISPOSITIVO}/{canal}"
                self.logger.info(f"   - Añadiendo canal {nombre_canal}...")
                self.task.ai_channels.add_ai_voltage_chan(
                    nombre_canal,
                    terminal_config=TerminalConfiguration.DIFF,
                    min_val=VOLTAJE_MIN,
                    max_val=VOLTAJE_MAX
                )
            
            self.logger.info("3. Configurando temporización...")
            # Configurar tiempo de adquisición y buffer
            self.task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=MUESTRAS_POR_BLOQUE * BUFFER_SIZE_FACTOR
            )
            
            # Configurar el tamaño del buffer explícitamente
            buffer_size = MUESTRAS_POR_BLOQUE * BUFFER_SIZE_FACTOR * len(CANALES)
            self.task.in_stream.input_buf_size = buffer_size
            self.logger.info(f"   - Tamaño de buffer configurado: {buffer_size} muestras")
            
            # Configurar política de lectura para evitar desbordamientos
            self.task.in_stream.auto_start = False
            self.task.in_stream.readall_overload = False
            
            # Configurar lectura relativa a la primera muestra disponible
            self.task.in_stream.relative_to = nidaqmx.constants.ReadRelativeTo.FIRST_SAMPLE
            self.task.in_stream.offset = 0
            
            # Configurar el número de muestras por canal por lectura
            self.task.in_stream.over_write = nidaqmx.constants.OverwriteMode.OVERWRITE_UNREAD_SAMPLES
            
            self.logger.info("4. Iniciando tarea...")
            self.task.start()
            
            self.logger.info("5. Leyendo datos (5 iteraciones)...")
            for i in range(5):
                try:
                    # Esperar un poco para que se llene el buffer
                    time.sleep(0.2)
                    
                    start_time = time.time()
                    
                    # Leer datos disponibles - utilizando el método read_analog_f64 directamente
                    # para mayor control y rendimiento
                    datos = self.task.read(
                        number_of_samples_per_channel=MUESTRAS_POR_BLOQUE,
                        timeout=10.0  # Timeout aún mayor
                    )
                    
                    end_time = time.time()
                    self.logger.info(f"   - Iteración {i+1}: {len(datos[0])} muestras leídas en {(end_time-start_time)*1000:.2f}ms")
                    
                    # Análisis estadístico básico de los datos capturados
                    if datos and len(datos) > 0:
                        for ch_idx, ch_data in enumerate(datos):
                            if len(ch_data) > 0:
                                self.logger.debug(f"     Canal {CANALES[ch_idx]}: Min={min(ch_data):.4f}V, Max={max(ch_data):.4f}V, Avg={sum(ch_data)/len(ch_data):.4f}V")
                    
                    # Dar más tiempo entre lecturas (ciclo completo)
                    time.sleep(0.8)
                    
                except Exception as e:
                    self.logger.error(f"Error en iteración {i+1}: {str(e)}")
                    # Continuar con la siguiente iteración
                    time.sleep(1)
            
            self.logger.info("6. Deteniendo tarea...")
            self.task.stop()
            
            self.logger.info("7. Cerrando tarea...")
            self.task.close()
            
            self.logger.info("Captura de comandos USB completada")
        
        except Exception as e:
            self.logger.error(f"Error durante la captura: {str(e)}", exc_info=True)
            if self.task:
                try:
                    self.task.close()
                except:
                    pass

# Añadir una función para implementar un enfoque basado en callback si el método anterior sigue fallando
def adquisicion_con_callback():
    """Implementación alternativa usando callbacks para una adquisición más robusta"""
    import threading
    
    logger = setup_logger("callback_logger")
    logger.info("Iniciando adquisición con callbacks")
    
    data_lock = threading.Lock()
    all_data = {canal: [] for canal in CANALES}
    stop_event = threading.Event()
    
    def callback(task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        """Callback que se ejecuta cuando hay nuevas muestras disponibles"""
        try:
            # Leer los datos disponibles
            datos = task.read(number_of_samples_per_channel=number_of_samples)
            
            # Almacenar datos en el buffer global
            with data_lock:
                for ch_idx, ch_data in enumerate(datos):
                    all_data[CANALES[ch_idx]].extend(ch_data)
            
            logger.debug(f"Callback: Leídas {number_of_samples} muestras de {len(CANALES)} canales")
            return 0
        except Exception as e:
            logger.error(f"Error en callback: {e}")
            return -1
    
    try:
        # Crear tarea
        task = nidaqmx.Task()
        
        # Configurar canales
        for canal in CANALES:
            nombre_canal = f"{DISPOSITIVO}/{canal}"
            task.ai_channels.add_ai_voltage_chan(
                nombre_canal,
                terminal_config=TerminalConfiguration.DIFF,
                min_val=VOLTAJE_MIN,
                max_val=VOLTAJE_MAX
            )
        
        # Configurar temporización
        task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=MUESTRAS_POR_BLOQUE * 10
        )
        
        # Registrar callback para que se ejecute cada MUESTRAS_POR_BLOQUE muestras
        task.register_every_n_samples_acquired_into_buffer_event(
            MUESTRAS_POR_BLOQUE,
            callback
        )
        
        # Iniciar tarea
        task.start()
        
        # Esperar mientras se adquieren datos (10 segundos)
        logger.info("Adquisición en curso, esperando 10 segundos...")
        for i in range(10):
            time.sleep(1)
            with data_lock:
                total_samples = len(next(iter(all_data.values())))
            logger.info(f"Total muestras adquiridas: {total_samples}")
        
        # Detener y cerrar tarea
        task.stop()
        task.close()
        
        # Guardar y analizar datos
        logger.info("Adquisición completada, analizando datos...")
        for canal, datos in all_data.items():
            if datos:
                logger.info(f"Canal {canal}: {len(datos)} muestras, "
                           f"Min={min(datos):.4f}V, Max={max(datos):.4f}V, "
                           f"Avg={sum(datos)/len(datos):.4f}V")
        
    except Exception as e:
        logger.error(f"Error en adquisición con callback: {e}", exc_info=True)
        if 'task' in locals():
            try:
                task.close()
            except:
                pass

if __name__ == "__main__":
    thread = AdquisicionThread()
    thread.run()
    
    # Si la adquisición estándar falla, descomenta la siguiente línea para probar el enfoque con callbacks
    # adquisicion_con_callback()

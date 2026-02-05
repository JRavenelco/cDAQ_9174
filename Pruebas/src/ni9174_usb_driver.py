import usb.core
import usb.util
import time
import numpy as np
import logging
import os
from datetime import datetime
from enum import Enum

# Configuración del logger
def setup_logger(name="ni9174_usb"):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s', 
                                       datefmt='%Y-%m-%d %H:%M:%S')
        
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

class TerminalConfiguration(Enum):
    """Emula la clase TerminalConfiguration de NI-DAQmx"""
    RSE = 0
    NRSE = 1
    DIFF = 2
    PSEUDODIFF = 3

class AcquisitionType(Enum):
    """Emula la clase AcquisitionType de NI-DAQmx"""
    FINITE = 0
    CONTINUOUS = 1
    HW_TIMED_SINGLE_POINT = 2

class NI9174Driver:
    """Controlador USB personalizado para NI-9174"""
    
    # Constantes del dispositivo (obtenidas de lsusb y Wireshark)
    VENDOR_ID = 0x3923  # National Instruments
    PRODUCT_ID = 0x74a5  # NI-9174 (¡verificar este valor con lsusb!)
    
    def __init__(self):
        self.logger = setup_logger()
        self.logger.info("Inicializando controlador USB para NI-9174")
        self.dev = None
        self.ep_in = None
        self.ep_out = None
        self.config = {
            'canales': [],
            'sample_rate': 0,
            'muestras_por_bloque': 0,
            'voltage_range': (-10.0, 10.0),
            'terminal_config': TerminalConfiguration.DIFF,
            'acquisition_mode': AcquisitionType.CONTINUOUS
        }
        self.is_task_running = False
        
    def find_device(self):
        """Busca y configura el dispositivo NI-9174"""
        self.logger.info(f"Buscando dispositivo NI-9174 (VID={self.VENDOR_ID:04x}, PID={self.PRODUCT_ID:04x})")
        self.dev = usb.core.find(idVendor=self.VENDOR_ID, idProduct=self.PRODUCT_ID)
        
        if self.dev is None:
            self.logger.error("Dispositivo NI-9174 no encontrado")
            raise ValueError("Dispositivo NI-9174 no encontrado")
            
        self.logger.info(f"Dispositivo encontrado: {self.dev}")
        
        # Resetear el dispositivo si está en uso
        if self.dev.is_kernel_driver_active(0):
            self.logger.info("Desactivando controlador del kernel")
            self.dev.detach_kernel_driver(0)
            
        # Configurar el dispositivo
        self.logger.info("Configurando dispositivo")
        self.dev.set_configuration()
        
        # Obtener la configuración activa
        cfg = self.dev.get_active_configuration()
        self.logger.debug(f"Configuración activa: {cfg}")
        
        # Obtener la interfaz (normalmente la primera)
        intf = cfg[(0, 0)]  # Interfaz 0, configuración alternativa 0
        self.logger.debug(f"Interfaz: {intf}")
        
        # Encontrar endpoints
        self.ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN
        )
        
        self.ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
        )
        
        if self.ep_in is None:
            self.logger.error("No se encontró endpoint IN")
            raise ValueError("No se encontró endpoint IN")
            
        self.logger.info(f"Endpoint IN: {self.ep_in}")
        if self.ep_out:
            self.logger.info(f"Endpoint OUT: {self.ep_out}")
        
        return True
        
    def send_control_command(self, request, value, index, data=None, timeout=1000):
        """Envía un comando de control USB al dispositivo"""
        request_type = 0x40  # Vendor request, OUT (ajustar según Wireshark)
        if data is None:
            data = bytearray()
            
        self.logger.debug(f"Enviando comando USB: req={request:02x}, val={value:04x}, idx={index:04x}, data={data.hex() if data else 'None'}")
        
        try:
            result = self.dev.ctrl_transfer(request_type, request, value, index, data, timeout)
            self.logger.debug(f"Resultado: {result}")
            return result
        except usb.core.USBError as e:
            self.logger.error(f"Error al enviar comando USB: {e}")
            raise
    
    def add_channel(self, channel_name, terminal_config=TerminalConfiguration.DIFF, min_val=-5.0, max_val=5.0):
        """Añade un canal para adquisición"""
        self.logger.info(f"Añadiendo canal: {channel_name}, config={terminal_config}, rango={min_val}V a {max_val}V")
        
        # Guardar configuración del canal
        self.config['canales'].append({
            'nombre': channel_name,
            'terminal_config': terminal_config,
            'min_val': min_val,
            'max_val': max_val
        })
        
        # Aquí enviaríamos los comandos USB específicos para configurar el canal
        # Esto depende de lo que captures con Wireshark
        modulo, canal = channel_name.split('/')
        modulo_num = int(modulo.replace('cDAQ1Mod', ''))
        canal_num = int(canal.replace('ai', ''))
        
        # Ejemplo hipotético (ajustar según capturas de Wireshark)
        # request 0x01 podría ser "configurar canal"
        # value podría codificar modulo_num y canal_num
        # index podría codificar terminal_config
        # data podría codificar min_val y max_val
        
        value = (modulo_num << 8) | canal_num
        index = terminal_config.value
        
        # Codificar min_val y max_val como float de 32 bits
        import struct
        data = bytearray(8)
        struct.pack_into('<f', data, 0, min_val)
        struct.pack_into('<f', data, 4, max_val)
        
        self.send_control_command(0x01, value, index, data)
        return True
        
    def configure_timing(self, rate, sample_mode=AcquisitionType.CONTINUOUS, samples_per_chan=1000):
        """Configura la temporización de la adquisición"""
        self.logger.info(f"Configurando temporización: rate={rate}Hz, mode={sample_mode}, samples={samples_per_chan}")
        
        # Guardar configuración
        self.config['sample_rate'] = rate
        self.config['muestras_por_bloque'] = samples_per_chan
        self.config['acquisition_mode'] = sample_mode
        
        # Ejemplo hipotético (ajustar según capturas de Wireshark)
        # request 0x02 podría ser "configurar temporización"
        # value podría codificar rate (parte baja)
        # index podría codificar rate (parte alta)
        # data podría codificar sample_mode y samples_per_chan
        
        value = rate & 0xFFFF
        index = (rate >> 16) & 0xFFFF
        
        # Codificar sample_mode y samples_per_chan
        data = bytearray(5)
        data[0] = sample_mode.value
        samples_bytes = samples_per_chan.to_bytes(4, byteorder='little')
        data[1:5] = samples_bytes
        
        self.send_control_command(0x02, value, index, data)
        return True
        
    def start(self):
        """Inicia la tarea de adquisición"""
        self.logger.info("Iniciando tarea de adquisición")
        
        if not self.config['canales']:
            self.logger.error("No se han configurado canales")
            raise ValueError("No se han configurado canales")
            
        # Ejemplo hipotético (ajustar según capturas de Wireshark)
        # request 0x03 podría ser "control de tarea"
        # value=1 podría significar "iniciar"
        
        self.send_control_command(0x03, 0x0001, 0x0000)
        self.is_task_running = True
        return True
        
    def read(self, number_of_samples_per_channel=None, timeout=10.0):
        """Lee datos del dispositivo"""
        if not self.is_task_running:
            self.logger.error("No hay una tarea en ejecución")
            raise RuntimeError("No hay una tarea en ejecución")
            
        if number_of_samples_per_channel is None:
            number_of_samples_per_channel = self.config['muestras_por_bloque']
            
        self.logger.info(f"Leyendo {number_of_samples_per_channel} muestras por canal, timeout={timeout}s")
        
        # Calcular tamaño de los datos a leer
        num_channels = len(self.config['canales'])
        bytes_per_sample = 2  # Asumiendo 16 bits por muestra
        total_bytes = number_of_samples_per_channel * num_channels * bytes_per_sample
        
        start_time = time.time()
        try:
            # Leer datos del endpoint IN
            data = self.ep_in.read(total_bytes, int(timeout * 1000))
            end_time = time.time()
            
            # Convertir datos a numpy array
            data_array = np.frombuffer(data, dtype=np.int16)
            
            # Reorganizar los datos por canal (depende del formato de datos real)
            # Si los datos están entrelazados (todas las muestras del canal 0, luego todas las del canal 1, etc.)
            data_array = data_array.reshape(num_channels, number_of_samples_per_channel)
            
            # Si los datos están entrelazados muestra a muestra (canal 0, canal 1, ..., canal 0, canal 1, ...)
            # data_array = data_array.reshape(number_of_samples_per_channel, num_channels).T
            
            self.logger.info(f"Leídas {data_array.shape[1]} muestras por canal en {(end_time - start_time) * 1000:.2f}ms")
            
            # Escalar datos según rangos configurados
            # Esta parte depende del formato de datos real que recibas
            scaled_data = []
            for i, chan_config in enumerate(self.config['canales']):
                min_val = chan_config['min_val']
                max_val = chan_config['max_val']
                # Asumiendo que los datos crudos están en el rango -32768 a 32767
                scale = (max_val - min_val) / 65536
                offset = (max_val + min_val) / 2
                scaled_chan_data = data_array[i] * scale + offset
                scaled_data.append(scaled_chan_data)
                
            return scaled_data
            
        except usb.core.USBError as e:
            self.logger.error(f"Error al leer datos: {e}")
            raise
            
    def stop(self):
        """Detiene la tarea de adquisición"""
        if not self.is_task_running:
            self.logger.warning("No hay una tarea en ejecución para detener")
            return True
            
        self.logger.info("Deteniendo tarea de adquisición")
        
        # Ejemplo hipotético (ajustar según capturas de Wireshark)
        # request 0x03 podría ser "control de tarea"
        # value=0 podría significar "detener"
        
        self.send_control_command(0x03, 0x0000, 0x0000)
        self.is_task_running = False
        return True
        
    def close(self):
        """Cierra la tarea y libera recursos"""
        self.logger.info("Cerrando tarea y liberando recursos")
        
        if self.is_task_running:
            self.stop()
            
        if self.dev:
            usb.util.dispose_resources(self.dev)
            
        self.config['canales'] = []
        self.logger.info("Recursos liberados")
        return True
        
    def __del__(self):
        """Destructor para asegurar que se liberan los recursos"""
        try:
            self.close()
        except:
            pass


class Task:
    """Emula la clase Task de NI-DAQmx"""
    
    def __init__(self):
        self.logger = setup_logger("ni9174_task")
        self.logger.info("Creando nueva tarea")
        self.driver = NI9174Driver()
        self.driver.find_device()
        self.ai_channels = AIChannelCollection(self.driver)
        self.timing = TimingConfiguration(self.driver)
        
    def start(self):
        """Inicia la tarea"""
        return self.driver.start()
        
    def read(self, number_of_samples_per_channel=None, timeout=10.0):
        """Lee datos de la tarea"""
        return self.driver.read(number_of_samples_per_channel, timeout)
        
    def stop(self):
        """Detiene la tarea"""
        return self.driver.stop()
        
    def close(self):
        """Cierra la tarea"""
        return self.driver.close()


class AIChannelCollection:
    """Emula la colección de canales AI de NI-DAQmx"""
    
    def __init__(self, driver):
        self.driver = driver
        self.logger = driver.logger
        
    def add_ai_voltage_chan(self, channel_name, terminal_config=TerminalConfiguration.DIFF, min_val=-5.0, max_val=5.0):
        """Añade un canal de voltaje"""
        self.logger.info(f"Añadiendo canal de voltaje: {channel_name}")
        return self.driver.add_channel(channel_name, terminal_config, min_val, max_val)


class TimingConfiguration:
    """Emula la configuración de temporización de NI-DAQmx"""
    
    def __init__(self, driver):
        self.driver = driver
        self.logger = driver.logger
        
    def cfg_samp_clk_timing(self, rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=1000):
        """Configura la temporización del muestreo"""
        self.logger.info(f"Configurando temporización del muestreo: {rate}Hz")
        return self.driver.configure_timing(rate, sample_mode, samps_per_chan)


# Ejemplo de uso (para pruebas):
if __name__ == "__main__":
    # Configuración
    DISPOSITIVO = "cDAQ1Mod1"
    CANALES = ["ai0", "ai1", "ai2", "ai3"]
    SAMPLE_RATE = 10000
    MUESTRAS_POR_BLOQUE = 1000
    VOLTAJE_MIN = -5.0
    VOLTAJE_MAX = 5.0

    # Crear y ejecutar la tarea
    logger = setup_logger("test_main")
    logger.info("== Iniciando prueba de controlador USB NI-9174 ==")
    
    task = Task()
    
    try:
        # Configurar canales
        logger.info("1. Configurando canales...")
        for canal in CANALES:
            nombre_canal = f"{DISPOSITIVO}/{canal}"
            task.ai_channels.add_ai_voltage_chan(
                nombre_canal,
                terminal_config=TerminalConfiguration.DIFF,
                min_val=VOLTAJE_MIN,
                max_val=VOLTAJE_MAX
            )
        
        # Configurar temporización
        logger.info("2. Configurando temporización...")
        task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=MUESTRAS_POR_BLOQUE * 10
        )
        
        # Iniciar tarea
        logger.info("3. Iniciando tarea...")
        task.start()
        
        # Leer datos
        logger.info("4. Leyendo datos (5 iteraciones)...")
        for i in range(5):
            start_time = time.time()
            datos = task.read(
                number_of_samples_per_channel=MUESTRAS_POR_BLOQUE,
                timeout=2.0
            )
            end_time = time.time()
            
            logger.info(f"   - Iteración {i+1}: {len(datos[0])} muestras leídas en {(end_time-start_time)*1000:.2f}ms")
            
            # Análisis estadístico básico
            for ch_idx, ch_data in enumerate(datos):
                logger.debug(f"     Canal {CANALES[ch_idx]}: Min={min(ch_data):.4f}V, Max={max(ch_data):.4f}V, Avg={sum(ch_data)/len(ch_data):.4f}V")
            
            time.sleep(1)
        
        # Detener y cerrar tarea
        logger.info("5. Deteniendo tarea...")
        task.stop()
        
        logger.info("6. Cerrando tarea...")
        task.close()
        
    except Exception as e:
        logger.error(f"Error durante la prueba: {str(e)}", exc_info=True)
        if task:
            try:
                task.close()
            except:
                pass
    
    logger.info("== Prueba finalizada ==")

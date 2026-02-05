# <<< --- FILE 1 (Main Application) --- >>>
import sys
import time
import queue
import threading
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QTabWidget, QSizePolicy, QGroupBox, QGridLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QSlider
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import queue
import threading
import time
from datetime import datetime
import os
import csv
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from scipy import signal
from scipy.signal import butter, filtfilt, sosfilt, cheby1, ellip, bessel
import traceback # AGREGADO PARA DEBUGGING
import os # AÑADIDO PARA GESTIÓN DE RUTAS
import csv # AÑADIDO PARA LOGGING DE EXPERIMENTOS

from SVMWearPredictor import SVMWearPredictor
from CBR import CBRHysteresis
import feature_extraction as fe

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, ExcitationSource, Coupling, BridgeUnits
try:
    PSEUDO_DIFF = TerminalConfiguration.PSEUDODIFFERENTIAL
except AttributeError:
    # Fallback for older nidaqmx versions
    class PseudoDiffDummy:
        value = 125
    PSEUDO_DIFF = PseudoDiffDummy()

from scipy.signal import savgol_filter
from scipy.optimize import least_squares
from scipy.fft import rfft, rfftfreq

# --------------------------------------------------
# CONSTANTES DE CONVERSIÓN Y CALIBRACIÓN
# --------------------------------------------------
FORCE_CONVERSION = 890.0   # N/V (valor original)
ACC_CONVERSION   = 1000.0 / 98.3 # g/V (calculado de 98.3 mV/g según certificado PCB 352C33 -> 1V = 1000mV; (1000mV) / (98.3 mV/g) = 10.173 g)

# Factor de amplificación (ej.: INA122 + LM358 amplifica por 10)
FORCE_AMPLIFICATION_FACTOR = 10.0
# Conversión efectiva considerando la amplificación
EFFECTIVE_FORCE_CONVERSION = FORCE_CONVERSION / FORCE_AMPLIFICATION_FACTOR

# --------------------------------------------------
# CONFIGURACIÓN DE ADQUISICIÓN
# --------------------------------------------------
DISPOSITIVO = "cDAQ1Mod1"
# IMPORTANT: Assuming ai0 is the input signal (e.g., Force X+) and ai1 is the output (e.g., Force X-) for Bouc-Wen
FORCE_CANALES = ["ai0", "ai1", "ai2", "ai3"] # Canales para el módulo 9205
FORCE_SAMPLE_RATE = 2000 # Tasa de muestreo para el módulo 9205
MUESTRAS_POR_BLOQUE = 100 # Muestras por bloque/lectura
TIME_WINDOW = 0.5 # 500ms para mostrar 5 ciclos completos de 10 Hz. Esto define la ventana de visualización, no la de adquisición.
FORCE_MIN_VOLTAGE = -1.5
FORCE_MAX_VOLTAGE = 1.5

VIB_DISPOSITIVO = "cDAQ1Mod2"
VIB_CANALES = ["ai0", "ai1"]
VIB_SAMPLE_RATE = 2000 # MODIFICADO: Establecido a 2000 Hz para el módulo NI 9234
VIB_MUESTRAS_POR_BLOQUE = 100 # Puedes ajustar esto. Con SR=2000, esto es 0.05 segundos por bloque.
# Nuevas constantes para los límites del acelerómetro en g
ACCEL_MIN_G = -0.2  # ±200mg = ±0.2g según imagen mostrada
ACCEL_MAX_G = 0.2

# Configuración NI 9219 (Celda de carga/Galga)
BRIDGE_DISPOSITIVO = "cDAQ1Mod3"
BRIDGE_CANALES = ["ai0", "ai1", "ai2", "ai3"]  # 4 canales de puente
BRIDGE_SAMPLE_RATE = 2000  # Hz - Igual que fuerza y vibración para coherencia
BRIDGE_MUESTRAS_POR_BLOQUE = 10  # Menos muestras por bloque para estabilidad
# Configuración de puente completo
BRIDGE_MIN_VV = -0.01    # ±0.01 V/V rango típico
BRIDGE_MAX_VV = 0.01
BRIDGE_NOMINAL_RESISTANCE = 350.0  # Ω - Resistencia nominal de galga estándar
BRIDGE_EXCITATION_VOLTAGE = 2.5  # V - Voltaje de excitación interno
# Conversión a microvolts (para visualización)
BRIDGE_TO_MICROVOLTS = 2.5e6  # Factor de conversión V/V -> µV

# Directorio para guardado automático y logs
DATOS_DIR = "datos_automaticos"
EXPERIMENTOS_LOG = "experimentos_log.csv"

# --- CONFIGURACIÓN DE FILTROS DIGITALES ---
FILTER_TYPES = {
    'none': 'Sin filtro',
    'lowpass': 'Pasa-bajas',
    'highpass': 'Pasa-altas', 
    'bandpass': 'Pasa-banda',
    'notch': 'Filtro Notch (50/60Hz)'
}

FILTER_METHODS = {
    'butter': 'Butterworth',
    'cheby1': 'Chebyshev I',
    'ellip': 'Elíptico',
    'bessel': 'Bessel'
}

AUTO_SAVE_BASE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATOS_DIR)
EXPERIMENT_LOG_FILE_PATH = os.path.join(AUTO_SAVE_BASE_DIRECTORY, EXPERIMENTOS_LOG)


datos_queue = queue.Queue(maxsize=10)
vib_queue = queue.Queue(maxsize=10)
bridge_queue = queue.Queue(maxsize=10)  # Cola para NI 9219 (puente/galga)
# Colas para el módulo de razonamiento inteligente
reasoning_input_queue = queue.Queue(maxsize=10)
reasoning_output_queue = queue.Queue(maxsize=10)

# --------------------------------------------------
# MODELO BOUC-WEN
# --------------------------------------------------
def bouc_wen_model(params, t, corriente): # 'corriente' here is the input u(t)
    A, B, C, n, k = params
    N = len(t)
    fuerza_modelada = np.zeros(N) # This is the modeled output F(t)
    z_vals = np.zeros(N)          # This is the internal state z(t)
    z = 0.0
    z_min, z_max = -1e3, 1e3
    for i in range(1, N):
        dt = t[i] - t[i-1]
        # Ensure input 'corriente' is treated as u
        du = (corriente[i] - corriente[i-1]) / dt if dt != 0 else 0.0
        delta_z = (A * du - B * abs(du) * z - C * du * (abs(z) ** n)) * dt
        z = z + delta_z
        z = np.clip(z, z_min, z_max)
        z_vals[i] = z
        # The model predicts the output force based on z
        # NOTE: The exact relationship might vary (e.g., alpha*k*u + (1-alpha)*k*z or other forms).
        # The current form k*z**2 is a specific simplification.
        fuerza_modelada[i] = k * (z ** 2)
    return fuerza_modelada, z_vals


def error_bouc_wen(params, t, corriente_input, fuerza_exp_output):
    # 'corriente_input' is u(t), 'fuerza_exp_output' is the measured F(t) to match
    # The Bouc-Wen model typically models the hysteretic force component.
    # The error function should compare the model's output (related to z)
    # with the corresponding experimental force component.
    fuerza_modelada, _ = bouc_wen_model(params, t, corriente_input)
    return fuerza_modelada - fuerza_exp_output

# --------------------------------------------------
# FUNCIONES FFT y NORMALIZACIÓN
# --------------------------------------------------
def medir_frecuencia_fft(signal, fs):
    N = len(signal)
    if N < 2:
        return 0.0
    window = np.hanning(N)
    sig_win = signal * window
    fft_vals = np.abs(rfft(sig_win))
    freqs = rfftfreq(N, 1.0/fs)
    idx = np.argmax(fft_vals[1:]) + 1
    return freqs[idx]

def fft_spectrum(signal, fs):
    N = len(signal)
    if N < 2:
        return np.array([0.0]), np.array([0.0])
    window = np.hanning(N)
    sig_win = signal * window
    fft_vals = np.abs(rfft(sig_win))
    freqs = rfftfreq(N, 1.0/fs)
    return freqs, fft_vals

def normalize_signal(signal):
    max_abs = np.max(np.abs(signal))
    return signal if max_abs == 0 else (signal / max_abs)

# --------------------------------------------------
# INTELLIGENT REASONING SYSTEM (MÓDULO DE RAZONAMIENTO INTELIGENTE)
# --------------------------------------------------
class IntelligentReasoningSystem(threading.Thread):
    """
    Hilo que implementa el razonamiento inteligente basado en casos.
    Recibe vectores de características desde una cola de entrada,
    procesa los datos (ej. mediante CBR) y envía el resultado
    a la cola de salida.
    """
    def __init__(self, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                features = self.input_queue.get(timeout=1)
                # Si self.running se vuelve False mientras está en .get(), saldrá después del timeout
                if not self.running:
                    break
                result = self.reasoning_algorithm(features)
                try:
                    self.output_queue.put_nowait(result)
                except queue.Full:
                    print("Reasoning thread: output_queue está llena. Descartando resultado.")
                    # Opcionalmente, intentar hacer espacio y reintentar una vez
                    # try:
                    #     self.output_queue.get_nowait() # Descartar el más antiguo
                    #     self.output_queue.put_nowait(result) # Reintentar
                    # except queue.Empty:
                    #     pass # No debería ocurrir si estaba llena
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in reasoning thread: {e}")
                traceback.print_exc() # AGREGADO PARA DEBUGGING


    def reasoning_algorithm(self, features):
        """
        Se simula el razonamiento basado en casos.
        Se ajusta la base de casos a una dimensión de 6 para que coincida con features.
        """
        # Ensure features is a numpy array
        features = np.asarray(features)

        case_base = {
            # Ensure case vectors are numpy arrays
            "Desgaste Bajo": np.array([0.1, 0.2, 0.1, 0.3, 0.1, 0.2]),
            "Desgaste Medio": np.array([0.5, 0.6, 0.5, 0.6, 0.5, 0.6]),
            "Desgaste Alto": np.array([0.9, 1.0, 0.9, 1.1, 0.9, 1.0])
        }
        min_distance = float('inf')
        best_match = "Indeterminado" # Default value
        for label, case_vector in case_base.items():
            # Ensure case_vector is also numpy array (already done above)
            try:
                 distance = np.linalg.norm(features - case_vector)
                 if distance < min_distance:
                     min_distance = distance
                     best_match = label
            except Exception as e:
                print(f"Error calculating distance for label {label}: {e}")
                print(f"Features shape: {features.shape}, Type: {type(features)}")
                print(f"Case vector shape: {case_vector.shape}, Type: {type(case_vector)}")

        return best_match


    def stop(self):
        self.running = False

# --------------------------------------------------
# SISTEMA DE FILTRADO DIGITAL

class DigitalFilter:
    """Clase para aplicar filtros digitales a las señales en tiempo real"""
    
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.reset_filter()
    
    def reset_filter(self):
        """Resetear el estado del filtro"""
        self.filter_type = 'none'
        self.filter_method = 'butter'
        self.order = 4
        self.cutoff_low = 1.0  # Hz
        self.cutoff_high = 100.0  # Hz
        self.notch_freq = 50.0  # Hz (50 o 60 Hz)
        self.notch_quality = 30.0  # Factor Q
        self.sos = None
        self.zi = None
        
    def design_filter(self, filter_type, filter_method='butter', order=4, 
                     cutoff_low=1.0, cutoff_high=100.0, notch_freq=50.0, notch_quality=30.0):
        """Diseñar el filtro con los parámetros especificados"""
        try:
            self.filter_type = filter_type
            self.filter_method = filter_method
            self.order = order
            self.cutoff_low = cutoff_low
            self.cutoff_high = cutoff_high
            self.notch_freq = notch_freq
            self.notch_quality = notch_quality
            
            nyquist = self.sample_rate / 2.0
            
            if filter_type == 'none':
                self.sos = None
                self.zi = None
                return True
                
            elif filter_type == 'lowpass':
                if cutoff_high >= nyquist:
                    cutoff_high = nyquist * 0.9
                if filter_method == 'butter':
                    self.sos = signal.butter(order, cutoff_high, btype='low', fs=self.sample_rate, output='sos')
                elif filter_method == 'cheby1':
                    self.sos = signal.cheby1(order, 0.5, cutoff_high, btype='low', fs=self.sample_rate, output='sos')
                elif filter_method == 'ellip':
                    self.sos = signal.ellip(order, 0.5, 40, cutoff_high, btype='low', fs=self.sample_rate, output='sos')
                elif filter_method == 'bessel':
                    self.sos = signal.bessel(order, cutoff_high, btype='low', fs=self.sample_rate, output='sos')
                    
            elif filter_type == 'highpass':
                if cutoff_low <= 0:
                    cutoff_low = 0.1
                if filter_method == 'butter':
                    self.sos = signal.butter(order, cutoff_low, btype='high', fs=self.sample_rate, output='sos')
                elif filter_method == 'cheby1':
                    self.sos = signal.cheby1(order, 0.5, cutoff_low, btype='high', fs=self.sample_rate, output='sos')
                elif filter_method == 'ellip':
                    self.sos = signal.ellip(order, 0.5, 40, cutoff_low, btype='high', fs=self.sample_rate, output='sos')
                elif filter_method == 'bessel':
                    self.sos = signal.bessel(order, cutoff_low, btype='high', fs=self.sample_rate, output='sos')
                    
            elif filter_type == 'bandpass':
                if cutoff_low <= 0:
                    cutoff_low = 0.1
                if cutoff_high >= nyquist:
                    cutoff_high = nyquist * 0.9
                if cutoff_low >= cutoff_high:
                    cutoff_low = cutoff_high * 0.1
                if filter_method == 'butter':
                    self.sos = signal.butter(order, [cutoff_low, cutoff_high], btype='band', fs=self.sample_rate, output='sos')
                elif filter_method == 'cheby1':
                    self.sos = signal.cheby1(order, 0.5, [cutoff_low, cutoff_high], btype='band', fs=self.sample_rate, output='sos')
                elif filter_method == 'ellip':
                    self.sos = signal.ellip(order, 0.5, 40, [cutoff_low, cutoff_high], btype='band', fs=self.sample_rate, output='sos')
                elif filter_method == 'bessel':
                    self.sos = signal.bessel(order, [cutoff_low, cutoff_high], btype='band', fs=self.sample_rate, output='sos')
                    
            elif filter_type == 'notch':
                # Filtro notch para eliminar 50/60 Hz
                if notch_freq >= nyquist:
                    notch_freq = 50.0
                self.sos = signal.iirnotch(notch_freq, notch_quality, fs=self.sample_rate)
                self.sos = np.array([self.sos])  # Convertir a formato sos
            
            # Inicializar condiciones iniciales
            if self.sos is not None:
                self.zi = signal.sosfilt_zi(self.sos)
                
            return True
            
        except Exception as e:
            print(f"Error diseñando filtro: {e}")
            self.sos = None
            self.zi = None
            return False
    
    def apply_filter(self, data):
        """Aplicar el filtro a los datos"""
        if self.sos is None or self.filter_type == 'none':
            return data
            
        try:
            # Aplicar filtro usando Second-Order Sections para estabilidad numérica
            if len(data) == 0:
                return data
                
            # Para datos en tiempo real, usar sosfilt con condiciones iniciales
            filtered_data, self.zi = signal.sosfilt(self.sos, data, zi=self.zi)
            return filtered_data
            
        except Exception as e:
            print(f"Error aplicando filtro: {e}")
            return data

# --------------------------------------------------
# HILOS DE ADQUISICIÓN
# --------------------------------------------------
class AdquisicionThread(threading.Thread):
    def __init__(self, dispositivo, canales, sample_rate, muestras_por_bloque, terminal_config):
        super().__init__()
        self.dispositivo = dispositivo
        self.canales = canales
        self.sample_rate = sample_rate
        self.muestras_por_bloque = muestras_por_bloque
        self.terminal_config = terminal_config
        self.running = True
        self.task = None

    def run(self):
        try:
            self.task = nidaqmx.Task()
            for canal in self.canales:
                nombre_canal = f"{self.dispositivo}/{canal}"
                self.task.ai_channels.add_ai_voltage_chan(
                    nombre_canal,
                    terminal_config=self.terminal_config,
                    min_val=FORCE_MIN_VOLTAGE,
                    max_val=FORCE_MAX_VOLTAGE
                )
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10 # Larger DAQ buffer
            )
            self.task.start()
            print(f"DEBUG: Hilo de adquisición para {self.dispositivo} iniciado con éxito. Leyendo {len(self.canales)} canales.")
            while self.running:
                try:
                    # Read data
                    datos = self.task.read(
                        number_of_samples_per_channel=self.muestras_por_bloque,
                        timeout=2.0 # Increased timeout
                    )
                    datos_np = np.array(datos)

                    # Put data in queue (handle potential full queue)
                    if not datos_queue.full():
                        datos_queue.put_nowait(datos_np)
                    else:
                        # If full, discard oldest and add newest
                        print(f"Advertencia ({self.dispositivo}): datos_queue está llena, descartando datos antiguos.") # AGREGADO PARA DEBUGGING
                        try:
                            datos_queue.get_nowait()
                            datos_queue.put_nowait(datos_np)
                        except queue.Empty:
                            # Should not happen if queue was full, but handle defensively
                            datos_queue.put_nowait(datos_np)

                except nidaqmx.errors.DaqReadError as e:
                     print(f"DAQ Read Error ({self.dispositivo}): {e}") # AGREGADO self.dispositivo
                     traceback.print_exc() # AGREGADO PARA DEBUGGING
                     # Optionally break or attempt recovery depending on error code
                     if e.error_code == -200279: # Timeout error, potentially recoverable
                         print("Timeout occurred, continuing...")
                         time.sleep(0.01) # Short pause
                     else:
                         print("Non-recoverable DAQ read error, stopping thread.")
                         break # Stop on other errors
                except Exception as read_ex:
                    print(f"Unexpected error during DAQ read ({self.dispositivo}): {read_ex}") # AGREGADO self.dispositivo
                    traceback.print_exc() # AGREGADO PARA DEBUGGING
                    break # Stop on unexpected errors

        except Exception as ex:
            print(f"Error initializing or running AdquisicionThread ({self.dispositivo}): {ex}")
            traceback.print_exc() # AGREGADO PARA DEBUGGING
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                    print(f"AdquisicionThread ({self.dispositivo}) stopped and closed.")
                except Exception as close_ex:
                    print(f"Error stopping/closing task for {self.dispositivo}: {close_ex}")
            self.task = None # Ensure task is None after closing

    def stop(self):
        print(f"Stopping AdquisicionThread ({self.dispositivo})...")
        self.running = False

class VibrationAcquisitionThread(threading.Thread):
    def __init__(self, dispositivo, canales, sample_rate, muestras_por_bloque, modo_entrada='Accelerometer'):
        super().__init__()
        self.dispositivo = dispositivo
        self.canales = canales
        self.sample_rate = sample_rate
        self.muestras_por_bloque = muestras_por_bloque
        self.modo_entrada = modo_entrada
        self.running = True
        self.task = None

    def run(self):
        try:
            self.task = nidaqmx.Task()
            for canal in self.canales:
                nombre_canal = f"{self.dispositivo}/{canal}"
                if self.modo_entrada == 'Accelerometer':
                    try:
                        # Try adding as accelerometer first
                        self.task.ai_channels.add_ai_accel_chan(
                            physical_channel=nombre_canal,
                            name_to_assign_to_channel=f"Accel_{canal}",
                            terminal_config=TerminalConfiguration.DEFAULT, # Or specify if needed
                            min_val=ACCEL_MIN_G, max_val=ACCEL_MAX_G, # Usar constantes definidas
                            sensitivity=98.3, # mV/g - Según certificado PCB 352C33
                            sensitivity_units=nidaqmx.constants.AccelSensitivityUnits.MILLIVOLTS_PER_G, # Or VOLTS_PER_G
                            current_excit_source=nidaqmx.constants.ExcitationSource.INTERNAL, # Or EXTERNAL if applicable
                            current_excit_val=0.004 # 4mA typical for IEPE
                        )
                        print(f"Configured {nombre_canal} as Accelerometer channel.")
                    except Exception as accel_e:
                        print(f"WARNING: Failed to configure {nombre_canal} as Accelerometer ({accel_e}). Falling back to Voltage with IEPE settings.")
                        self._configure_voltage_with_iepe(nombre_canal)
                else: # If mode is not Accelerometer, configure as voltage with IEPE attempt
                    self._configure_voltage_with_iepe(nombre_canal)

            # Configure timing
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10 # Larger DAQ buffer
            )
            self.task.start()

            while self.running:
                try:
                    # Read data
                    datos = self.task.read(
                        number_of_samples_per_channel=self.muestras_por_bloque,
                        timeout=2.0 # Increased timeout
                    )
                    datos_np = np.array(datos)

                    # Put data in queue
                    if not vib_queue.full():
                        vib_queue.put_nowait(datos_np)
                    else:
                        print(f"Advertencia ({self.dispositivo}): vib_queue está llena, descartando datos antiguos.") # AGREGADO PARA DEBUGGING
                        try:
                            vib_queue.get_nowait()
                            vib_queue.put_nowait(datos_np)
                        except queue.Empty:
                            vib_queue.put_nowait(datos_np)

                except nidaqmx.errors.DaqReadError as e:
                     print(f"Vibration DAQ Read Error ({self.dispositivo}): {e}") # AGREGADO self.dispositivo
                     traceback.print_exc() # AGREGADO PARA DEBUGGING
                     if e.error_code == -200279:
                         print("Vibration Timeout occurred, continuing...")
                         time.sleep(0.01)
                     else:
                         print("Non-recoverable Vibration DAQ read error, stopping thread.")
                         break
                except Exception as read_ex:
                    print(f"Unexpected error during Vibration DAQ read ({self.dispositivo}): {read_ex}") # AGREGADO self.dispositivo
                    traceback.print_exc() # AGREGADO PARA DEBUGGING
                    break

        except Exception as ex:
            print(f"Error initializing or running VibrationAcquisitionThread ({self.dispositivo}): {ex}")
            traceback.print_exc() # AGREGADO PARA DEBUGGING
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                    print(f"VibrationAcquisitionThread ({self.dispositivo}) stopped and closed.")
                except Exception as close_ex:
                    print(f"Error stopping/closing task for {self.dispositivo}: {close_ex}")
            self.task = None

    def _configure_voltage_with_iepe(self, physical_channel):
        """Helper to configure a voltage channel with typical IEPE settings."""
        try:
            ch = self.task.ai_channels.add_ai_voltage_chan(
                physical_channel=physical_channel,
                name_to_assign_to_channel=f"VoltageIEPE_{physical_channel.split('/')[-1]}",
                terminal_config=TerminalConfiguration.PSEUDODIFFERENTIAL, # Often suitable for single-ended IEPE
                min_val=-10.0, # Wider range often needed for AC + DC bias
                max_val=10.0
            )
            # Attempt to enable IEPE specific settings
            try:
                ch.ai_coupling = Coupling.AC # Filter DC bias
                print(f"Set {physical_channel} coupling to AC.")
            except Exception as coup_e:
                print(f"Warning: Could not set AC coupling for {physical_channel}: {coup_e}")
            try:
                ch.ai_exc_source = ExcitationSource.INTERNAL
                ch.ai_exc_val = 0.004 # 4mA
                print(f"Set {physical_channel} internal excitation to 4mA.")
            except Exception as exc_e:
                print(f"Warning: Could not set internal excitation for {physical_channel}: {exc_e}")
            # Some devices use ai_iepe_enable explicitly
            try:
                if hasattr(ch, 'ai_iepe_enable'):
                    ch.ai_iepe_enable = True
                    print(f"Enabled IEPE explicitly for {physical_channel}.")
            except Exception as iepe_e:
                 print(f"Warning: Could not enable IEPE explicitly for {physical_channel}: {iepe_e}")

        except Exception as e:
            print(f"CRITICAL: Failed even to add {physical_channel} as a basic voltage channel: {e}")
            # Handle this failure case, maybe raise exception or log critical error


    def stop(self):
        print(f"Stopping VibrationAcquisitionThread ({self.dispositivo})...")
        self.running = False

# --------------------------------------------------
# HILO DE ADQUISICIÓN PARA NI 9219 (PUENTE/GALGA)
# --------------------------------------------------
class BridgeAcquisitionThread(threading.Thread):
    """Hilo de adquisición para NI 9219 (puente completo/galga)"""
    def __init__(self, dispositivo, canales, sample_rate, muestras_por_bloque):
        super().__init__()
        self.dispositivo = dispositivo
        self.canales = canales
        self.sample_rate = sample_rate
        self.muestras_por_bloque = muestras_por_bloque
        self.running = True
        self.task = None

    def run(self):
        try:
            self.task = nidaqmx.Task()
            
            # Configurar canales de puente completo
            for canal in self.canales:
                nombre_canal = f"{self.dispositivo}/{canal}"
                base_name = nombre_canal.replace("/", "_")
                self.task.ai_channels.add_ai_bridge_chan(
                    physical_channel=nombre_canal,
                    name_to_assign_to_channel=f"Bridge_{base_name}",
                    min_val=BRIDGE_MIN_VV,
                    max_val=BRIDGE_MAX_VV,
                    units=BridgeUnits.VOLTS_PER_VOLT,
                    voltage_excit_val=BRIDGE_EXCITATION_VOLTAGE,
                    voltage_excit_source=ExcitationSource.INTERNAL,
                    nominal_bridge_resistance=BRIDGE_NOMINAL_RESISTANCE
                )
            
            # Configurar timing
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10
            )
            
            # Aumentar buffer para estabilidad
            self.task.in_stream.input_buf_size = 5000
            
            self.task.start()
            print(f"✅ Bridge thread iniciado: {self.dispositivo}")

            while self.running:
                try:
                    datos = self.task.read(
                        number_of_samples_per_channel=self.muestras_por_bloque,
                        timeout=2.0
                    )
                    datos_np = np.array(datos)
                    
                    # Intentar poner datos en cola sin bloquear
                    if not bridge_queue.full():
                        bridge_queue.put_nowait(datos_np)
                    else:
                        try:
                            bridge_queue.get_nowait()  # Remover dato viejo
                            bridge_queue.put_nowait(datos_np)  # Añadir nuevo
                        except queue.Empty:
                            bridge_queue.put_nowait(datos_np)
                            
                except nidaqmx.DaqError as e:
                    if e.error_code == -200279:  # Buffer underrun
                        print("Bridge timeout, continuando...")
                        time.sleep(0.01)
                    else:
                        print(f"Bridge DAQ error: {e}")
                        break
                except Exception as e:
                    print(f"Bridge unexpected error: {e}")
                    break

        except Exception as e:
            print(f"❌ Error Bridge thread: {e}")
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                    print(f"BridgeAcquisitionThread ({self.dispositivo}) stopped and closed.")
                except:
                    pass
            self.task = None

    def stop(self):
        print(f"Stopping BridgeAcquisitionThread ({self.dispositivo})...")
        self.running = False

# --------------------------------------------------
class PredictorHisteresis(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PREDICTIVE DAQ - v2.0")
        self.resize(1280, 720)
        
        # Coeficientes de interpolación polinomial para los canales de fuerza
        # Configurados para mostrar valores en milivolts (mV)
        # [c, a, b, d] para cada canal = c + a*x + b*x^2 + d*x^3
        self.poly_coeffs = {
            'ai0': [0.0, 1000.0, 0.0, 0.0],  # Muestra el valor en mV (voltios * 1000)
            'ai1': [0.0, 1000.0, 0.0, 0.0],  # Muestra el valor en mV (voltios * 1000)
            'ai2': [0.0, 1000.0, 0.0, 0.0],  # Muestra el valor en mV (voltios * 1000)
            'ai3': [0.0, 1000.0, 0.0, 0.0]   # Muestra el valor en mV (voltios * 1000)
        }

        # --- Internal State ---
        self.acquiring = False # Start in stopped state initially
        self.sample_rate = FORCE_SAMPLE_RATE # Default, will be updated by UI
        self.vib_sample_rate = VIB_SAMPLE_RATE # Default, will be updated by UI
        self.time_window = TIME_WINDOW
        self.buffer_size = max(int(self.time_window * self.sample_rate), 2)
        self.terminal_config = TerminalConfiguration.DIFF # Default

        # Inicializar filtros digitales para cada canal
        self.force_filters = [DigitalFilter(FORCE_SAMPLE_RATE) for _ in FORCE_CANALES]
        self.vib_filters = [DigitalFilter(VIB_SAMPLE_RATE) for _ in VIB_CANALES]
        self.bridge_filters = [DigitalFilter(BRIDGE_SAMPLE_RATE) for _ in BRIDGE_CANALES]

        # Variables para guardado automático y logging de sesión
        self.current_session_base_filename = None
        self.session_start_time = None
        self.session_start_time_str = ""

        # Crear directorio de guardado automático si no existe
        if not os.path.exists(AUTO_SAVE_BASE_DIRECTORY):
            try:
                os.makedirs(AUTO_SAVE_BASE_DIRECTORY, exist_ok=True) # exist_ok=True para no fallar si ya existe
                print(f"Directorio para guardado automático verificado/creado: {AUTO_SAVE_BASE_DIRECTORY}")
            except Exception as e:
                print(f"CRÍTICO: Error al crear directorio {AUTO_SAVE_BASE_DIRECTORY}: {e}")
                QtWidgets.QMessageBox.critical(self, "Error de Directorio", f"No se pudo crear el directorio de guardado automático:\n{AUTO_SAVE_BASE_DIRECTORY}\nError: {e}")
                # Considerar deshabilitar guardado automático o cerrar la app si el directorio es esencial.

        # Data Buffers
        self.all_data = [[] for _ in range(len(FORCE_CANALES))] # Stores longer history for force saving/optimization
        self.all_vib_data = [[] for _ in range(len(VIB_CANALES))] # Stores longer history for vibration saving
        self.all_bridge_data = [[] for _ in range(len(BRIDGE_CANALES))] # Stores longer history for bridge/strain saving
        # Use deque for efficient rolling buffers
        self.datos_buffer = [deque(maxlen=self.buffer_size) for _ in range(len(FORCE_CANALES))]
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
        # Adjust vib_buffer_size based on its specific sample rate and desired time window
        vib_buffer_size = max(int(self.time_window * self.vib_sample_rate), 2)
        self.vib_buffer = [deque(maxlen=vib_buffer_size) for _ in range(len(VIB_CANALES))]
        # Bridge buffer (slower sample rate, typically)
        bridge_buffer_size = max(int(self.time_window * BRIDGE_SAMPLE_RATE), 2)
        self.bridge_buffer = [deque(maxlen=bridge_buffer_size) for _ in range(len(BRIDGE_CANALES))]


        # Bouc-Wen Parameters
        self.params_opt = [0.9, 0.4, 0.4, 2.1, 0.9]  # Initial guess
        self._optimization_running = False
        self.executor = ThreadPoolExecutor(max_workers=1) # For optimization task

        # External Modules
        self.svm_predictor = SVMWearPredictor()
        self.cbr_analyzer = CBRHysteresis()

        # --- UI Setup ---
        self._setup_ui()

        # --- Timers ---
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        # Start timer only when acquisition starts

        self.opt_timer = QtCore.QTimer()
        self.opt_timer.timeout.connect(self.ejecutar_optimizacion)
        # Start opt_timer only when acquisition starts and data is available

        # --- Threads ---
        self.adquisicion_thread = None
        self.vib_thread = None
        self.bridge_thread = None  # NI 9219 thread
        self.reasoning_system = None
        # Start threads only when acquisition starts

        # Set initial state of button
        self.start_stop_button.setText("Iniciar")
        self._enable_controls(False) # Disable controls initially

    def _setup_ui(self):
        # --- Main layout setup ---
        # Crear widget central y layout principal
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Crear widget de pestañas
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.tabs, stretch=1)  # Añadir tabs al layout principal
        
        # Ahora puedes seguir con la creación de cada pestaña
        # --- TAB 1: Módulo 9205 (Fuerza) ---
        self.tab_9205 = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_9205, "Módulo 9205 (Voltaje)")
        # *** DEFINICIÓN CORRECTA ***
        layout_9205 = QtWidgets.QVBoxLayout(self.tab_9205)
        layout_9205.setContentsMargins(2, 2, 2, 2)
        layout_9205.setSpacing(2)

        # Info Labels (Grid)
        info_layout_9205 = QtWidgets.QGridLayout()
        # ... (code to setup info labels and add them to info_layout_9205) ...
        self.value_labels, self.freq_labels = [], []
        self.amp_labels, self.rms_labels = [], []
        force_labels = [f"Voltaje ({c})" for c in FORCE_CANALES]
        for i, label_text in enumerate(force_labels):
             header_label = QtWidgets.QLabel(label_text)
             header_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
             info_layout_9205.addWidget(header_label, 0, i)
             v_label = QtWidgets.QLabel("Valor: 0.0 V")
             # f_label = QtWidgets.QLabel("Frec: 0.0 Hz") # Optional FFT freq
             a_label = QtWidgets.QLabel("Amp: 0.0 V")
             r_label = QtWidgets.QLabel("RMS: 0.0 V")
             for row, lab in enumerate([v_label, a_label, r_label], start=1): # Removed f_label for now
                 lab.setStyleSheet("font-size: 10pt;")
                 info_layout_9205.addWidget(lab, row, i)
             self.value_labels.append(v_label)
             # self.freq_labels.append(f_label)
             self.amp_labels.append(a_label)
             self.rms_labels.append(r_label)
        # *** USO CORRECTO (añadir layout de info al layout principal de la tab) ***
        layout_9205.addLayout(info_layout_9205)

        # Plot Widgets
        self.plot_widgets = []
        self.plot_curves = []
        colors = ['r', 'b', 'g', 'm'] # Colors for ai0, ai1, ai2, ai3
        plot_container_9205 = QtWidgets.QWidget() # Container for plots
        plot_layout_9205 = QtWidgets.QGridLayout(plot_container_9205) # Grid layout *inside* the container
        plot_layout_9205.setContentsMargins(0,0,0,0)
        plot_layout_9205.setSpacing(1)
        # --- Add stretch factors for rows/columns ---
        num_rows_9205 = (len(FORCE_CANALES) + 1) // 2 # Calculate number of rows needed
        for r in range(num_rows_9205):
            plot_layout_9205.setRowStretch(r, 1)
        plot_layout_9205.setColumnStretch(0, 1)
        plot_layout_9205.setColumnStretch(1, 1)
        # --- End of added stretch factors ---

        # *** USO CORRECTO (añadir el contenedor de plots al layout principal de la tab) ***
        # Esta es la línea 474 que dio el error. Asegúrate que 'layout_9205' esté bien escrito.
        layout_9205.addWidget(plot_container_9205, stretch=1)

        num_cols = 2 # Arrange plots in 2 columns
        for i in range(len(FORCE_CANALES)):
            w = pg.PlotWidget(title=f"Canal {FORCE_CANALES[i]}")
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Voltaje', units='V')
            w.setLabel('bottom', 'Tiempo', units='s')
            w.showGrid(x=True, y=True)
            # Use EFFECTIVE_FORCE_CONVERSION for plotting range
            # Usar rango de -1500 a 1500 mV para visualización en milivolts
            w.setYRange(FORCE_MIN_VOLTAGE, FORCE_MAX_VOLTAGE)
            w.setXRange(-self.time_window, 0) # Initial range
            pen = pg.mkPen(color=colors[i % 4], width=1) # Thinner pen
            curve = w.plot(pen=pen)
            self.plot_widgets.append(w)
            self.plot_curves.append(curve)
            # *** USO CORRECTO (añadir plots individuales al *grid layout* dentro del contenedor) ***
            plot_layout_9205.addWidget(w, i // num_cols, i % num_cols)

        # --- TAB 2: Módulo 9234 + 9205 (Vibración + Fuerza) ---
        self.tab_9234 = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_9234, "Vibración + Fuerza (9234 + 9205)")
        layout_9234 = QtWidgets.QVBoxLayout(self.tab_9234)
        layout_9234.setContentsMargins(2, 2, 2, 2)
        layout_9234.setSpacing(2)

        # Info Labels (Grid)
        vib_info_layout = QtWidgets.QGridLayout()
        # layout_9234.addLayout(vib_info_layout) # Esto se hace después de poblarlo

        vib_labels_text = [f"Acel ({c})" for c in VIB_CANALES]
        self.vib_value_labels, self.vib_freq_labels = [], [] # Asumo que vib_freq_labels podría usarse después
        self.vib_amp_labels, self.vib_rms_labels = [], []
        
        # Fila para Headers de Canales de Vibración
        for i, label_text in enumerate(vib_labels_text):
            header_label = QtWidgets.QLabel(label_text)
            header_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
            vib_info_layout.addWidget(header_label, 0, i) # Fila 0

            # Suponiendo una estructura similar a la Tab 1 para las etiquetas de info
            v_label = QtWidgets.QLabel("Valor: 0.0 g")
            a_label = QtWidgets.QLabel("Amp: 0.0 g")
            r_label = QtWidgets.QLabel("RMS: 0.0 g")
            
            v_label.setStyleSheet("font-size: 10pt;")
            a_label.setStyleSheet("font-size: 10pt;")
            r_label.setStyleSheet("font-size: 10pt;")

            vib_info_layout.addWidget(v_label, 1, i) # Fila 1
            vib_info_layout.addWidget(a_label, 2, i) # Fila 2
            vib_info_layout.addWidget(r_label, 3, i) # Fila 3

            self.vib_value_labels.append(v_label)
            self.vib_amp_labels.append(a_label)
            self.vib_rms_labels.append(r_label)

        # --- Añadir Control de Ganancia para Emisiones Acústicas (VIB_CANALES[0]) ---
        # Asumiendo que las etiquetas anteriores ocupan hasta la fila 3.
        # Lo colocaremos en la fila 4, columnas 0 y 1.
        self.ae_gain_label = QtWidgets.QLabel(f"Ganancia Em. Acústica ({VIB_CANALES[0]}):")
        self.ae_gain_label.setStyleSheet("font-size: 10pt;")
        self.ae_gain_spinbox = QtWidgets.QDoubleSpinBox()
        self.ae_gain_spinbox.setRange(0.01, 10000.0)  # Rango de ganancia (ajustar según necesidad)
        self.ae_gain_spinbox.setValue(1.0)          # Ganancia por defecto (sin cambio)
        self.ae_gain_spinbox.setSingleStep(0.1)     # Incremento/decremento
        self.ae_gain_spinbox.setDecimals(2)         # Precisión decimal
        self.ae_gain_spinbox.setToolTip(f"Factor de ganancia para el canal de emisiones acústicas ({VIB_CANALES[0]})" )

        # Añadir al layout de información de vibración
        # Si VIB_CANALES tiene más de un canal, esto se alineará bien.
        # Si solo hay un VIB_CANAL, la columna 1 estaría vacía, pero aún funcional.
        current_row_for_gain = 4 # Siguiente fila después de las etiquetas de info de canal
        vib_info_layout.addWidget(self.ae_gain_label, current_row_for_gain, 0)
        vib_info_layout.addWidget(self.ae_gain_spinbox, current_row_for_gain, 1)
        # --- Fin de Añadir Control de Ganancia ---

        # --- Normalización de señales [-1, 1] ---
        # Checkboxes para activar normalización en plots (solo visualización)
        self.normalize_vib_checkbox = QtWidgets.QCheckBox("Normalizar Vibración [-1, 1]")
        self.normalize_vib_checkbox.setChecked(False)
        vib_info_layout.addWidget(self.normalize_vib_checkbox, current_row_for_gain + 1, 0)

        self.normalize_force_checkbox = QtWidgets.QCheckBox("Normalizar Fuerza [-1, 1]")
        self.normalize_force_checkbox.setChecked(False)
        vib_info_layout.addWidget(self.normalize_force_checkbox, current_row_for_gain + 1, 1)

        layout_9234.addLayout(vib_info_layout)
        
        # --- Vista y Ejes: Bloquear Y para permitir pan/zoom manual durante adquisición ---
        axes_group = QtWidgets.QGroupBox("Vista y Ejes")
        axes_layout = QtWidgets.QHBoxLayout(axes_group)
        self.lock_y_vib_checkbox = QtWidgets.QCheckBox("Bloquear Y Vibración")
        self.lock_y_vib_checkbox.setToolTip("Si está activo, no se reajusta el eje Y automáticamente; puedes pan/zoom manual.")
        self.lock_y_vib_checkbox.setChecked(False)
        self.lock_y_force_checkbox = QtWidgets.QCheckBox("Bloquear Y Fuerza")
        self.lock_y_force_checkbox.setToolTip("Si está activo, no se reajusta el eje Y automáticamente; puedes pan/zoom manual.")
        self.lock_y_force_checkbox.setChecked(False)
        axes_layout.addWidget(self.lock_y_vib_checkbox)
        axes_layout.addWidget(self.lock_y_force_checkbox)
        axes_layout.addStretch(1)
        layout_9234.addWidget(axes_group)
        
        # Container principal para todas las gráficas (6 ventanas: 2 vibración + 4 fuerza)
        all_plots_container = QtWidgets.QWidget()
        all_plots_layout = QtWidgets.QGridLayout(all_plots_container)
        all_plots_layout.setContentsMargins(5, 5, 5, 5)
        all_plots_layout.setSpacing(8)
        
        # Layout 3×2: 3 columnas, 2 filas para las 6 ventanas
        # Fila 0: [Vib ai0] [Vib ai1] [Fuerza ai0]
        # Fila 1: [Fuerza ai1] [Fuerza ai2] [Fuerza ai3]
        for r in range(2):  # 2 filas
            all_plots_layout.setRowStretch(r, 1)
        for c in range(3):  # 3 columnas  
            all_plots_layout.setColumnStretch(c, 1)
        
        # Añadir container al layout principal
        layout_9234.addWidget(all_plots_container, stretch=3)
        
        # Plot Widgets para Vibración
        self.vib_plot_widgets = []
        self.vib_plot_curves = []
        vib_colors = ['#FFD700', '#00FFFF']  # Dorado y Cian brillantes

        for i in range(len(VIB_CANALES)):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Aceleración', 'g')
            w.setLabel('bottom', 'Tiempo', 's')
            w.setTitle(f'Vibración {VIB_CANALES[i]} (g)')
            w.showGrid(x=True, y=True)
            # Habilitar pan/zoom con el ratón
            try:
                w.setMouseEnabled(x=True, y=True)
            except Exception:
                pass
            
            # Rango Y apropiado para aceleración
            w.setYRange(ACCEL_MIN_G, ACCEL_MAX_G)
            w.setXRange(-self.time_window, 0)
            
            # Líneas más gruesas para mejor visibilidad
            pen = pg.mkPen(color=vib_colors[i % len(vib_colors)], width=2)
            curve = w.plot(pen=pen)
            self.vib_plot_widgets.append(w)
            self.vib_plot_curves.append(curve)
            
            # Colocar en fila 0: vibración ai0 en (0,0), vibración ai1 en (0,1)
            all_plots_layout.addWidget(w, 0, i)
        
        # Plot Widgets para Fuerza (integrados en el mismo layout)
        force_colors = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7']  # Colores diferenciados para fuerza
        
        for i in range(len(FORCE_CANALES)):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Voltaje', 'V')
            w.setLabel('bottom', 'Tiempo', 's')
            w.setTitle(f'Fuerza {FORCE_CANALES[i]} (V)')
            w.showGrid(x=True, y=True)
            # Habilitar pan/zoom con el ratón
            try:
                w.setMouseEnabled(x=True, y=True)
            except Exception:
                pass
            
            # Rango Y apropiado para voltaje
            w.setYRange(FORCE_MIN_VOLTAGE, FORCE_MAX_VOLTAGE)
            w.setXRange(-self.time_window, 0)
            
            # Líneas más gruesas para mejor visibilidad
            pen = pg.mkPen(color=force_colors[i % len(force_colors)], width=2)
            curve = w.plot(pen=pen)
            self.plot_widgets.append(w)
            self.plot_curves.append(curve)
            
            # Distribución en el layout principal 3×2:
            # Fila 0: [Vib ai0] [Vib ai1] [Fuerza ai0]
            # Fila 1: [Fuerza ai1] [Fuerza ai2] [Fuerza ai3]  
            if i == 0:  # Fuerza ai0 en (0,2)
                all_plots_layout.addWidget(w, 0, 2)
            else:  # Fuerza ai1,ai2,ai3 en fila 1, columnas 0,1,2
                all_plots_layout.addWidget(w, 1, i-1)
        
        # --- TAB 3: Comparativa Fuerza vs Vibración ---
        self.tab_9219 = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_9219, "Comparativa (Fuerza vs Vibración)")
        layout_9219_tab = QtWidgets.QVBoxLayout(self.tab_9219)
        layout_9219_tab.setContentsMargins(2, 2, 2, 2)
        layout_9219_tab.setSpacing(2)

        compare_group = QtWidgets.QGroupBox("Comparativa Fuerza vs Vibración")
        compare_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        compare_layout = QtWidgets.QVBoxLayout(compare_group)

        # Controles: selección de canales y normalización
        compare_ctrl_layout = QtWidgets.QHBoxLayout()
        compare_ctrl_layout.addWidget(QtWidgets.QLabel("Fuerza:"))
        self.comp_force_combo = QtWidgets.QComboBox()
        for ch in FORCE_CANALES:
            self.comp_force_combo.addItem(ch)
        self.comp_force_combo.setCurrentIndex(0)
        compare_ctrl_layout.addWidget(self.comp_force_combo)

        compare_ctrl_layout.addSpacing(10)
        compare_ctrl_layout.addWidget(QtWidgets.QLabel("Vibración:"))
        self.comp_vib_combo = QtWidgets.QComboBox()
        for ch in VIB_CANALES:
            self.comp_vib_combo.addItem(ch)
        self.comp_vib_combo.setCurrentIndex(0)
        compare_ctrl_layout.addWidget(self.comp_vib_combo)

        compare_ctrl_layout.addStretch(1)
        self.comp_normalize_checkbox = QtWidgets.QCheckBox("Normalizar Comparativa [-1, 1]")
        self.comp_normalize_checkbox.setChecked(True)
        compare_ctrl_layout.addWidget(self.comp_normalize_checkbox)

        compare_layout.addLayout(compare_ctrl_layout)

        # Plots: superposición tiempo y scatter Lissajous (norm)
        compare_plots_container = QtWidgets.QWidget()
        compare_plots_grid = QtWidgets.QGridLayout(compare_plots_container)
        compare_plots_grid.setContentsMargins(0, 0, 0, 0)
        compare_plots_grid.setSpacing(4)

        # Plot de superposición en el tiempo
        self.comp_time_widget = pg.PlotWidget(title="Comparativa Tiempo (Fuerza vs Vibración)")
        self.comp_time_widget.setLabel('left', 'Valor (norm o unidades reales)')
        self.comp_time_widget.setLabel('bottom', 'Tiempo', 's')
        self.comp_time_widget.showGrid(x=True, y=True)
        self.comp_time_curve_force = self.comp_time_widget.plot(pen=pg.mkPen('#FF5722', width=2), name='Fuerza')
        self.comp_time_curve_vib = self.comp_time_widget.plot(pen=pg.mkPen('#FFD700', width=2), name='Vibración')
        compare_plots_grid.addWidget(self.comp_time_widget, 0, 0)

        # Scatter Lissajous: Fuerza (norm) vs Vibración (norm)
        self.comp_scatter_widget = pg.PlotWidget(title="Lissajous: Fuerza (norm) vs Vibración (norm)")
        self.comp_scatter_widget.setLabel('left', 'Vibración (norm)')
        self.comp_scatter_widget.setLabel('bottom', 'Fuerza (norm)')
        self.comp_scatter_widget.showGrid(x=True, y=True)
        self.comp_scatter_widget.setXRange(-1.1, 1.1)
        self.comp_scatter_widget.setYRange(-1.1, 1.1)
        self.comp_scatter_curve = self.comp_scatter_widget.plot(pen=None, symbol='o', symbolSize=3, symbolBrush='w', symbolPen=None)
        compare_plots_grid.addWidget(self.comp_scatter_widget, 0, 1)

        compare_layout.addWidget(compare_plots_container)
        layout_9219_tab.addWidget(compare_group)

        # --- TAB 4: Configuración de Interpolación ---
        self.tab_config = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_config, "Configuración")
        config_layout = QtWidgets.QVBoxLayout(self.tab_config)
        
        # Grupo de controles para interpolación polinómica
        poly_group = QtWidgets.QGroupBox("Interpolación Polinomial para Fuerza")
        poly_layout = QtWidgets.QVBoxLayout(poly_group)
        
        # Explicación
        poly_label = QtWidgets.QLabel("Configure los coeficientes para la ecuación: fuerza = c + a*[ai] + b*[ai]^2 + d*[ai]^3")
        poly_label.setStyleSheet("font-weight: bold;")
        poly_layout.addWidget(poly_label)
        
        # Tabla para los coeficientes
        coef_table = QtWidgets.QTableWidget()
        coef_table.setColumnCount(5)
        coef_table.setRowCount(len(FORCE_CANALES))
        coef_table.setHorizontalHeaderLabels(["Canal", "c (offset)", "a (lineal)", "b (cuadrático)", "d (cúbico)"])
        coef_table.setVerticalHeaderLabels([""] * len(FORCE_CANALES))
        
        # Configurar las celdas con valores iniciales
        for i, canal in enumerate(FORCE_CANALES):
            # Canal (solo lectura)
            canal_item = QtWidgets.QTableWidgetItem(canal)
            canal_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)  # Solo seleccionable, no editable
            coef_table.setItem(i, 0, canal_item)
            
            # Coeficientes a, b, c, d
            for j, coef in enumerate(self.poly_coeffs[canal]):
                coef_item = QtWidgets.QTableWidgetItem(str(coef))
                coef_table.setItem(i, j+1, coef_item)
        
        # Ajustar tabla
        header = coef_table.horizontalHeader()
        for i in range(5):
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
        coef_table.setMinimumHeight(150)
        poly_layout.addWidget(coef_table)
        self.coef_table = coef_table
        
        # Botón para aplicar cambios
        apply_poly_btn = QtWidgets.QPushButton("Aplicar Coeficientes")
        apply_poly_btn.clicked.connect(self.apply_poly_coefficients)
        poly_layout.addWidget(apply_poly_btn)
        
        config_layout.addWidget(poly_group)
        # Selector para habilitar canales analógicos
        channel_group = QtWidgets.QGroupBox("Canales 9205")
        channel_layout = QtWidgets.QHBoxLayout(channel_group)
        self.channel_checkboxes = []
        for canal in FORCE_CANALES:
            cb = QtWidgets.QCheckBox(canal)
            cb.setChecked(True)
            channel_layout.addWidget(cb)
            self.channel_checkboxes.append(cb)
        config_layout.addWidget(channel_group)
        
        # (Puente 9219 removido de Configuración)
        
        # --- GRUPO DE FILTROS DIGITALES ---
        filters_group = QtWidgets.QGroupBox("🔧 Filtros Digitales por Canal")
        filters_layout = QtWidgets.QVBoxLayout(filters_group)
        
        # Crear tabs para diferentes módulos
        filter_tabs = QtWidgets.QTabWidget()
        
        # Tab para filtros de fuerza (NI 9205)
        force_filter_tab = QtWidgets.QWidget()
        force_filter_layout = QtWidgets.QVBoxLayout(force_filter_tab)
        self.force_filter_controls = self.create_filter_controls("Fuerza", len(FORCE_CANALES), FORCE_CANALES)
        force_filter_layout.addWidget(self.force_filter_controls)
        filter_tabs.addTab(force_filter_tab, "🔋 Fuerza (9205)")
        
        # Tab para filtros de vibración (NI 9234)  
        vib_filter_tab = QtWidgets.QWidget()
        vib_filter_layout = QtWidgets.QVBoxLayout(vib_filter_tab)
        self.vib_filter_controls = self.create_filter_controls("Vibración", len(VIB_CANALES), VIB_CANALES)
        vib_filter_layout.addWidget(self.vib_filter_controls)
        filter_tabs.addTab(vib_filter_tab, "📈 Vibración (9234)")
        
        # (Pestaña de filtros para Puente removida)
        
        filters_layout.addWidget(filter_tabs)
        config_layout.addWidget(filters_group)
        
        config_layout.addStretch(1)

        # --- TAB 5: Modelado Bouc-Wen ---
        self.tab_modelo = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_modelo, "Modelado Bouc-Wen")
        layout_modelo = QtWidgets.QVBoxLayout(self.tab_modelo)
        layout_modelo.setContentsMargins(2, 2, 2, 2)
        layout_modelo.setSpacing(2)

        # Container for plots
        bw_plot_container = QtWidgets.QWidget()
        bw_plot_layout = QtWidgets.QGridLayout(bw_plot_container) # Use Grid for 3 plots
        bw_plot_layout.setContentsMargins(0,0,0,0)
        bw_plot_layout.setSpacing(1)
        # --- Add stretch factors for rows/columns ---
        bw_plot_layout.setRowStretch(0, 1) # Row for 2D and 3D plots
        bw_plot_layout.setRowStretch(1, 1) # Row for Hysteresis plot
        bw_plot_layout.setColumnStretch(0, 1)
        bw_plot_layout.setColumnStretch(1, 1)
        # --- End of added stretch factors ---
        layout_modelo.addWidget(bw_plot_container, stretch=1) # Keep stretch here

        # 2D Phase Plot (Lissajous - Normalized Force X+ vs Force X-)
        self.phase2d_widget = pg.PlotWidget(title="Fase 2D (ai1 norm vs ai0 norm)")
        self.phase2d_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.phase2d_widget.setLabel('left', f"{FORCE_CANALES[1]} (norm)")
        self.phase2d_widget.setLabel('bottom', f"{FORCE_CANALES[0]} (norm)")
        self.phase2d_widget.showGrid(x=True, y=True)
        self.phase2d_widget.setXRange(-1.1, 1.1) # Normalized range
        self.phase2d_widget.setYRange(-1.1, 1.1)
        self.phase2d_curve = self.phase2d_widget.plot(pen=pg.mkPen('c', width=1))
        bw_plot_layout.addWidget(self.phase2d_widget, 0, 0) # Row 0, Col 0

        # 3D Phase Plot (Time, Normalized Force X+, Normalized Force X-)
        self.glview = gl.GLViewWidget()
        self.glview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.glview.setCameraPosition(distance=4, azimuth=-45, elevation=20) # Adjusted view
        grid = gl.GLGridItem()
        grid.setSize(2, 2, 1) # Smaller grid for normalized data
        grid.scale(1, 1, 1) # Adjust scale if needed
        self.glview.addItem(grid)
        # Add axis labels (optional, can clutter)
        # xax = gl.GLAxisItem(glOptions='opaque'); xax.setSize(x=2); self.glview.addItem(xax)
        self.phase3d_line = gl.GLLinePlotItem(color=(0,255,0,200), width=1.5, antialias=True) # Green, slightly transparent
        self.glview.addItem(self.phase3d_line)
        bw_plot_layout.addWidget(self.glview, 0, 1) # Row 0, Col 1

        # Hysteresis Plot z(t) vs Input Signal Voltage
        self.bw_phase_widget = pg.PlotWidget(title="Ciclo Histéresis Modelo (z vs Input)")
        self.bw_phase_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bw_phase_widget.setLabel('left', 'z(t) (Estado Interno)')
        # *** CORRECTION HERE: Label updated ***
        self.bw_phase_widget.setLabel('bottom', f'Input Signal ({FORCE_CANALES[0]} - Volts)')
        self.bw_phase_widget.showGrid(x=True, y=True)
        # Set reasonable initial ranges, might need dynamic adjustment
        self.bw_phase_widget.setXRange(FORCE_MIN_VOLTAGE, FORCE_MAX_VOLTAGE)
        # self.bw_phase_widget.setYRange(-1500, 1500) # Adjust based on expected z range
        self.bw_phase_curve = self.bw_phase_widget.plot(pen=pg.mkPen('m', width=1))
        bw_plot_layout.addWidget(self.bw_phase_widget, 1, 0, 1, 2) # Row 1, spanning 2 columns

        # Optimization Controls
        optim_layout = QtWidgets.QHBoxLayout()
        self.optimizar_button = QtWidgets.QPushButton("Ajustar Modelo Bouc-Wen")
        self.optimizar_button.clicked.connect(self.ajustar_modelo)
        optim_layout.addWidget(self.optimizar_button)

        self.params_group = QtWidgets.QGroupBox("Parámetros Bouc-Wen Ajustados")
        form_layout = QtWidgets.QFormLayout(self.params_group)
        form_layout.setContentsMargins(5, 5, 5, 5)
        form_layout.setSpacing(4)
        self.param_labels = {}
        for param in ["A", "B", "C", "n", "k"]:
            lbl = QtWidgets.QLabel("-")
            lbl.setStyleSheet("font-size: 9pt;")
            form_layout.addRow(f"{param}:", lbl)
            self.param_labels[param] = lbl
        optim_layout.addWidget(self.params_group, stretch=1)
        layout_modelo.addLayout(optim_layout)


        # --- TAB 4: FFT/Espectros ---
        self.tab_fft = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_fft, "Espectros FFT")
        layout_fft = QtWidgets.QGridLayout(self.tab_fft) # Use grid layout
        layout_fft.setContentsMargins(2,2,2,2)
        layout_fft.setSpacing(1)
        # --- Add stretch factors for rows/columns ---
        layout_fft.setRowStretch(0, 1) # Row for Force FFTs
        layout_fft.setRowStretch(1, 1) # Row for Vib FFTs
        for c in range(max(len(FORCE_CANALES), len(VIB_CANALES))): # Stretch all potential columns
             layout_fft.setColumnStretch(c, 1)
        # --- End of added stretch factors ---

        # FFT Plots for 9205 (Force)
        self.fft_plots_9205 = []
        self.fft_curves_9205 = []
        c9205_colors = ['r','b','g','m']
        fft_titles_9205 = [f"FFT Fuerza ({c})" for c in FORCE_CANALES]
        for i in range(len(FORCE_CANALES)):
            w = pg.PlotWidget(title=fft_titles_9205[i])
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Magnitud')
            w.setLabel('bottom', 'Frecuencia', units='Hz')
            w.showGrid(x=True, y=True)
            w.setLogMode(x=False, y=True) # Log Y axis often useful for FFT
            pen = pg.mkPen(color=c9205_colors[i % 4], width=1)
            curve = w.plot(pen=pen)
            self.fft_plots_9205.append(w)
            self.fft_curves_9205.append(curve)
            layout_fft.addWidget(w, 0, i) # Add to first row of grid

        # FFT Plots for 9234 (Vibration)
        self.fft_plots_9234 = []
        self.fft_curves_9234 = []
        vib_colors2 = ['y','c']
        fft_titles_9234 = [f"FFT Acel ({c})" for c in VIB_CANALES]
        for i in range(len(VIB_CANALES)):
            w = pg.PlotWidget(title=fft_titles_9234[i])
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Magnitud')
            w.setLabel('bottom', 'Frecuencia', units='Hz')
            w.showGrid(x=True, y=True)
            w.setLogMode(x=False, y=True)
            pen = pg.mkPen(color=vib_colors2[i % 2], width=1)
            curve = w.plot(pen=pen)
            self.fft_plots_9234.append(w)
            self.fft_curves_9234.append(curve)
            layout_fft.addWidget(w, 1, i) # Add to second row of grid


        # --- TAB 5: Predicción de Desgaste (SVM) ---
        self.tab_svm = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_svm, "Predicción Desgaste (SVM)")
        layout_svm = QtWidgets.QVBoxLayout(self.tab_svm)
        layout_svm.setContentsMargins(10,10,10,10)
        layout_svm.setSpacing(10)
        layout_svm.addStretch(1) # Push content towards center
        self.svm_result_label = QtWidgets.QLabel("Estado de herramienta: Esperando datos...")
        self.svm_result_label.setStyleSheet("font-size: 18pt; font-weight: bold; qproperty-alignment: AlignCenter;")
        layout_svm.addWidget(self.svm_result_label)
        self.btn_prediccion = QtWidgets.QPushButton("Actualizar Predicción Manualmente")
        self.btn_prediccion.clicked.connect(self.actualizar_prediccion)
        layout_svm.addWidget(self.btn_prediccion, alignment=QtCore.Qt.AlignCenter)
        layout_svm.addStretch(1)


        # --- TAB 6: CBR Histeresis – Casos similares ---
        self.tab_cbr = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_cbr, "CBR Histeresis")
        layout_cbr = QtWidgets.QVBoxLayout(self.tab_cbr)
        layout_cbr.setContentsMargins(5,5,5,5)
        layout_cbr.setSpacing(5)
        self.cbr_result_label = QtWidgets.QLabel("Casos similares recuperados:")
        self.cbr_result_label.setStyleSheet("font-size: 14pt;")
        layout_cbr.addWidget(self.cbr_result_label)
        self.cbr_list = QtWidgets.QListWidget()
        self.cbr_list.setStyleSheet("font-size: 11pt;")
        layout_cbr.addWidget(self.cbr_list, stretch=1)


        # --- TAB 7: Razonamiento Inteligente (Visualización CBR) ---
        self.tab_reasoning = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_reasoning, "Razonamiento Inteligente (CBR)")
        layout_reasoning = QtWidgets.QVBoxLayout(self.tab_reasoning)
        layout_reasoning.setContentsMargins(5, 5, 5, 5)
        layout_reasoning.setSpacing(5)
        layout_reasoning.addStretch(1) # Center vertically

        self.reasoning_status_label = QtWidgets.QLabel("Estado Predicho (CBR): Esperando...")
        self.reasoning_status_label.setStyleSheet("font-size: 18pt; font-weight: bold; qproperty-alignment: AlignCenter;")
        layout_reasoning.addWidget(self.reasoning_status_label)

        # Simple visual indicator (e.g., a colored circle)
        self.reasoning_indicator = QtWidgets.QLabel()
        self.reasoning_indicator.setFixedSize(100, 100)
        self.reasoning_indicator.setStyleSheet("border-radius: 50px; background-color: gray;") # Default gray
        layout_reasoning.addWidget(self.reasoning_indicator, alignment=QtCore.Qt.AlignCenter)

        layout_reasoning.addStretch(1)

        self.reasoning_color_map = {
            "Desgaste Bajo": "background-color: lightgreen;",
            "Desgaste Medio": "background-color: yellow;",
            "Desgaste Alto": "background-color: salmon;",
            "Desgaste Indeterminado": "background-color: gray;", # Default/unknown
        }
        # Timer to check reasoning output queue
        self.reasoning_timer = QtCore.QTimer()
        self.reasoning_timer.timeout.connect(self.update_reasoning_visualization)
        # Start this timer only when acquisition starts

        # --- TAB 8: Diseño de Experimentos (DOE) ---
        self.tab_doe = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_doe, "Diseño de Experimentos (DOE)")
        layout_doe = QtWidgets.QVBoxLayout(self.tab_doe)
        layout_doe.setContentsMargins(10, 10, 10, 10)
        layout_doe.setSpacing(10)
        layout_doe.addStretch(1)

        doe_info_label = QtWidgets.QLabel("Esta sección está destinada a la configuración y ejecución de Diseños de Experimentos (DOE).")
        doe_info_label.setStyleSheet("font-size: 12pt; qproperty-alignment: AlignCenter;")
        doe_info_label.setWordWrap(True)
        layout_doe.addWidget(doe_info_label)

        # Example: Button to manage DOE
        self.btn_manage_doe = QtWidgets.QPushButton("Gestionar Experimentos DOE")
        # self.btn_manage_doe.clicked.connect(self.manage_doe_experiments) # Connect to a future method
        self.btn_manage_doe.setEnabled(False) # Enable when DOE logic is implemented
        layout_doe.addWidget(self.btn_manage_doe, alignment=QtCore.Qt.AlignCenter)

        layout_doe.addStretch(1)


        # --- Controles inferiores ---
        # Main control layout container
        bottom_controls_container = QtWidgets.QWidget()
        main_control_layout = QtWidgets.QVBoxLayout(bottom_controls_container) # QVBoxLayout for two rows
        main_control_layout.setContentsMargins(5,5,5,5)
        main_control_layout.setSpacing(5)

        # First row of controls (Start/Stop, Time Vis, Terminal Mode)
        control_layout_row1 = QtWidgets.QHBoxLayout()
        control_layout_row1.setSpacing(10)

        self.start_stop_button = QtWidgets.QPushButton("Iniciar") # Start state
        self.start_stop_button.setCheckable(True) # Make it toggle-like visually
        self.start_stop_button.setStyleSheet("font-size: 11pt; padding: 5px;")
        self.start_stop_button.clicked.connect(self.toggle_acquisition)
        control_layout_row1.addWidget(self.start_stop_button)

        control_layout_row1.addWidget(QtWidgets.QLabel("Tiempo Vis:"))
        self.time_combo = QtWidgets.QComboBox()
        self.time_combo.addItems(["10 ms", "20 ms", "50 ms", "100 ms", "200 ms", "500 ms", "1 s", "2 s"])
        self.time_combo.setCurrentText("50 ms") # Default
        self.time_combo.currentTextChanged.connect(self.update_timebase)
        control_layout_row1.addWidget(self.time_combo)

        control_layout_row1.addWidget(QtWidgets.QLabel("Modo Entrada (Fuerza):"))
        self.terminal_mode_combo = QtWidgets.QComboBox()
        self.terminal_modes = {
             "Diferencial": TerminalConfiguration.DIFF,
             "RSE": TerminalConfiguration.RSE,
             "NRSE": TerminalConfiguration.NRSE,
             "PseudoDif": PSEUDO_DIFF
             }
        self.terminal_mode_combo.addItems(list(self.terminal_modes.keys()))
        self.terminal_mode_combo.setCurrentText("Diferencial") # Default
        control_layout_row1.addWidget(self.terminal_mode_combo)

        self.apply_mode_button = QtWidgets.QPushButton("Aplicar Modo")
        self.apply_mode_button.clicked.connect(self.apply_mode_changes)
        control_layout_row1.addWidget(self.apply_mode_button)

        # Selector de fuente de vibración (NI-9234 o Digilent WaveForms)
        control_layout_row1.addWidget(QtWidgets.QLabel("Fuente Vib:"))
        self.vib_source_combo = QtWidgets.QComboBox()
        self.vib_source_combo.addItems(["NI-9234", "Digilent WF"])
        control_layout_row1.addWidget(self.vib_source_combo)
        control_layout_row1.addStretch(1) # Push save button to the right in this row if needed

        # Second row of controls (Sample Rates, Save)
        control_layout_row2 = QtWidgets.QHBoxLayout()
        control_layout_row2.setSpacing(10)

        control_layout_row2.addWidget(QtWidgets.QLabel("SR Fuerza (Hz):"))
        self.force_sample_rate_combo = QtWidgets.QComboBox()
        self.force_sample_rate_combo.addItems(["1000", "2000", "5000", "10000", "20000"])
        self.force_sample_rate_combo.setCurrentText(str(FORCE_SAMPLE_RATE))
        control_layout_row2.addWidget(self.force_sample_rate_combo)

        control_layout_row2.addWidget(QtWidgets.QLabel("SR Vib (Hz):"))
        self.vib_sample_rate_combo = QtWidgets.QComboBox()
        self.vib_sample_rate_combo.addItems(["2000", "5000", "10000", "20000", "25600", "51200"])
        self.vib_sample_rate_combo.setCurrentText(str(VIB_SAMPLE_RATE))
        control_layout_row2.addWidget(self.vib_sample_rate_combo)

        self.apply_sample_rates_button = QtWidgets.QPushButton("Aplicar SR")
        self.apply_sample_rates_button.clicked.connect(self.apply_sample_rate_changes)
        control_layout_row2.addWidget(self.apply_sample_rates_button)
        
        control_layout_row2.addStretch(1) # Push save button to the right

        main_control_layout.addLayout(control_layout_row1)
        main_control_layout.addLayout(control_layout_row2)
        
        main_layout.addWidget(bottom_controls_container) # Add container to main layout

    def _enable_controls(self, enable):
        """Enable/disable controls based on acquisition state."""
        self.time_combo.setEnabled(enable)
        self.terminal_mode_combo.setEnabled(enable)
        self.apply_mode_button.setEnabled(enable)
        self.force_sample_rate_combo.setEnabled(enable)
        self.vib_sample_rate_combo.setEnabled(enable)
        self.apply_sample_rates_button.setEnabled(enable)
        self.optimizar_button.setEnabled(enable)
        self.btn_prediccion.setEnabled(enable)
        # Fuente de vibración
        if hasattr(self, 'vib_source_combo'):
            self.vib_source_combo.setEnabled(enable)
        # Keep start/stop button always enabled
        self.start_stop_button.setEnabled(True)


    def toggle_acquisition(self):
        if self.acquiring:
            # --- Stop Acquisition ---
            print("Stopping acquisition...")
            self.acquiring = False
            self.start_stop_button.setChecked(False)
            self.start_stop_button.setText("Iniciar")

            # Stop timers
            print("Deteniendo temporizadores...") # AGREGADO PARA DEBUGGING
            self.update_timer.stop()
            self.opt_timer.stop()
            self.reasoning_timer.stop()
            print("Temporizadores detenidos.") # AGREGADO PARA DEBUGGING

            # Stop threads
            print("Deteniendo hilos...") # AGREGADO PARA DEBUGGING
            if self.adquisicion_thread and self.adquisicion_thread.is_alive():
                self.adquisicion_thread.stop()
            if self.vib_thread and self.vib_thread.is_alive():
                self.vib_thread.stop()
            if self.bridge_thread and self.bridge_thread.is_alive():
                self.bridge_thread.stop()
            if self.reasoning_system and self.reasoning_system.is_alive():
                self.reasoning_system.stop()

            # Wait for threads to finish (with timeout)
            print("Esperando a que los hilos finalicen...") # AGREGADO PARA DEBUGGING
            if self.adquisicion_thread: self.adquisicion_thread.join(timeout=1.5)
            if self.vib_thread: self.vib_thread.join(timeout=1.5)
            if self.bridge_thread: self.bridge_thread.join(timeout=1.5)
            if self.reasoning_system: self.reasoning_system.join(timeout=1.5)

            print("Threads stopped.")
            self._enable_controls(False) # Disable controls when stopped

            # Guardar datos de la sesión y registrar detalles del experimento
            if self.current_session_base_filename: # Solo si se inició una sesión
                print(f"DEBUG: Deteniendo. current_session_base_filename: {self.current_session_base_filename}")
                force_data_len = len(self.all_data[0]) if self.all_data and len(self.all_data) > 0 else 0
                vib_data_len = len(self.all_vib_data[0]) if self.all_vib_data and len(self.all_vib_data) > 0 else 0
                print(f"DEBUG: Longitud de self.all_data[0] antes de guardar: {force_data_len}")
                print(f"DEBUG: Longitud de self.all_vib_data[0] antes de guardar: {vib_data_len}")
                
                self.save_current_session_data()
                self.log_experiment_details()
                
                self.current_session_base_filename = None # Reset for next session
                self.session_start_time = None
                self.session_start_time_str = ""
            else:
                print("DEBUG: Deteniendo, pero current_session_base_filename no estaba definido. No se guardarán datos de sesión.")


        else:
            # --- Start Acquisition ---
            # Registrar hora de inicio y generar nombre de archivo para la sesión
            self.session_start_time = time.time()
            self.session_start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.session_start_time))
            timestamp_file = time.strftime('%Y%m%d_%H%M%S')
            self.current_session_base_filename = os.path.join(AUTO_SAVE_BASE_DIRECTORY, f"sesion_{timestamp_file}")
            print(f"Nueva sesión iniciada. Archivos se guardarán con base: {self.current_session_base_filename}")

            print("Starting acquisition...")
            self.acquiring = True
            self.start_stop_button.setChecked(True)
            self.start_stop_button.setText("Detener")
            self._enable_controls(True) # Enable controls when running

            # Update sample rates from UI just before starting
            self.sample_rate = int(self.force_sample_rate_combo.currentText())
            self.vib_sample_rate = int(self.vib_sample_rate_combo.currentText())
            print(f"Using Force SR: {self.sample_rate} Hz, Vibration SR: {self.vib_sample_rate} Hz")

            # Recalculate buffer sizes based on current sample rates and time window
            self.buffer_size = max(int(self.time_window * self.sample_rate), 2)
            vib_buffer_size = max(int(self.time_window * self.vib_sample_rate), 2)
            self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)


            # Clear old data
            self.all_data = [[] for _ in range(len(FORCE_CANALES))]
            self.all_vib_data = [[] for _ in range(len(VIB_CANALES))]
            self.all_bridge_data = [[] for _ in range(len(BRIDGE_CANALES))]
            # Re-initialize deques for fresh start with correct maxlen
            self.datos_buffer = [deque(maxlen=self.buffer_size) for _ in range(len(FORCE_CANALES))]
            self.vib_buffer = [deque(maxlen=vib_buffer_size) for _ in range(len(VIB_CANALES))]
            bridge_buffer_size = max(int(self.time_window * BRIDGE_SAMPLE_RATE), 2)
            self.bridge_buffer = [deque(maxlen=bridge_buffer_size) for _ in range(len(BRIDGE_CANALES))]

            # Clear plots maybe?
            for curve in self.plot_curves: curve.clear()
            for curve in self.vib_plot_curves: curve.clear()
            # Puente removido: limpiar solo si existe
            if hasattr(self, 'bridge_plot_curves'):
                for curve in self.bridge_plot_curves: curve.clear()  # Limpiar curvas del puente

            # Get current terminal config
            selected_mode = self.terminal_mode_combo.currentText()
            self.terminal_config = self.terminal_modes.get(selected_mode, TerminalConfiguration.DIFF)
            print(f"Using terminal configuration: {selected_mode}")

            # Clear queues before starting acquisition
            while not datos_queue.empty():
                try: datos_queue.get_nowait()
                except: break
            while not vib_queue.empty():
                try: vib_queue.get_nowait()
                except: break
            while not bridge_queue.empty():
                try: bridge_queue.get_nowait()
                except: break
            while not reasoning_input_queue.empty():
                try: reasoning_input_queue.get_nowait()
                except: break
            while not reasoning_output_queue.empty():
                try: reasoning_output_queue.get_nowait()
                except: break
            print("Colas limpiadas.")

            # Start threads
            print("Starting acquisition threads...")
            self.adquisicion_thread = AdquisicionThread(
                dispositivo=DISPOSITIVO, canales=FORCE_CANALES, sample_rate=self.sample_rate, # Use dynamic SR
                muestras_por_bloque=MUESTRAS_POR_BLOQUE, terminal_config=self.terminal_config
            )
            # Selección de fuente de vibración
            vib_source_text = self.vib_source_combo.currentText() if hasattr(self, 'vib_source_combo') else "NI-9234"
            if "Digilent" in vib_source_text:
                try:
                    # Importación perezosa para evitar fallos si pydwf no está instalado
                    from digilent_analogin_thread import DigilentAnalogInThread
                    # Usamos 2 canales por coherencia con VIB_CANALES (shape: 2 x N)
                    self.vib_thread = DigilentAnalogInThread(
                        data_queue=vib_queue,
                        channels=[0, 1],
                        sample_rate=float(self.vib_sample_rate),
                        channel_range=5.0,
                        poll_interval_s=0.002,
                        verbose=False,
                    )
                    print("Vibration source: Digilent WF (AnalogIn)")
                except Exception as e:
                    # Si falla la importación/configuración de Digilent, hacer fallback a NI y notificar
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Digilent no disponible",
                        f"No se pudo inicializar Digilent (pydwf). Se usará NI-9234.\n\nDetalle: {e}"
                    )
                    self.vib_thread = VibrationAcquisitionThread(
                        dispositivo=VIB_DISPOSITIVO, canales=VIB_CANALES, sample_rate=self.vib_sample_rate, # Use dynamic SR
                        muestras_por_bloque=VIB_MUESTRAS_POR_BLOQUE, modo_entrada="Accelerometer"
                    )
            else:
                self.vib_thread = VibrationAcquisitionThread(
                    dispositivo=VIB_DISPOSITIVO, canales=VIB_CANALES, sample_rate=self.vib_sample_rate, # Use dynamic SR
                    muestras_por_bloque=VIB_MUESTRAS_POR_BLOQUE, modo_entrada="Accelerometer"
                )
            self.reasoning_system = IntelligentReasoningSystem(reasoning_input_queue, reasoning_output_queue)
            
            # Bridge module deshabilitado: no crear hilo
            self.bridge_thread = None
            print("NI 9219 (Puente) deshabilitado. Usando pestaña de Comparativa.")

            self.adquisicion_thread.start()
            self.vib_thread.start()
            if self.bridge_thread:  # Solo iniciar si existe
                self.bridge_thread.start()
            self.reasoning_system.start()
            
            active_modules = ["Force", "Vibration"]
            if self.bridge_thread:
                active_modules.append("Bridge")
            print(f"Acquisition threads started: {', '.join(active_modules)}.")

            # Start timers
            print("Iniciando temporizadores...") # AGREGADO PARA DEBUGGING
            self.update_timer.start(30) # Update plots slightly less frequently (e.g., 30ms)
            self.opt_timer.start(15000) # Run optimization less often (e.g., every 15s)
            self.reasoning_timer.start(500) # Check reasoning output every 500ms
            print("Timers started.")


    def update_timebase(self):
        time_text = self.time_combo.currentText()
        try:
            if "ms" in time_text:
                time_val = float(time_text.split()[0]) / 1000.0
            else:
                time_val = float(time_text.split()[0])
            print(f"Updating time window to {time_val} s")

            self.time_window = time_val
            # Recalculate buffer sizes using current (potentially dynamic) sample rates
            self.buffer_size = max(int(self.time_window * self.sample_rate), 2)
            vib_buffer_size = max(int(self.time_window * self.vib_sample_rate), 2)

            self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
            # vib_tiempo for vibration plots if its time axis needs to be different due to buffer size / sample rate

            # Resize deques by re-initializing with new maxlen
            self.datos_buffer = [deque(maxlen=self.buffer_size) for _ in range(len(FORCE_CANALES))]
            self.vib_buffer = [deque(maxlen=vib_buffer_size) for _ in range(len(VIB_CANALES))]

            # Update plot X-ranges
            for w in self.plot_widgets:
                w.setXRange(-self.time_window, 0)
            for w in self.vib_plot_widgets:
                 w.setXRange(-self.time_window, 0) # Assuming you want same time window visually

        except ValueError:
            print(f"Error parsing time window value: {time_text}")


    def apply_mode_changes(self):
        if not self.acquiring:
             print("Cannot apply mode changes while acquisition is stopped.")
             # Optionally, just store the desired config and apply it on next start
             selected_mode = self.terminal_mode_combo.currentText()
             self.terminal_config = self.terminal_modes.get(selected_mode, TerminalConfiguration.DIFF)
             print(f"Terminal configuration set to {selected_mode} (will apply on next start).")
             return

        print("Applying terminal configuration change...")
        # 1. Stop Acquisition
        was_acquiring = self.acquiring
        if self.acquiring:
            self.toggle_acquisition() # This stops threads and timers

        # 2. Wait briefly for threads to fully stop
        time.sleep(0.6)

        # 3. Get the new configuration (already done if not acquiring)
        selected_mode = self.terminal_mode_combo.currentText()
        self.terminal_config = self.terminal_modes.get(selected_mode, TerminalConfiguration.DIFF)
        print(f"New terminal configuration selected: {selected_mode}")

        # 4. Restart Acquisition (if it was running before)
        if was_acquiring:
             print("Restarting acquisition with new mode...")
             self.toggle_acquisition() # This restarts threads and timers with new self.terminal_config
        else:
             # If it wasn't acquiring, just update the internal state
             print("Acquisition remains stopped. Mode will be applied on next start.")

    def apply_poly_coefficients(self):
        """Aplica los coeficientes de interpolación polinomial desde la tabla."""
        try:
            # Actualizar los coeficientes desde la tabla
            for i, canal in enumerate(FORCE_CANALES):
                coefs = []
                for j in range(1, 5):  # Columnas 1-4 (c, a, b, d)
                    item = self.coef_table.item(i, j)
                    if item is not None:
                        coefs.append(float(item.text()))
                    else:
                        coefs.append(0.0)  # Valor por defecto si está vacío
                
                # Actualizar diccionario de coeficientes
                self.poly_coeffs[canal] = coefs
            
            # Confirmar al usuario
            print("Coeficientes de interpolación polinomial actualizados:")
            for canal, coefs in self.poly_coeffs.items():
                print(f"{canal}: c={coefs[0]}, a={coefs[1]}, b={coefs[2]}, d={coefs[3]}")
                
            # Mostrar un mensaje en la barra de estado (si existe)
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage("Coeficientes de interpolación actualizados", 3000)
                
        except Exception as e:
            print(f"Error al aplicar coeficientes: {e}")
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"Error: {e}", 5000)
    
    def apply_sample_rate_changes(self):
        new_force_sr = int(self.force_sample_rate_combo.currentText())
        new_vib_sr = int(self.vib_sample_rate_combo.currentText())

        if new_force_sr == self.sample_rate and new_vib_sr == self.vib_sample_rate:
            print("Sample rates unchanged.")
            return

        print(f"Applying new sample rates: Force SR={new_force_sr} Hz, Vib SR={new_vib_sr} Hz")
        self.sample_rate = new_force_sr
        self.vib_sample_rate = new_vib_sr

        # Update timebase calculations (buffer sizes, time arrays)
        self.update_timebase() # This will use the new self.sample_rate and self.vib_sample_rate

        if self.acquiring:
            print("Restarting acquisition with new sample rates...")
            # Stop current acquisition
            self.toggle_acquisition() # This calls stop part
            # Wait briefly for threads to fully stop
            time.sleep(0.6) # Adjust if necessary
            # Restart acquisition
            self.toggle_acquisition() # This calls start part, which now uses new SRs
        else:
            print("Sample rates updated. Will be used on next acquisition start.")


    def update_plots(self):
        print("DEBUG: update_plots llamado")
        if not self.acquiring: 
            print("DEBUG: No actualizando plots porque self.acquiring = False")
            return # Don't update if not running

        # --- Update Force Data (Módulo 9205) ---
        force_data_updated = False
        print(f"DEBUG: ¿Cola de datos vacía? {datos_queue.empty()}")
        while not datos_queue.empty():
            try:
                nuevos_datos = datos_queue.get_nowait() # shape (num_channels, num_samples)
                num_nuevas = nuevos_datos.shape[1]

                for i in range(len(FORCE_CANALES)):
                    # Append to long-term storage
                    # Limit all_data size to prevent memory issues (e.g., last 60s)
                    max_all_data_len = int(60 * self.sample_rate) # Example: 60 seconds
                    self.all_data[i].extend(nuevos_datos[i,:])
                    if len(self.all_data[i]) > max_all_data_len:
                        self.all_data[i] = self.all_data[i][-max_all_data_len:]

                    # Update rolling buffer for display using deque's automatic handling
                    self.datos_buffer[i].extend(nuevos_datos[i,:])

                    # Convert to physical units (Force) for calculations/display
                    # Use data from deque, convert to numpy array for calculations
                    current_force_samples_volt = np.array(self.datos_buffer[i])
                    if current_force_samples_volt.size == 0: continue # Skip if buffer is empty
                    # Saltar canales deshabilitados
                    if hasattr(self, 'channel_checkboxes') and not self.channel_checkboxes[i].isChecked():
                        self.value_labels[i].setText("Valor: --")
                        self.amp_labels[i].setText("Amp: --")
                        self.rms_labels[i].setText("RMS: --")
                        self.plot_curves[i].clear()
                        continue
                    
                    # Procesar y mostrar datos de voltaje sin conversión polinomial
                    voltage_data = current_force_samples_volt
                    # Calcular métricas (usar voltaje)
                    amp = np.ptp(voltage_data)
                    rms = np.sqrt(np.mean(voltage_data**2))
                    # Actualizar etiquetas (último valor de voltaje)
                    self.value_labels[i].setText(f"Valor: {voltage_data[-1]:.3f} V")
                    self.amp_labels[i].setText(f"Amp: {amp:.3f} V")
                    self.rms_labels[i].setText(f"RMS: {rms:.3f} V")
                    # Actualizar curva en el plot (aplicar normalización si está activada)
                    current_time_axis = np.linspace(-self.time_window, 0, len(voltage_data))
                    plot_force = voltage_data
                    try:
                        if hasattr(self, 'normalize_force_checkbox') and self.normalize_force_checkbox.isChecked():
                            plot_force = normalize_signal(plot_force)
                            # Ajustar rango visual a [-1,1]
                            if i < len(self.plot_widgets):
                                if not (hasattr(self, 'lock_y_force_checkbox') and self.lock_y_force_checkbox.isChecked()):
                                    self.plot_widgets[i].setYRange(-1.1, 1.1)
                        else:
                            # Restaurar rango normal de voltaje
                            if i < len(self.plot_widgets):
                                if not (hasattr(self, 'lock_y_force_checkbox') and self.lock_y_force_checkbox.isChecked()):
                                    self.plot_widgets[i].setYRange(FORCE_MIN_VOLTAGE, FORCE_MAX_VOLTAGE)
                    except Exception:
                        pass
                    self.plot_curves[i].setData(current_time_axis, plot_force)

                force_data_updated = True # Mark that we processed force data

            except queue.Empty:
                break # No more data in queue
            except Exception as e:
                print(f"Error processing force data queue: {e}")
                break

        # --- Update Vibration Data (Módulo 9234) ---
        vib_data_updated = False
        # vib_buffer_size = len(self.vib_buffer[0]) # Not needed as deque handles maxlen
        
        # Obtener la ganancia actual del SpinBox ANTES del bucle de la cola
        current_ae_gain = 1.0 # Valor por defecto si el spinbox no existe (salvaguarda)
        if hasattr(self, 'ae_gain_spinbox'):
            current_ae_gain = self.ae_gain_spinbox.value()
            # DEBUG: Imprimir la ganancia leída
            print(f"DEBUG: Ganancia Leída del SpinBox: {current_ae_gain}")
        else:
            print("DEBUG: self.ae_gain_spinbox NO ENCONTRADO!")

        while not vib_queue.empty():
            try:
                nuevos_datos_vib = vib_queue.get_nowait()
                num_nuevas_vib = nuevos_datos_vib.shape[1]

                for i in range(len(VIB_CANALES)):
                     # Append to long-term storage for vibration
                    max_all_vib_data_len = int(60 * self.vib_sample_rate) # Example: 60 seconds of vib data
                    self.all_vib_data[i].extend(nuevos_datos_vib[i,:])
                    if len(self.all_vib_data[i]) > max_all_vib_data_len:
                        self.all_vib_data[i] = self.all_vib_data[i][-max_all_vib_data_len:]

                     # Update rolling buffer for display (using deque)
                    self.vib_buffer[i].extend(nuevos_datos_vib[i,:])

                    # Convert to physical units (Acceleration)
                    current_vib_samples = np.array(self.vib_buffer[i])
                    if current_vib_samples.size == 0: continue

                    # Assuming direct conversion V -> g if add_ai_accel_chan worked.
                    # If add_ai_accel_chan failed and it's reading voltage,
                    # ACC_CONVERSION would be needed here.
                    # For now, assuming data is already in 'g' as per original implication.
                    accel_data_phys = current_vib_samples # * ACC_CONVERSION # Apply if reading raw voltage

                    # --- Aplicar Ganancia para el Canal de Emisiones Acústicas ---
                    # Asumimos que el canal de emisiones acústicas es VIB_CANALES[0]
                    if i == 0: # Si es el primer canal de vibración (nuestro sensor AE)
                        # DEBUG: Imprimir datos antes de la ganancia para el canal 0
                        if accel_data_phys.size > 0:
                            print(f"DEBUG: Ch{i} ({VIB_CANALES[i]}) antes de ganancia (última muestra): {accel_data_phys[-1]:.4f}")
                        
                        accel_data_phys = accel_data_phys * current_ae_gain
                        
                        # DEBUG: Imprimir datos después de la ganancia para el canal 0
                        if accel_data_phys.size > 0:
                            print(f"DEBUG: Ch{i} ({VIB_CANALES[i]}) DESPUÉS de ganancia ({current_ae_gain:.2f}) (última muestra): {accel_data_phys[-1]:.4f}")
                    # --- Fin de Aplicar Ganancia ---


                    # Calculate metrics
                    # freq = medir_frecuencia_fft(accel_data_phys, self.vib_sample_rate)
                    amp = np.ptp(accel_data_phys)
                    rms = np.sqrt(np.mean(accel_data_phys**2))

                    # Update labels
                    # Para el canal AE, las unidades podrían no ser 'g' después de la ganancia.
                    # Considera cambiar la etiqueta de unidad o mostrar "unidades arbitrarias" o "mV" si la ganancia convierte a eso.
                    unit_label = "g"
                    if i == 0: # Si es el canal AE con ganancia
                        unit_label = "V (AE)" # O la unidad que corresponda después de la ganancia. Ajusta esto.
                                       # Si la ganancia es solo un multiplicador y la entrada era 'g', entonces podría seguir siendo 'g' escalado.
                                       # Si la entrada al DAQ es Voltios y la 'ganancia' es para amplificar esos voltios, la unidad es V.

                    self.vib_value_labels[i].setText(f"Valor: {accel_data_phys[-1]:.3f} {unit_label}")
                    # self.vib_freq_labels[i].setText(f"Frec: {freq:.1f} Hz")
                    self.vib_amp_labels[i].setText(f"Amp: {amp:.3f} {unit_label}")
                    self.vib_rms_labels[i].setText(f"RMS: {rms:.3f} {unit_label}")

                    # Update plot curve (aplicar normalización si está activada)
                    current_vib_time_axis = np.linspace(-self.time_window, 0, len(accel_data_phys))
                    plot_vib = accel_data_phys
                    try:
                        if hasattr(self, 'normalize_vib_checkbox') and self.normalize_vib_checkbox.isChecked():
                            plot_vib = normalize_signal(plot_vib)
                            if i < len(self.vib_plot_widgets):
                                if not (hasattr(self, 'lock_y_vib_checkbox') and self.lock_y_vib_checkbox.isChecked()):
                                    self.vib_plot_widgets[i].setYRange(-1.1, 1.1)
                        else:
                            if i < len(self.vib_plot_widgets):
                                if not (hasattr(self, 'lock_y_vib_checkbox') and self.lock_y_vib_checkbox.isChecked()):
                                    self.vib_plot_widgets[i].setYRange(ACCEL_MIN_G, ACCEL_MAX_G)
                    except Exception:
                        pass
                    # Ensure vib_plot_curves[i] exists and data is valid
                    self.vib_plot_curves[i].setData(current_vib_time_axis, plot_vib)

                vib_data_updated = True

            except queue.Empty:
                break
            except Exception as e:
                print(f"Error processing vibration data queue: {e}")
                break

        # --- Bridge module removido: omitir actualización de puente ---
        bridge_data_updated = False

        # Update bridge status
        if hasattr(self, 'bridge_status_label'):
            if bridge_data_updated:
                self.bridge_status_label.setText("Estado: ✅ Datos activos @ 2000 Hz")
            elif hasattr(self, 'bridge_thread') and self.bridge_thread:
                self.bridge_status_label.setText("Estado: ⏳ Esperando datos...")
            else:
                self.bridge_status_label.setText("Estado: ❌ Deshabilitado")

        # --- Update Comparative Plots (Force vs Vibration) ---
        try:
            if hasattr(self, 'comp_time_widget'):
                # Selección de canales
                sel_force = 0
                sel_vib = 0
                if hasattr(self, 'comp_force_combo'):
                    try:
                        ch_text = self.comp_force_combo.currentText()
                        sel_force = FORCE_CANALES.index(ch_text)
                    except Exception:
                        sel_force = 0
                if hasattr(self, 'comp_vib_combo'):
                    try:
                        ch_text = self.comp_vib_combo.currentText()
                        sel_vib = VIB_CANALES.index(ch_text)
                    except Exception:
                        sel_vib = 0

                # Extraer buffers
                force_buff = np.array(self.datos_buffer[sel_force]) if sel_force < len(self.datos_buffer) else np.array([])
                vib_buff = np.array(self.vib_buffer[sel_vib]) if sel_vib < len(self.vib_buffer) else np.array([])

                n = min(len(force_buff), len(vib_buff))
                if n >= 2:
                    f_seg = force_buff[-n:]
                    v_seg = vib_buff[-n:]

                    # Tiempo para overlay (usar ventana actual)
                    t_axis = np.linspace(-self.time_window, 0, n)

                    # Normalización opcional para overlay
                    if hasattr(self, 'comp_normalize_checkbox') and self.comp_normalize_checkbox.isChecked():
                        f_plot = normalize_signal(f_seg)
                        v_plot = normalize_signal(v_seg)
                        self.comp_time_widget.setYRange(-1.1, 1.1)
                    else:
                        f_plot = f_seg
                        v_plot = v_seg
                        # AutoRange para ver ambas unidades si no están normalizadas
                        self.comp_time_widget.enableAutoRange(axis='y')

                    # Actualizar overlay
                    self.comp_time_curve_force.setData(t_axis, f_plot)
                    self.comp_time_curve_vib.setData(t_axis, v_plot)

                    # Lissajous scatter SIEMPRE normalizado
                    f_norm = normalize_signal(f_seg)
                    v_norm = normalize_signal(v_seg)
                    self.comp_scatter_curve.setData(f_norm, v_norm)
        except Exception as e:
            print(f"Error updating comparative plots: {e}")

        # --- Update Bouc-Wen Plots (if force data was updated) ---
        if force_data_updated:
            n_fase = min(500, self.buffer_size) # Number of points for phase plots

            # Check if enough data exists in buffers (deques)
            # Convert deques to numpy arrays for processing
            buffer0_np = np.array(self.datos_buffer[0])
            buffer1_np = np.array(self.datos_buffer[1])

            if len(buffer0_np) >= 1 and len(buffer1_np) >=1: # Ensure not empty
                # Use all available data in the deque up to n_fase for these plots
                sig0_volt = buffer0_np[-n_fase:]
                sig1_volt = buffer1_np[-n_fase:]
                
                actual_n_fase = len(sig0_volt) # Actual number of points used

                # Lissajous 2D (Normalized Voltages)
                sig0n = normalize_signal(sig0_volt)
                sig1n = normalize_signal(sig1_volt)
                self.phase2d_curve.setData(sig0n, sig1n)

                # 3D Plot (Time, Normalized Voltages)
                time_axis_norm = np.linspace(-1, 0, actual_n_fase) # Use actual_n_fase
                pos = np.zeros((actual_n_fase, 3), dtype=np.float32) # Use actual_n_fase
                pos[:, 0] = time_axis_norm 
                pos[:, 1] = sig0n         
                pos[:, 2] = sig1n         
                self.phase3d_line.setData(pos=pos, color=(0,1,0,0.8), width=1.5)

                # Hysteresis Plot z(t) vs Input Voltage
                # *** CORRECTION HERE: Use voltage for plot ***
                t_data_bw_plot = np.linspace(0, (actual_n_fase - 1) / self.sample_rate, actual_n_fase) # Use actual_n_fase
                input_signal_volt = sig0_volt # Use raw voltage from buffer[0]

                # Calculate z(t) using the *voltage* input and current parameters
                _, z_vals_plot = bouc_wen_model(self.params_opt, t_data_bw_plot, input_signal_volt)

                # Plot z(t) vs Input Voltage
                self.bw_phase_curve.setData(input_signal_volt, z_vals_plot)
                # Optionally adjust Y range dynamically based on z_vals_plot min/max
                # self.bw_phase_widget.setYRange(np.min(z_vals_plot) * 1.1, np.max(z_vals_plot) * 1.1)


        # --- Update FFT Plots (if data updated) ---
        if force_data_updated:
            for i in range(len(FORCE_CANALES)):
                # Use physical units for FFT meaningfulness
                force_data_volt_np = np.array(self.datos_buffer[i])
                if force_data_volt_np.size < 2: continue # Not enough data for FFT
                force_data_phys = force_data_volt_np * EFFECTIVE_FORCE_CONVERSION
                freqs, mag = fft_spectrum(force_data_phys, self.sample_rate) # Use dynamic sample rate
                self.fft_curves_9205[i].setData(freqs, mag)
                self.fft_plots_9205[i].setXRange(0, self.sample_rate / 2) # Show up to Nyquist
                # Optionally adjust Y range
                # self.fft_plots_9205[i].enableAutoRange(axis='y')

        if vib_data_updated:
            for i in range(len(VIB_CANALES)):
                accel_data_np = np.array(self.vib_buffer[i])
                if accel_data_np.size < 2: continue # Not enough data for FFT
                accel_data_phys = accel_data_np # * ACC_CONVERSION # Assuming already in g
                freqs, mag = fft_spectrum(accel_data_phys, self.vib_sample_rate) # Use dynamic sample rate
                self.fft_curves_9234[i].setData(freqs, mag)
                self.fft_plots_9234[i].setXRange(0, self.vib_sample_rate / 2)
                # self.fft_plots_9234[i].enableAutoRange(axis='y')

        # --- Feature Extraction and Prediction/Reasoning ---
        # Run this less frequently or if significant data updated
        # Using force_data_updated or vib_data_updated as triggers
        if force_data_updated or vib_data_updated:
            # print("DEBUG: Calling run_prediction_and_reasoning()") # Descomentar para debugging intensivo
            self.run_prediction_and_reasoning()


    def run_prediction_and_reasoning(self):
        """Extract features and run SVM/CBR."""
        # print("DEBUG: Entered run_prediction_and_reasoning") # Descomentar para debugging intensivo
        n_feat = 500 # Number of samples for feature extraction
        
        # Convert deques to numpy arrays for feature extraction
        force_buffer0_np = np.array(self.datos_buffer[0])
        force_buffer1_np = np.array(self.datos_buffer[1])
        vib_buffer0_np = np.array(self.vib_buffer[0])

        if len(force_buffer0_np) >= n_feat and len(vib_buffer0_np) >= n_feat:
            try:
                # Prepare data (use physical units)
                input_force_volt = force_buffer0_np[-n_feat:]
                # output_force_volt = force_buffer1_np[-n_feat:] # If ai1 is output voltage

                input_force_phys = input_force_volt * EFFECTIVE_FORCE_CONVERSION
                # output_force_phys = output_force_volt * EFFECTIVE_FORCE_CONVERSION

                vib_data_raw = vib_buffer0_np[-n_feat:]
                vib_data_phys = vib_data_raw # * ACC_CONVERSION # Assuming already in g

                # Time vector for features
                t_feat = np.linspace(0, (n_feat - 1) / self.sample_rate, n_feat) # Use dynamic SR for force data time

                # --- Feature Extraction ---
                # Force features (e.g., from Bouc-Wen state or raw signals)
                # For Bouc-Wen feature extraction, decide if voltage or physical units are more appropriate for the model's input.
                # Using physical input force for consistency with optimization.
                _, bw_hyst_feat = bouc_wen_model(self.params_opt, t_feat, input_force_phys)
                force_features = fe.extract_force_features(bw_hyst_feat, t_feat)

                # Vibration features
                vib_features = fe.extract_vibration_features(vib_data_phys, self.vib_sample_rate) # Use dynamic SR for vib

                # Fuse features
                combined_features = fe.fuse_features(force_features, vib_features)

                # Ensure combined_features is a flat numpy array
                if isinstance(combined_features, (list, tuple)):
                   combined_features = np.array(combined_features).flatten()
                elif isinstance(combined_features, np.ndarray):
                   combined_features = combined_features.flatten()
                else:
                    print(f"Warning: Unexpected feature type: {type(combined_features)}")
                    return # Cannot proceed

                if combined_features is None or combined_features.size == 0:
                     print("Warning: Feature extraction resulted in empty features.")
                     return


                # --- SVM Prediction ---
                svm_label, svm_prob = self.svm_predictor.predict(combined_features)
                self.svm_result_label.setText(f"Estado (SVM): {svm_label} ({svm_prob*100:.1f}%)")

                # --- CBR Retrieval & Reasoning Input ---
                # Retrieve similar cases (optional display)
                casos_similares = self.cbr_analyzer.retrieve(combined_features, threshold=150.0) # Adjust threshold
                self.cbr_list.clear()
                for case, dist in casos_similares[:10]: # Display top 10
                    label = case.get('label', 'N/A') # Safe access to label
                    self.cbr_list.addItem(f"Caso: {label}, Dist: {dist:.2f}")

                # Send features to reasoning thread
                if not reasoning_input_queue.full():
                    reasoning_input_queue.put_nowait(combined_features)
                else:
                    # Handle full queue - discard oldest, add newest
                    try:
                        reasoning_input_queue.get_nowait()
                        reasoning_input_queue.put_nowait(combined_features)
                    except queue.Empty:
                         reasoning_input_queue.put_nowait(combined_features)


            except Exception as e:
                print(f"Error during feature extraction/prediction: {e}")
                import traceback
                traceback.print_exc()
            # finally: # Descomentar para debugging intensivo
                # print("DEBUG: Exiting run_prediction_and_reasoning")


    def update_reasoning_visualization(self):
        """Updates the CBR result display based on the reasoning thread output."""
        if not self.acquiring: return

        try:
            while not reasoning_output_queue.empty():
                result = reasoning_output_queue.get_nowait() # Get latest result
                # Update label
                self.reasoning_status_label.setText(f"Estado Predicho (CBR): {result}")
                # Update visual indicator
                style = self.reasoning_color_map.get(result, "background-color: gray;")
                self.reasoning_indicator.setStyleSheet(f"border-radius: 50px; {style}")

        except queue.Empty:
            pass # No new results
        except Exception as e:
            print(f"Error updating reasoning visualization: {e}")
            traceback.print_exc() # AGREGADO PARA DEBUGGING

    def ajustar_modelo(self):
        if self._optimization_running:
            print("Optimization already in progress.")
            return
        # Use data from all_data for more history
        N = min(5000, len(self.all_data[0])) # Use up to 5000 points from history
        if N < 100:
            QtWidgets.QMessageBox.warning(self, "Datos insuficientes", "No hay suficientes datos históricos para optimizar.")
            return

        self._optimization_running = True
        self.optimizar_button.setEnabled(False)
        self.optimizar_button.setText("Optimizando...")
        print(f"Starting Bouc-Wen optimization with N={N} points...")

        # Prepare data (use physical units for optimization)
        t_data = np.linspace(0, (N - 1) / self.sample_rate, N) # Use dynamic sample rate
        # Input signal (e.g., Force X+ in N)
        corriente_data_phys = np.array(self.all_data[0][-N:]) * EFFECTIVE_FORCE_CONVERSION
        # Experimental Output signal (e.g., Force X- in N)
        fuerza_data_phys = np.array(self.all_data[1][-N:]) * EFFECTIVE_FORCE_CONVERSION

        params0 = self.params_opt[:] # Use current params as starting guess

        def optimize_task():
            """The actual optimization calculation."""
            try:
                print("Applying Savitzky-Golay filter...")
                # Apply smoothing (adjust window/polyorder if needed)
                i_smooth = savgol_filter(corriente_data_phys, min(31, N//2 * 2 -1), 3) # Window must be odd and < N
                f_smooth = savgol_filter(fuerza_data_phys, min(31, N//2 * 2 - 1), 3)

                print("Running least_squares...")
                start_time = time.time()
                # Definir límites para los parámetros, especialmente para n (params[3])
                # A, B, C, n, k
                # Mantener n >= 0.1 (o algún valor pequeño positivo)
                # Los otros parámetros pueden ser menos restrictivos o basados en el conocimiento del dominio.
                # Por ahora, solo restringimos n.
                param_bounds_lower = [-np.inf, -np.inf, -np.inf, 0.1, -np.inf]
                param_bounds_upper = [np.inf,  np.inf,  np.inf, np.inf,  np.inf]

                res = least_squares(
                    error_bouc_wen,
                    params0,
                    args=(t_data, i_smooth, f_smooth),
                    bounds=(param_bounds_lower, param_bounds_upper), # LÍMITES AÑADIDOS
                    method='lm', # Levenberg-Marquardt
                    ftol=1e-5, xtol=1e-5, # Tighter tolerances
                    max_nfev=200, # More iterations allowed
                    verbose=0 # Set to 1 or 2 for optimization details
                )
                end_time = time.time()
                print(f"Optimization finished in {end_time - start_time:.2f}s. Success: {res.success}, Cost: {res.cost:.4e}")

                if res.success:
                    # Return optimized params and original (unsmoothed) data for comparison plot
                    return res.x, res.cost, t_data, corriente_data_phys, fuerza_data_phys
                else:
                    print(f"Optimization failed: {res.message}")
                    return None # Indicate failure
            except Exception as ex:
                print(f"Error during optimization task: {ex}")
                import traceback
                traceback.print_exc()
                return None

        def finish_optimization(future):
            """Callback executed in the main thread after optimization finishes."""
            try:
                result = future.result()
                if result is not None:
                    params_opt, final_err, t_opt, i_opt, f_opt = result
                    print(f"Optimized Parameters: A={params_opt[0]:.4f}, B={params_opt[1]:.4f}, C={params_opt[2]:.4f}, n={params_opt[3]:.4f}, k={params_opt[4]:.4f}")
                    self.params_opt = params_opt # Update parameters used for plotting/features
                    # Update UI labels
                    for k, v in zip(["A","B","C","n","k"], params_opt):
                        self.param_labels[k].setText(f"{v:.4f}")
                    # Show comparison plot
                    self.mostrar_grafico_comparacion(params_opt, t_opt, i_opt, f_opt)
                else:
                    QtWidgets.QMessageBox.warning(self, "Optimización Fallida", "No se pudo ajustar el modelo Bouc-Wen.")
            except Exception as e:
                 print(f"Error in finish_optimization callback: {e}")
                 traceback.print_exc() # AGREGADO PARA DEBUGGING
            finally:
                # Re-enable button and reset state regardless of success/failure
                self._optimization_running = False
                if self.acquiring: # Only enable if still acquiring
                    self.optimizar_button.setEnabled(True)
                self.optimizar_button.setText("Ajustar Modelo Bouc-Wen")
                print("Optimization UI update complete.")

        # Submit the optimization task to the executor
        future = self.executor.submit(optimize_task)
        # Connect the callback to be executed when the future is done
        future.add_done_callback(lambda f: QtCore.QTimer.singleShot(0, lambda: finish_optimization(f)))


    def ejecutar_optimizacion(self):
        """Called periodically by opt_timer to trigger optimization."""
        if self.acquiring and not self._optimization_running:
            # Only run if acquiring and no optimization is currently running
            # Check if enough data has accumulated in all_data
            if len(self.all_data[0]) > 2000: # Require at least 2000 points for auto-optimization
                print("Auto-triggering Bouc-Wen optimization...")
                # Limit history size before optimizing if it gets too large
                max_hist_opt = 30000 # Use last 30k points max for optimization data source
                if len(self.all_data[0]) > max_hist_opt:
                    print(f"Trimming history buffer to {max_hist_opt} points.")
                    for i in range(len(self.all_data)):
                        self.all_data[i] = self.all_data[i][-max_hist_opt:]

                self.ajustar_modelo()
            else:
                print("Skipping auto-optimization: Insufficient history data.")

    def mostrar_grafico_comparacion(self, params_opt, t_data, corriente_data, fuerza_data):
        """Plots experimental vs modeled force using Matplotlib."""
        try:
            print("Generating comparison plot...")
            fuerza_modelada, _ = bouc_wen_model(params_opt, t_data, corriente_data)

            plt.figure(figsize=(10, 6))
            plt.plot(t_data, fuerza_data, label="Fuerza Experimental (Medida)", alpha=0.7)
            plt.plot(t_data, fuerza_modelada, label="Fuerza Modelada (Bouc-Wen)", linestyle='--', color='red')
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Fuerza (N)")
            plt.title("Comparación Experimental vs. Modelo Bouc-Wen Ajustado")
            plt.legend()
            plt.grid(True)
            plt.show(block=False) # Show plot without blocking the main GUI thread
            print("Comparison plot displayed.")
        except Exception as e:
            print(f"Error displaying comparison plot: {e}")
            traceback.print_exc() # AGREGADO PARA DEBUGGING

    def save_current_session_data(self):
        """Guarda los datos de la sesión actual automáticamente sin pedir al usuario."""
        if not self.current_session_base_filename:
            print("Error: No hay nombre base de archivo para la sesión actual. No se guardarán datos.")
            return

        # Verificar existencia y permisos del directorio base
        if not os.path.exists(AUTO_SAVE_BASE_DIRECTORY):
            print(f"Error Crítico: El directorio de guardado automático {AUTO_SAVE_BASE_DIRECTORY} no existe al momento de guardar.")
            QtWidgets.QMessageBox.critical(self, "Error de Guardado", f"El directorio de guardado automático no existe:\n{AUTO_SAVE_BASE_DIRECTORY}")
            return
        if not os.access(AUTO_SAVE_BASE_DIRECTORY, os.W_OK):
            print(f"Error Crítico: No hay permisos de escritura en el directorio {AUTO_SAVE_BASE_DIRECTORY}.")
            QtWidgets.QMessageBox.critical(self, "Error de Guardado", f"No hay permisos de escritura en el directorio:\n{AUTO_SAVE_BASE_DIRECTORY}")
            return

        # Asegurarse de que las tasas de muestreo son válidas
        current_force_sr = self.sample_rate
        current_vib_sr = self.vib_sample_rate

        if current_force_sr <= 0:
            print(f"Error: Frecuencia de muestreo de fuerza inválida ({current_force_sr}). No se pueden generar datos de tiempo para fuerza.")
            # No retornamos aquí necesariamente, podríamos intentar guardar vibración si es válida
        if current_vib_sr <= 0:
            print(f"Error: Frecuencia de muestreo de vibración inválida ({current_vib_sr}). No se pueden generar datos de tiempo para vibración.")
            # No retornamos aquí necesariamente

        force_data_available = self.all_data and any(len(c) > 0 for c in self.all_data)
        vib_data_available = self.all_vib_data and any(len(c) > 0 for c in self.all_vib_data)
        bridge_data_available = self.all_bridge_data and any(len(c) > 0 for c in self.all_bridge_data)

        if not force_data_available and not vib_data_available and not bridge_data_available:
            print("No hay datos acumulados en la sesión actual (all_data, all_vib_data, all_bridge_data están vacíos). No se guardará nada.")
            return

        print(f"Intentando guardar datos de la sesión automáticamente con base: {self.current_session_base_filename}")
        guardado_exitoso_fuerza = False
        guardado_exitoso_vib = False

        # Save Force Data
        if force_data_available:
            filename_force = f"{self.current_session_base_filename}_fuerza.csv"
            print(f"DEBUG: Preparando para guardar datos de fuerza en: {filename_force}")
            try:
                min_len_force = 0
                # Determinar la longitud de los datos a guardar (basado en el primer canal con datos)
                for i, channel_data in enumerate(self.all_data):
                    if len(channel_data) > 0:
                        min_len_force = len(channel_data)
                        break
                
                if min_len_force > 0:
                    if current_force_sr <= 0:
                        print(f"Error: SR de fuerza ({current_force_sr}) inválido. No se puede guardar archivo de fuerza.")
                        QtWidgets.QMessageBox.warning(self, "Error SR Fuerza", f"Frecuencia de muestreo de fuerza inválida ({current_force_sr}). No se guardará el archivo de fuerza.")
                    else:
                        print(f"DEBUG: Guardando {min_len_force} muestras de datos de fuerza. SR: {current_force_sr} Hz.")
                        time_vector_force = np.linspace(0, (min_len_force - 1) / current_force_sr, min_len_force)
                        header_force_list = ["Tiempo(s)"] + [f"Fuerza_{c}(V)" for c in FORCE_CANALES]
                        header_force = ",".join(header_force_list)
                        
                        data_arrays_force = [time_vector_force]
                        for channel_idx, channel_data_list in enumerate(self.all_data):
                            current_ch_samples = np.array(channel_data_list[:min_len_force])
                            if len(current_ch_samples) < min_len_force:
                                full_ch_samples = np.full(min_len_force, np.nan) # Rellenar con NaN
                                if len(current_ch_samples) > 0:
                                    full_ch_samples[:len(current_ch_samples)] = current_ch_samples
                                data_arrays_force.append(full_ch_samples)
                            else:
                                data_arrays_force.append(current_ch_samples)

                        data_to_save_force = np.array(data_arrays_force).T
                        np.savetxt(filename_force, data_to_save_force, delimiter=",", header=header_force, comments="")
                        print(f"Datos de fuerza de la sesión guardados en: {filename_force}")
                        guardado_exitoso_fuerza = True
                else:
                    print("DEBUG: No hay datos de fuerza (min_len_force <= 0) en la sesión para guardar.")
            except Exception as e:
                print(f"Error CRÍTICO guardando datos de fuerza de la sesión: {e}")
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(self, "Error de Guardado (Fuerza)", f"No se pudo guardar el archivo de fuerza:\n{filename_force}\nError: {e}")
        else:
            print("DEBUG: No hay datos de fuerza disponibles en esta sesión (evaluado por force_data_available).")

        # Save Vibration Data
        if vib_data_available:
            filename_vib = f"{self.current_session_base_filename}_vibracion.csv"
            print(f"DEBUG: Preparando para guardar datos de vibración en: {filename_vib}")
            try:
                min_len_vib = 0
                for i, channel_data in enumerate(self.all_vib_data):
                    if len(channel_data) > 0:
                        min_len_vib = len(channel_data)
                        break
                
                if min_len_vib > 0:
                    if current_vib_sr <= 0:
                        print(f"Error: SR de vibración ({current_vib_sr}) inválido. No se puede guardar archivo de vibración.")
                        QtWidgets.QMessageBox.warning(self, "Error SR Vibración", f"Frecuencia de muestreo de vibración inválida ({current_vib_sr}). No se guardará el archivo de vibración.")
                    else:
                        print(f"DEBUG: Guardando {min_len_vib} muestras de datos de vibración. SR: {current_vib_sr} Hz.")
                        time_vector_vib = np.linspace(0, (min_len_vib - 1) / current_vib_sr, min_len_vib)
                        header_vib_list = ["Tiempo(s)"] + [f"Vibracion_{c}(g_or_V)" for c in VIB_CANALES]
                        header_vib = ",".join(header_vib_list)
                        
                        data_arrays_vib = [time_vector_vib]
                        for channel_idx, channel_data_list in enumerate(self.all_vib_data):
                            current_ch_samples = np.array(channel_data_list[:min_len_vib])
                            if len(current_ch_samples) < min_len_vib:
                                full_ch_samples = np.full(min_len_vib, np.nan) # Rellenar con NaN
                                if len(current_ch_samples) > 0:
                                    full_ch_samples[:len(current_ch_samples)] = current_ch_samples
                                data_arrays_vib.append(full_ch_samples)
                            else:
                                data_arrays_vib.append(current_ch_samples)
                                
                        data_to_save_vib = np.array(data_arrays_vib).T
                        np.savetxt(filename_vib, data_to_save_vib, delimiter=",", header=header_vib, comments="")
                        print(f"Datos de vibración de la sesión guardados en: {filename_vib}")
                        guardado_exitoso_vib = True
                else:
                    print("DEBUG: No hay datos de vibración (min_len_vib <= 0) en la sesión para guardar.")
            except Exception as e:
                print(f"Error CRÍTICO guardando datos de vibración de la sesión: {e}")
                traceback.print_exc()
                QtWidgets.QMessageBox.critical(self, "Error de Guardado (Vibración)", f"No se pudo guardar el archivo de vibración:\n{filename_vib}\nError: {e}")
        else:
            print("DEBUG: No hay datos de vibración disponibles en esta sesión (evaluado por vib_data_available).")

        # Save Bridge Data (NI 9219)
        # Puente removido: no guardar datos de puente
        bridge_data_available = False
        guardado_exitoso_bridge = False
        
        # (Guardado de puente omitido)

        if guardado_exitoso_fuerza or guardado_exitoso_vib or guardado_exitoso_bridge:
             QtWidgets.QMessageBox.information(self, "Guardado Automático", f"Datos de sesión guardados en la carpeta:\n{AUTO_SAVE_BASE_DIRECTORY}")
        elif force_data_available or vib_data_available or bridge_data_available: # Se intentó guardar pero al menos uno falló
             # Este mensaje podría aparecer si, por ejemplo, solo los datos de fuerza estaban disponibles y fallaron,
             # o si ambos estaban disponibles y ambos fallaron.
             # Si uno tuvo éxito y el otro falló, el mensaje de éxito anterior ya se mostró,
             # pero el error específico del que falló también se mostró como QMessageBox.critical.
             # Podríamos refinar esto, pero por ahora, si hubo un intento y no todo fue éxito, alertar.
             QtWidgets.QMessageBox.warning(self, "Fallo Guardado Automático", "Se intentó guardar datos pero ocurrió un error. Revise la consola y los mensajes de error anteriores.")
        # No mostrar mensaje si no había nada que guardar (ya cubierto por el return inicial).

    def log_experiment_details(self):
        """Registra los detalles del experimento en un archivo CSV."""
        print("DEBUG: Intentando registrar detalles del experimento...")
        if not self.session_start_time or not self.current_session_base_filename:
            print("Error: Faltan datos de la sesión (session_start_time o current_session_base_filename) para el registro.")
            return

        session_end_time = time.time()
        session_end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end_time))
        duration_seconds = session_end_time - self.session_start_time
        
        log_header = [
            "Timestamp Inicio", "Timestamp Fin", "Duracion (s)",
            "SR Fuerza (Hz)", "SR Vib (Hz)", "Modo Terminal Fuerza",
            "Archivo Base Sesion", "Archivo Fuerza", "Archivo Vibracion"
        ]
        
        force_file_name = f"{os.path.basename(self.current_session_base_filename)}_fuerza.csv"
        vib_file_name = f"{os.path.basename(self.current_session_base_filename)}_vibracion.csv"

        log_entry = [
            self.session_start_time_str,
            session_end_time_str,
            f"{duration_seconds:.2f}",
            self.sample_rate,
            self.vib_sample_rate,
            self.terminal_mode_combo.currentText(), # Modo terminal actual
            os.path.basename(self.current_session_base_filename),
            force_file_name if any(len(c) > 0 for c in self.all_data) else "N/A",
            vib_file_name if any(len(c) > 0 for c in self.all_vib_data) else "N/A"
        ]

        try:
            file_exists = os.path.isfile(EXPERIMENT_LOG_FILE_PATH)
            with open(EXPERIMENT_LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(EXPERIMENT_LOG_FILE_PATH) == 0:
                    writer.writerow(log_header)
                writer.writerow(log_entry)
            print(f"Detalles del experimento registrados en: {EXPERIMENT_LOG_FILE_PATH}")
        except Exception as e:
            print(f"Error al registrar detalles del experimento: {e}")
            traceback.print_exc()

    def closeEvent(self, event):
        """Ensures threads are stopped cleanly when closing the window."""
        print("Close event triggered. Stopping all background processes...")
        self.acquiring = False # Signal threads/timers to stop

        # Stop timers first
        print("Cerrando: Deteniendo temporizadores...") # AGREGADO PARA DEBUGGING
        self.update_timer.stop()
        self.opt_timer.stop()
        self.reasoning_timer.stop()
        print("Cerrando: Temporizadores detenidos.") # AGREGADO PARA DEBUGGING

        # Stop threads
        print("Cerrando: Deteniendo hilos...") # AGREGADO PARA DEBUGGING
        if self.adquisicion_thread and self.adquisicion_thread.is_alive():
            self.adquisicion_thread.stop()
        if self.vib_thread and self.vib_thread.is_alive():
            self.vib_thread.stop()
        if self.bridge_thread and self.bridge_thread.is_alive():
            self.bridge_thread.stop()
        if self.reasoning_system and self.reasoning_system.is_alive():
            self.reasoning_system.stop()

        # Wait for threads (increase timeout slightly if needed)
        print("Waiting for threads to join...")
        if self.adquisicion_thread:
            print("Cerrando: Esperando por adquisicion_thread...") # AGREGADO PARA DEBUGGING
            self.adquisicion_thread.join(timeout=2.0)
            if self.adquisicion_thread.is_alive():
                print("ADVERTENCIA: adquisicion_thread no finalizó a tiempo.") # AGREGADO PARA DEBUGGING
        if self.vib_thread:
            print("Cerrando: Esperando por vib_thread...") # AGREGADO PARA DEBUGGING
            self.vib_thread.join(timeout=2.0)
            if self.vib_thread.is_alive():
                print("ADVERTENCIA: vib_thread no finalizó a tiempo.") # AGREGADO PARA DEBUGGING
        if self.bridge_thread:
            print("Cerrando: Esperando por bridge_thread...") # AGREGADO PARA DEBUGGING
            self.bridge_thread.join(timeout=2.0)
            if self.bridge_thread.is_alive():
                print("ADVERTENCIA: bridge_thread no finalizó a tiempo.") # AGREGADO PARA DEBUGGING
        if self.reasoning_system:
            print("Cerrando: Esperando por reasoning_system...") # AGREGADO PARA DEBUGGING
            self.reasoning_system.join(timeout=2.0)
            if self.reasoning_system.is_alive():
                print("ADVERTENCIA: reasoning_system no finalizó a tiempo.") # AGREGADO PARA DEBUGGING


        # Shutdown thread pool executor
        print("Cerrando: Apagando ThreadPoolExecutor...") # AGREGADO PARA DEBUGGING
        self.executor.shutdown(wait=True)
        print("Cerrando: ThreadPoolExecutor apagado.") # AGREGADO PARA DEBUGGING

        print("All background processes stopped. Closing application.")
        event.accept() # Proceed with closing

    def create_filter_controls(self, module_name, num_channels, channel_names):
        """Crear controles de filtrado para un módulo específico"""
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_widget)
        
        # Scroll area para muchos canales
        scroll = QtWidgets.QScrollArea()
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_widget)
        
        filter_controls = []
        
        for i in range(num_channels):
            channel_group = QtWidgets.QGroupBox(f"Canal {channel_names[i]}")
            channel_layout = QtWidgets.QGridLayout(channel_group)
            
            # Tipo de filtro
            type_label = QtWidgets.QLabel("Tipo:")
            type_combo = QtWidgets.QComboBox()
            for key, value in FILTER_TYPES.items():
                type_combo.addItem(value, key)
            type_combo.setCurrentText("Sin filtro")
            
            # Método de filtro
            method_label = QtWidgets.QLabel("Método:")
            method_combo = QtWidgets.QComboBox()
            for key, value in FILTER_METHODS.items():
                method_combo.addItem(value, key)
            method_combo.setCurrentText("Butterworth")
            
            # Orden del filtro
            order_label = QtWidgets.QLabel("Orden:")
            order_spin = QtWidgets.QSpinBox()
            order_spin.setRange(1, 10)
            order_spin.setValue(4)
            
            # Frecuencia de corte baja
            low_freq_label = QtWidgets.QLabel("F. Baja (Hz):")
            low_freq_spin = QtWidgets.QDoubleSpinBox()
            low_freq_spin.setRange(0.1, 1000.0)
            low_freq_spin.setValue(1.0)
            low_freq_spin.setDecimals(1)
            
            # Frecuencia de corte alta
            high_freq_label = QtWidgets.QLabel("F. Alta (Hz):")
            high_freq_spin = QtWidgets.QDoubleSpinBox()
            high_freq_spin.setRange(1.0, 1000.0)
            high_freq_spin.setValue(100.0)
            high_freq_spin.setDecimals(1)
            
            # Frecuencia notch
            notch_freq_label = QtWidgets.QLabel("F. Notch (Hz):")
            notch_freq_spin = QtWidgets.QDoubleSpinBox()
            notch_freq_spin.setRange(1.0, 1000.0)
            notch_freq_spin.setValue(50.0)
            notch_freq_spin.setDecimals(1)
            
            # Factor Q para notch
            q_label = QtWidgets.QLabel("Factor Q:")
            q_spin = QtWidgets.QDoubleSpinBox()
            q_spin.setRange(1.0, 100.0)
            q_spin.setValue(30.0)
            q_spin.setDecimals(1)
            
            # Botón aplicar
            apply_btn = QPushButton("✅ Aplicar Filtro")
            apply_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            
            # Botón resetear
            reset_btn = QPushButton("🔄 Reset")
            reset_btn.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold;")
            
            # Layout de controles
            channel_layout.addWidget(type_label, 0, 0)
            channel_layout.addWidget(type_combo, 0, 1)
            channel_layout.addWidget(method_label, 0, 2)
            channel_layout.addWidget(method_combo, 0, 3)
            
            channel_layout.addWidget(order_label, 1, 0)
            channel_layout.addWidget(order_spin, 1, 1)
            channel_layout.addWidget(low_freq_label, 1, 2)
            channel_layout.addWidget(low_freq_spin, 1, 3)
            
            channel_layout.addWidget(high_freq_label, 2, 0)
            channel_layout.addWidget(high_freq_spin, 2, 1)
            channel_layout.addWidget(notch_freq_label, 2, 2)
            channel_layout.addWidget(notch_freq_spin, 2, 3)
            
            channel_layout.addWidget(q_label, 3, 0)
            channel_layout.addWidget(q_spin, 3, 1)
            channel_layout.addWidget(apply_btn, 3, 2)
            channel_layout.addWidget(reset_btn, 3, 3)
            
            # Conectar señales
            apply_btn.clicked.connect(lambda checked, mod=module_name, ch=i: self.apply_filter_to_channel(mod, ch))
            reset_btn.clicked.connect(lambda checked, mod=module_name, ch=i: self.reset_filter_channel(mod, ch))
            
            # Guardar referencias
            controls = {
                'type': type_combo,
                'method': method_combo,
                'order': order_spin,
                'low_freq': low_freq_spin,
                'high_freq': high_freq_spin,
                'notch_freq': notch_freq_spin,
                'q_factor': q_spin,
                'apply_btn': apply_btn,
                'reset_btn': reset_btn
            }
            
            filter_controls.append(controls)
            scroll_layout.addWidget(channel_group)
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        controls_layout.addWidget(scroll)
        
        # Guardar referencias para acceso posterior
        if module_name == "Fuerza":
            self.force_filter_ui = filter_controls
        elif module_name == "Vibración":
            self.vib_filter_ui = filter_controls
        elif module_name == "Puente":
            self.bridge_filter_ui = filter_controls
        
        return controls_widget
    
    def apply_filter_to_channel(self, module_name, channel_idx):
        """Aplicar filtro a un canal específico"""
        try:
            # Obtener controles según el módulo
            if module_name == "Fuerza":
                controls = self.force_filter_ui[channel_idx]
                filter_obj = self.force_filters[channel_idx]
            elif module_name == "Vibración":
                controls = self.vib_filter_ui[channel_idx]
                filter_obj = self.vib_filters[channel_idx]
            elif module_name == "Puente":
                controls = self.bridge_filter_ui[channel_idx]
                filter_obj = self.bridge_filters[channel_idx]
            else:
                return
            
            # Obtener parámetros de la UI
            filter_type = controls['type'].currentData()
            filter_method = controls['method'].currentData()
            order = controls['order'].value()
            low_freq = controls['low_freq'].value()
            high_freq = controls['high_freq'].value()
            notch_freq = controls['notch_freq'].value()
            q_factor = controls['q_factor'].value()
            
            # Diseñar y aplicar el filtro
            success = filter_obj.design_filter(
                filter_type=filter_type,
                filter_method=filter_method,
                order=order,
                cutoff_low=low_freq,
                cutoff_high=high_freq,
                notch_freq=notch_freq,
                notch_quality=q_factor
            )
            
            if success:
                print(f"✅ Filtro aplicado a {module_name} Canal {channel_idx}: {FILTER_TYPES[filter_type]}")
                controls['apply_btn'].setText("✅ Aplicado")
                controls['apply_btn'].setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
            else:
                print(f"❌ Error aplicando filtro a {module_name} Canal {channel_idx}")
                
        except Exception as e:
            print(f"Error en apply_filter_to_channel: {e}")
    
    def reset_filter_channel(self, module_name, channel_idx):
        """Resetear filtro de un canal específico"""
        try:
            # Obtener filtro según el módulo
            if module_name == "Fuerza":
                filter_obj = self.force_filters[channel_idx]
                controls = self.force_filter_ui[channel_idx]
            elif module_name == "Vibración":
                filter_obj = self.vib_filters[channel_idx]
                controls = self.vib_filter_ui[channel_idx]
            elif module_name == "Puente":
                filter_obj = self.bridge_filters[channel_idx]
                controls = self.bridge_filter_ui[channel_idx]
            else:
                return
            
            # Resetear filtro
            filter_obj.reset_filter()
            
            # Resetear UI
            controls['type'].setCurrentText("Sin filtro")
            controls['method'].setCurrentText("Butterworth")
            controls['order'].setValue(4)
            controls['low_freq'].setValue(1.0)
            controls['high_freq'].setValue(100.0)
            controls['notch_freq'].setValue(50.0)
            controls['q_factor'].setValue(30.0)
            
            controls['apply_btn'].setText("✅ Aplicar Filtro")
            controls['apply_btn'].setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            
            print(f"🔄 Filtro reseteado para {module_name} Canal {channel_idx}")
            
        except Exception as e:
            print(f"Error en reset_filter_channel: {e}")

    def actualizar_prediccion(self):
        """Manually trigger feature extraction and prediction."""
        if not self.acquiring:
            QtWidgets.QMessageBox.warning(self, "Detenido", "Inicie la adquisición para actualizar la predicción.")
            return
        print("Manual prediction update triggered.")
        self.run_prediction_and_reasoning()


# --- Main Execution ---
def main():
    # Set high DPI scaling for better look on modern displays (optional)
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    # Apply a style (optional)
    # app.setStyle('Fusion')
    ventana = PredictorHisteresis()
    # Instead of show(), use showMaximized() to fill the 2560x1080 screen
    ventana.showMaximized()
    # ventana.show() # <-- Replace this line
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

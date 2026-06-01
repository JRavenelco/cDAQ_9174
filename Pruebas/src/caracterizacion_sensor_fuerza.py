#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARACTERIZACIÓN DEL SENSOR DE FUERZA DYMH-105 CON SHAKER TIRA

Configuración:
- Shaker: TIRA TV 51144IN + Amplificador BAA 1000
- Generador: Keysight 33220A (manual)
- Celda de carga: Daysensor DYMH-105 (500 kg, 1.7 mV/V)
- Acondicionador: INA-4LC-8NTC (G=601)
- Acelerómetro: PCB 352C33 (100 mV/g)
- DAQ: NI cDAQ-9174 con NI 9205 (fuerza) + NI 9234 (aceleración)

Experimentos:
1. Barrido de frecuencia (sine sweep manual)
2. Excitación a frecuencias discretas
3. Cálculo de FRF y coherencia
4. Validación F = m*a
"""

import sys
import os
import time
import queue
import threading
import socket
import struct

# Fix Unicode emoji rendering on Windows terminals (cp1252)
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ─── Protocolo UDP compartido con Jetson ─────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_proto_dir = os.path.join(_here, "remote_monitoring")
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)
from cdaq_udp_protocol import (
    DATA_PORT, MAGIC, HDR_FMT, HDR_SIZE, pack_data,
)
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox,
    QTabWidget, QTextEdit, QComboBox, QCheckBox, QLineEdit, QFileDialog,
    QMessageBox, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime
from collections import deque
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
import pandas as pd

# Intentar importar nidaqmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    HAS_NIDAQMX = True
except ImportError:
    HAS_NIDAQMX = False
    print("⚠️ nidaqmx no disponible - Modo simulación")

# ============================================
# CONFIGURACIÓN DEL HARDWARE
# ============================================

# NI 9205 - Fuerza (celda de carga a través de INA-4LC)
FORCE_DEVICE = "cDAQ1Mod1"
FORCE_CHANNELS = ["ai0"]  # Canal de la celda de carga
FORCE_SAMPLE_RATE = 2500  # Hz
FORCE_MIN_V = -5.0
FORCE_MAX_V = 5.0
FORCE_TERMINAL = TerminalConfiguration.DIFF if HAS_NIDAQMX else None

# NI 9234 - Aceleración (1 CANAL: bancada)
ACCEL_DEVICE = "cDAQ1Mod2"
ACCEL_CHANNELS = ["ai0"]  # Solo acelerómetro en bancada
ACCEL_SAMPLE_RATE = 2000  # Hz

# SIN CALIBRACIÓN - Datos crudos en voltios
# La fuerza se guardará directamente en V
# La aceleración se guardará en g (conversión del NI 9234)
USE_RAW_DATA = True  # Sin conversiones
print("📊 Modo DATOS CRUDOS: Fuerza en V, Aceleración en g")

# Variables para compatibilidad con prints (no se usan para conversión)
CELDA_CAPACIDAD_KG = "N/A"
CELDA_SENSIBILIDAD_MV_V = "N/A"
INA849_GANANCIA = "N/A"

# Parámetros de los acelerómetros PCB 352C33
ACEL_SENSIBILIDAD_MV_G = 100.0  # mV/g (ambos sensores)
# ai0: Acelerómetro en BANCADA
# ai1: Acelerómetro en PIEZA

# Masa de prueba (ajustar según tu configuración)
MASA_PRUEBA_KG = 0.5  # kg

# Directorio de datos
DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
os.makedirs(DATOS_DIR, exist_ok=True)

# Colas de datos
force_queue = queue.Queue(maxsize=20)
accel_queue = queue.Queue(maxsize=20)  # Ahora contiene datos de 2 canales


# ============================================
# FUNCIONES DE CONVERSIÓN
# ============================================

def voltaje_a_fuerza_kg(V_medido):
    """
    MODO CRUDO: Devuelve voltaje directamente (V)
    """
    return V_medido  # Datos crudos - sin conversión

def voltaje_a_fuerza_N(V_medido):
    """MODO CRUDO: Devuelve voltaje directamente (V)"""
    return V_medido  # Datos crudos - sin conversión

def voltaje_a_aceleracion_g(V_medido):
    """
    El NI 9234 ya devuelve datos en g (IEPE calibrado)
    """
    return V_medido  # Ya viene en g del NI 9234

def voltaje_a_aceleracion_ms2(V_medido):
    """Devuelve en g (sin conversión a m/s²)"""
    return V_medido  # Ya viene en g


# ============================================
# HILOS DE ADQUISICIÓN
# ============================================

class ForceAcquisitionThread(threading.Thread):
    """Hilo para adquisición de fuerza (NI 9205)"""
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.task = None
        self.samples_per_read = 100
        
    def run(self):
        if not HAS_NIDAQMX:
            self._run_simulation()
            return
            
        try:
            self.task = nidaqmx.Task()
            for ch in FORCE_CHANNELS:
                self.task.ai_channels.add_ai_voltage_chan(
                    f"{FORCE_DEVICE}/{ch}",
                    terminal_config=FORCE_TERMINAL,
                    min_val=FORCE_MIN_V,
                    max_val=FORCE_MAX_V
                )
            self.task.timing.cfg_samp_clk_timing(
                rate=FORCE_SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples_per_read * 10
            )
            self.task.start()
            
            while self.running:
                try:
                    data = self.task.read(number_of_samples_per_channel=self.samples_per_read)
                    data_np = np.array(data).reshape(-1) if len(FORCE_CHANNELS) == 1 else np.array(data)
                    if not force_queue.full():
                        force_queue.put(data_np)
                except Exception as e:
                    if self.running:
                        print(f"Error lectura fuerza: {e}")
                    break
                    
        except Exception as e:
            print(f"Error inicializando fuerza: {e}")
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass
                    
    def _run_simulation(self):
        """Modo simulación sin hardware"""
        t = 0
        while self.running:
            # Simular señal de fuerza
            dt = self.samples_per_read / FORCE_SAMPLE_RATE
            t_arr = np.linspace(t, t + dt, self.samples_per_read)
            # Señal simulada: senoidal + ruido
            freq_sim = 40  # Hz
            data = 0.5 * np.sin(2 * np.pi * freq_sim * t_arr) + 0.02 * np.random.randn(self.samples_per_read)
            if not force_queue.full():
                force_queue.put(data)
            t += dt
            time.sleep(dt * 0.9)
            
    def stop(self):
        self.running = False


class AccelAcquisitionThread(threading.Thread):
    """
    Hilo para adquisición de aceleración (NI 9234) - 2 CANALES
    
    Canal 0 (ai0): Acelerómetro en bancada
    Canal 1 (ai1): Acelerómetro en pieza
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.task = None
        self.samples_per_read = 200
        self.n_channels = len(ACCEL_CHANNELS)
        
    def run(self):
        if not HAS_NIDAQMX:
            self._run_simulation()
            return
            
        try:
            self.task = nidaqmx.Task()
            for ch in ACCEL_CHANNELS:
                # NI 9234 con IEPE para cada canal
                self.task.ai_channels.add_ai_accel_chan(
                    f"{ACCEL_DEVICE}/{ch}",
                    sensitivity=ACEL_SENSIBILIDAD_MV_G,
                    min_val=-50.0,
                    max_val=50.0,
                    current_excit_val=0.004  # 4 mA IEPE
                )
            self.task.timing.cfg_samp_clk_timing(
                rate=ACCEL_SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples_per_read * 10
            )
            self.task.start()
            print(f"✅ NI 9234: {self.n_channels} canales de aceleración activos")
            
            while self.running:
                try:
                    data = self.task.read(number_of_samples_per_channel=self.samples_per_read)
                    # Con 2 canales, data es una lista de 2 arrays
                    data_np = np.array(data)  # Shape: (2, samples_per_read)
                    if not accel_queue.full():
                        accel_queue.put(data_np)
                except Exception as e:
                    if self.running:
                        print(f"Error lectura aceleración: {e}")
                    break
                    
        except Exception as e:
            print(f"Error inicializando aceleración IEPE: {e}")
            # Fallback a voltaje si falla IEPE
            self._run_voltage_mode()
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass
                    
    def _run_voltage_mode(self):
        """Modo voltaje como fallback para 2 canales"""
        try:
            self.task = nidaqmx.Task()
            for ch in ACCEL_CHANNELS:
                self.task.ai_channels.add_ai_voltage_chan(
                    f"{ACCEL_DEVICE}/{ch}",
                    min_val=-5.0,
                    max_val=5.0
                )
            self.task.timing.cfg_samp_clk_timing(
                rate=ACCEL_SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples_per_read * 10
            )
            self.task.start()
            print(f"⚠️ NI 9234: Modo voltaje (fallback) - {self.n_channels} canales")
            
            while self.running:
                try:
                    data = self.task.read(number_of_samples_per_channel=self.samples_per_read)
                    data_np = np.array(data)  # Shape: (2, samples_per_read)
                    # Convertir voltaje a g para ambos canales
                    data_g = voltaje_a_aceleracion_g(data_np)
                    if not accel_queue.full():
                        accel_queue.put(data_g)
                except:
                    break
        except Exception as e:
            print(f"Error modo voltaje: {e}")
            
    def _run_simulation(self):
        """Modo simulación con 2 canales"""
        t = 0
        while self.running:
            dt = self.samples_per_read / ACCEL_SAMPLE_RATE
            t_arr = np.linspace(t, t + dt, self.samples_per_read)
            freq_sim = 40
            
            # Canal 0: Acelerómetro bancada (original)
            phase0 = np.radians(15)
            acel0 = 0.3 * np.sin(2 * np.pi * freq_sim * t_arr + phase0) + 0.01 * np.random.randn(self.samples_per_read)
            
            # Canal 1: Acelerómetro en pieza (más cercano a la zona de corte)
            phase1 = np.radians(5)  # Menos desfase, más directo
            acel1 = 0.35 * np.sin(2 * np.pi * freq_sim * t_arr + phase1) + 0.01 * np.random.randn(self.samples_per_read)
            
            # Combinar en array 2D: (2, samples_per_read)
            data = np.vstack([acel0, acel1])
            
            if not accel_queue.full():
                accel_queue.put(data)
            t += dt
            time.sleep(dt * 0.9)
            
    def stop(self):
        self.running = False


# ============================================
# HILO DE ENVÍO UDP A JETSON
# ============================================

class UDPSenderThread(threading.Thread):
    """Hilo que envía bloques (fuerza + accel) vía UDP a la Jetson.

    Diseñado para no interferir con la captura en tiempo real:
    - send() es no-bloqueante (put_nowait, descarta si cola llena)
    - El socket UDP se crea en background; si la Jetson no responde,
      los paquetes simplemente se pierden sin afectar la adquisición.
    - Todo el trabajo pesado (pack_data, sendto) corre en este hilo,
      nunca en el hilo principal ni en los hilos de adquisición.
    """

    def __init__(self, jetson_ip: str, port: int = DATA_PORT):
        super().__init__(daemon=True)
        self.jetson_ip = jetson_ip
        self.port = port
        self.running = True
        self.queue = queue.Queue(maxsize=100)
        self.seq = 0
        self.sock = None
        self.bytes_sent = 0
        self.packets_sent = 0
        self.packets_dropped = 0
        self._sock_ready = threading.Event()

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(0.5)
            self._sock_ready.set()
        except Exception as e:
            print(f"[UDP] Error creando socket: {e}")
            return

        print(f"[UDP] Enviando a Jetson {self.jetson_ip}:{self.port}")

        while self.running:
            try:
                item = self.queue.get(timeout=0.3)
                if item is None:
                    break
                force_chunk, accel_chunk = item
                t_s = time.perf_counter()
                packet = pack_data(self.seq, t_s, FORCE_SAMPLE_RATE,
                                   np.asarray(force_chunk, dtype=np.float32),
                                   np.asarray(accel_chunk, dtype=np.float32))
                self.sock.sendto(packet, (self.jetson_ip, self.port))
                self.seq += 1
                self.bytes_sent += len(packet)
                self.packets_sent += 1
            except queue.Empty:
                continue
            except (OSError, socket.error) as e:
                # Jetson inalcanzable — descartar silenciosamente
                if self.running and self.packets_sent == 0:
                    print(f"[UDP] Jetson {self.jetson_ip}:{self.port} no alcanzable — paquetes se descartan")
                self.packets_dropped += 1
            except Exception as e:
                if self.running:
                    print(f"[UDP] Error: {e}")
                self.packets_dropped += 1

    def send(self, force_chunk, accel_chunk):
        """Encola un bloque para envío. No bloquea nunca: si la cola está
        llena, descarta el paquete para no frenar la adquisición."""
        if not self.running:
            return
        try:
            self.queue.put_nowait((force_chunk, accel_chunk))
        except queue.Full:
            self.packets_dropped += 1

    def stop(self):
        self.running = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        print(f"[UDP] Detenido — {self.packets_sent} enviados, "
              f"{self.packets_dropped} descartados, {self.bytes_sent/1024:.1f} kB")


# ============================================
# INTERFAZ PRINCIPAL
# ============================================

class CaracterizacionFuerzaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 Caracterización Sensor de Fuerza DYMH-105")
        self.setGeometry(100, 100, 1600, 900)
        
        # Estado
        self.acquiring = False
        self.force_thread = None
        self.accel_thread = None
        
        # Buffers de datos
        self.buffer_size = int(FORCE_SAMPLE_RATE * 2)  # 2 segundos
        self.force_buffer = deque(maxlen=self.buffer_size)
        self.accel_buffer_0 = deque(maxlen=self.buffer_size)  # ai0: bancada
        self.accel_buffer_1 = deque(maxlen=self.buffer_size)  # ai1: pieza
        
        # Datos acumulados para análisis
        self.all_force_data = []
        self.all_accel_data_0 = []  # Bancada
        self.all_accel_data_1 = []  # Pieza
        
        # Resultados de experimentos
        self.experimentos = []
        
        # Modo de experimento (SWEEP, TRIANGLE o CUTTING)
        self.experiment_mode = "SWEEP"
        self.use_single_accel = True   # Solo canal ai0 activo  # True = solo usar canal 0 (modo corte)
        
        # Buffers para análisis de fricción (desplazamiento integrado)
        self.displacement_buffer = deque(maxlen=self.buffer_size)
        
        # Timer de actualización
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)

        # UDP a Jetson
        self.udp_sender = None

        # Aplicar tema oscuro
        self.apply_dark_theme()
        
        # Configurar UI
        self._setup_ui()
        
    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        self.setPalette(palette)
        pg.setConfigOption('background', '#2b2b2b')
        pg.setConfigOption('foreground', 'w')
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # === PANEL SUPERIOR: Configuración ===
        config_group = QGroupBox("⚙️ Configuración del Experimento")
        config_layout = QGridLayout(config_group)
        
        # Masa de prueba
        config_layout.addWidget(QLabel("Masa de prueba (kg):"), 0, 0)
        self.masa_spin = QDoubleSpinBox()
        self.masa_spin.setRange(0.01, 100)
        self.masa_spin.setValue(MASA_PRUEBA_KG)
        self.masa_spin.setDecimals(3)
        config_layout.addWidget(self.masa_spin, 0, 1)
        
        # Frecuencia actual (manual del Keysight)
        config_layout.addWidget(QLabel("Frecuencia excitación (Hz):"), 0, 2)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.1, 1000)  # Permitir desde 0.1 Hz para fricción
        self.freq_spin.setValue(40)
        self.freq_spin.setDecimals(2)  # 2 decimales para frecuencias bajas
        config_layout.addWidget(self.freq_spin, 0, 3)
        
        # (Duración removida - grabación continua)
        self.is_recording = False  # Flag de grabación
        
        # Tipo de experimento
        config_layout.addWidget(QLabel("Tipo Experimento:"), 1, 0)
        self.exp_type_combo = QComboBox()
        self.exp_type_combo.addItems(["🔄 Barrido/Chirp (Inercia)", "📐 Triangular (Fricción)", "🔪 Fuerza de Corte (Histéresis)", "📡 Solo Acelerómetro"])
        self.exp_type_combo.currentIndexChanged.connect(self._on_exp_type_changed)
        config_layout.addWidget(self.exp_type_combo, 1, 1)
        
        # Velocidad objetivo (solo para modo triangular)
        config_layout.addWidget(QLabel("Velocidad objetivo (mm/s):"), 1, 2)
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(0.1, 1000)
        self.velocity_spin.setValue(10.0)
        self.velocity_spin.setDecimals(1)
        self.velocity_spin.setEnabled(False)  # Deshabilitado por defecto
        config_layout.addWidget(self.velocity_spin, 1, 3)
        
        # Parámetros de corte (solo para modo corte)
        config_layout.addWidget(QLabel("RPM Husillo:"), 2, 0)
        self.rpm_husillo_spin = QSpinBox()
        self.rpm_husillo_spin.setRange(0, 10000)
        self.rpm_husillo_spin.setValue(100)
        self.rpm_husillo_spin.setEnabled(False)
        config_layout.addWidget(self.rpm_husillo_spin, 2, 1)
        
        config_layout.addWidget(QLabel("RPM Avance X:"), 2, 2)
        self.rpm_avance_spin = QSpinBox()
        self.rpm_avance_spin.setRange(0, 1000)
        self.rpm_avance_spin.setValue(20)
        self.rpm_avance_spin.setEnabled(False)
        config_layout.addWidget(self.rpm_avance_spin, 2, 3)

        # Condición del cortador
        config_layout.addWidget(QLabel("🔪 Condición cortador:"), 2, 4)
        self.cutter_condition_combo = QComboBox()
        self.cutter_condition_combo.addItems(["Nuevo", "Medio uso", "Desgastado"])
        self.cutter_condition_combo.setToolTip("Estado del cortador: afecta la histéresis y el desgaste")
        self.cutter_condition_combo.setEnabled(False)  # Solo en modo corte
        config_layout.addWidget(self.cutter_condition_combo, 2, 5)

        # Notas
        config_layout.addWidget(QLabel("Notas:"), 1, 4)
        self.notas_edit = QLineEdit()
        self.notas_edit.setPlaceholderText("Descripción del experimento...")
        config_layout.addWidget(self.notas_edit, 1, 5)

        # ── Jetson UDP ──
        config_layout.addWidget(QLabel("🛰 Jetson UDP:"), 3, 0)
        self.udp_enabled_cb = QCheckBox("Enviar a Jetson")
        self.udp_enabled_cb.setToolTip("Activar envio UDP a la Jetson para inferencia remota")
        config_layout.addWidget(self.udp_enabled_cb, 3, 1)

        config_layout.addWidget(QLabel("IP Jetson:"), 3, 2)
        self.jetson_ip_edit = QLineEdit()
        self.jetson_ip_edit.setPlaceholderText("192.168.137.164")
        self.jetson_ip_edit.setText("192.168.137.164")
        config_layout.addWidget(self.jetson_ip_edit, 3, 3)

        config_layout.addWidget(QLabel("Puerto:"), 3, 4)
        self.udp_port_spin = QSpinBox()
        self.udp_port_spin.setRange(1024, 65535)
        self.udp_port_spin.setValue(DATA_PORT)
        config_layout.addWidget(self.udp_port_spin, 3, 5)

        main_layout.addWidget(config_group)

        # === PANEL DE FILTRADO ===
        filter_group = QGroupBox("🔧 Filtrado de Señales")
        filter_layout = QHBoxLayout(filter_group)
        
        # Checkbox para activar filtrado
        self.filter_enabled_cb = QCheckBox("Activar filtro")
        self.filter_enabled_cb.setChecked(False)
        self.filter_enabled_cb.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_enabled_cb)
        
        # Tipo de filtro
        filter_layout.addWidget(QLabel("Tipo:"))
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Savitzky-Golay", "Butterworth LP", "Media Móvil", "Mediana (spikes)", "Hampel (outliers)", "Auto (freq)", "Notch (elimina freq)"])
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_type_combo)
        
        # Parámetro de ventana/orden
        filter_layout.addWidget(QLabel("Ventana:"))
        self.filter_window_spin = QSpinBox()
        self.filter_window_spin.setRange(5, 201)
        self.filter_window_spin.setValue(21)
        self.filter_window_spin.setSingleStep(2)  # Solo impares
        self.filter_window_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_window_spin)
        
        # Orden del polinomio (Savitzky-Golay)
        filter_layout.addWidget(QLabel("Orden:"))
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 7)
        self.filter_order_spin.setValue(3)
        self.filter_order_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_order_spin)
        
        # Frecuencia de corte (Butterworth/Notch)
        filter_layout.addWidget(QLabel("Fc (Hz):"))
        self.filter_fc_spin = QDoubleSpinBox()
        self.filter_fc_spin.setRange(1, 1000)
        self.filter_fc_spin.setValue(50)  # 50 Hz por defecto (ruido ventilador)
        self.filter_fc_spin.setSingleStep(1)
        self.filter_fc_spin.setDecimals(1)
        self.filter_fc_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_fc_spin)
        
        # Indicador de reducción de ruido
        self.filter_snr_label = QLabel("SNR: --")
        self.filter_snr_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        filter_layout.addWidget(self.filter_snr_label)
        
        filter_layout.addStretch()
        main_layout.addWidget(filter_group)
        
        # === PANEL DE INFORMACIÓN EN TIEMPO REAL ===
        info_group = QGroupBox("📊 Mediciones en Tiempo Real")
        info_layout = QGridLayout(info_group)
        
        # Fuerza
        info_layout.addWidget(QLabel("⚡ FUERZA:"), 0, 0)
        self.force_volt_label = QLabel("Voltaje: 0.000 V")
        self.force_kg_label = QLabel("Fuerza: 0.000 kg")
        self.force_N_label = QLabel("Fuerza: 0.000 N")
        self.force_rms_label = QLabel("RMS: 0.000 V")
        info_layout.addWidget(self.force_volt_label, 0, 1)
        info_layout.addWidget(self.force_kg_label, 0, 2)
        info_layout.addWidget(self.force_N_label, 0, 3)
        info_layout.addWidget(self.force_rms_label, 0, 4)
        
        # Aceleración Canal 0 (Bancada)
        info_layout.addWidget(QLabel("📡 ACEL BANCADA (ai0):"), 1, 0)
        self.accel0_g_label = QLabel("Acel: 0.000 g")
        self.accel0_rms_label = QLabel("RMS: 0.000 g")
        info_layout.addWidget(self.accel0_g_label, 1, 1)
        info_layout.addWidget(self.accel0_rms_label, 1, 2)
        
        # Aceleración Canal 1 (Sobre sensor de fuerza)
        info_layout.addWidget(QLabel("📡 ACEL PIEZA (ai1):"), 2, 0)
        self.accel1_g_label = QLabel("Acel: 0.000 g")
        self.accel1_rms_label = QLabel("RMS: 0.000 g")
        info_layout.addWidget(self.accel1_g_label, 2, 1)
        info_layout.addWidget(self.accel1_rms_label, 2, 2)
        
        # Validación F=ma (usando acelerómetro sobre sensor)
        info_layout.addWidget(QLabel("🔄 VALIDACIÓN F=ma:"), 3, 0)
        self.fma_force_label = QLabel("F medida: 0.000 N")
        self.fma_calc_label = QLabel("m×a: 0.000 N")
        self.fma_error_label = QLabel("Error: 0.0 %")
        self.fma_error_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.fma_force_label, 3, 1)
        info_layout.addWidget(self.fma_calc_label, 3, 2)
        info_layout.addWidget(self.fma_error_label, 3, 3)
        
        # Transmisibilidad (relación entre acelerómetros)
        info_layout.addWidget(QLabel("📈 TRANSMISIBILIDAD:"), 3, 4)
        self.transmisibilidad_label = QLabel("T = 0.000")
        self.transmisibilidad_label.setStyleSheet("font-weight: bold; color: #2ecc71;")
        info_layout.addWidget(self.transmisibilidad_label, 3, 5)
        
        main_layout.addWidget(info_group)
        
        # === TABS PRINCIPALES ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background: #3a3a3a;
                color: white;
                padding: 8px 20px;
                font-weight: bold;
            }
            QTabBar::tab:selected { background: #2980b9; }
        """)
        
        # Tab 1: Señales en tiempo
        self._setup_time_tab()
        
        # Tab 2: FFT y FRF
        self._setup_fft_tab()
        
        # Tab 3: Lissajous / Fase
        self._setup_phase_tab()
        
        # Tab 4: Análisis de Fricción
        self._setup_friction_tab()
        
        # Tab 5: Resultados
        self._setup_results_tab()
        
        main_layout.addWidget(self.tabs)
        
        # === BOTONES DE CONTROL ===
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶️ Iniciar Adquisición")
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.start_btn.clicked.connect(self.toggle_acquisition)
        btn_layout.addWidget(self.start_btn)
        
        self.capture_btn = QPushButton("🔴 Iniciar Grabación")
        self.capture_btn.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.capture_btn.clicked.connect(self.toggle_recording)
        self.capture_btn.setEnabled(False)
        btn_layout.addWidget(self.capture_btn)
        
        self.autoset_btn = QPushButton("🎯 Auto-Set Y")
        self.autoset_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold; padding: 10px;")
        self.autoset_btn.clicked.connect(self.autoset_y)
        btn_layout.addWidget(self.autoset_btn)
        
        self.save_btn = QPushButton("💾 Guardar Resultados")
        self.save_btn.setStyleSheet("background-color: #e67e22; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.save_btn.clicked.connect(self.save_results)
        btn_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(btn_layout)
        
        # Status bar
        self.status_label = QLabel("Listo - Configure el Keysight y presione Iniciar")
        self.statusBar().addWidget(self.status_label)
        
    def _setup_time_tab(self):
        """Tab de señales en tiempo"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        splitter = QSplitter(Qt.Vertical)
        
        # Gráfica de fuerza
        self.force_plot = pg.PlotWidget(title="⚡ Fuerza (Voltaje)")
        self.force_plot.setLabel('left', 'Voltaje', 'V')
        self.force_plot.setLabel('bottom', 'Tiempo', 's')
        self.force_plot.showGrid(x=True, y=True, alpha=0.3)
        self.force_curve = self.force_plot.plot(pen=pg.mkPen('#3498db', width=2))
        splitter.addWidget(self.force_plot)
        
        # Gráfica de aceleración (2 canales)
        self.accel_plot = pg.PlotWidget(title="📡 Aceleración (2 sensores)")
        self.accel_plot.setLabel('left', 'Aceleración', 'g')
        self.accel_plot.setLabel('bottom', 'Tiempo', 's')
        self.accel_plot.showGrid(x=True, y=True, alpha=0.3)
        self.accel_curve = self.accel_plot.plot(pen=pg.mkPen('#e74c3c', width=2), name='Bancada (ai0)')
        self.accel_curve_1 = self.accel_plot.plot(pen=pg.mkPen('#f39c12', width=2), name='Pieza (ai1)')
        self.accel_plot.addLegend()
        splitter.addWidget(self.accel_plot)
        
        # Gráfica superpuesta (normalizada)
        self.overlay_plot = pg.PlotWidget(title="🔀 Superposición (Normalizada)")
        self.overlay_plot.setLabel('left', 'Amplitud', 'norm')
        self.overlay_plot.setLabel('bottom', 'Tiempo', 's')
        self.overlay_plot.showGrid(x=True, y=True, alpha=0.3)
        self.overlay_force_curve = self.overlay_plot.plot(pen=pg.mkPen('#3498db', width=2), name='Fuerza')
        self.overlay_accel_curve = self.overlay_plot.plot(pen=pg.mkPen('#e74c3c', width=2), name='Aceleración')
        self.overlay_plot.addLegend()
        splitter.addWidget(self.overlay_plot)
        
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "📈 Señales")
        
    def _setup_fft_tab(self):
        """Tab de FFT y FRF"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        splitter = QSplitter(Qt.Vertical)
        
        # FFT Fuerza
        self.fft_force_plot = pg.PlotWidget(title="FFT Fuerza")
        self.fft_force_plot.setLabel('left', 'Magnitud', 'dB')
        self.fft_force_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.fft_force_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_force_curve = self.fft_force_plot.plot(pen=pg.mkPen('#3498db', width=2))
        # Etiqueta de frecuencia pico
        self.fft_force_label = pg.TextItem(anchor=(0, 1), color='#3498db')
        self.fft_force_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.fft_force_plot.addItem(self.fft_force_label)
        splitter.addWidget(self.fft_force_plot)
        
        # FFT Aceleración (solo ai0)
        self.fft_accel_plot = pg.PlotWidget(title=f"FFT Aceleración ai0 (N=4096, Fs={ACCEL_SAMPLE_RATE} Hz)")
        self.fft_accel_plot.setLabel('left', 'Amplitud', 'g')
        self.fft_accel_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.fft_accel_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_accel_plot.setXRange(0, ACCEL_SAMPLE_RATE / 2)
        self.fft_accel_plot.enableAutoRange(axis='y')
        self.fft_accel_curve = self.fft_accel_plot.plot(pen=pg.mkPen('#00FF88', width=1.2))
        self.fft_accel_curve_1 = self.fft_accel_plot.plot(pen=pg.mkPen('#f39c12', width=2))  # unused, kept for compat
        # Línea vertical de pico
        self.fft_accel_peak_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.DashLine))
        self.fft_accel_plot.addItem(self.fft_accel_peak_line)
        # Etiqueta de frecuencia pico
        self.fft_accel_label_0 = pg.TextItem(anchor=(0, 1), color='#FF4444')
        self.fft_accel_label_0.setFont(QFont('Arial', 12, QFont.Bold))
        self.fft_accel_plot.addItem(self.fft_accel_label_0)
        self.fft_accel_label_1 = pg.TextItem(anchor=(0, 0), color='#f39c12')  # kept for compat
        self.fft_accel_plot.addItem(self.fft_accel_label_1)
        splitter.addWidget(self.fft_accel_plot)
        
        # FRF (Aceleración Sensor / Fuerza)
        self.frf_plot = pg.PlotWidget(title="FRF: H(f) = Acel_Sensor / Fuerza")
        self.frf_plot.setLabel('left', '|H(f)|', 'dB')
        self.frf_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.frf_plot.showGrid(x=True, y=True, alpha=0.3)
        self.frf_curve = self.frf_plot.plot(pen=pg.mkPen('#2ecc71', width=2))
        splitter.addWidget(self.frf_plot)
        
        # Transmisibilidad (Acel_Sensor / Acel_Bancada)
        self.trans_plot = pg.PlotWidget(title="Transmisibilidad: T(f) = Acel_Sensor / Acel_Bancada")
        self.trans_plot.setLabel('left', '|T(f)|', 'dB')
        self.trans_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.trans_plot.showGrid(x=True, y=True, alpha=0.3)
        self.transmisibilidad_curve = self.trans_plot.plot(pen=pg.mkPen('#9b59b6', width=2))
        splitter.addWidget(self.trans_plot)
        
        layout.addWidget(splitter)
        self.tabs.addTab(tab, "📊 FFT / FRF")
        
    def _setup_phase_tab(self):
        """Tab de diagrama de fase"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Lissajous 2D
        self.lissajous_plot = pg.PlotWidget(title="Diagrama de Lissajous (Fuerza vs Aceleración)")
        self.lissajous_plot.setLabel('left', 'Aceleración', 'g')
        self.lissajous_plot.setLabel('bottom', 'Fuerza', 'V')
        self.lissajous_plot.showGrid(x=True, y=True, alpha=0.3)
        self.lissajous_curve = self.lissajous_plot.plot(pen=pg.mkPen('#9b59b6', width=2))
        layout.addWidget(self.lissajous_plot)
        
        # Panel de información de fase
        phase_info = QGroupBox("📐 Análisis de Fase")
        phase_layout = QVBoxLayout(phase_info)
        
        self.phase_freq_label = QLabel("Frecuencia dominante: -- Hz")
        self.phase_angle_label = QLabel("Desfase: -- °")
        self.phase_delay_label = QLabel("Retardo: -- ms")
        self.phase_coherence_label = QLabel("Coherencia: -- ")
        
        for lbl in [self.phase_freq_label, self.phase_angle_label, 
                    self.phase_delay_label, self.phase_coherence_label]:
            lbl.setFont(QFont("Consolas", 12))
            phase_layout.addWidget(lbl)
            
        phase_layout.addStretch()
        layout.addWidget(phase_info)
        
        self.tabs.addTab(tab, "🔄 Fase")
        
    def _setup_friction_tab(self):
        """Tab de análisis de fricción (modo triangular)"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Gráfica principal: Fuerza vs Desplazamiento (histéresis)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.friction_plot = pg.PlotWidget(title="🧩 Histéresis: Fuerza vs Desplazamiento")
        self.friction_plot.setLabel('left', 'Fuerza', 'N')
        self.friction_plot.setLabel('bottom', 'Desplazamiento', 'mm')
        self.friction_plot.showGrid(x=True, y=True, alpha=0.3)
        self.friction_curve = self.friction_plot.plot(pen=pg.mkPen('#e74c3c', width=2))
        # Línea de referencia horizontal (F=0)
        self.friction_zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#666', width=1, style=Qt.DashLine))
        self.friction_plot.addItem(self.friction_zero_line)
        left_layout.addWidget(self.friction_plot)
        
        # Gráfica secundaria: Velocidad vs Tiempo
        self.velocity_plot = pg.PlotWidget(title="Velocidad Estimada (integración de aceleración)")
        self.velocity_plot.setLabel('left', 'Velocidad', 'mm/s')
        self.velocity_plot.setLabel('bottom', 'Tiempo', 's')
        self.velocity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.velocity_curve = self.velocity_plot.plot(pen=pg.mkPen('#3498db', width=2))
        left_layout.addWidget(self.velocity_plot)
        
        layout.addWidget(left_widget, stretch=3)
        
        # Panel derecho: Información de fricción
        right_widget = QGroupBox("📊 Análisis de Fricción")
        right_layout = QVBoxLayout(right_widget)
        
        # Métricas de fricción - Método Pico a Pico
        self.friction_fmax_label = QLabel("F_max: -- N")
        self.friction_fmin_label = QLabel("F_min: -- N")
        self.friction_fdyn_label = QLabel("F_fricción dinámica: -- N")
        self.friction_fdyn_label.setStyleSheet("font-weight: bold; color: #e74c3c; font-size: 14px;")
        
        right_layout.addWidget(QLabel("─── Método Cruce por Cero ───"))
        self.friction_fforward_label = QLabel("F_forward (x≈0): -- N")
        self.friction_fforward_label.setStyleSheet("color: #2ecc71;")
        self.friction_fbackward_label = QLabel("F_backward (x≈0): -- N")
        self.friction_fbackward_label.setStyleSheet("color: #3498db;")
        
        right_layout.addWidget(QLabel("─── Desplazamiento ───"))
        self.friction_xmax_label = QLabel("x_max: -- mm")
        self.friction_xmin_label = QLabel("x_min: -- mm")
        self.friction_stroke_label = QLabel("Carrera: -- mm")
        self.friction_velocity_label = QLabel("Velocidad media: -- mm/s")
        self.friction_energy_label = QLabel("Energía disipada: -- mJ")
        
        for lbl in [self.friction_fmax_label, self.friction_fmin_label, 
                    self.friction_fdyn_label, self.friction_fforward_label,
                    self.friction_fbackward_label, self.friction_xmax_label,
                    self.friction_xmin_label, self.friction_stroke_label,
                    self.friction_velocity_label, self.friction_energy_label]:
            lbl.setFont(QFont("Consolas", 10))
            right_layout.addWidget(lbl)
        
        right_layout.addSpacing(10)
        
        # Instrucciones para modo fricción
        instructions = QLabel(
            "📋 MÉTODO ZERO-CROSSING:\n"
            "• Encuentra puntos donde x ≈ 0\n"
            "• Separa por dirección (v>0 / v<0)\n"
            "• F_friction = (F_fwd - F_bwd) / 2\n"
            "• Más preciso que pico a pico\n"
            "\n⚙️ CONFIGURACIÓN:\n"
            "• Generador: Onda TRIANGULAR\n"
            "• Frecuencia: 0.5 - 5 Hz"
        )
        instructions.setStyleSheet("color: #888; font-size: 9px;")
        instructions.setWordWrap(True)
        right_layout.addWidget(instructions)
        
        right_layout.addStretch()
        layout.addWidget(right_widget, stretch=1)
        
        self.tabs.addTab(tab, "🧩 Fricción")
        
    def _on_exp_type_changed(self, index):
        """Callback cuando cambia el tipo de experimento"""
        if index == 0:  # Barrido/Chirp
            self.experiment_mode = "SWEEP"
            self.velocity_spin.setEnabled(False)
            self.rpm_husillo_spin.setEnabled(False)
            self.rpm_avance_spin.setEnabled(False)
            self.cutter_condition_combo.setEnabled(False)
            self.use_single_accel = True   # Solo canal ai0 activo
            self.status_label.setText("Modo: Barrido de frecuencia (identificacion de inercia)")
        elif index == 1:  # Triangular
            self.experiment_mode = "TRIANGLE"
            self.velocity_spin.setEnabled(True)
            self.rpm_husillo_spin.setEnabled(False)
            self.rpm_avance_spin.setEnabled(False)
            self.cutter_condition_combo.setEnabled(False)
            self.use_single_accel = True   # Solo canal ai0 activo
            self.status_label.setText("Modo: Onda triangular (caracterizacion de friccion)")
        elif index == 2:  # Fuerza de Corte
            self.experiment_mode = "CUTTING"
            self.velocity_spin.setEnabled(False)
            self.rpm_husillo_spin.setEnabled(True)
            self.rpm_avance_spin.setEnabled(True)
            self.cutter_condition_combo.setEnabled(True)
            self.use_single_accel = True
            self.status_label.setText("🔪 Modo: FUERZA DE CORTE - usando solo acelerometro ai0 + fuerza ai0")
        else:  # Solo Acelerómetro
            self.experiment_mode = "ACCEL_ONLY"
            self.velocity_spin.setEnabled(False)
            self.rpm_husillo_spin.setEnabled(True)
            self.rpm_avance_spin.setEnabled(False)
            self.cutter_condition_combo.setEnabled(False)
            self.use_single_accel = True
            self.status_label.setText("📡 Modo: SOLO ACELERÓMETRO - captura vibración sin sensor de fuerza")

    def _on_filter_changed(self):
        """Callback cuando cambian los parámetros del filtro"""
        filter_type = self.filter_type_combo.currentIndex()
        
        # Habilitar/deshabilitar controles según el tipo
        # 0=SG, 1=Butter, 2=Media, 3=Mediana, 4=Hampel, 5=Auto, 6=Notch
        self.filter_window_spin.setEnabled(filter_type in [0, 2, 3, 4])  # SG, Media, Mediana, Hampel
        self.filter_order_spin.setEnabled(filter_type in [0, 4])  # SG y Hampel (threshold)
        self.filter_fc_spin.setEnabled(filter_type in [1, 6])  # Butterworth y Notch
        
        # Ajustar label de orden para Hampel
        if filter_type == 4:
            # Para Hampel, el "orden" será el threshold en sigmas
            self.filter_order_spin.setRange(1, 10)
            self.filter_order_spin.setValue(3)  # 3 sigma es típico
        else:
            self.filter_order_spin.setRange(1, 7)
        
        # Modo Auto: calcular parámetros basados en frecuencia de excitación
        if filter_type == 5:  # Auto
            freq_exc = self.freq_spin.value()
            # Ventana óptima: ~10 puntos por período
            samples_per_period = FORCE_SAMPLE_RATE / freq_exc
            window = int(samples_per_period / 5)
            window = window if window % 2 == 1 else window + 1  # Asegurar impar
            window = max(5, min(window, 101))  # Limitar rango
            self.filter_window_spin.setValue(window)
            self.filter_fc_spin.setValue(freq_exc * 3)  # fc = 3x frecuencia fundamental
    
    def apply_filter(self, data):
        """
        Aplica el filtro seleccionado a los datos.
        Preserva la señal pero reduce el ruido de alta frecuencia.
        
        Args:
            data: numpy array de datos a filtrar
            
        Returns:
            tuple: (datos_filtrados, snr_mejora_db)
        """
        if not self.filter_enabled_cb.isChecked() or len(data) < 10:
            return data, 0.0
        
        filter_type = self.filter_type_combo.currentIndex()
        # 0=SG, 1=Butter, 2=Media, 3=Mediana, 4=Hampel, 5=Auto
        
        try:
            if filter_type == 0 or filter_type == 5:  # Savitzky-Golay o Auto
                window = self.filter_window_spin.value()
                order = self.filter_order_spin.value()
                # Asegurar que window sea impar y mayor que order
                window = window if window % 2 == 1 else window + 1
                window = max(order + 2, window)
                if window > len(data):
                    window = len(data) if len(data) % 2 == 1 else len(data) - 1
                filtered = savgol_filter(data, window, order)
                
            elif filter_type == 1:  # Butterworth paso bajo
                fc = self.filter_fc_spin.value()
                nyquist = FORCE_SAMPLE_RATE / 2
                if fc >= nyquist:
                    fc = nyquist * 0.9
                b, a = signal.butter(4, fc / nyquist, btype='low')
                filtered = signal.filtfilt(b, a, data)
                
            elif filter_type == 2:  # Media móvil
                window = self.filter_window_spin.value()
                kernel = np.ones(window) / window
                filtered = np.convolve(data, kernel, mode='same')
            
            elif filter_type == 3:  # Filtro de MEDIANA - excelente para spikes
                window = self.filter_window_spin.value()
                window = window if window % 2 == 1 else window + 1
                filtered = signal.medfilt(data, kernel_size=window)
            
            elif filter_type == 4:  # Filtro HAMPEL - detecta y reemplaza outliers
                window = self.filter_window_spin.value()
                threshold = self.filter_order_spin.value()  # En sigmas
                filtered = self._hampel_filter(data, window, threshold)
            
            elif filter_type == 6:  # Filtro NOTCH - elimina frecuencia específica
                fc = self.filter_fc_spin.value()  # Frecuencia a eliminar
                Q = 5  # Factor de calidad bajo = filtro más ancho (~fc/Q Hz de ancho)
                nyquist = FORCE_SAMPLE_RATE / 2
                if fc < nyquist:
                    # Aplicar notch y sus armónicos (50, 100, 150 Hz)
                    filtered = data.copy()
                    for harmonic in [1, 2, 3]:  # Fundamental + 2 armónicos
                        freq = fc * harmonic
                        if freq < nyquist:
                            b, a = signal.iirnotch(freq, Q, FORCE_SAMPLE_RATE)
                            filtered = signal.filtfilt(b, a, filtered)
                else:
                    filtered = data
                
            else:
                filtered = data
            
            # Calcular mejora en SNR (aproximada)
            noise_before = np.std(np.diff(data))
            noise_after = np.std(np.diff(filtered))
            if noise_after > 1e-10:
                snr_improvement = 20 * np.log10(noise_before / noise_after)
            else:
                snr_improvement = 0
                
            return filtered, snr_improvement
            
        except Exception as e:
            print(f"Error en filtrado: {e}")
            return data, 0.0
    
    def _hampel_filter(self, data, window_size=11, n_sigma=3):
        """
        Filtro Hampel: detecta outliers y los reemplaza por la mediana local.
        Excelente para eliminar spikes sin afectar la señal.
        
        Args:
            data: señal de entrada
            window_size: tamaño de ventana (impar)
            n_sigma: número de desviaciones estándar para considerar outlier
            
        Returns:
            señal filtrada
        """
        filtered = data.copy()
        n = len(data)
        k = window_size // 2
        
        for i in range(k, n - k):
            # Ventana local
            window = data[i - k:i + k + 1]
            
            # Mediana y MAD (Median Absolute Deviation)
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            
            # MAD a desviación estándar (factor 1.4826 para distribución normal)
            sigma = 1.4826 * mad
            
            # Si el punto está fuera de n_sigma, reemplazar por mediana
            if sigma > 1e-10:
                if np.abs(data[i] - median) > n_sigma * sigma:
                    filtered[i] = median
        
        return filtered
        
    def _setup_results_tab(self):
        """Tab de resultados acumulados"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Tabla de experimentos
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Frecuencia (Hz)", "Masa (kg)", "F_rms (N)", "a_rms (m/s²)",
            "m×a (N)", "Error F=ma (%)", "Fase (°)", "Notas"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.results_table)
        
        # Resumen
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Consolas", 10))
        self.summary_text.setMaximumHeight(150)
        layout.addWidget(self.summary_text)
        
        self.tabs.addTab(tab, "📋 Resultados")
        
    def toggle_acquisition(self):
        if not self.acquiring:
            self.start_acquisition()
        else:
            self.stop_acquisition()
            
    def start_acquisition(self):
        # Limpiar buffers (2 acelerómetros)
        self.force_buffer.clear()
        self.accel_buffer_0.clear()
        self.accel_buffer_1.clear()
        self.displacement_buffer.clear()
        self.all_force_data.clear()
        self.all_accel_data_0.clear()
        self.all_accel_data_1.clear()
        
        # Vaciar colas
        while not force_queue.empty():
            force_queue.get()
        while not accel_queue.empty():
            accel_queue.get()
            
        # Iniciar hilos
        if self.experiment_mode != "ACCEL_ONLY":
            self.force_thread = ForceAcquisitionThread()
            self.force_thread.start()
        else:
            self.force_thread = None
        self.accel_thread = AccelAcquisitionThread()
        self.accel_thread.start()
        
        # Iniciar timer
        self.update_timer.start(50)  # 20 Hz

        # Iniciar UDP a Jetson si está habilitado
        if self.udp_enabled_cb.isChecked():
            jetson_ip = self.jetson_ip_edit.text().strip() or "192.168.137.164"
            udp_port = self.udp_port_spin.value()
            self.udp_sender = UDPSenderThread(jetson_ip, udp_port)
            self.udp_sender.start()
            status_msg = f"⏺️ Adquiriendo + UDP → {jetson_ip}:{udp_port}..."
        else:
            status_msg = "⏺️ Adquiriendo..."

        self.acquiring = True
        self.start_btn.setText("⏹️ Detener Adquisición")
        self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.capture_btn.setEnabled(True)
        self.status_label.setText(status_msg)

    def stop_acquisition(self):
        self.acquiring = False
        self.update_timer.stop()
        
        if self.force_thread:
            self.force_thread.stop()
            self.force_thread.join(timeout=2)
        if self.accel_thread:
            self.accel_thread.stop()
            self.accel_thread.join(timeout=2)

        # Detener UDP sender
        if self.udp_sender:
            self.udp_sender.stop()
            self.udp_sender.join(timeout=2)
            self.udp_sender = None

        self.start_btn.setText("▶️ Iniciar Adquisición")
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        
        n_force = len(self.all_force_data)
        n_accel = len(self.all_accel_data_0)
        self.status_label.setText(f"⏸️ Detenido | Fuerza: {n_force} muestras | Aceleración: {n_accel} muestras (x2 canales)")
        
    def update_plots(self):
        """Actualiza gráficas en tiempo real"""
        # Procesar datos de fuerza (guardar último chunk para UDP)
        last_force_chunk = None
        if self.experiment_mode != "ACCEL_ONLY":
            while not force_queue.empty():
                try:
                    data = force_queue.get_nowait()
                    self.force_buffer.extend(data)
                    self.all_force_data.extend(data)
                    last_force_chunk = data
                except Exception:
                    break

        # Procesar datos de aceleración (2 canales)
        last_accel_chunk = None
        while not accel_queue.empty():
            try:
                data = accel_queue.get_nowait()
                # data tiene shape (2, samples_per_read)
                if data.ndim == 2 and data.shape[0] == 2:
                    self.accel_buffer_0.extend(data[0])  # Bancada
                    self.accel_buffer_1.extend(data[1])  # Pieza
                    self.all_accel_data_0.extend(data[0])
                    self.all_accel_data_1.extend(data[1])
                    last_accel_chunk = data[0]  # Bancada para UDP
                else:
                    # Fallback si solo hay 1 canal
                    flat = data.flatten()
                    self.accel_buffer_0.extend(flat)
                    self.accel_buffer_1.extend(flat)
                    self.all_accel_data_0.extend(flat)
                    self.all_accel_data_1.extend(flat)
                    last_accel_chunk = flat
            except Exception:
                break

        # Enviar último bloque vía UDP a la Jetson
        if self.udp_sender and last_force_chunk is not None and last_accel_chunk is not None:
            self.udp_sender.send(last_force_chunk, last_accel_chunk)

        # ── Modo ACCEL_ONLY: solo graficar acelerómetro ──
        if self.experiment_mode == "ACCEL_ONLY":
            if len(self.accel_buffer_0) > 10:
                accel_arr_0_raw = np.array(self.accel_buffer_0)
                accel_arr_0, snr_a0 = self.apply_filter(accel_arr_0_raw)
                n = len(accel_arr_0)
                t = np.linspace(-n / ACCEL_SAMPLE_RATE, 0, n)

                # Graficar acelerómetro
                self.accel_curve.setData(t, accel_arr_0)
                self.accel_curve_1.setData([], [])
                self.force_curve.setData([], [])
                self.overlay_force_curve.setData([], [])
                self.overlay_accel_curve.setData(t, accel_arr_0)

                # Labels de aceleración
                accel0_g = accel_arr_0[-1]
                accel0_rms = np.sqrt(np.mean(accel_arr_0**2))
                self.accel0_g_label.setText(f"Acel: {accel0_g:.4f} g")
                self.accel0_rms_label.setText(f"RMS: {accel0_rms:.4f} g")
                rpm = self.rpm_husillo_spin.value()
                self.force_volt_label.setText(f"RPM: {rpm}")
                self.force_kg_label.setText("N/A (solo acel)")
                self.force_N_label.setText("N/A (solo acel)")
                self.force_rms_label.setText("N/A (solo acel)")

                # SNR
                if self.filter_enabled_cb.isChecked():
                    self.filter_snr_label.setText(f"SNR: +{snr_a0:.1f} dB")
                else:
                    self.filter_snr_label.setText("SNR: --")

                # FFT solo acelerómetro (estilo interfaz_DAQ_acelerometro)
                NFFT = 4096
                if len(self.all_accel_data_0) % 1000 < 200 and n >= NFFT:
                    data_fft = accel_arr_0[-NFFT:]
                    data_fft = data_fft - np.mean(data_fft)
                    window = np.hanning(NFFT)
                    data_windowed = data_fft * window
                    yf = fft(data_windowed)
                    xf = fftfreq(NFFT, 1.0 / ACCEL_SAMPLE_RATE)[:NFFT // 2]
                    mag = 2.0 / NFFT * np.abs(yf[:NFFT // 2])
                    mag = mag / np.mean(window) * 2
                    mask = xf > 2
                    self.fft_accel_curve.setData(xf[mask], mag[mask])
                    self.fft_accel_curve_1.setData([], [])
                    self.fft_force_curve.setData([], [])
                    if np.any(mask) and np.max(mag[mask]) > 0:
                        idx_peak = np.argmax(mag[mask])
                        freq_pico = xf[mask][idx_peak]
                        amp_pico = mag[mask][idx_peak]
                        self.fft_accel_peak_line.setPos(freq_pico)
                        self.fft_accel_label_0.setText(f"Pico: {freq_pico:.1f} Hz ({amp_pico:.4f} g)")
                        self.fft_accel_label_0.setPos(freq_pico, amp_pico)
                        self.fft_accel_plot.setTitle(f"FFT Aceleración ai0 — PICO: {freq_pico:.1f} Hz ({amp_pico:.4f} g)")
            return
                
        # Actualizar gráficas si hay datos
        if len(self.force_buffer) > 10 and len(self.accel_buffer_0) > 10:
            force_arr_raw = np.array(self.force_buffer)
            accel_arr_0_raw = np.array(self.accel_buffer_0)  # Bancada
            accel_arr_1_raw = np.array(self.accel_buffer_1)  # Pieza
            
            n = min(len(force_arr_raw), len(accel_arr_0_raw), len(accel_arr_1_raw))
            force_arr_raw = force_arr_raw[-n:]
            accel_arr_0_raw = accel_arr_0_raw[-n:]
            accel_arr_1_raw = accel_arr_1_raw[-n:]
            
            # Aplicar filtrado si está habilitado
            force_arr, snr_f = self.apply_filter(force_arr_raw)
            accel_arr_0, snr_a0 = self.apply_filter(accel_arr_0_raw)
            accel_arr_1, snr_a1 = self.apply_filter(accel_arr_1_raw)
            
            # Actualizar indicador de SNR
            if self.filter_enabled_cb.isChecked():
                avg_snr = (snr_f + snr_a0 + snr_a1) / 3
                self.filter_snr_label.setText(f"SNR: +{avg_snr:.1f} dB")
                if avg_snr > 6:
                    self.filter_snr_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
                elif avg_snr > 3:
                    self.filter_snr_label.setStyleSheet("color: #f39c12; font-weight: bold;")
                else:
                    self.filter_snr_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
            else:
                self.filter_snr_label.setText("SNR: --")
                self.filter_snr_label.setStyleSheet("color: #888;")
            
            t = np.linspace(-n/FORCE_SAMPLE_RATE, 0, n)
            
            # Gráficas de tiempo (señales filtradas)
            self.force_curve.setData(t, force_arr)
            self.accel_curve.setData(t, accel_arr_0)  # Bancada en gráfica principal
            
            # Modo corte: solo canal 0, ocultar canal 1
            if self.use_single_accel:
                self.accel_curve_1.setData([], [])  # Ocultar canal 1
                accel_for_analysis = accel_arr_0  # Usar bancada para análisis
            else:
                self.accel_curve_1.setData(t, accel_arr_1)  # Pieza
                accel_for_analysis = accel_arr_1  # Usar pieza para análisis
            
            # Superposición normalizada
            if np.std(force_arr) > 1e-6 and np.std(accel_for_analysis) > 1e-6:
                force_norm = (force_arr - np.mean(force_arr)) / np.std(force_arr)
                accel_norm = (accel_for_analysis - np.mean(accel_for_analysis)) / np.std(accel_for_analysis)
                self.overlay_force_curve.setData(t, force_norm)
                self.overlay_accel_curve.setData(t, accel_norm)
                
            # Lissajous (Fuerza vs Aceleración)
            self.lissajous_curve.setData(force_arr, accel_for_analysis)
            
            # Actualizar labels
            self._update_info_labels(force_arr, accel_arr_0, accel_arr_1)
            
            # FFT y FRF (cada 10 actualizaciones)
            if len(self.all_force_data) % 1000 < 100:
                self._update_fft(force_arr, accel_arr_0, accel_arr_1)
            
            # Actualizar tab de fricción (si está en modo triangular o siempre para visualización)
            self._update_friction_analysis(force_arr, accel_for_analysis, t)
                
    def _update_info_labels(self, force_arr, accel_arr_0, accel_arr_1):
        """Actualiza etiquetas de información para 2 acelerómetros"""
        # Fuerza
        force_volt = force_arr[-1]
        force_kg = voltaje_a_fuerza_kg(force_volt)
        force_N = voltaje_a_fuerza_N(force_volt)
        force_rms = np.sqrt(np.mean(force_arr**2))
        
        self.force_volt_label.setText(f"Voltaje: {force_volt:.4f} V")
        self.force_kg_label.setText(f"Fuerza: {force_kg:.3f} kg")
        self.force_N_label.setText(f"Fuerza: {force_N:.3f} N")
        self.force_rms_label.setText(f"RMS: {force_rms:.4f} V")
        
        # Aceleración Canal 0 (Bancada)
        accel0_g = accel_arr_0[-1]
        accel0_rms = np.sqrt(np.mean(accel_arr_0**2))
        self.accel0_g_label.setText(f"Acel: {accel0_g:.4f} g")
        self.accel0_rms_label.setText(f"RMS: {accel0_rms:.4f} g")
        
        # Aceleración Canal 1 (Sobre sensor de fuerza)
        accel1_g = accel_arr_1[-1]
        accel1_rms = np.sqrt(np.mean(accel_arr_1**2))
        self.accel1_g_label.setText(f"Acel: {accel1_g:.4f} g")
        self.accel1_rms_label.setText(f"RMS: {accel1_rms:.4f} g")
        
        # Transmisibilidad: T = |a_sensor| / |a_bancada|
        if accel0_rms > 1e-6:
            transmisibilidad = accel1_rms / accel0_rms
        else:
            transmisibilidad = 0
        self.transmisibilidad_label.setText(f"T = {transmisibilidad:.3f}")
        
        # Validación F = m*a (usando acelerómetro sobre sensor - más directo)
        masa = self.masa_spin.value()
        F_medida = voltaje_a_fuerza_N(np.sqrt(np.mean(force_arr**2)))
        a_rms_ms2 = accel1_rms * 9.81  # Usar acelerómetro sobre sensor
        F_calculada = masa * a_rms_ms2
        
        if F_medida > 0.001:
            error_pct = abs(F_medida - F_calculada) / F_medida * 100
        else:
            error_pct = 0
            
        self.fma_force_label.setText(f"F medida: {F_medida:.4f} N")
        self.fma_calc_label.setText(f"m×a: {F_calculada:.4f} N")
        
        if error_pct < 10:
            color = "#27ae60"  # Verde
        elif error_pct < 25:
            color = "#f39c12"  # Amarillo
        else:
            color = "#e74c3c"  # Rojo
        self.fma_error_label.setText(f"Error: {error_pct:.1f} %")
        self.fma_error_label.setStyleSheet(f"font-weight: bold; color: {color};")
        
    def _update_fft(self, force_arr, accel_arr_0, accel_arr_1):
        """Actualiza FFT - solo ai0 con estilo interfaz_DAQ_acelerometro"""
        NFFT = 4096
        n = len(accel_arr_0)
        if n < NFFT:
            return
        
        # FFT Aceleración ai0 con ventana Hanning (amplitud lineal en g)
        data_fft = accel_arr_0[-NFFT:]
        data_fft = data_fft - np.mean(data_fft)
        window = np.hanning(NFFT)
        data_windowed = data_fft * window
        yf = fft(data_windowed)
        xf = fftfreq(NFFT, 1.0 / ACCEL_SAMPLE_RATE)[:NFFT // 2]
        mag = 2.0 / NFFT * np.abs(yf[:NFFT // 2])
        mag = mag / np.mean(window) * 2
        
        mask = xf > 2
        self.fft_accel_curve.setData(xf[mask], mag[mask])
        self.fft_accel_curve_1.setData([], [])
        
        # Pico aceleración
        if np.any(mask) and np.max(mag[mask]) > 0:
            idx_peak = np.argmax(mag[mask])
            freq_pico = xf[mask][idx_peak]
            amp_pico = mag[mask][idx_peak]
            self.fft_accel_peak_line.setPos(freq_pico)
            self.fft_accel_label_0.setText(f"Pico: {freq_pico:.1f} Hz ({amp_pico:.4f} g)")
            self.fft_accel_label_0.setPos(freq_pico, amp_pico)
            self.fft_accel_plot.setTitle(f"FFT Aceleración ai0 — PICO: {freq_pico:.1f} Hz ({amp_pico:.4f} g)")
        
        # FFT Fuerza (mantener en dB)
        n_f = len(force_arr)
        if n_f >= 256:
            freqs_f = fftfreq(n_f, 1/FORCE_SAMPLE_RATE)[:n_f//2]
            fft_force = np.abs(fft(force_arr - np.mean(force_arr)))[:n_f//2]
            fft_force_db = 20 * np.log10(fft_force + 1e-12)
            self.fft_force_curve.setData(freqs_f, fft_force_db)
            mask_f = freqs_f > 5
            if np.any(mask_f):
                idx_f = np.argmax(fft_force_db[mask_f])
                freq_pico_f = freqs_f[mask_f][idx_f]
                mag_pico_f = fft_force_db[mask_f][idx_f]
                self.fft_force_label.setText(f"PICO: {freq_pico_f:.1f} Hz")
                self.fft_force_label.setPos(freq_pico_f, mag_pico_f)
        
        # FRF y Transmisibilidad (usar datos completos)
        n_common = min(len(force_arr), len(accel_arr_0))
        if n_common >= 256:
            freqs = fftfreq(n_common, 1/FORCE_SAMPLE_RATE)[:n_common//2]
            fft_force_raw = np.abs(fft(force_arr[:n_common] - np.mean(force_arr[:n_common])))[:n_common//2]
            fft_accel_raw = np.abs(fft(accel_arr_0[:n_common] - np.mean(accel_arr_0[:n_common])))[:n_common//2]
            H = fft_accel_raw / (fft_force_raw + 1e-12)
            H_db = 20 * np.log10(np.abs(H) + 1e-12)
            self.frf_curve.setData(freqs, H_db)
            self.transmisibilidad_curve.setData([], [])
    
    def _update_friction_analysis(self, force_arr, accel_arr, t_arr):
        """
        Actualiza el análisis de fricción en tiempo real.
        Integra la aceleración dos veces para obtener desplazamiento.
        Usa el método de cruce por cero para calcular fricción dinámica.
        """
        n = len(accel_arr)
        if n < 100:
            return
            
        dt = 1.0 / FORCE_SAMPLE_RATE
        
        # 1. Preprocesar aceleración: remover DC y aplicar filtro pasa-altos
        #    para evitar drift en la integración
        accel_g = accel_arr - np.mean(accel_arr)
        
        # Filtro pasa-altos (Butterworth, fc=0.5 Hz) para remover drift
        try:
            fc = 0.5  # Frecuencia de corte
            b, a = signal.butter(2, fc / (FORCE_SAMPLE_RATE / 2), btype='high')
            accel_filtered = signal.filtfilt(b, a, accel_g)
        except:
            accel_filtered = accel_g
        
        # Convertir a m/s²
        accel_ms2 = accel_filtered * 9.81
        
        # 2. Primera integración: Aceleración → Velocidad
        velocity = cumulative_trapezoid(accel_ms2, dx=dt, initial=0)
        
        # Remover drift de velocidad (detrend lineal)
        velocity = signal.detrend(velocity)
        
        # Convertir a mm/s
        velocity_mms = velocity * 1000
        
        # 3. Segunda integración: Velocidad → Desplazamiento
        displacement = cumulative_trapezoid(velocity, dx=dt, initial=0)
        
        # Remover drift de desplazamiento (detrend lineal)
        displacement = signal.detrend(displacement)
        
        # Convertir a mm
        displacement_mm = displacement * 1000
        
        # 4. Convertir fuerza a Newtons
        force_N = np.array([voltaje_a_fuerza_N(v) for v in force_arr])
        force_N = force_N - np.mean(force_N)  # Centrar en cero
        
        # 5. Actualizar gráficas
        # Histéresis: Fuerza vs Desplazamiento
        self.friction_curve.setData(displacement_mm, force_N)
        
        # Velocidad vs Tiempo
        self.velocity_curve.setData(t_arr, velocity_mms)
        
        # 6. Calcular métricas de fricción
        F_max = np.max(force_N)
        F_min = np.min(force_N)
        F_friction_peak = (F_max - F_min) / 2  # Método pico a pico (simple)
        
        x_max = np.max(displacement_mm)
        x_min = np.min(displacement_mm)
        stroke = x_max - x_min  # Carrera
        
        v_mean = np.mean(np.abs(velocity_mms))
        
        # ============================================
        # MÉTODO DE CRUCE POR CERO (Zero-Crossing)
        # Más preciso que el método pico a pico
        # ============================================
        F_friction_zc = 0
        F_forward = 0
        F_backward = 0
        n_forward = 0
        n_backward = 0
        
        try:
            # Definir zona de cruce por cero: |x| < 10% del stroke
            x_threshold = stroke * 0.10 if stroke > 0.001 else 0.01
            
            # Encontrar puntos cerca del cruce por cero
            near_zero_mask = np.abs(displacement_mm) < x_threshold
            
            if np.sum(near_zero_mask) > 5:
                # Separar por dirección de velocidad
                forward_mask = near_zero_mask & (velocity_mms > 0)  # Avance
                backward_mask = near_zero_mask & (velocity_mms < 0)  # Retroceso
                
                if np.sum(forward_mask) > 2:
                    F_forward = np.mean(force_N[forward_mask])
                    n_forward = np.sum(forward_mask)
                    
                if np.sum(backward_mask) > 2:
                    F_backward = np.mean(force_N[backward_mask])
                    n_backward = np.sum(backward_mask)
                    
                # Fricción dinámica por cruce por cero
                if n_forward > 0 and n_backward > 0:
                    F_friction_zc = (F_forward - F_backward) / 2
        except:
            F_friction_zc = F_friction_peak  # Fallback al método simple
        
        # Usar el método de cruce por cero si es válido, sino el pico a pico
        if abs(F_friction_zc) > 0.001:
            F_friction_dyn = F_friction_zc
            method_used = "ZC"
        else:
            F_friction_dyn = F_friction_peak
            method_used = "P2P"
        
        # Energía disipada (área del lazo de histéresis)
        # E = ∮ F dx - calculada correctamente siguiendo el lazo
        try:
            # Calcular área usando la fórmula del polígono (Shoelace formula)
            # Más precisa que ordenar por x
            energy_mJ = 0.5 * np.abs(np.sum(displacement_mm[:-1] * force_N[1:] - 
                                            displacement_mm[1:] * force_N[:-1]))
        except:
            energy_mJ = 0
        
        # Guardar últimos valores calculados para captura
        self._last_friction_data = {
            'displacement_mm': displacement_mm,
            'velocity_mms': velocity_mms,
            'force_N': force_N,
            'F_friction_zc': F_friction_zc,
            'F_friction_peak': F_friction_peak,
            'F_forward': F_forward,
            'F_backward': F_backward,
            'n_forward': n_forward,
            'n_backward': n_backward,
            'stroke': stroke,
            'v_mean': v_mean,
            'energy_mJ': energy_mJ
        }
        
        # 7. Actualizar labels
        self.friction_fmax_label.setText(f"F_max: {F_max:.3f} N")
        self.friction_fmin_label.setText(f"F_min: {F_min:.3f} N")
        self.friction_fdyn_label.setText(f"F_fricción ({method_used}): {F_friction_dyn:.3f} N")
        self.friction_xmax_label.setText(f"x_max: {x_max:.3f} mm")
        self.friction_xmin_label.setText(f"x_min: {x_min:.3f} mm")
        self.friction_stroke_label.setText(f"Carrera: {stroke:.3f} mm")
        self.friction_velocity_label.setText(f"Velocidad media: {v_mean:.2f} mm/s")
        self.friction_energy_label.setText(f"Energía disipada: {energy_mJ:.3f} mJ")
        
        # Actualizar labels adicionales de cruce por cero
        if hasattr(self, 'friction_fforward_label'):
            self.friction_fforward_label.setText(f"F_forward (x≈0): {F_forward:.3f} N (n={n_forward})")
            self.friction_fbackward_label.setText(f"F_backward (x≈0): {F_backward:.3f} N (n={n_backward})")
        
    def autoset_y(self):
        """Auto-ajusta los ejes Y"""
        if len(self.force_buffer) > 10:
            force_arr = np.array(self.force_buffer)
            margin = (np.max(force_arr) - np.min(force_arr)) * 0.15 + 0.001
            self.force_plot.setYRange(np.min(force_arr) - margin, np.max(force_arr) + margin)
            
        if len(self.accel_buffer_0) > 10:
            accel_arr_0 = np.array(self.accel_buffer_0)
            accel_arr_1 = np.array(self.accel_buffer_1)
            all_accel = np.concatenate([accel_arr_0, accel_arr_1])
            margin = (np.max(all_accel) - np.min(all_accel)) * 0.15 + 0.001
            self.accel_plot.setYRange(np.min(all_accel) - margin, np.max(all_accel) + margin)
            
    def toggle_recording(self):
        """Inicia o detiene la grabación continua"""
        if not self.is_recording:
            # Iniciar grabación - limpiar buffers
            self.all_force_data.clear()
            self.all_accel_data_0.clear()
            self.all_accel_data_1.clear()
            self.is_recording = True
            self.capture_btn.setText("⏹ Detener Grabación")
            self.capture_btn.setStyleSheet("background-color: #c0392b; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
            self.status_label.setText("🔴 GRABANDO... (presiona Detener cuando termines)")
            self.recording_start_time = time.time()
        else:
            # Detener grabación
            self.is_recording = False
            duration = time.time() - self.recording_start_time
            self.capture_btn.setText("🔴 Iniciar Grabación")
            self.capture_btn.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
            n_samples = len(self.all_accel_data_0) if self.experiment_mode == "ACCEL_ONLY" else len(self.all_force_data)
            self.status_label.setText(f"✅ Grabación detenida: {n_samples} muestras ({duration:.1f}s) - Presiona 'Guardar' para exportar")
            
    def capture_experiment(self):
        """Captura los datos actuales como un experimento (2 acelerómetros)"""
        if len(self.all_force_data) < 100 or len(self.all_accel_data_0) < 100:
            QMessageBox.warning(self, "Sin datos", "No hay suficientes datos para capturar.\nInicia grabación primero.")
            return
            
        force_arr = np.array(self.all_force_data)
        accel_arr_0 = np.array(self.all_accel_data_0)  # Bancada
        accel_arr_1 = np.array(self.all_accel_data_1)  # Pieza
        
        # Calcular métricas
        freq = self.freq_spin.value()
        masa = self.masa_spin.value()
        
        force_rms_V = np.sqrt(np.mean(force_arr**2))
        force_rms_N = voltaje_a_fuerza_N(force_rms_V)
        
        # Usar acelerómetro en pieza para F=ma
        accel_rms_g = np.sqrt(np.mean(accel_arr_1**2))
        accel_rms_ms2 = accel_rms_g * 9.81
        
        # Transmisibilidad
        accel0_rms = np.sqrt(np.mean(accel_arr_0**2))
        transmisibilidad = accel_rms_g / accel0_rms if accel0_rms > 1e-6 else 0
        
        F_calculada = masa * accel_rms_ms2
        error_pct = abs(force_rms_N - F_calculada) / force_rms_N * 100 if force_rms_N > 0.001 else 0
        
        # Calcular fase (usar acelerómetro en pieza)
        n = min(len(force_arr), len(accel_arr_1))
        freqs = fftfreq(n, 1/FORCE_SAMPLE_RATE)[:n//2]
        fft_force = fft(force_arr[:n] - np.mean(force_arr[:n]))[:n//2]
        fft_accel = fft(accel_arr_1[:n] - np.mean(accel_arr_1[:n]))[:n//2]
        
        idx_peak = np.argmax(np.abs(fft_force[1:])) + 1
        phase_diff = np.angle(fft_accel[idx_peak], deg=True) - np.angle(fft_force[idx_peak], deg=True)
        while phase_diff > 180:
            phase_diff -= 360
        while phase_diff < -180:
            phase_diff += 360
            
        # Calcular métricas de fricción si está en modo TRIANGLE
        F_friction_dyn = 0
        F_friction_zc = 0
        F_friction_peak = 0
        F_forward = 0
        F_backward = 0
        velocity_target = 0
        stroke = 0
        energy_mJ = 0
        v_mean = 0
        
        if self.experiment_mode == "TRIANGLE":
            velocity_target = self.velocity_spin.value()
            
            # Calcular desplazamiento por integración doble
            dt = 1.0 / FORCE_SAMPLE_RATE
            accel_g = accel_arr_1 - np.mean(accel_arr_1)
            
            try:
                fc = 0.5
                b, a = signal.butter(2, fc / (FORCE_SAMPLE_RATE / 2), btype='high')
                accel_filtered = signal.filtfilt(b, a, accel_g)
            except:
                accel_filtered = accel_g
            
            accel_ms2 = accel_filtered * 9.81
            velocity = cumulative_trapezoid(accel_ms2, dx=dt, initial=0)
            velocity = signal.detrend(velocity)
            velocity_mms = velocity * 1000
            
            displacement = cumulative_trapezoid(velocity, dx=dt, initial=0)
            displacement = signal.detrend(displacement)
            displacement_mm = displacement * 1000
            
            force_N = np.array([voltaje_a_fuerza_N(v) for v in force_arr])
            force_N = force_N - np.mean(force_N)
            
            F_max = np.max(force_N)
            F_min = np.min(force_N)
            F_friction_peak = (F_max - F_min) / 2  # Método pico a pico
            stroke = np.max(displacement_mm) - np.min(displacement_mm)
            v_mean = np.mean(np.abs(velocity_mms))
            
            # ============================================
            # MÉTODO DE CRUCE POR CERO (Zero-Crossing)
            # ============================================
            try:
                x_threshold = stroke * 0.10 if stroke > 0.001 else 0.01
                near_zero_mask = np.abs(displacement_mm) < x_threshold
                
                if np.sum(near_zero_mask) > 5:
                    forward_mask = near_zero_mask & (velocity_mms > 0)
                    backward_mask = near_zero_mask & (velocity_mms < 0)
                    
                    if np.sum(forward_mask) > 2:
                        F_forward = np.mean(force_N[forward_mask])
                    if np.sum(backward_mask) > 2:
                        F_backward = np.mean(force_N[backward_mask])
                    
                    if np.sum(forward_mask) > 0 and np.sum(backward_mask) > 0:
                        F_friction_zc = (F_forward - F_backward) / 2
            except:
                F_friction_zc = 0
            
            # Usar método de cruce por cero si es válido
            F_friction_dyn = F_friction_zc if abs(F_friction_zc) > 0.001 else F_friction_peak
            
            # Energía disipada (Shoelace formula)
            try:
                energy_mJ = 0.5 * np.abs(np.sum(displacement_mm[:-1] * force_N[1:] - 
                                                displacement_mm[1:] * force_N[:-1]))
            except:
                energy_mJ = 0
            
        # Parámetros de corte (modo CUTTING)
        rpm_husillo = self.rpm_husillo_spin.value() if self.experiment_mode == "CUTTING" else 0
        rpm_avance = self.rpm_avance_spin.value() if self.experiment_mode == "CUTTING" else 0
        cutter_condition = self.cutter_condition_combo.currentText() if self.experiment_mode == "CUTTING" else "N/A"

        # Guardar experimento (con ambos acelerómetros)
        exp = {
            'experiment_type': self.experiment_mode,
            'frecuencia': freq,
            'masa': masa,
            'force_rms_N': force_rms_N,
            'accel_rms_ms2': accel_rms_ms2,
            'accel0_rms_g': accel0_rms,  # Bancada
            'accel1_rms_g': accel_rms_g,  # Pieza
            'transmisibilidad': transmisibilidad,
            'F_calculada': F_calculada,
            'error_pct': error_pct,
            'fase': phase_diff,
            # Métricas de fricción (modo TRIANGLE - Zero-Crossing Method)
            'F_friction_dyn': F_friction_dyn,
            'F_friction_zc': F_friction_zc,
            'F_friction_peak': F_friction_peak,
            'F_forward': F_forward,
            'F_backward': F_backward,
            'velocity_target_mms': velocity_target,
            'velocity_mean_mms': v_mean,
            'stroke_mm': stroke,
            'energy_mJ': energy_mJ,
            # Parámetros de corte (modo CUTTING)
            'rpm_husillo': rpm_husillo,
            'rpm_avance_x': rpm_avance,
            'cutter_condition': cutter_condition,
            'notas': self.notas_edit.text(),
            'force_data': force_arr.copy(),
            'accel_data_0': accel_arr_0.copy(),  # Bancada
            'accel_data_1': accel_arr_1.copy(),  # Pieza
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        self.experimentos.append(exp)
        
        # Actualizar tabla
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(f"{freq:.1f}"))
        self.results_table.setItem(row, 1, QTableWidgetItem(f"{masa:.3f}"))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{force_rms_N:.4f}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{accel_rms_ms2:.4f}"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{F_calculada:.4f}"))
        self.results_table.setItem(row, 5, QTableWidgetItem(f"{error_pct:.2f}"))
        self.results_table.setItem(row, 6, QTableWidgetItem(f"{phase_diff:.2f}"))
        self.results_table.setItem(row, 7, QTableWidgetItem(exp['notas']))
        
        # Limpiar buffers para siguiente experimento
        self.all_force_data.clear()
        self.all_accel_data_0.clear()
        self.all_accel_data_1.clear()
        
        self.status_label.setText(f"✅ Experimento #{len(self.experimentos)} capturado @ {freq} Hz")
        
        # Actualizar resumen
        self._update_summary()
        
    def _update_summary(self):
        """Actualiza el resumen de experimentos"""
        if not self.experimentos:
            return
            
        text = "=" * 50 + "\n"
        text += "RESUMEN DE CARACTERIZACIÓN\n"
        text += "=" * 50 + "\n\n"
        
        text += f"Total experimentos: {len(self.experimentos)}\n\n"
        
        errors = [e['error_pct'] for e in self.experimentos]
        text += f"Error F=ma promedio: {np.mean(errors):.2f} %\n"
        text += f"Error F=ma máximo: {np.max(errors):.2f} %\n"
        text += f"Error F=ma mínimo: {np.min(errors):.2f} %\n\n"
        
        freqs = [e['frecuencia'] for e in self.experimentos]
        text += f"Rango de frecuencias: {min(freqs):.1f} - {max(freqs):.1f} Hz\n"
        
        self.summary_text.setText(text)
        
    def save_results(self):
        """Guarda los datos grabados directamente"""
        # Si hay datos en buffer (grabación reciente), guardar directamente
        if len(self.all_force_data) >= 100 or (self.experiment_mode == "ACCEL_ONLY" and len(self.all_accel_data_0) >= 100):
            self._save_raw_recording()
            return
            
        if not self.experimentos:
            QMessageBox.warning(self, "Sin datos", "No hay datos para guardar.\nInicia grabación primero.")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _save_raw_recording(self):
        """Guarda grabación cruda directamente a archivo"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rpm_husillo = self.rpm_husillo_spin.value()

        # ── Modo ACCEL_ONLY: guardar solo acelerómetro ──
        if self.experiment_mode == "ACCEL_ONLY":
            accel_arr_0 = np.array(self.all_accel_data_0)
            n = len(accel_arr_0)
            t = np.arange(n) / ACCEL_SAMPLE_RATE

            df = pd.DataFrame({
                'tiempo_s': t,
                'aceleracion_sensor_g': accel_arr_0[:n],
            })

            filename = f"accel_{timestamp}_{rpm_husillo}rpm.csv"
            filepath = os.path.join(DATOS_DIR, filename)
            df.to_csv(filepath, index=False, sep='\t')

            import json as _json
            sidecar = {
                'file': filename,
                'timestamp': timestamp,
                'fs_hz': ACCEL_SAMPLE_RATE,
                'n_samples': int(n),
                'duration_s': round(float(n / ACCEL_SAMPLE_RATE), 3),
                'experiment_mode': 'ACCEL_ONLY',
                'rpm_husillo': rpm_husillo,
                'notas': self.notas_edit.text(),
                'canal_aceleracion': 'ai0 bancada (NI 9234)',
            }
            json_path = filepath.replace('.csv', '_meta.json')
            with open(json_path, 'w', encoding='utf-8') as jf:
                _json.dump(sidecar, jf, indent=2, ensure_ascii=False)

            duration = n / ACCEL_SAMPLE_RATE
            QMessageBox.information(self, "Guardado",
                f"Datos guardados (solo acel):\n{filepath}\n{json_path}\n\n"
                f"Muestras: {n}\nDuracion: {duration:.2f}s\nRPM: {rpm_husillo}")
            self.status_label.setText(f"💾 Guardado: {filename}")

            self.all_accel_data_0.clear()
            self.all_accel_data_1.clear()
            return

        # ── Modos con fuerza + acelerómetro ──
        force_arr = np.array(self.all_force_data)
        accel_arr_0 = np.array(self.all_accel_data_0)  # Canal 0 = Prensa
        accel_arr_1 = np.array(self.all_accel_data_1)  # Canal 1 = Pieza
        
        n = min(len(force_arr), len(accel_arr_0), len(accel_arr_1))
        t = np.arange(n) / FORCE_SAMPLE_RATE
        
        # Crear DataFrame con datos crudos.
        # En modo corte se guarda formato compatible con diciembre 2025:
        # tiempo_s, fuerza_V, aceleracion_sensor_g
        if self.use_single_accel:
            df = pd.DataFrame({
                'tiempo_s': t,
                'fuerza_V': force_arr[:n],
                'aceleracion_sensor_g': accel_arr_0[:n],
                'aceleracion_bancada_g': accel_arr_0[:n]
            })
        else:
            df = pd.DataFrame({
                'tiempo_s': t,
                'fuerza_V': force_arr[:n],
                'aceleracion_bancada_g': accel_arr_0[:n],  # ai0 = Bancada
                'aceleracion_pieza_g': accel_arr_1[:n]    # ai1 = Pieza
            })
        
        # Nombre del archivo (incluye condición del cortador)
        cutter_cond = self.cutter_condition_combo.currentText().lower().replace(" ", "_") if self.experiment_mode == "CUTTING" else "general"
        rpm_str = f"_{rpm_husillo}rpm" if self.experiment_mode == "CUTTING" else ""
        filename = f"corte_{timestamp}{rpm_str}_{cutter_cond}.csv"
        filepath = os.path.join(DATOS_DIR, filename)

        df.to_csv(filepath, index=False, sep='\t')

        # Guardar JSON sidecar con metadatos para la Jetson / Hailo
        import json as _json
        sidecar = {
            'file': filename,
            'timestamp': timestamp,
            'fs_hz': FORCE_SAMPLE_RATE,
            'n_samples': int(n),
            'duration_s': round(float(n / FORCE_SAMPLE_RATE), 3),
            'experiment_mode': self.experiment_mode,
            'cutter_condition': self.cutter_condition_combo.currentText() if self.experiment_mode == "CUTTING" else "N/A",
            'rpm_husillo': rpm_husillo if self.experiment_mode == "CUTTING" else 0,
            'rpm_avance_x': self.rpm_avance_spin.value() if self.experiment_mode == "CUTTING" else 0,
            'notas': self.notas_edit.text(),
            'canal_fuerza': 'ai0 (NI 9205)',
            'canal_aceleracion': 'ai0 bancada (NI 9234)',
        }
        json_path = filepath.replace('.csv', '_meta.json')
        with open(json_path, 'w', encoding='utf-8') as jf:
            _json.dump(sidecar, jf, indent=2, ensure_ascii=False)

        duration = n / FORCE_SAMPLE_RATE

        QMessageBox.information(self, "Guardado",
            f"Datos guardados:\n{filepath}\n{json_path}\n\n"
            f"Muestras: {n}\n"
            f"Duracion: {duration:.2f}s")

        self.status_label.setText(f"💾 Guardado: {filename}")
        
        # Limpiar buffers
        self.all_force_data.clear()
        self.all_accel_data_0.clear()
        self.all_accel_data_1.clear()
        
    def closeEvent(self, event):
        """Limpieza al cerrar"""
        self.stop_acquisition()
        event.accept()


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = CaracterizacionFuerzaGUI()
    window.show()
    
    print("\n" + "="*60)
    print("🔬 CARACTERIZACIÓN SENSOR DE FUERZA DYMH-105")
    print("="*60)
    print(f"   Fuerza: Canal ai0 (datos crudos en V)")
    print(f"   Acelerómetro: ai0 bancada (datos en g)")
    print(f"   Sample Rate: {FORCE_SAMPLE_RATE} Hz")
    print(f"   Datos en: {DATOS_DIR}")
    print("="*60)
    print("\n📋 INSTRUCCIONES:")
    print("   1. Selecciona el TIPO DE EXPERIMENTO:")
    print("      🔄 Barrido/Chirp: Para identificación de inercia (onda SENOIDAL)")
    print("      📐 Triangular: Para caracterización de fricción (onda TRIANGULAR)")
    print("      🔪 Fuerza de Corte: Para medición de histéresis en fresado")
    print("   2. Configura el Keysight a la frecuencia deseada")
    print("   3. Ajusta la ganancia del amplificador TIRA")
    print("   4. Presiona 'Iniciar Adquisición'")
    print("   5. Presiona 'Capturar Experimento' para guardar")
    print("   6. Repite para diferentes frecuencias/velocidades")
    print("   7. Presiona 'Guardar Resultados' al finalizar")
    print("")
    print("🧩 MODO FRICCIÓN (Triangular):")
    print("   - Configura el generador a onda TRIANGULAR")
    print("   - Usa frecuencias bajas (0.5-5 Hz) para velocidad constante")
    print("   - El desplazamiento se estima integrando la aceleración")
    print("   - El lazo F vs x debe formar un paralelogramo")
    print("   - F_fricción ≈ (F_max - F_min) / 2")
    print("")
    print("🔪 MODO FUERZA DE CORTE:")
    print("   - Configura RPM del husillo y avance")
    print("   - Mide fuerza y aceleración durante el corte real")
    print("   - Análisis: THD, Bouc-Wen, lazo F-a")
    print("   - Detecta histéresis del proceso de corte")
    print("")
    print("📡 MODO SOLO ACELERÓMETRO:")
    print("   - Captura solo vibración (sin sensor de fuerza)")
    print("   - Configura RPM del husillo para etiquetar la prueba")
    print("   - Archivos: accel_YYYYMMDD_HHMMSS_XXXrpm.csv")
    print("   - Útil para baseline de husillo sin corte")
    print("="*60 + "\n")
    
    sys.exit(app.exec_())

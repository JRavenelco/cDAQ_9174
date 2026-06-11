#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARACTERIZACIÓN DEL SENSOR DE FUERZA DYMH-105 CON SHAKER TIRA

Configuración:
- Shaker: TIRA TV 51144IN + Amplificador BAA 1000
- Generador: Keysight 33220A (manual)
- Celda de carga: Daysensor DYMH-105 (500 kg, 1.7 mV/V)
- Acondicionador: INA-4LC-8NTC (G=601)
- ADC FUERZA: ADS1219 (24-bit) leído por una Raspberry Pi Pico (maestro I2C local @1000 SPS)
    Cadena: celda -> INA-4LC (G=601) -> ADS1219 AIN0/AIN1 -> Pico (I2C) -> USB CDC -> PC
    Firmware del Pico: firmware_pico_ads1219/ (bare-metal multicore)
- Acelerómetro: PCB 352C33 (100 mV/g) -> NI cDAQ-9174 con NI 9234 (IEPE)

TASAS:
- Fuerza:       1000 Hz reales por el Pico (ADS1219 a 1000 SPS); el fs real se mide y se guarda
- Aceleración:  1000 Hz configurada en NI 9234 (sigue a FORCE_SAMPLE_RATE)

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
_src_root = os.path.dirname(_here)
_proto_dir = os.path.join(_src_root, "remote_monitoring")
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

# Intentar importar nidaqmx (solo para el ACELERÓMETRO en NI 9234)
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    try:
        from nidaqmx.constants import AccelUnits, AccelSensitivityUnits, ExcitationSource, Coupling
    except ImportError:
        AccelUnits = AccelSensitivityUnits = ExcitationSource = Coupling = None
    HAS_NIDAQMX = True
except ImportError:
    HAS_NIDAQMX = False
    AccelUnits = AccelSensitivityUnits = ExcitationSource = Coupling = None
    print("⚠️ nidaqmx no disponible - Aceleración en modo simulación")

# Comunicación con la Raspberry Pi Pico por USB CDC (puerto serie virtual)
try:
    import serial
    from serial.tools import list_ports
    HAS_SERIAL = True
except Exception as _e:
    HAS_SERIAL = False
    serial = None
    list_ports = None
    print(f"⚠️ pyserial no disponible - instálalo con: pip install pyserial ({_e})")
HAS_DWF = False  # esta versión NO usa WaveForms/Analog Discovery

# ============================================
# CONFIGURACIÓN DEL HARDWARE
# ============================================

# ── FUERZA: celda -> INA-4LC (G=601) -> ADS1219 (24-bit) -> Raspberry Pi Pico -> USB CDC ──
# El Pico (firmware en firmware_pico_ads1219/) lee el ADS1219 a 1000 SPS reales y envía
# bloques binarios por el puerto serie virtual. Aquí solo se desempaquetan.
PICO_PORT      = None        # None = autodetecta el Pico (VID 0x2E8A); o fija "COM7"
PICO_BAUD      = 115200      # USB CDC ignora el baud real, pero pyserial exige un valor
PICO_MAGIC0    = 0xA5
PICO_MAGIC1    = 0x5A
PICO_HDR_SIZE  = 16          # magic(2) + n(2) + seq(4) + t_us(4) + dropped(4)
PICO_BLK_MAX   = 64          # debe coincidir con BLK_MAX del firmware
ADS1219_VREF   = 2.048       # referencia interna del ADS1219 (V)
ADS1219_GAIN   = 4           # debe coincidir con ADS_GAIN del firmware del Pico
ADS1219_SPS    = 1000        # el Pico lee el ADS1219 a 1000 SPS reales
FORCE_SAMPLE_RATE = 1000     # nominal: el Pico entrega ~1000 Hz; el fs REAL se mide y se guarda igual

# NI 9234 - Aceleración (1 CANAL: bancada)
ACCEL_DEVICE = "cDAQ1Mod2"
ACCEL_CHANNELS = ["ai0"]  # Solo acelerómetro en bancada
# El NI 9234 NO puede muestrear a 990/1000 Hz (mínimo ~1652 Hz; solo hace 51200/n).
# Se corre a una tasa válida y sobre-muestreada; al guardar se re-muestrea a la rejilla
# del reloj medido del Pico (ver _save_raw_recording). La tasa REAL se lee de samp_clk_rate.
ACCEL_SAMPLE_RATE = 2048   # 51200/25 = 2048 Hz exactos (válido en el 9234)

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
ACEL_SENSIBILIDAD_MV_G = 102.4  # mV/g (ambos sensores)
# Rango esperado para DAQmx. El NI 9234 tiene rango de entrada fijo,
# pero acotar la escala fisica evita trabajar con una ventana excesiva.
ACCEL_MIN_G = -5.0
ACCEL_MAX_G = 5.0
ACCEL_IEPE_CURRENT_A = 0.004  # 4 mA IEPE interno del NI 9234
ACCEL_COUPLING = "AC"
# ai0: Acelerómetro en BANCADA

# Masa de prueba (ajustar según tu configuración)
MASA_PRUEBA_KG = 0.5  # kg

# Directorio de datos
DATOS_DIR = _here
os.makedirs(DATOS_DIR, exist_ok=True)

# Colas de datos
force_queue = queue.Queue(maxsize=40)
accel_queue = queue.Queue(maxsize=20)  # Un solo canal: ai0


# ============================================
# FUNCIONES DE CONVERSIÓN
# ============================================

def voltaje_a_fuerza_kg(V_medido):
    """MODO CRUDO: Devuelve voltaje directamente (V)"""
    return V_medido

def voltaje_a_fuerza_N(V_medido):
    """MODO CRUDO: Devuelve voltaje directamente (V)"""
    return V_medido

def voltaje_a_aceleracion_g(V_medido):
    """El NI 9234 ya devuelve datos en g (IEPE calibrado)"""
    return V_medido

def voltaje_a_aceleracion_ms2(V_medido):
    """Devuelve en g (sin conversión a m/s²)"""
    return V_medido


# ============================================
# HILOS DE ADQUISICIÓN
# ============================================

class ForceAcquisitionThread(threading.Thread):
    """FUERZA vía Raspberry Pi Pico: celda DYMH-105 -> INA-4LC (G=601) -> ADS1219 (24-bit)
    -> Pico (maestro I2C local @1000 SPS) -> USB CDC (bloques binarios) -> PC.

    El Pico marca el muestreo con el reloj del propio ADS1219 (sin round-trip USB por
    muestra, que era lo que topaba al Analog Discovery a ~487 Hz), bufferiza y envía
    bloques. Aquí se desempaquetan y se empuja (voltajes, t_perf_counter) a force_queue,
    MISMA interfaz que la versión Analog Discovery: el resto de la GUI (medición de fs
    real, guardado) no cambia.
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.ser = None
        self.dropped = 0          # muestras perdidas reportadas por el Pico (overflow de su cola)
        self.last_seq = None      # para detectar huecos de bloque en el enlace USB
        self.link_gaps = 0

    def _autodetect_port(self):
        if PICO_PORT:
            return PICO_PORT
        if list_ports is None:
            return None
        otros = []
        for p in list_ports.comports():
            if getattr(p, "vid", None) == 0x2E8A:   # VID de Raspberry Pi (Pico)
                return p.device
            otros.append(p.device)
        return otros[0] if len(otros) == 1 else None

    def _read_exact(self, n):
        buf = bytearray()
        while self.running and len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
        return bytes(buf)

    def run(self):
        if not HAS_SERIAL:
            print("Error fuerza (Pico): pyserial no disponible (pip install pyserial)")
            return
        port = self._autodetect_port()
        if not port:
            print("Error fuerza (Pico): no se encontró el puerto del Pico. "
                  "Conéctalo o fija PICO_PORT='COMx'.")
            return
        try:
            self.ser = serial.Serial(port, PICO_BAUD, timeout=0.2)
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            self.ser.write(b'S')          # START streaming
            self.ser.flush()
            print(f"✅ Pico ADS1219 en {port}: INA G=601, ADC G={ADS1219_GAIN}, "
                  f"{ADS1219_SPS} SPS reales, USB CDC binario, UI objetivo {FORCE_SAMPLE_RATE} Hz")

            scale = (ADS1219_VREF / ADS1219_GAIN) / (1 << 23)   # raw -> volts
            while self.running:
                # --- sincroniza con el magic del bloque ---
                b = self.ser.read(1)
                if not b or b[0] != PICO_MAGIC0:
                    continue
                b = self.ser.read(1)
                if not b or b[0] != PICO_MAGIC1:
                    continue

                hdr = self._read_exact(PICO_HDR_SIZE - 2)       # n, seq, t_us, dropped
                if len(hdr) < PICO_HDR_SIZE - 2:
                    break
                n, seq, t_us, dropped = struct.unpack("<HIII", hdr)
                if n == 0 or n > PICO_BLK_MAX:
                    continue                                    # header dudoso: re-sincroniza

                payload = self._read_exact(4 * n)
                if len(payload) < 4 * n:
                    break

                raws = np.frombuffer(payload, dtype="<i4", count=n).astype(np.float64)
                volts = raws * scale

                # Diagnóstico: drops del Pico y huecos de seq en el enlace USB
                self.dropped = dropped
                if self.last_seq is not None and seq != self.last_seq:
                    self.link_gaps += 1
                self.last_seq = (seq + n) & 0xFFFFFFFF

                if not force_queue.full():
                    force_queue.put((volts, time.perf_counter()))
        except Exception as e:
            if self.running:
                print(f"Error fuerza (Pico): {e}")
        finally:
            self._close()

    def _close(self):
        if self.ser is not None:
            try:
                self.ser.write(b'X'); self.ser.flush()          # STOP streaming
            except Exception:
                pass
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def stop(self):
        self.running = False


class AccelAcquisitionThread(threading.Thread):
    """
    Hilo para adquisición de aceleración (NI 9234)

    Canal 0 (ai0): Acelerómetro en bancada
    """
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.task = None
        self.samples_per_read = 25
        self.n_channels = len(ACCEL_CHANNELS)
        self.actual_fs = float(ACCEL_SAMPLE_RATE)   # tasa REAL del 9234 (samp_clk_rate), se lee al iniciar

    def run(self):
        if not HAS_NIDAQMX:
            self._run_simulation()
            return

        try:
            self.task = nidaqmx.Task()
            for ch in ACCEL_CHANNELS:
                # NI 9234 con IEPE interno, unidades en g y acoplamiento AC.
                accel_kwargs = {}
                if AccelUnits is not None:
                    accel_kwargs["units"] = AccelUnits.G
                if AccelSensitivityUnits is not None:
                    accel_kwargs["sensitivity_units"] = AccelSensitivityUnits.MILLIVOLTS_PER_G
                if ExcitationSource is not None:
                    accel_kwargs["current_excit_source"] = ExcitationSource.INTERNAL

                accel_chan = self.task.ai_channels.add_ai_accel_chan(
                    f"{ACCEL_DEVICE}/{ch}",
                    sensitivity=ACEL_SENSIBILIDAD_MV_G,
                    min_val=ACCEL_MIN_G,
                    max_val=ACCEL_MAX_G,
                    current_excit_val=ACCEL_IEPE_CURRENT_A,
                    **accel_kwargs
                )
                if Coupling is not None:
                    try:
                        accel_chan.ai_coupling = Coupling.AC
                    except Exception as e:
                        print(f"⚠️ No se pudo configurar AC coupling en {ch}: {e}")
            self.task.timing.cfg_samp_clk_timing(
                rate=ACCEL_SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples_per_read * 10
            )
            self.task.start()
            try:
                self.actual_fs = float(self.task.timing.samp_clk_rate)
            except Exception:
                self.actual_fs = float(ACCEL_SAMPLE_RATE)
            print(f"✅ NI 9234: {self.n_channels} canales activos | fs solicitado {ACCEL_SAMPLE_RATE} → real {self.actual_fs:.2f} Hz")

            while self.running:
                try:
                    data = self.task.read(number_of_samples_per_channel=self.samples_per_read)
                    data_np = np.array(data)  # Shape: (n_ch, samples_per_read)
                    if not accel_queue.full():
                        accel_queue.put(data_np)
                except Exception as e:
                    if self.running:
                        print(f"Error lectura aceleración: {e}")
                    break

        except Exception as e:
            print(f"Error inicializando aceleración IEPE: {e}")
            self._run_voltage_mode()
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass

    def _run_voltage_mode(self):
        """Modo voltaje como fallback"""
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
            try:
                self.actual_fs = float(self.task.timing.samp_clk_rate)
            except Exception:
                self.actual_fs = float(ACCEL_SAMPLE_RATE)
            print(f"⚠️ NI 9234: Modo voltaje (fallback) - {self.n_channels} canales | fs real {self.actual_fs:.2f} Hz")

            while self.running:
                try:
                    data = self.task.read(number_of_samples_per_channel=self.samples_per_read)
                    data_np = np.array(data)
                    data_g = voltaje_a_aceleracion_g(data_np)
                    if not accel_queue.full():
                        accel_queue.put(data_g)
                except:
                    break
        except Exception as e:
            print(f"Error modo voltaje: {e}")

    def _run_simulation(self):
        """Modo simulación con 1 canal"""
        t = 0
        while self.running:
            dt = self.samples_per_read / ACCEL_SAMPLE_RATE
            t_arr = np.linspace(t, t + dt, self.samples_per_read)
            freq_sim = 40

            phase0 = np.radians(15)
            acel0 = 0.3 * np.sin(2 * np.pi * freq_sim * t_arr + phase0) + 0.01 * np.random.randn(self.samples_per_read)

            if not accel_queue.full():
                accel_queue.put(acel0)
            t += dt
            time.sleep(dt * 0.9)

    def stop(self):
        self.running = False


# ============================================
# HILO DE ENVÍO UDP A JETSON
# ============================================

class UDPSenderThread(threading.Thread):
    """Hilo que envía bloques (fuerza + accel) vía UDP a la Jetson.

    Fuerza y aceleración se configuran a la misma tasa para mantener compatibilidad
    con pack_data(), que lleva un único campo de frecuencia de muestreo.
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
                if self.running and self.packets_sent == 0:
                    print(f"[UDP] Jetson {self.jetson_ip}:{self.port} no alcanzable — paquetes se descartan")
                self.packets_dropped += 1
            except Exception as e:
                if self.running:
                    print(f"[UDP] Error: {e}")
                self.packets_dropped += 1

    def send(self, force_chunk, accel_chunk):
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
        self.setWindowTitle("🔬 Caracterización Sensor de Fuerza DYMH-105 (ADS1219 + Pi Pico)")
        self.setGeometry(100, 100, 1600, 900)

        # Estado
        self.acquiring = False
        self.force_thread = None
        self.accel_thread = None

        # Buffers de datos
        self.force_buf_size = int(FORCE_SAMPLE_RATE * 2)   # 2 s de fuerza
        self.accel_buf_size = int(ACCEL_SAMPLE_RATE * 2)   # 2 s de aceleración
        self.force_buffer = deque(maxlen=self.force_buf_size)
        self.accel_buffer_0 = deque(maxlen=self.accel_buf_size)  # ai0: bancada

        # Datos acumulados para análisis / guardado
        self.all_force_data = []
        self.all_accel_data_0 = []

        # Medición de la tasa REAL de fuerza (timestamps del productor)
        self._force_rec_t0 = None      # perf_counter del último sample del 1er batch grabado
        self._force_rec_n0 = 0         # nº de muestras del 1er batch (excluidas del conteo)
        self._force_rec_count = 0      # muestras grabadas acumuladas
        self._force_rec_tlast = None   # perf_counter del último batch grabado
        # Anclas del RELOJ DEL PICO (cristal, t_us por bloque) → fs de fuerza exacto al guardar
        self._force_pico_seq0 = None
        self._force_pico_t0 = None
        self._force_pico_seq_last = None
        self._force_pico_t_last = None

        self.experimentos = []

        self.experiment_mode = "SWEEP"

        self.displacement_buffer = deque(maxlen=self.accel_buf_size)

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)

        self.udp_sender = None

        self.apply_dark_theme()
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

        config_layout.addWidget(QLabel("Masa de prueba (kg):"), 0, 0)
        self.masa_spin = QDoubleSpinBox()
        self.masa_spin.setRange(0.01, 100)
        self.masa_spin.setValue(MASA_PRUEBA_KG)
        self.masa_spin.setDecimals(3)
        config_layout.addWidget(self.masa_spin, 0, 1)

        config_layout.addWidget(QLabel("Frecuencia excitación (Hz):"), 0, 2)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.1, 1000)
        self.freq_spin.setValue(40)
        self.freq_spin.setDecimals(2)
        config_layout.addWidget(self.freq_spin, 0, 3)

        self.is_recording = False

        config_layout.addWidget(QLabel("Tipo Experimento:"), 1, 0)
        self.exp_type_combo = QComboBox()
        self.exp_type_combo.addItems(["🔄 Barrido/Chirp (Inercia)", "📐 Triangular (Fricción)", "🔪 Fuerza de Corte (Histéresis)", "📡 Solo Acelerómetro"])
        self.exp_type_combo.currentIndexChanged.connect(self._on_exp_type_changed)
        config_layout.addWidget(self.exp_type_combo, 1, 1)

        config_layout.addWidget(QLabel("Velocidad objetivo (mm/s):"), 1, 2)
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(0.1, 1000)
        self.velocity_spin.setValue(10.0)
        self.velocity_spin.setDecimals(1)
        self.velocity_spin.setEnabled(False)
        config_layout.addWidget(self.velocity_spin, 1, 3)

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

        config_layout.addWidget(QLabel("🔪 Condición cortador:"), 2, 4)
        self.cutter_condition_combo = QComboBox()
        self.cutter_condition_combo.addItems(["Nuevo", "Medio uso", "Desgastado"])
        self.cutter_condition_combo.setToolTip("Estado del cortador: afecta la histéresis y el desgaste")
        self.cutter_condition_combo.setEnabled(False)
        config_layout.addWidget(self.cutter_condition_combo, 2, 5)

        config_layout.addWidget(QLabel("Notas:"), 1, 4)
        self.notas_edit = QLineEdit()
        self.notas_edit.setPlaceholderText("Descripción del experimento...")
        config_layout.addWidget(self.notas_edit, 1, 5)

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

        self.filter_enabled_cb = QCheckBox("Activar filtro")
        self.filter_enabled_cb.setChecked(False)
        self.filter_enabled_cb.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_enabled_cb)

        filter_layout.addWidget(QLabel("Tipo:"))
        self.filter_type_combo = QComboBox()
        self.filter_type_combo.addItems(["Savitzky-Golay", "Butterworth LP", "Media Móvil", "Mediana (spikes)", "Hampel (outliers)", "Auto (freq)", "Notch (elimina freq)"])
        self.filter_type_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_type_combo)

        filter_layout.addWidget(QLabel("Ventana:"))
        self.filter_window_spin = QSpinBox()
        self.filter_window_spin.setRange(5, 201)
        self.filter_window_spin.setValue(21)
        self.filter_window_spin.setSingleStep(2)
        self.filter_window_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_window_spin)

        filter_layout.addWidget(QLabel("Orden:"))
        self.filter_order_spin = QSpinBox()
        self.filter_order_spin.setRange(1, 7)
        self.filter_order_spin.setValue(3)
        self.filter_order_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_order_spin)

        filter_layout.addWidget(QLabel("Fc (Hz):"))
        self.filter_fc_spin = QDoubleSpinBox()
        self.filter_fc_spin.setRange(1, 1000)
        self.filter_fc_spin.setValue(50)
        self.filter_fc_spin.setSingleStep(1)
        self.filter_fc_spin.setDecimals(1)
        self.filter_fc_spin.valueChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_fc_spin)

        self.filter_snr_label = QLabel("SNR: --")
        self.filter_snr_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
        filter_layout.addWidget(self.filter_snr_label)

        filter_layout.addStretch()
        main_layout.addWidget(filter_group)

        # === PANEL DE INFORMACIÓN EN TIEMPO REAL ===
        info_group = QGroupBox("📊 Mediciones en Tiempo Real")
        info_layout = QGridLayout(info_group)

        info_layout.addWidget(QLabel("⚡ FUERZA:"), 0, 0)
        self.force_volt_label = QLabel("Voltaje: 0.000 V")
        self.force_kg_label = QLabel("Fuerza: 0.000 kg")
        self.force_N_label = QLabel("Fuerza: 0.000 N")
        self.force_rms_label = QLabel("RMS: 0.000 V")
        self.force_fs_label = QLabel(f"fs_real: -- Hz (obj {FORCE_SAMPLE_RATE})")
        self.force_fs_label.setStyleSheet("color: #f1c40f; font-weight: bold;")
        info_layout.addWidget(self.force_volt_label, 0, 1)
        info_layout.addWidget(self.force_kg_label, 0, 2)
        info_layout.addWidget(self.force_N_label, 0, 3)
        info_layout.addWidget(self.force_rms_label, 0, 4)
        info_layout.addWidget(self.force_fs_label, 0, 5)

        info_layout.addWidget(QLabel("📡 ACEL BANCADA (ai0):"), 1, 0)
        self.accel0_g_label = QLabel("Acel: 0.000 g")
        self.accel0_rms_label = QLabel("RMS: 0.000 g")
        info_layout.addWidget(self.accel0_g_label, 1, 1)
        info_layout.addWidget(self.accel0_rms_label, 1, 2)

        info_layout.addWidget(QLabel(" VALIDACIÓN F=ma:"), 3, 0)
        self.fma_force_label = QLabel("F medida: 0.000 N")
        self.fma_calc_label = QLabel("m×a: 0.000 N")
        self.fma_error_label = QLabel("Error: 0.0 %")
        self.fma_error_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.fma_force_label, 3, 1)
        info_layout.addWidget(self.fma_calc_label, 3, 2)
        info_layout.addWidget(self.fma_error_label, 3, 3)

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

        self._setup_time_tab()
        self._setup_fft_tab()
        self._setup_phase_tab()
        self._setup_friction_tab()
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

        self.status_label = QLabel("Listo - Configure el Keysight y presione Iniciar")
        self.statusBar().addWidget(self.status_label)

    def _setup_time_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Vertical)

        self.force_plot = pg.PlotWidget(title=f"⚡ Fuerza (Voltaje, ADS1219 @ {FORCE_SAMPLE_RATE} Hz UI)")
        self.force_plot.setLabel('left', 'Voltaje', 'V')
        self.force_plot.setLabel('bottom', 'Tiempo', 's')
        self.force_plot.showGrid(x=True, y=True, alpha=0.3)
        self.force_curve = self.force_plot.plot(pen=pg.mkPen('#3498db', width=2))
        splitter.addWidget(self.force_plot)

        self.accel_plot = pg.PlotWidget(title=f"📡 Aceleración (NI 9234 @ {ACCEL_SAMPLE_RATE} SPS)")
        self.accel_plot.setLabel('left', 'Aceleración', 'g')
        self.accel_plot.setLabel('bottom', 'Tiempo', 's')
        self.accel_plot.showGrid(x=True, y=True, alpha=0.3)
        self.accel_curve = self.accel_plot.plot(pen=pg.mkPen('#e74c3c', width=2), name='Bancada (ai0)')
        self.accel_plot.addLegend()
        splitter.addWidget(self.accel_plot)

        self.overlay_plot = pg.PlotWidget(title="🔀 Superposición (Normalizada, rejilla común)")
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
        tab = QWidget()
        layout = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Vertical)

        self.fft_force_plot = pg.PlotWidget(title=f"FFT Fuerza (Fs={FORCE_SAMPLE_RATE} Hz)")
        self.fft_force_plot.setLabel('left', 'Magnitud', 'dB')
        self.fft_force_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.fft_force_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_force_curve = self.fft_force_plot.plot(pen=pg.mkPen('#3498db', width=2))
        self.fft_force_label = pg.TextItem(anchor=(0, 1), color='#3498db')
        self.fft_force_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.fft_force_plot.addItem(self.fft_force_label)
        splitter.addWidget(self.fft_force_plot)

        self.fft_accel_plot = pg.PlotWidget(title=f"FFT Aceleración ai0 (N=4096, Fs={ACCEL_SAMPLE_RATE} Hz)")
        self.fft_accel_plot.setLabel('left', 'Amplitud', 'g')
        self.fft_accel_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.fft_accel_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fft_accel_plot.setXRange(0, ACCEL_SAMPLE_RATE / 2)
        self.fft_accel_plot.enableAutoRange(axis='y')
        self.fft_accel_curve = self.fft_accel_plot.plot(pen=pg.mkPen('#00FF88', width=1.2))
        self.fft_accel_peak_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.DashLine))
        self.fft_accel_plot.addItem(self.fft_accel_peak_line)
        self.fft_accel_label_0 = pg.TextItem(anchor=(0, 1), color='#FF4444')
        self.fft_accel_label_0.setFont(QFont('Arial', 12, QFont.Bold))
        self.fft_accel_plot.addItem(self.fft_accel_label_0)
        splitter.addWidget(self.fft_accel_plot)

        self.frf_plot = pg.PlotWidget(title="FRF: H(f) = Acel / Fuerza (rejilla común, hasta 500 Hz)")
        self.frf_plot.setLabel('left', '|H(f)|', 'dB')
        self.frf_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.frf_plot.showGrid(x=True, y=True, alpha=0.3)
        self.frf_curve = self.frf_plot.plot(pen=pg.mkPen('#2ecc71', width=2))
        splitter.addWidget(self.frf_plot)

        layout.addWidget(splitter)
        self.tabs.addTab(tab, "📊 FFT / FRF")

    def _setup_phase_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        self.lissajous_plot = pg.PlotWidget(title="Diagrama de Lissajous (Fuerza vs Aceleración)")
        self.lissajous_plot.setLabel('left', 'Aceleración', 'g')
        self.lissajous_plot.setLabel('bottom', 'Fuerza', 'V')
        self.lissajous_plot.showGrid(x=True, y=True, alpha=0.3)
        self.lissajous_curve = self.lissajous_plot.plot(pen=pg.mkPen('#9b59b6', width=2))
        layout.addWidget(self.lissajous_plot)

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
        tab = QWidget()
        layout = QHBoxLayout(tab)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.friction_plot = pg.PlotWidget(title="🧩 Histéresis: Fuerza vs Desplazamiento")
        self.friction_plot.setLabel('left', 'Fuerza', 'N')
        self.friction_plot.setLabel('bottom', 'Desplazamiento', 'mm')
        self.friction_plot.showGrid(x=True, y=True, alpha=0.3)
        self.friction_curve = self.friction_plot.plot(pen=pg.mkPen('#e74c3c', width=2))
        self.friction_zero_line = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('#666', width=1, style=Qt.DashLine))
        self.friction_plot.addItem(self.friction_zero_line)
        left_layout.addWidget(self.friction_plot)

        self.velocity_plot = pg.PlotWidget(title="Velocidad Estimada (integración de aceleración)")
        self.velocity_plot.setLabel('left', 'Velocidad', 'mm/s')
        self.velocity_plot.setLabel('bottom', 'Tiempo', 's')
        self.velocity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.velocity_curve = self.velocity_plot.plot(pen=pg.mkPen('#3498db', width=2))
        left_layout.addWidget(self.velocity_plot)

        layout.addWidget(left_widget, stretch=3)

        right_widget = QGroupBox("📊 Análisis de Fricción")
        right_layout = QVBoxLayout(right_widget)

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
        if index == 0:
            self.experiment_mode = "SWEEP"
            self.velocity_spin.setEnabled(False)
            self.rpm_husillo_spin.setEnabled(False)
            self.rpm_avance_spin.setEnabled(False)
            self.cutter_condition_combo.setEnabled(False)
            self.status_label.setText("Modo: Barrido de frecuencia (identificacion de inercia)")
        elif index == 1:
            self.experiment_mode = "TRIANGLE"
            self.velocity_spin.setEnabled(True)
            self.rpm_husillo_spin.setEnabled(False)
            self.rpm_avance_spin.setEnabled(False)
            self.cutter_condition_combo.setEnabled(False)
            self.status_label.setText("Modo: Onda triangular (caracterizacion de friccion)")
        elif index == 2:
            self.experiment_mode = "CUTTING"
            self.velocity_spin.setEnabled(False)
            self.rpm_husillo_spin.setEnabled(True)
            self.rpm_avance_spin.setEnabled(True)
            self.cutter_condition_combo.setEnabled(True)
            self.status_label.setText("🔪 Modo: FUERZA DE CORTE - usando solo acelerometro ai0 + fuerza ai0")
        else:
            self.experiment_mode = "ACCEL_ONLY"
            self.velocity_spin.setEnabled(False)
            self.rpm_husillo_spin.setEnabled(True)
            self.rpm_avance_spin.setEnabled(False)
            self.cutter_condition_combo.setEnabled(False)
            self.status_label.setText("📡 Modo: SOLO ACELERÓMETRO - captura vibración sin sensor de fuerza")

    def _on_filter_changed(self):
        filter_type = self.filter_type_combo.currentIndex()
        self.filter_window_spin.setEnabled(filter_type in [0, 2, 3, 4])
        self.filter_order_spin.setEnabled(filter_type in [0, 4])
        self.filter_fc_spin.setEnabled(filter_type in [1, 6])

        if filter_type == 4:
            self.filter_order_spin.setRange(1, 10)
            self.filter_order_spin.setValue(3)
        else:
            self.filter_order_spin.setRange(1, 7)

        if filter_type == 5:
            freq_exc = self.freq_spin.value()
            samples_per_period = FORCE_SAMPLE_RATE / freq_exc
            window = int(samples_per_period / 5)
            window = window if window % 2 == 1 else window + 1
            window = max(5, min(window, 101))
            self.filter_window_spin.setValue(window)
            self.filter_fc_spin.setValue(freq_exc * 3)

    def apply_filter(self, data, fs=None):
        """Aplica el filtro seleccionado."""
        if fs is None:
            fs = FORCE_SAMPLE_RATE
        if not self.filter_enabled_cb.isChecked() or len(data) < 10:
            return data, 0.0

        filter_type = self.filter_type_combo.currentIndex()
        try:
            if filter_type == 0 or filter_type == 5:
                window = self.filter_window_spin.value()
                order = self.filter_order_spin.value()
                window = window if window % 2 == 1 else window + 1
                window = max(order + 2, window)
                if window > len(data):
                    window = len(data) if len(data) % 2 == 1 else len(data) - 1
                filtered = savgol_filter(data, window, order)
            elif filter_type == 1:
                fc = self.filter_fc_spin.value()
                nyquist = fs / 2
                if fc >= nyquist:
                    fc = nyquist * 0.9
                b, a = signal.butter(4, fc / nyquist, btype='low')
                filtered = signal.filtfilt(b, a, data)
            elif filter_type == 2:
                window = self.filter_window_spin.value()
                kernel = np.ones(window) / window
                filtered = np.convolve(data, kernel, mode='same')
            elif filter_type == 3:
                window = self.filter_window_spin.value()
                window = window if window % 2 == 1 else window + 1
                filtered = signal.medfilt(data, kernel_size=window)
            elif filter_type == 4:
                window = self.filter_window_spin.value()
                threshold = self.filter_order_spin.value()
                filtered = self._hampel_filter(data, window, threshold)
            elif filter_type == 6:
                fc = self.filter_fc_spin.value()
                Q = 5
                nyquist = fs / 2
                if fc < nyquist:
                    filtered = data.copy()
                    for harmonic in [1, 2, 3]:
                        freq = fc * harmonic
                        if freq < nyquist:
                            b, a = signal.iirnotch(freq, Q, fs)
                            filtered = signal.filtfilt(b, a, filtered)
                else:
                    filtered = data
            else:
                filtered = data

            noise_before = np.std(np.diff(data))
            noise_after = np.std(np.diff(filtered))
            snr_improvement = 20 * np.log10(noise_before / noise_after) if noise_after > 1e-10 else 0
            return filtered, snr_improvement
        except Exception as e:
            print(f"Error en filtrado: {e}")
            return data, 0.0

    def _hampel_filter(self, data, window_size=11, n_sigma=3):
        filtered = data.copy()
        n = len(data)
        k = window_size // 2
        for i in range(k, n - k):
            window = data[i - k:i + k + 1]
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            sigma = 1.4826 * mad
            if sigma > 1e-10 and np.abs(data[i] - median) > n_sigma * sigma:
                filtered[i] = median
        return filtered

    def _setup_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Frecuencia (Hz)", "Masa (kg)", "F_rms (N)", "a_rms (m/s²)",
            "m×a (N)", "Error F=ma (%)", "Fase (°)", "Notas"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.results_table)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setFont(QFont("Consolas", 10))
        self.summary_text.setMaximumHeight(150)
        layout.addWidget(self.summary_text)

        self.tabs.addTab(tab, "📋 Resultados")

    # ---------------------------------------------------------------
    #  Helper: ventana común
    # ---------------------------------------------------------------
    def _common_window(self, t_window=1.0):
        """Devuelve (force, accel0) sobre los últimos t_window segundos,
        usando muestras directas sin remuestreo."""
        nf = int(t_window * FORCE_SAMPLE_RATE)
        na = int(t_window * ACCEL_SAMPLE_RATE)
        if nf < 8 or len(self.force_buffer) < nf or len(self.accel_buffer_0) < na:
            return None
        f = np.array(self.force_buffer)[-nf:]
        a0 = np.array(self.accel_buffer_0)[-na:]
        n = min(len(f), len(a0))
        return f[-n:], a0[-n:]

    def toggle_acquisition(self):
        if not self.acquiring:
            self.start_acquisition()
        else:
            self.stop_acquisition()

    def start_acquisition(self):
        self.force_buffer.clear()
        self.accel_buffer_0.clear()
        self.displacement_buffer.clear()
        self.all_force_data.clear()
        self.all_accel_data_0.clear()

        while not force_queue.empty():
            force_queue.get()
        while not accel_queue.empty():
            accel_queue.get()

        if self.experiment_mode != "ACCEL_ONLY":
            self.force_thread = ForceAcquisitionThread()
            self.force_thread.start()
        else:
            self.force_thread = None
        self.accel_thread = AccelAcquisitionThread()
        self.accel_thread.start()

        self.update_timer.start(50)

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

        if self.udp_sender:
            self.udp_sender.stop()
            self.udp_sender.join(timeout=2)
            self.udp_sender = None

        self.start_btn.setText("▶️ Iniciar Adquisición")
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-size: 14px; font-weight: bold; padding: 10px;")

        n_force = len(self.all_force_data)
        n_accel = len(self.all_accel_data_0)
        self.status_label.setText(f"⏸️ Detenido | Fuerza: {n_force} muestras @{FORCE_SAMPLE_RATE} | Aceleración: {n_accel} muestras @{ACCEL_SAMPLE_RATE}")

    def update_plots(self):
        # --- vaciar colas hacia buffers ---
        last_force_chunk = None
        if self.experiment_mode != "ACCEL_ONLY":
            while not force_queue.empty():
                try:
                    item = force_queue.get_nowait()
                    arr, t_batch = item[0], item[1]
                    seq = item[2] if len(item) > 2 else None      # índice 1ª muestra del bloque (Pico)
                    t_us = item[3] if len(item) > 3 else None      # micros del Pico (cristal)
                    self.force_buffer.extend(arr)
                    if self.is_recording:
                        self.all_force_data.extend(arr)
                        n = len(arr)
                        if self._force_rec_t0 is None:
                            self._force_rec_t0 = t_batch
                            self._force_rec_n0 = n
                            self._force_rec_count = n
                        else:
                            self._force_rec_count += n
                        self._force_rec_tlast = t_batch
                        # Anclas del reloj del Pico (cristal) para fs exacto
                        if seq is not None and t_us is not None:
                            if self._force_pico_seq0 is None:
                                self._force_pico_seq0 = seq
                                self._force_pico_t0 = t_us
                            self._force_pico_seq_last = seq
                            self._force_pico_t_last = t_us
                    last_force_chunk = arr
                except Exception:
                    break

        last_accel_chunk = None
        while not accel_queue.empty():
            try:
                data = accel_queue.get_nowait()
                if data.ndim == 2:
                    accel_chunk = np.asarray(data[0], dtype=float).reshape(-1)
                else:
                    accel_chunk = np.asarray(data, dtype=float).reshape(-1)

                self.accel_buffer_0.extend(accel_chunk)
                if self.is_recording:
                    self.all_accel_data_0.extend(accel_chunk)
                last_accel_chunk = accel_chunk
            except Exception:
                break

        if self.udp_sender and last_force_chunk is not None and last_accel_chunk is not None:
            self.udp_sender.send(last_force_chunk, last_accel_chunk)

        # Etiqueta de tasa REAL de fuerza (medida con timestamps del productor)
        if self.experiment_mode != "ACCEL_ONLY":
            if self.is_recording and self._force_rec_count > self._force_rec_n0:
                self.force_fs_label.setText(f"fs_real: {self._force_fs_eff():.0f} Hz")
            else:
                self.force_fs_label.setText(f"fs_real: -- Hz (obj {FORCE_SAMPLE_RATE})")

        # ── Modo ACCEL_ONLY: solo graficar acelerómetro ──
        if self.experiment_mode == "ACCEL_ONLY":
            if len(self.accel_buffer_0) > 10:
                accel_arr_0_raw = np.array(self.accel_buffer_0)
                accel_arr_0, snr_a0 = self.apply_filter(accel_arr_0_raw, fs=ACCEL_SAMPLE_RATE)
                n = len(accel_arr_0)
                t = np.linspace(-n / ACCEL_SAMPLE_RATE, 0, n)

                self.accel_curve.setData(t, accel_arr_0)
                self.force_curve.setData([], [])
                self.overlay_force_curve.setData([], [])
                self.overlay_accel_curve.setData(t, accel_arr_0)

                accel0_g = accel_arr_0[-1]
                accel0_rms = np.sqrt(np.mean(accel_arr_0**2))
                self.accel0_g_label.setText(f"Acel: {accel0_g:.4f} g")
                self.accel0_rms_label.setText(f"RMS: {accel0_rms:.4f} g")
                rpm = self.rpm_husillo_spin.value()
                self.force_volt_label.setText(f"RPM: {rpm}")
                self.force_kg_label.setText("N/A (solo acel)")
                self.force_N_label.setText("N/A (solo acel)")
                self.force_rms_label.setText("N/A (solo acel)")

                if self.filter_enabled_cb.isChecked():
                    self.filter_snr_label.setText(f"SNR: +{snr_a0:.1f} dB")
                else:
                    self.filter_snr_label.setText("SNR: --")

                NFFT = 4096
                if len(self.all_accel_data_0) % 1000 < 200 and n >= NFFT:
                    data_fft = accel_arr_0[-NFFT:]
                    data_fft = data_fft - np.mean(data_fft)
                    window = np.hanning(NFFT)
                    yf = fft(data_fft * window)
                    xf = fftfreq(NFFT, 1.0 / ACCEL_SAMPLE_RATE)[:NFFT // 2]
                    mag = 2.0 / NFFT * np.abs(yf[:NFFT // 2])
                    mag = mag / np.mean(window) * 2
                    mask = xf > 2
                    self.fft_accel_curve.setData(xf[mask], mag[mask])
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

        # ── Gráficas en tiempo ──
        snr_f = snr_a0 = 0.0
        if len(self.force_buffer) > 10:
            f_full = np.array(self.force_buffer)
            f_full, snr_f = self.apply_filter(f_full, fs=FORCE_SAMPLE_RATE)
            tf = np.linspace(-len(f_full) / FORCE_SAMPLE_RATE, 0, len(f_full))
            self.force_curve.setData(tf, f_full)

        if len(self.accel_buffer_0) > 10:
            a0_full = np.array(self.accel_buffer_0)
            a0_full, snr_a0 = self.apply_filter(a0_full, fs=ACCEL_SAMPLE_RATE)
            ta = np.linspace(-len(a0_full) / ACCEL_SAMPLE_RATE, 0, len(a0_full))
            self.accel_curve.setData(ta, a0_full)

        if self.filter_enabled_cb.isChecked():
            avg_snr = (snr_f + snr_a0) / 2
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

        # Labels (RMS / último valor) sobre datos NATIVOS (rate-independiente)
        if len(self.force_buffer) > 10 and len(self.accel_buffer_0) > 10:
            f_native = np.array(self.force_buffer)
            a0_native = np.array(self.accel_buffer_0)
            self._update_info_labels(f_native, a0_native)

        # ── Análisis cruzado en ventana común sin remuestreo ──
        cw = self._common_window(1.0)
        if cw is None:
            return
        f, a0r = cw
        f, _ = self.apply_filter(f, fs=FORCE_SAMPLE_RATE)
        a0r, _ = self.apply_filter(a0r, fs=FORCE_SAMPLE_RATE)
        accel_for_analysis = a0r
        tc = np.linspace(-len(f) / FORCE_SAMPLE_RATE, 0, len(f))

        if np.std(f) > 1e-6 and np.std(accel_for_analysis) > 1e-6:
            self.overlay_force_curve.setData(tc, (f - np.mean(f)) / np.std(f))
            self.overlay_accel_curve.setData(tc, (accel_for_analysis - np.mean(accel_for_analysis)) / np.std(accel_for_analysis))

        self.lissajous_curve.setData(f, accel_for_analysis)

        if len(self.all_force_data) % 1000 < 100:
            self._update_fft(np.array(self.force_buffer), np.array(self.accel_buffer_0), f, a0r)

        self._update_friction_analysis(f, accel_for_analysis, tc)

    def _update_info_labels(self, force_arr, accel_arr_0):
        """Etiquetas de información. RMS y último valor son rate-independientes."""
        force_volt = force_arr[-1]
        force_kg = voltaje_a_fuerza_kg(force_volt)
        force_N = voltaje_a_fuerza_N(force_volt)
        force_rms = np.sqrt(np.mean(force_arr**2))

        self.force_volt_label.setText(f"Voltaje: {force_volt:.4f} V")
        self.force_kg_label.setText(f"Fuerza: {force_kg:.3f} kg")
        self.force_N_label.setText(f"Fuerza: {force_N:.3f} N")
        self.force_rms_label.setText(f"RMS: {force_rms:.4f} V")

        accel0_g = accel_arr_0[-1]
        accel0_rms = np.sqrt(np.mean(accel_arr_0**2))
        self.accel0_g_label.setText(f"Acel: {accel0_g:.4f} g")
        self.accel0_rms_label.setText(f"RMS: {accel0_rms:.4f} g")

        masa = self.masa_spin.value()
        F_medida = voltaje_a_fuerza_N(np.sqrt(np.mean(force_arr**2)))
        a_rms_ms2 = accel0_rms * 9.81
        F_calculada = masa * a_rms_ms2

        error_pct = abs(F_medida - F_calculada) / F_medida * 100 if F_medida > 0.001 else 0

        self.fma_force_label.setText(f"F medida: {F_medida:.4f} N")
        self.fma_calc_label.setText(f"m×a: {F_calculada:.4f} N")

        if error_pct < 10:
            color = "#27ae60"
        elif error_pct < 25:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        self.fma_error_label.setText(f"Error: {error_pct:.1f} %")
        self.fma_error_label.setStyleSheet(f"font-weight: bold; color: {color};")

    def _update_fft(self, force_native, accel0_native, f_common, a0_common):
        """FFT de aceleración/fuerza y FRF con ambos canales a la misma tasa."""
        # --- FFT Aceleración (tasa nativa) ---
        NFFT = 4096
        if len(accel0_native) >= NFFT:
            data_fft = accel0_native[-NFFT:] - np.mean(accel0_native[-NFFT:])
            window = np.hanning(NFFT)
            yf = fft(data_fft * window)
            xf = fftfreq(NFFT, 1.0 / ACCEL_SAMPLE_RATE)[:NFFT // 2]
            mag = 2.0 / NFFT * np.abs(yf[:NFFT // 2])
            mag = mag / np.mean(window) * 2
            mask = xf > 2
            self.fft_accel_curve.setData(xf[mask], mag[mask])
            if np.any(mask) and np.max(mag[mask]) > 0:
                idx = np.argmax(mag[mask])
                fp = xf[mask][idx]; ap = mag[mask][idx]
                self.fft_accel_peak_line.setPos(fp)
                self.fft_accel_label_0.setText(f"Pico: {fp:.1f} Hz ({ap:.4f} g)")
                self.fft_accel_label_0.setPos(fp, ap)
                self.fft_accel_plot.setTitle(f"FFT Aceleración ai0 — PICO: {fp:.1f} Hz ({ap:.4f} g)")

        # --- FFT Fuerza (tasa nativa de fuerza) ---
        n_f = len(force_native)
        if n_f >= 256:
            freqs_f = fftfreq(n_f, 1 / FORCE_SAMPLE_RATE)[:n_f // 2]
            fft_force = np.abs(fft(force_native - np.mean(force_native)))[:n_f // 2]
            fft_force_db = 20 * np.log10(fft_force + 1e-12)
            self.fft_force_curve.setData(freqs_f, fft_force_db)
            mask_f = freqs_f > 5
            if np.any(mask_f):
                idx_f = np.argmax(fft_force_db[mask_f])
                fpf = freqs_f[mask_f][idx_f]; mpf = fft_force_db[mask_f][idx_f]
                self.fft_force_label.setText(f"PICO: {fpf:.1f} Hz")
                self.fft_force_label.setPos(fpf, mpf)

        # --- FRF en ventana común (ambos a FORCE_SAMPLE_RATE) ---
        n_c = min(len(f_common), len(a0_common))
        if n_c >= 256:
            freqs = fftfreq(n_c, 1 / FORCE_SAMPLE_RATE)[:n_c // 2]
            Ff = np.abs(fft(f_common[:n_c] - np.mean(f_common[:n_c])))[:n_c // 2]
            Fa = np.abs(fft(a0_common[:n_c] - np.mean(a0_common[:n_c])))[:n_c // 2]
            H = Fa / (Ff + 1e-12)
            self.frf_curve.setData(freqs, 20 * np.log10(np.abs(H) + 1e-12))

    def _update_friction_analysis(self, force_arr, accel_arr, t_arr):
        """force_arr y accel_arr llegan a la misma tasa (FORCE_SAMPLE_RATE),
        así que dt = 1/FORCE_SAMPLE_RATE es correcto para integrar la aceleración."""
        n = len(accel_arr)
        if n < 100:
            return

        dt = 1.0 / FORCE_SAMPLE_RATE

        accel_g = accel_arr - np.mean(accel_arr)
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

        self.friction_curve.setData(displacement_mm, force_N)
        self.velocity_curve.setData(t_arr, velocity_mms)

        F_max = np.max(force_N)
        F_min = np.min(force_N)
        F_friction_peak = (F_max - F_min) / 2

        x_max = np.max(displacement_mm)
        x_min = np.min(displacement_mm)
        stroke = x_max - x_min
        v_mean = np.mean(np.abs(velocity_mms))

        F_friction_zc = 0
        F_forward = 0
        F_backward = 0
        n_forward = 0
        n_backward = 0

        try:
            x_threshold = stroke * 0.10 if stroke > 0.001 else 0.01
            near_zero_mask = np.abs(displacement_mm) < x_threshold
            if np.sum(near_zero_mask) > 5:
                forward_mask = near_zero_mask & (velocity_mms > 0)
                backward_mask = near_zero_mask & (velocity_mms < 0)
                if np.sum(forward_mask) > 2:
                    F_forward = np.mean(force_N[forward_mask]); n_forward = np.sum(forward_mask)
                if np.sum(backward_mask) > 2:
                    F_backward = np.mean(force_N[backward_mask]); n_backward = np.sum(backward_mask)
                if n_forward > 0 and n_backward > 0:
                    F_friction_zc = (F_forward - F_backward) / 2
        except:
            F_friction_zc = F_friction_peak

        if abs(F_friction_zc) > 0.001:
            F_friction_dyn = F_friction_zc; method_used = "ZC"
        else:
            F_friction_dyn = F_friction_peak; method_used = "P2P"

        try:
            energy_mJ = 0.5 * np.abs(np.sum(displacement_mm[:-1] * force_N[1:] -
                                            displacement_mm[1:] * force_N[:-1]))
        except:
            energy_mJ = 0

        self._last_friction_data = {
            'displacement_mm': displacement_mm, 'velocity_mms': velocity_mms,
            'force_N': force_N, 'F_friction_zc': F_friction_zc,
            'F_friction_peak': F_friction_peak, 'F_forward': F_forward,
            'F_backward': F_backward, 'n_forward': n_forward, 'n_backward': n_backward,
            'stroke': stroke, 'v_mean': v_mean, 'energy_mJ': energy_mJ
        }

        self.friction_fmax_label.setText(f"F_max: {F_max:.3f} N")
        self.friction_fmin_label.setText(f"F_min: {F_min:.3f} N")
        self.friction_fdyn_label.setText(f"F_fricción ({method_used}): {F_friction_dyn:.3f} N")
        self.friction_xmax_label.setText(f"x_max: {x_max:.3f} mm")
        self.friction_xmin_label.setText(f"x_min: {x_min:.3f} mm")
        self.friction_stroke_label.setText(f"Carrera: {stroke:.3f} mm")
        self.friction_velocity_label.setText(f"Velocidad media: {v_mean:.2f} mm/s")
        self.friction_energy_label.setText(f"Energía disipada: {energy_mJ:.3f} mJ")

        if hasattr(self, 'friction_fforward_label'):
            self.friction_fforward_label.setText(f"F_forward (x≈0): {F_forward:.3f} N (n={n_forward})")
            self.friction_fbackward_label.setText(f"F_backward (x≈0): {F_backward:.3f} N (n={n_backward})")

    def autoset_y(self):
        if len(self.force_buffer) > 10:
            force_arr = np.array(self.force_buffer)
            margin = (np.max(force_arr) - np.min(force_arr)) * 0.15 + 0.001
            self.force_plot.setYRange(np.min(force_arr) - margin, np.max(force_arr) + margin)

        if len(self.accel_buffer_0) > 10:
            accel_arr_0 = np.array(self.accel_buffer_0)
            margin = (np.max(accel_arr_0) - np.min(accel_arr_0)) * 0.15 + 0.001
            self.accel_plot.setYRange(np.min(accel_arr_0) - margin, np.max(accel_arr_0) + margin)

    def _clear_pending_queues(self):
        """Descarta muestras viejas que estaban esperando antes de grabar."""
        for q in (force_queue, accel_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break

    def toggle_recording(self):
        if not self.is_recording:
            self.all_force_data.clear()
            self.all_accel_data_0.clear()
            self._force_rec_t0 = None
            self._force_rec_n0 = 0
            self._force_rec_count = 0
            self._force_rec_tlast = None
            self._force_pico_seq0 = None
            self._force_pico_t0 = None
            self._force_pico_seq_last = None
            self._force_pico_t_last = None
            self._clear_pending_queues()
            self.is_recording = True
            self.capture_btn.setText("⏹ Detener Grabación")
            self.capture_btn.setStyleSheet("background-color: #c0392b; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
            self.status_label.setText("🔴 GRABANDO... (presiona Detener cuando termines)")
            self.recording_start_time = time.time()
        else:
            self.is_recording = False
            duration = time.time() - self.recording_start_time
            self.capture_btn.setText("🔴 Iniciar Grabación")
            self.capture_btn.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
            n_samples = len(self.all_accel_data_0) if self.experiment_mode == "ACCEL_ONLY" else len(self.all_force_data)
            self.status_label.setText(f"✅ Grabación detenida: {n_samples} muestras ({duration:.1f}s) - Presiona 'Guardar' para exportar")

    def capture_experiment(self):
        """Captura los datos actuales como un experimento."""
        if len(self.all_force_data) < 100 or len(self.all_accel_data_0) < 100:
            QMessageBox.warning(self, "Sin datos", "No hay suficientes datos para capturar.\nInicia grabación primero.")
            return

        force_arr = np.array(self.all_force_data)
        accel_arr_0 = np.array(self.all_accel_data_0)

        n_common = min(len(force_arr), len(accel_arr_0))
        force_common = force_arr[:n_common]
        accel_rs = accel_arr_0[:n_common]

        freq = self.freq_spin.value()
        masa = self.masa_spin.value()

        force_rms_V = np.sqrt(np.mean(force_common**2))
        force_rms_N = voltaje_a_fuerza_N(force_rms_V)

        accel_rms_g = np.sqrt(np.mean(accel_arr_0**2))
        accel_rms_ms2 = accel_rms_g * 9.81

        accel0_rms = np.sqrt(np.mean(accel_arr_0**2))

        F_calculada = masa * accel_rms_ms2
        error_pct = abs(force_rms_N - F_calculada) / force_rms_N * 100 if force_rms_N > 0.001 else 0

        # Fase sobre longitud común
        n = len(force_common)
        fft_force = fft(force_common - np.mean(force_common))[:n // 2]
        fft_accel = fft(accel_rs - np.mean(accel_rs))[:n // 2]
        idx_peak = np.argmax(np.abs(fft_force[1:])) + 1
        phase_diff = np.angle(fft_accel[idx_peak], deg=True) - np.angle(fft_force[idx_peak], deg=True)
        while phase_diff > 180:
            phase_diff -= 360
        while phase_diff < -180:
            phase_diff += 360

        F_friction_dyn = F_friction_zc = F_friction_peak = 0
        F_forward = F_backward = velocity_target = stroke = energy_mJ = v_mean = 0

        if self.experiment_mode == "TRIANGLE":
            velocity_target = self.velocity_spin.value()
            dt = 1.0 / FORCE_SAMPLE_RATE
            accel_g = accel_rs - np.mean(accel_rs)
            try:
                fc = 0.5
                b, a = signal.butter(2, fc / (FORCE_SAMPLE_RATE / 2), btype='high')
                accel_filtered = signal.filtfilt(b, a, accel_g)
            except:
                accel_filtered = accel_g
            accel_ms2 = accel_filtered * 9.81
            velocity = signal.detrend(cumulative_trapezoid(accel_ms2, dx=dt, initial=0))
            velocity_mms = velocity * 1000
            displacement = signal.detrend(cumulative_trapezoid(velocity, dx=dt, initial=0))
            displacement_mm = displacement * 1000

            force_N = np.array([voltaje_a_fuerza_N(v) for v in force_common])
            force_N = force_N - np.mean(force_N)
            # Igualar longitudes por seguridad, sin remuestreo
            m = min(len(force_N), len(displacement_mm), len(velocity_mms))
            force_N = force_N[:m]; displacement_mm = displacement_mm[:m]; velocity_mms = velocity_mms[:m]

            F_max = np.max(force_N); F_min = np.min(force_N)
            F_friction_peak = (F_max - F_min) / 2
            stroke = np.max(displacement_mm) - np.min(displacement_mm)
            v_mean = np.mean(np.abs(velocity_mms))
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
            F_friction_dyn = F_friction_zc if abs(F_friction_zc) > 0.001 else F_friction_peak
            try:
                energy_mJ = 0.5 * np.abs(np.sum(displacement_mm[:-1] * force_N[1:] -
                                                displacement_mm[1:] * force_N[:-1]))
            except:
                energy_mJ = 0

        rpm_husillo = self.rpm_husillo_spin.value() if self.experiment_mode == "CUTTING" else 0
        rpm_avance = self.rpm_avance_spin.value() if self.experiment_mode == "CUTTING" else 0
        cutter_condition = self.cutter_condition_combo.currentText() if self.experiment_mode == "CUTTING" else "N/A"

        exp = {
            'experiment_type': self.experiment_mode, 'frecuencia': freq, 'masa': masa,
            'force_rms_N': force_rms_N, 'accel_rms_ms2': accel_rms_ms2,
            'accel0_rms_g': accel0_rms, 'F_calculada': F_calculada,
            'error_pct': error_pct, 'fase': phase_diff,
            'F_friction_dyn': F_friction_dyn, 'F_friction_zc': F_friction_zc,
            'F_friction_peak': F_friction_peak, 'F_forward': F_forward, 'F_backward': F_backward,
            'velocity_target_mms': velocity_target, 'velocity_mean_mms': v_mean,
            'stroke_mm': stroke, 'energy_mJ': energy_mJ,
            'rpm_husillo': rpm_husillo, 'rpm_avance_x': rpm_avance, 'cutter_condition': cutter_condition,
            'notas': self.notas_edit.text(),
            'force_data': force_arr.copy(), 'accel_data_0': accel_arr_0.copy(),
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        self.experimentos.append(exp)

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

        self.all_force_data.clear()
        self.all_accel_data_0.clear()

        self.status_label.setText(f"✅ Experimento #{len(self.experimentos)} capturado @ {freq} Hz")
        self._update_summary()

    def _update_summary(self):
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
        if len(self.all_force_data) >= 100 or (self.experiment_mode == "ACCEL_ONLY" and len(self.all_accel_data_0) >= 100):
            self._save_raw_recording()
            return
        if not self.experimentos:
            QMessageBox.warning(self, "Sin datos", "No hay datos para guardar.\nInicia grabación primero.")
            return

    def _force_fs_eff(self):
        """Tasa REAL de fuerza medida con timestamps del productor (Hz).

        Cuenta las muestras posteriores al primer batch sobre el tiempo
        transcurrido entre el primer y el último batch grabados (estimador
        insesgado). Si no hay datos suficientes, devuelve FORCE_SAMPLE_RATE.
        """
        if (self._force_rec_t0 is not None and self._force_rec_tlast is not None
                and self._force_rec_tlast > self._force_rec_t0
                and self._force_rec_count > self._force_rec_n0):
            return (self._force_rec_count - self._force_rec_n0) / (self._force_rec_tlast - self._force_rec_t0)
        return float(FORCE_SAMPLE_RATE)

    def _force_fs_pico(self):
        """Tasa REAL de fuerza medida con el RELOJ DEL PICO (cristal, t_us por bloque).

        Más exacta que _force_fs_eff (que usa perf_counter del PC, con jitter de USB):
        usa el cristal del Pico (~±50 ppm). Devuelve None si no hay anclas suficientes.
        """
        if (self._force_pico_seq0 is not None and self._force_pico_seq_last is not None
                and self._force_pico_seq_last > self._force_pico_seq0):
            dt_us = (self._force_pico_t_last - self._force_pico_t0) & 0xFFFFFFFF
            if dt_us > 0:
                return (self._force_pico_seq_last - self._force_pico_seq0) / (dt_us * 1e-6)
        return None

    def _save_raw_recording(self):
        """Guarda grabación cruda: fuerza a su tasa REAL medida, aceleración a la del NI 9234."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rpm_husillo = self.rpm_husillo_spin.value()
        import json as _json

        # ── Modo ACCEL_ONLY: un solo archivo de aceleración ──
        if self.experiment_mode == "ACCEL_ONLY":
            fs_accel = float(getattr(self.accel_thread, "actual_fs", ACCEL_SAMPLE_RATE) or ACCEL_SAMPLE_RATE)
            accel_arr_0 = np.array(self.all_accel_data_0)
            n = len(accel_arr_0)
            t = np.arange(n) / fs_accel
            df = pd.DataFrame({'tiempo_s': t, 'aceleracion_sensor_g': accel_arr_0[:n]})
            acc_base = f"accel_{timestamp}_{rpm_husillo}rpm"
            out_dir = os.path.join(DATOS_DIR, acc_base)   # cada prueba en su propia carpeta
            os.makedirs(out_dir, exist_ok=True)
            filename = acc_base + ".csv"
            filepath = os.path.join(out_dir, filename)
            df.to_csv(filepath, index=False, sep='\t')
            sidecar = {
                'file': filename, 'timestamp': timestamp,
                'fs_hz': round(fs_accel, 4), 'fs_hz_accel': round(fs_accel, 4),
                'fs_hz_accel_nominal': ACCEL_SAMPLE_RATE,
                'accel_sensitivity_mV_g': ACEL_SENSIBILIDAD_MV_G,
                'accel_iepe_current_a': ACCEL_IEPE_CURRENT_A, 'accel_coupling': ACCEL_COUPLING,
                'n_samples': int(n), 'duration_s': round(float(n / fs_accel), 3),
                'experiment_mode': 'ACCEL_ONLY', 'rpm_husillo': rpm_husillo,
                'notas': self.notas_edit.text(), 'canal_aceleracion': 'ai0 bancada (NI 9234)',
            }
            json_path = filepath.replace('.csv', '_meta.json')
            with open(json_path, 'w', encoding='utf-8') as jf:
                _json.dump(sidecar, jf, indent=2, ensure_ascii=False)
            duration = n / fs_accel
            QMessageBox.information(self, "Guardado",
                f"Datos guardados (solo acel):\n{filepath}\n{json_path}\n\n"
                f"Muestras: {n} @ {fs_accel:.2f} Hz REAL\nDuracion: {duration:.2f}s\nRPM: {rpm_husillo}")
            self.status_label.setText(f"💾 Guardado en carpeta: {acc_base}\\")
            self.all_accel_data_0.clear()
            return

        # ── Modos con fuerza + acelerómetro ──
        force_arr = np.array(self.all_force_data)
        accel_arr_0 = np.array(self.all_accel_data_0)
        nf_raw = len(force_arr)
        na_raw = len(accel_arr_0)
        n_common = min(nf_raw, na_raw)

        if n_common < 100:
            QMessageBox.warning(
                self,
                "Sin datos suficientes",
                "No hay suficientes muestras comunes de fuerza y aceleración para guardar.\n"
                "Inicia Grabación y espera a que ambos canales estén corriendo."
            )
            return

        if nf_raw != na_raw:
            print(f"Fuerza {nf_raw} muestras @ su tasa y aceleración {na_raw} @ la suya "
                  f"(tasas distintas); se guardan SEPARADOS, cada uno con su base de tiempo.")

        # Cada canal se guarda POR SEPARADO con su PROPIA base de tiempo (SIN remuestrear).
        # Fuerza: reloj del PICO (cristal, t_us). Aceleración: samp_clk_rate real del 9234.
        fs_force = self._force_fs_pico()
        fs_source = "pico_clock"
        if fs_force is None:
            fs_force = self._force_fs_eff()
            fs_source = "pc_perf_counter"
        fs_accel = float(getattr(self.accel_thread, "actual_fs", ACCEL_SAMPLE_RATE) or ACCEL_SAMPLE_RATE)

        cutter_cond = self.cutter_condition_combo.currentText().lower().replace(" ", "_") if self.experiment_mode == "CUTTING" else "general"
        rpm_str = f"_{rpm_husillo}rpm" if self.experiment_mode == "CUTTING" else ""
        base = f"corte_{timestamp}{rpm_str}_{cutter_cond}"
        out_dir = os.path.join(DATOS_DIR, base)        # cada prueba en su propia carpeta
        os.makedirs(out_dir, exist_ok=True)

        # ── TXT FUERZA: su propia base de tiempo @ reloj del Pico (~990 Hz) ──
        nf = len(force_arr)
        tf = np.arange(nf) / fs_force
        force_file = base + "_fuerza.txt"
        force_path = os.path.join(out_dir, force_file)
        pd.DataFrame({'tiempo_s': tf, 'fuerza_V': force_arr}).to_csv(force_path, index=False, sep='\t')

        # ── TXT ACELERACIÓN: su propia base de tiempo @ tasa real del 9234 (2048 Hz) ──
        na = len(accel_arr_0)
        ta = np.arange(na) / fs_accel
        accel_file = base + "_accel.txt"
        accel_path = os.path.join(out_dir, accel_file)
        pd.DataFrame({'tiempo_s': ta, 'aceleracion_g': accel_arr_0}).to_csv(accel_path, index=False, sep='\t')

        # Sidecar: dos archivos INDEPENDIENTES, sin remuestreo
        sidecar = {
            'file_force': force_file,
            'file_accel': accel_file,
            'timestamp': timestamp,
            'fs_hz_force': round(float(fs_force), 4),            # tasa REAL de fuerza (reloj del Pico)
            'fs_hz_force_source': fs_source,                     # pico_clock | pc_perf_counter
            'fs_hz_force_nominal': FORCE_SAMPLE_RATE,
            'fs_hz_accel': round(float(fs_accel), 4),            # tasa REAL del 9234 (samp_clk_rate)
            'fs_hz_accel_nominal': ACCEL_SAMPLE_RATE,
            'sync_policy': 'archivos SEPARADOS sin remuestreo; cada canal con su propia base de tiempo',
            'n_samples_force': int(nf),
            'n_samples_accel': int(na),
            'duration_s_force': round(float(nf / fs_force), 3),
            'duration_s_accel': round(float(na / fs_accel), 3),
            'accel_sensitivity_mV_g': ACEL_SENSIBILIDAD_MV_G,
            'accel_iepe_current_a': ACCEL_IEPE_CURRENT_A,
            'accel_coupling': ACCEL_COUPLING,
            'experiment_mode': self.experiment_mode,
            'cutter_condition': self.cutter_condition_combo.currentText() if self.experiment_mode == "CUTTING" else "N/A",
            'rpm_husillo': rpm_husillo if self.experiment_mode == "CUTTING" else 0,
            'rpm_avance_x': self.rpm_avance_spin.value() if self.experiment_mode == "CUTTING" else 0,
            'notas': self.notas_edit.text(),
            'canal_fuerza': 'ADS1219 24-bit via Raspberry Pi Pico USB CDC (INA-4LC G=601, ADC G=%d)' % ADS1219_GAIN,
            'canal_aceleracion': 'ai0 bancada (NI 9234)',
        }
        json_path = os.path.join(out_dir, base + "_meta.json")
        with open(json_path, 'w', encoding='utf-8') as jf:
            _json.dump(sidecar, jf, indent=2, ensure_ascii=False)

        QMessageBox.information(self, "Guardado",
            f"Datos guardados (SEPARADOS, sin remuestreo):\n{force_path}\n{accel_path}\n{json_path}\n\n"
            f"Fuerza: {nf} muestras @ {fs_force:.2f} Hz REAL ({fs_source}) ({nf/fs_force:.2f}s)\n"
            f"Acel:   {na} muestras @ {fs_accel:.2f} Hz REAL (9234) ({na/fs_accel:.2f}s)")

        self.status_label.setText(f"💾 Guardado en carpeta: {base}\\ (fuerza.txt + accel.txt + meta, sin remuestreo)")

        self.all_force_data.clear()
        self.all_accel_data_0.clear()

    def closeEvent(self, event):
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
    print(f"   Fuerza: ADS1219 24-bit vía Raspberry Pi Pico (INA G=601, ADC G={ADS1219_GAIN}, {ADS1219_SPS} SPS)")
    print(f"           Puerto: {PICO_PORT or 'autodetect (VID 0x2E8A)'} | USB CDC binario | UI objetivo {FORCE_SAMPLE_RATE} Hz")
    print(f"   Acelerómetro: ai0 bancada (NI 9234, {ACCEL_SAMPLE_RATE} Hz, datos en g)")
    print(f"   Datos en: {DATOS_DIR}")
    print("="*60)
    print("\n🔌 CONEXIÓN DE LA FUERZA:")
    print("   celda DYMH-105 -> INA-4LC (G=601) -> ADS1219 AIN0/AIN1")
    print("   ADS1219 SDA->GP8, SCL->GP9, 3V3->3V3, GND->GND del Pico (pull-ups 10k)")
    print("   Requiere el Pico con firmware fuerza_pico_ads1219.uf2 y pyserial en el PC.")
    print("")
    print("⚠️ TASAS CONFIGURADAS:")
    print(f"   Fuerza {FORCE_SAMPLE_RATE} SPS (Nyquist ~{FORCE_SAMPLE_RATE//2} Hz)  |  Acel {ACCEL_SAMPLE_RATE} SPS")
    print("   Sin remuestreo: ambos canales se procesan con la tasa configurada.")
    print("   El guardado produce DOS CSV + un JSON sidecar que los enlaza.")
    print("")
    print("📋 INSTRUCCIONES:")
    print("   1. Selecciona el TIPO DE EXPERIMENTO")
    print("   2. Configura el Keysight a la frecuencia deseada")
    print("   3. Ajusta la ganancia del amplificador TIRA")
    print("   4. 'Iniciar Adquisición' -> 'Iniciar Grabación' -> 'Guardar Resultados'")
    print("="*60 + "\n")

    sys.exit(app.exec_())

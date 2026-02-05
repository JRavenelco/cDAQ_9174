import sys
import time
import queue
import threading
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtWidgets import QSizePolicy

# Importar nuestros módulos externos
from SVMWearPredictor import SVMWearPredictor
from CBR import CBRHysteresis
import feature_extraction as fe

import nidaqmx
from nidaqmx.constants import (
    AcquisitionType, Coupling, ExcitationSource,
    TerminalConfiguration
)
try:
    PSEUDO_DIFF = TerminalConfiguration.PSEUDODIFFERENTIAL
except AttributeError:
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
ACC_CONVERSION   = 10.0    # g/V

# Factor de amplificación (ej.: INA122 + LM358 amplifica por 10)
FORCE_AMPLIFICATION_FACTOR = 10.0
# Conversión efectiva considerando la amplificación
EFFECTIVE_FORCE_CONVERSION = FORCE_CONVERSION / FORCE_AMPLIFICATION_FACTOR

# --------------------------------------------------
# CONFIGURACIÓN DE ADQUISICIÓN
# --------------------------------------------------
DISPOSITIVO = "cDAQ1Mod1"            
CANALES = ["ai0", "ai1", "ai2", "ai3"]  # Por ejemplo: Fuerza X+, X-, Y+, Y-
SAMPLE_RATE = 10000
MUESTRAS_POR_BLOQUE = 1000
TIME_WINDOW = 0.05
VOLTAJE_MIN = -5.0
VOLTAJE_MAX = 5.0

VIB_DISPOSITIVO = "cDAQ1Mod2"        
VIB_CANALES = ["ai0", "ai1"]
VIB_SAMPLE_RATE = 20000
VIB_MUESTRAS_POR_BLOQUE = 1000

datos_queue = queue.Queue(maxsize=10)
vib_queue = queue.Queue(maxsize=10)
# Colas para el módulo de razonamiento inteligente
reasoning_input_queue = queue.Queue(maxsize=10)
reasoning_output_queue = queue.Queue(maxsize=10)

# --------------------------------------------------
# MODELO BOUC-WEN
# --------------------------------------------------
def bouc_wen_model(params, t, corriente):
    A, B, C, n, k = params
    N = len(t)
    fuerza_modelada = np.zeros(N)
    z_vals = np.zeros(N)
    z = 0.0
    z_min, z_max = -1e3, 1e3
    for i in range(1, N):
        dt = t[i] - t[i-1]
        du = (corriente[i] - corriente[i-1]) / dt if dt != 0 else 0.0
        delta_z = (A * du - B * abs(du) * z - C * du * (abs(z) ** n)) * dt
        z = z + delta_z
        z = np.clip(z, z_min, z_max)
        z_vals[i] = z
        fuerza_modelada[i] = k * (z ** 2)
    return fuerza_modelada, z_vals


def error_bouc_wen(params, t, corriente, fuerza_exp):
    fuerza_modelada, _ = bouc_wen_model(params, t, corriente)
    return fuerza_modelada - fuerza_exp

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
                result = self.reasoning_algorithm(features)
                self.output_queue.put(result)
                time.sleep(0.5)  # Simula tiempo de procesamiento
            except queue.Empty:
                continue

    def reasoning_algorithm(self, features):
        """
        Se simula el razonamiento basado en casos.
        Se ajusta la base de casos a una dimensión de 6 para que coincida con features.
        """
        case_base = {
            "Desgaste Bajo": np.array([0.1, 0.2, 0.1, 0.3, 0.1, 0.2]),
            "Desgaste Medio": np.array([0.5, 0.6, 0.5, 0.6, 0.5, 0.6]),
            "Desgaste Alto": np.array([0.9, 1.0, 0.9, 1.1, 0.9, 1.0])
        }
        min_distance = float('inf')
        best_match = None
        for label, case_vector in case_base.items():
            distance = np.linalg.norm(features - case_vector)
            if distance < min_distance:
                min_distance = distance
                best_match = label
        return best_match

    def stop(self):
        self.running = False

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
                    min_val=VOLTAJE_MIN,
                    max_val=VOLTAJE_MAX
                )
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10
            )
            self.task.start()
            while self.running:
                datos = self.task.read(
                    number_of_samples_per_channel=self.muestras_por_bloque,
                    timeout=1.0
                )
                datos_np = np.array(datos)
                if not datos_queue.full():
                    datos_queue.put_nowait(datos_np)
                else:
                    datos_queue.get_nowait()
                    datos_queue.put_nowait(datos_np)
        except Exception as ex:
            print("Error en AdquisicionThread:", ex)
        finally:
            if self.task:
                self.task.stop()
                self.task.close()

    def stop(self):
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
                        self.task.ai_channels.add_ai_accel_chan(
                            physical_channel=nombre_canal,
                            min_val=-50.0,
                            max_val=50.0,
                            sensitivity=100.0
                        )
                    except Exception as e:
                        print("FALLBACK a Voltage, error en add_ai_accel_chan:", e)
                        self._fallback_voltage_chan(nombre_canal)
                else:
                    self._fallback_voltage_chan(nombre_canal)
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10
            )
            self.task.start()
            while self.running:
                datos = self.task.read(
                    number_of_samples_per_channel=self.muestras_por_bloque,
                    timeout=1.0
                )
                datos_np = np.array(datos)
                if not vib_queue.full():
                    vib_queue.put_nowait(datos_np)
                else:
                    vib_queue.get_nowait()
                    vib_queue.put_nowait(datos_np)
        except Exception as ex:
            print("Error en VibrationAcquisitionThread:", ex)
        finally:
            if self.task:
                self.task.stop()
                self.task.close()

    def _fallback_voltage_chan(self, physical_channel):
        ch = self.task.ai_channels.add_ai_voltage_chan(
            physical_channel=physical_channel,
            min_val=-5.0,
            max_val=5.0
        )
        try:
            ch.ai_coupling = Coupling.AC
        except:
            pass
        try:
            ch.ai_excit_src = ExcitationSource.INTERNAL
            ch.ai_excit_val = 0.004
        except:
            pass
        try:
            ch.ai_iepe_enable = True
        except:
            pass

    def stop(self):
        self.running = False

# --------------------------------------------------
# CLASE PRINCIPAL DE LA INTERFAZ (DAQInterface)
# --------------------------------------------------
class PredictorHisteresis(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monitorización y Predicción de Desgaste de Herramienta")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.resize(1600, 900)

        self.acquiring = True
        self.sample_rate = SAMPLE_RATE
        self.time_window = TIME_WINDOW
        self.buffer_size = max(int(self.time_window * self.sample_rate), 2)
        self.terminal_config = TerminalConfiguration.DIFF

        self.all_data = [[] for _ in range(len(CANALES))]
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
        self.vib_buffer = [np.zeros(self.buffer_size) for _ in range(len(VIB_CANALES))]
        self.params_opt = [0.9, 0.4, 0.4, 2.1, 0.9]  # Para Bouc-Wen
        self._optimization_running = False
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Instanciar módulos SVM y CBR
        self.svm_predictor = SVMWearPredictor()
        self.cbr_analyzer = CBRHysteresis()

        central_widget = QtWidgets.QWidget()
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.tabs, stretch=1)

        # TAB 1: Módulo 9205 (Fuerza)
        self.tab_9205 = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_9205, "Módulo 9205")
        layout_9205 = QtWidgets.QVBoxLayout(self.tab_9205)
        layout_9205.setContentsMargins(0, 0, 0, 0)
        layout_9205.setSpacing(0)
        info_layout_9205 = QtWidgets.QHBoxLayout()
        layout_9205.addLayout(info_layout_9205)
        canal_labels = ["Fuerza X+", "Fuerza X-", "Fuerza Y+", "Fuerza Y-"]
        self.value_labels, self.freq_labels = [], []
        self.amp_labels, self.rms_labels = [], []
        for label_text in canal_labels:
            canal_layout = QtWidgets.QVBoxLayout()
            canal_label = QtWidgets.QLabel(label_text)
            canal_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
            canal_layout.addWidget(canal_label)
            v_label = QtWidgets.QLabel("Fuerza: 0.0 N")
            f_label = QtWidgets.QLabel("Frecuencia: 0.0 Hz")
            a_label = QtWidgets.QLabel("Amplitud: 0.0 N")
            r_label = QtWidgets.QLabel("RMS: 0.0 N")
            for lab in (v_label, f_label, a_label, r_label):
                lab.setStyleSheet("font-size: 12pt;")
                canal_layout.addWidget(lab)
            self.value_labels.append(v_label)
            self.freq_labels.append(f_label)
            self.amp_labels.append(a_label)
            self.rms_labels.append(r_label)
            info_layout_9205.addLayout(canal_layout)
        self.plot_widgets = []
        self.plot_curves = []
        colors = ['r', 'b', 'g', 'm']
        for i in range(len(CANALES)):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', canal_labels[i])
            w.setLabel('bottom', 'Tiempo (s)')
            w.showGrid(x=True, y=True)
            # Usar EFFECTIVE_FORCE_CONVERSION para ajustar la escala
            w.setYRange(VOLTAJE_MIN * EFFECTIVE_FORCE_CONVERSION, VOLTAJE_MAX * EFFECTIVE_FORCE_CONVERSION)
            pen = pg.mkPen(color=colors[i % 4], width=2)
            curve = w.plot(pen=pen)
            self.plot_widgets.append(w)
            self.plot_curves.append(curve)
            layout_9205.addWidget(w, stretch=1)

        # TAB 2: Módulo 9234 (Acelerómetro)
        self.tab_9234 = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_9234, "Módulo 9234")
        layout_9234 = QtWidgets.QVBoxLayout(self.tab_9234)
        layout_9234.setContentsMargins(0, 0, 0, 0)
        layout_9234.setSpacing(0)
        vib_info_layout = QtWidgets.QHBoxLayout()
        layout_9234.addLayout(vib_info_layout)
        vib_labels = ["Acel Canal ai0", "Acel Canal ai1"]
        self.vib_value_labels, self.vib_freq_labels = [], []
        self.vib_amp_labels, self.vib_rms_labels = [], []
        vib_colors = ['y', 'w']
        for label_text in vib_labels:
            canal_layout = QtWidgets.QVBoxLayout()
            canal_label = QtWidgets.QLabel(label_text)
            canal_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
            canal_layout.addWidget(canal_label)
            v_label = QtWidgets.QLabel("Aceleración: 0.0 g")
            f_label = QtWidgets.QLabel("Frecuencia: 0.0 Hz")
            a_label = QtWidgets.QLabel("Amplitud: 0.0 g")
            r_label = QtWidgets.QLabel("RMS: 0.0 g")
            for lab in (v_label, f_label, a_label, r_label):
                lab.setStyleSheet("font-size: 12pt;")
                canal_layout.addWidget(lab)
            self.vib_value_labels.append(v_label)
            self.vib_freq_labels.append(f_label)
            self.vib_amp_labels.append(a_label)
            self.vib_rms_labels.append(r_label)
            vib_info_layout.addLayout(canal_layout)
        self.vib_plot_widgets = []
        self.vib_plot_curves = []
        for i in range(len(VIB_CANALES)):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Aceleración (g)')
            w.setLabel('bottom', 'Tiempo (s)')
            w.showGrid(x=True, y=True)
            w.setYRange(VOLTAJE_MIN * ACC_CONVERSION, VOLTAJE_MAX * ACC_CONVERSION)
            pen = pg.mkPen(color=vib_colors[i % 2], width=2)
            curve = w.plot(pen=pen)
            self.vib_plot_widgets.append(w)
            self.vib_plot_curves.append(curve)
            layout_9234.addWidget(w, stretch=1)

        # TAB 3: Modelado Bouc-Wen
        self.tab_modelo = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_modelo, "Modelado Bouc-Wen")
        layout_modelo = QtWidgets.QVBoxLayout(self.tab_modelo)
        layout_modelo.setContentsMargins(0, 0, 0, 0)
        layout_modelo.setSpacing(0)
        self.phase2d_widget = pg.PlotWidget()
        self.phase2d_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.phase2d_widget.setLabel('left', "Canal1 (norm)")
        self.phase2d_widget.setLabel('bottom', "Canal0 (norm)")
        self.phase2d_widget.showGrid(x=True, y=True)
        self.phase2d_curve = self.phase2d_widget.plot(pen=pg.mkPen('c', width=2))
        layout_modelo.addWidget(self.phase2d_widget, stretch=1)
        self.glview = gl.GLViewWidget()
        self.glview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.glview.setCameraPosition(distance=10)
        grid = gl.GLGridItem()
        grid.setSize(10, 10, 1)
        self.glview.addItem(grid)
        self.phase3d_line = gl.GLLinePlotItem(color=(0,255,0,255), width=2, antialias=True)
        self.glview.addItem(self.phase3d_line)
        layout_modelo.addWidget(self.glview, stretch=1)
        self.bw_phase_widget = pg.PlotWidget()
        self.bw_phase_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bw_phase_widget.setLabel('left', 'z(t) (Bouc-Wen)')
        self.bw_phase_widget.setLabel('bottom', 'Corriente (N)')
        self.bw_phase_widget.showGrid(x=True, y=True)
        self.bw_phase_curve = self.bw_phase_widget.plot(pen=pg.mkPen('m', width=2))
        layout_modelo.addWidget(self.bw_phase_widget, stretch=1)
        optim_layout = QtWidgets.QHBoxLayout()
        self.optimizar_button = QtWidgets.QPushButton("Ajustar Modelo")
        self.optimizar_button.clicked.connect(self.ajustar_modelo)
        optim_layout.addWidget(self.optimizar_button)
        self.params_group = QtWidgets.QGroupBox("Parámetros Bouc-Wen Ajustados")
        form_layout = QtWidgets.QFormLayout()
        self.param_labels = {}
        for param in ["A", "B", "C", "n", "k"]:
            lbl = QtWidgets.QLabel("-")
            form_layout.addRow(f"{param}:", lbl)
            self.param_labels[param] = lbl
        self.params_group.setLayout(form_layout)
        optim_layout.addWidget(self.params_group)
        layout_modelo.addLayout(optim_layout)

        # TAB 4: FFT/Espectros
        self.tab_fft = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_fft, "Espectros FFT")
        layout_fft = QtWidgets.QVBoxLayout(self.tab_fft)
        layout_fft.setContentsMargins(0,0,0,0)
        layout_fft.setSpacing(0)
        self.fft_plots_9205 = []
        self.fft_curves_9205 = []
        c9205_colors = ['r','b','g','m']
        for i in range(len(CANALES)):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Magnitud')
            w.setLabel('bottom', 'Frecuencia (Hz)')
            w.showGrid(x=True, y=True)
            pen = pg.mkPen(color=c9205_colors[i % 4], width=2)
            curve = w.plot(pen=pen)
            self.fft_plots_9205.append(w)
            self.fft_curves_9205.append(curve)
            layout_fft.addWidget(w, stretch=1)
        self.fft_plots_9234 = []
        self.fft_curves_9234 = []
        vib_colors2 = ['y','w']
        for i in range(len(VIB_CANALES)):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Magnitud')
            w.setLabel('bottom', 'Frecuencia (Hz)')
            w.showGrid(x=True, y=True)
            pen = pg.mkPen(color=vib_colors2[i % 2], width=2)
            curve = w.plot(pen=pen)
            self.fft_plots_9234.append(w)
            self.fft_curves_9234.append(curve)
            layout_fft.addWidget(w, stretch=1)

        # TAB 5: Predicción de Desgaste (SVM)
        self.tab_svm = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_svm, "Predicción de Desgaste")
        layout_svm = QtWidgets.QVBoxLayout(self.tab_svm)
        layout_svm.setContentsMargins(0,0,0,0)
        layout_svm.setSpacing(0)
        self.svm_result_label = QtWidgets.QLabel("Estado de herramienta: -")
        self.svm_result_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout_svm.addWidget(self.svm_result_label)
        self.btn_prediccion = QtWidgets.QPushButton("Actualizar Predicción")
        self.btn_prediccion.clicked.connect(self.actualizar_prediccion)
        layout_svm.addWidget(self.btn_prediccion)

        # TAB 6: CBR Histeresis – Casos similares
        self.tab_cbr = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_cbr, "CBR Histeresis")
        layout_cbr = QtWidgets.QVBoxLayout(self.tab_cbr)
        layout_cbr.setContentsMargins(0,0,0,0)
        layout_cbr.setSpacing(0)
        self.cbr_result_label = QtWidgets.QLabel("Casos similares:")
        self.cbr_result_label.setStyleSheet("font-size: 14pt;")
        layout_cbr.addWidget(self.cbr_result_label)
        self.cbr_list = QtWidgets.QListWidget()
        layout_cbr.addWidget(self.cbr_list)

        # Nueva TAB: Razonamiento Inteligente
        self.tab_reasoning = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_reasoning, "Razonamiento Inteligente")
        layout_reasoning = QtWidgets.QVBoxLayout(self.tab_reasoning)
        layout_reasoning.setContentsMargins(0, 0, 0, 0)
        layout_reasoning.setSpacing(0)
        self.reasoning_plot = pg.PlotWidget()
        self.reasoning_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.reasoning_plot.setLabel('left', "Estado")
        self.reasoning_plot.setLabel('bottom', "Tiempo")
        self.reasoning_plot.showGrid(x=True, y=True)
        self.reasoning_plot.setYRange(0, 1)
        self.reasoning_curve = self.reasoning_plot.plot(pen=None, symbol='o', symbolSize=30)
        layout_reasoning.addWidget(self.reasoning_plot)
        self.reasoning_color_map = {
            "Desgaste Bajo": 'g',
            "Desgaste Medio": 'y',
            "Desgaste Alto": 'r'
        }
        self.reasoning_timer = QtCore.QTimer()
        self.reasoning_timer.timeout.connect(self.update_reasoning_visualization)
        self.reasoning_timer.start(500)

        main_layout.addWidget(self.tabs, stretch=1)

        # Controles inferiores
        control_layout = QtWidgets.QHBoxLayout()
        self.start_stop_button = QtWidgets.QPushButton("Detener")
        self.start_stop_button.clicked.connect(self.toggle_acquisition)
        control_layout.addWidget(self.start_stop_button)
        control_layout.addWidget(QtWidgets.QLabel("Tiempo:"))
        self.time_combo = QtWidgets.QComboBox()
        self.time_combo.addItems(["10 ms", "20 ms", "50 ms", "100 ms", "200 ms", "500 ms", "1 s"])
        self.time_combo.setCurrentText("50 ms")
        self.time_combo.currentTextChanged.connect(self.update_timebase)
        control_layout.addWidget(self.time_combo)
        control_layout.addWidget(QtWidgets.QLabel("Modo:"))
        self.terminal_mode_combo = QtWidgets.QComboBox()
        self.terminal_mode_combo.addItems(["Diferencial", "RSE", "NRSE"])
        self.terminal_mode_combo.setCurrentText("Diferencial")
        control_layout.addWidget(self.terminal_mode_combo)
        self.apply_mode_button = QtWidgets.QPushButton("Aplicar Modo")
        self.apply_mode_button.clicked.connect(self.apply_mode_changes)
        control_layout.addWidget(self.apply_mode_button)
        self.save_button = QtWidgets.QPushButton("Guardar Datos")
        self.save_button.clicked.connect(self.save_data)
        control_layout.addWidget(self.save_button)
        main_layout.addLayout(control_layout)

        # Iniciar hilos de adquisición
        self.adquisicion_thread = AdquisicionThread(
            dispositivo=DISPOSITIVO,
            canales=CANALES,
            sample_rate=SAMPLE_RATE,
            muestras_por_bloque=MUESTRAS_POR_BLOQUE,
            terminal_config=self.terminal_config
        )
        self.adquisicion_thread.start()
        self.vib_thread = VibrationAcquisitionThread(
            dispositivo=VIB_DISPOSITIVO,
            canales=VIB_CANALES,
            sample_rate=VIB_SAMPLE_RATE,
            muestras_por_bloque=VIB_MUESTRAS_POR_BLOQUE,
            modo_entrada="Accelerometer"
        )
        self.vib_thread.start()

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(20)

        self.opt_timer = QtCore.QTimer()
        self.opt_timer.timeout.connect(self.ejecutar_optimizacion)
        self.opt_timer.start(10000)

        # Iniciar el módulo de razonamiento inteligente en un hilo separado
        self.reasoning_system = IntelligentReasoningSystem(reasoning_input_queue, reasoning_output_queue)
        self.reasoning_system.start()

    def toggle_acquisition(self):
        if self.acquiring:
            self.acquiring = False
            self.start_stop_button.setText("Iniciar")
            self.adquisicion_thread.stop()
            self.vib_thread.stop()
        else:
            self.acquiring = True
            self.start_stop_button.setText("Detener")
            self.all_data = [[] for _ in range(len(CANALES))]
            self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
            self.vib_buffer = [np.zeros(self.buffer_size) for _ in range(len(VIB_CANALES))]
            self.adquisicion_thread = AdquisicionThread(
                dispositivo=DISPOSITIVO,
                canales=CANALES,
                sample_rate=SAMPLE_RATE,
                muestras_por_bloque=MUESTRAS_POR_BLOQUE,
                terminal_config=self.terminal_config
            )
            self.adquisicion_thread.start()
            self.vib_thread = VibrationAcquisitionThread(
                dispositivo=VIB_DISPOSITIVO,
                canales=VIB_CANALES,
                sample_rate=VIB_SAMPLE_RATE,
                muestras_por_bloque=VIB_MUESTRAS_POR_BLOQUE,
                modo_entrada="Accelerometer"
            )
            self.vib_thread.start()

    def update_timebase(self):
        time_text = self.time_combo.currentText()
        time_val = float(time_text.split()[0]) / 1000.0 if "ms" in time_text else float(time_text.split()[0])
        self.time_window = time_val
        self.buffer_size = max(int(self.time_window * self.sample_rate), 2)
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        for w in self.plot_widgets:
            w.setXRange(-self.time_window, 0)

    def apply_mode_changes(self):
        if self.acquiring:
            self.adquisicion_thread.stop()
            time.sleep(0.5)
        modo_text = self.terminal_mode_combo.currentText()
        if modo_text == "Diferencial":
            self.terminal_config = TerminalConfiguration.DIFF
        elif modo_text == "RSE":
            self.terminal_config = TerminalConfiguration.RSE
        elif modo_text == "NRSE":
            self.terminal_config = TerminalConfiguration.NRSE
        if self.acquiring:
            self.adquisicion_thread = AdquisicionThread(
                dispositivo=DISPOSITIVO,
                canales=CANALES,
                sample_rate=SAMPLE_RATE,
                muestras_por_bloque=MUESTRAS_POR_BLOQUE,
                terminal_config=self.terminal_config
            )
            self.adquisicion_thread.start()

    def update_plots(self):
        # Actualizar datos del módulo 9205 (Fuerza)
        while not datos_queue.empty():
            nuevos_datos = datos_queue.get_nowait()
            for i, canal_datos in enumerate(nuevos_datos):
                self.all_data[i].extend(canal_datos)
                n_new = len(canal_datos)
                if n_new >= self.buffer_size:
                    self.datos_buffer[i] = canal_datos[-self.buffer_size:]
                else:
                    self.datos_buffer[i][:-n_new] = self.datos_buffer[i][n_new:]
                    self.datos_buffer[i][-n_new:] = canal_datos

            for i in range(len(CANALES)):
                # Usar EFFECTIVE_FORCE_CONVERSION para calibrar la señal ya amplificada
                force_data = self.datos_buffer[i] * EFFECTIVE_FORCE_CONVERSION
                freq = medir_frecuencia_fft(force_data, SAMPLE_RATE)
                amp = np.max(force_data) - np.min(force_data)
                rms = np.sqrt(np.mean(force_data**2))

                self.value_labels[i].setText(f"Fuerza: {force_data[-1]:.1f} N")
                self.freq_labels[i].setText(f"Frecuencia: {freq:.1f} Hz")
                self.amp_labels[i].setText(f"Amplitud: {amp:.1f} N")
                self.rms_labels[i].setText(f"RMS: {rms:.1f} N")
                self.plot_curves[i].setData(self.tiempo, force_data)

        # Actualizar datos del módulo 9234 (Aceleración)
        while not vib_queue.empty():
            nuevos_datos_vib = vib_queue.get_nowait()
            for i in range(len(VIB_CANALES)):
                vib_canal = nuevos_datos_vib[i]
                n_new = len(vib_canal)
                if n_new >= self.buffer_size:
                    self.vib_buffer[i] = vib_canal[-self.buffer_size:]
                else:
                    self.vib_buffer[i][:-n_new] = self.vib_buffer[i][n_new:]
                    self.vib_buffer[i][-n_new:] = vib_canal

                accel_data = self.vib_buffer[i] * ACC_CONVERSION
                freq = medir_frecuencia_fft(accel_data, VIB_SAMPLE_RATE)
                amp = np.max(accel_data) - np.min(accel_data)
                rms = np.sqrt(np.mean(accel_data**2))

                self.vib_value_labels[i].setText(f"Aceleración: {accel_data[-1]:.2f} g")
                self.vib_freq_labels[i].setText(f"Frecuencia: {freq:.1f} Hz")
                self.vib_amp_labels[i].setText(f"Amplitud: {amp:.2f} g")
                self.vib_rms_labels[i].setText(f"RMS: {rms:.2f} g")
                self.vib_plot_curves[i].setData(self.tiempo, accel_data)

        # Actualizar Lissajous y 3D (Bouc-Wen)
        n_fase = 500
        if len(self.datos_buffer[0]) >= n_fase and len(self.datos_buffer[1]) >= n_fase:
            sig0 = self.datos_buffer[0][-n_fase:] * FORCE_CONVERSION
            sig1 = self.datos_buffer[1][-n_fase:] * FORCE_CONVERSION
            sig0n = normalize_signal(sig0)
            sig1n = normalize_signal(sig1)
            self.phase2d_curve.setData(sig0n, sig1n)

            time_axis = np.linspace(-1, 0, len(sig0n))
            pos = np.zeros((len(sig0n), 3), dtype=np.float32)
            pos[:, 0] = time_axis
            pos[:, 1] = sig0n
            pos[:, 2] = sig1n
            self.phase3d_line.setData(pos=pos)

            t_data_bw = np.linspace(0, (n_fase - 1)/SAMPLE_RATE, n_fase)
            corriente_segment = sig0
            _, z_vals = bouc_wen_model(self.params_opt, t_data_bw, corriente_segment)
            self.bw_phase_curve.setData(corriente_segment, z_vals)

        # Actualizar Tab FFT (9205)
        for i in range(len(CANALES)):
            force_data = self.datos_buffer[i] * FORCE_CONVERSION
            freqs, mag = fft_spectrum(force_data, SAMPLE_RATE)
            self.fft_curves_9205[i].setData(freqs, mag)
            self.fft_plots_9205[i].setXRange(0, SAMPLE_RATE/2)
        # Actualizar Tab FFT (9234)
        for i in range(len(VIB_CANALES)):
            accel_data = self.vib_buffer[i] * ACC_CONVERSION
            freqs, mag = fft_spectrum(accel_data, VIB_SAMPLE_RATE)
            self.fft_curves_9234[i].setData(freqs, mag)
            self.fft_plots_9234[i].setXRange(0, VIB_SAMPLE_RATE/2)

        # Extraer características y enviar al módulo de razonamiento inteligente
        if len(self.datos_buffer[0]) >= n_fase and len(self.vib_buffer[0]) >= n_fase:
            t_bw = np.linspace(0, (n_fase - 1)/SAMPLE_RATE, n_fase)
            bw_force, bw_hyst = bouc_wen_model(self.params_opt, t_bw, self.datos_buffer[0][-n_fase:] * FORCE_CONVERSION)
            force_features = fe.extract_force_features(bw_hyst, t_bw)
            vib_features = fe.extract_vibration_features(self.vib_buffer[0][-n_fase:] * ACC_CONVERSION, VIB_SAMPLE_RATE)
            combined_features = fe.fuse_features(force_features, vib_features)
            if reasoning_input_queue.full():
                try:
                    reasoning_input_queue.get_nowait()
                except queue.Empty:
                    pass
            reasoning_input_queue.put_nowait(combined_features)
            label, prob = self.svm_predictor.predict(combined_features)
            self.svm_result_label.setText(f"Estado de herramienta: {label} (Prob: {prob*100:.1f}%)")
            casos_similares = self.cbr_analyzer.retrieve(combined_features, threshold=100)
            self.cbr_list.clear()
            for case, dist in casos_similares:
                self.cbr_list.addItem(f"Label: {case['label']}, Dist: {dist:.2f}")

    def update_reasoning_visualization(self):
        """
        Consulta la cola de salida del razonamiento y actualiza la pestaña "Razonamiento Inteligente"
        mostrando un punto de color que representa el estado predicho.
        """
        if not reasoning_output_queue.empty():
            result = reasoning_output_queue.get_nowait()
            color = self.reasoning_color_map.get(result, 'w')
            self.reasoning_plot.clear()
            self.reasoning_plot.plot([0], [0.5], pen=None, symbol='o', symbolSize=30, symbolBrush=color)
            self.reasoning_plot.setTitle(f"Estado Predicho: {result}")

    def ajustar_modelo(self):
        if self._optimization_running:
            return
        N = min(3000, len(self.all_data[0]))
        if N < 100:
            QtWidgets.QMessageBox.warning(self, "Datos insuficientes", "No hay suficientes datos para optimizar.")
            return
        self._optimization_running = True
        self.optimizar_button.setEnabled(False)
        self.optimizar_button.setText("Optimizando...")
        t_data = np.linspace(0, N / SAMPLE_RATE, N)
        corriente_data = np.array(self.all_data[0][-N:]) * FORCE_CONVERSION
        fuerza_data = np.array(self.all_data[1][-N:]) * FORCE_CONVERSION
        params0 = [0.9, 0.4, 0.4, 2.1, 0.9]
        def optimize():
            try:
                i_smooth = savgol_filter(corriente_data, 15, 3)
                f_smooth = savgol_filter(fuerza_data, 15, 3)
                res = least_squares(error_bouc_wen, params0,
                                    args=(t_data, i_smooth, f_smooth),
                                    method='lm', ftol=1e-4, xtol=1e-4, max_nfev=100)
                return res.x, res.cost, t_data, corriente_data, fuerza_data
            except Exception as ex:
                print("Error en optimización:", ex)
                return None
        def finish_optimization(fut):
            try:
                result = fut.result()
                if result is None:
                    return
                params_opt, final_err, t_opt, i_opt, f_opt = result
                self.params_opt = params_opt
                for k, v in zip(["A","B","C","n","k"], params_opt):
                    self.param_labels[k].setText(f"{v:.4f}")
                self.mostrar_grafico_comparacion(params_opt, t_opt, i_opt, f_opt)
            finally:
                self._optimization_running = False
                self.optimizar_button.setEnabled(True)
                self.optimizar_button.setText("Ajustar Modelo")
        future = self.executor.submit(optimize)
        future.add_done_callback(lambda f: QtCore.QTimer.singleShot(0, lambda: finish_optimization(f)))

    def ejecutar_optimizacion(self):
        if not self._optimization_running:
            if len(self.all_data[0]) > 10000:
                for i in range(len(self.all_data)):
                    self.all_data[i] = self.all_data[i][-10000:]
            self.ajustar_modelo()

    def mostrar_grafico_comparacion(self, params_opt, t_data, corriente_data, fuerza_data):
        fuerza_modelada, _ = bouc_wen_model(params_opt, t_data, corriente_data)
        plt.figure()
        plt.plot(t_data, fuerza_data, label="Fuerza Experimental (N)")
        plt.plot(t_data, fuerza_modelada, label="Fuerza Modelada (N)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Fuerza (N)")
        plt.legend()
        plt.title("Comparación Fuerza Experimental vs. Modelo Bouc-Wen")
        plt.show(block=False)

    def save_data(self):
        if not self.all_data[0]:
            QtWidgets.QMessageBox.warning(self, "Sin datos", "No hay datos para guardar.")
            return
        filename = f"datos_predictor_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        datos_np = np.array([np.array(c) for c in self.all_data])
        np.savetxt(filename, datos_np.T, delimiter=",",
                   header=",".join([f"Canal_{c}" for c in CANALES]), comments="")
        QtWidgets.QMessageBox.information(self, "Datos Guardados", f"Datos guardados en '{filename}'.")

    def closeEvent(self, event):
        if self.adquisicion_thread and self.adquisicion_thread.is_alive():
            self.adquisicion_thread.stop()
            self.adquisicion_thread.join(timeout=1.0)
        if self.vib_thread and self.vib_thread.is_alive():
            self.vib_thread.stop()
            self.vib_thread.join(timeout=1.0)
        if self.reasoning_system and self.reasoning_system.is_alive():
            self.reasoning_system.stop()
            self.reasoning_system.join(timeout=1.0)
        super().closeEvent(event)
    
    def actualizar_prediccion(self):
        n_fase = 500
        if len(self.datos_buffer[0]) >= n_fase and len(self.vib_buffer[0]) >= n_fase:
            t_bw = np.linspace(0, (n_fase - 1)/SAMPLE_RATE, n_fase)
            bw_force, bw_hyst = bouc_wen_model(self.params_opt, t_bw, self.datos_buffer[0][-n_fase:] * FORCE_CONVERSION)
            from feature_extraction import extract_force_features, extract_vibration_features, fuse_features
            force_features = extract_force_features(bw_hyst, t_bw)
            vib_features = extract_vibration_features(self.vib_buffer[0][-n_fase:] * ACC_CONVERSION, VIB_SAMPLE_RATE)
            combined_features = fuse_features(force_features, vib_features)
            label, prob = self.svm_predictor.predict(combined_features)
            self.svm_result_label.setText(f"Estado de herramienta: {label} (Prob: {prob*100:.1f}%)")
            casos_similares = self.cbr_analyzer.retrieve(combined_features, threshold=100)
            self.cbr_list.clear()
            for case, dist in casos_similares:
                self.cbr_list.addItem(f"Label: {case['label']}, Dist: {dist:.2f}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    ventana = PredictorHisteresis()
    ventana.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

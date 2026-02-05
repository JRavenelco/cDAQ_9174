#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz combinada para adquisición de Vibración (NI 9234) + Fuerza (NI 9205)
- Basada en 'interfaz_DAQ_acelerometro.py' y reutilizando patrones de 'interfaz_DAQ_V2.py'
- Habilita/deshabilita canales individualmente
- Visualización en tiempo real con opción de bloquear eje Y (pan/zoom manual)
- Guardado de datos solo de canales habilitados (vibración y fuerza por separado)
"""

import sys
import time
import queue
import threading
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QSizePolicy, QGroupBox, QGridLayout, QCheckBox, QDoubleSpinBox,
    QComboBox, QSpinBox, QSplitter
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt
from datetime import datetime
import os
import csv
from collections import deque
import traceback
import re
try:
    import pywt  # Wavelet features (optional)
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, ExcitationSource, Coupling

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
# NI 9205 (Fuerza) - SOLO CANAL 0
FORCE_DISPOSITIVO = "cDAQ1Mod1"
FORCE_CANALES = ["ai0"]  # Solo fuerza canal 0
FORCE_SAMPLE_RATE = 2500  # Hz
FORCE_MUESTRAS_POR_BLOQUE = 100
FORCE_MIN_VOLTAGE = -1.5
FORCE_MAX_VOLTAGE = 1.5
FORCE_TERMINAL_CONFIG = TerminalConfiguration.DIFF

# NI 9234 (Vibración) - 2 CANALES
# ai0 = Acelerómetro en PRENSA (filtrado mecánico, envolvente)
# ai1 = Acelerómetro en PIEZA (directo, fuerza de corte)
VIB_DISPOSITIVO = "cDAQ1Mod2"
VIB_CANALES = ["ai0", "ai1"]  # Prensa + Pieza
VIB_SAMPLE_RATE = 2500  # Hz
VIB_MUESTRAS_POR_BLOQUE = 100
ACCEL_MIN_G = -0.2
ACCEL_MAX_G = 0.2
ACC_SENSITIVITY = 98.3  # mV/g (PCB 352C33)

# UI / Plot
TIME_WINDOW = 0.5  # segundos

# Guardado de datos
DATOS_DIR = "experimentos_mesa"
AUTO_SAVE_BASE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATOS_DIR)

# Colas
force_queue = queue.Queue(maxsize=10)
vib_queue = queue.Queue(maxsize=10)

# --------------------------------------------------
# HILOS DE ADQUISICIÓN
# --------------------------------------------------
class ForceAcquisitionThread(threading.Thread):
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
                samps_per_chan=self.muestras_por_bloque * 10
            )
            self.task.start()
            print(f"✅ Adquisición NI 9205 iniciada: {self.sample_rate} Hz, canales: {self.canales}")

            while self.running:
                try:
                    datos = self.task.read(
                        number_of_samples_per_channel=self.muestras_por_bloque,
                        timeout=2.0
                    )
                    datos_np = np.array(datos)
                    if not force_queue.full():
                        force_queue.put_nowait(datos_np)
                    else:
                        try:
                            force_queue.get_nowait()
                            force_queue.put_nowait(datos_np)
                        except queue.Empty:
                            force_queue.put_nowait(datos_np)
                except nidaqmx.errors.DaqReadError as e:
                    if e.error_code == -200279:
                        time.sleep(0.01)
                    else:
                        print(f"❌ Error DAQ 9205 no recuperable: {e}")
                        break
                except Exception as e:
                    print(f"❌ Error inesperado NI 9205: {e}")
                    break
        except Exception as ex:
            print(f"❌ Error inicializando NI 9205: {ex}")
            traceback.print_exc()
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                    print("✅ Tarea 9205 cerrada correctamente")
                except Exception as e:
                    print(f"Error cerrando tarea 9205: {e}")
            self.task = None

    def stop(self):
        print("🛑 Deteniendo adquisición 9205...")
        self.running = False


class VibrationAcquisitionThread(threading.Thread):
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
            for canal in self.canales:
                nombre_canal = f"{self.dispositivo}/{canal}"
                try:
                    self.task.ai_channels.add_ai_accel_chan(
                        physical_channel=nombre_canal,
                        name_to_assign_to_channel=f"Accel_{canal}",
                        terminal_config=TerminalConfiguration.DEFAULT,
                        min_val=ACCEL_MIN_G, max_val=ACCEL_MAX_G,
                        sensitivity=ACC_SENSITIVITY,
                        sensitivity_units=nidaqmx.constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                        current_excit_source=ExcitationSource.INTERNAL,
                        current_excit_val=0.004  # 4mA IEPE
                    )
                    print(f"✅ Configurado {nombre_canal} como acelerómetro")
                except Exception as accel_e:
                    print(f"⚠️ Fallo acelerómetro {nombre_canal}: {accel_e}")
                    self._configure_voltage_with_iepe(nombre_canal)

            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10
            )
            self.task.start()
            print(f"✅ Adquisición NI 9234 iniciada: {self.sample_rate} Hz")

            while self.running:
                try:
                    datos = self.task.read(
                        number_of_samples_per_channel=self.muestras_por_bloque,
                        timeout=2.0
                    )
                    datos_np = np.array(datos)
                    if not vib_queue.full():
                        vib_queue.put_nowait(datos_np)
                    else:
                        try:
                            vib_queue.get_nowait()
                            vib_queue.put_nowait(datos_np)
                        except queue.Empty:
                            vib_queue.put_nowait(datos_np)
                except nidaqmx.errors.DaqReadError as e:
                    if e.error_code == -200279:
                        time.sleep(0.01)
                    else:
                        print(f"❌ Error DAQ 9234 no recuperable: {e}")
                        break
                except Exception as e:
                    print(f"❌ Error inesperado NI 9234: {e}")
                    break
        except Exception as ex:
            print(f"❌ Error inicializando hilo 9234: {ex}")
            traceback.print_exc()
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                    print("✅ Tarea 9234 cerrada correctamente")
                except Exception as e:
                    print(f"Error cerrando tarea 9234: {e}")
            self.task = None

    def _configure_voltage_with_iepe(self, physical_channel):
        try:
            ch = self.task.ai_channels.add_ai_voltage_chan(
                physical_channel=physical_channel,
                name_to_assign_to_channel=f"VoltageIEPE_{physical_channel.split('/')[-1]}",
                terminal_config=TerminalConfiguration.PSEUDODIFFERENTIAL,
                min_val=-10.0, max_val=10.0
            )
            try:
                ch.ai_coupling = Coupling.AC
                ch.ai_exc_source = ExcitationSource.INTERNAL
                ch.ai_exc_val = 0.004
                if hasattr(ch, 'ai_iepe_enable'):
                    ch.ai_iepe_enable = True
                print(f"✅ Configurado {physical_channel} como voltaje IEPE")
            except Exception as e:
                print(f"⚠️ Configuración IEPE parcial: {e}")
        except Exception as e:
            print(f"❌ Fallo crítico configurando {physical_channel}: {e}")

    def stop(self):
        print("🛑 Deteniendo adquisición 9234...")
        self.running = False


# --------------------------------------------------
# VENTANA PRINCIPAL
# --------------------------------------------------
class CombinedDAQ(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔬 DAQ Multi-Módulo: Vibración + Fuerza")
        self.setGeometry(100, 100, 1600, 900)
        
        # Apply dark theme first
        self.apply_dark_theme()

        # Estado
        self.acquiring = False
        self.time_window = TIME_WINDOW
        self.force_sample_rate = FORCE_SAMPLE_RATE
        self.vib_sample_rate = VIB_SAMPLE_RATE

        # Buffers
        self.force_buffer_size = max(int(self.time_window * self.force_sample_rate), 2)
        self.vib_buffer_size = max(int(self.time_window * self.vib_sample_rate), 2)
        self.force_buffer = [deque(maxlen=self.force_buffer_size) for _ in range(len(FORCE_CANALES))]
        self.vib_buffer = [deque(maxlen=self.vib_buffer_size) for _ in range(len(VIB_CANALES))]
        self.all_force_data = [[] for _ in range(len(FORCE_CANALES))]
        self.all_vib_data = [[] for _ in range(len(VIB_CANALES))]

        # Espectrograma: parámetros y buffers
        self.spec_nfft = 256
        self.spec_hop = 128
        self.spec_max_cols = 200  # ~12.8s a 2kHz con hop=128
        self.spec_window = np.hanning(self.spec_nfft).astype(np.float32)
        self.spec_freq_bins = self.spec_nfft // 2 + 1
        self.spec_levels_min_db = -60.0
        self.spec_levels_max_db = 0.0
        self.spec_lut_name = 'viridis'
        # Buffers de imagen (dB normalizados 0..-inf)
        self.vib_spec_data = [np.zeros((self.spec_freq_bins, self.spec_max_cols), dtype=np.float32) for _ in range(len(VIB_CANALES))]
        self.force_spec_data = [np.zeros((self.spec_freq_bins, self.spec_max_cols), dtype=np.float32) for _ in range(len(FORCE_CANALES))]
        # Seguimiento de cuánto creció cada buffer para emitir nueva columna
        self.vib_spec_last_len = [0 for _ in range(len(VIB_CANALES))]
        self.force_spec_last_len = [0 for _ in range(len(FORCE_CANALES))]

        # Sesión
        self.current_session_base_filename = None
        self.session_start_time = None
        self.session_start_time_str = ""

        # Crear directorio
        if not os.path.exists(AUTO_SAVE_BASE_DIRECTORY):
            os.makedirs(AUTO_SAVE_BASE_DIRECTORY, exist_ok=True)
            print(f"✅ Directorio creado: {AUTO_SAVE_BASE_DIRECTORY}")

        # Hilos y timer
        self.force_thread = None
        self.vib_thread = None
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plots)

        self._setup_ui()
        self.start_stop_button.setText("Iniciar Adquisición")

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Título
        title = QLabel("📋 Adquisición Vibración (9234) + Fuerza (9205)")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        main_layout.addWidget(title)

        # --- METADATOS DE SESIÓN ---
        metadata_group = QGroupBox("📋 Metadatos de la Sesión")
        metadata_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        metadata_layout = QGridLayout(metadata_group)
        metadata_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        # Frecuencia de operación (Hz)
        metadata_layout.addWidget(QLabel("Frecuencia (Hz):"), 0, 0)
        self.frecuencia_spin = QDoubleSpinBox()
        self.frecuencia_spin.setRange(0.0, 10000.0)
        self.frecuencia_spin.setValue(0.0)
        self.frecuencia_spin.setSingleStep(0.1)
        self.frecuencia_spin.setDecimals(2)
        self.frecuencia_spin.setSuffix(" Hz")
        self.frecuencia_spin.setToolTip("Frecuencia de operación en Hz")
        metadata_layout.addWidget(self.frecuencia_spin, 0, 1)

        metadata_layout.addWidget(QLabel("Notas:"), 0, 2)
        self.notas_edit = QtWidgets.QLineEdit()
        self.notas_edit.setPlaceholderText("Observaciones opcionales...")
        metadata_layout.addWidget(self.notas_edit, 0, 3)

        main_layout.addWidget(metadata_group)

        # --- INFORMACIÓN DE VIBRACIÓN ---
        vib_info_group = QGroupBox("📡 Vibración (NI 9234)")
        vib_info_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        vib_info_layout = QGridLayout(vib_info_group)
        vib_info_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Info labels Vibración
        vib_info_layout.addWidget(QLabel("📊 Información de Vibración"), 0, 0, 1, 4)
        self.vib_value_labels = []
        self.vib_rms_labels = []
        self.vib_max_labels = []
        for i, canal in enumerate(VIB_CANALES):
            header = QLabel(f"Vib {canal}")
            header.setStyleSheet("font-size: 11pt; font-weight: bold;")
            vib_info_layout.addWidget(header, 1, i)
            v_label = QLabel("Valor: 0.00 g")
            r_label = QLabel("RMS: 0.00 g")
            m_label = QLabel("Max: 0.00 g")
            for row, lab in enumerate([v_label, r_label, m_label], start=2):
                lab.setStyleSheet("font-size: 10pt;")
                vib_info_layout.addWidget(lab, row, i)
            self.vib_value_labels.append(v_label)
            self.vib_rms_labels.append(r_label)
            self.vib_max_labels.append(m_label)
        vib_info_widget = QWidget()
        vib_info_widget.setLayout(vib_info_layout)
        vib_info_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        main_layout.addWidget(vib_info_widget)

        # --- INFORMACIÓN DE FUERZA ---
        force_info_group = QGroupBox("⚡ Fuerza (NI 9205)")
        force_info_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        force_info_layout = QGridLayout(force_info_group)
        force_info_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Info labels Fuerza
        force_info_layout.addWidget(QLabel("📊 Información de Fuerza"), 0, 0, 1, 4)
        self.force_value_labels = []
        self.force_rms_labels = []
        self.force_max_labels = []
        for i, canal in enumerate(FORCE_CANALES):
            header = QLabel(f"Fza {canal}")
            header.setStyleSheet("font-size: 11pt; font-weight: bold;")
            force_info_layout.addWidget(header, 1, i)
            v_label = QLabel("Valor: 0.000 V")
            r_label = QLabel("RMS: 0.000 V")
            m_label = QLabel("Max: 0.000 V")
            for row, lab in enumerate([v_label, r_label, m_label], start=2):
                lab.setStyleSheet("font-size: 10pt;")
                force_info_layout.addWidget(lab, row, i)
            self.force_value_labels.append(v_label)
            self.force_rms_labels.append(r_label)
            self.force_max_labels.append(m_label)
        force_info_widget = QWidget()
        force_info_widget.setLayout(force_info_layout)
        force_info_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        main_layout.addWidget(force_info_widget)

        # --- CANALES Y OPCIONES ---
        canales_group = QGroupBox("📈 Canales y Opciones")
        canales_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        canales_layout = QGridLayout(canales_group)
        canales_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        # Checkboxes Vibración
        self.vib_channel_checkboxes = {}
        for col, canal in enumerate(VIB_CANALES):
            cb = QCheckBox(f"Vib {canal}")
            cb.setChecked(True)
            cb.setToolTip(f"Habilita el procesamiento/guardado del canal {canal}")
            cb.stateChanged.connect(lambda state, ch=canal: self.on_vib_channel_toggled(ch, state))
            self.vib_channel_checkboxes[canal] = cb
            canales_layout.addWidget(cb, 0, col)

        # Checkboxes Fuerza
        self.force_channel_checkboxes = {}
        for col, canal in enumerate(FORCE_CANALES):
            cb = QCheckBox(f"Fza {canal}")
            cb.setChecked(True)
            cb.setToolTip(f"Habilita el procesamiento/guardado del canal {canal}")
            cb.stateChanged.connect(lambda state, ch=canal: self.on_force_channel_toggled(ch, state))
            self.force_channel_checkboxes[canal] = cb
            canales_layout.addWidget(cb, 1, col)

        # Auto-Set Y para ajustar automáticamente el rango de los ejes
        self.autoset_vib_btn = QPushButton("🎯 Auto-Set Vibración")
        self.autoset_vib_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold; padding: 5px;")
        self.autoset_vib_btn.clicked.connect(self.autoset_vib_y)
        self.autoset_force_btn = QPushButton("🎯 Auto-Set Fuerza")
        self.autoset_force_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold; padding: 5px;")
        self.autoset_force_btn.clicked.connect(self.autoset_force_y)
        self.autoset_all_btn = QPushButton("🎯 Auto-Set Todo")
        self.autoset_all_btn.setStyleSheet("background-color: #8e44ad; color: white; font-weight: bold; padding: 5px;")
        self.autoset_all_btn.clicked.connect(self.autoset_all_y)
        
        # Checkbox para auto-set continuo
        self.auto_scale_checkbox = QCheckBox("Auto-escala continua")
        self.auto_scale_checkbox.setToolTip("Ajusta automáticamente el eje Y en cada actualización")
        
        canales_layout.addWidget(self.autoset_vib_btn, 2, 0)
        canales_layout.addWidget(self.autoset_force_btn, 2, 1)
        canales_layout.addWidget(self.autoset_all_btn, 2, 2)
        canales_layout.addWidget(self.auto_scale_checkbox, 2, 3)

        main_layout.addWidget(canales_group)

        # --- Tabs para vistas ---
        self.plots_tabs = QtWidgets.QTabWidget()
        # Estilo para hacer las pestañas más grandes y visibles
        self.plots_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
            }
            QTabBar::tab {
                background: #3a3a3a;
                color: white;
                padding: 8px 20px;
                margin: 2px;
                border: 1px solid #555;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 11pt;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #2980b9;
                color: white;
            }
            QTabBar::tab:hover {
                background: #4a4a4a;
            }
        """)
        time_tab = QWidget()
        time_tab_layout = QVBoxLayout(time_tab)
        spec_tab = QWidget()
        spec_tab_layout = QVBoxLayout(spec_tab)
        self.plots_tabs.addTab(time_tab, "📈 Señales")
        self.plots_tabs.addTab(spec_tab, "🎨 Espectrogramas")
        comp_tab = QWidget()
        comp_layout = QGridLayout(comp_tab)
        comp_layout.setContentsMargins(0, 0, 0, 0)
        comp_layout.setSpacing(8)
        self.plots_tabs.addTab(comp_tab, "🔀 Comparación")

        # --- ANÁLISIS OFFLINE ---
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setContentsMargins(5, 5, 5, 5)
        analysis_layout.setSpacing(6)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Carpeta:"))
        self.analysis_dir_edit = QtWidgets.QLineEdit(AUTO_SAVE_BASE_DIRECTORY)
        path_row.addWidget(self.analysis_dir_edit)
        self.analysis_browse_btn = QPushButton("Explorar…")
        self.analysis_browse_btn.clicked.connect(self.browse_analysis_dir)
        path_row.addWidget(self.analysis_browse_btn)
        self.analysis_btn = QPushButton("Analizar")
        self.analysis_btn.clicked.connect(self.run_offline_analysis)
        path_row.addWidget(self.analysis_btn)
        analysis_layout.addLayout(path_row)

        self.analysis_text = QtWidgets.QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setFont(QFont("Courier", 9))
        analysis_layout.addWidget(self.analysis_text, stretch=1)

        self.plots_tabs.addTab(analysis_tab, "🔍 Análisis")

        # Subtabs de tiempo: vibración y fuerza
        time_subtabs = QtWidgets.QTabWidget()
        time_subtabs.setStyleSheet("""
            QTabBar::tab {
                background: #3a3a3a;
                color: white;
                padding: 6px 16px;
                margin: 2px;
                border: 1px solid #555;
                font-size: 10pt;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #27ae60;
            }
            QTabBar::tab:hover {
                background: #4a4a4a;
            }
        """)
        time_vib_tab = QWidget()
        time_force_tab = QWidget()
        time_subtabs.addTab(time_vib_tab, "📡 Vibración")
        time_subtabs.addTab(time_force_tab, "⚡ Fuerza")

        # Vibración - contenedor y layout
        vib_time_container = QWidget()
        vib_time_layout = QGridLayout(vib_time_container)
        vib_time_layout.setContentsMargins(0, 0, 0, 0)
        vib_time_layout.setSpacing(10)
        vib_time_layout.setRowStretch(0, 1)
        for c in range(max(2, len(VIB_CANALES))):
            vib_time_layout.setColumnStretch(c, 1)

        # Vibración plots
        self.vib_plot_widgets = []
        self.vib_plot_curves = []
        vib_colors = ['#FFD700', '#00FFFF']
        for i, canal in enumerate(VIB_CANALES):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setMinimumHeight(230)
            w.setBackground('#2b2b2b')
            w.setLabel('left', 'Aceleración', 'g')
            w.setLabel('bottom', 'Tiempo', 's')
            w.setTitle(f'Acelerómetro {canal}')
            w.showGrid(x=True, y=True, alpha=0.3)
            try:
                w.setMouseEnabled(x=True, y=True)
            except Exception:
                pass
            w.setYRange(ACCEL_MIN_G, ACCEL_MAX_G)
            w.setXRange(-self.time_window, 0)
            pen = pg.mkPen(color=vib_colors[i % len(vib_colors)], width=2)
            curve = w.plot(pen=pen)
            self.vib_plot_widgets.append(w)
            self.vib_plot_curves.append(curve)
            # Distribuir en una fila de 2 columnas
            vib_time_layout.addWidget(w, 0, i)

        time_vib_tab.setLayout(vib_time_layout)

        # Fuerza - contenedor y layout (2x2)
        force_time_container = QWidget()
        force_time_layout = QGridLayout(force_time_container)
        force_time_layout.setContentsMargins(0, 0, 0, 0)
        force_time_layout.setSpacing(10)
        force_time_layout.setRowStretch(0, 1)
        force_time_layout.setRowStretch(1, 1)
        for c in range(2):
            force_time_layout.setColumnStretch(c, 1)

        # Fuerza plots
        self.force_plot_widgets = []
        self.force_plot_curves = []
        force_colors = ['#FF5722', '#E91E63', '#9C27B0', '#673AB7']
        for i, canal in enumerate(FORCE_CANALES):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setMinimumHeight(230)
            w.setBackground('#2b2b2b')
            w.setLabel('left', 'Voltaje', 'V')
            w.setLabel('bottom', 'Tiempo', 's')
            w.setTitle(f'Fuerza {canal}')
            w.showGrid(x=True, y=True, alpha=0.3)
            try:
                w.setMouseEnabled(x=True, y=True)
            except Exception:
                pass
            w.setYRange(FORCE_MIN_VOLTAGE, FORCE_MAX_VOLTAGE)
            w.setXRange(-self.time_window, 0)
            pen = pg.mkPen(color=force_colors[i % len(force_colors)], width=2)
            curve = w.plot(pen=pen)
            self.force_plot_widgets.append(w)
            self.force_plot_curves.append(curve)
            # 2x2 grid
            force_time_layout.addWidget(w, i // 2, i % 2)

        time_force_tab.setLayout(force_time_layout)

        # Añadir subtabs de tiempo al tab principal de tiempo
        time_tab_layout.addWidget(time_subtabs, stretch=1)

        # --- COMPARACIÓN (Normalización + FFT empalmada) ---
        # Controles de normalización
        norm_group = QGroupBox("Normalización (solo vistas de comparación)")
        norm_layout = QHBoxLayout(norm_group)
        norm_layout.addWidget(QLabel("Modo:"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["Off", "Z-Score", "Min-Max [-1,1]", "RMS=1"])
        self.norm_combo.setCurrentText("Off")
        self.norm_mode = "Off"
        self.norm_combo.currentTextChanged.connect(lambda t: setattr(self, 'norm_mode', t))
        norm_layout.addWidget(self.norm_combo)
        # Remover DC
        self.remove_dc_cb = QCheckBox("Remover DC")
        self.remove_dc_cb.setChecked(True)
        norm_layout.addWidget(self.remove_dc_cb)

        # FFT empalmada
        self.comp_fft_plot = pg.PlotWidget()
        self.comp_fft_plot.setBackground('#2b2b2b')
        self.comp_fft_plot.setLabel('left', 'Amplitud', 'dB')
        self.comp_fft_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.comp_fft_plot.setTitle('FFT Normalizada (Vib vs Fuerza)')
        self.comp_fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.comp_fft_plot.addLegend()
        self.comp_fft_plot.showGrid(x=True, y=True)
        self.comp_fft_plot.setYRange(-80, 0)
        self.comp_fft_vib_curve = self.comp_fft_plot.plot(pen=pg.mkPen('#FFD700', width=2), name='Vib')
        self.comp_fft_force_curve = self.comp_fft_plot.plot(pen=pg.mkPen('#E91E63', width=2), name='Fuerza')

        # Columnas: izquierda Vibración, derecha Fuerza (señales normalizadas)
        self.comp_vib_time_plot = pg.PlotWidget()
        self.comp_vib_time_plot.setBackground('#2b2b2b')
        self.comp_vib_time_plot.setMinimumHeight(200)
        self.comp_vib_time_plot.setLabel('left', 'Vib (norm)')
        self.comp_vib_time_plot.setLabel('bottom', 'Tiempo', 's')
        self.comp_vib_time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.comp_vib_time_curve = self.comp_vib_time_plot.plot(pen=pg.mkPen('#FFD700', width=2))

        self.comp_force_time_plot = pg.PlotWidget()
        self.comp_force_time_plot.setBackground('#2b2b2b')
        self.comp_force_time_plot.setMinimumHeight(200)
        self.comp_force_time_plot.setLabel('left', 'Fza (norm)')
        self.comp_force_time_plot.setLabel('bottom', 'Tiempo', 's')
        self.comp_force_time_plot.showGrid(x=True, y=True, alpha=0.3)
        self.comp_force_time_curve = self.comp_force_time_plot.plot(pen=pg.mkPen('#E91E63', width=2))

        # FFT por columna
        self.comp_vib_fft_plot = pg.PlotWidget()
        self.comp_vib_fft_plot.setBackground('#2b2b2b')
        self.comp_vib_fft_plot.setLabel('left', 'Amplitud', 'dB')
        self.comp_vib_fft_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.comp_vib_fft_plot.setTitle('FFT Vib (normalizada)')
        self.comp_vib_fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.comp_vib_fft_plot.setYRange(-80, 0)
        self.comp_vib_fft_curve = self.comp_vib_fft_plot.plot(pen=pg.mkPen('#FFD700', width=2))

        self.comp_force_fft_plot = pg.PlotWidget()
        self.comp_force_fft_plot.setBackground('#2b2b2b')
        self.comp_force_fft_plot.setLabel('left', 'Amplitud', 'dB')
        self.comp_force_fft_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.comp_force_fft_plot.setTitle('FFT Fuerza (normalizada)')
        self.comp_force_fft_plot.showGrid(x=True, y=True, alpha=0.3)
        self.comp_force_fft_plot.setYRange(-80, 0)
        self.comp_force_fft_curve = self.comp_force_fft_plot.plot(pen=pg.mkPen('#E91E63', width=2))

        # Checkbox para mostrar/ocultar FFT empalmada
        self.show_overlay_cb = QCheckBox("Mostrar FFT empalmada (Vib vs Fuerza)")
        self.show_overlay_cb.setChecked(True)
        self.show_overlay_cb.stateChanged.connect(lambda s: self.comp_fft_plot.setVisible(self.show_overlay_cb.isChecked()))

        # Layout comparación: columnas por señal, overlay opcional abajo
        comp_layout.addWidget(norm_group, 0, 0, 1, 2)
        comp_layout.addWidget(self.comp_vib_time_plot, 1, 0)
        comp_layout.addWidget(self.comp_force_time_plot, 1, 1)
        comp_layout.addWidget(self.comp_vib_fft_plot, 2, 0)
        comp_layout.addWidget(self.comp_force_fft_plot, 2, 1)
        comp_layout.addWidget(self.show_overlay_cb, 3, 0, 1, 2)
        comp_layout.addWidget(self.comp_fft_plot, 4, 0, 1, 2)

        # FRF (Accel/Force) en dB
        self.comp_frf_plot = pg.PlotWidget()
        self.comp_frf_plot.setBackground('#2b2b2b')
        self.comp_frf_plot.setLabel('left', 'Accel/Force', 'dB')
        self.comp_frf_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.comp_frf_plot.setTitle('Relación A/F (FRF)')
        self.comp_frf_plot.showGrid(x=True, y=True, alpha=0.3)
        self.comp_frf_plot.setYRange(-80, 80)
        self.comp_frf_curve = self.comp_frf_plot.plot(pen=pg.mkPen('#03A9F4', width=2))
        comp_layout.addWidget(self.comp_frf_plot, 5, 0, 1, 2)

        # --- ESPECTROGRAMAS ---
        spec_subtabs = QtWidgets.QTabWidget()
        spec_subtabs.setStyleSheet("""
            QTabBar::tab {
                background: #3a3a3a;
                color: white;
                padding: 6px 16px;
                margin: 2px;
                border: 1px solid #555;
                font-size: 10pt;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #e67e22;
            }
            QTabBar::tab:hover {
                background: #4a4a4a;
            }
        """)
        spec_vib_tab = QWidget()
        spec_force_tab = QWidget()
        spec_subtabs.addTab(spec_vib_tab, "📡 Vibración")
        spec_subtabs.addTab(spec_force_tab, "⚡ Fuerza")

        # Vibración: espectrogramas (2)
        vib_spectro_group = QGroupBox("Espectrogramas (Vibración)")
        vib_spectro_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        vib_spectro_layout = QGridLayout(vib_spectro_group)
        vib_spectro_layout.setContentsMargins(0, 0, 0, 0)
        vib_spectro_layout.setSpacing(6)
        self.vib_spec_plot_widgets = []
        self.vib_spec_images = []
        dx_vib = self.spec_hop / self.vib_sample_rate
        dy_vib = self.vib_sample_rate / self.spec_nfft
        for i, canal in enumerate(VIB_CANALES):
            pw = pg.PlotWidget()
            pw.setBackground('#2b2b2b')
            pw.setTitle(f"Espectrograma Vib {canal}")
            pw.setLabel('left', 'Frecuencia', 'Hz')
            pw.setLabel('bottom', 'Tiempo', 's')
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setMinimumHeight(230)
            pw.setYRange(0, self.vib_sample_rate / 2)
            pw.setXRange(0, self.spec_max_cols * dx_vib)
            img = pg.ImageItem()
            # Escala a unidades físicas
            img.resetTransform()
            img.setTransform(QtGui.QTransform().scale(dx_vib, dy_vib))
            pw.addItem(img)
            # Inicializar imagen
            img.setImage(self.vib_spec_data[i], autoLevels=False, levels=(self.spec_levels_min_db, self.spec_levels_max_db))
            vib_spectro_layout.addWidget(pw, 0, i)
            self.vib_spec_plot_widgets.append(pw)
            self.vib_spec_images.append(img)

        # Fuerza: espectrogramas (4)
        force_spectro_group = QGroupBox("Espectrogramas (Fuerza)")
        force_spectro_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        force_spectro_layout = QGridLayout(force_spectro_group)
        force_spectro_layout.setContentsMargins(0, 0, 0, 0)
        force_spectro_layout.setSpacing(6)
        self.force_spec_plot_widgets = []
        self.force_spec_images = []
        dx_force = self.spec_hop / self.force_sample_rate
        dy_force = self.force_sample_rate / self.spec_nfft
        for i, canal in enumerate(FORCE_CANALES):
            pw = pg.PlotWidget()
            pw.setBackground('#2b2b2b')
            pw.setTitle(f"Espectrograma Fza {canal}")
            pw.setLabel('left', 'Frecuencia', 'Hz')
            pw.setLabel('bottom', 'Tiempo', 's')
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setMinimumHeight(230)
            pw.setYRange(0, self.force_sample_rate / 2)
            pw.setXRange(0, self.spec_max_cols * dx_force)
            img = pg.ImageItem()
            img.resetTransform()
            img.setTransform(QtGui.QTransform().scale(dx_force, dy_force))
            pw.addItem(img)
            img.setImage(self.force_spec_data[i], autoLevels=False, levels=(self.spec_levels_min_db, self.spec_levels_max_db))
            # 2x2
            force_spectro_layout.addWidget(pw, i // 2, i % 2)
            self.force_spec_plot_widgets.append(pw)
            self.force_spec_images.append(img)

        # Contenido de subtabs de espectrograma
        spec_vib_tab.setLayout(vib_spectro_layout)
        spec_force_tab.setLayout(force_spectro_layout)

        # Aplicar LUT inicial
        self.update_spec_lut()

        spec_tab_layout.addWidget(spec_subtabs, stretch=1)

        # --- CONTROLES DE ESPECTROGRAMA ---
        spec_ctrl_group = QGroupBox("Controles Espectrograma")
        spec_ctrl_layout = QGridLayout(spec_ctrl_group)
        row = 0
        spec_ctrl_layout.addWidget(QLabel("Colormap:"), row, 0)
        self.spec_cmap_combo = QComboBox()
        self.spec_cmap_combo.addItems(["viridis", "inferno", "hot", "gray"]) 
        self.spec_cmap_combo.setCurrentText(self.spec_lut_name)
        spec_ctrl_layout.addWidget(self.spec_cmap_combo, row, 1)

        row += 1
        spec_ctrl_layout.addWidget(QLabel("NFFT:"), row, 0)
        self.spec_nfft_spin = QSpinBox()
        self.spec_nfft_spin.setRange(64, 4096)
        self.spec_nfft_spin.setSingleStep(64)
        self.spec_nfft_spin.setValue(self.spec_nfft)
        spec_ctrl_layout.addWidget(self.spec_nfft_spin, row, 1)

        spec_ctrl_layout.addWidget(QLabel("Hop:"), row, 2)
        self.spec_hop_spin = QSpinBox()
        self.spec_hop_spin.setRange(8, 4096)
        self.spec_hop_spin.setSingleStep(8)
        self.spec_hop_spin.setValue(self.spec_hop)
        spec_ctrl_layout.addWidget(self.spec_hop_spin, row, 3)

        row += 1
        spec_ctrl_layout.addWidget(QLabel("Min dB:"), row, 0)
        self.spec_min_db_spin = QDoubleSpinBox()
        self.spec_min_db_spin.setRange(-160.0, 0.0)
        self.spec_min_db_spin.setDecimals(1)
        self.spec_min_db_spin.setSingleStep(1.0)
        self.spec_min_db_spin.setValue(self.spec_levels_min_db)
        spec_ctrl_layout.addWidget(self.spec_min_db_spin, row, 1)

        spec_ctrl_layout.addWidget(QLabel("Max dB:"), row, 2)
        self.spec_max_db_spin = QDoubleSpinBox()
        self.spec_max_db_spin.setRange(-60.0, 20.0)
        self.spec_max_db_spin.setDecimals(1)
        self.spec_max_db_spin.setSingleStep(1.0)
        self.spec_max_db_spin.setValue(self.spec_levels_max_db)
        spec_ctrl_layout.addWidget(self.spec_max_db_spin, row, 3)

        row += 1
        self.spec_apply_btn = QPushButton("Aplicar")
        self.spec_apply_btn.clicked.connect(self.apply_spectrogram_settings)
        spec_ctrl_layout.addWidget(self.spec_apply_btn, row, 0, 1, 4)

        spec_tab_layout.addWidget(spec_ctrl_group)

        # Añadir tabs al layout principal con más prioridad de espacio
        main_layout.addWidget(self.plots_tabs, stretch=2)

        # --- BOTONES PRINCIPALES ---
        btn_layout = QHBoxLayout()
        self.start_stop_button = QPushButton("▶️ Iniciar Adquisición")
        self.start_stop_button.setStyleSheet("background-color: #2ecc71; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.start_stop_button.clicked.connect(self.toggle_acquisition)
        btn_layout.addWidget(self.start_stop_button)

        self.save_button = QPushButton("💾 Guardar Datos")
        self.save_button.setStyleSheet("background-color: #3498db; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.save_button.clicked.connect(self.save_current_session_data)
        self.save_button.setEnabled(False)
        btn_layout.addWidget(self.save_button)

        main_layout.addLayout(btn_layout)

        # Status bar
        self.status_label = QLabel("Listo")
        self.statusBar().addWidget(self.status_label)

    def autoset_vib_y(self):
        """Auto-ajusta el eje Y de los gráficos de vibración basándose en los datos actuales."""
        for i, widget in enumerate(self.vib_plot_widgets):
            if len(self.vib_buffer[i]) > 10:
                data = np.array(self.vib_buffer[i])
                data_min = np.min(data)
                data_max = np.max(data)
                margin = (data_max - data_min) * 0.15 + 0.001  # 15% de margen + mínimo
                widget.setYRange(data_min - margin, data_max + margin)
        print("🎯 Auto-Set Vibración aplicado")

    def autoset_force_y(self):
        """Auto-ajusta el eje Y de los gráficos de fuerza basándose en los datos actuales."""
        for i, widget in enumerate(self.force_plot_widgets):
            if len(self.force_buffer[i]) > 10:
                data = np.array(self.force_buffer[i])
                data_min = np.min(data)
                data_max = np.max(data)
                margin = (data_max - data_min) * 0.15 + 0.001  # 15% de margen + mínimo
                widget.setYRange(data_min - margin, data_max + margin)
        print("🎯 Auto-Set Fuerza aplicado")

    def autoset_all_y(self):
        """Auto-ajusta todos los gráficos."""
        self.autoset_vib_y()
        self.autoset_force_y()
        print("🎯 Auto-Set Todo aplicado")

    def toggle_acquisition(self):
        if not self.acquiring:
            self.start_acquisition()
        else:
            self.stop_acquisition()

    def start_acquisition(self):
        # Limpiar buffers
        for buf in self.vib_buffer:
            buf.clear()
        for buf in self.force_buffer:
            buf.clear()
        for data_list in self.all_vib_data:
            data_list.clear()
        for data_list in self.all_force_data:
            data_list.clear()

        # Crear nombre de sesión
        now = datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')
        self.current_session_base_filename = os.path.join(AUTO_SAVE_BASE_DIRECTORY, f"sesion_{timestamp_str}")
        self.session_start_time = time.time()
        self.session_start_time_str = now.strftime('%Y-%m-%d %H:%M:%S')

        # Iniciar hilos
        self.force_thread = ForceAcquisitionThread(
            FORCE_DISPOSITIVO, FORCE_CANALES, FORCE_SAMPLE_RATE, FORCE_MUESTRAS_POR_BLOQUE, FORCE_TERMINAL_CONFIG
        )
        self.vib_thread = VibrationAcquisitionThread(
            VIB_DISPOSITIVO, VIB_CANALES, VIB_SAMPLE_RATE, VIB_MUESTRAS_POR_BLOQUE
        )
        self.force_thread.start()
        self.vib_thread.start()

        # Iniciar timer
        self.update_timer.start(50)  # 20 Hz

        self.acquiring = True
        self.start_stop_button.setText("⏹️ Detener Adquisición")
        self.start_stop_button.setStyleSheet("background-color: #e74c3c; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.save_button.setEnabled(False)
        self.status_label.setText("⏺️ Iniciando adquisición...")
        print(f"✅ Adquisición iniciada: {self.current_session_base_filename}")

    def stop_acquisition(self):
        self.acquiring = False

        # Detener timer
        self.update_timer.stop()

        # Detener hilos
        if self.force_thread:
            self.force_thread.stop()
            self.force_thread.join(timeout=2.0)
        if self.vib_thread:
            self.vib_thread.stop()
            self.vib_thread.join(timeout=2.0)

        # Mostrar estadísticas finales
        vib_samples = sum(len(buf) for buf in self.all_vib_data)
        force_samples = sum(len(buf) for buf in self.all_force_data)
        
        self.start_stop_button.setText("▶️ Iniciar Adquisición")
        self.start_stop_button.setStyleSheet("background-color: #2ecc71; color: white; font-size: 14px; font-weight: bold; padding: 10px;")
        self.save_button.setEnabled(True)
        self.status_label.setText(
            f"⏸️ Detenido | 📡 Vib: {vib_samples:,} muestras | ⚡ Fuerza: {force_samples:,} muestras"
        )
        print(f"🛑 Adquisición detenida. Capturadas {vib_samples} muestras vib, {force_samples} muestras fuerza")

    def update_plots(self):
        # Procesar datos de vibración
        while not vib_queue.empty():
            try:
                datos_np = vib_queue.get_nowait()
                for i in range(len(VIB_CANALES)):
                    if i < datos_np.shape[0]:
                        canal = VIB_CANALES[i]
                        enabled = self.vib_channel_checkboxes.get(canal, None)
                        enabled = True if enabled is None else enabled.isChecked()
                        datos_canal_raw = np.asarray(datos_np[i])
                        # Guardar crudo
                        self.all_vib_data[i].extend(datos_canal_raw.tolist())
                        if enabled:
                            self.vib_buffer[i].extend(datos_canal_raw)
                            if len(datos_canal_raw) > 0:
                                valor = float(datos_canal_raw[-1])
                                rms = float(np.sqrt(np.mean(datos_canal_raw ** 2)))
                                max_val = float(np.max(np.abs(datos_canal_raw)))
                                self.vib_value_labels[i].setText(f"Valor: {valor:.3f} g")
                                self.vib_rms_labels[i].setText(f"RMS: {rms:.3f} g")
                                self.vib_max_labels[i].setText(f"Max: {max_val:.3f} g")
                            # Plot
                            if len(self.vib_buffer[i]) >= 2:
                                data = np.array(self.vib_buffer[i])
                                n = len(data)
                                t = np.linspace(-n / self.vib_sample_rate, 0, n)
                                self.vib_plot_curves[i].setData(t, data)
                                # Auto-escala continua si está habilitada
                                if self.auto_scale_checkbox.isChecked():
                                    data_min = np.min(data)
                                    data_max = np.max(data)
                                    margin = (data_max - data_min) * 0.15 + 0.001
                                    self.vib_plot_widgets[i].setYRange(data_min - margin, data_max + margin)
                        else:
                            self.vib_plot_curves[i].setData([], [])
                            self.vib_value_labels[i].setText("Valor: - (off)")
                            self.vib_rms_labels[i].setText("RMS: - (off)")
                            self.vib_max_labels[i].setText("Max: - (off)")
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error actualizando plots vib: {e}")

        # Procesar datos de fuerza
        while not force_queue.empty():
            try:
                datos_np = force_queue.get_nowait()
                for i in range(len(FORCE_CANALES)):
                    if i < datos_np.shape[0]:
                        canal = FORCE_CANALES[i]
                        enabled = self.force_channel_checkboxes.get(canal, None)
                        enabled = True if enabled is None else enabled.isChecked()
                        datos_canal_raw = np.asarray(datos_np[i])
                        # Guardar crudo
                        self.all_force_data[i].extend(datos_canal_raw.tolist())
                        if enabled:
                            self.force_buffer[i].extend(datos_canal_raw)
                            if len(datos_canal_raw) > 0:
                                valor = float(datos_canal_raw[-1])
                                rms = float(np.sqrt(np.mean(datos_canal_raw ** 2)))
                                max_val = float(np.max(np.abs(datos_canal_raw)))
                                self.force_value_labels[i].setText(f"Valor: {valor:.3f} V")
                                self.force_rms_labels[i].setText(f"RMS: {rms:.3f} V")
                                self.force_max_labels[i].setText(f"Max: {max_val:.3f} V")
                            # Plot
                            if len(self.force_buffer[i]) >= 2:
                                data = np.array(self.force_buffer[i])
                                n = len(data)
                                t = np.linspace(-n / self.force_sample_rate, 0, n)
                                self.force_plot_curves[i].setData(t, data)
                                # Auto-escala continua si está habilitada
                                if self.auto_scale_checkbox.isChecked():
                                    data_min = np.min(data)
                                    data_max = np.max(data)
                                    margin = (data_max - data_min) * 0.15 + 0.001
                                    self.force_plot_widgets[i].setYRange(data_min - margin, data_max + margin)
                        else:
                            self.force_plot_curves[i].setData([], [])
                            self.force_value_labels[i].setText("Valor: - (off)")
                            self.force_rms_labels[i].setText("RMS: - (off)")
                            self.force_max_labels[i].setText("Max: - (off)")
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error actualizando plots fuerza: {e}")

        # Actualizar espectrogramas
        try:
            self.update_spectrograms()
        except Exception as e:
            print(f"Error actualizando espectrogramas: {e}")

        # Actualizar vistas de comparación
        try:
            self.update_comparison_views()
        except Exception as e:
            print(f"Error actualizando comparación: {e}")
        
        # Actualizar contador de muestras en status bar
        if self.acquiring:
            vib_samples = sum(len(buf) for buf in self.all_vib_data)
            force_samples = sum(len(buf) for buf in self.all_force_data)
            self.status_label.setText(
                f"⏺️ Adquiriendo... | 📡 Vib: {vib_samples:,} muestras | ⚡ Fuerza: {force_samples:,} muestras"
            )

    def _spec_mag_db(self, segment):
        """Calcula una columna de espectrograma normalizada en dB (0 dB = pico del segmento)."""
        seg = np.asarray(segment)
        if seg.size != self.spec_nfft:
            return np.zeros((self.spec_freq_bins,), dtype=np.float32)
        win = self.spec_window
        xw = seg * win
        spec = np.abs(np.fft.rfft(xw))
        peak = np.max(spec) + 1e-12
        spec_norm = spec / peak
        col_db = 20.0 * np.log10(spec_norm + 1e-12)
        return col_db.astype(np.float32)

    def update_spectrograms(self):
        """Actualiza las imágenes de espectrograma para canales habilitados."""
        # Vibración
        for i, canal in enumerate(VIB_CANALES):
            enabled_cb = self.vib_channel_checkboxes.get(canal, None)
            enabled = True if enabled_cb is None else enabled_cb.isChecked()
            if not enabled:
                continue
            buf_len = len(self.vib_buffer[i])
            if buf_len >= self.spec_nfft and (buf_len - self.vib_spec_last_len[i] >= self.spec_hop):
                segment = np.array(self.vib_buffer[i])[-self.spec_nfft:]
                col = self._spec_mag_db(segment)
                # Desplazar izquierda y agregar nueva columna al final
                self.vib_spec_data[i] = np.roll(self.vib_spec_data[i], -1, axis=1)
                self.vib_spec_data[i][:, -1] = col
                self.vib_spec_last_len[i] = buf_len
                # Actualizar imagen
                self.vib_spec_images[i].setImage(self.vib_spec_data[i], autoLevels=False, levels=(self.spec_levels_min_db, self.spec_levels_max_db))

        # Fuerza
        for i, canal in enumerate(FORCE_CANALES):
            enabled_cb = self.force_channel_checkboxes.get(canal, None)
            enabled = True if enabled_cb is None else enabled_cb.isChecked()
            if not enabled:
                continue
            buf_len = len(self.force_buffer[i])
            if buf_len >= self.spec_nfft and (buf_len - self.force_spec_last_len[i] >= self.spec_hop):
                segment = np.array(self.force_buffer[i])[-self.spec_nfft:]
                col = self._spec_mag_db(segment)
                self.force_spec_data[i] = np.roll(self.force_spec_data[i], -1, axis=1)
                self.force_spec_data[i][:, -1] = col
                self.force_spec_last_len[i] = buf_len
                self.force_spec_images[i].setImage(self.force_spec_data[i], autoLevels=False, levels=(self.spec_levels_min_db, self.spec_levels_max_db))

    def apply_normalization(self, x, mode):
        x = np.asarray(x)
        if x.size == 0 or mode == 'Off':
            return x
        if mode == 'Z-Score':
            mu = float(np.mean(x))
            sigma = float(np.std(x))
            return (x - mu) / (sigma + 1e-12)
        if mode.startswith('Min-Max'):
            mn = float(np.min(x))
            mx = float(np.max(x))
            rng = mx - mn
            if rng <= 1e-12:
                return np.zeros_like(x)
            return ((x - mn) / rng) * 2.0 - 1.0
        if mode == 'RMS=1':
            rms = float(np.sqrt(np.mean(x**2)))
            return x / (rms + 1e-12)
        return x

    def _compute_fft_db(self, x, fs, nfft=None):
        x = np.asarray(x)
        if x.size < 8:
            return np.array([]), np.array([])
        if nfft is None:
            nfft = max(128, int(2 ** np.floor(np.log2(min(4096, x.size)))))
        win = np.hanning(nfft)
        if x.size < nfft:
            # zero-pad
            xz = np.zeros(nfft)
            xz[:x.size] = x
            x = xz
        else:
            x = x[-nfft:]
        x = x - np.mean(x)
        X = np.fft.rfft(x * win)
        mag = np.abs(X)
        if mag.max() <= 1e-12:
            mag = mag + 1e-12
        mag_db = 20.0 * np.log10(mag / np.max(mag))
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        return freqs, mag_db

    def _compute_fft_mag(self, x, fs, nfft=None):
        x = np.asarray(x)
        if x.size < 8:
            return np.array([]), np.array([])
        if nfft is None:
            nfft = max(128, int(2 ** np.floor(np.log2(min(4096, x.size)))))
        win = np.hanning(nfft)
        if x.size < nfft:
            xz = np.zeros(nfft)
            xz[:x.size] = x
            x = xz
        else:
            x = x[-nfft:]
        x = x - np.mean(x)
        X = np.fft.rfft(x * win)
        mag = np.abs(X)
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        return freqs, mag

    def update_comparison_views(self):
        # Elegir primer canal habilitado de cada módulo (fallback ai0)
        vib_idx = 0
        for i, c in enumerate(VIB_CANALES):
            cb = self.vib_channel_checkboxes.get(c)
            if cb is None or cb.isChecked():
                vib_idx = i
                break
        force_idx = 0
        for i, c in enumerate(FORCE_CANALES):
            cb = self.force_channel_checkboxes.get(c)
            if cb is None or cb.isChecked():
                force_idx = i
                break

        # Señales procesadas (opcional remover DC) y luego normalizadas para vista
        y_vib = np.array(self.vib_buffer[vib_idx])
        y_force = np.array(self.force_buffer[force_idx])
        y_vib_n = None
        y_force_n = None
        # Remover DC si está habilitado
        if y_vib.size and self.remove_dc_cb.isChecked():
            y_vib = y_vib - np.mean(y_vib)
        if y_force.size and self.remove_dc_cb.isChecked():
            y_force = y_force - np.mean(y_force)

        if y_vib.size:
            t_vib = np.linspace(-len(y_vib) / self.vib_sample_rate, 0, len(y_vib))
            y_vib_n = self.apply_normalization(y_vib, self.norm_mode)
            self.comp_vib_time_curve.setData(t_vib, y_vib_n)
            if self.norm_mode == 'Off':
                self.comp_vib_time_plot.setYRange(ACCEL_MIN_G, ACCEL_MAX_G)
            else:
                self.comp_vib_time_plot.setYRange(-3, 3)
        else:
            self.comp_vib_time_curve.setData([], [])

        if y_force.size:
            t_force = np.linspace(-len(y_force) / self.force_sample_rate, 0, len(y_force))
            y_force_n = self.apply_normalization(y_force, self.norm_mode)
            self.comp_force_time_curve.setData(t_force, y_force_n)
            if self.norm_mode == 'Off':
                self.comp_force_time_plot.setYRange(FORCE_MIN_VOLTAGE, FORCE_MAX_VOLTAGE)
            else:
                self.comp_force_time_plot.setYRange(-3, 3)
        else:
            self.comp_force_time_curve.setData([], [])

        # FFT empalmada (dB normalizados a su pico) usando señales procesadas
        if y_vib.size >= 128:
            fv, dv = self._compute_fft_db(y_vib, self.vib_sample_rate)
            self.comp_fft_vib_curve.setData(fv, dv)
        else:
            self.comp_fft_vib_curve.setData([], [])
        if y_force.size >= 128:
            ff, df = self._compute_fft_db(y_force, self.force_sample_rate)
            self.comp_fft_force_curve.setData(ff, df)
        else:
            self.comp_fft_force_curve.setData([], [])

        # FFT por columna
        if y_vib.size >= 128:
            self.comp_vib_fft_curve.setData(fv, dv)
        else:
            self.comp_vib_fft_curve.setData([], [])
        if y_force.size >= 128:
            self.comp_force_fft_curve.setData(ff, df)
        else:
            self.comp_force_fft_curve.setData([], [])

        # FRF Accel/Force (dB) con NFFT común
        if y_vib.size >= 128 and y_force.size >= 128:
            nfft_common = int(max(128, 2 ** np.floor(np.log2(min(4096, len(y_vib), len(y_force))))))
            fv2, mv = self._compute_fft_mag(y_vib, self.vib_sample_rate, nfft=nfft_common)
            ff2, mf = self._compute_fft_mag(y_force, self.force_sample_rate, nfft=nfft_common)
            if mv.size and mf.size:
                m = min(mv.size, mf.size)
                eps = 1e-12
                frf_db = 20.0 * np.log10((mv[:m] + eps) / (mf[:m] + eps))
                self.comp_frf_curve.setData(fv2[:m], frf_db)
            else:
                self.comp_frf_curve.setData([], [])
        else:
            self.comp_frf_curve.setData([], [])

    def build_colormap(self, name):
        """Construye un pg.ColorMap aproximado para el nombre dado."""
        name = (name or '').lower()
        if name == 'gray':
            positions = [0.0, 1.0]
            colors = [(0, 0, 0), (255, 255, 255)]
        elif name == 'hot':
            positions = [0.0, 0.3, 0.6, 1.0]
            colors = [(0, 0, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255)]
        elif name == 'inferno':
            positions = [0.0, 0.25, 0.5, 0.75, 1.0]
            colors = [(0, 0, 4), (84, 15, 109), (187, 55, 84), (249, 142, 8), (252, 255, 164)]
        else:  # viridis (default)
            positions = [0.0, 0.25, 0.5, 0.75, 1.0]
            colors = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
        return pg.ColorMap(positions, colors)

    def update_spec_lut(self):
        """Aplica el LUT a todas las imágenes de espectrograma."""
        try:
            cm = self.build_colormap(self.spec_lut_name)
            lut = cm.getLookupTable(0.0, 1.0, 256)
            for img in getattr(self, 'vib_spec_images', []):
                img.setLookupTable(lut)
            for img in getattr(self, 'force_spec_images', []):
                img.setLookupTable(lut)
        except Exception as e:
            print(f"Advertencia: no se pudo aplicar LUT ({self.spec_lut_name}): {e}")

    def apply_spectrogram_settings(self):
        """Lee controles y aplica nueva configuración de espectrograma."""
        try:
            # Leer controles
            nfft = int(self.spec_nfft_spin.value())
            hop = int(self.spec_hop_spin.value())
            if hop > nfft:
                hop = max(8, nfft // 2)
                self.spec_hop_spin.setValue(hop)
            min_db = float(self.spec_min_db_spin.value())
            max_db = float(self.spec_max_db_spin.value())
            if max_db <= min_db:
                max_db = min_db + 1.0
                self.spec_max_db_spin.setValue(max_db)
            cmap_name = self.spec_cmap_combo.currentText()

            # Actualizar estado interno
            self.spec_nfft = nfft
            self.spec_hop = hop
            self.spec_window = np.hanning(self.spec_nfft).astype(np.float32)
            self.spec_freq_bins = self.spec_nfft // 2 + 1
            self.spec_levels_min_db = min_db
            self.spec_levels_max_db = max_db
            self.spec_lut_name = cmap_name

            # Reinicializar buffers e imágenes
            self.vib_spec_data = [np.zeros((self.spec_freq_bins, self.spec_max_cols), dtype=np.float32) for _ in range(len(VIB_CANALES))]
            self.force_spec_data = [np.zeros((self.spec_freq_bins, self.spec_max_cols), dtype=np.float32) for _ in range(len(FORCE_CANALES))]
            self.vib_spec_last_len = [0 for _ in range(len(VIB_CANALES))]
            self.force_spec_last_len = [0 for _ in range(len(FORCE_CANALES))]

            # Reescalar imágenes según dx/dy nuevos y resetear vistas
            dx_vib = self.spec_hop / self.vib_sample_rate
            dy_vib = self.vib_sample_rate / self.spec_nfft
            for i, img in enumerate(self.vib_spec_images):
                img.resetTransform()
                img.setTransform(QtGui.QTransform().scale(dx_vib, dy_vib))
                img.setImage(self.vib_spec_data[i], autoLevels=False, levels=(self.spec_levels_min_db, self.spec_levels_max_db))
                self.vib_spec_plot_widgets[i].setXRange(0, self.spec_max_cols * dx_vib)
                self.vib_spec_plot_widgets[i].setYRange(0, self.vib_sample_rate / 2)

            dx_force = self.spec_hop / self.force_sample_rate
            dy_force = self.force_sample_rate / self.spec_nfft
            for i, img in enumerate(self.force_spec_images):
                img.resetTransform()
                img.setTransform(QtGui.QTransform().scale(dx_force, dy_force))
                img.setImage(self.force_spec_data[i], autoLevels=False, levels=(self.spec_levels_min_db, self.spec_levels_max_db))
                self.force_spec_plot_widgets[i].setXRange(0, self.spec_max_cols * dx_force)
                self.force_spec_plot_widgets[i].setYRange(0, self.force_sample_rate / 2)

            # LUT
            self.update_spec_lut()
        except Exception as e:
            print(f"Error aplicando configuración de espectrograma: {e}")

    def save_current_session_data(self):
        if not self.current_session_base_filename:
            QtWidgets.QMessageBox.warning(self, "Error", "No hay sesión activa para guardar")
            return

        # Metadatos
        frecuencia = self.frecuencia_spin.value()
        notas = self.notas_edit.text()

        # Canales habilitados
        enabled_vib_indices = [i for i, c in enumerate(VIB_CANALES) if self.vib_channel_checkboxes.get(c, None) is None or self.vib_channel_checkboxes[c].isChecked()]
        enabled_force_indices = [i for i, c in enumerate(FORCE_CANALES) if self.force_channel_checkboxes.get(c, None) is None or self.force_channel_checkboxes[c].isChecked()]

        # Construir un solo archivo con todos los datos (tiempo + vib + fuerza)
        try:
            lengths = []
            for i in enabled_vib_indices:
                lengths.append(len(self.all_vib_data[i]))
            for i in enabled_force_indices:
                lengths.append(len(self.all_force_data[i]))
            
            # Mensaje de error mejorado con información detallada
            if len(lengths) == 0:
                QtWidgets.QMessageBox.warning(
                    self, "Sin canales habilitados", 
                    "No hay canales habilitados para guardar.\n\n"
                    "Por favor, habilita al menos un canal de Vibración o Fuerza."
                )
                return
            
            if min(lengths) == 0:
                # Mostrar información de cuántos datos hay
                vib_info = ", ".join([f"{VIB_CANALES[i]}: {len(self.all_vib_data[i])}" for i in enabled_vib_indices])
                force_info = ", ".join([f"{FORCE_CANALES[i]}: {len(self.all_force_data[i])}" for i in enabled_force_indices])
                
                msg = "No hay datos suficientes para guardar.\n\n"
                msg += "📊 Estado de buffers:\n"
                if vib_info:
                    msg += f"  Vibración: {vib_info} muestras\n"
                if force_info:
                    msg += f"  Fuerza: {force_info} muestras\n"
                msg += "\n💡 Solución:\n"
                msg += "  1. Presiona '▶️ Iniciar Adquisición'\n"
                msg += "  2. Espera unos segundos para acumular datos\n"
                msg += "  3. Luego presiona '💾 Guardar Datos'"
                
                QtWidgets.QMessageBox.warning(self, "Sin datos", msg)
                return
            min_len = min(lengths)
            # Asumimos SR iguales (2000 Hz) para ambas familias
            sr = int(self.vib_sample_rate)
            time_vector = np.linspace(0, (min_len - 1) / sr, min_len)

            header_cols = ["Tiempo(s)"]
            data_arrays = [time_vector]
            for i in enabled_vib_indices:
                header_cols.append(f"Acel_{VIB_CANALES[i]}(g)")
                data_arrays.append(np.array(self.all_vib_data[i][:min_len]))
            for i in enabled_force_indices:
                header_cols.append(f"Volt_{FORCE_CANALES[i]}(V)")
                data_arrays.append(np.array(self.all_force_data[i][:min_len]))
            data_to_save = np.array(data_arrays).T

            # Calcular features de resumen (FFT + Wavelet) por canal
            def peak_freq(signal_arr, fs):
                x = np.asarray(signal_arr)
                if x.size < 2:
                    return 0.0
                spec = np.abs(np.fft.rfft(x))
                freqs = np.fft.rfftfreq(x.size, 1.0/fs)
                if spec.size <= 1:
                    return 0.0
                idx = int(np.argmax(spec[1:]) + 1)
                return float(freqs[idx])

            def spectral_centroid(signal_arr, fs):
                x = np.asarray(signal_arr)
                if x.size < 2:
                    return 0.0
                mag = np.abs(np.fft.rfft(x))
                freqs = np.fft.rfftfreq(x.size, 1.0/fs)
                denom = np.sum(mag)
                if denom == 0:
                    return 0.0
                return float(np.sum(freqs * mag) / denom)

            def wavelet_energy(signal_arr):
                if not HAS_PYWT:
                    return np.nan
                try:
                    coeffs = pywt.wavedec(signal_arr, 'db4', level=3)
                    # Excluir coeficiente de aproximación (cA3), sumar energía de detalles
                    details = coeffs[1:]
                    energy = float(sum(np.sum(np.square(d)) for d in details))
                    return energy
                except Exception:
                    return np.nan

            features_lines = []
            features_lines.append("# FEATURES SUMMARY (por canal)")
            features_lines.append("# Canal,Tipo,RMS,MaxAbs,Mean,Std,PeakFreq(Hz),SpectralCentroid(Hz),WaveletEnergy")
            # Vibración
            for i in enabled_vib_indices:
                x = np.array(self.all_vib_data[i][:min_len])
                rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
                maxabs = float(np.max(np.abs(x))) if x.size else 0.0
                mean = float(np.mean(x)) if x.size else 0.0
                std = float(np.std(x)) if x.size else 0.0
                pf = peak_freq(x, self.vib_sample_rate)
                sc = spectral_centroid(x, self.vib_sample_rate)
                we = wavelet_energy(x)
                features_lines.append(f"{VIB_CANALES[i]},VIB,{rms:.6f},{maxabs:.6f},{mean:.6f},{std:.6f},{pf:.3f},{sc:.3f},{we:.6f}")
            # Fuerza
            for i in enabled_force_indices:
                x = np.array(self.all_force_data[i][:min_len])
                rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
                maxabs = float(np.max(np.abs(x))) if x.size else 0.0
                mean = float(np.mean(x)) if x.size else 0.0
                std = float(np.std(x)) if x.size else 0.0
                pf = peak_freq(x, self.force_sample_rate)
                sc = spectral_centroid(x, self.force_sample_rate)
                we = wavelet_energy(x)
                features_lines.append(f"{FORCE_CANALES[i]},FORCE,{rms:.6f},{maxabs:.6f},{mean:.6f},{std:.6f},{pf:.3f},{sc:.3f},{we:.6f}")

            # Metadata
            canales_vib_guardados = ", ".join([VIB_CANALES[i] for i in enabled_vib_indices]) if enabled_vib_indices else ""
            canales_force_guardados = ", ".join([FORCE_CANALES[i] for i in enabled_force_indices]) if enabled_force_indices else ""
            metadata_header = (
                f"# Frecuencia: {frecuencia:.2f} Hz\n"
                f"# Notas: {notas}\n"
                f"# Canales vibración guardados: {canales_vib_guardados}\n"
                f"# Canales fuerza guardados: {canales_force_guardados}\n"
                f"# SR Vib: {self.vib_sample_rate} Hz\n"
                f"# SR Fuerza: {self.force_sample_rate} Hz\n"
                f"# Duración (s): {min_len/sr:.3f}\n"
            )

            filename_all = f"{self.current_session_base_filename}.csv"
            with open(filename_all, 'w', encoding='utf-8') as f:
                f.write(metadata_header)
                f.write(",".join(header_cols) + "\n")
                np.savetxt(f, data_to_save, delimiter=",", fmt='%.6f')
                f.write("\n")
                for line in features_lines:
                    f.write(line + "\n")
            print(f"✅ Datos combinados guardados: {filename_all}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error guardando datos combinados:\n{e}")
            print(f"❌ Error guardando datos combinados: {e}")
            traceback.print_exc()

        # Log
        try:
            self.log_experiment_details()
            QtWidgets.QMessageBox.information(
                self, "Guardado exitoso",
                f"Datos guardados en: \n{AUTO_SAVE_BASE_DIRECTORY}\n"
            )
        except Exception as e:
            print(f"⚠️ Error guardando log: {e}")

    def log_experiment_details(self):
        if not self.session_start_time or not self.current_session_base_filename:
            return
        log_file = os.path.join(AUTO_SAVE_BASE_DIRECTORY, "experimentos_log.csv")
        session_end = time.time()
        duration = session_end - self.session_start_time

        # Metadatos
        frecuencia = self.frecuencia_spin.value()
        notas = self.notas_edit.text()

        try:
            file_exists = os.path.isfile(log_file)
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(log_file) == 0:
                    writer.writerow([
                        "Timestamp Inicio", "Timestamp Fin", "Duracion (s)",
                        "SR Vib (Hz)", "SR Fuerza (Hz)", "Frecuencia (Hz)",
                        "Notas", "Base Archivo"
                    ])
                writer.writerow([
                    self.session_start_time_str,
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end)),
                    f"{duration:.2f}",
                    self.vib_sample_rate,
                    self.force_sample_rate,
                    f"{frecuencia:.2f}",
                    notas if notas else "",
                    os.path.basename(self.current_session_base_filename)
                ])
            print(f"✅ Log guardado: {log_file}")
        except Exception as e:
            print(f"⚠️ Error guardando log: {e}")

    def on_vib_channel_toggled(self, canal, state):
        try:
            idx = VIB_CANALES.index(canal)
            self.vib_buffer[idx].clear()
            self.vib_plot_curves[idx].setData([], [])
            self.vib_value_labels[idx].setText("Valor: - (off)")
            self.vib_rms_labels[idx].setText("RMS: - (off)")
            self.vib_max_labels[idx].setText("Max: - (off)")
        except Exception as e:
            print(f"Error al togglear vib {canal}: {e}")

    def on_force_channel_toggled(self, canal, state):
        try:
            idx = FORCE_CANALES.index(canal)
            self.force_buffer[idx].clear()
            self.force_plot_curves[idx].setData([], [])
            self.force_value_labels[idx].setText("Valor: - (off)")
            self.force_rms_labels[idx].setText("RMS: - (off)")
            self.force_max_labels[idx].setText("Max: - (off)")
        except Exception as e:
            print(f"Error al togglear fuerza {canal}: {e}")

    def browse_analysis_dir(self):
        """Abre un diálogo para seleccionar la carpeta de análisis."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Seleccionar carpeta de análisis", self.analysis_dir_edit.text()
        )
        if directory:
            self.analysis_dir_edit.setText(directory)

    def run_offline_analysis(self):
        """Ejecuta el script de análisis offline sobre la carpeta seleccionada."""
        folder = self.analysis_dir_edit.text()
        if not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(self, "Error", f"La carpeta no existe:\n{folder}")
            return

        self.analysis_text.setText(f"Iniciando análisis en: {folder}\n...")
        
        # Buscar el script de análisis en el mismo directorio que este script
        script_name = "analisis_offline_experimentos.py"
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        
        if not os.path.exists(script_path):
             self.analysis_text.append(f"⚠️ No se encontró el script de análisis en: {script_path}")
             return

        import subprocess
        try:
            # Ejecutar el script pasando la carpeta como argumento
            # Usamos sys.executable para usar el mismo entorno Python
            cmd = [sys.executable, script_path, folder]
            
            self.analysis_text.append(f"Ejecutando: {' '.join(cmd)}\n")
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Leer salida en tiempo real (bloqueante en GUI simple, idealmente usaría QProcess)
            stdout, stderr = process.communicate()
            
            if stdout:
                self.analysis_text.append("--- SALIDA ---\n" + stdout)
            if stderr:
                self.analysis_text.append("\n--- ERRORES ---\n" + stderr)
                
            if process.returncode == 0:
                self.analysis_text.append("\n✅ Análisis completado exitosamente.")
            else:
                self.analysis_text.append(f"\n❌ El análisis terminó con código {process.returncode}.")
                
        except Exception as e:
            self.analysis_text.append(f"\n❌ Error ejecutando subprocess: {e}")
            traceback.print_exc()

    def apply_dark_theme(self):
        """Apply dark color scheme like CubeSat GUI"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
        self.setPalette(palette)
        
        # Update plot backgrounds
        pg.setConfigOption('background', '#2b2b2b')
        pg.setConfigOption('foreground', 'w')

    def closeEvent(self, event):
        if self.acquiring:
            self.stop_acquisition()
        self.update_timer.stop()
        if self.force_thread and self.force_thread.is_alive():
            self.force_thread.stop()
            self.force_thread.join(timeout=2.0)
        if self.vib_thread and self.vib_thread.is_alive():
            self.vib_thread.stop()
            self.vib_thread.join(timeout=2.0)
        print("✅ Aplicación cerrada")
        event.accept()


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CombinedDAQ()
    window.show()
    sys.exit(app.exec_())

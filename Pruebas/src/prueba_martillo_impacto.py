#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRUEBA DE TIEMPO DE RESPUESTA - MARTILLO DE IMPACTO
====================================================
Mide el tiempo de respuesta del sensor de fuerza comparando
con un acelerómetro de referencia.

Basado en metodología de tesis: Tiempo desde impulso hasta pico de respuesta.
Resultado esperado: ~140 µs con error <3%

Hardware:
- NI 9205: Sensor de fuerza (ai0)
- NI 9234: Acelerómetro IEPE (ai0) - usado como referencia de impacto
"""

import sys
import os
import numpy as np
import threading
import queue
from datetime import datetime
from collections import deque

# PyQt5
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QPushButton,
                             QSpinBox, QDoubleSpinBox, QGroupBox, QTableWidget,
                             QTableWidgetItem, QMessageBox, QStatusBar, QSplitter,
                             QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

# PyQtGraph
import pyqtgraph as pg

# Pandas para guardar datos
import pandas as pd

# Scipy para análisis
from scipy import signal
from scipy.ndimage import maximum_filter1d

# NI-DAQmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    try:
        from nidaqmx.constants import TerminalConfiguration
    except ImportError:
        TerminalConfiguration = None
    HAS_NIDAQMX = True
except ImportError:
    HAS_NIDAQMX = False
    print("⚠️ NI-DAQmx no disponible - Modo simulación")

# =============================================================================
# CONFIGURACIÓN DE HARDWARE
# =============================================================================
FORCE_DEVICE = "cDAQ1Mod1"      # NI 9205
FORCE_CHANNEL = "ai0"
ACCEL_DEVICE = "cDAQ1Mod2"      # NI 9234
ACCEL_CHANNEL = "ai0"

# Sample rate alto para capturar tiempos de respuesta (~µs)
SAMPLE_RATE = 51200  # Hz (máximo del NI 9234)
BUFFER_SIZE = int(SAMPLE_RATE * 2)  # 2 segundos de buffer

# Umbral para detección de impacto
IMPACT_THRESHOLD_G = 0.5  # g - umbral para detectar impacto en acelerómetro
FORCE_THRESHOLD_V = 0.01  # V - umbral para detectar respuesta en fuerza

# Sensibilidad del acelerómetro
ACCEL_SENSITIVITY = 100.0  # mV/g (PCB 352C33)

# Directorio de datos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(SCRIPT_DIR, "pruebas_martillo")
os.makedirs(DATOS_DIR, exist_ok=True)

# Colas para datos
data_queue = queue.Queue(maxsize=100)

# =============================================================================
# HILO DE ADQUISICIÓN SINCRONIZADA
# =============================================================================
class SyncAcquisitionThread(threading.Thread):
    """Adquisición sincronizada de fuerza y aceleración a alta velocidad"""
    
    def __init__(self, sample_rate=SAMPLE_RATE):
        super().__init__(daemon=True)
        self.running = True
        self.sample_rate = sample_rate
        self.samples_per_read = 512
        
    def run(self):
        if not HAS_NIDAQMX:
            self._run_simulation()
            return
            
        try:
            # Crear tarea combinada
            self.task_force = nidaqmx.Task("ForceTask")
            self.task_accel = nidaqmx.Task("AccelTask")
            
            # Configurar canal de fuerza (NI 9205)
            if TerminalConfiguration:
                self.task_force.ai_channels.add_ai_voltage_chan(
                    f"{FORCE_DEVICE}/{FORCE_CHANNEL}",
                    terminal_config=TerminalConfiguration.DIFF,
                    min_val=-1.0,
                    max_val=1.0
                )
            else:
                self.task_force.ai_channels.add_ai_voltage_chan(
                    f"{FORCE_DEVICE}/{FORCE_CHANNEL}",
                    min_val=-1.0,
                    max_val=1.0
                )
            self.task_force.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples_per_read * 10
            )
            
            # Configurar canal de aceleración (NI 9234 con IEPE)
            try:
                self.task_accel.ai_channels.add_ai_accel_chan(
                    f"{ACCEL_DEVICE}/{ACCEL_CHANNEL}",
                    sensitivity=ACCEL_SENSITIVITY,
                    min_val=-50.0,
                    max_val=50.0,
                    current_excit_val=0.004
                )
            except:
                # Fallback a voltaje
                self.task_accel.ai_channels.add_ai_voltage_chan(
                    f"{ACCEL_DEVICE}/{ACCEL_CHANNEL}",
                    min_val=-5.0,
                    max_val=5.0
                )
                
            self.task_accel.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples_per_read * 10
            )
            
            self.task_force.start()
            self.task_accel.start()
            
            print(f"✅ Adquisición iniciada @ {self.sample_rate} Hz")
            
            while self.running:
                try:
                    force_data = self.task_force.read(
                        number_of_samples_per_channel=self.samples_per_read
                    )
                    accel_data = self.task_accel.read(
                        number_of_samples_per_channel=self.samples_per_read
                    )
                    
                    if not data_queue.full():
                        data_queue.put({
                            'force': np.array(force_data),
                            'accel': np.array(accel_data),
                            'timestamp': datetime.now()
                        })
                except Exception as e:
                    if self.running:
                        print(f"Error lectura: {e}")
                    break
                    
        except Exception as e:
            print(f"Error inicialización DAQ: {e}")
            self._run_simulation()
        finally:
            self._cleanup()
            
    def _run_simulation(self):
        """Modo simulación con impactos aleatorios"""
        print("🔄 Ejecutando en modo simulación")
        t = 0
        impact_time = np.random.uniform(1, 3)
        
        while self.running:
            # Generar datos simulados
            dt = self.samples_per_read / self.sample_rate
            t_arr = np.linspace(t, t + dt, self.samples_per_read)
            
            # Ruido base
            force = np.random.normal(0, 0.001, self.samples_per_read)
            accel = np.random.normal(0, 0.05, self.samples_per_read)
            
            # Simular impacto
            if t < impact_time < t + dt:
                idx = int((impact_time - t) / dt * self.samples_per_read)
                # Impulso en acelerómetro (más rápido)
                accel[idx:idx+5] += np.array([2, 5, 3, 1, 0.5])[:min(5, len(accel)-idx)]
                # Respuesta en fuerza (ligeramente retrasada ~150µs = ~8 muestras @ 51.2kHz)
                delay_samples = 8
                if idx + delay_samples < len(force):
                    force[idx+delay_samples:idx+delay_samples+10] += np.array([0.02, 0.05, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.005, 0.002])[:min(10, len(force)-idx-delay_samples)]
                
                impact_time = t + np.random.uniform(2, 5)
            
            t += dt
            
            if not data_queue.full():
                data_queue.put({
                    'force': force,
                    'accel': accel,
                    'timestamp': datetime.now()
                })
                
            threading.Event().wait(dt * 0.9)
            
    def _cleanup(self):
        try:
            if hasattr(self, 'task_force'):
                self.task_force.stop()
                self.task_force.close()
            if hasattr(self, 'task_accel'):
                self.task_accel.stop()
                self.task_accel.close()
        except:
            pass
            
    def stop(self):
        self.running = False


# =============================================================================
# INTERFAZ GRÁFICA
# =============================================================================
class PruebaMartilloGUI(QMainWindow):
    """Interfaz para prueba de tiempo de respuesta con martillo"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔨 Prueba de Tiempo de Respuesta - Martillo de Impacto")
        self.setGeometry(100, 100, 1400, 900)
        
        # Estado
        self.acquiring = False
        self.armed = False  # Esperando impacto
        self.acq_thread = None
        
        # Buffers
        self.force_buffer = deque(maxlen=BUFFER_SIZE)
        self.accel_buffer = deque(maxlen=BUFFER_SIZE)
        
        # Resultados de impactos
        self.impactos = []
        
        # Configuración de detección
        self.threshold_accel = IMPACT_THRESHOLD_G
        self.threshold_force = FORCE_THRESHOLD_V
        self.pre_trigger_samples = int(0.005 * SAMPLE_RATE)  # 5ms antes
        self.post_trigger_samples = int(0.020 * SAMPLE_RATE)  # 20ms después
        
        self._setup_ui()
        self._setup_timer()
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Panel superior: Controles
        controls = self._create_controls()
        layout.addWidget(controls)
        
        # Splitter para gráficas y resultados
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo: Gráficas
        graphs_widget = self._create_graphs()
        splitter.addWidget(graphs_widget)
        
        # Panel derecho: Resultados
        results_widget = self._create_results_panel()
        splitter.addWidget(results_widget)
        
        splitter.setSizes([900, 500])
        layout.addWidget(splitter)
        
        # Barra de estado
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Listo - Presione 'Iniciar' y luego 'Armar' para detectar impactos")
        self.status_bar.addWidget(self.status_label)
        
    def _create_controls(self):
        group = QGroupBox("⚙️ Controles")
        layout = QHBoxLayout(group)
        
        # Botón iniciar/detener
        self.start_btn = QPushButton("▶️ Iniciar Adquisición")
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        self.start_btn.clicked.connect(self.toggle_acquisition)
        layout.addWidget(self.start_btn)
        
        # Botón armar
        self.arm_btn = QPushButton("🎯 Armar Detección")
        self.arm_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 10px;")
        self.arm_btn.clicked.connect(self.toggle_arm)
        self.arm_btn.setEnabled(False)
        layout.addWidget(self.arm_btn)
        
        layout.addSpacing(20)
        
        # Umbral acelerómetro
        layout.addWidget(QLabel("Umbral Acel (g):"))
        self.threshold_accel_spin = QDoubleSpinBox()
        self.threshold_accel_spin.setRange(0.1, 10.0)
        self.threshold_accel_spin.setValue(self.threshold_accel)
        self.threshold_accel_spin.setSingleStep(0.1)
        self.threshold_accel_spin.valueChanged.connect(lambda v: setattr(self, 'threshold_accel', v))
        layout.addWidget(self.threshold_accel_spin)
        
        # Umbral fuerza
        layout.addWidget(QLabel("Umbral Fuerza (V):"))
        self.threshold_force_spin = QDoubleSpinBox()
        self.threshold_force_spin.setRange(0.001, 1.0)
        self.threshold_force_spin.setValue(self.threshold_force)
        self.threshold_force_spin.setSingleStep(0.005)
        self.threshold_force_spin.setDecimals(3)
        self.threshold_force_spin.valueChanged.connect(lambda v: setattr(self, 'threshold_force', v))
        layout.addWidget(self.threshold_force_spin)
        
        layout.addSpacing(20)
        
        # Botón limpiar
        clear_btn = QPushButton("🗑️ Limpiar Resultados")
        clear_btn.clicked.connect(self.clear_results)
        layout.addWidget(clear_btn)
        
        # Botón guardar
        save_btn = QPushButton("💾 Guardar Resultados")
        save_btn.clicked.connect(self.save_results)
        layout.addWidget(save_btn)
        
        layout.addStretch()
        
        return group
        
    def _create_graphs(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Configurar estilo oscuro
        pg.setConfigOptions(antialias=True)
        
        # Gráfica de señales en tiempo real
        self.realtime_plot = pg.PlotWidget(title="📊 Señales en Tiempo Real")
        self.realtime_plot.setLabel('left', 'Amplitud')
        self.realtime_plot.setLabel('bottom', 'Tiempo', 's')
        self.realtime_plot.showGrid(x=True, y=True, alpha=0.3)
        self.realtime_plot.addLegend()
        
        self.force_curve = self.realtime_plot.plot(pen=pg.mkPen('#3498db', width=2), name='Fuerza (V)')
        self.accel_curve = self.realtime_plot.plot(pen=pg.mkPen('#e74c3c', width=2), name='Aceleración (g)')
        
        # Líneas de umbral
        self.threshold_line_accel = pg.InfiniteLine(pos=self.threshold_accel, angle=0, 
                                                     pen=pg.mkPen('#e74c3c', width=1, style=Qt.DashLine))
        self.threshold_line_force = pg.InfiniteLine(pos=self.threshold_force, angle=0,
                                                     pen=pg.mkPen('#3498db', width=1, style=Qt.DashLine))
        self.realtime_plot.addItem(self.threshold_line_accel)
        self.realtime_plot.addItem(self.threshold_line_force)
        
        layout.addWidget(self.realtime_plot)
        
        # Gráfica de último impacto capturado
        self.impact_plot = pg.PlotWidget(title="🔨 Último Impacto Capturado")
        self.impact_plot.setLabel('left', 'Amplitud (normalizada)')
        self.impact_plot.setLabel('bottom', 'Tiempo', 'ms')
        self.impact_plot.showGrid(x=True, y=True, alpha=0.3)
        self.impact_plot.addLegend()
        
        self.impact_force_curve = self.impact_plot.plot(pen=pg.mkPen('#3498db', width=2), name='Fuerza')
        self.impact_accel_curve = self.impact_plot.plot(pen=pg.mkPen('#e74c3c', width=2), name='Aceleración')
        
        # Marcadores de picos
        self.accel_peak_marker = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('#e74c3c'))
        self.force_peak_marker = pg.ScatterPlotItem(size=15, brush=pg.mkBrush('#3498db'))
        self.impact_plot.addItem(self.accel_peak_marker)
        self.impact_plot.addItem(self.force_peak_marker)
        
        layout.addWidget(self.impact_plot)
        
        return widget
        
    def _create_results_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Estadísticas
        stats_group = QGroupBox("📈 Estadísticas")
        stats_layout = QGridLayout(stats_group)
        
        self.n_impacts_label = QLabel("Impactos: 0")
        self.mean_time_label = QLabel("Tiempo promedio: -- µs")
        self.std_time_label = QLabel("Desv. estándar: -- µs")
        self.error_label = QLabel("Error relativo: -- %")
        self.min_time_label = QLabel("Mínimo: -- µs")
        self.max_time_label = QLabel("Máximo: -- µs")
        
        for i, lbl in enumerate([self.n_impacts_label, self.mean_time_label, 
                                  self.std_time_label, self.error_label,
                                  self.min_time_label, self.max_time_label]):
            lbl.setFont(QFont("Consolas", 11))
            stats_layout.addWidget(lbl, i // 2, i % 2)
            
        layout.addWidget(stats_group)
        
        # Tabla de resultados
        table_group = QGroupBox("📋 Resultados Individuales")
        table_layout = QVBoxLayout(table_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "# Impacto", "Tiempo (µs)", "Pico Acel (g)", "Pico Fuerza (V)", "Timestamp"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.results_table)
        
        layout.addWidget(table_group)
        
        return widget
        
    def _setup_timer(self):
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        
    def toggle_acquisition(self):
        if not self.acquiring:
            self.start_acquisition()
        else:
            self.stop_acquisition()
            
    def start_acquisition(self):
        # Limpiar buffers
        self.force_buffer.clear()
        self.accel_buffer.clear()
        
        # Vaciar cola
        while not data_queue.empty():
            data_queue.get()
            
        # Iniciar hilo
        self.acq_thread = SyncAcquisitionThread(SAMPLE_RATE)
        self.acq_thread.start()
        
        # Iniciar timer
        self.update_timer.start(30)  # ~33 Hz
        
        self.acquiring = True
        self.start_btn.setText("⏹️ Detener")
        self.start_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
        self.arm_btn.setEnabled(True)
        self.status_label.setText("⏺️ Adquiriendo... Presione 'Armar' para detectar impactos")
        
    def stop_acquisition(self):
        self.acquiring = False
        self.armed = False
        self.update_timer.stop()
        
        if self.acq_thread:
            self.acq_thread.stop()
            self.acq_thread.join(timeout=2)
            
        self.start_btn.setText("▶️ Iniciar Adquisición")
        self.start_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        self.arm_btn.setEnabled(False)
        self.arm_btn.setText("🎯 Armar Detección")
        self.arm_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 10px;")
        self.status_label.setText(f"⏸️ Detenido | {len(self.impactos)} impactos capturados")
        
    def toggle_arm(self):
        self.armed = not self.armed
        if self.armed:
            self.arm_btn.setText("🔴 ARMADO - Esperando impacto...")
            self.arm_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
            self.status_label.setText("🎯 ARMADO - Golpee con el martillo")
        else:
            self.arm_btn.setText("🎯 Armar Detección")
            self.arm_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 10px;")
            self.status_label.setText("⏺️ Adquiriendo...")
            
    def update_plots(self):
        # Procesar datos de la cola
        while not data_queue.empty():
            try:
                data = data_queue.get_nowait()
                self.force_buffer.extend(data['force'])
                self.accel_buffer.extend(data['accel'])
            except:
                break
                
        if len(self.force_buffer) < 1000:
            return
            
        # Convertir a arrays
        force_arr = np.array(self.force_buffer)
        accel_arr = np.array(self.accel_buffer)
        
        # Actualizar gráfica en tiempo real (últimos 0.5 segundos)
        n_show = min(len(force_arr), int(0.5 * SAMPLE_RATE))
        t = np.linspace(-n_show/SAMPLE_RATE, 0, n_show)
        
        # Normalizar para visualización
        force_show = force_arr[-n_show:]
        accel_show = accel_arr[-n_show:]
        
        self.force_curve.setData(t, force_show * 10)  # Escalar fuerza para visualización
        self.accel_curve.setData(t, accel_show)
        
        # Actualizar líneas de umbral
        self.threshold_line_accel.setValue(self.threshold_accel)
        self.threshold_line_force.setValue(self.threshold_force * 10)
        
        # Detectar impacto si está armado
        if self.armed:
            self._detect_impact(force_arr, accel_arr)
            
    def _detect_impact(self, force_arr, accel_arr):
        """Detecta impacto y calcula tiempo de respuesta"""
        # Buscar pico en acelerómetro que supere umbral
        n = len(accel_arr)
        if n < self.pre_trigger_samples + self.post_trigger_samples:
            return
            
        # Buscar en los últimos datos
        search_region = accel_arr[-self.post_trigger_samples*2:]
        
        # Detectar cruce de umbral
        above_threshold = np.abs(search_region) > self.threshold_accel
        
        if not np.any(above_threshold):
            return
            
        # Encontrar primer cruce
        trigger_idx = np.argmax(above_threshold)
        
        # Verificar que tenemos suficientes datos antes y después
        global_idx = n - len(search_region) + trigger_idx
        
        if global_idx < self.pre_trigger_samples or global_idx > n - self.post_trigger_samples:
            return
            
        # Extraer ventana de análisis
        start_idx = global_idx - self.pre_trigger_samples
        end_idx = global_idx + self.post_trigger_samples
        
        force_window = force_arr[start_idx:end_idx]
        accel_window = accel_arr[start_idx:end_idx]
        
        # Encontrar picos
        accel_peak_idx = np.argmax(np.abs(accel_window))
        
        # Buscar pico de fuerza después del pico de aceleración
        force_after_accel = force_window[accel_peak_idx:]
        if len(force_after_accel) < 10:
            return
            
        # Remover DC de fuerza
        force_after_accel = force_after_accel - np.mean(force_window[:self.pre_trigger_samples])
        
        # Buscar primer pico significativo en fuerza
        force_peak_idx_rel = np.argmax(np.abs(force_after_accel))
        force_peak_idx = accel_peak_idx + force_peak_idx_rel
        
        # Verificar que el pico de fuerza supera umbral
        if np.abs(force_window[force_peak_idx] - np.mean(force_window[:self.pre_trigger_samples])) < self.threshold_force:
            return
            
        # Calcular tiempo de respuesta
        time_diff_samples = force_peak_idx - accel_peak_idx
        time_diff_us = time_diff_samples / SAMPLE_RATE * 1e6
        
        # Validar resultado (debe ser positivo y razonable)
        if time_diff_us < 0 or time_diff_us > 5000:  # Max 5ms
            return
            
        # Guardar resultado
        impact = {
            'numero': len(self.impactos) + 1,
            'tiempo_us': time_diff_us,
            'pico_accel_g': np.max(np.abs(accel_window)),
            'pico_force_V': np.max(np.abs(force_window - np.mean(force_window[:self.pre_trigger_samples]))),
            'timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'force_window': force_window.copy(),
            'accel_window': accel_window.copy(),
            'accel_peak_idx': accel_peak_idx,
            'force_peak_idx': force_peak_idx
        }
        self.impactos.append(impact)
        
        # Actualizar visualización
        self._update_impact_plot(impact)
        self._update_results_table()
        self._update_statistics()
        
        # Desarmar después de captura
        self.armed = False
        self.arm_btn.setText("🎯 Armar Detección")
        self.arm_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 10px;")
        self.status_label.setText(f"✅ Impacto #{impact['numero']} capturado: {time_diff_us:.1f} µs")
        
        # Limpiar buffers para evitar re-detección
        self.force_buffer.clear()
        self.accel_buffer.clear()
        
    def _update_impact_plot(self, impact):
        """Actualiza gráfica del último impacto"""
        force = impact['force_window']
        accel = impact['accel_window']
        
        # Normalizar para comparación
        force_norm = (force - np.mean(force[:self.pre_trigger_samples]))
        force_norm = force_norm / (np.max(np.abs(force_norm)) + 1e-12)
        
        accel_norm = accel / (np.max(np.abs(accel)) + 1e-12)
        
        # Tiempo en ms
        t_ms = np.arange(len(force)) / SAMPLE_RATE * 1000
        t_ms = t_ms - t_ms[self.pre_trigger_samples]  # Centrar en trigger
        
        self.impact_force_curve.setData(t_ms, force_norm)
        self.impact_accel_curve.setData(t_ms, accel_norm)
        
        # Marcadores de picos
        self.accel_peak_marker.setData(
            [t_ms[impact['accel_peak_idx']]], 
            [accel_norm[impact['accel_peak_idx']]]
        )
        self.force_peak_marker.setData(
            [t_ms[impact['force_peak_idx']]], 
            [force_norm[impact['force_peak_idx']]]
        )
        
    def _update_results_table(self):
        """Actualiza tabla de resultados"""
        self.results_table.setRowCount(len(self.impactos))
        
        for i, imp in enumerate(self.impactos):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(imp['numero'])))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{imp['tiempo_us']:.2f}"))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{imp['pico_accel_g']:.3f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{imp['pico_force_V']:.4f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(imp['timestamp']))
            
        self.results_table.scrollToBottom()
        
    def _update_statistics(self):
        """Actualiza estadísticas"""
        n = len(self.impactos)
        self.n_impacts_label.setText(f"Impactos: {n}")
        
        if n == 0:
            return
            
        tiempos = [imp['tiempo_us'] for imp in self.impactos]
        mean_t = np.mean(tiempos)
        std_t = np.std(tiempos) if n > 1 else 0
        error_rel = (std_t / mean_t * 100) if mean_t > 0 else 0
        
        self.mean_time_label.setText(f"Tiempo promedio: {mean_t:.2f} µs")
        self.std_time_label.setText(f"Desv. estándar: {std_t:.2f} µs")
        self.error_label.setText(f"Error relativo: {error_rel:.2f} %")
        self.min_time_label.setText(f"Mínimo: {np.min(tiempos):.2f} µs")
        self.max_time_label.setText(f"Máximo: {np.max(tiempos):.2f} µs")
        
    def clear_results(self):
        """Limpia resultados"""
        self.impactos = []
        self.results_table.setRowCount(0)
        self._update_statistics()
        self.impact_force_curve.setData([], [])
        self.impact_accel_curve.setData([], [])
        self.accel_peak_marker.setData([], [])
        self.force_peak_marker.setData([], [])
        self.status_label.setText("🗑️ Resultados limpiados")
        
    def save_results(self):
        """Guarda resultados a CSV"""
        if not self.impactos:
            QMessageBox.warning(self, "Sin datos", "No hay impactos para guardar")
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar resumen
        df = pd.DataFrame([{
            'numero': imp['numero'],
            'tiempo_us': imp['tiempo_us'],
            'pico_accel_g': imp['pico_accel_g'],
            'pico_force_V': imp['pico_force_V'],
            'timestamp': imp['timestamp']
        } for imp in self.impactos])
        
        csv_path = os.path.join(DATOS_DIR, f"martillo_{timestamp}_resumen.csv")
        df.to_csv(csv_path, index=False)
        
        # Calcular estadísticas
        tiempos = df['tiempo_us'].values
        stats = {
            'n_impactos': len(tiempos),
            'tiempo_promedio_us': np.mean(tiempos),
            'desv_estandar_us': np.std(tiempos),
            'error_relativo_pct': np.std(tiempos) / np.mean(tiempos) * 100,
            'tiempo_min_us': np.min(tiempos),
            'tiempo_max_us': np.max(tiempos)
        }
        
        stats_path = os.path.join(DATOS_DIR, f"martillo_{timestamp}_estadisticas.txt")
        with open(stats_path, 'w') as f:
            f.write("PRUEBA DE TIEMPO DE RESPUESTA - MARTILLO DE IMPACTO\n")
            f.write("="*50 + "\n\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
                
        QMessageBox.information(self, "Guardado", 
                                f"Resultados guardados:\n{csv_path}\n{stats_path}")
        self.status_label.setText(f"💾 Guardado: {os.path.basename(csv_path)}")
        
    def closeEvent(self, event):
        self.stop_acquisition()
        event.accept()


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("🔨 PRUEBA DE TIEMPO DE RESPUESTA - MARTILLO DE IMPACTO")
    print("="*60)
    print(f"   Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   Resolución temporal: {1/SAMPLE_RATE*1e6:.2f} µs")
    print(f"   Fuerza: {FORCE_DEVICE}/{FORCE_CHANNEL}")
    print(f"   Acelerómetro: {ACCEL_DEVICE}/{ACCEL_CHANNEL}")
    print(f"   Datos en: {DATOS_DIR}")
    print("="*60)
    print("\n📋 INSTRUCCIONES:")
    print("   1. Presione 'Iniciar Adquisición'")
    print("   2. Presione 'Armar Detección'")
    print("   3. Golpee con el martillo cerca del sensor")
    print("   4. El sistema capturará automáticamente el impacto")
    print("   5. Repita para obtener estadísticas")
    print("="*60 + "\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Tema oscuro
    palette = app.palette()
    from PyQt5.QtGui import QColor
    palette.setColor(palette.Window, QColor(53, 53, 53))
    palette.setColor(palette.WindowText, QColor(255, 255, 255))
    palette.setColor(palette.Base, QColor(25, 25, 25))
    palette.setColor(palette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(palette.Text, QColor(255, 255, 255))
    palette.setColor(palette.Button, QColor(53, 53, 53))
    palette.setColor(palette.ButtonText, QColor(255, 255, 255))
    palette.setColor(palette.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.Link, QColor(42, 130, 218))
    palette.setColor(palette.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = PruebaMartilloGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

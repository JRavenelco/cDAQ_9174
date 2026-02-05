#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALIBRACIÓN ESTÁTICA DE CELDAS DE CARGA
=======================================

Soporta:
- DYMH-105 (500 kg, 1.7 mV/V)
- OMEGA LC302 (varias capacidades, 1 mV/V)

Funciones:
1. Tareo (zero) - Establecer el cero sin carga
2. Calibración con masa conocida (ej: tu peso)
3. Medición de tiempo de respuesta (step response)
4. Guardar factores de calibración

Autor: Cascade AI
Fecha: 2025-12-03
"""

import sys
import time
import queue
import threading
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QPushButton, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox,
    QTextEdit, QMessageBox, QProgressBar, QFrame, QComboBox
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime
import os
import json

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

FORCE_DEVICE = "cDAQ1Mod1"
FORCE_CHANNEL = "ai0"
FORCE_SAMPLE_RATE = 2500  # Hz
FORCE_MIN_V = -5.0
FORCE_MAX_V = 5.0
FORCE_TERMINAL = TerminalConfiguration.DIFF if HAS_NIDAQMX else None

# ============================================
# DEFINICIÓN DE CELDAS DE CARGA
# ============================================

CELDAS_DISPONIBLES = {
    "DYMH-105 (500kg)": {
        "capacidad_kg": 500.0,
        "sensibilidad_mV_V": 1.7,
        "excitacion_V": 5.0,
        "ganancia": 601.0,
        "descripcion": "Daysensor DYMH-105, 500kg, 1.7mV/V"
    },
    "LC302-25 (11kg)": {
        "capacidad_kg": 11.34,  # 25 lb
        "sensibilidad_mV_V": 1.0,
        "excitacion_V": 5.0,
        "ganancia": 601.0,
        "descripcion": "OMEGA LC302-25, 25lb/111N, 1mV/V"
    },
    "LC302-50 (23kg)": {
        "capacidad_kg": 22.68,  # 50 lb
        "sensibilidad_mV_V": 1.0,
        "excitacion_V": 5.0,
        "ganancia": 601.0,
        "descripcion": "OMEGA LC302-50, 50lb/222N, 1mV/V"
    },
    "LC302-100 (45kg)": {
        "capacidad_kg": 45.36,  # 100 lb
        "sensibilidad_mV_V": 1.0,
        "excitacion_V": 5.0,
        "ganancia": 601.0,
        "descripcion": "OMEGA LC302-100, 100lb/445N, 1mV/V"
    },
    "LC302-250 (113kg)": {
        "capacidad_kg": 113.4,  # 250 lb
        "sensibilidad_mV_V": 1.0,
        "excitacion_V": 5.0,
        "ganancia": 601.0,
        "descripcion": "OMEGA LC302-250, 250lb/1112N, 1mV/V"
    },
    "LC302-500 (227kg)": {
        "capacidad_kg": 226.8,  # 500 lb
        "sensibilidad_mV_V": 1.0,
        "excitacion_V": 5.0,
        "ganancia": 601.0,
        "descripcion": "OMEGA LC302-500, 500lb/2224N, 1mV/V"
    },
    "LC302-1K (454kg)": {
        "capacidad_kg": 453.6,  # 1000 lb
        "sensibilidad_mV_V": 1.0,
        "excitacion_V": 5.0,
        "ganancia": 6.5,  # Medido experimentalmente (INA826 + OPA197)
        "descripcion": "OMEGA LC302-1K, 1000lb/4448N, 1mV/V"
    },
    "Personalizada": {
        "capacidad_kg": 100.0,
        "sensibilidad_mV_V": 2.0,
        "excitacion_V": 5.0,
        "ganancia": 601.0,
        "descripcion": "Celda personalizada - editar parámetros"
    }
}

# Celda por defecto
CELDA_ACTUAL = "DYMH-105 (500kg)"

# Directorio de datos
DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "caracterizacion_fuerza")
os.makedirs(DATOS_DIR, exist_ok=True)

# Archivo de calibración
CALIBRACION_FILE = os.path.join(DATOS_DIR, "calibracion_celda.json")


class CalibrationWindow(QMainWindow):
    """Ventana principal de calibración"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibración de Celdas de Carga")
        self.setMinimumSize(1100, 800)
        
        # Celda seleccionada
        self.celda_nombre = CELDA_ACTUAL
        self.celda_params = CELDAS_DISPONIBLES[self.celda_nombre].copy()
        
        # Variables de calibración
        self.voltage_offset = 0.0  # Offset de tareo (V)
        self.voltage_per_kg = None  # Factor V/kg calibrado
        self.calibration_done = False
        
        # Datos de adquisición
        self.voltage_buffer = []
        self.time_buffer = []
        self.acquiring = False
        self.task = None
        
        # Cargar calibración previa si existe
        self.load_calibration()
        
        # Configurar UI
        self.setup_ui()
        
        # Timer de actualización
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        
        # Iniciar adquisición continua
        self.start_acquisition()
        
    def setup_ui(self):
        """Configura la interfaz de usuario"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Aplicar tema oscuro
        self.apply_dark_theme()
        
        # === TÍTULO ===
        title = QLabel("🔧 CALIBRACIÓN DE CELDAS DE CARGA")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # === SELECTOR DE CELDA ===
        celda_group = QGroupBox("📦 Selección de Celda de Carga")
        celda_layout = QGridLayout(celda_group)
        
        celda_layout.addWidget(QLabel("Celda:"), 0, 0)
        self.celda_combo = QComboBox()
        self.celda_combo.addItems(list(CELDAS_DISPONIBLES.keys()))
        self.celda_combo.setCurrentText(self.celda_nombre)
        self.celda_combo.currentTextChanged.connect(self.on_celda_changed)
        celda_layout.addWidget(self.celda_combo, 0, 1)
        
        self.celda_desc_label = QLabel(self.celda_params['descripcion'])
        self.celda_desc_label.setStyleSheet("color: #888;")
        celda_layout.addWidget(self.celda_desc_label, 0, 2)
        
        # Parámetros editables
        celda_layout.addWidget(QLabel("Capacidad (kg):"), 1, 0)
        self.capacidad_spin = QDoubleSpinBox()
        self.capacidad_spin.setRange(1, 10000)
        self.capacidad_spin.setValue(self.celda_params['capacidad_kg'])
        self.capacidad_spin.setDecimals(2)
        self.capacidad_spin.valueChanged.connect(self.on_param_changed)
        celda_layout.addWidget(self.capacidad_spin, 1, 1)
        
        celda_layout.addWidget(QLabel("Sensibilidad (mV/V):"), 1, 2)
        self.sensibilidad_spin = QDoubleSpinBox()
        self.sensibilidad_spin.setRange(0.1, 10)
        self.sensibilidad_spin.setValue(self.celda_params['sensibilidad_mV_V'])
        self.sensibilidad_spin.setDecimals(2)
        self.sensibilidad_spin.valueChanged.connect(self.on_param_changed)
        celda_layout.addWidget(self.sensibilidad_spin, 1, 3)
        
        celda_layout.addWidget(QLabel("Excitación (V):"), 2, 0)
        self.excitacion_spin = QDoubleSpinBox()
        self.excitacion_spin.setRange(1, 15)
        self.excitacion_spin.setValue(self.celda_params['excitacion_V'])
        self.excitacion_spin.setDecimals(1)
        self.excitacion_spin.valueChanged.connect(self.on_param_changed)
        celda_layout.addWidget(self.excitacion_spin, 2, 1)
        
        celda_layout.addWidget(QLabel("Ganancia amp:"), 2, 2)
        self.ganancia_spin = QDoubleSpinBox()
        self.ganancia_spin.setRange(1, 2000)
        self.ganancia_spin.setValue(self.celda_params['ganancia'])
        self.ganancia_spin.setDecimals(0)
        self.ganancia_spin.valueChanged.connect(self.on_param_changed)
        celda_layout.addWidget(self.ganancia_spin, 2, 3)
        
        # Factor nominal calculado
        self.factor_nominal_label = QLabel("")
        self.factor_nominal_label.setStyleSheet("color: yellow;")
        celda_layout.addWidget(self.factor_nominal_label, 3, 0, 1, 4)
        self.update_factor_nominal()
        
        layout.addWidget(celda_group)
        
        # === PANEL SUPERIOR: Lectura actual ===
        reading_group = QGroupBox("📊 Lectura Actual")
        reading_layout = QGridLayout(reading_group)
        
        # Voltaje crudo
        reading_layout.addWidget(QLabel("Voltaje (V):"), 0, 0)
        self.voltage_label = QLabel("0.0000")
        self.voltage_label.setFont(QFont("Courier", 24, QFont.Bold))
        self.voltage_label.setStyleSheet("color: cyan;")
        reading_layout.addWidget(self.voltage_label, 0, 1)
        
        # Voltaje - offset
        reading_layout.addWidget(QLabel("V - Offset:"), 0, 2)
        self.voltage_offset_label = QLabel("0.0000")
        self.voltage_offset_label.setFont(QFont("Courier", 24, QFont.Bold))
        self.voltage_offset_label.setStyleSheet("color: lime;")
        reading_layout.addWidget(self.voltage_offset_label, 0, 3)
        
        # Fuerza calculada
        reading_layout.addWidget(QLabel("Fuerza (kg):"), 1, 0)
        self.force_kg_label = QLabel("---")
        self.force_kg_label.setFont(QFont("Courier", 24, QFont.Bold))
        self.force_kg_label.setStyleSheet("color: orange;")
        reading_layout.addWidget(self.force_kg_label, 1, 1)
        
        reading_layout.addWidget(QLabel("Fuerza (N):"), 1, 2)
        self.force_n_label = QLabel("---")
        self.force_n_label.setFont(QFont("Courier", 24, QFont.Bold))
        self.force_n_label.setStyleSheet("color: orange;")
        reading_layout.addWidget(self.force_n_label, 1, 3)
        
        layout.addWidget(reading_group)
        
        # === GRÁFICA EN TIEMPO REAL ===
        self.plot_widget = pg.PlotWidget(title="Señal de Voltaje")
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Voltaje', 'V')
        self.plot_widget.setLabel('bottom', 'Tiempo', 's')
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('c', width=2))
        self.offset_line = self.plot_widget.addLine(y=0, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        layout.addWidget(self.plot_widget)
        
        # === PANEL DE CALIBRACIÓN ===
        cal_layout = QHBoxLayout()
        
        # --- TAREO / PRECARGA ---
        tare_group = QGroupBox("1️⃣ TAREO / PRECARGA")
        tare_layout = QVBoxLayout(tare_group)
        
        tare_layout.addWidget(QLabel("Establecer punto de trabajo:"))
        
        # Opción 1: Tareo normal (zero)
        self.tare_btn = QPushButton("⚖️ TAREAR (Zero actual)")
        self.tare_btn.setMinimumHeight(40)
        self.tare_btn.clicked.connect(self.do_tare)
        tare_layout.addWidget(self.tare_btn)
        
        # Opción 2: Precarga a voltaje específico
        preload_layout = QHBoxLayout()
        preload_layout.addWidget(QLabel("Precarga (V):"))
        self.preload_spin = QDoubleSpinBox()
        self.preload_spin.setRange(-5.0, 5.0)
        self.preload_spin.setValue(2.5)
        self.preload_spin.setDecimals(2)
        self.preload_spin.setSingleStep(0.1)
        preload_layout.addWidget(self.preload_spin)
        tare_layout.addLayout(preload_layout)
        
        self.preload_btn = QPushButton("🎯 AJUSTAR PRECARGA")
        self.preload_btn.setMinimumHeight(40)
        self.preload_btn.setStyleSheet("background-color: #4a6fa5;")
        self.preload_btn.clicked.connect(self.do_preload)
        tare_layout.addWidget(self.preload_btn)
        
        self.tare_status = QLabel("Offset actual: 0.0000 V")
        tare_layout.addWidget(self.tare_status)
        
        # Indicador de rango útil
        self.range_label = QLabel("Rango: ±5V desde zero")
        self.range_label.setStyleSheet("color: #888;")
        tare_layout.addWidget(self.range_label)
        
        cal_layout.addWidget(tare_group)
        
        # --- CALIBRACIÓN ---
        cal_group = QGroupBox("2️⃣ CALIBRACIÓN")
        cal_group_layout = QVBoxLayout(cal_group)
        
        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("Masa conocida (kg):"))
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.1, 200)
        self.mass_spin.setValue(70.0)  # Tu peso aproximado
        self.mass_spin.setDecimals(2)
        self.mass_spin.setSuffix(" kg")
        mass_layout.addWidget(self.mass_spin)
        cal_group_layout.addLayout(mass_layout)
        
        cal_group_layout.addWidget(QLabel("Con la masa sobre la celda:"))
        self.cal_btn = QPushButton("📏 CALIBRAR")
        self.cal_btn.setMinimumHeight(50)
        self.cal_btn.clicked.connect(self.do_calibration)
        cal_group_layout.addWidget(self.cal_btn)
        
        self.cal_status = QLabel("Factor: No calibrado")
        cal_group_layout.addWidget(self.cal_status)
        
        cal_layout.addWidget(cal_group)
        
        # --- TIEMPO DE RESPUESTA ---
        step_group = QGroupBox("3️⃣ TIEMPO DE RESPUESTA")
        step_layout = QVBoxLayout(step_group)
        
        step_layout.addWidget(QLabel("Aplica/retira carga rápidamente:"))
        self.step_btn = QPushButton("⏱️ MEDIR STEP RESPONSE")
        self.step_btn.setMinimumHeight(50)
        self.step_btn.clicked.connect(self.measure_step_response)
        step_layout.addWidget(self.step_btn)
        
        self.step_status = QLabel("Tiempo de respuesta: ---")
        step_layout.addWidget(self.step_status)
        
        cal_layout.addWidget(step_group)
        
        layout.addLayout(cal_layout)
        
        # === RESULTADOS ===
        results_group = QGroupBox("📋 Resultados de Calibración")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        self.results_text.setFont(QFont("Courier", 10))
        results_layout.addWidget(self.results_text)
        
        # Botones de guardar/cargar
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 Guardar Calibración")
        self.save_btn.clicked.connect(self.save_calibration)
        btn_layout.addWidget(self.save_btn)
        
        self.load_btn = QPushButton("📂 Cargar Calibración")
        self.load_btn.clicked.connect(self.load_calibration)
        btn_layout.addWidget(self.load_btn)
        
        results_layout.addLayout(btn_layout)
        layout.addWidget(results_group)
        
        self.update_results_display()
        
    def apply_dark_theme(self):
        """Aplica tema oscuro"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(45, 45, 45))
        palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(60, 60, 60))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        self.setPalette(palette)
    
    def on_celda_changed(self, nombre):
        """Maneja el cambio de celda seleccionada"""
        self.celda_nombre = nombre
        self.celda_params = CELDAS_DISPONIBLES[nombre].copy()
        
        # Actualizar spinboxes
        self.capacidad_spin.blockSignals(True)
        self.sensibilidad_spin.blockSignals(True)
        self.excitacion_spin.blockSignals(True)
        self.ganancia_spin.blockSignals(True)
        
        self.capacidad_spin.setValue(self.celda_params['capacidad_kg'])
        self.sensibilidad_spin.setValue(self.celda_params['sensibilidad_mV_V'])
        self.excitacion_spin.setValue(self.celda_params['excitacion_V'])
        self.ganancia_spin.setValue(self.celda_params['ganancia'])
        
        self.capacidad_spin.blockSignals(False)
        self.sensibilidad_spin.blockSignals(False)
        self.excitacion_spin.blockSignals(False)
        self.ganancia_spin.blockSignals(False)
        
        self.celda_desc_label.setText(self.celda_params['descripcion'])
        self.update_factor_nominal()
        
        # Resetear calibración
        self.voltage_offset = 0.0
        self.voltage_per_kg = None
        self.calibration_done = False
        self.tare_status.setText("Offset actual: 0.0000 V")
        self.cal_status.setText("Factor: No calibrado")
        self.update_results_display()
        
        self.log_result(f"\n🔄 Celda cambiada a: {nombre}")
    
    def on_param_changed(self):
        """Maneja cambio en parámetros de celda"""
        self.celda_params['capacidad_kg'] = self.capacidad_spin.value()
        self.celda_params['sensibilidad_mV_V'] = self.sensibilidad_spin.value()
        self.celda_params['excitacion_V'] = self.excitacion_spin.value()
        self.celda_params['ganancia'] = self.ganancia_spin.value()
        self.update_factor_nominal()
    
    def update_factor_nominal(self):
        """Actualiza el factor nominal calculado"""
        factor = (self.celda_params['sensibilidad_mV_V'] / 1000) * \
                 self.celda_params['excitacion_V'] * \
                 self.celda_params['ganancia'] / \
                 self.celda_params['capacidad_kg']
        self.factor_nominal_label.setText(
            f"Factor nominal: {factor:.6f} V/kg  |  "
            f"Rango: 0 - {self.celda_params['capacidad_kg']:.1f} kg  |  "
            f"Salida max: {factor * self.celda_params['capacidad_kg']:.3f} V"
        )
        
    def start_acquisition(self):
        """Inicia adquisición continua"""
        self.acquiring = True
        self.acq_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.acq_thread.start()
        self.update_timer.start(50)  # 20 Hz update
        
    def _acquisition_loop(self):
        """Loop de adquisición en hilo separado"""
        if not HAS_NIDAQMX:
            self._simulation_loop()
            return
            
        try:
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(
                f"{FORCE_DEVICE}/{FORCE_CHANNEL}",
                terminal_config=FORCE_TERMINAL,
                min_val=FORCE_MIN_V,
                max_val=FORCE_MAX_V
            )
            self.task.timing.cfg_samp_clk_timing(
                rate=FORCE_SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=1000
            )
            self.task.start()
            
            samples_per_read = 100
            
            while self.acquiring:
                try:
                    data = self.task.read(number_of_samples_per_channel=samples_per_read)
                    data_np = np.array(data)
                    
                    # Agregar al buffer
                    t_now = time.time()
                    for i, v in enumerate(data_np):
                        self.voltage_buffer.append(v)
                        self.time_buffer.append(t_now + i / FORCE_SAMPLE_RATE)
                    
                    # Mantener solo últimos 5 segundos
                    max_samples = 5 * FORCE_SAMPLE_RATE
                    if len(self.voltage_buffer) > max_samples:
                        self.voltage_buffer = self.voltage_buffer[-max_samples:]
                        self.time_buffer = self.time_buffer[-max_samples:]
                        
                except Exception as e:
                    if self.acquiring:
                        print(f"Error lectura: {e}")
                    break
                    
        except Exception as e:
            print(f"Error inicializando DAQ: {e}")
            self._simulation_loop()
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass
                    
    def _simulation_loop(self):
        """Modo simulación"""
        t = 0
        while self.acquiring:
            dt = 0.04  # 25 Hz
            samples = int(dt * FORCE_SAMPLE_RATE)
            
            # Simular voltaje con algo de ruido
            noise = 0.001 * np.random.randn(samples)
            base = 0.1 + 0.05 * np.sin(2 * np.pi * 0.5 * t)  # Drift lento
            data = base + noise
            
            t_now = time.time()
            for i, v in enumerate(data):
                self.voltage_buffer.append(v)
                self.time_buffer.append(t_now + i / FORCE_SAMPLE_RATE)
            
            max_samples = 5 * FORCE_SAMPLE_RATE
            if len(self.voltage_buffer) > max_samples:
                self.voltage_buffer = self.voltage_buffer[-max_samples:]
                self.time_buffer = self.time_buffer[-max_samples:]
            
            t += dt
            time.sleep(dt)
            
    def update_display(self):
        """Actualiza la visualización"""
        if len(self.voltage_buffer) < 10:
            return
            
        # Voltaje actual (promedio de últimas muestras)
        recent = self.voltage_buffer[-100:] if len(self.voltage_buffer) >= 100 else self.voltage_buffer
        v_mean = np.mean(recent)
        v_std = np.std(recent)
        
        self.voltage_label.setText(f"{v_mean:.5f} ± {v_std:.5f}")
        
        # Voltaje - offset
        v_corrected = v_mean - self.voltage_offset
        self.voltage_offset_label.setText(f"{v_corrected:.5f}")
        
        # Fuerza (si está calibrado)
        if self.voltage_per_kg is not None and self.voltage_per_kg != 0:
            force_kg = v_corrected / self.voltage_per_kg
            force_n = force_kg * 9.81
            self.force_kg_label.setText(f"{force_kg:.2f}")
            self.force_n_label.setText(f"{force_n:.2f}")
        else:
            # Usar factor nominal de la celda seleccionada
            factor_nominal = (self.celda_params['sensibilidad_mV_V'] / 1000) * \
                            self.celda_params['excitacion_V'] * \
                            self.celda_params['ganancia'] / \
                            self.celda_params['capacidad_kg']
            force_kg = v_corrected / factor_nominal if factor_nominal != 0 else 0
            force_n = force_kg * 9.81
            self.force_kg_label.setText(f"{force_kg:.2f} (nom)")
            self.force_n_label.setText(f"{force_n:.2f} (nom)")
        
        # Actualizar gráfica
        if len(self.time_buffer) > 0:
            t_arr = np.array(self.time_buffer) - self.time_buffer[0]
            v_arr = np.array(self.voltage_buffer)
            self.plot_curve.setData(t_arr, v_arr)
            self.offset_line.setValue(self.voltage_offset)
            
    def do_tare(self):
        """Realiza el tareo (establece el zero)"""
        if len(self.voltage_buffer) < 500:
            QMessageBox.warning(self, "Espera", "Esperando datos suficientes...")
            return
            
        # Promediar últimos 2 segundos
        samples = min(2 * FORCE_SAMPLE_RATE, len(self.voltage_buffer))
        recent = self.voltage_buffer[-samples:]
        
        self.voltage_offset = np.mean(recent)
        std = np.std(recent)
        
        self.tare_status.setText(f"Offset: {self.voltage_offset:.5f} V (σ={std:.5f})")
        self.update_range_label()
        self.log_result(f"TAREO: Offset = {self.voltage_offset:.5f} V (σ={std:.5f} V)")
        
        QMessageBox.information(self, "Tareo Completado", 
                               f"Offset establecido: {self.voltage_offset:.5f} V\n"
                               f"Desviación estándar: {std:.5f} V")
    
    def do_preload(self):
        """Ajusta el offset para trabajar con precarga"""
        if len(self.voltage_buffer) < 500:
            QMessageBox.warning(self, "Espera", "Esperando datos suficientes...")
            return
        
        target_voltage = self.preload_spin.value()
        
        # Voltaje actual
        samples = min(2 * FORCE_SAMPLE_RATE, len(self.voltage_buffer))
        recent = self.voltage_buffer[-samples:]
        v_actual = np.mean(recent)
        std = np.std(recent)
        
        # El offset se calcula para que: v_actual - offset = target
        # offset = v_actual - target
        self.voltage_offset = v_actual - target_voltage
        
        self.tare_status.setText(f"Offset: {self.voltage_offset:.5f} V → Precarga: {target_voltage:.2f} V")
        self.update_range_label()
        
        self.log_result(f"\nPRECARGA AJUSTADA:")
        self.log_result(f"  Voltaje actual: {v_actual:.5f} V")
        self.log_result(f"  Punto de trabajo: {target_voltage:.2f} V")
        self.log_result(f"  Offset calculado: {self.voltage_offset:.5f} V")
        
        # Calcular rango útil en kg si está calibrado
        if self.voltage_per_kg is not None and self.voltage_per_kg != 0:
            rango_pos = (5.0 - target_voltage) / self.voltage_per_kg
            rango_neg = (target_voltage + 5.0) / self.voltage_per_kg
            self.log_result(f"  Rango: -{rango_neg:.1f} kg a +{rango_pos:.1f} kg")
        
        QMessageBox.information(self, "Precarga Ajustada", 
                               f"Voltaje actual: {v_actual:.5f} V\n"
                               f"Punto de trabajo: {target_voltage:.2f} V\n"
                               f"Offset: {self.voltage_offset:.5f} V\n\n"
                               f"Ahora el display mostrará {target_voltage:.2f} V como referencia.")
    
    def update_range_label(self):
        """Actualiza el indicador de rango útil"""
        if self.voltage_per_kg is not None and self.voltage_per_kg != 0:
            # Calcular rango en kg
            v_trabajo = -self.voltage_offset if hasattr(self, 'preload_spin') else 0
            rango_pos_v = 5.0 - v_trabajo
            rango_neg_v = 5.0 + v_trabajo
            rango_pos_kg = rango_pos_v / self.voltage_per_kg
            rango_neg_kg = rango_neg_v / self.voltage_per_kg
            self.range_label.setText(f"Rango: -{rango_neg_kg:.0f} kg a +{rango_pos_kg:.0f} kg")
        else:
            self.range_label.setText(f"Rango: ±5V desde offset")
        
    def do_calibration(self):
        """Realiza la calibración con masa conocida"""
        if len(self.voltage_buffer) < 500:
            QMessageBox.warning(self, "Espera", "Esperando datos suficientes...")
            return
            
        mass_kg = self.mass_spin.value()
        
        # Promediar últimos 2 segundos
        samples = min(2 * FORCE_SAMPLE_RATE, len(self.voltage_buffer))
        recent = self.voltage_buffer[-samples:]
        
        v_mean = np.mean(recent)
        v_corrected = v_mean - self.voltage_offset
        
        if abs(v_corrected) < 0.0001:
            QMessageBox.warning(self, "Error", 
                               "Voltaje muy bajo. ¿Está la masa sobre la celda?\n"
                               "¿Ya hiciste el tareo sin carga?")
            return
            
        # Calcular factor
        self.voltage_per_kg = v_corrected / mass_kg
        self.calibration_done = True
        
        # Factor nominal para comparar (usando parámetros de celda seleccionada)
        factor_nominal = (self.celda_params['sensibilidad_mV_V'] / 1000) * \
                        self.celda_params['excitacion_V'] * \
                        self.celda_params['ganancia'] / \
                        self.celda_params['capacidad_kg']
        ratio = self.voltage_per_kg / factor_nominal if factor_nominal != 0 else 0
        
        self.cal_status.setText(f"Factor: {self.voltage_per_kg:.6f} V/kg")
        
        self.log_result(f"\nCALIBRACIÓN ({self.celda_nombre}):")
        self.log_result(f"  Masa aplicada: {mass_kg:.2f} kg")
        self.log_result(f"  Voltaje medido: {v_mean:.5f} V")
        self.log_result(f"  Voltaje corregido: {v_corrected:.5f} V")
        self.log_result(f"  Factor calibrado: {self.voltage_per_kg:.6f} V/kg")
        self.log_result(f"  Factor nominal: {factor_nominal:.6f} V/kg")
        self.log_result(f"  Ratio (cal/nom): {ratio:.3f}")
        
        QMessageBox.information(self, "Calibración Completada",
                               f"Celda: {self.celda_nombre}\n"
                               f"Masa: {mass_kg:.2f} kg\n"
                               f"Factor calibrado: {self.voltage_per_kg:.6f} V/kg\n"
                               f"Factor nominal: {factor_nominal:.6f} V/kg\n"
                               f"Ratio: {ratio:.3f}")
        
    def measure_step_response(self):
        """Mide el tiempo de respuesta ante un escalón"""
        QMessageBox.information(self, "Instrucciones",
                               "1. Prepara una masa para aplicar rápidamente\n"
                               "2. Haz clic en OK\n"
                               "3. Aplica la masa en los próximos 5 segundos\n"
                               "4. Mantén la masa por 2 segundos más")
        
        # Capturar datos por 7 segundos
        self.step_btn.setEnabled(False)
        self.step_btn.setText("⏳ Capturando...")
        
        # Limpiar buffer y capturar nuevos datos
        start_time = time.time()
        capture_duration = 7.0
        
        captured_data = []
        captured_time = []
        
        while time.time() - start_time < capture_duration:
            if len(self.voltage_buffer) > 0:
                captured_data.extend(self.voltage_buffer[-100:])
                captured_time.extend(self.time_buffer[-100:])
            time.sleep(0.05)
            QApplication.processEvents()
        
        self.step_btn.setEnabled(True)
        self.step_btn.setText("⏱️ MEDIR STEP RESPONSE")
        
        if len(captured_data) < 1000:
            QMessageBox.warning(self, "Error", "No se capturaron suficientes datos")
            return
            
        # Analizar step response
        data = np.array(captured_data)
        t = np.array(captured_time) - captured_time[0]
        
        # Detectar el escalón
        data_smooth = np.convolve(data, np.ones(50)/50, mode='same')
        diff = np.abs(np.diff(data_smooth))
        
        # Encontrar punto de máximo cambio
        step_idx = np.argmax(diff)
        
        if step_idx < 100 or step_idx > len(data) - 100:
            QMessageBox.warning(self, "Error", "No se detectó un escalón claro")
            return
            
        # Valores antes y después del escalón
        v_before = np.mean(data[max(0, step_idx-500):step_idx-50])
        v_after = np.mean(data[step_idx+500:min(len(data), step_idx+1000)])
        delta_v = v_after - v_before
        
        # Calcular tiempo de respuesta (10% a 90%)
        v_10 = v_before + 0.1 * delta_v
        v_90 = v_before + 0.9 * delta_v
        
        # Buscar índices
        search_start = step_idx - 50
        search_end = min(step_idx + 500, len(data))
        
        idx_10 = None
        idx_90 = None
        
        for i in range(search_start, search_end):
            if idx_10 is None and data[i] >= v_10:
                idx_10 = i
            if idx_10 is not None and data[i] >= v_90:
                idx_90 = i
                break
                
        if idx_10 is not None and idx_90 is not None:
            rise_time = (t[idx_90] - t[idx_10]) * 1000  # ms
            self.step_status.setText(f"Tiempo de subida (10-90%): {rise_time:.2f} ms")
            
            self.log_result(f"\nTIEMPO DE RESPUESTA:")
            self.log_result(f"  ΔV: {delta_v:.5f} V")
            self.log_result(f"  Tiempo 10-90%: {rise_time:.2f} ms")
            self.log_result(f"  Frecuencia de corte aprox: {0.35/rise_time*1000:.1f} Hz")
            
            # Graficar
            self.plot_step_response(t, data, step_idx, idx_10, idx_90, v_10, v_90)
        else:
            self.step_status.setText("No se pudo calcular tiempo de respuesta")
            
    def plot_step_response(self, t, data, step_idx, idx_10, idx_90, v_10, v_90):
        """Grafica el step response"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t * 1000, data, 'c-', linewidth=0.5, label='Señal')
        ax.axhline(y=v_10, color='g', linestyle='--', label='10%')
        ax.axhline(y=v_90, color='r', linestyle='--', label='90%')
        ax.axvline(x=t[idx_10]*1000, color='g', linestyle=':')
        ax.axvline(x=t[idx_90]*1000, color='r', linestyle=':')
        ax.set_xlabel('Tiempo (ms)')
        ax.set_ylabel('Voltaje (V)')
        ax.set_title('Step Response - Celda de Carga')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Zoom al escalón
        ax.set_xlim([t[step_idx]*1000 - 100, t[step_idx]*1000 + 200])
        
        plt.tight_layout()
        
        # Guardar
        filename = os.path.join(DATOS_DIR, f"step_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename, dpi=150)
        plt.show()
        
        self.log_result(f"  Gráfica guardada: {filename}")
        
    def log_result(self, text):
        """Agrega texto al log de resultados"""
        self.results_text.append(text)
        
    def update_results_display(self):
        """Actualiza el display de resultados"""
        self.results_text.clear()
        self.results_text.append(f"=== ESTADO DE CALIBRACIÓN ===\n")
        self.results_text.append(f"Celda: {self.celda_nombre}")
        self.results_text.append(f"Offset (tareo): {self.voltage_offset:.5f} V")
        
        if self.voltage_per_kg is not None:
            self.results_text.append(f"Factor calibrado: {self.voltage_per_kg:.6f} V/kg")
            self.results_text.append(f"Calibración: ✅ COMPLETADA")
        else:
            factor_nominal = (self.celda_params['sensibilidad_mV_V'] / 1000) * \
                            self.celda_params['excitacion_V'] * \
                            self.celda_params['ganancia'] / \
                            self.celda_params['capacidad_kg']
            self.results_text.append(f"Factor nominal: {factor_nominal:.6f} V/kg")
            self.results_text.append(f"Calibración: ⚠️ USANDO VALORES NOMINALES")
            
    def save_calibration(self):
        """Guarda la calibración a archivo"""
        # Nombre de archivo específico para cada celda
        safe_name = self.celda_nombre.replace(" ", "_").replace("(", "").replace(")", "")
        cal_file = os.path.join(DATOS_DIR, f"calibracion_{safe_name}.json")
        
        cal_data = {
            'celda_nombre': self.celda_nombre,
            'celda_params': self.celda_params,
            'voltage_offset': self.voltage_offset,
            'voltage_per_kg': self.voltage_per_kg,
            'timestamp': datetime.now().isoformat(),
            'notes': f"Calibrado con masa de {self.mass_spin.value()} kg"
        }
        
        with open(cal_file, 'w') as f:
            json.dump(cal_data, f, indent=2)
            
        self.log_result(f"\n💾 Calibración guardada en: {cal_file}")
        QMessageBox.information(self, "Guardado", f"Calibración guardada en:\n{cal_file}")
        
    def load_calibration(self):
        """Carga calibración desde archivo para la celda actual"""
        # Buscar archivo específico de la celda
        safe_name = self.celda_nombre.replace(" ", "_").replace("(", "").replace(")", "")
        cal_file = os.path.join(DATOS_DIR, f"calibracion_{safe_name}.json")
        
        # Si no existe, intentar con el archivo genérico antiguo
        if not os.path.exists(cal_file):
            cal_file = CALIBRACION_FILE
        
        if os.path.exists(cal_file):
            try:
                with open(cal_file, 'r') as f:
                    cal_data = json.load(f)
                    
                self.voltage_offset = cal_data.get('voltage_offset', 0.0)
                self.voltage_per_kg = cal_data.get('voltage_per_kg', None)
                
                # Actualizar UI si ya existe
                if hasattr(self, 'tare_status'):
                    self.tare_status.setText(f"Offset: {self.voltage_offset:.5f} V (cargado)")
                if hasattr(self, 'cal_status') and self.voltage_per_kg:
                    self.cal_status.setText(f"Factor: {self.voltage_per_kg:.6f} V/kg (cargado)")
                    self.calibration_done = True
                    
                if hasattr(self, 'results_text'):
                    self.update_results_display()
                    self.log_result(f"\n📂 Calibración cargada desde: {cal_file}")
                    self.log_result(f"   Celda: {cal_data.get('celda_nombre', 'N/A')}")
                    self.log_result(f"   Fecha: {cal_data.get('timestamp', 'N/A')}")
                    self.log_result(f"   Notas: {cal_data.get('notes', 'N/A')}")
                
            except Exception as e:
                print(f"Error cargando calibración: {e}")
                
    def closeEvent(self, event):
        """Maneja el cierre de la ventana"""
        self.acquiring = False
        self.update_timer.stop()
        time.sleep(0.2)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = CalibrationWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

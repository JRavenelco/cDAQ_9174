#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz simplificada para adquisición de acelerómetro (NI 9234)
Solo registra datos de vibración y los guarda automáticamente.
"""

import sys
import time
import queue
import threading
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QSizePolicy, QGroupBox, QGridLayout, QCheckBox
from datetime import datetime
import os
import csv
from collections import deque
import traceback

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, ExcitationSource, Coupling

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
VIB_DISPOSITIVO = "cDAQ1Mod2"
VIB_CANALES = ["ai0", "ai1"]
VIB_SAMPLE_RATE = 2000  # Hz
VIB_MUESTRAS_POR_BLOQUE = 100
TIME_WINDOW = 0.5  # segundos
ACCEL_MIN_G = -0.2
ACCEL_MAX_G = 0.2
ACC_SENSITIVITY = 98.3  # mV/g (PCB 352C33)

DATOS_DIR = "experimentos_caja_planetaria"
AUTO_SAVE_BASE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATOS_DIR)

vib_queue = queue.Queue(maxsize=10)

# --------------------------------------------------
# HILO DE ADQUISICIÓN ACELERÓMETRO
# --------------------------------------------------
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
                    print(f"⚠️ Fallo configuración acelerómetro {nombre_canal}: {accel_e}")
                    self._configure_voltage_with_iepe(nombre_canal)

            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10
            )
            self.task.start()
            print(f"✅ Adquisición acelerómetro iniciada: {self.sample_rate} Hz")

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
                        print(f"❌ Error DAQ no recuperable: {e}")
                        break
                except Exception as e:
                    print(f"❌ Error inesperado: {e}")
                    break

        except Exception as ex:
            print(f"❌ Error inicializando hilo: {ex}")
            traceback.print_exc()
        finally:
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                    print("✅ Tarea cerrada correctamente")
                except Exception as e:
                    print(f"Error cerrando tarea: {e}")
            self.task = None

    def _configure_voltage_with_iepe(self, physical_channel):
        """Configurar canal como voltaje con IEPE si falla modo acelerómetro"""
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
        print("🛑 Deteniendo adquisición...")
        self.running = False

# --------------------------------------------------
# VENTANA PRINCIPAL
# --------------------------------------------------
class AcelerometroDAQ(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DAQ Acelerómetro - NI 9234")
        self.resize(1000, 600)
        
        # Estado
        self.acquiring = False
        self.sample_rate = VIB_SAMPLE_RATE
        self.time_window = TIME_WINDOW
        buffer_size = max(int(self.time_window * self.sample_rate), 2)
        
        # Buffers
        self.all_vib_data = [[] for _ in range(len(VIB_CANALES))]
        self.vib_buffer = [deque(maxlen=buffer_size) for _ in range(len(VIB_CANALES))]
        self.tiempo = np.linspace(-self.time_window, 0, buffer_size)
        
        # Sesión
        self.current_session_base_filename = None
        self.session_start_time = None
        self.session_start_time_str = ""
        
        # Crear directorio
        if not os.path.exists(AUTO_SAVE_BASE_DIRECTORY):
            os.makedirs(AUTO_SAVE_BASE_DIRECTORY, exist_ok=True)
            print(f"✅ Directorio creado: {AUTO_SAVE_BASE_DIRECTORY}")
        
        # Hilo y timer
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
        title = QLabel("Adquisición de Acelerómetro (NI 9234)")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        main_layout.addWidget(title)
        
        # --- METADATOS EXPERIMENTALES ---
        metadata_group = QGroupBox("Condiciones del Experimento")
        metadata_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        metadata_layout = QGridLayout(metadata_group)
        
        # Voltaje
        metadata_layout.addWidget(QLabel("Voltaje (V):"), 0, 0)
        self.voltaje_spin = QtWidgets.QDoubleSpinBox()
        self.voltaje_spin.setRange(0.0, 120.0)
        self.voltaje_spin.setValue(0.0)
        self.voltaje_spin.setSingleStep(0.1)
        self.voltaje_spin.setDecimals(2)
        self.voltaje_spin.setSuffix(" V")
        self.voltaje_spin.setToolTip("Voltaje de operación en volts")
        metadata_layout.addWidget(self.voltaje_spin, 0, 1)
        
        # Nivel de Desbalanceo
        metadata_layout.addWidget(QLabel("Nivel Desbalanceo:"), 0, 2)
        self.nivel_desbalanceo_spin = QtWidgets.QSpinBox()
        self.nivel_desbalanceo_spin.setRange(0, 10)
        self.nivel_desbalanceo_spin.setValue(0)
        self.nivel_desbalanceo_spin.setToolTip("Nivel de desbalanceo del experimento (0-10)")
        metadata_layout.addWidget(self.nivel_desbalanceo_spin, 0, 3)
        
        # Nivel de Desalineamiento
        metadata_layout.addWidget(QLabel("Nivel Desalineamiento:"), 1, 0)
        self.nivel_desalineamiento_spin = QtWidgets.QSpinBox()
        self.nivel_desalineamiento_spin.setRange(0, 10)
        self.nivel_desalineamiento_spin.setValue(0)
        self.nivel_desalineamiento_spin.setToolTip("Nivel de desalineamiento (0-10)")
        metadata_layout.addWidget(self.nivel_desalineamiento_spin, 1, 1)
        
        # Nivel de Frenado
        metadata_layout.addWidget(QLabel("Nivel Frenado:"), 1, 2)
        self.nivel_frenado_spin = QtWidgets.QSpinBox()
        self.nivel_frenado_spin.setRange(0, 10)
        self.nivel_frenado_spin.setValue(0)
        self.nivel_frenado_spin.setToolTip("Nivel de frenado (0-10)")
        metadata_layout.addWidget(self.nivel_frenado_spin, 1, 3)
        
        # Orden de Corrida (manual)
        metadata_layout.addWidget(QLabel("Orden Corrida:"), 2, 0)
        self.orden_corrida_spin = QtWidgets.QSpinBox()
        self.orden_corrida_spin.setRange(1, 9999)
        self.orden_corrida_spin.setValue(1)
        self.orden_corrida_spin.setToolTip("Número de corrida (manual)")
        metadata_layout.addWidget(self.orden_corrida_spin, 2, 1)
        
        # Notas adicionales
        metadata_layout.addWidget(QLabel("Notas:"), 2, 2)
        self.notas_edit = QtWidgets.QLineEdit()
        self.notas_edit.setPlaceholderText("Observaciones opcionales...")
        metadata_layout.addWidget(self.notas_edit, 2, 3, 1, 1)
        
        main_layout.addWidget(metadata_group)
        
        # --- CANALES Y OPCIONES ---
        canales_group = QGroupBox("Canales y Opciones")
        canales_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        canales_layout = QGridLayout(canales_group)
        
        # Checkboxes por canal
        self.channel_checkboxes = {}
        for col, canal in enumerate(VIB_CANALES):
            cb = QCheckBox(f"Habilitar {canal}")
            cb.setChecked(True)
            cb.setToolTip(f"Habilita el procesamiento/guardado del canal {canal}")
            # Limpiar curva y buffer al deshabilitar
            cb.stateChanged.connect(lambda state, ch=canal: self.on_channel_toggled(ch, state))
            self.channel_checkboxes[canal] = cb
            canales_layout.addWidget(cb, 0, col)
        
        # Checkbox de normalización (z-score)
        self.normalize_checkbox = QCheckBox("Normalizar (z-score)")
        self.normalize_checkbox.setChecked(False)
        self.normalize_checkbox.setToolTip("Aplica z-score por bloque para visualización y al guardar (si está activado)")
        self.normalize_checkbox.toggled.connect(self.on_normalize_toggled)
        canales_layout.addWidget(self.normalize_checkbox, 1, 0, 1, len(VIB_CANALES))
        
        main_layout.addWidget(canales_group)
        
        # Info labels
        info_layout = QGridLayout()
        self.vib_value_labels = []
        self.vib_rms_labels = []
        self.vib_max_labels = []
        
        for i, canal in enumerate(VIB_CANALES):
            header = QLabel(f"Canal {canal}")
            header.setStyleSheet("font-size: 11pt; font-weight: bold;")
            info_layout.addWidget(header, 0, i)
            
            v_label = QLabel("Valor: 0.00 g")
            r_label = QLabel("RMS: 0.00 g")
            m_label = QLabel("Max: 0.00 g")
            
            for row, lab in enumerate([v_label, r_label, m_label], start=1):
                lab.setStyleSheet("font-size: 10pt;")
                info_layout.addWidget(lab, row, i)
            
            self.vib_value_labels.append(v_label)
            self.vib_rms_labels.append(r_label)
            self.vib_max_labels.append(m_label)
        
        main_layout.addLayout(info_layout)
        
        # Plots
        plot_container = QWidget()
        plot_layout = QHBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        self.vib_plot_widgets = []
        self.vib_plot_curves = []
        colors = ['#FFD700', '#00FFFF']
        
        for i, canal in enumerate(VIB_CANALES):
            w = pg.PlotWidget()
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            w.setLabel('left', 'Aceleración', 'g')
            w.setLabel('bottom', 'Tiempo', 's')
            w.setTitle(f'Acelerómetro {canal}')
            w.showGrid(x=True, y=True)
            w.setYRange(ACCEL_MIN_G, ACCEL_MAX_G)
            w.setXRange(-self.time_window, 0)
            
            pen = pg.mkPen(color=colors[i % len(colors)], width=2)
            curve = w.plot(pen=pen)
            
            self.vib_plot_widgets.append(w)
            self.vib_plot_curves.append(curve)
            plot_layout.addWidget(w)
        
        main_layout.addWidget(plot_container, stretch=1)
        
        # Botones
        button_layout = QHBoxLayout()
        self.start_stop_button = QPushButton("Iniciar")
        self.start_stop_button.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.start_stop_button.clicked.connect(self.toggle_acquisition)
        button_layout.addWidget(self.start_stop_button)
        
        self.save_button = QPushButton("Guardar Datos")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_current_session_data)
        button_layout.addWidget(self.save_button)
        
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Listo")
        self.statusBar().addWidget(self.status_label)

    def toggle_acquisition(self):
        if not self.acquiring:
            self.start_acquisition()
        else:
            self.stop_acquisition()

    def start_acquisition(self):
        # Limpiar buffers
        for buf in self.vib_buffer:
            buf.clear()
        for data_list in self.all_vib_data:
            data_list.clear()
        
        # Crear nombre de sesión
        now = datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')
        self.current_session_base_filename = os.path.join(
            AUTO_SAVE_BASE_DIRECTORY,
            f"sesion_{timestamp_str}"
        )
        self.session_start_time = time.time()
        self.session_start_time_str = now.strftime('%Y-%m-%d %H:%M:%S')
        
        # Iniciar hilo
        self.vib_thread = VibrationAcquisitionThread(
            VIB_DISPOSITIVO, VIB_CANALES, VIB_SAMPLE_RATE, VIB_MUESTRAS_POR_BLOQUE
        )
        self.vib_thread.start()
        
        # Iniciar timer
        self.update_timer.start(50)  # 20 Hz actualización UI
        
        self.acquiring = True
        self.start_stop_button.setText("Detener Adquisición")
        self.save_button.setEnabled(False)
        self.status_label.setText("Adquiriendo...")
        print(f"✅ Adquisición iniciada: {self.current_session_base_filename}")

    def stop_acquisition(self):
        self.acquiring = False
        
        # Detener timer
        self.update_timer.stop()
        
        # Detener hilo
        if self.vib_thread:
            self.vib_thread.stop()
            self.vib_thread.join(timeout=2.0)
        
        self.start_stop_button.setText("Iniciar Adquisición")
        self.save_button.setEnabled(True)
        self.status_label.setText("Detenido")
        print("🛑 Adquisición detenida")

    def update_plots(self):
        while not vib_queue.empty():
            try:
                datos_np = vib_queue.get_nowait()
                
                for i in range(len(VIB_CANALES)):
                    if i < datos_np.shape[0]:
                        canal = VIB_CANALES[i]
                        enabled = True
                        try:
                            enabled = self.channel_checkboxes[canal].isChecked()
                        except Exception:
                            pass
                        datos_canal_raw = datos_np[i]
                        
                        # Siempre almacenar datos crudos para posible guardado posterior
                        self.all_vib_data[i].extend(datos_canal_raw.tolist())
                        
                        # Preparar datos para visualización (normalizados o crudos)
                        if self.normalize_checkbox.isChecked():
                            mu = float(np.mean(datos_canal_raw))
                            sigma = float(np.std(datos_canal_raw))
                            if sigma <= 1e-12:
                                sigma = 1e-12
                            datos_display = (datos_canal_raw - mu) / sigma
                            units = "z"
                        else:
                            datos_display = datos_canal_raw
                            units = "g"
                        
                        # Actualización según esté habilitado el canal
                        if enabled:
                            self.vib_buffer[i].extend(datos_display)
                            # Actualizar labels
                            if len(datos_display) > 0:
                                valor = float(datos_display[-1])
                                rms = float(np.sqrt(np.mean(datos_display ** 2)))
                                max_val = float(np.max(np.abs(datos_display)))
                                self.vib_value_labels[i].setText(f"Valor: {valor:.3f} {units}")
                                self.vib_rms_labels[i].setText(f"RMS: {rms:.3f} {units}")
                                self.vib_max_labels[i].setText(f"Max: {max_val:.3f} {units}")
                            # Actualizar plot
                            if len(self.vib_buffer[i]) >= 2:
                                data = np.array(self.vib_buffer[i])
                                n = len(data)
                                t = np.linspace(-n / self.sample_rate, 0, n)
                                self.vib_plot_curves[i].setData(t, data)
                        else:
                            # Canal deshabilitado: limpiar curva y etiquetas
                            self.vib_plot_curves[i].setData([], [])
                            self.vib_value_labels[i].setText("Valor: - (off)")
                            self.vib_rms_labels[i].setText("RMS: - (off)")
                            self.vib_max_labels[i].setText("Max: - (off)")
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error actualizando plots: {e}")
    
    def on_channel_toggled(self, canal, state):
        """Limpia buffers y curva cuando un canal se deshabilita."""
        try:
            idx = VIB_CANALES.index(canal)
            self.vib_buffer[idx].clear()
            # Limpiar curva y etiquetas
            self.vib_plot_curves[idx].setData([], [])
            self.vib_value_labels[idx].setText("Valor: - (off)")
            self.vib_rms_labels[idx].setText("RMS: - (off)")
            self.vib_max_labels[idx].setText("Max: - (off)")
        except Exception as e:
            print(f"Error al togglear canal {canal}: {e}")
    
    def on_normalize_toggled(self, checked):
        """Ajusta los rangos de Y según normalización."""
        try:
            if checked:
                for w in self.vib_plot_widgets:
                    w.setYRange(-4.0, 4.0)
            else:
                for w in self.vib_plot_widgets:
                    w.setYRange(ACCEL_MIN_G, ACCEL_MAX_G)
        except Exception as e:
            print(f"Error actualizando rangos por normalización: {e}")

    def save_current_session_data(self):
        if not self.current_session_base_filename:
            QtWidgets.QMessageBox.warning(self, "Error", "No hay sesión activa para guardar")
            return
        
        if not any(len(c) > 0 for c in self.all_vib_data):
            QtWidgets.QMessageBox.warning(self, "Sin datos", "No hay datos para guardar")
            return
        
        # Obtener metadatos
        voltaje = self.voltaje_spin.value()
        nivel_desbalanceo = self.nivel_desbalanceo_spin.value()
        nivel_desalineamiento = self.nivel_desalineamiento_spin.value()
        nivel_frenado = self.nivel_frenado_spin.value()
        orden_corrida = self.orden_corrida_spin.value()
        notas = self.notas_edit.text()
        normalizado = self.normalize_checkbox.isChecked()
        
        # Canales habilitados
        enabled_indices = [i for i, c in enumerate(VIB_CANALES) if self.channel_checkboxes.get(c) is None or self.channel_checkboxes[c].isChecked()]
        if len(enabled_indices) == 0:
            QtWidgets.QMessageBox.warning(self, "Sin canales", "No hay canales habilitados para guardar")
            return
        
        # Longitudes disponibles de canales habilitados
        available_lengths = [len(self.all_vib_data[i]) for i in enabled_indices if len(self.all_vib_data[i]) > 0]
        if len(available_lengths) == 0:
            QtWidgets.QMessageBox.warning(self, "Sin datos", "No hay datos en los canales habilitados para guardar")
            return
        
        # Construir nombre con metadatos (voltaje formateado sin decimales si es entero)
        voltaje_str = f"{voltaje:.2f}".rstrip('0').rstrip('.')
        filename = f"{self.current_session_base_filename}_V{voltaje_str}_D{nivel_desbalanceo}_A{nivel_desalineamiento}_F{nivel_frenado}_O{orden_corrida}_vibracion.csv"
        
        try:
            min_len = min(available_lengths)
            time_vector = np.linspace(0, (min_len - 1) / self.sample_rate, min_len)
            
            # Header con metadatos como comentarios
            canales_guardados = ", ".join([VIB_CANALES[i] for i in enabled_indices])
            metadata_header = (
                f"# Voltaje: {voltaje:.2f} V\n"
                f"# Nivel Desbalanceo: {nivel_desbalanceo}\n"
                f"# Nivel Desalineamiento: {nivel_desalineamiento}\n"
                f"# Nivel Frenado: {nivel_frenado}\n"
                f"# Orden Corrida: {orden_corrida}\n"
                f"# Notas: {notas}\n"
                f"# Normalizado (z-score): {normalizado}\n"
                f"# Canales guardados: {canales_guardados}\n"
                f"# Sample Rate: {self.sample_rate} Hz\n"
                f"# Duracion: {min_len/self.sample_rate:.3f} s\n"
            )
            
            unidades = "z" if normalizado else "g"
            header_cols = ["Tiempo(s)"] + [f"Acel_{VIB_CANALES[i]}({unidades})" for i in enabled_indices]
            
            data_arrays = [time_vector]
            for i in enabled_indices:
                arr_raw = np.array(self.all_vib_data[i][:min_len])
                if normalizado:
                    mu = float(np.mean(arr_raw))
                    sigma = float(np.std(arr_raw))
                    if sigma <= 1e-12:
                        sigma = 1e-12
                    arr = (arr_raw - mu) / sigma
                else:
                    arr = arr_raw
                data_arrays.append(arr)
            
            data_to_save = np.array(data_arrays).T
            
            # Guardar con metadatos en header
            with open(filename, 'w') as f:
                f.write(metadata_header)
                f.write(",".join(header_cols) + "\n")
                np.savetxt(f, data_to_save, delimiter=",", fmt='%.6f')
            
            # Guardar log con metadatos
            self.log_experiment_details()
            
            QtWidgets.QMessageBox.information(
                self, "Guardado exitoso",
                f"Datos guardados:\n{os.path.basename(filename)}\n\n"
                f"Voltaje: {voltaje:.2f} V\n"
                f"Desbalanceo: {nivel_desbalanceo}, Desalineamiento: {nivel_desalineamiento}, Frenado: {nivel_frenado}\n"
                f"Orden: {orden_corrida}\n"
                f"Muestras: {min_len}, Duración: {min_len/self.sample_rate:.2f} s"
            )
            print(f"✅ Datos guardados: {filename}")
            
            # Auto-incrementar orden de corrida
            self.orden_corrida_spin.setValue(orden_corrida + 1)
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Error guardando:\n{e}")
            print(f"❌ Error guardando: {e}")
            traceback.print_exc()

    def log_experiment_details(self):
        """Registrar detalles en CSV de log"""
        if not self.session_start_time or not self.current_session_base_filename:
            return
        
        log_file = os.path.join(AUTO_SAVE_BASE_DIRECTORY, "experimentos_log.csv")
        session_end = time.time()
        duration = session_end - self.session_start_time
        
        # Obtener metadatos
        voltaje = self.voltaje_spin.value()
        nivel_desbalanceo = self.nivel_desbalanceo_spin.value()
        nivel_desalineamiento = self.nivel_desalineamiento_spin.value()
        nivel_frenado = self.nivel_frenado_spin.value()
        orden_corrida = self.orden_corrida_spin.value()
        notas = self.notas_edit.text()
        
        header = [
            "Timestamp Inicio", "Timestamp Fin", "Duracion (s)", "SR (Hz)",
            "Voltaje (V)", "Nivel Desbalanceo", "Nivel Desalineamiento", "Nivel Frenado", 
            "Orden Corrida", "Notas", "Archivo"
        ]
        entry = [
            self.session_start_time_str,
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_end)),
            f"{duration:.2f}",
            self.sample_rate,
            voltaje,
            nivel_desbalanceo,
            nivel_desalineamiento,
            nivel_frenado,
            orden_corrida,
            notas if notas else "",
            os.path.basename(self.current_session_base_filename) + f"_V{voltaje}_D{nivel_desbalanceo}_A{nivel_desalineamiento}_F{nivel_frenado}_O{orden_corrida}_vibracion.csv"
        ]
        
        try:
            file_exists = os.path.isfile(log_file)
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(log_file) == 0:
                    writer.writerow(header)
                writer.writerow(entry)
            print(f"✅ Log guardado: {log_file}")
        except Exception as e:
            print(f"⚠️ Error guardando log: {e}")

    def closeEvent(self, event):
        if self.acquiring:
            self.stop_acquisition()
        
        self.update_timer.stop()
        
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
    window = AcelerometroDAQ()
    window.show()
    sys.exit(app.exec_())

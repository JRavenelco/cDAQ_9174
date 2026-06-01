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

from scipy.fft import fft, fftfreq
from scipy.signal import welch

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, ExcitationSource, Coupling

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------
VIB_DISPOSITIVO = "cDAQ1Mod2"
VIB_CANALES = ["ai0"]
VIB_SAMPLE_RATE = 2000  # Hz
VIB_MUESTRAS_POR_BLOQUE = 200
TIME_WINDOW = 2.0  # segundos
ACCEL_MIN_G = -50.0
ACCEL_MAX_G = 50.0
ACC_SENSITIVITY = 100.0  # mV/g (PCB 352C33)
FFT_NFFT = 4096  # puntos para FFT

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
        self.setWindowTitle("DAQ Acelerómetro - NI 9234 (ai0 + FFT)")
        self.resize(1200, 700)
        
        # Estado
        self.acquiring = False
        self.sample_rate = VIB_SAMPLE_RATE
        self.time_window = TIME_WINDOW
        buffer_size = max(int(self.time_window * self.sample_rate), 2)
        
        # Buffers - solo 1 canal
        self.all_vib_data = [[]]
        self.vib_buffer = deque(maxlen=buffer_size)
        self.fft_buffer = deque(maxlen=FFT_NFFT)  # buffer para FFT
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
        title = QLabel("Adquisición Acelerómetro ai0 (NI 9234) + FFT")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
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
        metadata_layout.addWidget(self.voltaje_spin, 0, 1)
        
        # Nivel de Desbalanceo
        metadata_layout.addWidget(QLabel("Desbalanceo:"), 0, 2)
        self.nivel_desbalanceo_spin = QtWidgets.QSpinBox()
        self.nivel_desbalanceo_spin.setRange(0, 10)
        self.nivel_desbalanceo_spin.setValue(0)
        metadata_layout.addWidget(self.nivel_desbalanceo_spin, 0, 3)
        
        # Nivel de Desalineamiento
        metadata_layout.addWidget(QLabel("Desalineamiento:"), 0, 4)
        self.nivel_desalineamiento_spin = QtWidgets.QSpinBox()
        self.nivel_desalineamiento_spin.setRange(0, 10)
        self.nivel_desalineamiento_spin.setValue(0)
        metadata_layout.addWidget(self.nivel_desalineamiento_spin, 0, 5)
        
        # Nivel de Frenado
        metadata_layout.addWidget(QLabel("Frenado:"), 1, 0)
        self.nivel_frenado_spin = QtWidgets.QSpinBox()
        self.nivel_frenado_spin.setRange(0, 10)
        self.nivel_frenado_spin.setValue(0)
        metadata_layout.addWidget(self.nivel_frenado_spin, 1, 1)
        
        # Orden de Corrida
        metadata_layout.addWidget(QLabel("Orden:"), 1, 2)
        self.orden_corrida_spin = QtWidgets.QSpinBox()
        self.orden_corrida_spin.setRange(1, 9999)
        self.orden_corrida_spin.setValue(1)
        metadata_layout.addWidget(self.orden_corrida_spin, 1, 3)
        
        # Notas
        metadata_layout.addWidget(QLabel("Notas:"), 1, 4)
        self.notas_edit = QtWidgets.QLineEdit()
        self.notas_edit.setPlaceholderText("Observaciones...")
        metadata_layout.addWidget(self.notas_edit, 1, 5)
        
        main_layout.addWidget(metadata_group)
        
        # Info labels
        info_layout = QHBoxLayout()
        self.lbl_valor = QLabel("Valor: 0.00 g")
        self.lbl_rms = QLabel("RMS: 0.00 g")
        self.lbl_max = QLabel("Max: 0.00 g")
        self.lbl_muestras = QLabel("Muestras: 0")
        self.lbl_freq_pico = QLabel("Pico FFT: -- Hz")
        for lbl in [self.lbl_valor, self.lbl_rms, self.lbl_max, self.lbl_muestras, self.lbl_freq_pico]:
            lbl.setStyleSheet("font-size: 10pt; font-weight: bold;")
            info_layout.addWidget(lbl)
        main_layout.addLayout(info_layout)
        
        # --- PLOTS: Tiempo (izq) + FFT (der) ---
        plot_container = QWidget()
        plot_layout = QHBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot tiempo
        self.time_plot = pg.PlotWidget()
        self.time_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.time_plot.setLabel('left', 'Aceleración', 'g')
        self.time_plot.setLabel('bottom', 'Tiempo', 's')
        self.time_plot.setTitle('Señal Temporal - ai0')
        self.time_plot.showGrid(x=True, y=True)
        self.time_plot.setXRange(-self.time_window, 0)
        self.time_plot.enableAutoRange(axis='y')
        self.time_curve = self.time_plot.plot(pen=pg.mkPen(color='#FFD700', width=1.5))
        plot_layout.addWidget(self.time_plot)
        
        # Plot FFT
        self.fft_plot = pg.PlotWidget()
        self.fft_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fft_plot.setLabel('left', 'Amplitud', 'g')
        self.fft_plot.setLabel('bottom', 'Frecuencia', 'Hz')
        self.fft_plot.setTitle(f'FFT (N={FFT_NFFT}, Fs={VIB_SAMPLE_RATE} Hz)')
        self.fft_plot.showGrid(x=True, y=True)
        self.fft_plot.setXRange(0, VIB_SAMPLE_RATE / 2)
        self.fft_plot.enableAutoRange(axis='y')
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen(color='#00FF88', width=1.2))
        # Línea de pico
        self.fft_peak_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=1.5, style=QtCore.Qt.DashLine))
        self.fft_plot.addItem(self.fft_peak_line)
        plot_layout.addWidget(self.fft_plot)
        
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
        self.vib_buffer.clear()
        self.fft_buffer.clear()
        self.all_vib_data = [[]]
        
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
        self.status_label.setText(f"Detenido - {len(self.all_vib_data[0])} muestras")
        print("🛑 Adquisición detenida")

    def update_plots(self):
        while not vib_queue.empty():
            try:
                datos_np = vib_queue.get_nowait()
                
                # Solo canal 0 (ai0)
                if datos_np.ndim == 1:
                    datos_canal = datos_np
                else:
                    datos_canal = datos_np[0] if datos_np.shape[0] >= 1 else datos_np.flatten()
                
                # Almacenar datos crudos
                self.all_vib_data[0].extend(datos_canal.tolist())
                
                # Buffer para visualización temporal
                self.vib_buffer.extend(datos_canal)
                
                # Buffer para FFT
                self.fft_buffer.extend(datos_canal)
                
                # Actualizar labels
                valor = float(datos_canal[-1])
                rms = float(np.sqrt(np.mean(np.array(datos_canal) ** 2)))
                max_val = float(np.max(np.abs(datos_canal)))
                self.lbl_valor.setText(f"Valor: {valor:.4f} g")
                self.lbl_rms.setText(f"RMS: {rms:.4f} g")
                self.lbl_max.setText(f"Max: {max_val:.4f} g")
                self.lbl_muestras.setText(f"Muestras: {len(self.all_vib_data[0])}")
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"Error actualizando plots: {e}")
        
        # Actualizar plot temporal
        if len(self.vib_buffer) >= 2:
            data = np.array(self.vib_buffer)
            n = len(data)
            t = np.linspace(-n / self.sample_rate, 0, n)
            self.time_curve.setData(t, data)
        
        # Actualizar FFT cuando haya suficientes datos
        if len(self.fft_buffer) >= FFT_NFFT:
            data_fft = np.array(self.fft_buffer)[-FFT_NFFT:]
            data_fft = data_fft - np.mean(data_fft)
            
            # Ventana Hanning
            window = np.hanning(FFT_NFFT)
            data_windowed = data_fft * window
            
            # FFT
            yf = fft(data_windowed)
            xf = fftfreq(FFT_NFFT, 1.0 / self.sample_rate)[:FFT_NFFT // 2]
            mag = 2.0 / FFT_NFFT * np.abs(yf[:FFT_NFFT // 2])
            
            # Compensar ventana
            mag = mag / np.mean(window) * 2
            
            # Graficar FFT (solo > 2 Hz para evitar DC)
            mask = xf > 2
            self.fft_curve.setData(xf[mask], mag[mask])
            
            # Detectar pico principal
            if np.any(mask) and np.max(mag[mask]) > 0:
                idx_peak = np.argmax(mag[mask])
                freq_peak = xf[mask][idx_peak]
                amp_peak = mag[mask][idx_peak]
                self.fft_peak_line.setPos(freq_peak)
                self.lbl_freq_pico.setText(f"Pico FFT: {freq_peak:.1f} Hz ({amp_peak:.4f} g)")

    def save_current_session_data(self):
        if not self.current_session_base_filename:
            QtWidgets.QMessageBox.warning(self, "Error", "No hay sesión activa para guardar")
            return
        
        if len(self.all_vib_data[0]) == 0:
            QtWidgets.QMessageBox.warning(self, "Sin datos", "No hay datos para guardar")
            return
        
        # Obtener metadatos
        voltaje = self.voltaje_spin.value()
        nivel_desbalanceo = self.nivel_desbalanceo_spin.value()
        nivel_desalineamiento = self.nivel_desalineamiento_spin.value()
        nivel_frenado = self.nivel_frenado_spin.value()
        orden_corrida = self.orden_corrida_spin.value()
        notas = self.notas_edit.text()
        
        # Construir nombre con metadatos
        voltaje_str = f"{voltaje:.2f}".rstrip('0').rstrip('.')
        filename = f"{self.current_session_base_filename}_V{voltaje_str}_D{nivel_desbalanceo}_A{nivel_desalineamiento}_F{nivel_frenado}_O{orden_corrida}_vibracion.csv"
        
        try:
            n_samples = len(self.all_vib_data[0])
            time_vector = np.linspace(0, (n_samples - 1) / self.sample_rate, n_samples)
            
            metadata_header = (
                f"# Voltaje: {voltaje:.2f} V\n"
                f"# Nivel Desbalanceo: {nivel_desbalanceo}\n"
                f"# Nivel Desalineamiento: {nivel_desalineamiento}\n"
                f"# Nivel Frenado: {nivel_frenado}\n"
                f"# Orden Corrida: {orden_corrida}\n"
                f"# Notas: {notas}\n"
                f"# Normalizado (z-score): False\n"
                f"# Canales guardados: ai0\n"
                f"# Sample Rate: {self.sample_rate} Hz\n"
                f"# Duracion: {n_samples/self.sample_rate:.3f} s\n"
            )
            
            header_cols = ["Tiempo(s)", "Acel_ai0(g)"]
            arr = np.array(self.all_vib_data[0])
            data_to_save = np.column_stack([time_vector, arr])
            
            with open(filename, 'w') as f:
                f.write(metadata_header)
                f.write(",".join(header_cols) + "\n")
                np.savetxt(f, data_to_save, delimiter=",", fmt='%.6f')
            
            # Guardar log
            self.log_experiment_details()
            
            QtWidgets.QMessageBox.information(
                self, "Guardado exitoso",
                f"Datos guardados:\n{os.path.basename(filename)}\n\n"
                f"Voltaje: {voltaje:.2f} V\n"
                f"Desbalanceo: {nivel_desbalanceo}, Desalineamiento: {nivel_desalineamiento}, Frenado: {nivel_frenado}\n"
                f"Orden: {orden_corrida}\n"
                f"Muestras: {n_samples}, Duración: {n_samples/self.sample_rate:.2f} s"
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

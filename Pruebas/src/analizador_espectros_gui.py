#!/usr/bin/env python3
"""
Analizador de Espectros en Tiempo Real - Estilo Analog Discovery
Análisis espectral avanzado con espectrograma, filtros digitales y análisis de frecuencias
"""

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QPushButton, QTabWidget, QSizePolicy, 
                            QGroupBox, QGridLayout, QCheckBox, QSpinBox, 
                            QDoubleSpinBox, QComboBox, QSlider)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches
from scipy import signal
from scipy.signal import butter, filtfilt, sosfilt, spectrogram, welch
from scipy.fft import fft, fftfreq
from collections import deque
import queue
import threading
import time
from datetime import datetime


class SpectrogramWidget(FigureCanvas):
    """Widget personalizado para mostrar espectrograma como Analog Discovery"""
    
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        # Crear figure con estilo oscuro
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='black')
        super(SpectrogramWidget, self).__init__(self.fig)
        self.setParent(parent)
        
        # Configurar estilo
        plt.style.use('dark_background')
        
        # Crear subplot para espectrograma
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        
        # Configurar colores y estilo
        self.ax.tick_params(colors='white', which='both')
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        
        # Variables para datos
        self.sample_rate = 2000
        self.max_time_window = 10.0  # 10 segundos
        self.freq_range = [0, 1000]  # Hz
        self.colormap = 'plasma'  # Similar a Analog Discovery
        
        # Buffer para datos
        self.data_buffer = deque(maxlen=int(self.sample_rate * self.max_time_window))
        self.time_buffer = deque(maxlen=int(self.sample_rate * self.max_time_window))
        
        # Configurar subplot inicial
        self.setup_spectrogram()
        
    def setup_spectrogram(self):
        """Configurar el espectrograma inicial"""
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.set_xlabel('Tiempo (s)', color='white', fontsize=10)
        self.ax.set_ylabel('Frecuencia (Hz)', color='white', fontsize=10)
        self.ax.set_title('Espectrograma en Tiempo Real', color='white', fontsize=12, fontweight='bold')
        
        # Colores de ticks
        self.ax.tick_params(colors='white', which='both', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        # Grid sutil
        self.ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
        
        self.fig.tight_layout()
    
    def update_spectrogram(self, new_data, sample_rate=None):
        """Actualizar espectrograma con nuevos datos"""
        if sample_rate:
            self.sample_rate = sample_rate
            
        # Añadir nuevos datos al buffer
        if isinstance(new_data, (list, np.ndarray)) and len(new_data) > 0:
            current_time = time.time()
            for i, sample in enumerate(new_data):
                self.data_buffer.append(sample)
                self.time_buffer.append(current_time + i/self.sample_rate)
        
        # Verificar que tenemos suficientes datos
        if len(self.data_buffer) < 512:  # Mínimo para espectrograma
            return
            
        try:
            # Convertir buffers a arrays
            data_array = np.array(list(self.data_buffer))
            time_array = np.array(list(self.time_buffer))
            
            # Calcular espectrograma
            nperseg = min(512, len(data_array) // 4)  # Ventana adaptativa
            noverlap = nperseg // 2
            
            frequencies, times, Sxx = spectrogram(
                data_array, 
                fs=self.sample_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
            
            # Filtrar rango de frecuencias
            freq_mask = (frequencies >= self.freq_range[0]) & (frequencies <= self.freq_range[1])
            frequencies = frequencies[freq_mask]
            Sxx = Sxx[freq_mask, :]
            
            # Convertir a dB
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            
            # Limpiar axes
            self.ax.clear()
            
            # Plotear espectrograma
            im = self.ax.pcolormesh(
                times, frequencies, Sxx_db,
                cmap=self.colormap,
                shading='gouraud',
                alpha=0.9
            )
            
            # Configurar ejes
            self.ax.set_xlabel('Tiempo Relativo (s)', color='white', fontsize=10)
            self.ax.set_ylabel('Frecuencia (Hz)', color='white', fontsize=10)
            self.ax.set_title('Espectrograma en Tiempo Real', color='white', fontsize=12, fontweight='bold')
            
            # Limitar rango Y
            self.ax.set_ylim(self.freq_range)
            
            # Colores de ticks
            self.ax.tick_params(colors='white', which='both', labelsize=8)
            for spine in self.ax.spines.values():
                spine.set_color('white')
            
            # Grid sutil
            self.ax.grid(True, alpha=0.2, color='gray', linewidth=0.5)
            
            # Colorbar
            if hasattr(self, 'cbar'):
                self.cbar.remove()
            self.cbar = self.fig.colorbar(im, ax=self.ax)
            self.cbar.set_label('Potencia (dB)', color='white', fontsize=9)
            self.cbar.ax.tick_params(colors='white', labelsize=8)
            
            self.fig.tight_layout()
            self.draw()
            
        except Exception as e:
            print(f"Error actualizando espectrograma: {e}")
    
    def set_frequency_range(self, min_freq, max_freq):
        """Configurar rango de frecuencias"""
        self.freq_range = [min_freq, max_freq]
    
    def set_colormap(self, colormap):
        """Cambiar mapa de colores"""
        self.colormap = colormap


class FFTWidget(pg.PlotWidget):
    """Widget para mostrar FFT en tiempo real"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configurar estilo oscuro
        self.setBackground('black')
        self.setLabel('left', 'Magnitud (dB)', color='white', size='10pt')
        self.setLabel('bottom', 'Frecuencia (Hz)', color='white', size='10pt')
        self.setTitle('Análisis FFT en Tiempo Real', color='white', size='12pt')
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # Configurar rango inicial
        self.setYRange(-80, 20)  # dB
        self.setXRange(0, 1000)  # Hz
        
        # Curva para FFT
        self.fft_curve = self.plot(pen=pg.mkPen(color='#00FFFF', width=2))
        
        # Variables
        self.sample_rate = 2000
        self.data_buffer = deque(maxlen=4096)  # Buffer para FFT
        
    def update_fft(self, new_data, sample_rate=None):
        """Actualizar FFT con nuevos datos"""
        if sample_rate:
            self.sample_rate = sample_rate
            
        # Añadir datos al buffer
        if isinstance(new_data, (list, np.ndarray)) and len(new_data) > 0:
            self.data_buffer.extend(new_data)
        
        # Verificar que tenemos suficientes datos
        if len(self.data_buffer) < 512:
            return
            
        try:
            # Convertir a array
            data_array = np.array(list(self.data_buffer))
            
            # Aplicar ventana
            windowed_data = data_array * np.hanning(len(data_array))
            
            # Calcular FFT
            fft_data = np.abs(fft(windowed_data))
            frequencies = fftfreq(len(windowed_data), 1/self.sample_rate)
            
            # Solo frecuencias positivas
            pos_mask = frequencies > 0
            frequencies = frequencies[pos_mask]
            fft_data = fft_data[pos_mask]
            
            # Convertir a dB
            fft_db = 20 * np.log10(fft_data + 1e-10)
            
            # Actualizar curva
            self.fft_curve.setData(frequencies, fft_db)
            
        except Exception as e:
            print(f"Error actualizando FFT: {e}")


class AnalizadorEspectrosGUI(QMainWindow):
    """Interfaz principal del analizador de espectros"""
    
    def __init__(self):
        super().__init__()
        
        # Variables de estado
        self.sample_rate = 2000
        self.is_running = False
        self.current_channel = 0
        
        # Simulación de datos (se conectará con DAQ real)
        self.simulation_timer = QTimer()
        self.simulation_timer.timeout.connect(self.generate_test_data)
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        self.setWindowTitle('🔬 Analizador de Espectros - Estilo Analog Discovery')
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # === PANEL SUPERIOR - CONTROLES ===
        controls_frame = QtWidgets.QFrame()
        controls_frame.setMaximumHeight(120)
        controls_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Grupo de configuración
        config_group = QGroupBox("⚙️ Configuración")
        config_layout = QGridLayout(config_group)
        
        # Sample Rate
        config_layout.addWidget(QLabel("Sample Rate:"), 0, 0)
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["1000 Hz", "2000 Hz", "5000 Hz", "10000 Hz"])
        self.sample_rate_combo.setCurrentText("2000 Hz")
        self.sample_rate_combo.currentTextChanged.connect(self.update_sample_rate)
        config_layout.addWidget(self.sample_rate_combo, 0, 1)
        
        # Rango de frecuencias
        config_layout.addWidget(QLabel("Freq Min:"), 1, 0)
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0, 5000)
        self.freq_min_spin.setValue(0)
        self.freq_min_spin.setSuffix(" Hz")
        config_layout.addWidget(self.freq_min_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Freq Max:"), 2, 0)
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(1, 5000)
        self.freq_max_spin.setValue(1000)
        self.freq_max_spin.setSuffix(" Hz")
        config_layout.addWidget(self.freq_max_spin, 2, 1)
        
        # Grupo de control
        control_group = QGroupBox("🎛️ Control")
        control_layout = QVBoxLayout(control_group)
        
        self.start_btn = QPushButton("▶️ Iniciar Análisis")
        self.start_btn.setMinimumHeight(35)
        self.start_btn.clicked.connect(self.toggle_analysis)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.setMinimumHeight(35)
        self.reset_btn.clicked.connect(self.reset_analysis)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.reset_btn)
        
        # Grupo de colormaps
        colormap_group = QGroupBox("🎨 Visualización")
        colormap_layout = QVBoxLayout(colormap_group)
        
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["plasma", "viridis", "hot", "jet", "magma", "inferno"])
        self.colormap_combo.setCurrentText("plasma")
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        colormap_layout.addWidget(self.colormap_combo)
        
        controls_layout.addWidget(config_group, 2)
        controls_layout.addWidget(control_group, 1)
        controls_layout.addWidget(colormap_group, 1)
        
        main_layout.addWidget(controls_frame)
        
        # === PANEL CENTRAL - VISUALIZACIÓN ===
        # Splitter para dividir espectrograma y FFT
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        
        # Widget de espectrograma
        self.spectrogram_widget = SpectrogramWidget(self, width=14, height=8, dpi=100)
        splitter.addWidget(self.spectrogram_widget)
        
        # Widget de FFT
        self.fft_widget = FFTWidget(self)
        splitter.addWidget(self.fft_widget)
        
        # Proporciones: 70% espectrograma, 30% FFT
        splitter.setSizes([700, 300])
        
        main_layout.addWidget(splitter)
        
        # Barra de estado
        self.statusBar().showMessage("Listo para análisis espectral...")
        
    def setup_style(self):
        """Configurar estilo oscuro similar a Analog Discovery"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2a2a2a;
                color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #00FFFF;
            }
            QPushButton {
                background-color: #0078D4;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
            QComboBox, QDoubleSpinBox {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
            }
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #555555;
            }
        """)
    
    def update_sample_rate(self, text):
        """Actualizar sample rate"""
        try:
            self.sample_rate = int(text.split()[0])
            self.fft_widget.sample_rate = self.sample_rate
            self.spectrogram_widget.sample_rate = self.sample_rate
            print(f"Sample rate actualizado: {self.sample_rate} Hz")
        except:
            pass
    
    def update_colormap(self, colormap):
        """Actualizar colormap del espectrograma"""
        self.spectrogram_widget.set_colormap(colormap)
        print(f"Colormap actualizado: {colormap}")
    
    def toggle_analysis(self):
        """Iniciar/detener análisis"""
        if not self.is_running:
            # Iniciar
            self.is_running = True
            self.start_btn.setText("⏸️ Detener Análisis")
            self.start_btn.setStyleSheet("background-color: #FF5722; color: white; font-weight: bold;")
            
            # Configurar rango de frecuencias
            min_freq = self.freq_min_spin.value()
            max_freq = self.freq_max_spin.value()
            self.spectrogram_widget.set_frequency_range(min_freq, max_freq)
            self.fft_widget.setXRange(min_freq, max_freq)
            
            # Iniciar simulación (reemplazar con datos reales del DAQ)
            self.simulation_timer.start(50)  # 50ms = 20 FPS
            
            self.statusBar().showMessage("🟢 Análisis espectral activo...")
            
        else:
            # Detener
            self.is_running = False
            self.start_btn.setText("▶️ Iniciar Análisis")
            self.start_btn.setStyleSheet("background-color: #0078D4; color: white; font-weight: bold;")
            
            self.simulation_timer.stop()
            self.statusBar().showMessage("🔴 Análisis detenido")
    
    def reset_analysis(self):
        """Resetear análisis"""
        # Detener si está corriendo
        if self.is_running:
            self.toggle_analysis()
        
        # Limpiar buffers
        self.fft_widget.data_buffer.clear()
        self.spectrogram_widget.data_buffer.clear()
        self.spectrogram_widget.time_buffer.clear()
        
        # Resetear visualizaciones 
        self.spectrogram_widget.setup_spectrogram()
        
        self.statusBar().showMessage("🔄 Análisis reseteado")
    
    def generate_test_data(self):
        """Generar datos de prueba (reemplazar con datos reales del DAQ)"""
        # Simular señal con múltiples componentes de frecuencia + ruido
        t = np.linspace(0, 0.05, int(self.sample_rate * 0.05))  # 50ms de datos
        
        # Señal base con múltiples frecuencias
        signal_data = (
            0.5 * np.sin(2 * np.pi * 10 * t) +      # 10 Hz
            0.3 * np.sin(2 * np.pi * 50 * t) +      # 50 Hz
            0.2 * np.sin(2 * np.pi * 120 * t) +     # 120 Hz
            0.1 * np.sin(2 * np.pi * 200 * t) +     # 200 Hz
            0.05 * np.random.randn(len(t))           # Ruido
        )
        
        # Añadir variación temporal
        time_factor = time.time() % 10
        if time_factor < 3:
            # Añadir componente de alta frecuencia temporal
            signal_data += 0.2 * np.sin(2 * np.pi * 300 * t + time_factor)
        
        # Actualizar visualizaciones
        self.fft_widget.update_fft(signal_data, self.sample_rate)
        self.spectrogram_widget.update_spectrogram(signal_data, self.sample_rate)
    
    def connect_to_daq_data(self, data_queue):
        """Conectar con datos reales del DAQ (para integración futura)"""
        # Esta función se usará para conectar con interfaz_DAQ_V2.py
        pass


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    
    # Configurar estilo oscuro
    app.setStyle('Fusion')
    
    # Crear y mostrar ventana principal
    window = AnalizadorEspectrosGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Visualizador Simple de Datos CSV - Sin dependencias pesadas
Basado en el estilo de ProjectAD para cargar y visualizar datos capturados
"""

import sys
import os
import csv
import numpy as np
from datetime import datetime

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QGroupBox, QFileDialog, QMessageBox, QTextEdit,
                            QSplitter, QFrame)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class SimpleDataCanvas(FigureCanvas):
    """Canvas simplificado para visualización de datos"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#353535')
        
        # Crear subplots manualmente
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Fuerza - Tiempo
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Fuerza - FFT
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Vibración - Tiempo
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Vibración - FFT
        
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        
        super(SimpleDataCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        self.setup_dark_style()
        
    def setup_dark_style(self):
        """Configurar estilo oscuro"""
        plt.style.use('dark_background')
        axes_list = [self.ax1, self.ax2, self.ax3, self.ax4]
        
        for ax in axes_list:
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white', which='both')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
    
    def plot_data(self, data_fuerza=None, data_vibracion=None):
        """Plotear datos usando numpy arrays"""
        # Limpiar plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        self.setup_dark_style()
        
        colors = ['#42A5F5', '#FFA726', '#66BB6A', '#EF5350']
        
        # Plot datos de fuerza
        if data_fuerza is not None and len(data_fuerza) > 0:
            time_data = data_fuerza[:, 0]  # Primera columna = tiempo
            fs = 1.0 / (time_data[1] - time_data[0]) if len(time_data) > 1 else 1000
            
            # Serie de tiempo
            for i in range(1, data_fuerza.shape[1]):  # Saltar columna de tiempo
                self.ax1.plot(time_data, data_fuerza[:, i], 
                             color=colors[i-1], alpha=0.8, linewidth=1, label=f'CH{i}')
            
            self.ax1.set_xlabel('Tiempo (s)', color='white')
            self.ax1.set_ylabel('Fuerza (N)', color='white')
            self.ax1.set_title(f'Fuerza - Serie de Tiempo\nFs: {fs:.0f} Hz, Muestras: {len(data_fuerza):,}', 
                              color='white', fontsize=10)
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3, color='gray')
            
            # FFT de fuerza
            for i in range(1, data_fuerza.shape[1]):
                y = data_fuerza[:, i] - np.mean(data_fuerza[:, i])
                Y = np.abs(np.fft.fft(y))
                freqs = np.fft.fftfreq(len(y), 1/fs)
                
                pos_mask = freqs > 0
                self.ax2.semilogy(freqs[pos_mask], Y[pos_mask], 
                                 color=colors[i-1], alpha=0.8, linewidth=1, label=f'CH{i}')
            
            self.ax2.set_xlabel('Frecuencia (Hz)', color='white')
            self.ax2.set_ylabel('Magnitud FFT', color='white')
            self.ax2.set_title('Fuerza - Espectro FFT', color='white', fontsize=10)
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3, color='gray')
        
        # Plot datos de vibración
        if data_vibracion is not None and len(data_vibracion) > 0:
            time_data = data_vibracion[:, 0]
            fs = 1.0 / (time_data[1] - time_data[0]) if len(time_data) > 1 else 5000
            
            # Serie de tiempo
            for i in range(1, data_vibracion.shape[1]):
                self.ax3.plot(time_data, data_vibracion[:, i], 
                             color=colors[i-1], alpha=0.8, linewidth=1, label=f'CH{i}')
            
            self.ax3.set_xlabel('Tiempo (s)', color='white')
            self.ax3.set_ylabel('Vibración (g)', color='white')
            self.ax3.set_title(f'Vibración - Serie de Tiempo\nFs: {fs:.0f} Hz, Muestras: {len(data_vibracion):,}', 
                              color='white', fontsize=10)
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3, color='gray')
            
            # FFT de vibración
            for i in range(1, data_vibracion.shape[1]):
                y = data_vibracion[:, i] - np.mean(data_vibracion[:, i])
                Y = np.abs(np.fft.fft(y))
                freqs = np.fft.fftfreq(len(y), 1/fs)
                
                pos_mask = freqs > 0
                self.ax4.semilogy(freqs[pos_mask], Y[pos_mask], 
                                 color=colors[i-1], alpha=0.8, linewidth=1, label=f'CH{i}')
            
            self.ax4.set_xlabel('Frecuencia (Hz)', color='white')
            self.ax4.set_ylabel('Magnitud FFT', color='white')
            self.ax4.set_title('Vibración - Espectro FFT', color='white', fontsize=10)
            self.ax4.legend()
            self.ax4.grid(True, alpha=0.3, color='gray')
        
        self.draw()


class VisualizadorSimple(QMainWindow):
    """Interfaz simple del visualizador"""
    
    def __init__(self):
        super().__init__()
        
        self.data_fuerza = None
        self.data_vibracion = None
        self.files = {"fuerza": None, "vibracion": None}
        
        self.setup_ui()
        self.setup_style()
        self.setup_connections()
        
        self.default_dir = os.path.join(os.path.dirname(__file__), "datos_automaticos")
        if not os.path.exists(self.default_dir):
            self.default_dir = os.path.dirname(__file__)
    
    def setup_ui(self):
        """Configurar interfaz"""
        self.setWindowTitle('Visualizador Simple - Estilo ProjectAD')
        self.setMinimumSize(QSize(1200, 800))
        self.resize(1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Panel de controles
        controls_frame = QFrame()
        controls_frame.setMaximumHeight(150)
        controls_frame.setFrameStyle(QFrame.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Grupo de archivos
        file_group = QGroupBox("📂 Cargar Archivos CSV")
        file_layout = QGridLayout(file_group)
        
        self.btn_load_fuerza = QPushButton("Cargar Fuerza")
        self.btn_load_fuerza.setMinimumHeight(35)
        self.btn_load_vibracion = QPushButton("Cargar Vibración")
        self.btn_load_vibracion.setMinimumHeight(35)
        self.btn_clear = QPushButton("Limpiar")
        self.btn_clear.setMinimumHeight(35)
        
        self.lbl_fuerza = QLabel("Sin archivo")
        self.lbl_vibracion = QLabel("Sin archivo")
        
        file_layout.addWidget(self.btn_load_fuerza, 0, 0)
        file_layout.addWidget(self.lbl_fuerza, 0, 1)
        file_layout.addWidget(self.btn_load_vibracion, 1, 0)
        file_layout.addWidget(self.lbl_vibracion, 1, 1)
        file_layout.addWidget(self.btn_clear, 0, 2, 2, 1)
        
        # Grupo de estadísticas
        stats_group = QGroupBox("📊 Información")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(100)
        self.stats_text.setReadOnly(True)
        self.update_stats_display()
        
        stats_layout.addWidget(self.stats_text)
        
        controls_layout.addWidget(file_group, 2)
        controls_layout.addWidget(stats_group, 1)
        
        main_layout.addWidget(controls_frame)
        
        # Canvas y toolbar
        self.canvas = SimpleDataCanvas(self, width=12, height=8, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        
        self.statusBar().showMessage("Listo para cargar datos CSV...")
    
    def setup_style(self):
        """Configurar estilo ProjectAD"""
        self.setStyleSheet("""
            QMainWindow { background-color: #353535; color: white; }
            QGroupBox {
                font-weight: bold; border: 2px solid #555555; border-radius: 5px;
                margin-top: 10px; padding-top: 10px; background-color: #2a2a2a; color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; color: #42A5F5;
            }
            QPushButton {
                background-color: #42A5F5; border: none; color: white; padding: 8px 16px;
                border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #0D47A1; }
            QLabel { color: white; font-weight: bold; }
            QTextEdit {
                background-color: #2a2a2a; color: white; border: 1px solid #555555;
                border-radius: 4px; padding: 5px; font-family: 'Courier New'; font-size: 9pt;
            }
            QFrame { background-color: #2a2a2a; border: 1px solid #555555; }
        """)
    
    def setup_connections(self):
        """Conectar señales"""
        self.btn_load_fuerza.clicked.connect(self.load_fuerza)
        self.btn_load_vibracion.clicked.connect(self.load_vibracion)
        self.btn_clear.clicked.connect(self.clear_data)
    
    def load_csv_file(self, filename):
        """Cargar archivo CSV usando csv reader"""
        try:
            data_list = []
            with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
                # Detectar separador
                sample = csvfile.read(1024)
                csvfile.seek(0)
                
                if ',' in sample:
                    delimiter = ','
                elif ';' in sample:
                    delimiter = ';'
                else:
                    delimiter = ','
                
                reader = csv.reader(csvfile, delimiter=delimiter)
                
                # Saltar header si existe
                first_row = next(reader)
                try:
                    # Si la primera fila son números, incluirla
                    [float(x) for x in first_row]
                    data_list.append([float(x) for x in first_row])
                except ValueError:
                    # Es header, saltarla
                    pass
                
                # Leer datos numéricos
                for row in reader:
                    try:
                        numeric_row = [float(x) for x in row if x.strip()]
                        if len(numeric_row) > 0:
                            data_list.append(numeric_row)
                    except ValueError:
                        continue
            
            return np.array(data_list) if data_list else None
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error cargando archivo:\n{str(e)}")
            return None
    
    def load_fuerza(self):
        """Cargar archivo de fuerza"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de fuerza",
            self.default_dir, "CSV files (*.csv);;All files (*.*)")
        
        if file_path:
            self.data_fuerza = self.load_csv_file(file_path)
            if self.data_fuerza is not None:
                self.files["fuerza"] = file_path
                filename = os.path.basename(file_path)
                self.lbl_fuerza.setText(f"✅ {filename} ({len(self.data_fuerza):,} muestras)")
                self.update_visualization()
                self.update_stats_display()
                self.statusBar().showMessage(f"Cargado: {filename}")
    
    def load_vibracion(self):
        """Cargar archivo de vibración"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de vibración",
            self.default_dir, "CSV files (*.csv);;All files (*.*)")
        
        if file_path:
            self.data_vibracion = self.load_csv_file(file_path)
            if self.data_vibracion is not None:
                self.files["vibracion"] = file_path
                filename = os.path.basename(file_path)
                self.lbl_vibracion.setText(f"✅ {filename} ({len(self.data_vibracion):,} muestras)")
                self.update_visualization()
                self.update_stats_display()
                self.statusBar().showMessage(f"Cargado: {filename}")
    
    def clear_data(self):
        """Limpiar datos"""
        self.data_fuerza = None
        self.data_vibracion = None
        self.files = {"fuerza": None, "vibracion": None}
        self.lbl_fuerza.setText("Sin archivo")
        self.lbl_vibracion.setText("Sin archivo")
        self.update_visualization()
        self.update_stats_display()
        self.statusBar().showMessage("Datos limpiados")
    
    def update_visualization(self):
        """Actualizar gráficas"""
        self.canvas.plot_data(self.data_fuerza, self.data_vibracion)
    
    def calculate_fs(self, data):
        """Calcular frecuencia de muestreo"""
        if data is None or len(data) < 2:
            return 1000
        dt = data[1, 0] - data[0, 0]  # Primera columna = tiempo
        return 1.0 / dt if dt > 0 else 1000
    
    def update_stats_display(self):
        """Actualizar estadísticas"""
        stats = "📊 ESTADÍSTICAS:\n"
        
        if self.data_fuerza is not None:
            fs = self.calculate_fs(self.data_fuerza)
            duration = len(self.data_fuerza) / fs
            stats += f"🔋 Fuerza: {len(self.data_fuerza):,} muestras, {fs:.0f} Hz, {duration:.1f}s\n"
        
        if self.data_vibracion is not None:
            fs = self.calculate_fs(self.data_vibracion)
            duration = len(self.data_vibracion) / fs
            stats += f"📈 Vibración: {len(self.data_vibracion):,} muestras, {fs:.0f} Hz, {duration:.1f}s\n"
        
        if self.data_fuerza is None and self.data_vibracion is None:
            stats += "No hay datos cargados."
        
        self.stats_text.setText(stats)


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    
    # Estilo ProjectAD
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    
    app.setPalette(palette)
    
    font = QFont()
    font.setPointSize(10)
    font.setBold(True)
    app.setFont(font)
    
    window = VisualizadorSimple()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

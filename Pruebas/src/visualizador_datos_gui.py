#!/usr/bin/env python3
"""
Visualizador de Datos CSV - Interfaz Gráfica
Basado en el estilo de ProjectAD para cargar y visualizar datos capturados
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt5.QtCore import QSize, QTimer, Qt
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QGroupBox, QFileDialog, QMessageBox, QTextEdit,
                            QCheckBox, QProgressBar, QFrame, QSplitter)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.fft import fft, fftfreq


class DataVisualizationCanvas(FigureCanvas):
    """Canvas personalizado para visualización de datos"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Crear figure con subplots
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#353535')
        self.axes = self.fig.subplots(2, 2)
        self.fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.4)
        
        super(DataVisualizationCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # Configurar estilo oscuro para las gráficas
        self.setup_dark_style()
        
    def setup_dark_style(self):
        """Configurar estilo oscuro para matplotlib"""
        plt.style.use('dark_background')
        for ax in self.axes.flat:
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white', which='both')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
    
    def plot_data(self, data_fuerza=None, data_vibracion=None):
        """Plotear datos de fuerza y vibración"""
        # Limpiar axes
        for ax in self.axes.flat:
            ax.clear()
            ax.set_facecolor('#2a2a2a')
        
        if data_fuerza is not None:
            self.plot_dataset(data_fuerza, "Fuerza", 0, "N")
            
        if data_vibracion is not None:
            self.plot_dataset(data_vibracion, "Vibración", 1, "g")
        
        self.draw()
    
    def plot_dataset(self, data, title, row_idx, unit):
        """Plotear un dataset específico"""
        if data is None or len(data) == 0:
            return
            
        time_col = data.columns[0]
        data_cols = data.columns[1:]
        
        # Calcular frecuencia de muestreo
        if len(data) > 1:
            dt = data[time_col].iloc[1] - data[time_col].iloc[0]
            fs = 1.0 / dt if dt > 0 else 1000
        else:
            fs = 1000
        
        # Plot serie de tiempo
        ax_time = self.axes[row_idx, 0]
        colors = ['#42A5F5', '#FFA726', '#66BB6A', '#EF5350']
        
        for i, col in enumerate(data_cols):
            color = colors[i % len(colors)]
            ax_time.plot(data[time_col], data[col], 
                        label=col, color=color, alpha=0.8, linewidth=1)
        
        ax_time.set_xlabel('Tiempo (s)', color='white')
        ax_time.set_ylabel(f'{title} ({unit})', color='white')
        ax_time.set_title(f'{title} - Serie de Tiempo\nFs: {fs:.0f} Hz, Muestras: {len(data):,}', 
                         color='white', fontsize=10)
        ax_time.legend(facecolor='#353535', edgecolor='white', labelcolor='white')
        ax_time.grid(True, alpha=0.3, color='gray')
        
        # Plot FFT
        ax_fft = self.axes[row_idx, 1]
        for i, col in enumerate(data_cols):
            color = colors[i % len(colors)]
            y = data[col].values
            
            # Calcular FFT
            Y = np.abs(fft(y - np.mean(y)))
            freqs = fftfreq(len(y), 1/fs)
            
            # Solo frecuencias positivas
            pos_mask = freqs > 0
            ax_fft.semilogy(freqs[pos_mask], Y[pos_mask], 
                           label=col, color=color, alpha=0.8, linewidth=1)
        
        ax_fft.set_xlabel('Frecuencia (Hz)', color='white')
        ax_fft.set_ylabel('Magnitud FFT', color='white')
        ax_fft.set_title(f'{title} - Espectro FFT', color='white', fontsize=10)
        ax_fft.legend(facecolor='#353535', edgecolor='white', labelcolor='white')
        ax_fft.grid(True, alpha=0.3, color='gray')


class VisualizadorDatosGUI(QMainWindow):
    """Interfaz principal del visualizador de datos"""
    
    def __init__(self):
        super().__init__()
        
        # Datos
        self.data_fuerza = None
        self.data_vibracion = None
        self.current_files = {"fuerza": None, "vibracion": None}
        
        # Configurar interfaz
        self.setup_ui()
        self.setup_style()
        self.setup_connections()
        
        # Directorio por defecto
        self.default_dir = os.path.join(os.path.dirname(__file__), "datos_automaticos")
        if not os.path.exists(self.default_dir):
            self.default_dir = os.path.dirname(__file__)
    
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        self.setWindowTitle('Visualizador de Datos CSV - Estilo ProjectAD')
        self.setMinimumSize(QSize(1400, 900))
        self.resize(1600, 1000)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # === PANEL SUPERIOR - CONTROLES ===
        controls_frame = QFrame()
        controls_frame.setMaximumHeight(200)
        controls_frame.setFrameStyle(QFrame.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Grupo de carga de archivos
        file_group = QGroupBox("Cargar Archivos CSV")
        file_layout = QGridLayout(file_group)
        
        # Botones de carga
        self.btn_load_fuerza = QPushButton("📂 Cargar Fuerza")
        self.btn_load_fuerza.setMinimumHeight(40)
        self.btn_load_vibracion = QPushButton("📂 Cargar Vibración")
        self.btn_load_vibracion.setMinimumHeight(40)
        self.btn_clear_all = QPushButton("🗑️ Limpiar Todo")
        self.btn_clear_all.setMinimumHeight(40)
        
        # Labels de estado
        self.lbl_fuerza_status = QLabel("Sin archivo")
        self.lbl_vibracion_status = QLabel("Sin archivo")
        
        file_layout.addWidget(QLabel("Fuerza:"), 0, 0)
        file_layout.addWidget(self.btn_load_fuerza, 0, 1)
        file_layout.addWidget(self.lbl_fuerza_status, 0, 2)
        
        file_layout.addWidget(QLabel("Vibración:"), 1, 0)
        file_layout.addWidget(self.btn_load_vibracion, 1, 1)
        file_layout.addWidget(self.lbl_vibracion_status, 1, 2)
        
        file_layout.addWidget(self.btn_clear_all, 0, 3, 2, 1)
        
        # Grupo de análisis
        analysis_group = QGroupBox("Análisis y Exportación")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.btn_show_stats = QPushButton("📊 Mostrar Estadísticas")
        self.btn_export_analysis = QPushButton("💾 Exportar Análisis")
        
        analysis_layout.addWidget(self.btn_show_stats)
        analysis_layout.addWidget(self.btn_export_analysis)
        
        controls_layout.addWidget(file_group, 2)
        controls_layout.addWidget(analysis_group, 1)
        
        main_layout.addWidget(controls_frame)
        
        # === PANEL CENTRAL - VISUALIZACIÓN ===
        # Splitter para dividir gráficas y estadísticas
        splitter = QSplitter(Qt.Horizontal)
        
        # Canvas de visualización
        self.canvas = DataVisualizationCanvas(self, width=12, height=8, dpi=100)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        # Panel de estadísticas
        stats_widget = QWidget()
        stats_widget.setMaximumWidth(350)
        stats_layout = QVBoxLayout(stats_widget)
        
        stats_group = QGroupBox("📈 Estadísticas")
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(300)
        self.stats_text.setReadOnly(True)
        
        stats_group_layout = QVBoxLayout(stats_group)
        stats_group_layout.addWidget(self.stats_text)
        
        stats_layout.addWidget(stats_group)
        stats_layout.addStretch()
        
        splitter.addWidget(plot_widget)
        splitter.addWidget(stats_widget)
        splitter.setSizes([1000, 300])
        
        main_layout.addWidget(splitter)
        
        # Barra de estado
        self.statusBar().showMessage("Listo para cargar datos...")
    
    def setup_style(self):
        """Configurar estilo oscuro similar a ProjectAD"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
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
                color: #42A5F5;
            }
            QPushButton {
                background-color: #42A5F5;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #2a2a2a;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-family: 'Courier New';
                font-size: 9pt;
            }
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #555555;
            }
        """)
    
    def setup_connections(self):
        """Configurar conexiones de señales"""
        self.btn_load_fuerza.clicked.connect(self.load_fuerza_file)
        self.btn_load_vibracion.clicked.connect(self.load_vibracion_file)
        self.btn_clear_all.clicked.connect(self.clear_all_data)
        self.btn_show_stats.clicked.connect(self.show_statistics)
        self.btn_export_analysis.clicked.connect(self.export_analysis)
    
    def load_fuerza_file(self):
        """Cargar archivo de fuerza"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de fuerza",
            self.default_dir,
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.data_fuerza = pd.read_csv(file_path)
                self.current_files["fuerza"] = file_path
                filename = os.path.basename(file_path)
                self.lbl_fuerza_status.setText(f"✅ {filename} ({len(self.data_fuerza):,} muestras)")
                self.statusBar().showMessage(f"Cargado: {filename}")
                self.update_visualization()
                self.update_statistics()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{str(e)}")
    
    def load_vibracion_file(self):
        """Cargar archivo de vibración"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo de vibración",
            self.default_dir,
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.data_vibracion = pd.read_csv(file_path)
                self.current_files["vibracion"] = file_path
                filename = os.path.basename(file_path)
                self.lbl_vibracion_status.setText(f"✅ {filename} ({len(self.data_vibracion):,} muestras)")
                self.statusBar().showMessage(f"Cargado: {filename}")
                self.update_visualization()
                self.update_statistics()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{str(e)}")
    
    def clear_all_data(self):
        """Limpiar todos los datos"""
        reply = QMessageBox.question(self, "Confirmar", 
                                   "¿Limpiar todos los datos cargados?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.data_fuerza = None
            self.data_vibracion = None
            self.current_files = {"fuerza": None, "vibracion": None}
            self.lbl_fuerza_status.setText("Sin archivo")
            self.lbl_vibracion_status.setText("Sin archivo")
            self.update_visualization()
            self.update_statistics()
            self.statusBar().showMessage("Datos limpiados")
    
    def update_visualization(self):
        """Actualizar visualización de gráficas"""
        self.canvas.plot_data(self.data_fuerza, self.data_vibracion)
    
    def calculate_sample_rate(self, data):
        """Calcular frecuencia de muestreo"""
        if data is None or len(data) < 2:
            return 1000
        
        time_col = data.columns[0]
        dt = data[time_col].iloc[1] - data[time_col].iloc[0]
        return 1.0 / dt if dt > 0 else 1000
    
    def update_statistics(self):
        """Actualizar panel de estadísticas"""
        stats_text = "📊 ESTADÍSTICAS DE DATOS\n"
        stats_text += "=" * 30 + "\n\n"
        
        if self.data_fuerza is not None:
            fs = self.calculate_sample_rate(self.data_fuerza)
            duration = len(self.data_fuerza) / fs
            stats_text += f"🔋 FUERZA:\n"
            stats_text += f"  Muestras: {len(self.data_fuerza):,}\n"
            stats_text += f"  Frecuencia: {fs:.0f} Hz\n"
            stats_text += f"  Duración: {duration:.1f} s\n"
            
            for col in self.data_fuerza.columns[1:]:
                data_col = self.data_fuerza[col]
                stats_text += f"  {col}: μ={data_col.mean():.3f}, σ={data_col.std():.3f}\n"
            stats_text += "\n"
        
        if self.data_vibracion is not None:
            fs = self.calculate_sample_rate(self.data_vibracion)
            duration = len(self.data_vibracion) / fs
            stats_text += f"📈 VIBRACIÓN:\n"
            stats_text += f"  Muestras: {len(self.data_vibracion):,}\n"
            stats_text += f"  Frecuencia: {fs:.0f} Hz\n"
            stats_text += f"  Duración: {duration:.1f} s\n"
            
            for col in self.data_vibracion.columns[1:]:
                data_col = self.data_vibracion[col]
                stats_text += f"  {col}: μ={data_col.mean():.3f}, σ={data_col.std():.3f}\n"
        
        if self.data_fuerza is None and self.data_vibracion is None:
            stats_text += "No hay datos cargados.\n"
            stats_text += "Use los botones de carga para\n"
            stats_text += "seleccionar archivos CSV."
        
        self.stats_text.setText(stats_text)
    
    def show_statistics(self):
        """Mostrar estadísticas detalladas"""
        if self.data_fuerza is None and self.data_vibracion is None:
            QMessageBox.information(self, "Sin datos", "No hay datos cargados para mostrar estadísticas.")
            return
        
        stats_dialog = QMessageBox(self)
        stats_dialog.setWindowTitle("Estadísticas Detalladas")
        stats_dialog.setIcon(QMessageBox.Information)
        
        detailed_stats = self.generate_detailed_statistics()
        stats_dialog.setText(detailed_stats)
        stats_dialog.exec_()
    
    def generate_detailed_statistics(self):
        """Generar estadísticas detalladas"""
        stats = "ESTADÍSTICAS DETALLADAS\n" + "="*50 + "\n\n"
        
        if self.data_fuerza is not None:
            stats += "DATOS DE FUERZA:\n"
            stats += f"Archivo: {os.path.basename(self.current_files['fuerza'])}\n"
            stats += str(self.data_fuerza.describe()) + "\n\n"
        
        if self.data_vibracion is not None:
            stats += "DATOS DE VIBRACIÓN:\n"
            stats += f"Archivo: {os.path.basename(self.current_files['vibracion'])}\n"
            stats += str(self.data_vibracion.describe()) + "\n\n"
        
        return stats
    
    def export_analysis(self):
        """Exportar análisis a archivo"""
        if self.data_fuerza is None and self.data_vibracion is None:
            QMessageBox.information(self, "Sin datos", "No hay datos para exportar.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analisis_datos_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("ANÁLISIS DE DATOS CAPTURADOS\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(self.generate_detailed_statistics())
            
            QMessageBox.information(self, "Exportado", f"Análisis exportado a:\n{filename}")
            self.statusBar().showMessage(f"Análisis exportado: {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo exportar el análisis:\n{str(e)}")


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    
    # Configurar estilo similar a ProjectAD
    app.setStyle('Fusion')
    
    # Paleta de colores oscura
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
    
    # Fuente
    font = QFont()
    font.setPointSize(10)
    font.setBold(True)
    app.setFont(font)
    
    # Crear y mostrar ventana principal
    window = VisualizadorDatosGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

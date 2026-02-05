#!/usr/bin/env python3
"""
Visor CSV Básico - Sin pandas, solo numpy y matplotlib
Estilo ProjectAD para visualizar datos de fuerza y vibración
"""

import sys
import os
import numpy as np
from datetime import datetime

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                            QGroupBox, QFileDialog, QMessageBox, QTextEdit)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class CSVViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.data_fuerza = None
        self.data_vibracion = None
        
        self.setWindowTitle('Visor CSV - Datos de Fuerza y Vibración')
        self.setMinimumSize(QSize(1200, 700))
        self.resize(1400, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        layout = QVBoxLayout(central_widget)
        
        # === CONTROLES ===
        controls_group = QGroupBox("📂 Carga de Archivos")
        controls_layout = QHBoxLayout(controls_group)
        
        self.btn_fuerza = QPushButton("🔋 Cargar Fuerza")
        self.btn_fuerza.setMinimumHeight(40)
        self.btn_fuerza.clicked.connect(self.cargar_fuerza)
        
        self.btn_vibracion = QPushButton("📈 Cargar Vibración")
        self.btn_vibracion.setMinimumHeight(40)
        self.btn_vibracion.clicked.connect(self.cargar_vibracion)
        
        self.btn_limpiar = QPushButton("🗑️ Limpiar")
        self.btn_limpiar.setMinimumHeight(40)
        self.btn_limpiar.clicked.connect(self.limpiar_datos)
        
        self.lbl_estado = QLabel("Listo para cargar archivos CSV...")
        
        controls_layout.addWidget(self.btn_fuerza)
        controls_layout.addWidget(self.btn_vibracion)
        controls_layout.addWidget(self.btn_limpiar)
        controls_layout.addWidget(self.lbl_estado)
        
        layout.addWidget(controls_group)
        
        # === GRÁFICAS ===
        # Figure con 4 subplots
        self.figure = Figure(figsize=(14, 8), facecolor='#353535')
        self.canvas = FigureCanvas(self.figure)
        
        # Toolbar de navegación con controles de zoom, pan, etc.
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar { 
                background-color: #2a2a2a; 
                border: 1px solid #555; 
                spacing: 3px;
                padding: 5px;
            }
            QToolButton { 
                background-color: #42A5F5; 
                border: 1px solid #1976D2; 
                border-radius: 3px; 
                padding: 5px;
                margin: 2px;
            }
            QToolButton:hover { 
                background-color: #1976D2; 
            }
            QToolButton:pressed { 
                background-color: #0D47A1; 
            }
        """)
        
        # Crear subplots
        self.ax1 = self.figure.add_subplot(2, 2, 1)  # Fuerza tiempo
        self.ax2 = self.figure.add_subplot(2, 2, 2)  # Fuerza FFT  
        self.ax3 = self.figure.add_subplot(2, 2, 3)  # Vibración tiempo
        self.ax4 = self.figure.add_subplot(2, 2, 4)  # Vibración FFT
        
        self.figure.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.95, wspace=0.25, hspace=0.35)
        
        # Estilo oscuro
        self.setup_dark_style()
        
        # Agregar toolbar y canvas al layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # === INFORMACIÓN ===
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(120)
        self.info_text.setReadOnly(True)
        self.actualizar_info()
        
        layout.addWidget(self.info_text)
        
        # Aplicar estilo
        self.aplicar_estilo_projectad()
        
        # Directorio por defecto
        self.directorio_datos = os.path.join(os.path.dirname(__file__), "datos_automaticos")
        if not os.path.exists(self.directorio_datos):
            self.directorio_datos = os.path.dirname(__file__)
    
    def setup_dark_style(self):
        """Configurar matplotlib con estilo oscuro"""
        axes = [self.ax1, self.ax2, self.ax3, self.ax4]
        
        for ax in axes:
            ax.set_facecolor('#2a2a2a')
            ax.tick_params(colors='white', which='both', labelsize=9)
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.grid(True, alpha=0.3, color='gray')
    
    def cargar_csv(self, archivo):
        """Cargar archivo CSV usando numpy"""
        try:
            # Intentar cargar con numpy
            data = np.loadtxt(archivo, delimiter=',', skiprows=1)
            return data
        except:
            try:
                # Intentar con separador punto y coma
                data = np.loadtxt(archivo, delimiter=';', skiprows=1)
                return data
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{str(e)}")
                return None
    
    def cargar_fuerza(self):
        """Cargar archivo de fuerza"""
        archivo, _ = QFileDialog.getOpenFileName(
            self, "Cargar datos de fuerza",
            self.directorio_datos,
            "CSV files (*.csv);;All files (*.*)")
        
        if archivo:
            self.data_fuerza = self.cargar_csv(archivo)
            if self.data_fuerza is not None:
                nombre = os.path.basename(archivo)
                self.lbl_estado.setText(f"✅ Fuerza: {nombre} ({len(self.data_fuerza):,} muestras)")
                self.graficar_datos()
                self.actualizar_info()
    
    def cargar_vibracion(self):
        """Cargar archivo de vibración"""
        archivo, _ = QFileDialog.getOpenFileName(
            self, "Cargar datos de vibración",
            self.directorio_datos,
            "CSV files (*.csv);;All files (*.*)")
        
        if archivo:
            self.data_vibracion = self.cargar_csv(archivo)
            if self.data_vibracion is not None:
                nombre = os.path.basename(archivo)
                self.lbl_estado.setText(f"✅ Vibración: {nombre} ({len(self.data_vibracion):,} muestras)")
                self.graficar_datos()
                self.actualizar_info()
    
    def limpiar_datos(self):
        """Limpiar todos los datos"""
        self.data_fuerza = None
        self.data_vibracion = None
        
        # Limpiar gráficas
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.setup_dark_style()
        self.canvas.draw()
        
        self.lbl_estado.setText("Datos limpiados")
        self.actualizar_info()
    
    def graficar_datos(self):
        """Crear todas las gráficas"""
        # Limpiar
        self.ax1.clear()
        self.ax2.clear() 
        self.ax3.clear()
        self.ax4.clear()
        self.setup_dark_style()
        
        colores = ['#42A5F5', '#FFA726', '#66BB6A', '#EF5350']
        
        # === FUERZA ===
        if self.data_fuerza is not None:
            tiempo = self.data_fuerza[:, 0]
            fs_fuerza = 1.0 / (tiempo[1] - tiempo[0]) if len(tiempo) > 1 else 1000
            
            # Serie de tiempo fuerza
            for i in range(1, self.data_fuerza.shape[1]):
                self.ax1.plot(tiempo, self.data_fuerza[:, i], 
                             color=colores[i-1], alpha=0.8, linewidth=1, label=f'CH{i}')
            
            self.ax1.set_xlabel('Tiempo (s)')
            self.ax1.set_ylabel('Fuerza (N)')
            self.ax1.set_title(f'Fuerza - Tiempo\n{fs_fuerza:.0f} Hz - {len(self.data_fuerza):,} muestras', fontsize=10)
            self.ax1.legend(fontsize=8)
            
            # FFT fuerza
            for i in range(1, self.data_fuerza.shape[1]):
                y = self.data_fuerza[:, i] - np.mean(self.data_fuerza[:, i])
                Y = np.abs(np.fft.fft(y))
                freqs = np.fft.fftfreq(len(y), 1/fs_fuerza)
                
                pos = freqs > 0
                self.ax2.semilogy(freqs[pos], Y[pos], 
                                 color=colores[i-1], alpha=0.8, linewidth=1, label=f'CH{i}')
            
            self.ax2.set_xlabel('Frecuencia (Hz)')
            self.ax2.set_ylabel('Magnitud FFT')
            self.ax2.set_title('Fuerza - FFT', fontsize=10)
            self.ax2.legend(fontsize=8)
        
        # === VIBRACIÓN ===  
        if self.data_vibracion is not None:
            tiempo = self.data_vibracion[:, 0]
            fs_vib = 1.0 / (tiempo[1] - tiempo[0]) if len(tiempo) > 1 else 5000
            
            # Serie de tiempo vibración
            for i in range(1, self.data_vibracion.shape[1]):
                self.ax3.plot(tiempo, self.data_vibracion[:, i],
                             color=colores[i-1], alpha=0.8, linewidth=1, label=f'CH{i}')
            
            self.ax3.set_xlabel('Tiempo (s)')
            self.ax3.set_ylabel('Vibración (g)')
            self.ax3.set_title(f'Vibración - Tiempo\n{fs_vib:.0f} Hz - {len(self.data_vibracion):,} muestras', fontsize=10)
            self.ax3.legend(fontsize=8)
            
            # FFT vibración
            for i in range(1, self.data_vibracion.shape[1]):
                y = self.data_vibracion[:, i] - np.mean(self.data_vibracion[:, i])
                Y = np.abs(np.fft.fft(y))
                freqs = np.fft.fftfreq(len(y), 1/fs_vib)
                
                pos = freqs > 0
                self.ax4.semilogy(freqs[pos], Y[pos],
                                 color=colores[i-1], alpha=0.8, linewidth=1, label=f'CH{i}')
            
            self.ax4.set_xlabel('Frecuencia (Hz)')
            self.ax4.set_ylabel('Magnitud FFT') 
            self.ax4.set_title('Vibración - FFT', fontsize=10)
            self.ax4.legend(fontsize=8)
        
        self.canvas.draw()
    
    def actualizar_info(self):
        """Actualizar información mostrada"""
        info = "📊 INFORMACIÓN DE DATOS CARGADOS\n"
        info += "=" * 45 + "\n"
        
        if self.data_fuerza is not None:
            tiempo = self.data_fuerza[:, 0]
            fs = 1.0 / (tiempo[1] - tiempo[0]) if len(tiempo) > 1 else 1000
            duracion = len(self.data_fuerza) / fs
            info += f"🔋 FUERZA: {len(self.data_fuerza):,} muestras | {fs:.0f} Hz | {duracion:.1f} segundos\n"
            
            for i in range(1, self.data_fuerza.shape[1]):
                canal = self.data_fuerza[:, i]
                info += f"   CH{i}: μ={canal.mean():.3f}, σ={canal.std():.3f}, min={canal.min():.3f}, max={canal.max():.3f}\n"
        
        if self.data_vibracion is not None:
            tiempo = self.data_vibracion[:, 0]
            fs = 1.0 / (tiempo[1] - tiempo[0]) if len(tiempo) > 1 else 5000
            duracion = len(self.data_vibracion) / fs
            info += f"📈 VIBRACIÓN: {len(self.data_vibracion):,} muestras | {fs:.0f} Hz | {duracion:.1f} segundos\n"
            
            for i in range(1, self.data_vibracion.shape[1]):
                canal = self.data_vibracion[:, i]
                info += f"   CH{i}: μ={canal.mean():.3f}, σ={canal.std():.3f}, min={canal.min():.3f}, max={canal.max():.3f}\n"
        
        if self.data_fuerza is None and self.data_vibracion is None:
            info += "No hay datos cargados. Use los botones de arriba para cargar archivos CSV."
        
        self.info_text.setText(info)
    
    def aplicar_estilo_projectad(self):
        """Aplicar estilo similar a ProjectAD"""
        self.setStyleSheet("""
            QMainWindow { background-color: #353535; color: white; }
            QGroupBox {
                font-weight: bold; border: 2px solid #555; border-radius: 5px;
                margin-top: 10px; padding-top: 10px; background-color: #2a2a2a; color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #42A5F5;
            }
            QPushButton {
                background-color: #42A5F5; border: none; color: white; 
                padding: 10px 20px; border-radius: 5px; font-weight: bold; font-size: 11px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #0D47A1; }
            QLabel { color: white; font-weight: bold; font-size: 11px; }
            QTextEdit {
                background-color: #2a2a2a; color: white; border: 1px solid #555;
                border-radius: 4px; padding: 8px; font-family: 'Consolas', 'Courier New';
                font-size: 10px; font-weight: bold;
            }
        """)


def main():
    app = QApplication(sys.argv)
    
    # Configurar estilo Fusion + colores oscuros
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
    
    # Fuente
    font = QFont()
    font.setPointSize(10)
    font.setBold(True)
    app.setFont(font)
    
    # Crear y mostrar ventana
    viewer = CSVViewer()
    viewer.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

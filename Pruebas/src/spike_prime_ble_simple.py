import sys
import asyncio
import platform
from bleak import BleakScanner, BleakClient
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, 
                           QLabel, QComboBox, QWidget, QMessageBox, QHBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import numpy as np
import pyqtgraph as pg

# Configuración de UUIDs (reemplazar con los del SPIKE Prime)
SPIKE_SERVICE_UUID = "00001623-1212-efde-1623-785feabcd123"
ACCEL_CHAR_UUID   = "00001624-1212-efde-1623-785feabcd123"

class BLEScanner(QThread):
    """Hilo para escanear dispositivos BLE."""
    devices_found = pyqtSignal(list)
    
    def run(self):
        try:
            # Configurar el bucle de eventos para Windows
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            # Crear un nuevo bucle de eventos
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ejecutar el escaneo
            devices = loop.run_until_complete(self._scan())
            self.devices_found.emit(devices)
            
        except Exception as e:
            print(f"Error en BLEScanner: {e}")
            self.devices_found.emit([])
    
    async def _scan(self):
        """Función asíncrona para escanear dispositivos."""
        print("Iniciando escaneo BLE...")
        devices = await BleakScanner.discover(timeout=10.0)
        print(f"Dispositivos encontrados: {len(devices)}")
        return devices

class SpikePrimeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.client = None
        self.scan_thread = None
        self.setWindowTitle("SPIKE Prime BLE Controller")
        self.setGeometry(100, 100, 800, 600)
        
        # Configuración de la ventana
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Controles superiores
        top_layout = QHBoxLayout()
        self.btn_scan = QPushButton("🔍 Escanear")
        self.btn_scan.setStyleSheet("font-size: 14px; padding: 8px;")
        self.btn_scan.clicked.connect(self.start_scan)
        
        self.cb_devices = QComboBox()
        self.cb_devices.setStyleSheet("font-size: 14px; padding: 8px; min-width: 300px;")
        self.cb_devices.setPlaceholderText("Selecciona un dispositivo")
        
        self.btn_connect = QPushButton("🚀 Conectar")
        self.btn_connect.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_connect.setEnabled(False)
        
        top_layout.addWidget(self.btn_scan)
        top_layout.addWidget(self.cb_devices)
        top_layout.addWidget(self.btn_connect)
        
        # Gráfico
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Acelerómetro SPIKE Prime", color="#333333", size="14pt")
        self.plot_widget.setLabel('left', 'Aceleración (g)')
        self.plot_widget.setLabel('bottom', 'Muestras')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen(color='#2196F3', width=2))
        self.plot_data = np.zeros(100)
        
        # Estado
        self.lbl_status = QLabel("Estado: Desconectado")
        self.lbl_status.setStyleSheet("font-size: 12px; color: #666666;")
        
        # Layout
        layout.addLayout(top_layout)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.lbl_status)
        
        # Timer para actualizar la interfaz
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)  # 20 FPS
        
        # Iniciar escaneo automático
        QTimer.singleShot(1000, self.start_scan)
    
    def start_scan(self):
        """Inicia el escaneo de dispositivos BLE."""
        if self.scan_thread and self.scan_thread.isRunning():
            return
            
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("Escaneando...")
        self.cb_devices.clear()
        self.lbl_status.setText("Escaneando dispositivos BLE...")
        
        self.scan_thread = BLEScanner()
        self.scan_thread.devices_found.connect(self.on_devices_found)
        self.scan_thread.finished.connect(self.on_scan_finished)
        self.scan_thread.start()
    
    def on_scan_finished(self):
        """Se llama cuando termina el escaneo."""
        self.btn_scan.setEnabled(True)
        self.btn_scan.setText("🔍 Escanear")
    
    def on_devices_found(self, devices):
        """Maneja los dispositivos encontrados."""
        self.cb_devices.clear()
        
        if not devices:
            self.lbl_status.setText("No se encontraron dispositivos BLE")
            return
            
        for device in devices:
            name = device.name if device.name else "Dispositivo desconocido"
            self.cb_devices.addItem(f"{name} ({device.address})", device.address)
        
        self.btn_connect.setEnabled(len(devices) > 0)
        self.lbl_status.setText(f"Se encontraron {len(devices)} dispositivos")
    
    def toggle_connection(self):
        """Conecta/desconecta el dispositivo seleccionado."""
        if self.client and self.client.is_connected:
            asyncio.create_task(self.disconnect_device())
        else:
            self.connect_device()
    
    async def disconnect_device(self):
        """Desconecta el dispositivo BLE."""
        try:
            if self.client:
                await self.client.disconnect()
                self.lbl_status.setText("Desconectado")
                self.btn_connect.setText("🚀 Conectar")
                self.btn_connect.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                    }
                    QPushButton:disabled {
                        background-color: #cccccc;
                    }
                """)
        except Exception as e:
            self.lbl_status.setText(f"Error al desconectar: {str(e)}")
    
    def connect_device(self):
        """Inicia la conexión al dispositivo BLE."""
        address = self.cb_devices.currentData()
        if not address:
            QMessageBox.warning(self, "Error", "Por favor selecciona un dispositivo primero")
            return
            
        self.lbl_status.setText(f"Conectando a {address}...")
        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Conectando...")
        
        # Usar un hilo separado para la conexión
        self.connection_thread = ConnectionThread(address)
        self.connection_thread.connection_result.connect(self.on_connection_result)
        self.connection_thread.start()
    
    def on_connection_result(self, result):
        """Maneja el resultado de la conexión."""
        success, message, client = result
        self.client = client
        
        if success:
            self.lbl_status.setText(f"Conectado a {self.cb_devices.currentText()}")
            self.btn_connect.setText("🔌 Desconectar")
            self.btn_connect.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                }
            """)
        else:
            self.lbl_status.setText(f"Error de conexión: {message}")
            self.btn_connect.setText("🚀 Conectar")
            self.btn_connect.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                }
            """)
        
        self.btn_connect.setEnabled(True)
    
    def update_ui(self):
        """Actualiza la interfaz de usuario."""
        if hasattr(self, 'plot_curve'):
            # Simular datos si no hay conexión
            if not self.client or not self.client.is_connected:
                self.plot_data = np.roll(self.plot_data, -1)
                self.plot_data[-1] = np.random.normal(0, 0.1)
                
            self.plot_curve.setData(self.plot_data)
    
    def closeEvent(self, event):
        """Limpia los recursos al cerrar la aplicación."""
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.quit()
            self.scan_thread.wait()
            
        if self.client and self.client.is_connected:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.client.disconnect())
            
        event.accept()

class ConnectionThread(QThread):
    """Hilo para manejar la conexión BLE."""
    connection_result = pyqtSignal(tuple)  # (success, message, client)
    
    def __init__(self, address):
        super().__init__()
        self.address = address
    
    def run(self):
        try:
            # Configurar el bucle de eventos para Windows
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            # Crear un nuevo bucle de eventos
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ejecutar la conexión
            client = loop.run_until_complete(self._connect())
            self.connection_result.emit((True, "Conexión exitosa", client))
            
        except Exception as e:
            self.connection_result.emit((False, str(e), None))
    
    async def _connect(self):
        """Establece la conexión BLE."""
        client = BleakClient(self.address)
        await client.connect(timeout=10.0)
        return client

def main():
    # Configura el bucle de eventos para Windows
    if platform.system() == "Windows":
        import sys
        if sys.version_info >= (3, 8) and sys.platform.lower().startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    app = QApplication(sys.argv)
    
    # Establecer estilo
    app.setStyle("Fusion")
    
    window = SpikePrimeApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

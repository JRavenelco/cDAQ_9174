import asyncio
import sys
import platform
from bleak import BleakScanner, BleakClient
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QComboBox, QWidget, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import numpy as np
import pyqtgraph as pg

# Configuración de UUIDs (reemplaza con los del SPIKE Prime)
SPIKE_SERVICE_UUID = "00001623-1212-efde-1623-785feabcd123"
ACCEL_CHAR_UUID   = "00001624-1212-efde-1623-785feabcd123"

class BLEScanner(QThread):
    """Hilo para escanear dispositivos BLE."""
    devices_found = pyqtSignal(list)
    
    def run(self):
        asyncio.run(self._scan())
    
    async def _scan(self):
        try:
            devices = await BleakScanner.discover(timeout=5.0)
            self.devices_found.emit(devices)
        except Exception as e:
            print(f"Error en escaneo BLE: {e}")
            self.devices_found.emit([])

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
        
        # Controles
        self.btn_scan = QPushButton("Escanear dispositivos")
        self.btn_scan.clicked.connect(self.start_scan)
        
        self.cb_devices = QComboBox()
        self.cb_devices.setPlaceholderText("Selecciona un dispositivo")
        
        self.btn_connect = QPushButton("Conectar")
        self.btn_connect.clicked.connect(self.toggle_connection)
        self.btn_connect.setEnabled(False)
        
        # Gráfico
        self.plot_widget = pg.PlotWidget()
        self.plot_curve = self.plot_widget.plot(pen='y')
        self.plot_data = np.zeros(100)
        
        # Estado
        self.lbl_status = QLabel("Estado: Desconectado")
        
        # Layout
        layout.addWidget(self.btn_scan)
        layout.addWidget(self.cb_devices)
        layout.addWidget(self.btn_connect)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.lbl_status)
        
        # Timer para actualizar la interfaz
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)
    
    def start_scan(self):
        """Inicia el escaneo de dispositivos BLE."""
        self.btn_scan.setEnabled(False)
        self.cb_devices.clear()
        self.lbl_status.setText("Escaneando...")
        
        self.scan_thread = BLEScanner()
        self.scan_thread.devices_found.connect(self.on_devices_found)
        self.scan_thread.finished.connect(lambda: self.btn_scan.setEnabled(True))
        self.scan_thread.start()
    
    def on_devices_found(self, devices):
        """Maneja los dispositivos encontrados."""
        self.cb_devices.clear()
        if not devices:
            self.lbl_status.setText("No se encontraron dispositivos")
            return
            
        for device in devices:
            name = device.name if device.name else "Dispositivo desconocido"
            self.cb_devices.addItem(f"{name} ({device.address})", device.address)
        
        self.btn_connect.setEnabled(len(devices) > 0)
        self.lbl_status.setText(f"Encontrados {len(devices)} dispositivos")
    
    def toggle_connection(self):
        """Conecta/desconecta el dispositivo seleccionado."""
        if self.client and self.client.is_connected:
            asyncio.create_task(self.disconnect_device())
        else:
            self.connect_device()
    
    async def disconnect_device(self):
        """Desconecta el dispositivo BLE."""
        if self.client:
            await self.client.disconnect()
            self.lbl_status.setText("Desconectado")
            self.btn_connect.setText("Conectar")
    
    def connect_device(self):
        """Inicia la conexión al dispositivo BLE."""
        address = self.cb_devices.currentData()
        if not address:
            QMessageBox.warning(self, "Error", "Selecciona un dispositivo primero")
            return
            
        self.lbl_status.setText(f"Conectando a {address}...")
        asyncio.create_task(self._connect_ble(address))
    
    async def _connect_ble(self, address):
        """Conexión asíncrona al dispositivo BLE."""
        try:
            self.client = BleakClient(address)
            await self.client.connect()
            self.lbl_status.setText(f"Conectado a {address}")
            self.btn_connect.setText("Desconectar")
            
            # Iniciar notificaciones (ajusta el UUID según corresponda)
            await self.client.start_notify(ACCEL_CHAR_UUID, self.handle_notification)
            
        except Exception as e:
            self.lbl_status.setText(f"Error de conexión: {e}")
    
    def handle_notification(self, sender, data):
        """Maneja las notificaciones BLE."""
        # Procesa los datos del acelerómetro
        try:
            # Ajusta según el formato de datos del SPIKE Prime
            values = np.frombuffer(data, dtype=np.int16) / 1000.0  # Ajusta la escala según sea necesario
            if len(values) >= 3:  # Asumiendo 3 ejes (x, y, z)
                self.plot_data = np.roll(self.plot_data, -1)
                self.plot_data[-1] = np.linalg.norm(values[:3])  # Magnitud del vector
        except Exception as e:
            print(f"Error procesando datos: {e}")
    
    def update_ui(self):
        """Actualiza la interfaz de usuario."""
        if hasattr(self, 'plot_curve'):
            self.plot_curve.setData(self.plot_data)
    
    def closeEvent(self, event):
        """Limpia los recursos al cerrar la aplicación."""
        if self.client and self.client.is_connected:
            asyncio.create_task(self.disconnect_device())
        event.accept()

def main():
    # Configura el bucle de eventos para Windows
    if platform.system() == "Windows":
        import sys
        if sys.version_info >= (3, 8) and sys.platform.lower().startswith("win"):
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    app = QApplication(sys.argv)
    window = SpikePrimeApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

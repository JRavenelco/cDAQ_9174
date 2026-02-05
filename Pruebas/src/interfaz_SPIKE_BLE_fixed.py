import sys
import asyncio
import threading
import queue
import os

# Asegurar que el directorio lib esté en el path
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if lib_path not in sys.path:
    sys.path.append(lib_path)

try:
    from bleak import BleakClient, BleakScanner
except ImportError as e:
    print(f"Error al importar Bleak: {e}")
    print("Asegúrate de instalar las dependencias con: pip install bleak pyqtgraph PyQt5")
    print("Si usas una versión reciente de Python, prueba con: python -m pip install --upgrade pip")
    sys.exit(1)

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


class SpikeBLEThread(threading.Thread):
    """Hilo para suscribirse a notificaciones BLE del Hub SPIKE Prime."""
    def __init__(self, address: str, char_uuid: str, data_queue: queue.Queue):
        super().__init__()
        self.address = address
        self.char_uuid = char_uuid
        self.data_queue = data_queue
        self.running = False

    def run(self):
        asyncio.run(self._ble_loop())

    async def _ble_loop(self):
        try:
            async with BleakClient(self.address) as client:
                self.running = True
                await client.start_notify(self.char_uuid, self._notification_handler)
                while self.running:
                    await asyncio.sleep(0.1)
                await client.stop_notify(self.char_uuid)
        except Exception as e:
            print(f"Error en conexión BLE: {e}")

    def _notification_handler(self, sender, data: bytearray):
        try:
            vals = np.frombuffer(data, dtype=np.int16).astype(float)
            self.data_queue.put(vals)
        except Exception as e:
            print(f"Error procesando datos BLE: {e}")

    def stop(self):
        self.running = False


class SpikePrimeBLEApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spike Prime BLE Interface")
        self.queue = queue.Queue()
        self.thread = None
        self.connected = False
        self.data_buffer = np.zeros(512)  # Buffer para datos
        
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QGridLayout(self)
        
        # Controles BLE
        self.btn_scan = QtWidgets.QPushButton("Escanear BLE")
        self.cb_devices = QtWidgets.QComboBox()
        self.btn_connect = QtWidgets.QPushButton("Conectar")
        self.le_uuid = QtWidgets.QLineEdit()
        self.le_uuid.setPlaceholderText("UUID de característica")
        
        # Gráfica
        self.plot = pg.PlotWidget(title="Datos del Acelerómetro")
        self.curve = self.plot.plot(pen='y')
        
        # Estado
        self.status = QtWidgets.QLabel("Desconectado")
        
        # Layout
        layout.addWidget(self.btn_scan, 0, 0)
        layout.addWidget(self.cb_devices, 0, 1, 1, 2)
        layout.addWidget(self.btn_connect, 0, 3)
        layout.addWidget(self.le_uuid, 0, 4)
        layout.addWidget(self.plot, 1, 0, 1, 5)
        layout.addWidget(self.status, 2, 0, 1, 5)
        
        # Conexiones
        self.btn_scan.clicked.connect(self.scan_devices)
        self.btn_connect.clicked.connect(self.toggle_connection)
        
        # Timer para actualizar gráfica
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

    async def _scan_devices_async(self):
        """Escanea dispositivos BLE de forma asíncrona."""
        devices = await BleakScanner.discover()
        return devices

    def scan_devices(self):
        """Busca dispositivos BLE cercanos."""
        self.cb_devices.clear()
        self.status.setText("Escaneando...")
        
        # Ejecutar escaneo en un hilo separado
        def scan_done(future):
            try:
                devices = future.result()
                for d in devices:
                    name = d.name or 'Desconocido'
                    self.cb_devices.addItem(f"{name} ({d.address})", d.address)
                self.status.setText(f"Encontrados {len(devices)} dispositivos")
            except Exception as e:
                self.status.setText(f"Error en escaneo: {e}")
        
        # Crear un nuevo bucle de eventos para el escaneo
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = asyncio.ensure_future(self._scan_devices_async())
        future.add_done_callback(scan_done)
        loop.run_until_complete(future)

    def toggle_connection(self):
        """Conecta/desconecta del dispositivo seleccionado."""
        if not self.connected:
            addr = self.cb_devices.currentData()
            uuid = self.le_uuid.text().strip()
            
            if not addr or not uuid:
                QtWidgets.QMessageBox.warning(
                    self, 
                    "Error", 
                    "Selecciona un dispositivo e ingresa el UUID de característica"
                )
                return
                
            try:
                self.thread = SpikeBLEThread(addr, uuid, self.queue)
                self.thread.start()
                self.connected = True
                self.btn_connect.setText("Desconectar")
                self.status.setText(f"Conectado a {addr}")
            except Exception as e:
                self.status.setText(f"Error de conexión: {e}")
        else:
            if self.thread:
                self.thread.stop()
                self.thread.join(timeout=1.0)
                self.thread = None
            self.connected = False
            self.btn_connect.setText("Conectar")
            self.status.setText("Desconectado")

    def update_plot(self):
        """Actualiza la gráfica con nuevos datos BLE."""
        try:
            while True:
                vals = self.queue.get_nowait()
                # Actualizar buffer con los nuevos datos
                n = len(vals)
                self.data_buffer = np.roll(self.data_buffer, -n)
                self.data_buffer[-n:] = vals
            self.curve.setData(self.data_buffer)
        except queue.Empty:
            pass

    def closeEvent(self, event):
        """Limpia recursos al cerrar la ventana."""
        if self.thread and self.thread.is_alive():
            self.thread.stop()
            self.thread.join(timeout=1.0)
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Establecer estilo
    app.setStyle('Fusion')
    
    window = SpikePrimeBLEApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

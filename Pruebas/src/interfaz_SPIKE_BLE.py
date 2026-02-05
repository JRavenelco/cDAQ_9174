import sys
import asyncio
import threading
import queue
from bleak import BleakClient, BleakScanner
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
        async with BleakClient(self.address) as client:
            self.running = True
            await client.start_notify(self.char_uuid, self._notification_handler)
            while self.running:
                await asyncio.sleep(0.1)
            await client.stop_notify(self.char_uuid)

    def _notification_handler(self, sender, data: bytearray):
        # Parsear bytes a valores numéricos (ajustar según formato del Hub)
        try:
            vals = np.frombuffer(data, dtype=np.int16).astype(float)
            self.data_queue.put(vals)
        except Exception as e:
            print(f"Notification parse error: {e}")

    def stop(self):
        self.running = False


class SpikePrimeBLEApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spike Prime BLE Interface")
        self.queue = queue.Queue()
        self.thread = None
        self.connected = False

        layout = QtWidgets.QGridLayout(self)
        # Controles BLE
        self.btn_scan = QtWidgets.QPushButton("Scan BLE")
        self.cb_devices = QtWidgets.QComboBox()
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.le_uuid = QtWidgets.QLineEdit()
        self.le_uuid.setPlaceholderText("Characteristic UUID")
        # Gráfica
        self.plot = pg.PlotWidget(title="Accelerometer Data")
        self.curve = self.plot.plot(pen='y')
        # Estado
        self.status = QtWidgets.QLabel("Idle")

        layout.addWidget(self.btn_scan, 0, 0)
        layout.addWidget(self.cb_devices, 0, 1)
        layout.addWidget(self.btn_connect, 0, 2)
        layout.addWidget(self.le_uuid, 0, 3)
        layout.addWidget(self.plot, 1, 0, 1, 4)
        layout.addWidget(self.status, 2, 0, 1, 4)

        self.btn_scan.clicked.connect(self.scan_devices)
        self.btn_connect.clicked.connect(self.toggle_connection)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

        # Buffer de datos de ejemplo
        self.data_buffer = np.zeros(512)

    def scan_devices(self):
        """Busca dispositivos BLE cercanos y llena el combo box."""
        self.cb_devices.clear()
        devices = asyncio.run(BleakScanner.discover())
        for d in devices:
            name = d.name or 'Unknown'
            self.cb_devices.addItem(f"{name} ({d.address})", d.address)

    def toggle_connection(self):
        if not self.connected:
            addr = self.cb_devices.currentData()
            uuid = self.le_uuid.text().strip()
            if not addr or not uuid:
                QtWidgets.QMessageBox.warning(self, "Error", "Select device and enter UUID")
                return
            # Iniciar hilo BLE
            self.thread = SpikeBLEThread(addr, uuid, self.queue)
            self.thread.start()
            self.connected = True
            self.btn_connect.setText("Disconnect")
            self.status.setText(f"Connected to {addr}")
        else:
            # Detener hilo
            if self.thread:
                self.thread.stop()
                self.thread.join(timeout=1.0)
            self.connected = False
            self.btn_connect.setText("Connect")
            self.status.setText("Disconnected")

    def update_plot(self):
        """Actualiza la gráfica con nuevos datos BLE."""
        try:
            while True:
                vals = self.queue.get_nowait()
                # Desplazar y actualizar buffer
                n = len(vals)
                self.data_buffer = np.roll(self.data_buffer, -n)
                self.data_buffer[-n:] = vals
            self.curve.setData(self.data_buffer)
        except queue.Empty:
            pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SpikePrimeBLEApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

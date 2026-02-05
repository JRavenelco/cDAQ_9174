import sys
import asyncio
import platform
from bleak import BleakScanner, BleakClient
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, 
                           QLabel, QComboBox, QWidget, QMessageBox, QHBoxLayout, QTextEdit)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject, pyqtSlot
import numpy as np
import pyqtgraph as pg
from struct import unpack
import struct
import logging

# UUIDs for SPIKE Prime App Protocol
SPIKE_APP_SERVICE_UUID = "0000FD02-0000-1000-8000-00805F9B34FB"
SPIKE_APP_RX_CHAR_UUID = "0000FD02-0001-1000-8000-00805F9B34FB"  # Hub's RX, so we WRITE to it
SPIKE_APP_TX_CHAR_UUID = "0000FD02-0002-1000-8000-00805F9B34FB"  # Hub's TX, so we get NOTIFICATIONS from it

# Configure basic logging to see INFO and DEBUG messages in the console
logging.basicConfig(
    level=logging.DEBUG, # Capture DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s',
    stream=sys.stdout  # Explicitly send to stdout
)
logging.debug("Root logger configured for DEBUG level output to stdout.")

class BLEDevice(QObject):
    """Clase para manejar la comunicación BLE con el SPIKE Prime."""
    data_received = pyqtSignal(int, int, int)
    status_update = pyqtSignal(str)
    connection_changed = pyqtSignal(bool)
    
    def __init__(self, loop):
        super().__init__()
        self.loop = loop
        self.client = None
        self.info_response_received_event = asyncio.Event()
        self.device_notification_response_event = asyncio.Event()
        self.hub_info = None
        self.device_notification_status = None
        self.is_handler_active = False
    
    async def connect_to_device(self, address):
        """Conecta al dispositivo BLE."""
        logging.info(f"Intentando conectar a {address}")
        try:
            self.client = BleakClient(address, loop=self.loop)
            await self.client.connect()
            logging.info(f"Conectado a {self.client.address}. Obteniendo servicios...")
            self.status_update.emit(f"Conectado a {self.client.address}. Obteniendo servicios...")
            
            # Verificar si el servicio SPIKE App Protocol está presente
            services = await self.client.get_services()
            logging.info(f"Servicios disponibles en {self.client.address}:")
            spike_service_found = False
            for service in services:
                logging.info(f"  Servicio UUID descubierto: {service.uuid.lower()}")
                if service.uuid.lower() == SPIKE_APP_SERVICE_UUID.lower():
                    spike_service_found = True
                    # Opcional: listar características del servicio encontrado
                    logging.info(f"    Características para el servicio {service.uuid}:")
                    for char in service.characteristics:
                        logging.info(f"      Característica: {char.uuid}, Propiedades: {char.properties}")
                        if char.uuid.lower() == SPIKE_APP_RX_CHAR_UUID.lower():
                            logging.info(f"        Encontrada RX Característica: {char.uuid}")
                        if char.uuid.lower() == SPIKE_APP_TX_CHAR_UUID.lower():
                            logging.info(f"        Encontrada TX Característica: {char.uuid}")
            
            if not spike_service_found:
                logging.error(f"Servicio SPIKE App Protocol ({SPIKE_APP_SERVICE_UUID}) no encontrado.")
                self.status_update.emit("Error: Servicio SPIKE App no encontrado.")
                await self.disconnect_device()
                return False

            logging.info(f"Servicio SPIKE App Protocol ({SPIKE_APP_SERVICE_UUID}) encontrado.")
            self.connection_changed.emit(True)
            return True
        except Exception as e:
            logging.error(f"Error inesperado al conectar: {e}")
            self.status_update.emit(f"Error inesperado al conectar: {e}")
        self.connection_changed.emit(False)
        return False
    
    async def disconnect_device(self):
        """Desconecta el dispositivo BLE."""
        if self.client and self.client.is_connected:
            logging.info("Desconectando del dispositivo...")
            if self.is_handler_active:
                try:
                    await self.client.stop_notify(SPIKE_APP_TX_CHAR_UUID)
                    logging.info("Notificaciones detenidas.")
                except Exception as e:
                    logging.warning(f"Error al detener notificaciones: {e}")
                self.is_handler_active = False
            await self.client.disconnect()
            logging.info("Desconectado.")
            self.status_update.emit("Desconectado exitosamente")
        else:
            logging.info("El cliente no está conectado o no existe.")
        self.connection_changed.emit(False)
    
    async def send_info_request_and_wait(self):
        """Envía un InfoRequest y espera la respuesta."""
        if not self.client or not self.client.is_connected:
            return False
        try:
            self.info_response_received_event.clear()
            self.hub_info = None
            info_request_payload = bytes([0x00]) # InfoRequest
            logging.info(f"Enviando InfoRequest: {info_request_payload.hex()}")
            await self.client.write_gatt_char(SPIKE_APP_RX_CHAR_UUID, info_request_payload, response=False) # response=False es típico aquí
            self.status_update.emit("InfoRequest enviado, esperando respuesta...")
            logging.info("InfoRequest enviado. Esperando InfoResponse (timeout 5s)...")
            await asyncio.wait_for(self.info_response_received_event.wait(), timeout=5.0)
            logging.info(f"InfoResponse recibido: {self.hub_info}")
            self.status_update.emit(f"InfoResponse recibido.")
            return True
        except asyncio.TimeoutError:
            logging.error("Timeout esperando InfoResponse.")
            self.status_update.emit("Error: Timeout esperando InfoResponse.")
            return False
        except Exception as e:
            logging.error(f"Error en send_info_request_and_wait: {e}")
            self.status_update.emit(f"Error enviando InfoRequest: {e}")
            return False
    
    async def send_device_notification_config(self):
        """Envía la configuración de notificación del dispositivo."""
        if not self.client or not self.client.is_connected:
            return False
        try:
            self.device_notification_response_event.clear()
            self.device_notification_status = None
            delta_interval_ms = 100
            port_id = 0x64
            mode = 0x00
            notification_enabled = 0x01
            delta_bytes = delta_interval_ms.to_bytes(4, byteorder='little')
            config_data = bytes([
                0x28, port_id, mode,
                delta_bytes[0], delta_bytes[1], delta_bytes[2], delta_bytes[3],
                notification_enabled
            ])
            logging.info(f"Enviando DeviceNotificationRequest (hex): {config_data.hex()} para Puerto {port_id}, Modo {mode}, Intervalo {delta_interval_ms}ms")
            await self.client.write_gatt_char(SPIKE_APP_RX_CHAR_UUID, config_data, response=False) # CAMBIADO a response=False
            self.status_update.emit("DeviceNotificationRequest enviado, esperando confirmación...")
            logging.info("DeviceNotificationRequest enviado (sin respuesta directa esperada de la escritura). Esperando DeviceNotificationResponse vía notificación (timeout 5s)...")
            await asyncio.wait_for(self.device_notification_response_event.wait(), timeout=5.0)
            if self.device_notification_status == 0x00: # Success
                logging.info("DeviceNotificationResponse OK recibido.")
                self.status_update.emit("Suscrito a notificaciones del acelerómetro.")
                return True
            else:
                logging.error(f"DeviceNotificationResponse indicó error o estado inesperado: {self.device_notification_status}")
                self.status_update.emit(f"Error en confirmación de suscripción: {self.device_notification_status}")
                return False
        except asyncio.TimeoutError:
            logging.error("Timeout esperando DeviceNotificationResponse.")
            self.status_update.emit("Error: Timeout confirmando suscripción.")
            return False
        except Exception as e:
            logging.error(f"Error en send_device_notification_config: {e}") # Esto capturará el 'Write Not Permitted'
            self.status_update.emit(f"Error configurando notificaciones: {e}")
            return False
    
    def _notification_handler(self, sender, data):
        """Maneja las notificaciones."""
        logging.debug(f"_notification_handler llamado por {sender} con datos (hex): {data.hex() if data else 'None'}")
        if not data:
            return

        msg_type = data[0]
        if msg_type == 0x01: # InfoResponse
            logging.info(f"InfoResponse (0x01) recibido: {data.hex()}")
            # Placeholder para parsear info real si es necesario
            self.hub_info = {
                'raw': data.hex(),
                'rpc_version': f"{data[1]}.{data[2]}.{data[3]}.{data[4]}" if len(data) > 4 else "N/A",
                'firmware_version': f"{data[5]}.{data[6]}.{data[7]}.{data[8]}" if len(data) > 8 else "N/A"
            }
            self.info_response_received_event.set()

        elif msg_type == 0x3C: # DeviceNotification
            if len(data) > 1:
                sub_msg_type = data[1]
                if sub_msg_type == 0x00: # DeviceNotificationResponse (ack for DeviceNotificationRequest)
                    status = data[2] if len(data) > 2 else None
                    logging.info(f"DeviceNotificationResponse (0x3C, 0x00) recibido. Status: {status if status is not None else 'N/A'}. Payload: {data.hex()}")
                    self.device_notification_status = status
                    self.device_notification_response_event.set()
                
                elif sub_msg_type == 0x01: # DeviceImuValues
                    logging.debug(f"DeviceImuValues (0x3C, 0x01) recibido: {data.hex()}")
                    # Formato: type(01), hub_face_up, hub_face_yaw, yaw, pitch, roll, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
                    #           1B      1B           1B            2B   2B     2B    2B     2B     2B    2B      2B      2B
                    # Total esperado: 1 + 1 + 1 + 2*6 = 15 bytes para DeviceImuValues payload (sin contar el 0x3C inicial)
                    if len(data) >= 17: # 0x3C + 0x01 + 15 bytes de payload IMU
                        # Extraer acc_x, acc_y, acc_z (son los bytes 8,9; 10,11; 12,13 del payload IMU, o 10,11; 12,13; 14,15 de 'data')
                        try:
                            acc_x = struct.unpack('<h', data[9:11])[0]  # Offset 9,10 from start of 'data' (0x3C, 0x01, face, yaw_face, yaw, pitch, roll, THEN acc_x)
                            acc_y = struct.unpack('<h', data[11:13])[0] # Offset 11,12
                            acc_z = struct.unpack('<h', data[13:15])[0] # Offset 13,14
                            # Los índices correctos según la documentación de LEGO (App Protocol v1.0.0.21, p.22)
                            # Data: [0x3C, 0x01, HubFaceUp, HubFaceYaw, Yaw(2B), Pitch(2B), Roll(2B), AccX(2B), AccY(2B), AccZ(2B), ...]
                            # Index:   0,    1,      2       ,     3     ,  4-5   ,   6-7   ,  8-9   ,  10-11  ,  12-13  ,  14-15
                            acc_x = struct.unpack('<h', data[10:12])[0]
                            acc_y = struct.unpack('<h', data[12:14])[0]
                            acc_z = struct.unpack('<h', data[14:16])[0]
                            logging.info(f"IMU Data: Acc(X:{acc_x}, Y:{acc_y}, Z:{acc_z})")
                            self.data_received.emit(acc_x, acc_y, acc_z)
                        except struct.error as e:
                            logging.error(f"Error desempaquetando datos IMU: {e}. Datos: {data.hex()}")
                    else:
                        logging.warning(f"DeviceImuValues payload demasiado corto: {len(data)} bytes. Esperados >= 17. Data: {data.hex()}")
            else:
                logging.warning(f"DeviceNotification (0x3C) payload demasiado corto para sub-tipo: {len(data)} bytes. Data: {data.hex()}")
        else:
            logging.debug(f"Notificación no manejada recibida. Tipo: {msg_type:02X}, Datos: {data.hex()}")
    
    async def connect_and_subscribe(self, address):
        """Conecta al dispositivo y se suscribe a las notificaciones."""
        self.status_update.emit(f"Iniciando conexión y suscripción a {address}...")
        if not await self.connect_to_device(address):
            self.status_update.emit("Fallo al conectar al dispositivo.")
            logging.error(f"Fallo en connect_to_device para {address}")
            return False

        try:
            logging.info("Iniciando notificaciones para la comunicación inicial...")
            await self.client.start_notify(SPIKE_APP_TX_CHAR_UUID, self._notification_handler)
            self.is_handler_active = True
            await asyncio.sleep(0.1) # Dar tiempo para que se establezca el handler

            if not await self.send_info_request_and_wait():
                self.status_update.emit("Fallo en el handshake de InfoRequest/InfoResponse.")
                logging.error("Fallo en send_info_request_and_wait.")
                await self.disconnect_device()
                return False
            
            logging.info("Handshake InfoRequest/InfoResponse exitoso. Procediendo a configurar notificaciones del acelerómetro.")
            
            if await self.send_device_notification_config():
                self.status_update.emit("Conectado y suscrito exitosamente al acelerómetro.")
                logging.info("Conectado y suscrito exitosamente al acelerómetro.")
                # No detenemos las notificaciones aquí, ya que las necesitamos para los datos del IMU.
                return True # Éxito final
            else:
                self.status_update.emit("Fallo al configurar las notificaciones del acelerómetro.")
                logging.error("Fallo en send_device_notification_config.")
                await self.disconnect_device()
                return False

        except Exception as e:
            logging.error(f"Error durante connect_and_subscribe: {e}")
            self.status_update.emit(f"Error en el proceso de conexión/suscripción: {e}")
            await self.disconnect_device()
            return False

class BLEScanner(QThread):
    """Hilo para escanear dispositivos BLE."""
    devices_found = pyqtSignal(list)
    
    def run(self):
        try:
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            devices = loop.run_until_complete(self._scan())
            self.devices_found.emit(devices)
            
        except Exception as e:
            print(f"Error en BLEScanner: {e}")
            self.devices_found.emit([])
    
    async def _scan(self):
        """Escanea dispositivos BLE."""
        print("Iniciando escaneo BLE...")
        devices = await BleakScanner.discover(timeout=10.0)
        print(f"Dispositivos encontrados: {len(devices)}")
        return devices

class SpikePrimeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ble_device = BLEDevice(asyncio.new_event_loop())
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
        
        # Gráfico para acelerómetro (X, Y, Z)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        
        # Configurar gráfico X
        self.plot_x = self.plot_widget.addPlot(title="Aceleración X")
        self.plot_x.setLabel('left', 'Aceleración (g)')
        self.plot_x.setLabel('bottom', 'Muestras')
        self.plot_x.showGrid(x=True, y=True, alpha=0.3)
        self.curve_x = self.plot_x.plot(pen=pg.mkPen(color='r', width=2))
        
        # Configurar gráfico Y
        self.plot_widget.nextRow()
        self.plot_y = self.plot_widget.addPlot(title="Aceleración Y")
        self.plot_y.setLabel('left', 'Aceleración (g)')
        self.plot_y.setLabel('bottom', 'Muestras')
        self.plot_y.showGrid(x=True, y=True, alpha=0.3)
        self.curve_y = self.plot_y.plot(pen=pg.mkPen(color='g', width=2))
        
        # Configurar gráfico Z
        self.plot_widget.nextRow()
        self.plot_z = self.plot_widget.addPlot(title="Aceleración Z")
        self.plot_z.setLabel('left', 'Aceleración (g)')
        self.plot_z.setLabel('bottom', 'Muestras')
        self.plot_z.showGrid(x=True, y=True, alpha=0.3)
        self.curve_z = self.plot_z.plot(pen=pg.mkPen(color='b', width=2))
        
        # Datos del gráfico
        self.data_x = np.zeros(100)
        self.data_y = np.zeros(100)
        self.data_z = np.zeros(100)
        
        # Consola de estado
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(100)
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
                font-family: monospace;
            }
        """)
        
        # Estado
        self.lbl_status = QLabel("Estado: Desconectado")
        self.lbl_status.setStyleSheet("font-size: 12px; color: #666666;")
        
        # Layout
        layout.addLayout(top_layout)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.console)
        layout.addWidget(self.lbl_status)
        
        # Timer para actualizar la interfaz
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)  # 20 FPS
        
        # Conectar señales BLE
        self.ble_device.data_received.connect(self.on_data_received)
        self.ble_device.status_update.connect(self.update_status)
        self.ble_device.connection_changed.connect(self.on_connection_changed)
        
        # Iniciar escaneo automático
        QTimer.singleShot(1000, self.start_scan)
    
    def log(self, message):
        """Agrega un mensaje a la consola."""
        self.console.append(f"> {message}")
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum()
        )
    
    def update_status(self, message):
        """Actualiza el estado y lo muestra en la consola."""
        self.lbl_status.setText(f"Estado: {message}")
        self.log(message)
    
    def start_scan(self):
        """Inicia el escaneo de dispositivos BLE."""
        if self.scan_thread and self.scan_thread.isRunning():
            return
            
        self.btn_scan.setEnabled(False)
        self.btn_scan.setText("Escaneando...")
        self.cb_devices.clear()
        self.update_status("Escaneando dispositivos BLE...")
        
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
            self.update_status("No se encontraron dispositivos BLE")
            return
            
        for device in devices:
            name = device.name if device.name else "Dispositivo desconocido"
            self.cb_devices.addItem(f"{name} ({device.address})", device.address)
        
        self.btn_connect.setEnabled(len(devices) > 0)
        self.update_status(f"Se encontraron {len(devices)} dispositivos")
    
    def toggle_connection(self):
        """Conecta/desconecta el dispositivo seleccionado."""
        if self.ble_device.client and self.ble_device.client.is_connected:
            self.disconnect_device()
        else:
            self.connect_device()
    
    async def _disconnect_async(self):
        """Método asíncrono para desconectar."""
        await self.ble_device.disconnect()
        
    def disconnect_device(self):
        """Desconecta el dispositivo BLE."""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._disconnect_async())
        except Exception as e:
            self.status_update.emit(f"Error en desconexión: {e}")
        finally:
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
    
    def connect_device(self):
        """Inicia la conexión al dispositivo BLE."""
        address = self.cb_devices.currentData()
        if not address:
            QMessageBox.warning(self, "Error", "Por favor selecciona un dispositivo primero")
            return
            
        self.update_status(f"Conectando a {address}...")
        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Conectando...")
        
        # Usar un hilo separado para la conexión
        self.connection_thread = ConnectionThread(self.ble_device, address)
        self.connection_thread.connection_result.connect(self.on_connection_result)
        self.connection_thread.start()
    
    def on_connection_result(self, success):
        """Maneja el resultado de la conexión."""
        if success:
            self.update_status(f"Conectado a {self.cb_devices.currentText()}")
            self.btn_connect.setText("🔌 Desconectar")
            self.btn_connect.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                }
            """)
            
            # Suscribirse a notificaciones después de conectar
            self.subscribe_thread = SubscribeThread(self.ble_device)
            self.subscribe_thread.subscription_result.connect(self.on_subscription_result)
            self.subscribe_thread.start()
            
        else:
            self.update_status("Error al conectar al dispositivo")
            self.btn_connect.setText("🚀 Conectar")
            self.btn_connect.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                }
            """)
        
        self.btn_connect.setEnabled(True)
    
    def on_subscription_result(self, success):
        """Maneja el resultado de la suscripción a notificaciones."""
        if success:
            self.update_status("Listo para recibir datos del acelerómetro")
        else:
            self.update_status("Error al suscribirse a las notificaciones")
    
    def on_connection_changed(self, connected):
        """Maneja el cambio de estado de la conexión."""
        if connected:
            self.update_status("Conectado")
        else:
            self.update_status("Desconectado")
    
    def on_data_received(self, x, y, z):
        """Procesa los datos recibidos del acelerómetro."""
        try:
            # Actualizar datos del gráfico
            self.data_x = np.roll(self.data_x, -1)
            self.data_y = np.roll(self.data_y, -1)
            self.data_z = np.roll(self.data_z, -1)
            
            self.data_x[-1] = x
            self.data_y[-1] = y
            self.data_z[-1] = z
            
        except Exception as e:
            self.log(f"Error procesando datos: {e}")
    
    def update_ui(self):
        """Actualiza la interfaz de usuario."""
        self.curve_x.setData(self.data_x)
        self.curve_y.setData(self.data_y)
        self.curve_z.setData(self.data_z)
    
    def closeEvent(self, event):
        """Limpia los recursos al cerrar la aplicación."""
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.quit()
            self.scan_thread.wait()
            
        if hasattr(self, 'connection_thread') and self.connection_thread.isRunning():
            self.connection_thread.quit()
            self.connection_thread.wait()
            
        if hasattr(self, 'subscribe_thread') and self.subscribe_thread.isRunning():
            self.subscribe_thread.quit()
            self.subscribe_thread.wait()
            
        # Asegurarse de que el dispositivo BLE se desconecte
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_device.disconnect())
            
        event.accept()

class ConnectionThread(QThread):
    """Hilo para manejar la conexión BLE."""
    connection_result = pyqtSignal(bool)  # success
    
    def __init__(self, ble_device, address):
        super().__init__()
        self.ble_device = ble_device
        self.address = address
    
    def run(self):
        try:
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            connected = loop.run_until_complete(self.ble_device.connect_and_subscribe(self.address))
            self.connection_result.emit(connected)
            
        except Exception as e:
            print(f"Error en ConnectionThread: {e}")
            self.connection_result.emit(False)

class SubscribeThread(QThread):
    """Hilo para manejar la suscripción a notificaciones BLE."""
    subscription_result = pyqtSignal(bool)  # success
    
    def __init__(self, ble_device):
        super().__init__()
        self.ble_device = ble_device
    
    def run(self):
        try:
            if platform.system() == "Windows":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            subscribed = loop.run_until_complete(self.ble_device.subscribe_to_notifications())
            self.subscription_result.emit(subscribed)
            
        except Exception as e:
            print(f"Error en SubscribeThread: {e}")
            self.subscription_result.emit(False)

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

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit
from PyQt6.QtCore import Qt
try:
    from pydwf import DwfLibrary, DeviceEnumeration, PyDwfError
    from pydwf.utilities import openDwfDevice 
except ImportError as e:
    print(f"Error al importar componentes de pydwf: {e}")
    print("Asegúrate de que 'pydwf' esté instalado y accesible.")
    exit()

class DigilentSPIInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digilent WaveForms SPI Interface")
        self.setGeometry(100, 100, 400, 300)

        # Initialize Digilent Device variables
        self.dwf_library = None
        self.device_enumerator = None
        self.device = None
        self.device_name = "Desconocido"
        self.serial_number = "Desconocido"

        # Setup UI
        self.setup_ui()

        # Connect to device on startup
        self.connect_to_device()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Device Status Label
        self.status_label = QLabel("Estado del Dispositivo: Desconectado")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Device Info Label
        self.info_label = QLabel("Dispositivo: Desconocido (S/N: Desconocido)")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        # Connect/Disconnect Buttons
        self.connect_button = QPushButton("Conectar Dispositivo")
        self.connect_button.clicked.connect(self.connect_to_device)
        layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton("Desconectar Dispositivo")
        self.disconnect_button.clicked.connect(self.disconnect_device)
        self.disconnect_button.setEnabled(False)
        layout.addWidget(self.disconnect_button)

        # SPI Capture Button
        self.capture_button = QPushButton("Capturar Datos SPI")
        self.capture_button.clicked.connect(self.capture_spi_data)
        self.capture_button.setEnabled(False)
        layout.addWidget(self.capture_button)

        # Log Area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def log(self, message):
        self.log_text.append(message)

    def connect_to_device(self):
        self.log("Buscando dispositivos Digilent WaveForms...")
        try:
            if self.dwf_library is None:
                self.dwf_library = DwfLibrary()
            if self.device_enumerator is None:
                self.device_enumerator = DeviceEnumeration(self.dwf_library)

            num_devices = 0
            if hasattr(self.device_enumerator, 'enumerateDevices'):
                num_devices = self.device_enumerator.enumerateDevices()
                self.log(f"Número de dispositivos encontrados: {num_devices}")
            else:
                self.log("El método 'enumerateDevices' no fue encontrado en device_enumerator.")
                return

            if isinstance(num_devices, int) and num_devices > 0:
                self.log(f"Se encontraron {num_devices} dispositivos.")

                # Get device info
                try:
                    if hasattr(self.device_enumerator, 'deviceName') and hasattr(self.device_enumerator, 'serialNumber'):
                        self.log("Obteniendo información del primer dispositivo (índice 0)...")
                        self.device_name = self.device_enumerator.deviceName(0)
                        self.serial_number = self.device_enumerator.serialNumber(0)
                        self.log(f"  Nombre del dispositivo: {self.device_name}")
                        self.log(f"  Número de serie: {self.serial_number}")
                    else:
                        self.log("Los métodos deviceName o serialNumber no están en device_enumerator.")
                except PyDwfError as e_info:
                    self.log(f"Error al obtener información del dispositivo desde el enumerador: {e_info}")
                except Exception as e_generic_info:
                    self.log(f"Error genérico al obtener información del dispositivo: {e_generic_info}")

                # Open the device if not already opened
                if self.device is None:
                    self.log("Abriendo el primer dispositivo lógicamente disponible...")
                    self.device = openDwfDevice(self.dwf_library)
                    self.log("Dispositivo abierto exitosamente.")
                    self.status_label.setText("Estado del Dispositivo: Conectado")
                    self.info_label.setText(f"Dispositivo: {self.device_name} (S/N: {self.serial_number})")
                    self.connect_button.setEnabled(False)
                    self.disconnect_button.setEnabled(True)
                    self.capture_button.setEnabled(True)
                else:
                    self.log("El dispositivo ya está abierto.")
            else:
                self.log("No se encontraron dispositivos Digilent WaveForms.")
                self.status_label.setText("Estado del Dispositivo: Desconectado")
                self.info_label.setText("Dispositivo: Desconocido (S/N: Desconocido)")
                self.connect_button.setEnabled(True)
                self.disconnect_button.setEnabled(False)
                self.capture_button.setEnabled(False)
        except PyDwfError as e:
            self.log(f"Ocurrió un error de PyDwf: {e}")
            if "DWF library not found" in str(e) or "Error loading DWF library" in str(e):
                self.log("--- Posible Solución ---")
                self.log("Asegúrate de que el software WaveForms de Digilent esté instalado correctamente.")
                self.log("Este software instala las librerías y controladores necesarios.")
                self.log("Puedes descargarlo desde: https://digilent.com/reference/software/waveforms/waveforms-3/start")
            self.status_label.setText("Estado del Dispositivo: Error")
        except Exception as e:
            self.log(f"Ocurrió un error inesperado: {e}")
            self.status_label.setText("Estado del Dispositivo: Error")

    def disconnect_device(self):
        if self.device is not None:
            self.log("Cerrando el dispositivo...")
            self.device.close()
            self.device = None
            self.log("Dispositivo cerrado.")
            self.status_label.setText("Estado del Dispositivo: Desconectado")
            self.info_label.setText("Dispositivo: Desconocido (S/N: Desconocido)")
            self.connect_button.setEnabled(True)
            self.disconnect_button.setEnabled(False)
            self.capture_button.setEnabled(False)
        else:
            self.log("No hay dispositivo abierto para cerrar.")

    def capture_spi_data(self):
        if self.device is not None:
            self.log("Iniciando captura de datos SPI...")
            # Aquí se implementará la lógica para captura SPI
            self.log("Captura de datos SPI no implementada aún.")
        else:
            self.log("No hay dispositivo conectado para capturar datos.")

    def closeEvent(self, event):
        if self.device is not None:
            self.device.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DigilentSPIInterface()
    window.show()
    sys.exit(app.exec())

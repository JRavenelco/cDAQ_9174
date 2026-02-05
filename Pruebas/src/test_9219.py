import sys
import time
import nidaqmx
from nidaqmx.constants import AcquisitionType, ExcitationSource, BridgeUnits
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# -------------------------------------------------
# Hilo de adquisición para 4 canales NI 9219
# -------------------------------------------------
class AcquisitionThread(QtCore.QThread):
    """
    Hilo que configura la tarea NI 9219 con una tasa de muestreo y un número de muestras
    por lectura. Emite una lista de 4 valores (V/V) promedio en la señal data_signal.
    """
    data_signal = QtCore.pyqtSignal(list)

    def __init__(self, rate=100.0, samples_per_channel=1, parent=None):
        super().__init__(parent)
        self.running = True
        self.rate = rate
        self.samples_per_channel = samples_per_channel
        self.task = None

        try:
            self.task = nidaqmx.Task()
            channels = ["cDAQ1Mod3/ai0", "cDAQ1Mod3/ai1", "cDAQ1Mod3/ai2", "cDAQ1Mod3/ai3"]
            for ch in channels:
                base_name = ch.replace("/", "_")
                self.task.ai_channels.add_ai_bridge_chan(
                    physical_channel=ch,
                    name_to_assign_to_channel=f"LoadCell_{base_name}",
                    min_val=-0.01,  # ±0.01 V/V
                    max_val=0.01,
                    units=BridgeUnits.VOLTS_PER_VOLT,
                    voltage_excit_val=2.5,
                    voltage_excit_source=ExcitationSource.INTERNAL,
                    nominal_bridge_resistance=350.0
                )
            # Configurar la tasa de muestreo y modo continuo
            self.task.timing.cfg_samp_clk_timing(
                rate=self.rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=1000  # Búfer interno
            )
            # Aumentar el búfer interno para evitar -200279
            self.task.in_stream.input_buf_size = 5000

            self.task.start()
        except Exception as e:
            print("Error al configurar la tarea NI 9219:", e)
            self.running = False

    def run(self):
        """
        Lee 'samples_per_channel' muestras cada ~10 ms,
        promedia por canal y emite data_signal.
        """
        while self.running:
            try:
                data = self.task.read(
                    number_of_samples_per_channel=self.samples_per_channel,
                    timeout=1.0
                )
                # data => [[ch0_samples], [ch1_samples], ...]
                avg_values = []
                for channel_data in data:
                    avg = sum(channel_data)/len(channel_data)
                    avg_values.append(avg)
                self.data_signal.emit(avg_values)
            except Exception as e:
                print("Error en adquisición:", e)
            self.msleep(10)  # 10 ms de retardo = 100 ciclos por segundo

    def stop(self):
        self.running = False
        try:
            if self.task:
                self.task.stop()
                self.task.close()
        except nidaqmx.DaqError:
            pass
        self.wait()

# -------------------------------------------------
# Widget para un sensor: muestra microvolts
# con offset y ganancia configurables
# -------------------------------------------------
class SensorWidget(QtWidgets.QWidget):
    def __init__(self, sensor_index, parent=None):
        super().__init__(parent)
        self.sensor_index = sensor_index

        # Por defecto: factor para convertir V/V a µV => 2.5 * 1e6
        self.base_factor = 2.5e6
        self.offset_uv = 0.0   # offset en µV
        self.gain = 1.0        # factor de amplificación adimensional

        self.buffer_size = 100
        self.buffer = []

        layout = QtWidgets.QVBoxLayout(self)

        self.title_label = QtWidgets.QLabel(f"Sensor {sensor_index+1} - Microvolts")
        layout.addWidget(self.title_label)

        self.value_label = QtWidgets.QLabel("Lectura: --- µV")
        layout.addWidget(self.value_label)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-100, 1000)
        self.plot_curve = self.plot_widget.plot(pen='y')
        layout.addWidget(self.plot_widget)

        # Sección de offset y ganancia
        form_layout = QtWidgets.QFormLayout()
        self.offset_line = QtWidgets.QLineEdit("0.0")
        self.gain_line = QtWidgets.QLineEdit("1.0")
        form_layout.addRow("Offset (µV):", self.offset_line)
        form_layout.addRow("Ganancia:", self.gain_line)
        layout.addLayout(form_layout)

        # Botón "Aplicar" para leer offset y ganancia
        self.apply_btn = QtWidgets.QPushButton("Aplicar Offset/Ganancia")
        self.apply_btn.clicked.connect(self.apply_offset_gain)
        layout.addWidget(self.apply_btn)

    def update_data(self, raw_vv):
        """
        raw_vv: Lectura en V/V
        1) Convertir a µV => raw_vv * base_factor
        2) Restar offset (µV)
        3) Multiplicar por ganancia
        """
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(raw_vv)

        uv = raw_vv * self.base_factor
        corrected_uv = (uv - self.offset_uv) * self.gain

        self.value_label.setText(f"Lectura: {corrected_uv:.2f} µV")

        # Graficar todos los datos del buffer con offset/gain
        plot_data = []
        for val in self.buffer:
            uv_i = val * self.base_factor
            corrected_i = (uv_i - self.offset_uv) * self.gain
            plot_data.append(corrected_i)

        self.plot_curve.setData(plot_data)

    def apply_offset_gain(self):
        """
        Lee los valores de offset y ganancia desde los QLineEdit,
        actualiza self.offset_uv y self.gain
        """
        try:
            offset_text = self.offset_line.text()
            gain_text = self.gain_line.text()
            self.offset_uv = float(offset_text)
            self.gain = float(gain_text)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          "Valores de offset/gain inválidos.")

# -------------------------------------------------
# Ventana principal
# -------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lectura en Microvolts - Ajuste Rate y Samples (NI 9219)")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # Widgets para configurar Rate y Samples
        config_layout = QtWidgets.QHBoxLayout()
        # Actualizamos los valores por defecto: 100 Hz y 1 muestra por lectura
        self.rate_line = QtWidgets.QLineEdit("100.0")
        self.samples_line = QtWidgets.QLineEdit("1")
        config_layout.addWidget(QtWidgets.QLabel("Rate (Hz):"))
        config_layout.addWidget(self.rate_line)
        config_layout.addWidget(QtWidgets.QLabel("Samples/Read:"))
        config_layout.addWidget(self.samples_line)
        self.apply_config_btn = QtWidgets.QPushButton("Aplicar Configuración")
        self.apply_config_btn.clicked.connect(self.apply_acquisition_config)
        config_layout.addWidget(self.apply_config_btn)

        layout.addLayout(config_layout, 0, 0, 1, 2)

        # Crear 4 widgets de sensor
        self.sensor_widgets = []
        for i in range(4):
            sw = SensorWidget(i)
            self.sensor_widgets.append(sw)
            row = (i // 2) + 1
            col = i % 2
            layout.addWidget(sw, row, col)

        # Iniciar hilo de adquisición con los nuevos valores por defecto
        self.acq_thread = None
        self.create_acquisition_thread(rate=100.0, samples=1)

        # Botón para detener
        self.stop_btn = QtWidgets.QPushButton("Detener Adquisición")
        self.stop_btn.clicked.connect(self.stop_acquisition)
        layout.addWidget(self.stop_btn, 3, 0, 1, 2)

    def create_acquisition_thread(self, rate, samples):
        """Crea y lanza el hilo de adquisición con rate y samples dados."""
        if self.acq_thread:
            self.acq_thread.stop()
        self.acq_thread = AcquisitionThread(rate=rate, samples_per_channel=samples)
        self.acq_thread.data_signal.connect(self.update_sensors)
        self.acq_thread.start()

    def apply_acquisition_config(self):
        """Lee Rate y Samples del GUI, recrea el hilo de adquisición."""
        try:
            new_rate = float(self.rate_line.text())
            new_samples = int(self.samples_line.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Error",
                                          "Valores de Rate/Samples inválidos.")
            return
        self.create_acquisition_thread(new_rate, new_samples)

    def update_sensors(self, raw_values):
        for i, raw_val in enumerate(raw_values):
            self.sensor_widgets[i].update_data(raw_val)

    def stop_acquisition(self):
        if self.acq_thread:
            self.acq_thread.stop()
        QtWidgets.QMessageBox.information(self, "Adquisición", "Adquisición detenida.")

    def closeEvent(self, event):
        if self.acq_thread:
            self.acq_thread.stop()
        event.accept()

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

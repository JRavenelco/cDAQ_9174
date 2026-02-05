import nidaqmx
from nidaqmx.system import System
import numpy as np
import time
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# --------------------------------------------------------------------------------
# CONFIGURACIÓN PRINCIPAL
# --------------------------------------------------------------------------------
DISPOSITIVO = "cDAQ1Mod1"       # Ajustar si tu módulo se llama distinto en NI MAX
CANALES = ["ai0", "ai1"]        # Dos canales para capturar dos señales
SAMPLE_RATE = 10000             # 10 kHz de muestreo por defecto
MUESTRAS_POR_BLOQUE = 1000      # Cantidad de muestras por lectura
TIME_WINDOW = 0.05              # 50 ms de ventana de tiempo en la gráfica
VOLTAJE_MIN = -5.0              # Rango mínimo
VOLTAJE_MAX = 5.0               # Rango máximo

# Mapeo para el ComboBox de modo → objeto TerminalConfiguration
TERMINAL_CONFIG_MAP = {
    "Diferencial": TerminalConfiguration.DIFF,
    "RSE": TerminalConfiguration.RSE,
    "NRSE": TerminalConfiguration.NRSE
}

# Cola para pasar datos del hilo de adquisición a la GUI
datos_queue = queue.Queue(maxsize=10)

# --------------------------------------------------------------------------------
# HILO DE ADQUISICIÓN
# --------------------------------------------------------------------------------
class AdquisicionThread(threading.Thread):
    """
    Hilo que ejecuta la adquisición continua con NI-DAQmx.
    Lee datos en bloques de MUESTRAS_POR_BLOQUE y los envía a datos_queue.
    """
    def __init__(self, dispositivo, canales, sample_rate, muestras_por_bloque, terminal_config):
        super().__init__()
        self.dispositivo = dispositivo
        self.canales = canales
        self.sample_rate = sample_rate
        self.muestras_por_bloque = muestras_por_bloque
        self.terminal_config = terminal_config
        self.running = True
        self.task = None
        
    def run(self):
        try:
            # Crear la tarea
            self.task = nidaqmx.Task()
            
            # Añadir canales con el modo de entrada seleccionado
            for canal in self.canales:
                nombre_canal = f"{self.dispositivo}/{canal}"
                print(f"Configurando canal: {nombre_canal} en modo {self.terminal_config}")
                self.task.ai_channels.add_ai_voltage_chan(
                    nombre_canal,
                    terminal_config=self.terminal_config,
                    min_val=VOLTAJE_MIN,
                    max_val=VOLTAJE_MAX
                )
            
            # Configurar adquisición continua
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10
            )
            
            print(f"Iniciando adquisición a {self.sample_rate} Hz en {self.canales}")
            self.task.start()
            
            # Bucle principal de adquisición
            while self.running:
                try:
                    datos = self.task.read(
                        number_of_samples_per_channel=self.muestras_por_bloque,
                        timeout=1.0
                    )
                    datos_np = np.array(datos)
                    
                    # Pasar datos a la cola
                    try:
                        datos_queue.put(datos_np, block=False)
                    except queue.Full:
                        # Si la cola está llena, sacamos el más antiguo
                        datos_queue.get_nowait()
                        datos_queue.put(datos_np, block=False)
                
                except nidaqmx.errors.DaqError as e:
                    print(f"Error en adquisición: {e}")
                    if "timeout" not in str(e).lower():
                        break
            
            # Al salir, detenemos y cerramos la tarea
            if self.task:
                self.task.stop()
                self.task.close()
                print("Tarea de adquisición detenida y cerrada.")
                
        except Exception as e:
            print(f"Error en el hilo de adquisición: {e}")
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass
    
    def stop(self):
        self.running = False

# --------------------------------------------------------------------------------
# CLASE PRINCIPAL DE LA APLICACIÓN (OSCILOSCOPIO)
# --------------------------------------------------------------------------------
class OsciloscopioApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Osciloscopio - Dos Señales con Selección de Modo')
        self.resize(1200, 800)
        
        # Estado de adquisición
        self.acquiring = True
        
        # Tasa de muestreo y ventana de tiempo (por defecto)
        self.sample_rate = SAMPLE_RATE
        self.time_window = TIME_WINDOW
        
        # Modo de entrada (por defecto, Diferencial)
        self.terminal_config = TerminalConfiguration.DIFF
        
        # Calcular tamaño del buffer
        self.buffer_size = int(self.time_window * self.sample_rate)
        if self.buffer_size < 2:
            self.buffer_size = 2
        
        # Crear buffers para cada canal
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
        
        # Interfaz
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(layout)
        
        # Sección de info de la señal
        info_layout = QtWidgets.QHBoxLayout()
        self.value_labels = []
        self.freq_labels = []
        self.amp_labels = []
        self.rms_labels = []
        
        for canal in CANALES:
            canal_layout = QtWidgets.QVBoxLayout()
            
            canal_label = QtWidgets.QLabel(f"Canal {canal}")
            canal_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
            canal_layout.addWidget(canal_label)
            
            value_label = QtWidgets.QLabel("Voltaje: 0.000 V")
            value_label.setStyleSheet("font-size: 12pt;")
            canal_layout.addWidget(value_label)
            self.value_labels.append(value_label)
            
            freq_label = QtWidgets.QLabel("Frecuencia: 0.0 Hz")
            freq_label.setStyleSheet("font-size: 12pt;")
            canal_layout.addWidget(freq_label)
            self.freq_labels.append(freq_label)
            
            amp_label = QtWidgets.QLabel("Amplitud: 0.000 Vpp")
            amp_label.setStyleSheet("font-size: 12pt;")
            canal_layout.addWidget(amp_label)
            self.amp_labels.append(amp_label)
            
            rms_label = QtWidgets.QLabel("RMS: 0.000 V")
            rms_label.setStyleSheet("font-size: 12pt;")
            canal_layout.addWidget(rms_label)
            self.rms_labels.append(rms_label)
            
            info_layout.addLayout(canal_layout)
        
        layout.addLayout(info_layout)
        
        # Gráficas
        self.plot_widgets = []
        self.plot_curves = []
        
        for i, canal in enumerate(CANALES):
            plot_widget = pg.PlotWidget()
            plot_widget.setLabel('left', 'Voltaje', units='V')
            if i == len(CANALES) - 1:
                plot_widget.setLabel('bottom', 'Tiempo', units='s')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setYRange(VOLTAJE_MIN, VOLTAJE_MAX)
            
            pen = pg.mkPen(color=('r', 'b')[i % 2], width=2)
            plot_curve = plot_widget.plot(pen=pen)
            
            layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            self.plot_curves.append(plot_curve)
        
        # Botones de control
        control_layout = QtWidgets.QHBoxLayout()
        
        self.start_stop_button = QtWidgets.QPushButton("Detener")
        self.start_stop_button.clicked.connect(self.toggle_acquisition)
        control_layout.addWidget(self.start_stop_button)
        
        # ComboBox para la base de tiempo
        control_layout.addWidget(QtWidgets.QLabel("Tiempo:"))
        self.time_combo = QtWidgets.QComboBox()
        self.time_combo.addItems(["10 ms", "20 ms", "50 ms", "100 ms", "200 ms", "500 ms", "1 s"])
        self.time_combo.setCurrentText("50 ms")
        self.time_combo.currentTextChanged.connect(self.update_timebase)
        control_layout.addWidget(self.time_combo)
        
        # ComboBox para el modo de entrada
        control_layout.addWidget(QtWidgets.QLabel("Modo:"))
        self.terminal_mode_combo = QtWidgets.QComboBox()
        self.terminal_mode_combo.addItems(["Diferencial", "RSE", "NRSE"])
        self.terminal_mode_combo.setCurrentText("Diferencial")  # por defecto
        control_layout.addWidget(self.terminal_mode_combo)
        
        # Botón para aplicar los cambios de modo
        self.apply_mode_button = QtWidgets.QPushButton("Aplicar Modo")
        self.apply_mode_button.clicked.connect(self.apply_mode_changes)
        control_layout.addWidget(self.apply_mode_button)
        
        # Botón para guardar datos
        self.save_button = QtWidgets.QPushButton("Guardar Datos")
        self.save_button.clicked.connect(self.save_data)
        control_layout.addWidget(self.save_button)
        
        layout.addLayout(control_layout)
        
        # Hilo de adquisición (arranca de inmediato)
        self.adquisicion_thread = AdquisicionThread(
            DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config
        )
        self.adquisicion_thread.start()
        
        # Timer para refrescar la gráfica
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(20)  # ~50 FPS
        
        # Almacenar todos los datos (por si deseas guardarlos)
        self.all_data = [[] for _ in range(len(CANALES))]
    
    # --------------------------------------------------------------------------------
    # Funciones de interfaz
    # --------------------------------------------------------------------------------
    def toggle_acquisition(self):
        if self.acquiring:
            # Detener
            self.acquiring = False
            self.start_stop_button.setText("Iniciar")
            self.adquisicion_thread.stop()
        else:
            # Iniciar
            self.acquiring = True
            self.start_stop_button.setText("Detener")
            
            # Vaciar buffers
            self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
            self.all_data = [[] for _ in range(len(CANALES))]
            
            # Crear nuevo hilo
            self.adquisicion_thread = AdquisicionThread(
                DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config
            )
            self.adquisicion_thread.start()
    
    def update_timebase(self):
        """
        Cambia la ventana de tiempo de la gráfica (base de tiempo).
        """
        time_text = self.time_combo.currentText()
        if "ms" in time_text:
            time_val = float(time_text.split()[0]) / 1000.0
        else:
            time_val = float(time_text.split()[0])
        
        self.time_window = time_val
        self.buffer_size = int(self.time_window * self.sample_rate)
        if self.buffer_size < 2:
            self.buffer_size = 2
        
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        
        for widget in self.plot_widgets:
            widget.setXRange(-self.time_window, 0)
    
    def apply_mode_changes(self):
        """
        Aplica el modo de entrada seleccionado (Diferencial, RSE, NRSE).
        Detiene la adquisición si está activa, crea una nueva tarea y la reinicia.
        """
        # Detener si está en adquisición
        if self.acquiring:
            self.adquisicion_thread.stop()
            time.sleep(0.5)
        
        # Leer el texto del ComboBox
        modo_text = self.terminal_mode_combo.currentText()  # "Diferencial", "RSE", "NRSE"
        self.terminal_config = TERMINAL_CONFIG_MAP[modo_text]
        
        print(f"Aplicando modo de entrada: {modo_text}")
        
        # Crear nueva tarea con el modo seleccionado
        self.adquisicion_thread = AdquisicionThread(
            DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config
        )
        
        # Si estaba adquiriendo, la reiniciamos
        if self.acquiring:
            self.adquisicion_thread.start()
            self.start_stop_button.setText("Detener")
    
    def update_plots(self):
        """
        Llamada periódicamente por el QTimer. Lee la cola, actualiza los buffers
        y grafica en tiempo real.
        """
        try:
            while not datos_queue.empty():
                nuevos_datos = datos_queue.get_nowait()
                
                # nuevos_datos: array con forma [Ncanales, Nsamples]
                for i, canal_datos in enumerate(nuevos_datos):
                    self.all_data[i].extend(canal_datos)
                    
                    num_nuevas = len(canal_datos)
                    if num_nuevas >= self.buffer_size:
                        self.datos_buffer[i][:] = canal_datos[-self.buffer_size:]
                    else:
                        self.datos_buffer[i][:-num_nuevas] = self.datos_buffer[i][num_nuevas:]
                        self.datos_buffer[i][-num_nuevas:] = canal_datos
                
                # Calcular métricas
                for i, canal in enumerate(CANALES):
                    freq, amp, rms = self.calcular_metricas_senal(
                        self.datos_buffer[i], self.sample_rate
                    )
                    valor_actual = self.datos_buffer[i][-1]
                    
                    self.value_labels[i].setText(f"Voltaje: {valor_actual:.3f} V")
                    self.freq_labels[i].setText(f"Frecuencia: {freq:.1f} Hz")
                    self.amp_labels[i].setText(f"Amplitud: {amp:.3f} Vpp")
                    self.rms_labels[i].setText(f"RMS: {rms:.3f} V")
        
        except Exception as e:
            print(f"Error al actualizar la gráfica: {e}")
        
        # Dibujar
        for i, curve in enumerate(self.plot_curves):
            curve.setData(self.tiempo, self.datos_buffer[i])
    
    def calcular_metricas_senal(self, datos, sample_rate):
        """
        Calcula frecuencia, amplitud pico a pico y RMS a partir de 'datos'.
        """
        rms = np.sqrt(np.mean(np.square(datos)))
        amplitud = np.max(datos) - np.min(datos)
        
        zero_mean_data = datos - np.mean(datos)
        cruces = np.where(np.diff(np.signbit(zero_mean_data)))[0]
        
        if len(cruces) >= 4:
            periodos = []
            for i in range(0, len(cruces) - 2, 2):
                periodos.append(cruces[i+2] - cruces[i])
            if periodos:
                periodo_promedio = np.mean(periodos)
                frecuencia = sample_rate / periodo_promedio if periodo_promedio > 0 else 0
            else:
                frecuencia = 0
        else:
            frecuencia = 0
        
        return frecuencia, amplitud, rms
    
    def save_data(self):
        """
        Guarda los datos acumulados en un CSV.
        Evita el '#' en la primera línea con 'comments=""' para que
        las columnas se llamen exactamente "Canal_ai0", "Canal_ai1", etc.
        """
        if not self.all_data[0]:
            QtWidgets.QMessageBox.warning(self, "Sin datos", 
                                          "No hay datos para guardar. Inicia la adquisición primero.")
            return
        
        filename = f"datos_osciloscopio_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        datos_np = np.array([np.array(canal_data) for canal_data in self.all_data])
        
        np.savetxt(
            filename,
            np.transpose(datos_np),
            delimiter=",",
            header=",".join([f"Canal_{canal}" for canal in CANALES]),
            comments=""  # <--- Evita que NumPy añada '#' en la primera línea
        )
        
        # Limpiar buffer de guardado
        self.all_data = [[] for _ in range(len(CANALES))]
        
        QtWidgets.QMessageBox.information(
            self, "Datos Guardados",
            f"Los datos se han guardado correctamente en '{filename}'."
        )
    
    def closeEvent(self, event):
        # Detener el hilo al cerrar
        if hasattr(self, 'adquisicion_thread') and self.adquisicion_thread.is_alive():
            self.adquisicion_thread.stop()
            self.adquisicion_thread.join(timeout=1.0)
        event.accept()

# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------
def mostrar_info_hardware():
    system = System.local()
    print("\n=== INFORMACIÓN DEL HARDWARE NI-DAQ ===")
    for device in system.devices:
        print(f"\nDispositivo: {device.name}")
        print(f"  Producto: {device.product_type}")
        print(f"  Número de serie: {device.serial_num}")
        try:
            print("\n  Canales analógicos de entrada:")
            for i, channel in enumerate(device.ai_physical_chans):
                print(f"    {i}: {channel.name}")
            print("\n  Capacidades de timing:")
            print(f"    AI_Max_Multi_Chan_Rate: {device.ai_max_multi_chan_rate}")
            print(f"    AI_Max_Single_Chan_Rate: {device.ai_max_single_chan_rate}")
        except Exception as e:
            print(f"  Error al obtener info: {e}")
    print("\n========================================")

def main():
    mostrar_info_hardware()
    print(f"\nIniciando osciloscopio con selección de modo de entrada:")
    print(f"  Dispositivo: {DISPOSITIVO}")
    print(f"  Canales: {CANALES}")
    print(f"  Muestreo: {SAMPLE_RATE} Hz")
    print(f"  Ventana de tiempo inicial: {TIME_WINDOW*1000} ms (base de tiempo)")
    
    app = QtWidgets.QApplication([])
    ventana = OsciloscopioApp()
    ventana.show()
    app.exec_()

if __name__ == "__main__":
    main()

import nidaqmx
from nidaqmx.system import System
import numpy as np
import time
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
# --- IMPORTANTE: para 3D con PyQtGraph ---
import pyqtgraph.opengl as gl
# ------------------------------------------
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Opcional, solo si sigues usando Matplotlib
import matplotlib.animation as animation
from scipy.signal import savgol_filter

# --------------------------------------------------------------------------------
# CONFIGURACIÓN PRINCIPAL
# --------------------------------------------------------------------------------
DISPOSITIVO = "cDAQ1Mod1"       # Ajustar según NI MAX
CANALES = ["ai0", "ai1"]        # Dos canales para capturar señales
SAMPLE_RATE = 10000             # 10 kHz por defecto
MUESTRAS_POR_BLOQUE = 1000      # Muestras por lectura
TIME_WINDOW = 0.05              # 50 ms de ventana en la gráfica
VOLTAJE_MIN = -5.0              # Rango mínimo
VOLTAJE_MAX = 5.0               # Rango máximo

# Mapeo para el ComboBox de modo → TerminalConfiguration
TERMINAL_CONFIG_MAP = {
    "Diferencial": TerminalConfiguration.DIFF,
    "RSE": TerminalConfiguration.RSE,
    "NRSE": TerminalConfiguration.NRSE
}

# Cola para pasar datos del hilo de adquisición a la GUI
datos_queue = queue.Queue(maxsize=10)

# --------------------------------------------------------------------------------
# Función de normalización
# --------------------------------------------------------------------------------
def normalize_signal(signal):
    """Normaliza la señal para que su valor máximo absoluto sea 1.
       Si la señal es constante (todo 0), se devuelve tal cual."""
    max_abs = np.max(np.abs(signal))
    if max_abs == 0:
        return signal
    return signal / max_abs

# --------------------------------------------------------------------------------
# HILO DE ADQUISICIÓN
# --------------------------------------------------------------------------------
class AdquisicionThread(threading.Thread):
    """
    Hilo que ejecuta la adquisición continua con NI-DAQmx.
    Lee datos en bloques y los envía a datos_queue.
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
            self.task = nidaqmx.Task()
            # Configurar cada canal con el modo seleccionado
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
            
            while self.running:
                try:
                    datos = self.task.read(
                        number_of_samples_per_channel=self.muestras_por_bloque,
                        timeout=1.0
                    )
                    datos_np = np.array(datos)
                    try:
                        datos_queue.put(datos_np, block=False)
                    except queue.Full:
                        # Si la cola está llena, descartar el dato más antiguo
                        datos_queue.get_nowait()
                        datos_queue.put(datos_np, block=False)
                except nidaqmx.errors.DaqError as e:
                    print(f"Error en adquisición: {e}")
                    if "timeout" not in str(e).lower():
                        break
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
        self.setWindowTitle('Osciloscopio - 2D y 3D en Tiempo Real')
        self.resize(1500, 800)
        
        # Estado de adquisición y parámetros
        self.acquiring = True
        self.sample_rate = SAMPLE_RATE
        self.time_window = TIME_WINDOW
        self.terminal_config = TerminalConfiguration.DIFF
        self.buffer_size = max(int(self.time_window * self.sample_rate), 2)
        
        # Buffers para datos y tiempo
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
        
        # Configuración de la interfaz
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(main_layout)
        
        # Sección de información de la señal
        info_layout = QtWidgets.QHBoxLayout()
        self.value_labels, self.freq_labels = [], []
        self.amp_labels, self.rms_labels = [], []
        
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
        main_layout.addLayout(info_layout)
        
        # Layout para colocar las gráficas 2D y 3D
        plots_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(plots_layout, stretch=1)
        
        # ============= Gráficas en 2D (PyQtGraph) =============
        self.plot_widgets = []
        self.plot_curves = []
        
        signals_layout = QtWidgets.QVBoxLayout()
        for i, canal in enumerate(CANALES):
            plot_widget = pg.PlotWidget()
            plot_widget.setLabel('left', 'Voltaje', units='V')
            if i == len(CANALES) - 1:
                plot_widget.setLabel('bottom', 'Tiempo', units='s')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setYRange(VOLTAJE_MIN, VOLTAJE_MAX)
            pen = pg.mkPen(color=('r', 'b')[i % 2], width=2)
            plot_curve = plot_widget.plot(pen=pen)
            signals_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            self.plot_curves.append(plot_curve)
        
        # ============= Diagrama de fase 2D en tiempo real (opcional) =============
        # Si quieres un Lissajous 2D en tiempo real:
        self.phase2d_widget = pg.PlotWidget()
        self.phase2d_widget.setLabel('left', f"{CANALES[1]} (norm)")
        self.phase2d_widget.setLabel('bottom', f"{CANALES[0]} (norm)")
        self.phase2d_widget.showGrid(x=True, y=True)
        self.phase2d_curve = self.phase2d_widget.plot(pen=pg.mkPen('g', width=2))
        signals_layout.addWidget(self.phase2d_widget)
        
        # Añadimos layout de señales al contenedor
        plots_layout.addLayout(signals_layout, stretch=1)
        
        # ============= Diagrama 3D en tiempo real (PyQtGraph OpenGL) =============
        self.glview = gl.GLViewWidget()
        self.glview.setCameraPosition(distance=10)  # Ajusta la distancia a la escena
        # Añadimos una malla o grid para referencia
        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        grid.setSize(10, 10, 1)
        self.glview.addItem(grid)
        
        # Creamos un GLLinePlotItem para graficar en 3D
        self.phase3d_line = gl.GLLinePlotItem(
            color=(0, 255, 0, 255),  # verde
            width=2,
            antialias=True
        )
        self.glview.addItem(self.phase3d_line)
        
        # Añadimos la vista 3D al layout
        plots_layout.addWidget(self.glview, stretch=1)
        
        # ============= Controles y selección de diagramas =============
        control_layout = QtWidgets.QHBoxLayout()
        
        self.start_stop_button = QtWidgets.QPushButton("Detener")
        self.start_stop_button.clicked.connect(self.toggle_acquisition)
        control_layout.addWidget(self.start_stop_button)
        
        control_layout.addWidget(QtWidgets.QLabel("Tiempo:"))
        self.time_combo = QtWidgets.QComboBox()
        self.time_combo.addItems(["10 ms", "20 ms", "50 ms", "100 ms", "200 ms", "500 ms", "1 s"])
        self.time_combo.setCurrentText("50 ms")
        self.time_combo.currentTextChanged.connect(self.update_timebase)
        control_layout.addWidget(self.time_combo)
        
        control_layout.addWidget(QtWidgets.QLabel("Modo:"))
        self.terminal_mode_combo = QtWidgets.QComboBox()
        self.terminal_mode_combo.addItems(["Diferencial", "RSE", "NRSE"])
        self.terminal_mode_combo.setCurrentText("Diferencial")
        control_layout.addWidget(self.terminal_mode_combo)
        
        self.apply_mode_button = QtWidgets.QPushButton("Aplicar Modo")
        self.apply_mode_button.clicked.connect(self.apply_mode_changes)
        control_layout.addWidget(self.apply_mode_button)
        
        self.save_button = QtWidgets.QPushButton("Guardar Datos")
        self.save_button.clicked.connect(self.save_data)
        control_layout.addWidget(self.save_button)
        
        main_layout.addLayout(control_layout)
        
        # Iniciar el hilo de adquisición
        self.adquisicion_thread = AdquisicionThread(
            DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config
        )
        self.adquisicion_thread.start()
        
        # Timer para actualizar la gráfica en tiempo real
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(20)  # ~50 FPS
        
        # Almacenar datos acumulados
        self.all_data = [[] for _ in range(len(CANALES))]
    
    # -------------------- Funciones de Interfaz --------------------
    def toggle_acquisition(self):
        if self.acquiring:
            self.acquiring = False
            self.start_stop_button.setText("Iniciar")
            self.adquisicion_thread.stop()
        else:
            self.acquiring = True
            self.start_stop_button.setText("Detener")
            self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
            self.all_data = [[] for _ in range(len(CANALES))]
            self.adquisicion_thread = AdquisicionThread(
                DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config
            )
            self.adquisicion_thread.start()
    
    def update_timebase(self):
        time_text = self.time_combo.currentText()
        time_val = float(time_text.split()[0]) / 1000.0 if "ms" in time_text else float(time_text.split()[0])
        self.time_window = time_val
        self.buffer_size = max(int(self.time_window * self.sample_rate), 2)
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        for widget in self.plot_widgets:
            widget.setXRange(-self.time_window, 0)
    
    def apply_mode_changes(self):
        if self.acquiring:
            self.adquisicion_thread.stop()
            time.sleep(0.5)
        modo_text = self.terminal_mode_combo.currentText()
        self.terminal_config = TERMINAL_CONFIG_MAP[modo_text]
        print(f"Aplicando modo de entrada: {modo_text}")
        self.adquisicion_thread = AdquisicionThread(
            DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config
        )
        if self.acquiring:
            self.adquisicion_thread.start()
            self.start_stop_button.setText("Detener")
    
    def update_plots(self):
        """
        Llamada periódicamente (~50 FPS). Lee la cola, actualiza los buffers
        y dibuja en tiempo real:
         - Señales en 2D
         - Diagrama 2D Lissajous
         - Diagrama 3D con pyqtgraph.opengl
        """
        try:
            while not datos_queue.empty():
                nuevos_datos = datos_queue.get_nowait()
                for i, canal_datos in enumerate(nuevos_datos):
                    self.all_data[i].extend(canal_datos)
                    num_nuevas = len(canal_datos)
                    if num_nuevas >= self.buffer_size:
                        self.datos_buffer[i][:] = canal_datos[-self.buffer_size:]
                    else:
                        self.datos_buffer[i][:-num_nuevas] = self.datos_buffer[i][num_nuevas:]
                        self.datos_buffer[i][-num_nuevas:] = canal_datos
                
                # Calcular métricas en cada canal
                for i, canal in enumerate(CANALES):
                    freq, amp, rms = self.calcular_metricas_senal(self.datos_buffer[i], self.sample_rate)
                    valor_actual = self.datos_buffer[i][-1]
                    self.value_labels[i].setText(f"Voltaje: {valor_actual:.3f} V")
                    self.freq_labels[i].setText(f"Frecuencia: {freq:.1f} Hz")
                    self.amp_labels[i].setText(f"Amplitud: {amp:.3f} Vpp")
                    self.rms_labels[i].setText(f"RMS: {rms:.3f} V")
        
        except Exception as e:
            print(f"Error al actualizar la gráfica: {e}")
        
        # Actualizar gráficas 2D (señales vs tiempo)
        for i, curve in enumerate(self.plot_curves):
            curve.setData(self.tiempo, self.datos_buffer[i])
        
        # -------------------- Diagrama 2D (Lissajous) --------------------
        n_fase = 500
        sig0 = self.datos_buffer[0][-n_fase:]
        sig1 = self.datos_buffer[1][-n_fase:]
        sig0n = normalize_signal(sig0)
        sig1n = normalize_signal(sig1)
        self.phase2d_curve.setData(sig0n, sig1n)
        
        # -------------------- Diagrama 3D en tiempo real --------------------
        # Usamos el mismo subset n_fase, y un eje 'tiempo' que va de -1 a 0 (por ejemplo).
        time_axis = np.linspace(-1, 0, len(sig0n))
        
        # Creamos un array (N,3) con (x,y,z)
        pos = np.zeros((len(sig0n), 3), dtype=np.float32)
        pos[:, 0] = time_axis       # Eje X => tiempo
        pos[:, 1] = sig0n          # Eje Y => canal 0
        pos[:, 2] = sig1n          # Eje Z => canal 1
        
        # Actualizamos la línea 3D
        self.phase3d_line.setData(pos=pos)
    
    def calcular_metricas_senal(self, datos, sample_rate):
        rms = np.sqrt(np.mean(np.square(datos)))
        amplitud = np.max(datos) - np.min(datos)
        zero_mean_data = datos - np.mean(datos)
        cruces = np.where(np.diff(np.signbit(zero_mean_data)))[0]
        if len(cruces) >= 4:
            periodos = [cruces[i+2] - cruces[i] for i in range(0, len(cruces)-2, 2)]
            if periodos:
                periodo_promedio = np.mean(periodos)
                frecuencia = sample_rate / periodo_promedio if periodo_promedio > 0 else 0
            else:
                frecuencia = 0
        else:
            frecuencia = 0
        return frecuencia, amplitud, rms
    
    def save_data(self):
        if not self.all_data[0]:
            QtWidgets.QMessageBox.warning(self, "Sin datos", "No hay datos para guardar. Inicia la adquisición primero.")
            return
        filename = f"datos_osciloscopio_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        datos_np = np.array([np.array(canal_data) for canal_data in self.all_data])
        np.savetxt(filename, np.transpose(datos_np), delimiter=",",
                   header=",".join([f"Canal_{canal}" for canal in CANALES]), comments="")
        self.all_data = [[] for _ in range(len(CANALES))]
        QtWidgets.QMessageBox.information(self, "Datos Guardados",
                                          f"Los datos se han guardado correctamente en '{filename}'.")
    
    def closeEvent(self, event):
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
    print(f"\nIniciando osciloscopio con diagramas 2D y 3D en tiempo real:")
    print(f"  Dispositivo: {DISPOSITIVO}")
    print(f"  Canales: {CANALES}")
    print(f"  Muestreo: {SAMPLE_RATE} Hz")
    print(f"  Ventana de tiempo inicial: {TIME_WINDOW * 1000} ms")
    
    app = QtWidgets.QApplication([])
    ventana = OsciloscopioApp()
    ventana.show()
    app.exec_()

if __name__ == "__main__":
    main()

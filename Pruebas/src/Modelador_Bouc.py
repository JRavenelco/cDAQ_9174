import nidaqmx
from nidaqmx.system import System
import numpy as np
import time
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necesario para gráficos 3D
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
        self.setWindowTitle('Osciloscopio - Múltiples Diagramas de Fase')
        self.resize(1200, 800)
        
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
        
        # Gráficas en PyQtGraph
        self.plot_widgets, self.plot_curves = [], []
        for i, canal in enumerate(CANALES):
            plot_widget = pg.PlotWidget()
            plot_widget.setLabel('left', 'Voltaje', units='V')
            if i == len(CANALES) - 1:
                plot_widget.setLabel('bottom', 'Tiempo', units='s')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setYRange(VOLTAJE_MIN, VOLTAJE_MAX)
            pen = pg.mkPen(color=('r', 'b')[i % 2], width=2)
            plot_curve = plot_widget.plot(pen=pen)
            main_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            self.plot_curves.append(plot_curve)
        
        # Controles y selección de diagramas
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
        
        # Selector para diagramas de fase
        self.add_diagram_selector_to_ui(control_layout)
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
        
        # Almacenar datos acumulados para análisis y gráficos de fase
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
                for i, canal in enumerate(CANALES):
                    freq, amp, rms = self.calcular_metricas_senal(self.datos_buffer[i], self.sample_rate)
                    valor_actual = self.datos_buffer[i][-1]
                    self.value_labels[i].setText(f"Voltaje: {valor_actual:.3f} V")
                    self.freq_labels[i].setText(f"Frecuencia: {freq:.1f} Hz")
                    self.amp_labels[i].setText(f"Amplitud: {amp:.3f} Vpp")
                    self.rms_labels[i].setText(f"RMS: {rms:.3f} V")
        except Exception as e:
            print(f"Error al actualizar la gráfica: {e}")
        for i, curve in enumerate(self.plot_curves):
            curve.setData(self.tiempo, self.datos_buffer[i])
    
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
    
    # -------------------- Funciones para Diagramas de Fase --------------------
    def add_diagram_selector_to_ui(self, control_layout):
        control_layout.addWidget(QtWidgets.QLabel("Tipo de Diagrama:"))
        self.phase_type_combo = QtWidgets.QComboBox()
        self.phase_type_combo.addItems(["Lissajous 2D", "Diagrama 3D", "Animación", "Multi-vista"])
        control_layout.addWidget(self.phase_type_combo)
        self.phase_button = QtWidgets.QPushButton("Mostrar Diagrama")
        self.phase_button.clicked.connect(self.show_selected_phase_diagram)
        control_layout.addWidget(self.phase_button)
    
    def show_selected_phase_diagram(self):
        diagram_type = self.phase_type_combo.currentText()
        if diagram_type == "Lissajous 2D":
            self.plot_phase_diagram_2d()
        elif diagram_type == "Diagrama 3D":
            self.plot_phase_diagram_3d()
        elif diagram_type == "Animación":
            self.plot_phase_animation()
        elif diagram_type == "Multi-vista":
            self.plot_multi_view_phase_diagram()
    
    def plot_phase_diagram_2d(self):
        num_pts = 5000
        if len(self.all_data[0]) < num_pts:
            num_pts = len(self.all_data[0])
        if num_pts < 2:
            QtWidgets.QMessageBox.warning(self, "Datos insuficientes", "No hay suficientes datos para graficar el diagrama de fase.")
            return
        # Obtener y normalizar los datos
        signal1 = normalize_signal(np.array(self.all_data[0][-num_pts:]))
        signal2 = normalize_signal(np.array(self.all_data[1][-num_pts:]))
        
        plt.figure(figsize=(10, 8), dpi=100)
        scatter = plt.scatter(signal1, signal2, c=np.arange(num_pts), cmap='viridis', s=3, alpha=0.8)
        plt.plot(signal1, signal2, 'k-', linewidth=0.5, alpha=0.5)
        plt.plot(signal1[-1], signal2[-1], 'ro', markersize=8)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Tiempo (muestras)')
        plt.xlabel(f"Canal {CANALES[0]} (normalizado)", fontsize=12)
        plt.ylabel(f"Canal {CANALES[1]} (normalizado)", fontsize=12)
        plt.title("Diagrama de Fase 2D (Lissajous)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        try:
            freq1, _, _ = self.calcular_metricas_senal(signal1, self.sample_rate)
            freq2, _, _ = self.calcular_metricas_senal(signal2, self.sample_rate)
            relacion_texto = f"{freq1:.1f} Hz / {freq2:.1f} Hz = {freq1/freq2:.3f}" if freq2 > 0 else "No se puede calcular"
            plt.figtext(0.02, 0.02, f"Relación de frecuencias: {relacion_texto}", fontsize=10, ha='left')
        except Exception as ex:
            print(f"Error al calcular la relación de frecuencias: {ex}")
        plt.tight_layout()
        plt.show()
    
    def plot_phase_diagram_3d(self):
        num_pts = 5000
        if len(self.all_data[0]) < num_pts:
            num_pts = len(self.all_data[0])
        if num_pts < 2:
            QtWidgets.QMessageBox.warning(self, "Datos insuficientes", "No hay suficientes datos para graficar el diagrama de fase 3D.")
            return
        signal1 = normalize_signal(np.array(self.all_data[0][-num_pts:]))
        signal2 = normalize_signal(np.array(self.all_data[1][-num_pts:]))
        time_vector = np.linspace(0, num_pts / self.sample_rate, num_pts)
        fig = plt.figure(figsize=(12, 10), dpi=100)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        ax3d = fig.add_subplot(gs[0, :], projection='3d')
        scatter = ax3d.scatter(time_vector, signal1, signal2, c=np.arange(num_pts),
                               cmap='viridis', s=5, alpha=0.8)
        ax3d.plot(time_vector, signal1, signal2, 'k-', linewidth=0.5, alpha=0.5)
        ax3d.set_xlabel("Tiempo (s)", fontsize=12)
        ax3d.set_ylabel(f"Canal {CANALES[0]} (normalizado)", fontsize=12)
        ax3d.set_zlabel(f"Canal {CANALES[1]} (normalizado)", fontsize=12)
        ax3d.set_title("Diagrama de Fase 3D", fontsize=14)
        cbar = fig.colorbar(scatter, ax=ax3d, pad=0.1)
        cbar.set_label('Tiempo (muestras)')
        ax3d.view_init(elev=30, azim=-45)
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(time_vector, signal1, 'r-')
        ax1.set_xlabel("Tiempo (s)")
        ax1.set_ylabel(f"Canal {CANALES[0]} (normalizado)")
        ax1.grid(True)
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.plot(time_vector, signal2, 'b-')
        ax2.set_xlabel("Tiempo (s)")
        ax2.set_ylabel(f"Canal {CANALES[1]} (normalizado)")
        ax2.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_multi_view_phase_diagram(self):
        num_pts = 5000
        if len(self.all_data[0]) < num_pts:
            num_pts = len(self.all_data[0])
        if num_pts < 2:
            QtWidgets.QMessageBox.warning(self, "Datos insuficientes", "No hay suficientes datos para graficar las múltiples vistas.")
            return
        signal1 = normalize_signal(np.array(self.all_data[0][-num_pts:]))
        signal2 = normalize_signal(np.array(self.all_data[1][-num_pts:]))
        dsignal1 = savgol_filter(np.gradient(signal1), 15, 2)
        time_vector = np.linspace(0, num_pts / self.sample_rate, num_pts)
        fig = plt.figure(figsize=(15, 10), dpi=100)
        gs = fig.add_gridspec(3, 3)
        ax3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        scatter = ax3d.scatter(signal1, signal2, dsignal1, c=np.arange(num_pts),
                               cmap='viridis', s=5, alpha=0.8)
        ax3d.plot(signal1, signal2, dsignal1, 'k-', linewidth=0.5, alpha=0.5)
        ax3d.set_xlabel(f"Canal {CANALES[0]} (normalizado)")
        ax3d.set_ylabel(f"Canal {CANALES[1]} (normalizado)")
        ax3d.set_zlabel("Derivada (norm)")
        ax3d.set_title("Diagrama de Fase 3D con Derivada")
        ax3d.view_init(elev=30, azim=45)
        ax_xy = fig.add_subplot(gs[0, 2])
        ax_xy.scatter(signal1, signal2, c=np.arange(num_pts), cmap='viridis', s=3, alpha=0.6)
        ax_xy.plot(signal1, signal2, 'k-', linewidth=0.5, alpha=0.5)
        ax_xy.set_xlabel(f"Canal {CANALES[0]} (normalizado)")
        ax_xy.set_ylabel(f"Canal {CANALES[1]} (normalizado)")
        ax_xy.set_title("Proyección XY (Lissajous)")
        ax_xy.grid(True)
        ax_xz = fig.add_subplot(gs[1, 2])
        ax_xz.scatter(signal1, dsignal1, c=np.arange(num_pts), cmap='viridis', s=3, alpha=0.6)
        ax_xz.plot(signal1, dsignal1, 'k-', linewidth=0.5, alpha=0.5)
        ax_xz.set_xlabel(f"Canal {CANALES[0]} (normalizado)")
        ax_xz.set_ylabel("Derivada (norm)")
        ax_xz.set_title("Proyección X-Derivada")
        ax_xz.grid(True)
        ax_time1 = fig.add_subplot(gs[2, 0])
        ax_time1.plot(time_vector, signal1, 'r-')
        ax_time1.set_xlabel("Tiempo (s)")
        ax_time1.set_ylabel(f"Canal {CANALES[0]} (normalizado)")
        ax_time1.set_title(f"Señal Canal {CANALES[0]}")
        ax_time1.grid(True)
        ax_time2 = fig.add_subplot(gs[2, 1])
        ax_time2.plot(time_vector, signal2, 'b-')
        ax_time2.set_xlabel("Tiempo (s)")
        ax_time2.set_ylabel(f"Canal {CANALES[1]} (normalizado)")
        ax_time2.set_title(f"Señal Canal {CANALES[1]}")
        ax_time2.grid(True)
        ax_deriv = fig.add_subplot(gs[2, 2])
        ax_deriv.plot(time_vector, dsignal1, 'g-')
        ax_deriv.set_xlabel("Tiempo (s)")
        ax_deriv.set_ylabel("Derivada (norm)")
        ax_deriv.set_title("Derivada de la Señal")
        ax_deriv.grid(True)
        cbar = fig.colorbar(scatter, ax=[ax3d, ax_xy, ax_xz], pad=0.01)
        cbar.set_label('Tiempo (muestras)')
        plt.tight_layout()
        plt.show()
    
    def plot_phase_animation(self):
        """
        Crea una animación del diagrama de fase (Lissajous) que muestra la evolución
        de la señal con el tiempo. Se fijan los ejes para que la visualización sea estable.
        Los datos se normalizan para que siempre se grafique en el rango -1 a 1.
        """
        num_pts = 5000
        if len(self.all_data[0]) < num_pts:
            num_pts = len(self.all_data[0])
        if num_pts < 100:
            QtWidgets.QMessageBox.warning(self, "Datos insuficientes", "No hay suficientes datos para la animación.")
            return
        
        # Normalizar los datos
        signal1 = normalize_signal(np.array(self.all_data[0][-num_pts:]))
        signal2 = normalize_signal(np.array(self.all_data[1][-num_pts:]))
        
        # Fijar ejes según los datos normalizados (rango -1 a 1)
        centro_x = 0
        centro_y = 0
        radio = 1.0
        
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.set_xlim(centro_x - radio, centro_x + radio)
        ax.set_ylim(centro_y - radio, centro_y + radio)
        ax.set_xlabel(f"Canal {CANALES[0]} (normalizado)")
        ax.set_ylabel(f"Canal {CANALES[1]} (normalizado)")
        ax.set_title("Animación del Diagrama de Fase")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Inicialización de la línea y el punto de la animación
        line, = ax.plot([], [], 'b-', lw=1.5, alpha=0.8)
        point, = ax.plot([], [], 'ro', ms=6)
        
        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point
        
        def update(frame):
            x = signal1[:frame]
            y = signal2[:frame]
            line.set_data(x, y)
            if frame > 0:
                # Envolver los valores escalares en listas
                point.set_data([x[-1]], [y[-1]])
            return line, point
        
        ani = animation.FuncAnimation(fig, update, frames=range(1, num_pts+1, 10),
                                      init_func=init, blit=True, interval=20)
        plt.tight_layout()
        plt.show()
    
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
    print(f"\nIniciando osciloscopio con selección de modo:")
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

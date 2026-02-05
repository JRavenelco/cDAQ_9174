import nidaqmx
from nidaqmx.system import System
import numpy as np
import time
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl
from nidaqmx.constants import AcquisitionType, TerminalConfiguration
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------
# Intentar importar CuPy para GPU (opcional)
# --------------------------------------------------------------------
try:
    import cupy as cp
    from cupyx.scipy.optimize import least_squares as cp_least_squares
    USE_GPU = True
    print("Se utilizará CuPy para la optimización en GPU.")
except ImportError:
    USE_GPU = False
    from scipy.optimize import least_squares
    print("CuPy no disponible. Se usará optimización en CPU.")

# --------------------------------------------------------------------
# Kernel personalizado (opcional, placeholder)
# --------------------------------------------------------------------
from cupy import RawKernel

kernel_code = r'''
extern "C" __global__ void bouc_wen_parallel(
    const float* dt, const float* du,
    float* z_out, float A, float B, float C, float n,
    int N, int num_series) {
    
    int i = blockIdx.x; // Cada bloque procesa una serie (índice i)
    if (i >= num_series) return;
    
    int out_offset = i * N;
    int delta_offset = i * (N - 1);
    
    float z = 0.0f;
    z_out[out_offset] = z;
    for (int j = 1; j < N; j++) {
        float dti = dt[delta_offset + j - 1];
        float dui = du[delta_offset + j - 1];
        float delta_z = (A * dui - B * fabsf(dui) * z - C * dui * powf(fabsf(z), n)) * dti;
        z = z + delta_z;
        if (z < -1000.0f) z = -1000.0f;
        if (z > 1000.0f) z = 1000.0f;
        z_out[out_offset + j] = z;
    }
}
'''
bouc_wen_kernel = RawKernel(kernel_code, 'bouc_wen_parallel')

def bouc_wen_model_gpu_parallel(params, t, corriente_batch):
    # (Placeholder, si quisieras usar GPU para múltiples series)
    pass

# --------------------------------------------------------------------
# Función para mostrar info del hardware NI-DAQ
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# CONFIGURACIÓN PRINCIPAL
# --------------------------------------------------------------------
DISPOSITIVO = "cDAQ1Mod1"
CANALES = ["ai0", "ai1"]  # ai0: corriente, ai1: fuerza
SAMPLE_RATE = 10000
MUESTRAS_POR_BLOQUE = 1000
TIME_WINDOW = 0.05
VOLTAJE_MIN = -5.0
VOLTAJE_MAX = 5.0

TERMINAL_CONFIG_MAP = {
    "Diferencial": TerminalConfiguration.DIFF,
    "RSE": TerminalConfiguration.RSE,
    "NRSE": TerminalConfiguration.NRSE
}

datos_queue = queue.Queue(maxsize=10)

def normalize_signal(signal):
    max_abs = np.max(np.abs(signal))
    if max_abs == 0:
        return signal
    return signal / max_abs

# --------------------------------------------------------------------
# Modelo Bouc-Wen (CPU)
# --------------------------------------------------------------------
def bouc_wen_model(params, t, corriente):
    A, B, C, n, k = params
    N = len(t)
    fuerza_modelada = np.zeros(N)
    z_vals = np.zeros(N)
    z = 0.0
    z_min, z_max = -1e3, 1e3
    for i in range(1, N):
        dt = t[i] - t[i-1]
        du = (corriente[i] - corriente[i-1]) / dt if dt != 0 else 0.0
        delta_z = (A * du - B * abs(du) * z - C * du * (abs(z) ** n)) * dt
        z = z + delta_z
        z = np.clip(z, z_min, z_max)
        z_vals[i] = z
        fuerza_modelada[i] = k * (z ** 2)
    return fuerza_modelada, z_vals

def error_bouc_wen(params, t, corriente, fuerza_exp):
    fuerza_modelada, _ = bouc_wen_model(params, t, corriente)
    return fuerza_modelada - fuerza_exp

# --------------------------------------------------------------------
# Hilo de Adquisición
# --------------------------------------------------------------------
class AdquisicionThread(threading.Thread):
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
            print("[AdquisicionThread] Creando Task NI-DAQmx...")
            self.task = nidaqmx.Task()
            for canal in self.canales:
                nombre_canal = f"{self.dispositivo}/{canal}"
                print(f"[AdquisicionThread] Configurando canal: {nombre_canal} en modo {self.terminal_config}")
                self.task.ai_channels.add_ai_voltage_chan(
                    nombre_canal,
                    terminal_config=self.terminal_config,
                    min_val=VOLTAJE_MIN,
                    max_val=VOLTAJE_MAX
                )
            self.task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10
            )
            print(f"[AdquisicionThread] Iniciando adquisición a {self.sample_rate} Hz en {self.canales}")
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
                        datos_queue.get_nowait()
                        datos_queue.put(datos_np, block=False)
                except nidaqmx.errors.DaqError as e:
                    print(f"[AdquisicionThread] Error en adquisición: {e}")
                    if "timeout" not in str(e).lower():
                        break
            if self.task:
                self.task.stop()
                self.task.close()
                print("[AdquisicionThread] Tarea de adquisición detenida y cerrada.")
        except Exception as e:
            print("[AdquisicionThread] ERROR en el hilo de adquisición:", e)
            if self.task:
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass

    def stop(self):
        print("[AdquisicionThread] stop() llamado, deteniendo la adquisición.")
        self.running = False

# --------------------------------------------------------------------
# CLASE PRINCIPAL: PREDICTOR DE HISTÉRESIS
# --------------------------------------------------------------------
class PredictorHisteresis(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        print("[PredictorHisteresis] __init__ iniciado.")
        self.setWindowTitle('Predictor de Histeresis - Modelo Bouc-Wen')
        self.resize(1500, 900)

        # Parámetros por defecto (se sobrescriben tras optimización)
        self.params_opt = [0.9, 0.4, 0.4, 2.1, 0.9]

        self.acquiring = True
        self.sample_rate = SAMPLE_RATE
        self.time_window = TIME_WINDOW
        self.terminal_config = TerminalConfiguration.DIFF
        self.buffer_size = max(int(self.time_window * self.sample_rate), 2)

        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)

        self._optimization_running = False
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        # Panel de información de señales
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

        # Layout para gráficas (2 columnas)
        plots_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(plots_layout, stretch=1)

        # ------------------ Columna Izquierda ------------------
        left_layout = QtWidgets.QVBoxLayout()

        # Señales vs. tiempo (una por canal)
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
            left_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            self.plot_curves.append(plot_curve)

        # Lissajous 2D (corriente vs. fuerza normalizados)
        self.phase2d_widget = pg.PlotWidget()
        self.phase2d_widget.setLabel('left', f"{CANALES[1]} (norm)")
        self.phase2d_widget.setLabel('bottom', f"{CANALES[0]} (norm)")
        self.phase2d_widget.showGrid(x=True, y=True)
        self.phase2d_curve = self.phase2d_widget.plot(pen=pg.mkPen('g', width=2))
        left_layout.addWidget(self.phase2d_widget)

        # Agregamos la columna izquierda al layout principal
        plots_layout.addLayout(left_layout, stretch=1)

        # ------------------ Columna Derecha ------------------
        right_layout = QtWidgets.QVBoxLayout()

        # Gráfica 3D con OpenGL
        self.glview = gl.GLViewWidget()
        self.glview.setCameraPosition(distance=10)
        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        grid.setSize(10, 10, 1)
        self.glview.addItem(grid)
        self.phase3d_line = gl.GLLinePlotItem(color=(0, 255, 0, 255), width=2, antialias=True)
        self.glview.addItem(self.phase3d_line)
        right_layout.addWidget(self.glview, stretch=1)

        # Diagrama de histéresis: z(t) vs. corriente(t)
        self.bw_phase_widget = pg.PlotWidget()
        self.bw_phase_widget.setLabel('left', 'z(t) (Bouc-Wen)')
        self.bw_phase_widget.setLabel('bottom', 'Corriente (A)')
        self.bw_phase_widget.showGrid(x=True, y=True)
        self.bw_phase_curve = self.bw_phase_widget.plot(pen=pg.mkPen('m', width=2))
        right_layout.addWidget(self.bw_phase_widget, stretch=1)

        # Agregamos la columna derecha al layout principal
        plots_layout.addLayout(right_layout, stretch=1)

        # Panel de Optimización
        optim_layout = QtWidgets.QHBoxLayout()
        self.optimizar_button = QtWidgets.QPushButton("Ajustar Modelo")
        self.optimizar_button.clicked.connect(self.ajustar_modelo)
        optim_layout.addWidget(self.optimizar_button)

        self.params_group = QtWidgets.QGroupBox("Parámetros Bouc-Wen Ajustados")
        form_layout = QtWidgets.QFormLayout()
        self.param_labels = {}
        for param in ["A", "B", "C", "n", "k"]:
            lbl = QtWidgets.QLabel("-")
            form_layout.addRow(f"{param}:", lbl)
            self.param_labels[param] = lbl
        self.params_group.setLayout(form_layout)
        optim_layout.addWidget(self.params_group)
        main_layout.addLayout(optim_layout)

        # Controles generales
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

        print("[PredictorHisteresis] Creando AdquisicionThread...")
        self.adquisicion_thread = AdquisicionThread(DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config)
        self.adquisicion_thread.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(20)

        print("[PredictorHisteresis] Creando opt_timer con 10s...")
        self.opt_timer = QtCore.QTimer()
        self.opt_timer.timeout.connect(self.ejecutar_optimizacion)
        self.opt_timer.start(10000)

        self.all_data = [[] for _ in range(len(CANALES))]
        print("[PredictorHisteresis] __init__ finalizado.")

    def toggle_acquisition(self):
        if self.acquiring:
            print("[toggle_acquisition] Deteniendo la adquisición.")
            self.acquiring = False
            self.start_stop_button.setText("Iniciar")
            self.adquisicion_thread.stop()
        else:
            print("[toggle_acquisition] Iniciando la adquisición.")
            self.acquiring = True
            self.start_stop_button.setText("Detener")
            self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
            self.all_data = [[] for _ in range(len(CANALES))]
            self.adquisicion_thread = AdquisicionThread(DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config)
            self.adquisicion_thread.start()

    def update_timebase(self):
        print("[update_timebase] Cambio de base de tiempo.")
        time_text = self.time_combo.currentText()
        time_val = float(time_text.split()[0]) / 1000.0 if "ms" in time_text else float(time_text.split()[0])
        self.time_window = time_val
        self.buffer_size = max(int(self.time_window * self.sample_rate), 2)
        self.tiempo = np.linspace(-self.time_window, 0, self.buffer_size)
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        for widget in self.plot_widgets:
            widget.setXRange(-self.time_window, 0)

    def apply_mode_changes(self):
        print("[apply_mode_changes] Aplicando modo de entrada.")
        if self.acquiring:
            self.adquisicion_thread.stop()
            time.sleep(0.5)
        modo_text = self.terminal_mode_combo.currentText()
        self.terminal_config = TERMINAL_CONFIG_MAP[modo_text]
        print(f"[apply_mode_changes] Modo seleccionado: {modo_text}")
        self.adquisicion_thread = AdquisicionThread(DISPOSITIVO, CANALES, self.sample_rate, MUESTRAS_POR_BLOQUE, self.terminal_config)
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
            print(f"[update_plots] Error al actualizar la gráfica: {e}")

        # Actualizar curvas de señales vs. tiempo
        for i, curve in enumerate(self.plot_curves):
            curve.setData(self.tiempo, self.datos_buffer[i])

        # Diagrama Lissajous 2D (corriente vs. fuerza normalizados)
        n_fase = 500
        sig0 = self.datos_buffer[0][-n_fase:]
        sig1 = self.datos_buffer[1][-n_fase:]
        sig0n = normalize_signal(sig0)
        sig1n = normalize_signal(sig1)
        self.phase2d_curve.setData(sig0n, sig1n)

        # Diagrama 3D
        time_axis = np.linspace(-1, 0, len(sig0n))
        pos = np.zeros((len(sig0n), 3), dtype=np.float32)
        pos[:, 0] = time_axis
        pos[:, 1] = sig0n
        pos[:, 2] = sig1n
        self.phase3d_line.setData(pos=pos)

        # Diagrama de histéresis z(t) vs. corriente(t)
        if len(self.all_data[0]) >= n_fase:
            t_data_bw = np.linspace(0, (n_fase - 1)/self.sample_rate, n_fase)
            corriente_segment = np.array(self.all_data[0][-n_fase:])
            params_bw = self.params_opt
            _, z_vals = bouc_wen_model(params_bw, t_data_bw, corriente_segment)
            self.bw_phase_curve.setData(corriente_segment, z_vals)

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

    def ajustar_modelo(self):
        print("[ajustar_modelo] Iniciando...")
        if self._optimization_running:
            print("[ajustar_modelo] Ya hay una optimización en curso. Espere...")
            return

        N = min(3000, len(self.all_data[0]))
        print(f"[ajustar_modelo] N calculado = {N}, total all_data[0] = {len(self.all_data[0])}")
        if N < 100:
            print("[ajustar_modelo] Datos insuficientes, se aborta.")
            QtWidgets.QMessageBox.warning(self, "Datos insuficientes", "No hay suficientes datos para optimizar.")
            return

        self._optimization_running = True
        self.optimizar_button.setEnabled(False)
        self.optimizar_button.setText("Optimizando...")

        t_data = np.linspace(0, N / self.sample_rate, N)
        corriente_data = np.array(self.all_data[0][-N:])
        fuerza_data = np.array(self.all_data[1][-N:])

        params0 = [0.9, 0.4, 0.4, 2.1, 0.9]

        def optimize():
            try:
                print("[optimize] Iniciando optimización real...")
                start_opt = time.time()
                corriente_smooth = savgol_filter(corriente_data, 15, 3)
                fuerza_smooth = savgol_filter(fuerza_data, 15, 3)
                if USE_GPU:
                    print("[optimize] GPU route (ejemplo). Retornando params0 sin cambios.")
                    params_opt = params0
                    final_error = 999.99
                else:
                    print("[optimize] CPU route con SciPy least_squares.")
                    from scipy.optimize import least_squares
                    res = least_squares(
                        error_bouc_wen, params0,
                        args=(t_data, corriente_smooth, fuerza_smooth),
                        method='lm', ftol=1e-4, xtol=1e-4, max_nfev=100
                    )
                    params_opt = res.x
                    final_error = res.cost
                elapsed = time.time() - start_opt
                print(f"[optimize] Optimización finalizada en {elapsed:.3f} s")
                return params_opt, final_error, t_data, corriente_data, fuerza_data
            except Exception as ex:
                print(f"[optimize] ERROR durante la optimización: {ex}")
                raise

        def update_ui(fut):
            print("[update_ui] Iniciando actualización de la UI...")
            try:
                result = fut.result()
                print("[update_ui] future.result() =", result)
                if result is None:
                    print("[update_ui] La optimización falló (result es None).")
                    return
                params_opt, final_error, t_data_res, corriente_res, fuerza_res = result
                print("[update_ui] params_opt =", params_opt)
                print("[update_ui] final_error =", final_error)
                if len(params_opt) != 5:
                    raise ValueError(f"[update_ui] params_opt no tiene 5 valores: {params_opt}")
                keys = ["A", "B", "C", "n", "k"]
                for k, value in zip(keys, params_opt):
                    print(f"[update_ui] Asignando {k} = {value}")
                    self.param_labels[k].setText(f"{value:.4f}")

                # Guardar los parámetros optimizados en self.params_opt
                self.params_opt = params_opt

                print(f"[update_ui] Parámetros ajustados (Bouc-Wen): {params_opt}")
                print(f"[update_ui] Error final: {final_error}")
                print("[update_ui] Llamando a mostrar_grafico_comparacion...")
                self.executor.submit(self.mostrar_grafico_comparacion, params_opt, t_data_res, corriente_res, fuerza_res)
            except Exception as ex:
                print("[update_ui] ERROR al actualizar la UI:", ex)
                raise
            finally:
                self._optimization_running = False
                self.optimizar_button.setEnabled(True)
                self.optimizar_button.setText("Ajustar Modelo")

        print("[ajustar_modelo] Llamando a optimize() en segundo plano.")
        future = self.executor.submit(optimize)
        future.add_done_callback(lambda fut: QtCore.QTimer.singleShot(0, lambda: update_ui(fut)))

    def ejecutar_optimizacion(self):
        print(f"[ejecutar_optimizacion] Timer fired. _optimization_running = {self._optimization_running}")
        if not self._optimization_running:
            if len(self.all_data[0]) > 10000:
                for i in range(len(self.all_data)):
                    self.all_data[i] = self.all_data[i][-10000:]
            self.ajustar_modelo()

    def mostrar_grafico_comparacion(self, params_opt, t_data, corriente_data, fuerza_data):
        print("[mostrar_grafico_comparacion] Iniciando...")
        try:
            print("[mostrar_grafico_comparacion] Calculando fuerza_modelada con bouc_wen_model")
            fuerza_modelada, _ = bouc_wen_model(params_opt, t_data, corriente_data)
            print("[mostrar_grafico_comparacion] fuerza_modelada shape =", fuerza_modelada.shape)
            plt.figure()
            plt.plot(t_data, fuerza_data, label="Fuerza Experimental")
            plt.plot(t_data, fuerza_modelada, label="Fuerza Modelada")
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Fuerza")
            plt.legend()
            plt.title("Comparación Fuerza Experimental vs. Modelo")
            plt.show(block=False)
            print("[mostrar_grafico_comparacion] Gráfica mostrada (Matplotlib).")
        except Exception as ex:
            print("[mostrar_grafico_comparacion] ERROR:", ex)
            raise

    def save_data(self):
        print("[save_data] Guardando datos.")
        if not self.all_data[0]:
            QtWidgets.QMessageBox.warning(self, "Sin datos", "No hay datos para guardar. Inicia la adquisición primero.")
            return
        filename = f"datos_predictor_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        datos_np = np.array([np.array(canal_data) for canal_data in self.all_data])
        np.savetxt(filename, np.transpose(datos_np), delimiter=",",
                   header=",".join([f"Canal_{c}" for c in CANALES]), comments="")
        self.all_data = [[] for _ in range(len(CANALES))]
        QtWidgets.QMessageBox.information(self, "Datos Guardados", f"Los datos se han guardado en '{filename}'.")

    def closeEvent(self, event):
        print("[PredictorHisteresis] closeEvent, deteniendo hilos.")
        if hasattr(self, 'adquisicion_thread') and self.adquisicion_thread.is_alive():
            self.adquisicion_thread.stop()
            self.adquisicion_thread.join(timeout=1.0)
        event.accept()

# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    mostrar_info_hardware()
    print(f"\nIniciando Predictor de Histeresis:")
    print(f"  Dispositivo: {DISPOSITIVO}")
    print(f"  Canales: {CANALES}")
    print(f"  Muestreo: {SAMPLE_RATE} Hz")
    print(f"  Ventana de tiempo inicial: {TIME_WINDOW * 1000} ms")

    app = QtWidgets.QApplication([])
    ventana = PredictorHisteresis()
    ventana.show()
    app.exec_()

if __name__ == "__main__":
    main()

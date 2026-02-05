import nidaqmx
from nidaqmx.system import System
import numpy as np
import time
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# Configuración de la adquisición de datos analógicos
DISPOSITIVO = "cDAQ1Mod1"
CANALES = ["ai0", "ai1"]
TASA_MUESTREO = 10000         # Aumentada para mejor resolución temporal
MUESTRAS_POR_BLOQUE = 1000    # Muestras por actualización
TIEMPO_VISUALIZACION = 0.2    # Tiempo visualizado en segundos
VOLTAJE_MIN = -10.0
VOLTAJE_MAX = 10.0

# Cola para comunicación entre hilos
datos_queue = queue.Queue(maxsize=10)

# Clase para la adquisición de datos en un hilo separado
class AdquisicionThread(threading.Thread):
    def __init__(self, dispositivo, canales, tasa_muestreo, muestras_por_bloque):
        threading.Thread.__init__(self)
        self.dispositivo = dispositivo
        self.canales = canales
        self.tasa_muestreo = tasa_muestreo
        self.muestras_por_bloque = muestras_por_bloque
        self.running = True
        self.task = None
        
    def run(self):
        try:
            # Configurar la tarea
            self.task = nidaqmx.Task()
            
            # Añadir canales
            for canal in self.canales:
                nombre_canal = f"{self.dispositivo}/{canal}"
                print(f"Configurando canal: {nombre_canal}")
                self.task.ai_channels.add_ai_voltage_chan(
                    nombre_canal,
                    terminal_config=TerminalConfiguration.RSE,
                    min_val=VOLTAJE_MIN,
                    max_val=VOLTAJE_MAX
                )
            
            # Configurar timing
            self.task.timing.cfg_samp_clk_timing(
                rate=self.tasa_muestreo,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.muestras_por_bloque * 10
            )
            
            # Iniciar la tarea
            print(f"Iniciando adquisición a {self.tasa_muestreo} Hz")
            self.task.start()
            
            # Bucle de adquisición
            while self.running:
                try:
                    # Leer datos
                    datos = self.task.read(
                        number_of_samples_per_channel=self.muestras_por_bloque,
                        timeout=0.1
                    )
                    
                    # Convertir a numpy array
                    datos_np = np.array(datos)
                    
                    # Poner en la cola
                    try:
                        datos_queue.put(datos_np, block=False)
                    except queue.Full:
                        # Si la cola está llena, descartar los datos más antiguos
                        try:
                            datos_queue.get_nowait()
                            datos_queue.put(datos_np, block=False)
                        except:
                            pass
                
                except nidaqmx.errors.DaqError as e:
                    print(f"Error en adquisición: {e}")
                    if "timeout" not in str(e).lower():
                        break
            
            # Detener y cerrar la tarea
            if self.task:
                self.task.stop()
                self.task.close()
                print("Tarea de adquisición detenida y cerrada")
                
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

# Clase para la aplicación de osciloscopio
class OsciloscopioApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Osciloscopio Digital - NI-DAQmx')
        self.resize(1200, 800)
        
        # Crear un widget central
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Crear un layout vertical
        layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(layout)
        
        # Crear widgets de gráficas para cada canal
        self.plot_widgets = []
        self.plot_curves = []
        self.value_labels = []
        
        for i, canal in enumerate(CANALES):
            # Etiqueta para mostrar el valor actual
            value_label = QtWidgets.QLabel(f"Canal {canal}: 0.000 V")
            value_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
            layout.addWidget(value_label)
            self.value_labels.append(value_label)
            
            # Widget de gráfica
            plot_widget = pg.PlotWidget()
            plot_widget.setLabel('left', 'Voltaje', units='V')
            if i == len(CANALES) - 1:  # Último canal
                plot_widget.setLabel('bottom', 'Tiempo', units='s')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setYRange(VOLTAJE_MIN, VOLTAJE_MAX)
            
            # Añadir curva para los datos
            pen = pg.mkPen(color=('r', 'g', 'b', 'c', 'm', 'y')[i % 6], width=2)
            plot_curve = plot_widget.plot(pen=pen)
            
            # Añadir a los layouts y listas
            layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            self.plot_curves.append(plot_curve)
        
        # Crear controles
        control_layout = QtWidgets.QHBoxLayout()
        
        # Botón para iniciar/detener
        self.start_stop_button = QtWidgets.QPushButton("Detener")
        self.start_stop_button.clicked.connect(self.toggle_acquisition)
        control_layout.addWidget(self.start_stop_button)
        
        # Selector de tiempo de visualización
        control_layout.addWidget(QtWidgets.QLabel("Tiempo:"))
        self.time_combo = QtWidgets.QComboBox()
        self.time_combo.addItems(["50 ms", "100 ms", "200 ms", "500 ms", "1 s", "2 s", "5 s"])
        self.time_combo.setCurrentText("200 ms")
        self.time_combo.currentTextChanged.connect(self.update_timebase)
        control_layout.addWidget(self.time_combo)
        
        # Botón para guardar datos
        self.save_button = QtWidgets.QPushButton("Guardar Datos")
        self.save_button.clicked.connect(self.save_data)
        control_layout.addWidget(self.save_button)
        
        layout.addLayout(control_layout)
        
        # Variables para almacenar datos
        self.buffer_size = int(TIEMPO_VISUALIZACION * TASA_MUESTREO)
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        self.tiempo = np.linspace(-TIEMPO_VISUALIZACION, 0, self.buffer_size)
        
        # Crear hilo de adquisición
        self.adquisicion_thread = AdquisicionThread(
            DISPOSITIVO, CANALES, TASA_MUESTREO, MUESTRAS_POR_BLOQUE
        )
        self.adquisicion_thread.start()
        
        # Timer para actualizar la visualización
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(20)  # 50 Hz de actualización visual
        
        # Variable para almacenar todos los datos para guardar
        self.all_data = [[] for _ in range(len(CANALES))]
        
        # Estado de adquisición
        self.acquiring = True
    
    def update_timebase(self):
        # Actualizar la base de tiempo según la selección
        time_text = self.time_combo.currentText()
        if "ms" in time_text:
            time_val = float(time_text.split()[0]) / 1000
        else:
            time_val = float(time_text.split()[0])
        
        self.buffer_size = int(time_val * TASA_MUESTREO)
        self.tiempo = np.linspace(-time_val, 0, self.buffer_size)
        self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
        
        # Actualizar los ejes de tiempo
        for widget in self.plot_widgets:
            widget.setXRange(-time_val, 0)
    
    def update_plots(self):
        # Obtener nuevos datos de la cola
        try:
            while not datos_queue.empty():
                nuevos_datos = datos_queue.get_nowait()
                
                # Almacenar para guardar después
                for i, canal_datos in enumerate(nuevos_datos):
                    self.all_data[i].extend(canal_datos)
                
                # Actualizar el buffer circular
                for i, canal_datos in enumerate(nuevos_datos):
                    # Número de nuevas muestras
                    num_nuevas = len(canal_datos)
                    
                    # Desplazar datos antiguos
                    if num_nuevas >= self.buffer_size:
                        # Si hay más datos nuevos que el buffer, solo usar los más recientes
                        self.datos_buffer[i][:] = canal_datos[-self.buffer_size:]
                    else:
                        # Desplazar datos antiguos y añadir nuevos
                        self.datos_buffer[i][:-num_nuevas] = self.datos_buffer[i][num_nuevas:]
                        self.datos_buffer[i][-num_nuevas:] = canal_datos
                
                # Actualizar etiquetas de valores actuales
                for i, canal in enumerate(CANALES):
                    valor_actual = self.datos_buffer[i][-1]
                    self.value_labels[i].setText(f"Canal {canal}: {valor_actual:.3f} V")
        except:
            pass
        
        # Actualizar los gráficos
        for i, curve in enumerate(self.plot_curves):
            curve.setData(self.tiempo, self.datos_buffer[i])
    
    def toggle_acquisition(self):
        if self.acquiring:
            # Detener adquisición
            self.acquiring = False
            self.start_stop_button.setText("Iniciar")
            self.adquisicion_thread.stop()
        else:
            # Reiniciar adquisición
            self.acquiring = True
            self.start_stop_button.setText("Detener")
            
            # Reiniciar buffers
            self.datos_buffer = [np.zeros(self.buffer_size) for _ in range(len(CANALES))]
            
            # Crear nuevo hilo
            self.adquisicion_thread = AdquisicionThread(
                DISPOSITIVO, CANALES, TASA_MUESTREO, MUESTRAS_POR_BLOQUE
            )
            self.adquisicion_thread.start()
    
    def save_data(self):
        try:
            # Convertir a numpy array
            datos_np = np.array([np.array(canal_data) for canal_data in self.all_data])
            
            # Guardar en archivo
            np.savetxt(
                f"datos_osciloscopio_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                np.transpose(datos_np),
                delimiter=",",
                header=",".join([f"Canal_{canal}" for canal in CANALES])
            )
            
            # Mostrar mensaje
            QtWidgets.QMessageBox.information(
                self, "Datos Guardados", 
                "Los datos han sido guardados correctamente en un archivo CSV."
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error al Guardar", 
                f"Ocurrió un error al guardar los datos: {e}"
            )
    
    def closeEvent(self, event):
        # Detener adquisición al cerrar
        if self.adquisicion_thread.is_alive():
            self.adquisicion_thread.stop()
            self.adquisicion_thread.join(timeout=1.0)
        event.accept()

# Función principal
def main():
    # Mostrar información del sistema
    system = System.local()
    print("Dispositivos detectados:")
    for device in system.devices:
        print(f"Nombre del dispositivo: {device.name}")
        print(f"Tipo de producto:      {device.product_type}")
        print(f"Número de serie:       {device.serial_num}")
        print("-----------------------")
    
    # Crear aplicación
    app = QtWidgets.QApplication([])
    osciloscopio = OsciloscopioApp()
    osciloscopio.show()
    
    # Ejecutar bucle principal
    app.exec_()

if __name__ == "__main__":
    main()
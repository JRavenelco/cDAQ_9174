import sys
import time
import queue
import threading
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtWidgets import (QSizePolicy, QTabWidget, QGridLayout, QPushButton, 
                           QLabel, QComboBox, QSpinBox, QDoubleSpinBox, 
                           QFileDialog, QGroupBox, QProgressBar, QCheckBox,
                           QRadioButton, QButtonGroup, QMessageBox, QTextEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from collections import deque
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, ExcitationSource

# Definir PSEUDO_DIFF conforme a V2
try:
    PSEUDO_DIFF = TerminalConfiguration.PSEUDODIFFERENTIAL
except AttributeError:
    class PseudoDiffDummy:
        value = 125
    PSEUDO_DIFF = PseudoDiffDummy()
import pandas as pd
import os
import pickle
import scipy.signal as signal
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime

# Intentar importar scikit-learn (opcional para la implementación completa de ML)
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn no está disponible. La clasificación será simulada.")

# Configuración de la adquisición
DISPOSITIVO = "cDAQ1Mod2"  # Módulo 9234
CANALES = ["ai0", "ai1"]  # Dos canales de aceleración
SAMPLE_RATE = 51200  # Frecuencia de muestreo máxima para NI 9234
MUESTRAS_POR_BLOQUE = 1024
TIEMPO_VENTANA = 2.56  # Segundos de datos a mostrar (común en HAR)

# Límites del acelerómetro en g
ACCEL_MIN_G = -50.0
ACCEL_MAX_G = 50.0

# Factor de conversión para el acelerómetro 352c33 (100 mV/g)
ACCEL_CONVERSION = 10.0  # g/V (1V / 100mV/g = 10g)

# Fenómenos de maquinado a clasificar
MACHINING_PHENOMENA = ["Corte Normal", "Chatter", "Desgaste", "Fricción Excesiva", "Impacto"]

# Mantenemos la variable ACTIVITIES por compatibilidad, pero apuntando a los fenómenos de maquinado
ACTIVITIES = MACHINING_PHENOMENA

class AdquisicionAceleracionThread(threading.Thread):
    """Hilo para la adquisición de datos del acelerómetro."""
    def __init__(self, dispositivo, canales, sample_rate, muestras_por_bloque, modo_iepe=True, sensibilidad=100.0):
        super().__init__()
        self.dispositivo = dispositivo
        self.canales = canales
        self.sample_rate = sample_rate
        self.muestras_por_bloque = muestras_por_bloque
        self.modo_iepe = modo_iepe
        self.sensibilidad = sensibilidad  # en mV/g
        self.running = True
        self.task = None
        self.data_queue = queue.Queue(maxsize=10)

    def run(self):
        try:
            with nidaqmx.Task() as self.task:
                # Configurar canales según el modo seleccionado
                for canal in self.canales:
                    if self.modo_iepe:
                        # Configuración para acelerómetros IEPE
                        self.task.ai_channels.add_ai_accel_chan(
                            f"{self.dispositivo}/{canal}",
                            sensitivity=self.sensibilidad,
                            sensitivity_units=nidaqmx.constants.AccelSensitivityUnits.MILLIVOLTS_PER_G,
                            current_excit_source=ExcitationSource.INTERNAL,
                            current_excit_val=0.002,  # 2mA para NI 9234
                            min_val=ACCEL_MIN_G,
                            max_val=ACCEL_MAX_G,
                            terminal_config=PSEUDO_DIFF
                        )
                    else:
                        # Configuración para entrada de voltaje estándar
                        self.task.ai_channels.add_ai_voltage_chan(
                            f"{self.dispositivo}/{canal}",
                            terminal_config=PSEUDO_DIFF,
                            min_val=-5.0,
                            max_val=5.0,
                            units=nidaqmx.constants.VoltageUnits.VOLTS
                        )
                
                # Configurar temporización
                self.task.timing.cfg_samp_clk_timing(
                    rate=self.sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=self.muestras_por_bloque
                )
                
                # Iniciar la tarea
                self.task.start()
                
                # Bucle de adquisición
                while self.running:
                    try:
                        # Leer datos
                        data = self.task.read(
                            number_of_samples_per_channel=self.muestras_por_bloque,
                            timeout=2.0
                        )
                        
                        # Convertir a array de numpy
                        data = np.array(data)
                        
                        # Si no está en modo IEPE, convertir a g usando el factor de conversión
                        if not self.modo_iepe:
                            data = data * (self.sensibilidad / 1000.0)  # mV/g a g/V
                            
                        # Poner en la cola si hay espacio
                        try:
                            self.data_queue.put_nowait(data)
                        except queue.Full:
                            pass  # Descartar datos si la cola está llena
                            
                    except Exception as e:
                        print(f"Error en adquisición: {e}")
                        time.sleep(0.1)
                        
        except Exception as e:
            print(f"Error en el hilo de adquisición: {e}")

    def stop(self):
        self.running = False
        # No manual stop here; the run() context manager handles closing the task

class FeatureExtractor:
    """Clase para extraer características de señales de aceleración para clasificación de fenómenos de maquinado."""
    
    @staticmethod
    def extract_time_domain_features(window):
        """Extrae características del dominio del tiempo."""
        features = []
        
        # Calcula características para cada canal
        for channel_data in window:
            # Características estadísticas básicas
            features.append(np.mean(channel_data))       # Media
            features.append(np.std(channel_data))        # Desviación estándar
            features.append(np.var(channel_data))        # Varianza
            features.append(np.max(channel_data))        # Máximo
            features.append(np.min(channel_data))        # Mínimo
            features.append(np.median(channel_data))     # Mediana
            features.append(np.percentile(channel_data, 25))  # Primer cuartil
            features.append(np.percentile(channel_data, 75))  # Tercer cuartil
            
            # Características adicionales
            features.append(np.sum(np.abs(channel_data))) # Suma absoluta
            features.append(np.sqrt(np.mean(channel_data**2)))  # RMS
            
            # Cruces por cero (útil para análisis de movimiento)
            zero_crossings = np.sum(np.diff(np.signbit(channel_data)))
            features.append(zero_crossings)
        
        return features
    
    @staticmethod
    def extract_frequency_domain_features(window, sample_rate):
        """Extrae características del dominio de la frecuencia."""
        features = []
        
        for channel_data in window:
            # Aplicar ventana Hanning para reducir fugas espectrales
            windowed_data = channel_data * np.hanning(len(channel_data))
            
            # Calcular FFT
            fft_values = np.abs(rfft(windowed_data))
            freqs = rfftfreq(len(windowed_data), 1.0/sample_rate)
            
            # Evitar división por cero
            if len(fft_values) == 0:
                features.extend([0, 0, 0, 0, 0])
                continue
                
            # Características espectrales
            features.append(np.max(fft_values))  # Amplitud máxima
            dominant_freq_idx = np.argmax(fft_values)
            features.append(freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0)  # Frecuencia dominante
            
            # Energía espectral
            energy = np.sum(fft_values**2)
            features.append(energy)
            
            # Centroide espectral (frecuencia promedio ponderada por amplitud)
            if np.sum(fft_values) > 0:
                spectral_centroid = np.sum(freqs * fft_values) / np.sum(fft_values)
                features.append(spectral_centroid)
            else:
                features.append(0)
            
            # Bandas de frecuencia - dividir en 4 bandas y sumar la energía en cada una
            if len(freqs) > 4:
                band_size = len(freqs) // 4
                for i in range(4):
                    start = i * band_size
                    end = (i+1) * band_size if i < 3 else len(freqs)
                    band_energy = np.sum(fft_values[start:end]**2)
                    features.append(band_energy)
            else:
                features.extend([0, 0, 0, 0])  # Si no hay suficientes puntos, añadir ceros
        
        return features
    
    @staticmethod
    def extract_all_features(windows, sample_rate):
        """Extrae todas las características disponibles de una ventana de datos."""
        all_features = []
        
        for window in windows:
            # Características del dominio del tiempo
            time_features = FeatureExtractor.extract_time_domain_features(window)
            
            # Características del dominio de la frecuencia
            freq_features = FeatureExtractor.extract_frequency_domain_features(window, sample_rate)
            
            # Combinar todas las características
            features = time_features + freq_features
            all_features.append(features)
        
        return all_features
    
    @staticmethod
    def segment_signal(data, window_size, overlap=0.5):
        """Segmenta la señal en ventanas con superposición."""
        windows = []
        step = int(window_size * (1 - overlap))
        
        # Asegurar que hay suficientes datos
        if len(data[0]) < window_size:
            return windows
        
        # Crear ventanas
        for i in range(0, len(data[0]) - window_size + 1, step):
            window = [channel[i:i+window_size] for channel in data]
            windows.append(window)
        
        return windows


class MachiningClassifier:
    """Clase para clasificar fenómenos de maquinado basándose en características extraídas."""
    
    def __init__(self, model_type='svm'):
        self.model = None
        self.scaler = None
        self.model_type = model_type
        self.is_trained = False
        
        # Comprobar si scikit-learn está disponible
        if not SKLEARN_AVAILABLE and model_type != 'dummy':
            print("scikit-learn no está disponible, usando clasificador simulado.")
            self.model_type = 'dummy'
    
    def train(self, X, y):
        """Entrena el clasificador con datos y etiquetas."""
        if self.model_type == 'dummy':
            # Simulación de entrenamiento
            self.is_trained = True
            accuracy = 0.85  # Valor simulado
            conf_matrix = np.eye(len(np.unique(y))) * 0.85  # Matriz de confusión simulada
            return {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'report': "Reporte simulado: Precisión aproximada de 85%"
            }
        
        # Implementación real con scikit-learn
        try:
            # Normalizar los datos
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42)
            
            # Crear el modelo según el tipo
            if self.model_type == 'svm':
                self.model = SVC(kernel='rbf', probability=True)
            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier(n_estimators=100)
            elif self.model_type == 'mlp':
                self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
            
            # Entrenar el modelo
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluar el modelo
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': conf_matrix,
                'report': report
            }
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")
            self.is_trained = False
            return None
    
    def predict(self, X):
        """Predice la clase para nuevas características."""
        if not self.is_trained:
            return None, None
        
        if self.model_type == 'dummy':
            # Simulación de predicción
            activity_idx = np.random.choice(len(ACTIVITIES), p=[0.7, 0.075, 0.075, 0.075, 0.075])
            activity = ACTIVITIES[activity_idx]
            probabilities = np.zeros(len(ACTIVITIES))
            probabilities[activity_idx] = 0.85
            remaining_prob = 0.15 / (len(ACTIVITIES) - 1)
            for i in range(len(ACTIVITIES)):
                if i != activity_idx:
                    probabilities[i] = remaining_prob
            return activity, probabilities
        
        try:
            # Normalizar los datos
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            
            # Realizar predicción
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            return prediction, probabilities
            
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            return None, None
    
    def save(self, filename):
        """Guarda el modelo entrenado en un archivo."""
        if not self.is_trained:
            return False
        
        try:
            model_data = {
                'model_type': self.model_type,
                'is_trained': self.is_trained
            }
            
            if self.model_type != 'dummy':
                model_data['model'] = self.model
                model_data['scaler'] = self.scaler
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        except Exception as e:
            print(f"Error al guardar el modelo: {e}")
            return False
    
    def load(self, filename):
        """Carga un modelo entrenado desde un archivo."""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']
            
            if self.model_type != 'dummy' and SKLEARN_AVAILABLE:
                self.model = model_data['model']
                self.scaler = model_data['scaler']
            else:
                # Si scikit-learn no está disponible, usar modo simulado
                self.model_type = 'dummy'
            
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False


class MachiningClassifierApp(QtWidgets.QMainWindow):
    """Aplicación para la clasificación de fenómenos de maquinado mediante sensores."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clasificación de Fenómenos de Maquinado mediante Sensores")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.showMaximized()  # Iniciar maximizada
        
        # Crear directorios para guardar datos y modelos
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_data")
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_models")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        # Directorio para guardar datos de experimentos
        self.experiments_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experimentos")
        os.makedirs(self.experiments_dir, exist_ok=True)
        # Flag y buffer para guardado de experimento
        self.experiment_running = False
        self.experiment_data = None
        
        # Configuración de la ventana principal
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # Configuración de parámetros
        self.sample_rate = SAMPLE_RATE
        self.muestras_por_bloque = MUESTRAS_POR_BLOQUE
        self.tiempo_ventana = TIEMPO_VENTANA
        self.buffer_size = max(int(self.tiempo_ventana * self.sample_rate), 2)
        
        # Parámetros para la extracción de características
        self.window_size = 128  # 2.56 segundos a 50 Hz
        self.window_overlap = 0.5  # 50% de superposición
        
        # Buffers de datos
        self.datos_buffer = [deque(maxlen=self.buffer_size) for _ in range(len(CANALES))]
        self.tiempo = np.linspace(-self.tiempo_ventana, 0, self.buffer_size)
        
        # Datos para entrenamiento
        self.all_data = {}  # {actividad: [características]}
        self.current_activity = None
        self.recording = False
        self.recording_start_time = None
        self.recording_duration = 10  # segundos
        
        # Inicializar componentes
        self.feature_extractor = FeatureExtractor()
        self.classifier = MachiningClassifier()
        self.adquisicion_thread = None
        
        # Estado general
        self.adquiriendo = False
        self.classification_running = False
        
        # Configurar pestañas
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # Pestaña de adquisición
        self.tab_adquisicion = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_adquisicion, "Adquisición de Datos")
        self._setup_adquisicion_tab()
        
        # Pestaña de entrenamiento
        self.tab_entrenamiento = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_entrenamiento, "Entrenamiento de Modelo")
        self._setup_entrenamiento_tab()
        
        # Pestaña de clasificación en tiempo real
        self.tab_clasificacion = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_clasificacion, "Clasificación en Tiempo Real")
        self._setup_clasificacion_tab()
        
        # Barra de estado
        self.statusBar().showMessage("Listo. Inicie la adquisición para comenzar.")
        
        # Timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.actualizar_graficos)
        self.update_timer.start(50)  # Actualizar cada 50ms
        
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.actualizar_grabacion)
        
        self.classification_timer = QTimer()
        self.classification_timer.timeout.connect(self.realizar_clasificacion)
    
    def _setup_adquisicion_tab(self):
        """Configura la pestaña de adquisición de datos."""
        layout = QtWidgets.QVBoxLayout(self.tab_adquisicion)
        
        # Panel superior con controles
        control_panel = QtWidgets.QHBoxLayout()
        
        # Botón de inicio/parada
        self.btn_iniciar = QPushButton("Iniciar Adquisición")
        self.btn_iniciar.clicked.connect(self.toggle_adquisicion)
        self.btn_iniciar.setMinimumWidth(150)
        control_panel.addWidget(self.btn_iniciar)
        
        # Frecuencia de muestreo
        control_panel.addWidget(QLabel("Frecuencia (Hz):"))
        self.combo_frecuencia = QComboBox()
        self.combo_frecuencia.addItems(["50", "100", "200", "500", "1000", "2000", "5000", "10000", "20000", "51200"])
        self.combo_frecuencia.setCurrentText(str(self.sample_rate))
        self.combo_frecuencia.currentTextChanged.connect(self.cambiar_frecuencia)
        control_panel.addWidget(self.combo_frecuencia)
        
        # Configuración IEPE
        self.chk_iepe = QCheckBox("Modo IEPE")
        self.chk_iepe.setChecked(True)
        control_panel.addWidget(self.chk_iepe)
        
        # Sensibilidad
        control_panel.addWidget(QLabel("Sensibilidad (mV/g):"))
        self.spin_sensibilidad = QDoubleSpinBox()
        self.spin_sensibilidad.setRange(1, 1000)
        self.spin_sensibilidad.setValue(100)  # 100 mV/g para PCB 352c33
        self.spin_sensibilidad.setSingleStep(1)
        control_panel.addWidget(self.spin_sensibilidad)
        
        control_panel.addStretch(1)
        layout.addLayout(control_panel)
        
        # Panel de gráficos
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, 1)  # Darle más espacio al gráfico
        
        # Gráfico de aceleración
        self.plot_aceleracion = self.plot_widget.addPlot(title="Señales de Aceleración")
        self.plot_aceleracion.setLabel('left', 'Aceleración', 'g')
        self.plot_aceleracion.setLabel('bottom', 'Tiempo', 's')
        self.plot_aceleracion.addLegend()
        
        # Curvas para cada canal
        self.curvas_aceleracion = []
        colores = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, canal in enumerate(CANALES):
            curva = self.plot_aceleracion.plot(
                self.tiempo,
                [0] * len(self.tiempo),
                pen=colores[i % len(colores)],
                name=f"Canal {i+1}"
            )
            self.curvas_aceleracion.append(curva)
        
        # Gráfico de FFT
        self.plot_widget.nextRow()
        self.plot_fft = self.plot_widget.addPlot(title="Espectro de Frecuencia")
        self.plot_fft.setLabel('left', 'Amplitud', 'g')
        self.plot_fft.setLabel('bottom', 'Frecuencia', 'Hz')
        self.plot_fft.setLogMode(x=True, y=False)
        
        # Curvas para FFT
        self.curvas_fft = []
        for i, canal in enumerate(CANALES):
            curva = self.plot_fft.plot(
                [1], [0],  # Datos iniciales dummy
                pen=colores[i % len(colores)],
                name=f"FFT Canal {i+1}"
            )
            self.curvas_fft.append(curva)
        
        # Panel de grabación de actividades
        grabacion_panel = QtWidgets.QGroupBox("Grabación de Actividades")
        grabacion_layout = QtWidgets.QHBoxLayout(grabacion_panel)
        
        # Selector de actividad
        grabacion_layout.addWidget(QLabel("Actividad:"))
        self.combo_actividad = QComboBox()
        self.combo_actividad.addItems(ACTIVITIES)
        grabacion_layout.addWidget(self.combo_actividad)
        
        # Tiempo de grabación
        grabacion_layout.addWidget(QLabel("Duración (s):"))
        self.spin_duracion = QSpinBox()
        self.spin_duracion.setRange(5, 60)
        self.spin_duracion.setValue(10)
        grabacion_layout.addWidget(self.spin_duracion)
        
        # Botón de grabación
        self.btn_grabar = QPushButton("Iniciar Grabación")
        self.btn_grabar.clicked.connect(self.toggle_grabacion)
        grabacion_layout.addWidget(self.btn_grabar)
        
        # Barra de progreso
        self.progress_grabacion = QProgressBar()
        self.progress_grabacion.setRange(0, 100)
        grabacion_layout.addWidget(self.progress_grabacion, 1)  # Dar más espacio
        
        layout.addWidget(grabacion_panel)
        
        # Panel inferior con botones
        botones_panel = QtWidgets.QHBoxLayout()
        
        # Botón para guardar datos
        self.btn_guardar = QPushButton("Guardar Datos")
        self.btn_guardar.clicked.connect(self.guardar_datos)
        botones_panel.addWidget(self.btn_guardar)
        
        # Botón para cargar datos
        self.btn_cargar = QPushButton("Cargar Datos")
        self.btn_cargar.clicked.connect(self.cargar_datos)
        botones_panel.addWidget(self.btn_cargar)
        
        # Botón para limpiar datos
        self.btn_limpiar = QPushButton("Limpiar Datos")
        self.btn_limpiar.clicked.connect(self.limpiar_datos)
        botones_panel.addWidget(self.btn_limpiar)
        # Botón para guardado de experimento
        self.btn_experimento = QPushButton("Guardar Experimento")
        self.btn_experimento.clicked.connect(self.toggle_experimento)
        botones_panel.addWidget(self.btn_experimento)
        
        layout.addLayout(botones_panel)
    
    def _setup_entrenamiento_tab(self):
        """Configura la pestaña de entrenamiento del modelo."""
        layout = QtWidgets.QVBoxLayout(self.tab_entrenamiento)
        
        # Panel de resumen de datos
        datos_group = QGroupBox("Datos de Entrenamiento")
        datos_layout = QtWidgets.QVBoxLayout(datos_group)
        
        self.lbl_datos_info = QLabel("No hay datos cargados")
        datos_layout.addWidget(self.lbl_datos_info)
        
        # Tabla de datos
        self.tabla_datos = QtWidgets.QTableWidget()
        self.tabla_datos.setColumnCount(2)
        self.tabla_datos.setHorizontalHeaderLabels(["Actividad", "Muestras"])
        self.tabla_datos.horizontalHeader().setStretchLastSection(True)
        datos_layout.addWidget(self.tabla_datos)
        
        layout.addWidget(datos_group)
        
        # Panel de configuración del modelo
        modelo_group = QGroupBox("Configuración del Modelo")
        modelo_layout = QtWidgets.QGridLayout(modelo_group)
        
        # Tipo de modelo
        modelo_layout.addWidget(QLabel("Tipo de Modelo:"), 0, 0)
        self.combo_modelo = QComboBox()
        self.combo_modelo.addItems(["SVM", "Random Forest", "MLP", "Dummy (Simulado)"])
        modelo_layout.addWidget(self.combo_modelo, 0, 1)
        
        # Parámetros de entrenamiento
        modelo_layout.addWidget(QLabel("Test Split (%):"), 1, 0)
        self.spin_test_split = QSpinBox()
        self.spin_test_split.setRange(10, 50)
        self.spin_test_split.setValue(30)
        modelo_layout.addWidget(self.spin_test_split, 1, 1)
        
        # Botón para entrenar
        self.btn_entrenar = QPushButton("Entrenar Modelo")
        self.btn_entrenar.clicked.connect(self.entrenar_modelo)
        modelo_layout.addWidget(self.btn_entrenar, 2, 0, 1, 2)
        
        layout.addWidget(modelo_group)
        
        # Panel de resultados
        resultados_group = QGroupBox("Resultados del Entrenamiento")
        resultados_layout = QtWidgets.QVBoxLayout(resultados_group)
        
        # Área de texto para mostrar resultados
        self.txt_resultados = QTextEdit()
        self.txt_resultados.setReadOnly(True)
        resultados_layout.addWidget(self.txt_resultados)
        
        # Botón para guardar modelo
        self.btn_guardar_modelo = QPushButton("Guardar Modelo Entrenado")
        self.btn_guardar_modelo.clicked.connect(self.guardar_modelo)
        resultados_layout.addWidget(self.btn_guardar_modelo)
        
        layout.addWidget(resultados_group)
    
    def _setup_clasificacion_tab(self):
        """Configura la pestaña de clasificación en tiempo real."""
        layout = QtWidgets.QVBoxLayout(self.tab_clasificacion)
        
        # Panel de control
        control_panel = QtWidgets.QHBoxLayout()
        
        # Botón para cargar modelo
        self.btn_cargar_modelo = QPushButton("Cargar Modelo")
        self.btn_cargar_modelo.clicked.connect(self.cargar_modelo)
        control_panel.addWidget(self.btn_cargar_modelo)
        
        # Etiqueta de modelo cargado
        self.lbl_modelo_cargado = QLabel("No hay modelo cargado")
        control_panel.addWidget(self.lbl_modelo_cargado)
        
        # Botón para iniciar/detener clasificación
        self.btn_clasificar = QPushButton("Iniciar Clasificación")
        self.btn_clasificar.clicked.connect(self.toggle_clasificacion)
        self.btn_clasificar.setEnabled(False)  # Deshabilitado hasta que se cargue un modelo
        control_panel.addWidget(self.btn_clasificar)
        
        layout.addLayout(control_panel)
        
        # Panel central con gráfico de aceleración y clasificación
        central_panel = QtWidgets.QHBoxLayout()
        
        # Gráfico de aceleración (70% del espacio)
        clasificacion_plot_widget = pg.GraphicsLayoutWidget()
        self.plot_clasificacion = clasificacion_plot_widget.addPlot(title="Señal de Aceleración en Tiempo Real")
        self.plot_clasificacion.setLabel('left', 'Aceleración', 'g')
        self.plot_clasificacion.setLabel('bottom', 'Tiempo', 's')
        self.plot_clasificacion.addLegend()
        
        # Curvas para cada canal
        self.curvas_clasificacion = []
        colores = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, canal in enumerate(CANALES):
            curva = self.plot_clasificacion.plot(
                self.tiempo,
                [0] * len(self.tiempo),
                pen=colores[i % len(colores)],
                name=f"Canal {i+1}"
            )
            self.curvas_clasificacion.append(curva)
        
        central_panel.addWidget(clasificacion_plot_widget, 7)  # 70% del espacio
        
        # Panel de resultados de clasificación (30% del espacio)
        resultados_widget = QtWidgets.QWidget()
        resultados_layout = QtWidgets.QVBoxLayout(resultados_widget)
        
        # Etiqueta de actividad detectada
        self.lbl_actividad = QLabel("Actividad: No detectada")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.lbl_actividad.setFont(font)
        self.lbl_actividad.setAlignment(Qt.AlignCenter)
        resultados_layout.addWidget(self.lbl_actividad)
        
        # Barras de probabilidad para cada actividad
        self.barras_prob = {}
        barras_layout = QtWidgets.QVBoxLayout()
        for actividad in ACTIVITIES:
            actividad_layout = QtWidgets.QHBoxLayout()
            actividad_layout.addWidget(QLabel(f"{actividad}:"))
            barra = QProgressBar()
            barra.setRange(0, 100)
            barra.setValue(0)
            actividad_layout.addWidget(barra)
            self.barras_prob[actividad] = barra
            barras_layout.addLayout(actividad_layout)
        
        resultados_layout.addLayout(barras_layout)
        resultados_layout.addStretch(1)
        
        # Área de registro de actividades
        registro_group = QGroupBox("Registro de Actividades")
        registro_layout = QtWidgets.QVBoxLayout(registro_group)
        
        self.txt_registro = QTextEdit()
        self.txt_registro.setReadOnly(True)
        registro_layout.addWidget(self.txt_registro)
        
        resultados_layout.addWidget(registro_group)
        
        central_panel.addWidget(resultados_widget, 3)  # 30% del espacio
        
        layout.addLayout(central_panel, 1)  # Dar más espacio al panel central
    
    # Funciones de adquisición de datos
    def toggle_adquisicion(self):
        """Inicia o detiene la adquisición de datos."""
        if self.adquiriendo:
            # Detener adquisición
            if self.adquisicion_thread is not None:
                self.adquisicion_thread.stop()
                self.adquisicion_thread = None
            self.btn_iniciar.setText("Iniciar Adquisición")
            self.statusBar().showMessage("Adquisición detenida")
            self.adquiriendo = False
            
            # Deshabilitar controles de grabación
            self.btn_grabar.setEnabled(False)
            
            # Detener grabación si está activa
            if self.recording:
                self.toggle_grabacion()
        else:
            # Iniciar adquisición
            modo_iepe = self.chk_iepe.isChecked()
            sensibilidad = self.spin_sensibilidad.value()
            
            self.adquisicion_thread = AdquisicionAceleracionThread(
                DISPOSITIVO,
                CANALES,
                self.sample_rate,
                self.muestras_por_bloque,
                modo_iepe=modo_iepe,
                sensibilidad=sensibilidad
            )
            self.adquisicion_thread.start()
            
            self.btn_iniciar.setText("Detener Adquisición")
            self.statusBar().showMessage("Adquisición iniciada")
            self.adquiriendo = True
            
            # Habilitar controles de grabación
            self.btn_grabar.setEnabled(True)
    
    def cambiar_frecuencia(self, frecuencia):
        """Cambia la frecuencia de muestreo."""
        try:
            nueva_frecuencia = int(frecuencia)
            if nueva_frecuencia != self.sample_rate:
                self.sample_rate = nueva_frecuencia
                self.buffer_size = max(int(self.tiempo_ventana * self.sample_rate), 2)
                self.tiempo = np.linspace(-self.tiempo_ventana, 0, self.buffer_size)
                self.datos_buffer = [deque(maxlen=self.buffer_size) for _ in range(len(CANALES))]
                
                # Reiniciar adquisición si estaba activa
                if self.adquiriendo:
                    self.toggle_adquisicion()
                    self.toggle_adquisicion()
                
                self.statusBar().showMessage(f"Frecuencia cambiada a {nueva_frecuencia} Hz")
        except ValueError:
            pass
    
    def actualizar_graficos(self):
        """Actualiza los gráficos con los nuevos datos adquiridos."""
        if not self.adquiriendo or self.adquisicion_thread is None:
            return
        
        # Procesar datos de la cola
        datos_procesados = 0
        while datos_procesados < 5:  # Procesar hasta 5 paquetes por vez
            try:
                datos = self.adquisicion_thread.data_queue.get_nowait()
                
                # Extender los buffers con los nuevos datos
                for i, canal_data in enumerate(datos):
                    if i < len(self.datos_buffer):
                        self.datos_buffer[i].extend(canal_data)
                        # Capturar datos de experimento
                        if self.experiment_running and self.experiment_data is not None:
                            if i < len(self.experiment_data):
                                self.experiment_data[i].extend(canal_data.tolist())
                
                datos_procesados += 1
            except queue.Empty:
                break
        
        # Actualizar gráficos si hay datos
        if datos_procesados > 0:
            # Actualizar gráfico de aceleración
            for i, curva in enumerate(self.curvas_aceleracion):
                if i < len(self.datos_buffer):
                    datos_canal = list(self.datos_buffer[i])[-len(self.tiempo):]
                    if len(datos_canal) < len(self.tiempo):
                        # Rellenar con ceros si no hay suficientes datos
                        datos_canal = [0] * (len(self.tiempo) - len(datos_canal)) + datos_canal
                    curva.setData(self.tiempo, datos_canal)
            
            # Actualizar gráfico de FFT
            self.actualizar_fft()
            
            # Actualizar también los gráficos de clasificación si está activa
            if self.classification_running:
                for i, curva in enumerate(self.curvas_clasificacion):
                    if i < len(self.datos_buffer):
                        datos_canal = list(self.datos_buffer[i])[-len(self.tiempo):]
                        if len(datos_canal) < len(self.tiempo):
                            datos_canal = [0] * (len(self.tiempo) - len(datos_canal)) + datos_canal
                        curva.setData(self.tiempo, datos_canal)
    
    def actualizar_fft(self):
        """Actualiza el gráfico de FFT."""
        for i, curva in enumerate(self.curvas_fft):
            if i < len(self.datos_buffer) and len(self.datos_buffer[i]) > 10:
                # Obtener los últimos N datos para FFT
                n_datos = min(1024, len(self.datos_buffer[i]))
                datos = list(self.datos_buffer[i])[-n_datos:]
                
                # Aplicar ventana Hanning
                datos_ventana = np.array(datos) * np.hanning(len(datos))
                
                # Calcular FFT
                fft = np.abs(rfft(datos_ventana))
                frecuencias = rfftfreq(len(datos), 1.0/self.sample_rate)
                
                # Filtrar frecuencias muy bajas para mejor visualización
                mask = frecuencias > 0.5
                if np.any(mask):
                    curva.setData(frecuencias[mask], fft[mask])
    
    # Funciones de grabación de actividades
    def toggle_experimento(self):
        """Inicia o detiene el guardado de datos del experimento."""
        if not self.adquiriendo:
            QMessageBox.warning(self, "Advertencia", "La adquisición no está activa.")
            return
        if self.experiment_running:
            # Detener guardado de experimento
            self.experiment_running = False
            self.btn_experimento.setText("Guardar Experimento")
            # Guardar datos del experimento
            filename = datetime.datetime.now().strftime("experimento_%Y%m%d_%H%M%S.pkl")
            filepath = os.path.join(self.experiments_dir, filename)
            try:
                with open(filepath, "wb") as f:
                    pickle.dump(self.experiment_data, f)
                self.statusBar().showMessage(f"Experimento guardado en {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar el experimento:\n{e}")
        else:
            # Iniciar guardado de experimento
            self.experiment_data = [[] for _ in range(len(CANALES))]
            self.experiment_running = True
            self.btn_experimento.setText("Detener Experimento")
            self.statusBar().showMessage("Guardando experimento...")
    def toggle_grabacion(self):
        """Inicia o detiene la grabación de datos para entrenamiento."""
        if not self.adquiriendo:
            return
        
        if self.recording:
            # Detener grabación
            self.recording = False
            self.recording_timer.stop()
            self.btn_grabar.setText("Iniciar Grabación")
            self.progress_grabacion.setValue(0)
            self.current_activity = None
            self.statusBar().showMessage("Grabación detenida")
        else:
            # Iniciar grabación
            self.current_activity = self.combo_actividad.currentText()
            self.recording_duration = self.spin_duracion.value()
            self.recording_start_time = time.time()
            
            # Inicializar la entrada en all_data si no existe
            if self.current_activity not in self.all_data:
                self.all_data[self.current_activity] = []
            
            self.recording = True
            self.btn_grabar.setText("Detener Grabación")
            self.statusBar().showMessage(f"Grabando actividad: {self.current_activity}")
            
            # Iniciar timer para actualizar la barra de progreso
            self.recording_timer.start(100)  # Actualizar cada 100ms
    
    def actualizar_grabacion(self):
        """Actualiza el estado de la grabación y procesa los datos."""
        if not self.recording:
            return
        
        tiempo_transcurrido = time.time() - self.recording_start_time
        progreso = int((tiempo_transcurrido / self.recording_duration) * 100)
        
        if progreso >= 100:
            # Finalizar grabación automáticamente
            self.progress_grabacion.setValue(100)
            self.toggle_grabacion()
            self.actualizar_tabla_datos()
            return
        
        self.progress_grabacion.setValue(progreso)
        
        # Procesar datos para segmentar y extraer características
        if len(self.datos_buffer[0]) >= self.window_size:  # Al menos una ventana completa
            datos_actuales = [list(buffer)[-self.window_size:] for buffer in self.datos_buffer]
            windows = [datos_actuales]  # Una sola ventana con los datos más recientes
            
            # Extraer características
            features = FeatureExtractor.extract_all_features(windows, self.sample_rate)
            
            # Guardar características con la etiqueta de actividad
            if features and len(features) > 0:
                self.all_data[self.current_activity].append(features[0])
    
    def actualizar_tabla_datos(self):
        """Actualiza la tabla de datos de entrenamiento."""
        self.tabla_datos.setRowCount(0)  # Limpiar tabla
        
        total_samples = 0
        row = 0
        
        for actividad, features_list in self.all_data.items():
            num_samples = len(features_list)
            if num_samples > 0:
                self.tabla_datos.insertRow(row)
                self.tabla_datos.setItem(row, 0, QtWidgets.QTableWidgetItem(actividad))
                self.tabla_datos.setItem(row, 1, QtWidgets.QTableWidgetItem(str(num_samples)))
                row += 1
                total_samples += num_samples
        
        # Actualizar etiqueta de resumen
        if total_samples > 0:
            self.lbl_datos_info = f"Total: {total_samples} muestras en {len(self.all_data)} actividades"
        else:
            self.lbl_datos_info.setText("No hay datos cargados")
    
    def guardar_datos(self):
        """Guarda los datos recopilados en un archivo."""
        if not self.all_data or len(self.all_data) == 0:
            QMessageBox.warning(self, "Sin datos", "No hay datos para guardar.")
            return
        
        # Generar nombre de archivo con fecha y hora
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"har_data_{timestamp}.pkl")
        
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'data': self.all_data,
                    'sample_rate': self.sample_rate,
                    'window_size': self.window_size,
                    'activities': list(self.all_data.keys())
                }, f)
            
            QMessageBox.information(self, "Éxito", f"Datos guardados en:\n{filename}")
            self.statusBar().showMessage(f"Datos guardados en: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al guardar datos: {e}")
    
    def cargar_datos(self):
        """Carga datos desde un archivo."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Cargar Datos", self.data_dir, "Pickle Files (*.pkl)"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'rb') as f:
                data_dict = pickle.load(f)
            
            self.all_data = data_dict['data']
            self.window_size = data_dict.get('window_size', 128)
            
            self.actualizar_tabla_datos()
            QMessageBox.information(self, "Éxito", f"Datos cargados desde:\n{filename}")
            self.statusBar().showMessage(f"Datos cargados desde: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar datos: {e}")
    
    def limpiar_datos(self):
        """Limpia todos los datos recopilados."""
        if not self.all_data:
            return
        
        reply = QMessageBox.question(
            self, "Confirmar", "¿Está seguro de que desea eliminar todos los datos recopilados?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.all_data = {}
            self.actualizar_tabla_datos()
            self.statusBar().showMessage("Datos eliminados")
    
    # Funciones de entrenamiento y clasificación
    def entrenar_modelo(self):
        """Entrena un modelo con los datos recopilados."""
        if not self.all_data or len(self.all_data) == 0:
            QMessageBox.warning(self, "Sin datos", "No hay datos para entrenar el modelo.")
            return
        
        # Preparar datos para entrenamiento
        X = []
        y = []
        for actividad, features_list in self.all_data.items():
            for features in features_list:
                X.append(features)
                y.append(actividad)
        
        X = np.array(X)
        
        # Seleccionar tipo de modelo
        model_type = self.combo_modelo.currentText().lower()
        if "random forest" in model_type:
            model_type = "random_forest"
        elif "mlp" in model_type:
            model_type = "mlp"
        elif "dummy" in model_type:
            model_type = "dummy"
        else:
            model_type = "svm"
        
        # Crear        # Clasificador
        self.machining_classifier = MachiningClassifier(model_type=self.current_model_type)
        
        self.statusBar().showMessage("Entrenando modelo...")
        self.txt_resultados.append(f"Entrenando modelo {model_type} con {len(X)} muestras...")
        QtWidgets.QApplication.processEvents()  # Actualizar interfaz
        
        # Entrenar modelo
        results = self.classifier.train(X, y)
        
        if results:
            # Mostrar resultados
            self.txt_resultados.append(f"\nEntrenamiento completado con éxito.")
            self.txt_resultados.append(f"Precisión: {results['accuracy']:.4f}")
            
            if isinstance(results['report'], str):
                self.txt_resultados.append(f"\nInforme de clasificación:\n{results['report']}")
            
            self.statusBar().showMessage("Modelo entrenado con éxito")
            
            # Habilitar botón de guardar modelo
            self.btn_guardar_modelo.setEnabled(True)
        else:
            self.txt_resultados.append("\nError durante el entrenamiento.")
            self.statusBar().showMessage("Error durante el entrenamiento")
    
    def guardar_modelo(self):
        """Guarda el modelo entrenado en un archivo."""
        if not self.classifier or not self.classifier.is_trained:
            QMessageBox.warning(self, "Sin modelo", "No hay modelo entrenado para guardar.")
            return
        
        # Generar nombre de archivo con fecha y hora
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.models_dir, f"har_model_{timestamp}.pkl")
        
        if self.classifier.save(filename):
            QMessageBox.information(self, "Éxito", f"Modelo guardado en:\n{filename}")
            self.statusBar().showMessage(f"Modelo guardado en: {filename}")
        else:
            QMessageBox.critical(self, "Error", "Error al guardar el modelo")
    
    def cargar_modelo(self):
        """Carga un modelo entrenado desde un archivo."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Cargar Modelo", self.models_dir, "Pickle Files (*.pkl)"
        )
        
        if not filename:
            return
        
        self.classifier = ActivityClassifier()
        if self.classifier.load(filename):
            self.lbl_modelo_cargado.setText(f"Modelo cargado: {os.path.basename(filename)}")
            self.btn_clasificar.setEnabled(True)
            self.statusBar().showMessage(f"Modelo cargado desde: {filename}")
        else:
            QMessageBox.critical(self, "Error", "Error al cargar el modelo")
    
    def toggle_clasificacion(self):
        """Inicia o detiene la clasificación en tiempo real."""
        if not self.classifier or not self.classifier.is_trained:
            QMessageBox.warning(self, "Sin modelo", "No hay modelo entrenado para clasificar.")
            return
        
        if not self.adquiriendo:
            QMessageBox.warning(self, "Sin adquisición", "Inicie la adquisición primero.")
            return
        
        if self.classification_running:
            # Detener clasificación
            self.classification_timer.stop()
            self.btn_clasificar.setText("Iniciar Clasificación")
            self.classification_running = False
            self.statusBar().showMessage("Clasificación detenida")
        else:
            # Iniciar clasificación
            self.classification_timer.start(200)  # Clasificar cada 200ms
            self.btn_clasificar.setText("Detener Clasificación")
            self.classification_running = True
            self.statusBar().showMessage("Clasificación iniciada")
    
    def realizar_clasificacion(self):
        """Realiza la clasificación en tiempo real con los datos actuales."""
        if not self.classification_running or len(self.datos_buffer[0]) < self.window_size:
            return
        
        # Obtener datos actuales para clasificación
        datos_actuales = [list(buffer)[-self.window_size:] for buffer in self.datos_buffer]
        
        # Extraer características
        features = FeatureExtractor.extract_all_features([datos_actuales], self.sample_rate)
        
        if features and len(features) > 0:
            # Realizar predicción
            activity, probabilities = self.classifier.predict(np.array(features[0]))
            
            if activity is not None:
                # Actualizar etiqueta de actividad
                self.lbl_actividad.setText(f"Actividad: {activity}")
                
                # Actualizar barras de probabilidad
                for i, act in enumerate(ACTIVITIES):
                    if i < len(probabilities):
                        prob = int(probabilities[i] * 100)
                        self.barras_prob[act].setValue(prob)
                
                # Registrar actividad si cambió
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                self.txt_registro.append(f"[{timestamp}] {activity} ({probabilities[ACTIVITIES.index(activity)]:.2f})")
                self.txt_registro.moveCursor(QtWidgets.QTextCursor.End)
    
    def closeEvent(self, event):
        """Manejador para el cierre de la ventana."""
        # Detener hilos y timers
        if self.adquisicion_thread is not None:
            self.adquisicion_thread.stop()
            # Dar tiempo para que termine (sin usar wait)
            self.adquisicion_thread.join(timeout=1.0)
        
        self.update_timer.stop()
        self.recording_timer.stop()
        self.classification_timer.stop()
        
        event.accept()
        
        # Gráfico de dominio del tiempo
        self.plot_tiempo = self.graphics_widget.addPlot(title="Dominio del Tiempo")
        self.plot_tiempo.setLabel('left', 'Aceleración', 'g')
        self.plot_tiempo.setLabel('bottom', 'Tiempo', 's')
        self.plot_tiempo.addLegend()
        
        # Curvas para cada canal
        self.curvas_tiempo = []
        colores = ['r', 'g', 'b', 'y']
        for i, canal in enumerate(CANALES):
            curva = self.plot_tiempo.plot(
                self.tiempo, 
                [0]*len(self.tiempo), 
                pen=colores[i], 
                name=f'Canal {i+1}'
            )
            self.curvas_tiempo.append(curva)
        
        # Gráfico de FFT
        self.graphics_widget.nextRow()
        self.plot_fft = self.graphics_widget.addPlot(title="Espectro de Frecuencia")
        self.plot_fft.setLabel('left', 'Amplitud', 'g')
        self.plot_fft.setLabel('bottom', 'Frecuencia', 'Hz')
        self.plot_fft.setLogMode(x=True, y=True)
        
        # Curvas FFT para cada canal
        self.curvas_fft = []
        for i, canal in enumerate(CANALES):
            curva = self.plot_fft.plot(
                [1], [1],  # Datos dummy
                pen=colores[i],
                name=f'Canal {i+1} FFT'
            )
            self.curvas_fft.append(curva)
    
    def _setup_controls(self):
        """Configura los controles de la interfaz."""
        # Contenedor de controles
        control_layout = QtWidgets.QHBoxLayout()
        
        # Botón de inicio/detención
        self.btn_inicio = QtWidgets.QPushButton("Detener")
        self.btn_inicio.clicked.connect(self.toggle_adquisicion)
        control_layout.addWidget(self.btn_inicio)
        
        # Selector de frecuencia de muestreo
        self.combo_sample_rate = QtWidgets.QComboBox()
        self.combo_sample_rate.addItems(["51200", "25600", "12800", "6400", "3200"])
        self.combo_sample_rate.setCurrentText(str(self.sample_rate))
        self.combo_sample_rate.currentTextChanged.connect(self.cambiar_sample_rate)
        control_layout.addWidget(QtWidgets.QLabel("Frecuencia de Muestreo (Hz):"))
        control_layout.addWidget(self.combo_sample_rate)
        
        # Selector de tiempo de ventana
        self.combo_ventana = QtWidgets.QComboBox()
        self.combo_ventana.addItems(["0.5", "1.0", "2.0", "5.0", "10.0"])
        self.combo_ventana.setCurrentText(str(self.tiempo_ventana))
        self.combo_ventana.currentTextChanged.connect(self.cambiar_tiempo_ventana)
        control_layout.addWidget(QtWidgets.QLabel("Ventana de Tiempo (s):"))
        control_layout.addWidget(self.combo_ventana)
        
        # Añadir controles al layout principal
        self.layout.addLayout(control_layout)
    
    def cambiar_sample_rate(self, text):
        """Cambia la frecuencia de muestreo."""
        try:
            nueva_tasa = int(text)
            if nueva_tasa != self.sample_rate:
                self.sample_rate = nueva_tasa
                self.reiniciar_adquisicion()
        except ValueError:
            pass
    
    def cambiar_tiempo_ventana(self, text):
        """Cambia el tiempo de la ventana de visualización."""
        try:
            nuevo_tiempo = float(text)
            if nuevo_tiempo > 0 and nuevo_tiempo != self.tiempo_ventana:
                self.tiempo_ventana = nuevo_tiempo
                self.buffer_size = max(int(self.tiempo_ventana * self.sample_rate), 2)
                self.tiempo = np.linspace(-self.tiempo_ventana, 0, self.buffer_size)
                
                # Reiniciar buffers
                with threading.Lock():
                    self.datos_buffer = [deque(maxlen=self.buffer_size) for _ in range(len(CANALES))]
        except ValueError:
            pass
    
    def toggle_adquisicion(self):
        """Inicia o detiene la adquisición de datos."""
        if self.adquiriendo:
            # Detener adquisición
            if self.adquisicion_thread is not None:
                self.adquisicion_thread.stop()
                self.adquisicion_thread.join(timeout=1.0)
                self.adquisicion_thread = None
            self.btn_iniciar.setText("Iniciar")
        else:
            # Iniciar adquisición
            self.adquisicion_thread = AdquisicionAceleracionThread(
                DISPOSITIVO, 
                CANALES, 
                self.sample_rate, 
                self.muestras_por_bloque
            )
            self.adquisicion_thread.start()
            self.btn_iniciar.setText("Detener")
        
        self.adquiriendo = not self.adquiriendo
    
    def reiniciar_adquisicion(self):
        """Reinicia la adquisición con los nuevos parámetros."""
        if self.adquiriendo:
            estado_anterior = True
            self.toggle_adquisicion()
            time.sleep(0.1)
            self.toggle_adquisicion()
    
    def actualizar_graficos(self):
        """Actualiza los gráficos con los nuevos datos."""
        if self.adquisicion_thread is None or not self.adquiriendo:
            return
        
        # Procesar todos los paquetes de datos disponibles
        datos_procesados = 0
        while True:
            try:
                datos = self.adquisicion_thread.data_queue.get_nowait()
                datos_procesados += 1
                
                # Evitar sobrecargar la interfaz
                if datos_procesados > 5:  # Máximo 5 paquetes por actualización
                    break
                
                # Procesar datos para cada canal
                for i in range(len(CANALES)):
                    if i < len(datos):
                        # Convertir a g y agregar al buffer
                        datos_g = datos[i] * ACCEL_CONVERSION
                        self.datos_buffer[i].extend(datos_g)
                
            except queue.Empty:
                break
        
        # Actualizar gráficos si hay datos
        if any(self.datos_buffer):
            self.actualizar_grafico_tiempo()
            self.actualizar_grafico_fft()
    
    def actualizar_grafico_tiempo(self):
        """Actualiza el gráfico en el dominio del tiempo."""
        # Obtener los datos más recientes
        datos_actuales = []
        for i in range(len(CANALES)):
            datos_canal = list(self.datos_buffer[i])
            if len(datos_canal) < len(self.tiempo):
                # Rellenar con ceros al principio si no hay suficientes datos
                datos_canal = [0] * (len(self.tiempo) - len(datos_canal)) + datos_canal
            else:
                # Tomar solo los últimos N puntos
                datos_canal = datos_canal[-len(self.tiempo):]
            
            datos_actuales.append(datos_canal)
        
        # Actualizar curvas
        for i, curva in enumerate(self.curvas_tiempo):
            if i < len(datos_actuales):
                curva.setData(self.tiempo, datos_actuales[i])
    
    def actualizar_grafico_fft(self):
        """Actualiza el gráfico de FFT."""
        for i, curva in enumerate(self.curvas_fft):
            if i < len(self.datos_buffer) and len(self.datos_buffer[i]) > 10:
                # Calcular FFT
                datos = np.array(self.datos_buffer[i])
                n = len(datos)
                if n < 2:
                    continue
                
                # Aplicar ventana de Hann
                ventana = np.hanning(n)
                datos_ventana = datos * ventana
                
                # Calcular FFT
                fft_result = np.fft.rfft(datos_ventana)
                fft_mag = np.abs(fft_result) / n * 2  # Escalado de amplitud
                fft_mag[0] /= 2  # Corregir el componente DC
                
                # Calcular frecuencias
                frecuencias = np.fft.rfftfreq(n, 1.0/self.sample_rate)
                
                # Filtrar frecuencias por encima de 0.1 Hz para evitar ruido de DC
                mask = frecuencias > 0.1
                frecuencias = frecuencias[mask]
                fft_mag = fft_mag[mask]
                
                # Actualizar curva
                curva.setData(frecuencias, fft_mag)
    
    def closeEvent(self, event):
        """Maneja el cierre de la ventana."""
        if self.adquisicion_thread is not None:
            self.adquisicion_thread.stop()
            self.adquisicion_thread.join(timeout=1.0)
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Configurar estilo visual
    app.setStyle('Fusion')
    
    # Crear y mostrar la ventana principal
    window = MachiningClassifierApp()
    
    # Ejecutar aplicación
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

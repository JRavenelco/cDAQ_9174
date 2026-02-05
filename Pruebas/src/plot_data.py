import sys
import csv
import os
import numpy as np
import pandas as pd # Usaremos pandas para facilitar el manejo de datos
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QComboBox, QLineEdit, QGroupBox, QFormLayout, QTabWidget, QMessageBox,
    QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar # Importar la barra de navegación
from matplotlib.figure import Figure
from scipy.signal import savgol_filter # Para Savitzky-Golay
from scipy.fft import rfft, rfftfreq # Para FFT


# --- Funciones de Procesamiento de Datos ---
def moving_average(data, window_size):
    if window_size <= 0 or window_size > len(data):
        return data # Devolver original si la ventana no es válida
    return pd.Series(data).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

def savitzky_golay(data, window_size, poly_order):
    if window_size <= 0 or window_size > len(data) or poly_order < 0:
        return data
    if window_size % 2 == 0: # window_size debe ser impar
        window_size += 1
    if poly_order >= window_size:
        poly_order = window_size - 1 
        if poly_order < 0: return data # No se puede aplicar
    try:
        return savgol_filter(data, window_size, poly_order)
    except Exception as e:
        print(f"Error en Savitzky-Golay: {e}")
        return data

def calculate_fft(data, sample_spacing):
    if len(data) < 2 or sample_spacing <= 0:
        return np.array([]), np.array([])
    N = len(data)
    yf = rfft(data)
    xf = rfftfreq(N, sample_spacing)
    return xf, np.abs(yf)


class DataProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visor y Procesador de Datos CSV")
        # Aumentar el tamaño de la ventana principal
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        self.setGeometry(50, 50, int(screen_geometry.width() * 0.9), int(screen_geometry.height() * 0.9))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Almacenamiento de datos
        self.data_file1 = None # DataFrame de Pandas
        self.data_file2 = None # DataFrame de Pandas
        self.processed_data_file1 = {} # { 'col_name': { 'method': data_array } }
        self.processed_data_file2 = {}

        self.filepath1 = ""
        self.filepath2 = ""

        self._setup_ui()

    def _setup_ui(self):
        # --- Sección de Carga de Archivos ---
        load_layout = QHBoxLayout()
        self.btn_load1 = QPushButton("Cargar Archivo 1")
        self.btn_load1.clicked.connect(lambda: self._load_file(1))
        self.lbl_file1 = QLabel("Archivo 1: No cargado")
        load_layout.addWidget(self.btn_load1)
        load_layout.addWidget(self.lbl_file1, 1) # Stretch factor

        self.btn_load2 = QPushButton("Cargar Archivo 2 (Opcional)")
        self.btn_load2.clicked.connect(lambda: self._load_file(2))
        self.lbl_file2 = QLabel("Archivo 2: No cargado")
        load_layout.addWidget(self.btn_load2)
        load_layout.addWidget(self.lbl_file2, 1)

        self.layout.addLayout(load_layout)

        # --- Sección de Selección de Columnas y Procesamiento ---
        main_processing_layout = QHBoxLayout()
        main_processing_layout.setContentsMargins(0, 0, 0, 0)  # Reducir márgenes

        # Panel Izquierdo: Selección de Columnas
        column_selection_group = QGroupBox("Selección de Columnas")
        column_selection_group.setMaximumWidth(300)  # Limitar el ancho del panel lateral
        col_sel_layout = QVBoxLayout(column_selection_group)
        col_sel_layout.setContentsMargins(5, 15, 5, 5)  # Reducir márgenes internos
        
        self.lbl_cols_file1 = QLabel("Columnas Archivo 1:")
        self.list_cols_file1 = QListWidget()
        self.list_cols_file1.setSelectionMode(QListWidget.MultiSelection)
        
        self.lbl_cols_file2 = QLabel("Columnas Archivo 2:")
        self.list_cols_file2 = QListWidget()
        self.list_cols_file2.setSelectionMode(QListWidget.MultiSelection)

        col_sel_layout.addWidget(self.lbl_cols_file1)
        col_sel_layout.addWidget(self.list_cols_file1)
        col_sel_layout.addWidget(self.lbl_cols_file2)
        col_sel_layout.addWidget(self.list_cols_file2)
        column_selection_group.setLayout(col_sel_layout)
        main_processing_layout.addWidget(column_selection_group, 1)


        # Panel Derecho: Pestañas de Procesamiento
        processing_tabs = QTabWidget()
        processing_tabs.setMinimumWidth(300)  # Ancho mínimo para las pestañas
        processing_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Pestaña Media Móvil
        self.tab_ma = QWidget()
        ma_layout = QFormLayout(self.tab_ma)
        self.ma_window_edit = QLineEdit("5")
        self.btn_apply_ma = QPushButton("Aplicar Media Móvil")
        self.btn_apply_ma.clicked.connect(self._apply_moving_average)
        ma_layout.addRow("Tamaño de Ventana:", self.ma_window_edit)
        ma_layout.addRow(self.btn_apply_ma)
        processing_tabs.addTab(self.tab_ma, "Media Móvil")

        # Pestaña Savitzky-Golay
        self.tab_sg = QWidget()
        sg_layout = QFormLayout(self.tab_sg)
        self.sg_window_edit = QLineEdit("11")
        self.sg_poly_edit = QLineEdit("2")
        self.btn_apply_sg = QPushButton("Aplicar Savitzky-Golay")
        self.btn_apply_sg.clicked.connect(self._apply_savgol)
        sg_layout.addRow("Tamaño de Ventana (impar):", self.sg_window_edit)
        sg_layout.addRow("Orden Polinomial:", self.sg_poly_edit)
        sg_layout.addRow(self.btn_apply_sg)
        processing_tabs.addTab(self.tab_sg, "Savitzky-Golay")
        
        # Pestaña FFT (para visualización, no procesamiento que modifique la serie temporal)
        self.tab_fft = QWidget()
        fft_layout = QFormLayout(self.tab_fft)
        self.fft_sample_spacing_edit = QLineEdit("0.001") # Asumir 1kHz por defecto
        self.btn_show_fft = QPushButton("Mostrar FFT de Columnas Seleccionadas")
        self.btn_show_fft.clicked.connect(self._update_time_series_plot)  # Conectar a la función de actualización
        fft_layout.addRow("Espaciado Muestras (1/SR):", self.fft_sample_spacing_edit)
        fft_layout.addRow(self.btn_show_fft)
        processing_tabs.addTab(self.tab_fft, "FFT")


        main_processing_layout.addWidget(processing_tabs, 2) 
        self.layout.addLayout(main_processing_layout)

        # --- Sección de Gráficos ---
        plot_group = QGroupBox("")
        plot_group.setStyleSheet("QGroupBox { border: 1px solid gray; border-radius: 5px; margin-top: 0.5em; }")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(5, 15, 5, 5)  # Reducir márgenes internos
        
        # Configuración común para las figuras
        # Hacer los gráficos más grandes para llenar más espacio
        self.fig_width = 16  # Aumentado de 14 a 16
        self.fig_height = 10  # Aumentado de 8 a 10
        self.dpi = 100        # Mantener DPI para calidad
        
        # Gráfico de Series Temporales
        self.fig_time_series = Figure(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)
        self.canvas_time_series = FigureCanvas(self.fig_time_series)
        self.toolbar_time_series = NavigationToolbar(self.canvas_time_series, self) 
        self.ax_time_series = self.fig_time_series.add_subplot(111)
        
        # Configurar el gráfico de series temporales
        self.ax_time_series.tick_params(axis='both', which='major', labelsize=10)
        self.ax_time_series.grid(True, linestyle='--', alpha=0.7)
        self.ax_time_series.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
        self.ax_time_series.set_ylabel('Amplitud', fontsize=12, fontweight='bold')
        self.ax_time_series.set_title('Series Temporales', fontsize=14, fontweight='bold')
        self.fig_time_series.tight_layout(pad=3.0)  # Añadir más padding
        
        # Gráfico de FFT
        self.fig_fft = Figure(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)
        self.canvas_fft = FigureCanvas(self.fig_fft)
        self.toolbar_fft = NavigationToolbar(self.canvas_fft, self) 
        self.ax_fft = self.fig_fft.add_subplot(111)
        
        # Configurar el gráfico de FFT
        self.ax_fft.tick_params(axis='both', which='major', labelsize=10)
        self.ax_fft.grid(True, linestyle='--', alpha=0.7)
        self.ax_fft.set_xlabel('Frecuencia (Hz)', fontsize=12, fontweight='bold')
        self.ax_fft.set_ylabel('Magnitud', fontsize=12, fontweight='bold')
        self.ax_fft.set_title('Espectro de Frecuencia (FFT)', fontsize=14, fontweight='bold')
        self.fig_fft.tight_layout(pad=3.0)  # Añadir más padding

        plot_layout.addWidget(self.toolbar_time_series) 
        plot_layout.addWidget(self.canvas_time_series, 1)  # El 1 hace que el gráfico se expanda
        plot_layout.addWidget(self.toolbar_fft) 
        plot_layout.addWidget(self.canvas_fft, 1)  # El 1 hace que el gráfico se expanda
        
        # Añadir el grupo de gráficos al layout principal con más prioridad de estiramiento
        self.layout.addWidget(plot_group, 2)  # Aumentar la prioridad de estiramiento
        
        # Ajustar el espaciado general
        self.layout.setSpacing(5)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Asegurar que los gráficos se expandan verticalmente
        self.central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_time_series.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_fft.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Botón de Actualizar/Limpiar Gráficos
        graph_controls_layout = QHBoxLayout()
        self.btn_update_plot = QPushButton("Actualizar Gráfico de Series Temporales")
        self.btn_update_plot.clicked.connect(self._update_time_series_plot)
        self.btn_clear_processed = QPushButton("Limpiar Datos Procesados y Gráficos")
        self.btn_clear_processed.clicked.connect(self._clear_all)
        graph_controls_layout.addWidget(self.btn_update_plot)
        graph_controls_layout.addWidget(self.btn_clear_processed)
        self.layout.addLayout(graph_controls_layout)

    def _load_file(self, file_num):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        initial_dir = os.path.join(script_dir, 'datos_automaticos')
        if not os.path.isdir(initial_dir): initial_dir = script_dir
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, f"Seleccionar Archivo CSV {file_num}", initial_dir, "CSV Files (*.csv)"
        )
        if not filepath: return

        try:
            df = pd.read_csv(filepath, comment='#') # comment='#' para ignorar líneas de cabecera de np.savetxt
            # Intentar convertir todas las columnas a numérico, errores a NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Eliminar filas donde la primera columna (X) es NaN, ya que son inútiles para graficar
            if not df.empty and df.columns[0] is not None:
                 df.dropna(subset=[df.columns[0]], inplace=True)


            if file_num == 1:
                self.data_file1 = df
                self.filepath1 = filepath
                self.lbl_file1.setText(f"Archivo 1: {os.path.basename(filepath)}")
                self.processed_data_file1.clear()
                self._populate_column_list(self.list_cols_file1, df)
            else:
                self.data_file2 = df
                self.filepath2 = filepath
                self.lbl_file2.setText(f"Archivo 2: {os.path.basename(filepath)}")
                self.processed_data_file2.clear()
                self._populate_column_list(self.list_cols_file2, df)
            
            self._update_time_series_plot() # Actualizar gráfico con datos originales
            self._clear_fft_plot() # Limpiar gráfico FFT al cargar nuevos datos

        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar", f"No se pudo cargar o procesar el archivo:\n{filepath}\nError: {e}")
            if file_num == 1:
                self.data_file1 = None
                self.lbl_file1.setText("Archivo 1: Error")
                self.list_cols_file1.clear()
            else:
                self.data_file2 = None
                self.lbl_file2.setText("Archivo 2: Error")
                self.list_cols_file2.clear()

    def _populate_column_list(self, list_widget, dataframe):
        list_widget.clear()
        if dataframe is None or dataframe.empty: return
        # No añadir la primera columna (X) a la lista de selección de Y
        for col_name in dataframe.columns[1:]:
            item = QListWidgetItem(col_name)
            list_widget.addItem(item)

    def _get_selected_columns(self, list_widget, data_file_num):
        selected_items = list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Sin Selección", f"Por favor, seleccione al menos una columna del Archivo {data_file_num}.")
            return None, None
        
        df = self.data_file1 if data_file_num == 1 else self.data_file2
        if df is None or df.empty:
            QMessageBox.warning(self, "Sin Datos", f"No hay datos cargados para el Archivo {data_file_num}.")
            return None, None
            
        x_col_name = df.columns[0]
        x_data = df[x_col_name].to_numpy()
        
        selected_y_cols = {} # {'col_name': y_data_array}
        for item in selected_items:
            col_name = item.text()
            if col_name in df.columns:
                selected_y_cols[col_name] = df[col_name].to_numpy()
        return x_data, selected_y_cols

    def _apply_moving_average(self):
        try:
            window = int(self.ma_window_edit.text())
            if window <= 0:
                QMessageBox.warning(self, "Parámetro Inválido", "El tamaño de ventana debe ser positivo.")
                return
        except ValueError:
            QMessageBox.warning(self, "Entrada Inválida", "El tamaño de ventana debe ser un número entero.")
            return

        self._apply_processing(1, "MA", lambda data: moving_average(data, window))
        self._apply_processing(2, "MA", lambda data: moving_average(data, window))
        self._update_time_series_plot()

    def _apply_savgol(self):
        try:
            window = int(self.sg_window_edit.text())
            polyorder = int(self.sg_poly_edit.text())
            if window <= 0 or polyorder < 0:
                QMessageBox.warning(self, "Parámetro Inválido", "Ventana debe ser >0, Orden Pol. >=0.")
                return
            if window % 2 == 0:
                 QMessageBox.information(self, "Ajuste Ventana SG", "Ventana para Savitzky-Golay debe ser impar. Se ajustará a " + str(window+1))
                 window +=1 # Asegurar impar
            if polyorder >= window:
                QMessageBox.warning(self, "Parámetro Inválido", "Orden polinomial debe ser menor que el tamaño de ventana.")
                return

        except ValueError:
            QMessageBox.warning(self, "Entrada Inválida", "Ventana y Orden Pol. deben ser números enteros.")
            return
        
        self._apply_processing(1, "SG", lambda data: savitzky_golay(data, window, polyorder))
        self._apply_processing(2, "SG", lambda data: savitzky_golay(data, window, polyorder))
        self._update_time_series_plot()


    def _apply_processing(self, file_num, method_name, processing_func):
        list_widget = self.list_cols_file1 if file_num == 1 else self.list_cols_file2
        processed_data_dict = self.processed_data_file1 if file_num == 1 else self.processed_data_file2
        df_original = self.data_file1 if file_num == 1 else self.data_file2

        if df_original is None or df_original.empty: return

        _, selected_y_cols = self._get_selected_columns(list_widget, file_num)
        if not selected_y_cols: return

        for col_name, y_data_orig in selected_y_cols.items():
            # Filtrar NaNs antes de procesar, pero mantener la estructura original para el plot
            valid_indices = ~np.isnan(y_data_orig)
            y_data_to_process = y_data_orig[valid_indices]
            
            if len(y_data_to_process) > 0:
                processed_segment = processing_func(y_data_to_process)
                
                # Reinsertar el segmento procesado en un array lleno de NaNs del tamaño original
                y_data_processed_full = np.full_like(y_data_orig, np.nan)
                y_data_processed_full[valid_indices] = processed_segment
            else:
                y_data_processed_full = np.full_like(y_data_orig, np.nan)


            if col_name not in processed_data_dict:
                processed_data_dict[col_name] = {}
            processed_data_dict[col_name][method_name] = y_data_processed_full
        
        print(f"Procesamiento '{method_name}' aplicado a columnas seleccionadas del Archivo {file_num}.")


    def _update_time_series_plot(self):
        self.ax_time_series.clear()
        plot_styles = ['-', '--']
        
        # Plot File 1
        if self.data_file1 is not None and not self.data_file1.empty:
            x_col_name1 = self.data_file1.columns[0]
            x_data1 = self.data_file1[x_col_name1].to_numpy()
            selected_cols1_items = self.list_cols_file1.selectedItems()
            selected_cols1_names = [item.text() for item in selected_cols1_items]

            for y_col_name in self.data_file1.columns[1:]:
                if y_col_name not in selected_cols1_names and selected_cols1_names: # Si hay selección, solo graficar seleccionadas
                    continue
                y_data_orig1 = self.data_file1[y_col_name].to_numpy()
                valid_idx1 = ~np.isnan(x_data1) & ~np.isnan(y_data_orig1)
                self.ax_time_series.plot(x_data1[valid_idx1], y_data_orig1[valid_idx1], 
                                         label=f"{y_col_name} (F1 Orig)", linestyle=plot_styles[0], alpha=0.7, linewidth=2)
        
        # Plot File 2
        if self.data_file2 is not None and not self.data_file2.empty:
            x_col_name2 = self.data_file2.columns[0]
            x_data2 = self.data_file2[x_col_name2].to_numpy()
            selected_cols2_items = self.list_cols_file2.selectedItems()
            selected_cols2_names = [item.text() for item in selected_cols2_items]

            for y_col_name in self.data_file2.columns[1:]:
                if y_col_name not in selected_cols2_names and selected_cols2_names:
                    continue
                y_data_orig2 = self.data_file2[y_col_name].to_numpy()
                valid_idx2 = ~np.isnan(x_data2) & ~np.isnan(y_data_orig2)
                self.ax_time_series.plot(x_data2[valid_idx2], y_data_orig2[valid_idx2], 
                                         label=f"{y_col_name} (F2 Orig)", linestyle=plot_styles[1], alpha=0.7, linewidth=2)
        
        # Configuración del gráfico
        self.ax_time_series.set_xlabel('Tiempo (s)', fontsize=12, fontweight='bold')
        self.ax_time_series.set_ylabel('Amplitud', fontsize=12, fontweight='bold')
        self.ax_time_series.set_title('Series Temporales', fontsize=14, fontweight='bold')
        self.ax_time_series.grid(True, linestyle='--', alpha=0.7)
        
        # Mostrar leyenda si hay datos
        if (self.data_file1 is not None and not self.data_file1.empty) or \
           (self.data_file2 is not None and not self.data_file2.empty):
            legend = self.ax_time_series.legend(prop={'size': 10}, loc='best')
            legend.set_draggable(True)  # Hacer la leyenda arrastrable
        
        # Ajustar márgenes
        self.fig_time_series.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.1, hspace=0.3)
        
        # Actualizar el canvas
        self.canvas_time_series.draw()
        
        # Procesar FFT si es necesario
        try:
            sample_spacing_str = self.fft_sample_spacing_edit.text()
            if not sample_spacing_str:
                return
                
            sample_spacing = float(sample_spacing_str)
            if sample_spacing <= 0:
                return
                
            fft_plotted = False
            self.ax_fft.clear()
            
            # FFT para Archivo 1
            if self.data_file1 is not None and not self.data_file1.empty:
                _, selected_y_cols1 = self._get_selected_columns(self.list_cols_file1, 1)
                if selected_y_cols1:
                    for col_name, y_data in selected_y_cols1.items():
                        y_clean = y_data[~np.isnan(y_data)]  # Quitar NaNs para FFT
                        if len(y_clean) > 1:
                            xf, yf = calculate_fft(y_clean, sample_spacing)
                            self.ax_fft.plot(xf, yf, label=f"FFT {col_name} (F1)", linewidth=2)
                            fft_plotted = True
            
            # FFT para Archivo 2
            if self.data_file2 is not None and not self.data_file2.empty:
                _, selected_y_cols2 = self._get_selected_columns(self.list_cols_file2, 2)
                if selected_y_cols2:
                    for col_name, y_data in selected_y_cols2.items():
                        y_clean = y_data[~np.isnan(y_data)]
                        if len(y_clean) > 1:
                            xf, yf = calculate_fft(y_clean, sample_spacing)
                            self.ax_fft.plot(xf, yf, label=f"FFT {col_name} (F2)", linestyle='--', linewidth=2)
                            fft_plotted = True
            
            if fft_plotted:
                self.ax_fft.set_title('Espectro de Frecuencia (FFT)', fontsize=14, fontweight='bold')
                self.ax_fft.set_xlabel('Frecuencia (Hz)', fontsize=12, fontweight='bold')
                self.ax_fft.set_ylabel('Magnitud', fontsize=12, fontweight='bold')
                self.ax_fft.grid(True, linestyle='--', alpha=0.7)
                legend = self.ax_fft.legend(prop={'size': 10}, loc='best')
                legend.set_draggable(True)
            else:
                self.ax_fft.text(0.5, 0.5, "Cargue datos y seleccione columnas para ver FFT.", 
                                horizontalalignment='center', verticalalignment='center')
                self.ax_fft.set_title('Espectro de Frecuencia (FFT)', fontsize=14, fontweight='bold')
                self.ax_fft.set_xlabel('Frecuencia (Hz)', fontsize=12, fontweight='bold')
                self.ax_fft.set_ylabel('Magnitud', fontsize=12, fontweight='bold')
                self.ax_fft.grid(True, linestyle='--', alpha=0.7)
                
            self.canvas_fft.draw()
            self.fig_fft.tight_layout()
            
        except ValueError:
            # Ignorar errores de conversión de FFT
            pass

    def _clear_fft_plot(self):
        self.ax_fft.clear()
        self.ax_fft.text(0.5, 0.5, "Cargue datos y seleccione columnas para ver FFT.", 
                         horizontalalignment='center', verticalalignment='center')
        self.ax_fft.set_title("Transformada Rápida de Fourier (FFT)") 
        self.ax_fft.set_xlabel("Frecuencia (Hz)")
        self.ax_fft.set_ylabel("Magnitud")
        self.ax_fft.grid(True)
        self.canvas_fft.draw()
        self.fig_fft.tight_layout() # Asegurar que se llama

    def _clear_all(self):
        self.data_file1 = None
        self.data_file2 = None
        self.processed_data_file1.clear()
        self.processed_data_file2.clear()
        self.filepath1 = ""
        self.filepath2 = ""
        self.lbl_file1.setText("Archivo 1: No cargado")
        self.lbl_file2.setText("Archivo 2: No cargado")
        self.list_cols_file1.clear()
        self.list_cols_file2.clear()
        
        self.ax_time_series.clear()
        self.ax_time_series.set_title("Gráfico de Series Temporales") # Mantener título
        self.ax_time_series.set_xlabel("Eje X (Tiempo/Muestras)") # Mantener etiqueta X
        self.ax_time_series.set_ylabel("Valor") # Mantener etiqueta Y
        self.ax_time_series.grid(True) # Mantener rejilla
        self.canvas_time_series.draw()
        self.fig_time_series.tight_layout() # Asegurar que se llama
        
        self._clear_fft_plot() # Esto ya llama a tight_layout para FFT
        print("Datos procesados y gráficos limpiados.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = DataProcessorApp()
    main_window.show()
    sys.exit(app.exec_())

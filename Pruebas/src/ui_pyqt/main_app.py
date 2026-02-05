import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                             QTableWidget, QTableWidgetItem, QFileDialog, 
                             QSplitter, QTextEdit, QGroupBox, QFormLayout, 
                             QDoubleSpinBox, QSpinBox)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import math
import csv

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulación y Análisis de Fresado - PyQt5")
        self.setGeometry(100, 100, 1200, 800)  # x, y, width, height

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Create a splitter to make panels resizable
        splitter = QSplitter(Qt.Horizontal)

        # --- Left Panel (Simulation) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # --- Simulation Controls ---
        controls_group = QGroupBox("Parámetros de Simulación")
        controls_layout = QFormLayout()

        # Tool Parameters
        self.tool_diameter_input = QDoubleSpinBox(value=10.0, minimum=1.0, maximum=100.0, suffix=" mm")
        self.tool_flutes_input = QSpinBox(value=4, minimum=1, maximum=16)
        controls_layout.addRow("Diámetro Herramienta:", self.tool_diameter_input)
        controls_layout.addRow("Número de Filos:", self.tool_flutes_input)

        # Process Parameters
        self.process_rpm_input = QSpinBox(value=3000, minimum=100, maximum=20000, suffix=" RPM")
        self.process_feed_rate_input = QDoubleSpinBox(value=500, minimum=10, maximum=5000, suffix=" mm/min")
        self.process_ap_input = QDoubleSpinBox(value=1.0, minimum=0.1, maximum=50.0, decimals=2, suffix=" mm") # Axial depth
        self.process_ae_input = QDoubleSpinBox(value=5.0, minimum=0.1, maximum=100.0, decimals=2, suffix=" mm") # Radial depth
        controls_layout.addRow("Velocidad de Giro (RPM):", self.process_rpm_input)
        controls_layout.addRow("Velocidad de Avance:", self.process_feed_rate_input)
        controls_layout.addRow("Profundidad Axial (ap):", self.process_ap_input)
        controls_layout.addRow("Profundidad Radial (ae):", self.process_ae_input)

        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)

        self.start_sim_button = QPushButton("Iniciar/Actualizar Simulación")
        self.start_sim_button.clicked.connect(self.start_animation)
        left_layout.addWidget(self.start_sim_button)

        # --- Simulation Visualization ---
        self.sim_plot_xy = pg.PlotWidget(title="Vista Superior (Plano XY)")
        self.sim_plot_xy.setAspectLocked(True)
        self.sim_plot_xy.setLabel('bottom', 'X (mm)')
        self.sim_plot_xy.setLabel('left', 'Y (mm)')
        self.sim_plot_xy.showGrid(x=True, y=True)
        left_layout.addWidget(self.sim_plot_xy)
        
        self.sim_plot_xz = pg.PlotWidget(title="Vista Lateral (Plano XZ)")
        self.sim_plot_xz.setAspectLocked(True)
        self.sim_plot_xz.setLabel('bottom', 'X (mm)')
        self.sim_plot_xz.setLabel('left', 'Z (mm)')
        self.sim_plot_xz.showGrid(x=True, y=True)
        left_layout.addWidget(self.sim_plot_xz)

        # Initialize simulation variables
        self.initial_tool_click_x = -15.0  # Default initial X position
        self.initial_tool_click_y = None   # None means centered on workpiece Y
        
        # Animation variables
        self.animation_timer = None
        self.animation_step = 0
        self.animation_max_steps = 100
        self.animation_in_progress = False
        self.animation_phase = 0  # 0: not started, 1: descent, 2: cutting
        self.current_tool_x = -15.0
        self.current_tool_y = 25.0  # Default Y position
        self.current_tool_z = 30.0  # Start above workpiece
        self.target_tool_z = 0.0    # Target depth (will be set to -ap)

        # --- Storage for force vector arrows so they can be cleared/redrawn ---
        self.force_arrow_xy = []
        self.force_arrow_xz = []
        
        # Connect the run button and mouse clicks
        self.start_sim_button.clicked.connect(self.start_animation)
        self.sim_plot_xy.scene().sigMouseClicked.connect(self.on_plot_xy_clicked)

        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # --- Right Panel (Analysis) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.tab_widget = QTabWidget()
        
        # Time Domain Tab
        self.time_domain_tab = QWidget()
        self.time_domain_layout = QVBoxLayout(self.time_domain_tab)

        # Layout for Force Data Section
        force_data_group = QWidget()
        force_data_layout = QVBoxLayout(force_data_group)
        force_data_layout.setContentsMargins(0,0,0,0) # Remove margins for tighter packing

        self.load_force_button = QPushButton("Cargar CSV de Fuerza")
        self.load_force_button.clicked.connect(lambda: self.load_csv_data('force'))
        force_data_layout.addWidget(self.load_force_button)

        self.force_summary_label = QLabel("Datos de Fuerza: No cargados")
        force_data_layout.addWidget(self.force_summary_label)

        self.force_data_table = QTableWidget()
        force_data_layout.addWidget(self.force_data_table)
        self.force_plot_widget = pg.PlotWidget(title="Gráfico de Fuerza")
        force_data_layout.addWidget(self.force_plot_widget)
        self.time_domain_layout.addWidget(force_data_group)

        # Layout for Vibration Data Section
        vibration_data_group = QWidget()
        vibration_data_layout = QVBoxLayout(vibration_data_group)
        vibration_data_layout.setContentsMargins(0,0,0,0) # Remove margins

        self.load_vibration_button = QPushButton("Cargar CSV de Vibración")
        self.load_vibration_button.clicked.connect(lambda: self.load_csv_data('vibration'))
        vibration_data_layout.addWidget(self.load_vibration_button)

        self.vibration_summary_label = QLabel("Datos de Vibración: No cargados")
        vibration_data_layout.addWidget(self.vibration_summary_label)

        self.vibration_data_table = QTableWidget()
        vibration_data_layout.addWidget(self.vibration_data_table)
        self.vibration_plot_widget = pg.PlotWidget(title="Gráfico de Vibración")
        vibration_data_layout.addWidget(self.vibration_plot_widget)
        self.time_domain_layout.addWidget(vibration_data_group)
        self.tab_widget.addTab(self.time_domain_tab, "Dominio del Tiempo")
        
        # Frequency Domain Tab
        self.frequency_domain_tab = QWidget()
        self.frequency_domain_layout = QVBoxLayout(self.frequency_domain_tab)

        # FFT Plot for Force Data
        self.force_fft_plot_widget = pg.PlotWidget(title="FFT de Fuerza")
        self.frequency_domain_layout.addWidget(self.force_fft_plot_widget)

        # FFT Plot for Vibration Data
        self.vibration_fft_plot_widget = pg.PlotWidget(title="FFT de Vibración")
        self.frequency_domain_layout.addWidget(self.vibration_fft_plot_widget)

        self.tab_widget.addTab(self.frequency_domain_tab, "Dominio de la Frecuencia")
        
        # Diagnostics and Log Tab
        self.diag_log_tab = QWidget()
        self.diag_log_layout = QVBoxLayout(self.diag_log_tab)
        self.diag_log_layout.addWidget(QLabel("Contenido de Diagnóstico y Log"))
        # TODO: Add diagnostic results, log messages
        self.tab_widget.addTab(self.diag_log_tab, "Diagnóstico y Log")
        
        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_panel)

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Set initial sizes for splitter panels (optional)
        splitter.setSizes([600, 600]) # Adjust as needed

    def on_plot_xy_clicked(self, event):
        # Ensure click is with the left mouse button
        if event.button() == Qt.LeftButton:
            pos = event.scenePos() # Get position in scene coordinates
            # Map scene position to view coordinates (the coordinates of the data in the plot)
            view_coords = self.sim_plot_xy.getViewBox().mapSceneToView(pos)
            self.initial_tool_click_x = view_coords.x()
            self.initial_tool_click_y = view_coords.y()
            self.current_tool_x = self.initial_tool_click_x
            self.current_tool_y = self.initial_tool_click_y
            print(f"Tool start position set by click: X={self.initial_tool_click_x:.2f}, Y={self.initial_tool_click_y:.2f}")
            self.run_simulation() # Redraw simulation with new tool position
            
    def start_animation(self):
        """Inicia la animación de la herramienta"""
        # Reset variables de animación
        self.animation_step = 0
        self.animation_phase = 1  # Comenzar en fase de descenso (1)
        self.animation_in_progress = True
        
        # Obtener parámetros actuales
        ap = self.process_ap_input.value()
        self.target_tool_z = -ap  # Profundidad objetivo
        
        # Establecer posición inicial
        if self.initial_tool_click_y is None:
            self.current_tool_y = 25.0  # Default Y centered on workpiece
        else:
            self.current_tool_y = self.initial_tool_click_y
        
        self.current_tool_x = self.initial_tool_click_x
        self.current_tool_z = 30.0  # Comenzar por encima de la pieza
        
        # Configurar y arrancar el timer
        if self.animation_timer is None:
            from PyQt5.QtCore import QTimer
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.update_animation)
        
        self.animation_timer.start(50)  # Actualizar cada 50ms
        print("Animación iniciada: Fase de descenso de la herramienta")
        
        # Dibujar estado inicial
        self.run_simulation()
        
    def update_animation(self):
        """Actualizar la animación en cada paso del timer"""
        if not self.animation_in_progress:
            return
        
        if self.animation_phase == 1:  # Fase de descenso
            # Calcular descenso progresivo
            descent_step = (30.0 - self.target_tool_z) / 20  # 20 pasos para el descenso
            self.current_tool_z -= descent_step
            
            # Verificar si llegamos a la profundidad objetivo
            if self.current_tool_z <= self.target_tool_z:
                self.current_tool_z = self.target_tool_z
                self.animation_phase = 2  # Cambiar a fase de corte
                print("Animación: Cambio a fase de corte")
                
        elif self.animation_phase == 2:  # Fase de corte (movimiento en X)
            # Calcular avance en X
            feed_step = 2.0  # mm por paso (puede escalarse con feed rate)
            self.current_tool_x += feed_step
            
            # Verificar si llegamos al final del corte (longitud pieza 100 mm + pequeño margen)
            if self.current_tool_x > 100.0:
                self.animation_in_progress = False
                self.animation_timer.stop()
                print("Animación: Corte completado")
        
        # Actualizar visualización
        self.run_simulation(False)  # False = no resetear variables de animación
        
        # Incrementar contador de pasos
        # Calcular y dibujar fuerzas sólo durante la fase de corte
        if self.animation_phase == 2:
            feed_rate = self.process_feed_rate_input.value()
            rpm = self.process_rpm_input.value()
            flutes = self.tool_flutes_input.value()
            ap = self.process_ap_input.value()
            # Evitar división por cero
            if rpm > 0 and flutes > 0:
                feed_per_rev = feed_rate / rpm  # mm/rev
                chip_thk = feed_per_rev / flutes  # mm/tooth
            else:
                chip_thk = 0.0
            Fx, Fy, Fz = self.calculate_cutting_forces(chip_thk, ap)
            self.draw_force_vectors(Fx, Fy, Fz)

        self.animation_step += 1

    # ------------------------------------------------------------------
    # ---  Cálculo de fuerzas y dibujo de vectores                      ---
    # ------------------------------------------------------------------
    def calculate_cutting_forces(self, chip_thickness, ap):
        """Devuelve Fx, Fy, Fz aproximados (N). Modelo simplificado"""
        # Coeficiente de corte (N/mm^2) – simplificación
        Kc = 1500.0
        Kr = 0.3  # Relación fuerza radial/tangencial

        Fc = Kc * chip_thickness * ap  # Fuerza tangencial (dirección avance X)
        Fr = Kr * Fc                   # Fuerza radial (normal a corte, eje Z)

        Fx = Fc
        Fy = 0.0
        Fz = -Fr  # Negativo porque apunta hacia abajo en Z
        return Fx, Fy, Fz

    def draw_force_vectors(self, Fx, Fy, Fz):
        """Dibuja flechas de fuerza en las vistas XY y XZ"""
        # Escala simple para convertir Newtons a pixels
        scale = 50.0  # Ajustar según se necesite

        # ----------------------------------------------------------------
        # Limpiar flechas previas
        # ----------------------------------------------------------------
        for arr in self.force_arrow_xy:
            try:
                self.sim_plot_xy.removeItem(arr)
            except Exception:
                pass
        self.force_arrow_xy.clear()

        if hasattr(self, 'workpiece_poly_xz_item'):
            self.sim_plot_xz.removeItem(self.workpiece_poly_xz_item)
        for arr in self.force_arrow_xz:
            try:
                self.sim_plot_xz.removeItem(arr)
            except Exception:
                pass
        self.force_arrow_xz.clear()

        # ----------------------------------------------------------------
        # Flecha XY (Fx, Fy)
        # ----------------------------------------------------------------
        mag_xy = math.hypot(Fx, Fy)
        if mag_xy > 0:
            angle_xy = math.degrees(math.atan2(Fy, Fx))
            tail_len_xy = mag_xy / scale
            arrow_xy = pg.ArrowItem(angle=angle_xy, headLen=8, tipAngle=30, tailLen=tail_len_xy, brush='r')
            arrow_xy.setPos(self.current_tool_x, self.current_tool_y)
            self.sim_plot_xy.addItem(arrow_xy)
            self.force_arrow_xy.append(arrow_xy)

        # ----------------------------------------------------------------
        # Flecha XZ (Fx, Fz)
        # ----------------------------------------------------------------
        mag_xz = math.hypot(Fx, Fz)
        if mag_xz > 0:
            angle_xz = math.degrees(math.atan2(Fz, Fx))
            tail_len_xz = mag_xz / scale
            arrow_xz = pg.ArrowItem(angle=angle_xz, headLen=8, tipAngle=30, tailLen=tail_len_xz, brush='r')
            arrow_xz.setPos(self.current_tool_x, self.current_tool_z)
            self.sim_plot_xz.addItem(arrow_xz)
            self.force_arrow_xz.append(arrow_xz)

    # ------------------------------------------------------------------
    # --- Simulación y gráficos principales                             ---
    # ------------------------------------------------------------------
    def run_simulation(self, reset_animation=True):
        tool_diameter = self.tool_diameter_input.value()
        tool_flutes = self.tool_flutes_input.value()
        rpm = self.process_rpm_input.value()
        feed_rate = self.process_feed_rate_input.value()
        ap = self.process_ap_input.value() # Axial depth
        ae = self.process_ae_input.value() # Radial depth

        print("--- Parámetros de Simulación ---")
        print(f"Diámetro Herramienta: {tool_diameter} mm")
        print(f"Número de Filos: {tool_flutes}")
        print(f"RPM: {rpm}")
        print(f"Avance: {feed_rate} mm/min")
        print(f"Profundidad Axial (ap): {ap} mm")
        print(f"Profundidad Radial (ae): {ae} mm")
        print("---------------------------------")

        # --- Define workpiece dimensions (fixed for now) ---
        wp_length_x = 100.0  # mm
        wp_width_y = 50.0   # mm
        wp_height_z = 20.0  # mm (material thickness)

        # --- Clear previous plots ---
        self.sim_plot_xy.clear()
        self.sim_plot_xz.clear()

        # --- Tool parameters ---
        D = tool_diameter
        R = D / 2.0

        wp_pen = pg.mkPen('w', width=1)
        wp_brush = pg.mkBrush(173, 216, 230, 150) # Light blue fill (R, G, B, Alpha)

        # XY View (Top View) - Workpiece is a rectangle
        wp_xy_x_coords = [0, wp_length_x, wp_length_x, 0, 0]
        wp_xy_y_coords = [0, 0, wp_width_y, wp_width_y, 0]
        self.sim_plot_xy.plot(wp_xy_x_coords, wp_xy_y_coords, pen=wp_pen, fillLevel=0, brush=wp_brush)

        # XZ View (Side View) - Workpiece es trapezoidal
        wp_x_start = 0
        wp_z_bottom = -wp_height_z  # Base of trapezoid rests on top of fixture
        wp_z_top = 0  # Top surface of trapezoid is at Z=0
        wp_top_width = wp_length_x
        # slope_indent_x makes the bottom width wp_top_width - 2 * slope_indent_x
        # For a bottom width of 60% of top width, each indent is 20% of top width.
        slope_indent_x = wp_top_width * 0.2 

        # Define the points for the trapezoid to be plotted
        trap_x = [wp_x_start, wp_x_start + wp_top_width, wp_x_start + wp_top_width - slope_indent_x, wp_x_start + slope_indent_x, wp_x_start]
        trap_z = [wp_z_bottom, wp_z_bottom, wp_z_top, wp_z_top, wp_z_bottom]
        
        # Plot the trapezoid shape
        self.sim_plot_xz.plot(trap_x, trap_z, pen=wp_pen, fillLevel=0, brush=wp_brush)

        # Almacenar los puntos del perfil para referencia (sensores, etc.)
        self.wp_xz_x_profile = trap_x
        self.wp_xz_z_profile = trap_z

        # --- Configurar posición inicial de la herramienta si es necesario ---
        if reset_animation and not self.animation_in_progress:
            # Solo reiniciar posiciones si no estamos en medio de una animación
            self.current_tool_x = self.initial_tool_click_x
            if self.initial_tool_click_y is None:
                self.current_tool_y = wp_width_y / 2.0  # Default Y centered on workpiece
            else:
                self.current_tool_y = self.initial_tool_click_y
            self.current_tool_z = 0  # Posición inicial en altura (superficie)
        
        # --- Draw Tool (using animation position variables) --- 
        tool_color = pg.mkColor('c') # Cyan for tool
        tool_pen = pg.mkPen(tool_color, width=2)
        tool_brush = pg.mkBrush(tool_color.red(), tool_color.green(), tool_color.blue(), 120)
        
        # XY View (Top View) - Tool as a circle
        num_points_circle = 100
        theta = np.linspace(0, 2 * np.pi, num_points_circle)
        tool_circle_x = self.current_tool_x + R * np.cos(theta)
        tool_circle_y = self.current_tool_y + R * np.sin(theta)
        self.sim_plot_xy.plot(tool_circle_x, tool_circle_y, pen=tool_pen, fillLevel=0, brush=tool_brush)

        # XZ View (Side View) - Tool as a rectangle
        visual_tool_length_z = D * 1.5 
        tool_rect_xz_x = [self.current_tool_x - R, self.current_tool_x + R, self.current_tool_x + R, self.current_tool_x - R, self.current_tool_x - R]
        tool_rect_xz_z = [self.current_tool_z, self.current_tool_z, self.current_tool_z + visual_tool_length_z, self.current_tool_z + visual_tool_length_z, self.current_tool_z]
        self.sim_plot_xz.plot(tool_rect_xz_x, tool_rect_xz_z, pen=tool_pen, fillLevel=0, brush=tool_brush)

        # --- Draw Fixture (bancada) ---
        # Dibujar la bancada (más grande que la pieza) en color más oscuro
        fixture_pen = pg.mkPen('w', width=1)
        fixture_brush = pg.mkBrush(100, 100, 100, 150) # Gris para la bancada
        
        # Dimensiones de la bancada
        fixture_margin = 30      # Margen alrededor de la pieza
        fixture_height = 30      # Altura total de la bancada
        fixture_width_y = wp_width_y + 2 * fixture_margin
        fixture_length_x = wp_length_x + 2 * fixture_margin
        
        # XY View - Bancada como rectángulo más grande
        fixture_xy_x = [-fixture_margin, wp_length_x + fixture_margin, wp_length_x + fixture_margin, -fixture_margin, -fixture_margin]
        fixture_xy_y = [-fixture_margin, -fixture_margin, wp_width_y + fixture_margin, wp_width_y + fixture_margin, -fixture_margin]
        self.sim_plot_xy.plot(fixture_xy_x, fixture_xy_y, pen=fixture_pen, fillLevel=0, brush=fixture_brush)
        
        # XZ View - Bancada como rectángulo bajo la pieza
        fixture_xz_x = [-fixture_margin, wp_length_x + fixture_margin, wp_length_x + fixture_margin, -fixture_margin, -fixture_margin]
        fixture_xz_z = [-wp_height_z - fixture_height, -wp_height_z - fixture_height, -wp_height_z, -wp_height_z, -wp_height_z - fixture_height]
        self.sim_plot_xz.plot(fixture_xz_x, fixture_xz_z, pen=fixture_pen, fillLevel=0, brush=fixture_brush)
        
        # --- Draw Sensors (Yellow Circles) ---
        sensor_pen = pg.mkPen(color=(255, 255, 0), width=1)
        sensor_brush = pg.mkBrush(255, 255, 0, 150) # Yellow
        sensor_size = 8 # Diameter for scatter plot

        # Sensores alrededor de la bancada - Vista XY
        sensors_xy_data = [
            {'pos': (-fixture_margin - 10, wp_width_y / 2), 'size': sensor_size, 'pen': sensor_pen, 'brush': sensor_brush}, # Izquierda
            {'pos': (wp_length_x / 2, -fixture_margin - 10), 'size': sensor_size, 'pen': sensor_pen, 'brush': sensor_brush}, # Abajo
            {'pos': (wp_length_x + fixture_margin + 10, wp_width_y / 2), 'size': sensor_size, 'pen': sensor_pen, 'brush': sensor_brush}, # Derecha
            {'pos': (wp_length_x / 2, wp_width_y + fixture_margin + 10), 'size': sensor_size, 'pen': sensor_pen, 'brush': sensor_brush}  # Arriba
        ]
        self.sim_plot_xy.addItem(pg.ScatterPlotItem(sensors_xy_data))

        # Sensores alrededor de la bancada - Vista XZ
        sensors_xz_data = [
            {'pos': (self.current_tool_x - R - 15, -5), 'size': sensor_size, 'pen': sensor_pen, 'brush': sensor_brush}, # Cerca de la herramienta
            {'pos': (20, -wp_height_z - fixture_height - 5), 'size': sensor_size, 'pen': sensor_pen, 'brush': sensor_brush}, # Debajo de bancada inicio
            {'pos': (70, -wp_height_z - fixture_height - 5), 'size': sensor_size, 'pen': sensor_pen, 'brush': sensor_brush}, # Debajo de bancada final
            {'pos': (wp_length_x + fixture_margin + 5, -wp_height_z - 10), 'size': sensor_size, 'pen': sensor_pen, 'brush': sensor_brush}  # Lado derecho
        ]
        # self.sim_plot_xz.addItem(pg.ScatterPlotItem(sensors_xz_data))

        # --- Adjust plot ranges to include sensors, workpiece, and fixture ---
        padding = 40 # Padding ampliado para visualización completa
        self.sim_plot_xy.setRange(xRange=[-fixture_margin - padding, wp_length_x + fixture_margin + padding], 
                                  yRange=[-fixture_margin - padding, wp_width_y + fixture_margin + padding])
        self.sim_plot_xz.setRange(xRange=[-fixture_margin - padding, wp_length_x + fixture_margin + padding], 
                                  yRange=[-wp_height_z - fixture_height - padding, visual_tool_length_z + padding])

    def load_csv_data(self, data_type):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog # Uncomment if native dialog causes issues
        file_path, _ = QFileDialog.getOpenFileName(self, "Cargar Archivo CSV", "", 
                                                   "Archivos CSV (*.csv);;Todos los Archivos (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'r', newline='', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    headers = next(reader, None) # Read headers
                    data_rows = list(reader)   # Read all data rows

                if headers:
                    num_cols = len(headers)
                    num_rows = len(data_rows)
                    summary_text = f"Archivo: {file_path.split('/')[-1]}\nColumnas: {num_cols}, Filas de datos: {num_rows}"
                    
                    target_table = None
                    target_summary_label = None

                    if data_type == 'force':
                        self.force_headers = headers
                        self.force_data_rows = data_rows
                        target_table = self.force_data_table
                        target_summary_label = self.force_summary_label
                        prefix_summary = "Datos de Fuerza: "
                    elif data_type == 'vibration':
                        self.vibration_headers = headers
                        self.vibration_data_rows = data_rows
                        target_table = self.vibration_data_table
                        target_summary_label = self.vibration_summary_label
                        prefix_summary = "Datos de Vibración: "
                    
                    if target_table is not None and target_summary_label is not None:
                        target_summary_label.setText(prefix_summary + summary_text)
                        target_table.setColumnCount(num_cols)
                        target_table.setHorizontalHeaderLabels(headers)
                        rows_to_display = min(num_rows, 100)
                        target_table.setRowCount(rows_to_display)

                        # Prepare data for plotting
                        plot_data_columns = []
                        time_column_index = -1

                        # Try to identify time column (e.g., named 'Tiempo' or 'Time', or first column)
                        for idx, h_name in enumerate(headers):
                            if h_name.lower() in ['tiempo', 'time']:
                                time_column_index = idx
                                break
                        if time_column_index == -1 and num_cols > 0: # Default to first column if no specific name found
                            try: # Check if first column is numeric
                                float(data_rows[0][0])
                                time_column_index = 0 
                            except (ValueError, IndexError):
                                pass # First column is not numeric or data is empty
                        
                        # Extract numerical data columns for plotting
                        for j in range(num_cols):
                            if j == time_column_index:
                                continue # Skip time column itself for y-axis
                            try:
                                # Convert entire column to float for plotting
                                col_data = [float(row[j]) for row in data_rows if j < len(row)]
                                plot_data_columns.append({'name': headers[j], 'data': col_data})
                            except (ValueError, IndexError):
                                print(f"Warning: Column '{headers[j]}' could not be converted to numeric for plotting or is incomplete.")
                                pass # Skip non-numeric columns or incomplete columns

                        # Extract time data if time column was identified
                        time_data = None
                        if time_column_index != -1:
                            try:
                                time_data = [float(row[time_column_index]) for row in data_rows if time_column_index < len(row)]
                            except (ValueError, IndexError):
                                time_data = None # Fallback if time column is not numeric
                                print(f"Warning: Time column '{headers[time_column_index]}' could not be converted to numeric or is incomplete.")
                        
                        # If no time data, use simple index for x-axis
                        if time_data is None and plot_data_columns:
                            max_len = max(len(col['data']) for col in plot_data_columns) if plot_data_columns else 0
                            time_data = list(range(max_len))
                            print("Warning: No valid time column found. Using data index for X-axis.")

                        # Update table preview
                        for i in range(rows_to_display):
                            for j in range(num_cols):
                                if j < len(data_rows[i]):
                                    item_value = data_rows[i][j]
                                    try:
                                        item_value_float = float(item_value)
                                        item = QTableWidgetItem(str(item_value_float))
                                    except ValueError:
                                        item = QTableWidgetItem(item_value)
                                    target_table.setItem(i, j, item)
                                else:
                                    target_table.setItem(i, j, QTableWidgetItem(""))
                        target_table.resizeColumnsToContents()

                        # Plotting
                        plot_widget = self.force_plot_widget if data_type == 'force' else self.vibration_plot_widget
                        plot_widget.clear() # Clear previous plots
                        plot_widget.addLegend()
                        if time_data and plot_data_columns:
                            min_len = len(time_data)
                            for col in plot_data_columns:
                                min_len = min(min_len, len(col['data']))
                            
                            for col_idx, col in enumerate(plot_data_columns):
                                # Ensure data and time arrays are of the same length for plotting
                                y_data = col['data'][:min_len]
                                x_data = time_data[:min_len]
                                if x_data and y_data:
                                     # Cycle through some default pen colors
                                    pen_color = pg.intColor(col_idx, hues=len(plot_data_columns))
                                    plot_widget.plot(x_data, y_data, pen=pen_color, name=col['name'])
                        plot_widget.setTitle(f"Gráfico de {data_type.capitalize()} - {file_path.split('/')[-1]}")

                        # Update FFT plots
                        self.update_fft_plot(data_type, headers, data_rows, time_column_index)

                else:
                    error_msg = "Error: El archivo CSV está vacío o no tiene cabeceras."
                    if data_type == 'force':
                        self.force_summary_label.setText("Datos de Fuerza: " + error_msg)
                        self.force_data_table.setRowCount(0)
                        self.force_data_table.setColumnCount(0)
                    elif data_type == 'vibration':
                        self.vibration_summary_label.setText("Datos de Vibración: " + error_msg)
                        self.vibration_data_table.setRowCount(0)
                        self.vibration_data_table.setColumnCount(0)

            except Exception as e:
                error_msg = f"Error al leer el archivo: {e}"
                if data_type == 'force':
                    self.force_summary_label.setText("Datos de Fuerza: " + error_msg)
                    self.force_data_table.setRowCount(0)
                    self.force_data_table.setColumnCount(0)
                elif data_type == 'vibration':
                    self.vibration_summary_label.setText("Datos de Vibración: " + error_msg)
                    self.vibration_data_table.setRowCount(0)
                    self.vibration_data_table.setColumnCount(0)
                print(f"Error loading CSV ({data_type}): {e}\n{traceback.format_exc()}")

    def update_fft_plot(self, data_type, headers, data_rows, time_column_index):
        if not data_rows:
            return

        plot_widget_fft = self.force_fft_plot_widget if data_type == 'force' else self.vibration_fft_plot_widget
        plot_widget_fft.clear()
        plot_widget_fft.addLegend()
        
        num_cols = len(headers)
        N = len(data_rows) # Number of samples
        if N <= 1: # Need at least 2 samples for FFT and sample rate calculation
            plot_widget_fft.setTitle(f"FFT de {data_type.capitalize()} - Datos insuficientes")
            return

        sample_rate = None
        if time_column_index != -1:
            try:
                time_values = [float(row[time_column_index]) for row in data_rows if time_column_index < len(row)]
                if len(time_values) > 1:
                    # Calculate sample rate assuming time is somewhat regular
                    # More robust: check for uniform sampling, but this is a start
                    total_time = time_values[-1] - time_values[0]
                    if total_time > 0:
                        sample_rate = (len(time_values) -1) / total_time
                    else: # Handle case where total_time is zero or negative (e.g. all timestamps are same)
                        sample_rate = 1 # Default to 1 Hz if time difference is zero
                        print(f"Warning: Could not determine valid sample rate from time column for {data_type}. Assuming 1 Hz.") 
            except (ValueError, IndexError):
                print(f"Warning: Could not parse time column for {data_type} to determine sample rate.")
        
        if sample_rate is None: # If still no sample rate (no time col or parse error)
            # Try to infer from the first row's first two time values if available and numeric
            # This is a fallback and might not be robust for all CSV formats.
            # A better approach might be to ask the user or assume a default if not found.
            print(f"Warning: Sample rate for {data_type} not determined from time column. Plotting against sample index.")

            # Fallback: plot FFT vs sample index, not true frequency
            # Or, alternatively, prompt user for sample rate here.
            # For now, we'll proceed but frequencies will be in terms of 1/N, 2/N etc.
            sample_rate = 1 # Default to 1 for now, meaning frequencies are 1/N, 2/N etc.

        plot_widget_fft.setLabel('bottom', 'Frecuencia (Hz)')
        plot_widget_fft.setLabel('left', 'Magnitud')

        for j in range(num_cols):
            if j == time_column_index:
                continue
            try:
                signal_data = np.array([float(row[j]) for row in data_rows if j < len(row)])
                if len(signal_data) < 2 : continue # Not enough data for FFT
                # Perform FFT
                # Using rfft for real-valued input signals
                fft_values = np.fft.rfft(signal_data)
                fft_freq = np.fft.rfftfreq(len(signal_data), d=1.0/sample_rate)
                
                magnitude = np.abs(fft_values)

                # Plot only positive frequencies
                pen_color = pg.intColor(j, hues=num_cols)
                plot_widget_fft.plot(fft_freq, magnitude, pen=pen_color, name=headers[j])
                
            except (ValueError, IndexError) as e:
                print(f"Skipping FFT for column '{headers[j]}' in {data_type} due to error: {e}")
                continue
        
        file_name = "" # Get filename if available, e.g., from self.force_summary_label.text()
        if data_type == 'force' and hasattr(self, 'force_summary_label'):
            # Extract filename from summary label if possible (this is a bit indirect)
            # A better way would be to store filename when loaded.
            summary_parts = self.force_summary_label.text().split("Archivo: ")
            if len(summary_parts) > 1:
                file_name = summary_parts[1].split('\n')[0]
        elif data_type == 'vibration' and hasattr(self, 'vibration_summary_label'):
            summary_parts = self.vibration_summary_label.text().split("Archivo: ")
            if len(summary_parts) > 1:
                file_name = summary_parts[1].split('\n')[0]

        plot_widget_fft.setTitle(f"FFT de {data_type.capitalize()} - {file_name}")


if __name__ == '__main__':
    import traceback # Import traceback for detailed error logging
    app = QApplication(sys.argv)
    # pg.setConfigOptions(antialias=True) # Optional: for smoother graphics
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())

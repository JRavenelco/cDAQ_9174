#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRUEBA DE TIEMPO DE RESPUESTA - MARTILLO DE IMPACTO v2
=======================================================
Versión simplificada y más robusta.

Detecta impactos comparando acelerómetro vs sensor de fuerza.
"""

import sys
import os
import numpy as np
import threading
import queue
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QPushButton,
                             QDoubleSpinBox, QGroupBox, QTableWidget,
                             QTableWidgetItem, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

import pyqtgraph as pg
import pandas as pd

# NI-DAQmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    HAS_NIDAQMX = True
except ImportError:
    HAS_NIDAQMX = False
    print("⚠️ NI-DAQmx no disponible - Modo simulación")

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
FORCE_DEVICE = "cDAQ1Mod1"
FORCE_CHANNEL = "ai0"
ACCEL_DEVICE = "cDAQ1Mod2"
ACCEL_CHANNEL = "ai0"

SAMPLE_RATE = 25600  # Hz - más estable que 51.2k
ACCEL_SENSITIVITY = 100.0  # mV/g

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(SCRIPT_DIR, "pruebas_martillo")
os.makedirs(DATOS_DIR, exist_ok=True)

# Cola global
data_queue = queue.Queue(maxsize=200)


# =============================================================================
# HILO DE ADQUISICIÓN
# =============================================================================
class AcquisitionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.samples_per_read = 256
        
    def run(self):
        if not HAS_NIDAQMX:
            self._simulate()
            return
            
        try:
            # Tarea de fuerza
            self.task_f = nidaqmx.Task()
            self.task_f.ai_channels.add_ai_voltage_chan(
                f"{FORCE_DEVICE}/{FORCE_CHANNEL}",
                min_val=-1.0, max_val=1.0
            )
            self.task_f.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples_per_read * 10
            )
            
            # Tarea de aceleración
            self.task_a = nidaqmx.Task()
            try:
                self.task_a.ai_channels.add_ai_accel_chan(
                    f"{ACCEL_DEVICE}/{ACCEL_CHANNEL}",
                    sensitivity=ACCEL_SENSITIVITY,
                    min_val=-50.0, max_val=50.0,
                    current_excit_val=0.004
                )
            except:
                self.task_a.ai_channels.add_ai_voltage_chan(
                    f"{ACCEL_DEVICE}/{ACCEL_CHANNEL}",
                    min_val=-5.0, max_val=5.0
                )
                
            self.task_a.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples_per_read * 10
            )
            
            self.task_f.start()
            self.task_a.start()
            print(f"✅ DAQ iniciado @ {SAMPLE_RATE} Hz")
            
            while self.running:
                try:
                    f = np.array(self.task_f.read(self.samples_per_read))
                    a = np.array(self.task_a.read(self.samples_per_read))
                    data_queue.put({'force': f, 'accel': a})
                except Exception as e:
                    if self.running:
                        print(f"Error: {e}")
                    break
                    
        except Exception as e:
            print(f"Error DAQ: {e}")
            self._simulate()
        finally:
            self._cleanup()
            
    def _simulate(self):
        """Simulación con impactos periódicos"""
        print("🔄 Modo simulación activo")
        t = 0
        next_impact = 2.0
        
        while self.running:
            dt = self.samples_per_read / SAMPLE_RATE
            
            # Ruido base
            f = np.random.normal(0, 0.002, self.samples_per_read)
            a = np.random.normal(0, 0.1, self.samples_per_read)
            
            # Simular impacto
            if t <= next_impact < t + dt:
                idx = int((next_impact - t) / dt * self.samples_per_read)
                idx = min(idx, self.samples_per_read - 20)
                
                # Impulso en acelerómetro (instantáneo)
                a[idx:idx+3] = [3.0, 5.0, 2.0]
                
                # Respuesta en fuerza (retrasada ~150µs ≈ 4 muestras @ 25.6kHz)
                delay = 4
                if idx + delay + 5 < self.samples_per_read:
                    f[idx+delay:idx+delay+5] = [0.02, 0.06, 0.04, 0.02, 0.01]
                
                next_impact = t + np.random.uniform(3, 6)
            
            t += dt
            data_queue.put({'force': f, 'accel': a})
            threading.Event().wait(dt * 0.8)
            
    def _cleanup(self):
        for task in ['task_f', 'task_a']:
            if hasattr(self, task):
                try:
                    getattr(self, task).stop()
                    getattr(self, task).close()
                except:
                    pass
                    
    def stop(self):
        self.running = False


# =============================================================================
# INTERFAZ
# =============================================================================
class MartilloGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔨 Tiempo de Respuesta - Martillo v2")
        self.setGeometry(100, 100, 1300, 800)
        
        self.acquiring = False
        self.armed = False
        self.thread = None
        
        # Buffers circulares
        self.buffer_size = int(SAMPLE_RATE * 1.0)  # 1 segundo
        self.force_buf = deque(maxlen=self.buffer_size)
        self.accel_buf = deque(maxlen=self.buffer_size)
        
        # Resultados
        self.impactos = []
        
        # Parámetros de detección
        self.umbral_accel = 1.0  # g
        self.umbral_force = 0.01  # V
        self.cooldown = 0  # Evitar detecciones múltiples
        
        self._setup_ui()
        
        # Timer de actualización
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # === CONTROLES ===
        ctrl = QGroupBox("Controles")
        ctrl_lay = QHBoxLayout(ctrl)
        
        self.btn_start = QPushButton("▶️ Iniciar")
        self.btn_start.setStyleSheet("background:#27ae60; color:white; font-weight:bold; padding:8px;")
        self.btn_start.clicked.connect(self._toggle_acq)
        ctrl_lay.addWidget(self.btn_start)
        
        self.btn_arm = QPushButton("🎯 Armar")
        self.btn_arm.setStyleSheet("background:#3498db; color:white; font-weight:bold; padding:8px;")
        self.btn_arm.clicked.connect(self._toggle_arm)
        self.btn_arm.setEnabled(False)
        ctrl_lay.addWidget(self.btn_arm)
        
        ctrl_lay.addWidget(QLabel("Umbral Acel (g):"))
        self.spin_accel = QDoubleSpinBox()
        self.spin_accel.setRange(0.1, 20.0)
        self.spin_accel.setValue(self.umbral_accel)
        self.spin_accel.valueChanged.connect(lambda v: setattr(self, 'umbral_accel', v))
        ctrl_lay.addWidget(self.spin_accel)
        
        ctrl_lay.addWidget(QLabel("Umbral Fuerza (V):"))
        self.spin_force = QDoubleSpinBox()
        self.spin_force.setRange(0.001, 0.5)
        self.spin_force.setValue(self.umbral_force)
        self.spin_force.setDecimals(3)
        self.spin_force.valueChanged.connect(lambda v: setattr(self, 'umbral_force', v))
        ctrl_lay.addWidget(self.spin_force)
        
        btn_clear = QPushButton("🗑️ Limpiar")
        btn_clear.clicked.connect(self._clear)
        ctrl_lay.addWidget(btn_clear)
        
        btn_save = QPushButton("💾 Guardar")
        btn_save.clicked.connect(self._save)
        ctrl_lay.addWidget(btn_save)
        
        ctrl_lay.addStretch()
        layout.addWidget(ctrl)
        
        # === GRÁFICAS Y RESULTADOS ===
        splitter = QSplitter(Qt.Horizontal)
        
        # Gráficas
        graphs = QWidget()
        g_lay = QVBoxLayout(graphs)
        
        # Gráfica tiempo real
        self.plot_rt = pg.PlotWidget(title="Señales en Tiempo Real")
        self.plot_rt.setLabel('bottom', 'Tiempo', 's')
        self.plot_rt.showGrid(True, True, 0.3)
        self.plot_rt.addLegend()
        self.curve_f = self.plot_rt.plot(pen=pg.mkPen('#3498db', width=2), name='Fuerza (×50)')
        self.curve_a = self.plot_rt.plot(pen=pg.mkPen('#e74c3c', width=2), name='Aceleración (g)')
        self.line_th = pg.InfiniteLine(pos=self.umbral_accel, angle=0, 
                                        pen=pg.mkPen('#f39c12', width=1, style=Qt.DashLine))
        self.plot_rt.addItem(self.line_th)
        g_lay.addWidget(self.plot_rt)
        
        # Gráfica impacto
        self.plot_imp = pg.PlotWidget(title="Último Impacto Detectado")
        self.plot_imp.setLabel('bottom', 'Tiempo', 'ms')
        self.plot_imp.showGrid(True, True, 0.3)
        self.plot_imp.addLegend()
        self.curve_if = self.plot_imp.plot(pen=pg.mkPen('#3498db', width=2), name='Fuerza (norm)')
        self.curve_ia = self.plot_imp.plot(pen=pg.mkPen('#e74c3c', width=2), name='Aceleración (norm)')
        self.marker_a = pg.ScatterPlotItem(size=12, brush='#e74c3c')
        self.marker_f = pg.ScatterPlotItem(size=12, brush='#3498db')
        self.plot_imp.addItem(self.marker_a)
        self.plot_imp.addItem(self.marker_f)
        g_lay.addWidget(self.plot_imp)
        
        splitter.addWidget(graphs)
        
        # Panel resultados
        results = QWidget()
        r_lay = QVBoxLayout(results)
        
        # Estadísticas
        stats = QGroupBox("📊 Estadísticas")
        s_lay = QGridLayout(stats)
        
        self.lbl_n = QLabel("Impactos: 0")
        self.lbl_mean = QLabel("Promedio: -- µs")
        self.lbl_std = QLabel("Desv. Est.: -- µs")
        self.lbl_err = QLabel("Error: -- %")
        self.lbl_min = QLabel("Mín: -- µs")
        self.lbl_max = QLabel("Máx: -- µs")
        
        for i, lbl in enumerate([self.lbl_n, self.lbl_mean, self.lbl_std, 
                                  self.lbl_err, self.lbl_min, self.lbl_max]):
            lbl.setFont(QFont("Consolas", 11))
            s_lay.addWidget(lbl, i//2, i%2)
        r_lay.addWidget(stats)
        
        # Tabla
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["#", "Tiempo (µs)", "Acel (g)", "Fuerza (V)"])
        r_lay.addWidget(self.table)
        
        splitter.addWidget(results)
        splitter.setSizes([800, 400])
        layout.addWidget(splitter)
        
        # Status
        self.status = QLabel("Listo")
        self.status.setStyleSheet("padding: 5px; background: #2c3e50; color: white;")
        layout.addWidget(self.status)
        
    def _toggle_acq(self):
        if not self.acquiring:
            # Limpiar
            self.force_buf.clear()
            self.accel_buf.clear()
            while not data_queue.empty():
                data_queue.get()
                
            self.thread = AcquisitionThread()
            self.thread.start()
            self.timer.start(40)  # 25 Hz
            
            self.acquiring = True
            self.btn_start.setText("⏹️ Detener")
            self.btn_start.setStyleSheet("background:#e74c3c; color:white; font-weight:bold; padding:8px;")
            self.btn_arm.setEnabled(True)
            self.status.setText("⏺️ Adquiriendo - Presione Armar para detectar")
        else:
            self.acquiring = False
            self.armed = False
            self.timer.stop()
            if self.thread:
                self.thread.stop()
                self.thread.join(timeout=1)
            self.btn_start.setText("▶️ Iniciar")
            self.btn_start.setStyleSheet("background:#27ae60; color:white; font-weight:bold; padding:8px;")
            self.btn_arm.setEnabled(False)
            self.btn_arm.setText("🎯 Armar")
            self.btn_arm.setStyleSheet("background:#3498db; color:white; font-weight:bold; padding:8px;")
            self.status.setText(f"⏸️ Detenido - {len(self.impactos)} impactos")
            
    def _toggle_arm(self):
        self.armed = not self.armed
        if self.armed:
            self.btn_arm.setText("🔴 ARMADO")
            self.btn_arm.setStyleSheet("background:#e74c3c; color:white; font-weight:bold; padding:8px;")
            self.status.setText("🎯 ARMADO - Golpee con el martillo!")
            self.cooldown = 0
        else:
            self.btn_arm.setText("🎯 Armar")
            self.btn_arm.setStyleSheet("background:#3498db; color:white; font-weight:bold; padding:8px;")
            self.status.setText("⏺️ Adquiriendo")
            
    def _update(self):
        # Leer datos de la cola
        count = 0
        while not data_queue.empty() and count < 20:
            try:
                d = data_queue.get_nowait()
                self.force_buf.extend(d['force'])
                self.accel_buf.extend(d['accel'])
                count += 1
            except:
                break
                
        if len(self.force_buf) < 500:
            return
            
        # Arrays
        f = np.array(self.force_buf)
        a = np.array(self.accel_buf)
        
        # Mostrar últimos 0.3s
        n = min(len(f), int(0.3 * SAMPLE_RATE))
        t = np.linspace(-n/SAMPLE_RATE, 0, n)
        
        self.curve_f.setData(t, f[-n:] * 50)  # Escalar fuerza
        self.curve_a.setData(t, a[-n:])
        self.line_th.setValue(self.umbral_accel)
        
        # Cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            return
            
        # Detectar impacto
        if self.armed:
            self._detect(f, a)
            
    def _detect(self, f, a):
        """Detección simple: buscar pico de aceleración > umbral"""
        # Buscar en últimos datos
        n_search = min(len(a), int(0.1 * SAMPLE_RATE))  # Últimos 100ms
        a_search = a[-n_search:]
        
        # ¿Hay pico sobre umbral?
        max_idx = np.argmax(np.abs(a_search))
        max_val = np.abs(a_search[max_idx])
        
        if max_val < self.umbral_accel:
            return
            
        # Encontramos impacto! Extraer ventana
        # Necesitamos datos antes y después del pico
        pre = int(0.002 * SAMPLE_RATE)   # 2ms antes
        post = int(0.015 * SAMPLE_RATE)  # 15ms después
        
        global_idx = len(a) - n_search + max_idx
        
        if global_idx < pre or global_idx + post > len(a):
            return
            
        # Extraer ventanas
        f_win = f[global_idx - pre : global_idx + post].copy()
        a_win = a[global_idx - pre : global_idx + post].copy()
        
        # Remover DC
        f_win = f_win - np.mean(f_win[:pre])
        a_win = a_win - np.mean(a_win[:pre])
        
        # Encontrar picos
        a_peak = np.argmax(np.abs(a_win))
        
        # Buscar pico de fuerza DESPUÉS del pico de aceleración
        f_after = np.abs(f_win[a_peak:])
        if len(f_after) < 5:
            return
            
        f_peak_rel = np.argmax(f_after)
        f_peak = a_peak + f_peak_rel
        
        # Verificar que fuerza supera umbral
        if np.abs(f_win[f_peak]) < self.umbral_force:
            self.status.setText(f"⚠️ Impacto detectado pero fuerza muy baja ({np.abs(f_win[f_peak])*1000:.1f} mV)")
            self.cooldown = 25  # 1 segundo
            return
            
        # Calcular tiempo de respuesta
        dt_samples = f_peak - a_peak
        dt_us = dt_samples / SAMPLE_RATE * 1e6
        
        # Validar
        if dt_us < 0 or dt_us > 2000:
            self.status.setText(f"⚠️ Tiempo inválido: {dt_us:.0f} µs")
            self.cooldown = 25
            return
            
        # Guardar
        imp = {
            'n': len(self.impactos) + 1,
            'tiempo_us': dt_us,
            'pico_a': max_val,
            'pico_f': np.max(np.abs(f_win)),
            'f_win': f_win,
            'a_win': a_win,
            'a_peak': a_peak,
            'f_peak': f_peak
        }
        self.impactos.append(imp)
        
        # Actualizar UI
        self._show_impact(imp)
        self._update_stats()
        self._update_table()
        
        # Desarmar
        self.armed = False
        self.btn_arm.setText("🎯 Armar")
        self.btn_arm.setStyleSheet("background:#3498db; color:white; font-weight:bold; padding:8px;")
        self.status.setText(f"✅ Impacto #{imp['n']}: {dt_us:.1f} µs (Acel: {max_val:.2f}g, Fuerza: {imp['pico_f']*1000:.1f}mV)")
        
        # Cooldown para evitar re-detección
        self.cooldown = 50  # ~2 segundos
        
        # Limpiar buffers
        self.force_buf.clear()
        self.accel_buf.clear()
        
    def _show_impact(self, imp):
        """Mostrar gráfica del impacto"""
        f = imp['f_win']
        a = imp['a_win']
        
        # Normalizar
        f_n = f / (np.max(np.abs(f)) + 1e-12)
        a_n = a / (np.max(np.abs(a)) + 1e-12)
        
        # Tiempo en ms
        t = np.arange(len(f)) / SAMPLE_RATE * 1000
        t = t - t[int(0.002 * SAMPLE_RATE)]  # Centrar
        
        self.curve_if.setData(t, f_n)
        self.curve_ia.setData(t, a_n)
        
        self.marker_a.setData([t[imp['a_peak']]], [a_n[imp['a_peak']]])
        self.marker_f.setData([t[imp['f_peak']]], [f_n[imp['f_peak']]])
        
    def _update_stats(self):
        n = len(self.impactos)
        self.lbl_n.setText(f"Impactos: {n}")
        
        if n == 0:
            return
            
        ts = [i['tiempo_us'] for i in self.impactos]
        mean = np.mean(ts)
        std = np.std(ts) if n > 1 else 0
        
        self.lbl_mean.setText(f"Promedio: {mean:.1f} µs")
        self.lbl_std.setText(f"Desv. Est.: {std:.1f} µs")
        self.lbl_err.setText(f"Error: {std/mean*100:.1f} %" if mean > 0 else "Error: -- %")
        self.lbl_min.setText(f"Mín: {np.min(ts):.1f} µs")
        self.lbl_max.setText(f"Máx: {np.max(ts):.1f} µs")
        
    def _update_table(self):
        self.table.setRowCount(len(self.impactos))
        for i, imp in enumerate(self.impactos):
            self.table.setItem(i, 0, QTableWidgetItem(str(imp['n'])))
            self.table.setItem(i, 1, QTableWidgetItem(f"{imp['tiempo_us']:.1f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{imp['pico_a']:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{imp['pico_f']*1000:.1f} mV"))
        self.table.scrollToBottom()
        
    def _clear(self):
        self.impactos = []
        self.table.setRowCount(0)
        self._update_stats()
        self.curve_if.setData([], [])
        self.curve_ia.setData([], [])
        self.marker_a.setData([], [])
        self.marker_f.setData([], [])
        self.status.setText("🗑️ Limpiado")
        
    def _save(self):
        if not self.impactos:
            QMessageBox.warning(self, "Sin datos", "No hay impactos")
            return
            
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        df = pd.DataFrame([{
            'n': i['n'],
            'tiempo_us': i['tiempo_us'],
            'pico_accel_g': i['pico_a'],
            'pico_fuerza_mV': i['pico_f'] * 1000
        } for i in self.impactos])
        
        path = os.path.join(DATOS_DIR, f"martillo_{ts}.csv")
        df.to_csv(path, index=False)
        
        # Estadísticas
        tiempos = df['tiempo_us'].values
        stats_txt = f"""PRUEBA MARTILLO - {ts}
{'='*40}
Impactos: {len(tiempos)}
Promedio: {np.mean(tiempos):.2f} µs
Desv. Est.: {np.std(tiempos):.2f} µs
Error Rel.: {np.std(tiempos)/np.mean(tiempos)*100:.2f} %
Mínimo: {np.min(tiempos):.2f} µs
Máximo: {np.max(tiempos):.2f} µs
"""
        stats_path = os.path.join(DATOS_DIR, f"martillo_{ts}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(stats_txt)
            
        QMessageBox.information(self, "Guardado", f"Guardado en:\n{path}")
        self.status.setText(f"💾 Guardado: {os.path.basename(path)}")
        
    def closeEvent(self, event):
        if self.acquiring:
            self._toggle_acq()
        event.accept()


def main():
    print("\n" + "="*50)
    print("🔨 PRUEBA MARTILLO v2")
    print("="*50)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Resolución: {1/SAMPLE_RATE*1e6:.1f} µs")
    print("="*50 + "\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Tema oscuro
    p = app.palette()
    p.setColor(p.Window, QColor(53, 53, 53))
    p.setColor(p.WindowText, QColor(255, 255, 255))
    p.setColor(p.Base, QColor(25, 25, 25))
    p.setColor(p.Text, QColor(255, 255, 255))
    p.setColor(p.Button, QColor(53, 53, 53))
    p.setColor(p.ButtonText, QColor(255, 255, 255))
    p.setColor(p.Highlight, QColor(42, 130, 218))
    app.setPalette(p)
    
    win = MartilloGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

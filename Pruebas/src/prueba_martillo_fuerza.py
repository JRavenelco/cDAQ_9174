#!/usr/bin/env python3
"""
PRUEBA MARTILLO - SOLO SENSOR DE FUERZA
=======================================
Mide tiempo de respuesta del sensor de fuerza ante impactos.
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

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    HAS_NIDAQMX = True
except ImportError:
    HAS_NIDAQMX = False
    print("⚠️ Modo simulación")

# Configuración
DEVICE = "cDAQ1Mod1"  # NI 9205
CHANNEL = "ai0"
SAMPLE_RATE = 10000  # Hz
UMBRAL_DEFAULT = 0.01  # V

DATOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pruebas_martillo")
os.makedirs(DATOS_DIR, exist_ok=True)

data_queue = queue.Queue(maxsize=100)


class ForceThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.samples = 100
        
    def run(self):
        if not HAS_NIDAQMX:
            self._sim()
            return
            
        try:
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(
                f"{DEVICE}/{CHANNEL}",
                min_val=-1.0, max_val=1.0
            )
            self.task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.samples * 10
            )
            self.task.start()
            print(f"✅ Fuerza @ {SAMPLE_RATE} Hz")
            
            while self.running:
                try:
                    d = np.array(self.task.read(self.samples))
                    data_queue.put(d)
                except:
                    break
        except Exception as e:
            print(f"Error: {e}")
            self._sim()
        finally:
            if hasattr(self, 'task'):
                try:
                    self.task.stop()
                    self.task.close()
                except:
                    pass
                    
    def _sim(self):
        print("🔄 Simulación")
        t = 0
        next_hit = 2.0
        while self.running:
            dt = self.samples / SAMPLE_RATE
            d = np.random.normal(0, 0.002, self.samples)
            
            if t <= next_hit < t + dt:
                idx = int((next_hit - t) / dt * self.samples)
                idx = min(idx, self.samples - 10)
                # Impulso típico de martillo
                d[idx:idx+8] = [0.01, 0.05, 0.12, 0.08, 0.04, 0.02, 0.01, 0.005]
                next_hit = t + np.random.uniform(3, 6)
            
            t += dt
            data_queue.put(d)
            threading.Event().wait(dt * 0.9)
            
    def stop(self):
        self.running = False


class MartilloFuerzaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔨 Prueba Martillo - Sensor de Fuerza")
        self.setGeometry(100, 100, 1200, 700)
        
        self.running = False
        self.armed = False
        self.thread = None
        
        self.buf = deque(maxlen=int(SAMPLE_RATE * 0.5))  # 500ms
        self.impactos = []
        self.umbral = UMBRAL_DEFAULT
        self.cooldown = 0
        
        self._ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        
    def _ui(self):
        w = QWidget()
        self.setCentralWidget(w)
        lay = QVBoxLayout(w)
        
        # Controles
        ctrl = QHBoxLayout()
        
        self.btn_start = QPushButton("▶️ Iniciar")
        self.btn_start.setStyleSheet("background:#27ae60;color:white;font-weight:bold;padding:10px;")
        self.btn_start.clicked.connect(self._toggle)
        ctrl.addWidget(self.btn_start)
        
        self.btn_arm = QPushButton("🎯 Armar")
        self.btn_arm.setStyleSheet("background:#3498db;color:white;font-weight:bold;padding:10px;")
        self.btn_arm.clicked.connect(self._arm)
        self.btn_arm.setEnabled(False)
        ctrl.addWidget(self.btn_arm)
        
        ctrl.addWidget(QLabel("Umbral (V):"))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.001, 0.5)
        self.spin.setValue(self.umbral)
        self.spin.setDecimals(3)
        self.spin.setSingleStep(0.005)
        self.spin.valueChanged.connect(lambda v: setattr(self, 'umbral', v))
        ctrl.addWidget(self.spin)
        
        btn_clr = QPushButton("🗑️ Limpiar")
        btn_clr.clicked.connect(self._clear)
        ctrl.addWidget(btn_clr)
        
        btn_save = QPushButton("💾 Guardar")
        btn_save.clicked.connect(self._save)
        ctrl.addWidget(btn_save)
        
        ctrl.addStretch()
        lay.addLayout(ctrl)
        
        # Gráficas + Resultados
        split = QSplitter(Qt.Horizontal)
        
        # Gráficas
        gw = QWidget()
        gl = QVBoxLayout(gw)
        
        self.plot_rt = pg.PlotWidget(title="Señal en Tiempo Real")
        self.plot_rt.setLabel('left', 'Voltaje', 'V')
        self.plot_rt.setLabel('bottom', 'Tiempo', 's')
        self.plot_rt.showGrid(True, True, 0.3)
        self.curve_rt = self.plot_rt.plot(pen=pg.mkPen('#3498db', width=2))
        self.line_th = pg.InfiniteLine(pos=self.umbral, angle=0, 
                                        pen=pg.mkPen('#e74c3c', width=2, style=Qt.DashLine))
        self.plot_rt.addItem(self.line_th)
        gl.addWidget(self.plot_rt)
        
        self.plot_imp = pg.PlotWidget(title="Último Impacto")
        self.plot_imp.setLabel('left', 'Voltaje', 'V')
        self.plot_imp.setLabel('bottom', 'Tiempo', 'ms')
        self.plot_imp.showGrid(True, True, 0.3)
        self.curve_imp = self.plot_imp.plot(pen=pg.mkPen('#2ecc71', width=2))
        self.marker = pg.ScatterPlotItem(size=15, brush='#e74c3c')
        self.plot_imp.addItem(self.marker)
        gl.addWidget(self.plot_imp)
        
        split.addWidget(gw)
        
        # Resultados
        rw = QWidget()
        rl = QVBoxLayout(rw)
        
        stats = QGroupBox("📊 Estadísticas")
        sl = QGridLayout(stats)
        self.lbl_n = QLabel("Impactos: 0")
        self.lbl_mean = QLabel("Promedio: -- µs")
        self.lbl_std = QLabel("Desv: -- µs")
        self.lbl_err = QLabel("Error: -- %")
        self.lbl_peak = QLabel("Último pico: -- mV")
        self.lbl_rise = QLabel("Rise time: -- µs")
        
        for i, l in enumerate([self.lbl_n, self.lbl_mean, self.lbl_std, 
                               self.lbl_err, self.lbl_peak, self.lbl_rise]):
            l.setFont(QFont("Consolas", 11))
            sl.addWidget(l, i//2, i%2)
        rl.addWidget(stats)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["#", "Rise (µs)", "Pico (mV)", "Ancho (µs)"])
        rl.addWidget(self.table)
        
        split.addWidget(rw)
        split.setSizes([700, 400])
        lay.addWidget(split)
        
        self.status = QLabel("Listo - Presione Iniciar")
        self.status.setStyleSheet("background:#34495e;color:white;padding:5px;")
        lay.addWidget(self.status)
        
    def _toggle(self):
        if not self.running:
            self.buf.clear()
            while not data_queue.empty():
                data_queue.get()
            self.thread = ForceThread()
            self.thread.start()
            self.timer.start(50)
            self.running = True
            self.btn_start.setText("⏹️ Detener")
            self.btn_start.setStyleSheet("background:#e74c3c;color:white;font-weight:bold;padding:10px;")
            self.btn_arm.setEnabled(True)
            self.status.setText("⏺️ Adquiriendo")
        else:
            self.running = False
            self.armed = False
            self.timer.stop()
            if self.thread:
                self.thread.stop()
                self.thread.join(timeout=1)
            self.btn_start.setText("▶️ Iniciar")
            self.btn_start.setStyleSheet("background:#27ae60;color:white;font-weight:bold;padding:10px;")
            self.btn_arm.setEnabled(False)
            self.btn_arm.setText("🎯 Armar")
            self.btn_arm.setStyleSheet("background:#3498db;color:white;font-weight:bold;padding:10px;")
            self.status.setText(f"⏸️ Detenido - {len(self.impactos)} impactos")
            
    def _arm(self):
        self.armed = not self.armed
        if self.armed:
            self.btn_arm.setText("🔴 ARMADO")
            self.btn_arm.setStyleSheet("background:#e74c3c;color:white;font-weight:bold;padding:10px;")
            self.status.setText("🎯 ARMADO - ¡Golpee!")
            self.cooldown = 0
        else:
            self.btn_arm.setText("🎯 Armar")
            self.btn_arm.setStyleSheet("background:#3498db;color:white;font-weight:bold;padding:10px;")
            
    def _update(self):
        # Leer cola
        while not data_queue.empty():
            try:
                self.buf.extend(data_queue.get_nowait())
            except:
                break
                
        if len(self.buf) < 200:
            return
            
        d = np.array(self.buf)
        
        # Gráfica tiempo real
        n = min(len(d), int(0.2 * SAMPLE_RATE))
        t = np.linspace(-n/SAMPLE_RATE, 0, n)
        self.curve_rt.setData(t, d[-n:])
        self.line_th.setValue(self.umbral)
        
        if self.cooldown > 0:
            self.cooldown -= 1
            return
            
        if self.armed:
            self._detect(d)
            
    def _detect(self, d):
        # Remover DC
        dc = np.mean(d[:100])
        d_ac = d - dc
        
        # Buscar pico sobre umbral en últimos 50ms
        n_search = min(len(d_ac), int(0.05 * SAMPLE_RATE))
        search = np.abs(d_ac[-n_search:])
        
        max_idx = np.argmax(search)
        max_val = search[max_idx]
        
        if max_val < self.umbral:
            return
            
        # ¡Impacto detectado!
        global_idx = len(d_ac) - n_search + max_idx
        
        # Extraer ventana: 2ms antes, 10ms después
        pre = int(0.002 * SAMPLE_RATE)
        post = int(0.010 * SAMPLE_RATE)
        
        if global_idx < pre or global_idx + post > len(d_ac):
            return
            
        win = d_ac[global_idx - pre : global_idx + post].copy()
        
        # Encontrar pico
        peak_idx = np.argmax(np.abs(win))
        peak_val = win[peak_idx]
        
        # Rise time: desde 10% hasta 90% del pico
        abs_win = np.abs(win)
        th_10 = 0.1 * np.abs(peak_val)
        th_90 = 0.9 * np.abs(peak_val)
        
        # Buscar cruce 10%
        idx_10 = 0
        for i in range(peak_idx):
            if abs_win[i] >= th_10:
                idx_10 = i
                break
                
        # Buscar cruce 90%
        idx_90 = peak_idx
        for i in range(idx_10, peak_idx):
            if abs_win[i] >= th_90:
                idx_90 = i
                break
                
        rise_samples = idx_90 - idx_10
        rise_us = rise_samples / SAMPLE_RATE * 1e6
        
        # Ancho a mitad de altura (FWHM)
        th_50 = 0.5 * np.abs(peak_val)
        idx_start = peak_idx
        idx_end = peak_idx
        for i in range(peak_idx, -1, -1):
            if abs_win[i] < th_50:
                idx_start = i
                break
        for i in range(peak_idx, len(abs_win)):
            if abs_win[i] < th_50:
                idx_end = i
                break
        fwhm_us = (idx_end - idx_start) / SAMPLE_RATE * 1e6
        
        # Guardar
        imp = {
            'n': len(self.impactos) + 1,
            'rise_us': rise_us,
            'peak_mV': peak_val * 1000,
            'fwhm_us': fwhm_us,
            'win': win,
            'peak_idx': peak_idx
        }
        self.impactos.append(imp)
        
        # Mostrar
        self._show(imp)
        self._stats()
        self._table()
        
        # Desarmar
        self.armed = False
        self.btn_arm.setText("🎯 Armar")
        self.btn_arm.setStyleSheet("background:#3498db;color:white;font-weight:bold;padding:10px;")
        self.status.setText(f"✅ #{imp['n']}: Rise={rise_us:.0f}µs, Pico={imp['peak_mV']:.1f}mV")
        
        self.cooldown = 20
        self.buf.clear()
        
    def _show(self, imp):
        win = imp['win']
        t = np.arange(len(win)) / SAMPLE_RATE * 1000
        t = t - t[int(0.002 * SAMPLE_RATE)]
        
        self.curve_imp.setData(t, win * 1000)  # mV
        self.marker.setData([t[imp['peak_idx']]], [win[imp['peak_idx']] * 1000])
        
    def _stats(self):
        n = len(self.impactos)
        self.lbl_n.setText(f"Impactos: {n}")
        
        if n == 0:
            return
            
        rises = [i['rise_us'] for i in self.impactos]
        mean = np.mean(rises)
        std = np.std(rises) if n > 1 else 0
        
        self.lbl_mean.setText(f"Promedio: {mean:.0f} µs")
        self.lbl_std.setText(f"Desv: {std:.0f} µs")
        self.lbl_err.setText(f"Error: {std/mean*100:.1f} %" if mean > 0 else "Error: -- %")
        self.lbl_peak.setText(f"Último pico: {self.impactos[-1]['peak_mV']:.1f} mV")
        self.lbl_rise.setText(f"Rise time: {self.impactos[-1]['rise_us']:.0f} µs")
        
    def _table(self):
        self.table.setRowCount(len(self.impactos))
        for i, imp in enumerate(self.impactos):
            self.table.setItem(i, 0, QTableWidgetItem(str(imp['n'])))
            self.table.setItem(i, 1, QTableWidgetItem(f"{imp['rise_us']:.0f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{imp['peak_mV']:.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{imp['fwhm_us']:.0f}"))
        self.table.scrollToBottom()
        
    def _clear(self):
        self.impactos = []
        self.table.setRowCount(0)
        self._stats()
        self.curve_imp.setData([], [])
        self.marker.setData([], [])
        
    def _save(self):
        if not self.impactos:
            QMessageBox.warning(self, "Sin datos", "No hay impactos")
            return
            
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        df = pd.DataFrame([{
            'n': i['n'],
            'rise_us': i['rise_us'],
            'peak_mV': i['peak_mV'],
            'fwhm_us': i['fwhm_us']
        } for i in self.impactos])
        
        path = os.path.join(DATOS_DIR, f"fuerza_martillo_{ts}.csv")
        df.to_csv(path, index=False)
        
        QMessageBox.information(self, "Guardado", f"Guardado:\n{path}")
        
    def closeEvent(self, e):
        if self.running:
            self._toggle()
        e.accept()


def main():
    print("\n🔨 PRUEBA MARTILLO - SENSOR DE FUERZA")
    print(f"   Device: {DEVICE}/{CHANNEL}")
    print(f"   Sample Rate: {SAMPLE_RATE} Hz\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    p = app.palette()
    p.setColor(p.Window, QColor(53, 53, 53))
    p.setColor(p.WindowText, QColor(255, 255, 255))
    p.setColor(p.Base, QColor(25, 25, 25))
    p.setColor(p.Text, QColor(255, 255, 255))
    p.setColor(p.Button, QColor(53, 53, 53))
    p.setColor(p.ButtonText, QColor(255, 255, 255))
    app.setPalette(p)
    
    win = MartilloFuerzaGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

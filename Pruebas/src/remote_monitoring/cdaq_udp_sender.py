#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cDAQ UDP Sender  –  Corre en WINDOWS con NI-DAQmx instalado.

Adquiere de:
  • NI 9205 (cDAQ1Mod1/ai0)  → fuerza / voltaje analógico
  • NI 9234 (cDAQ1Mod2/ai0)  → acelerómetro IEPE

Envía bloques UDP al Jetson (receiver) usando el protocolo definido
en cdaq_udp_protocol.py.

Uso:
    python cdaq_udp_sender.py --jetson-ip 192.168.137.164
    python cdaq_udp_sender.py --simulate          # sin hardware
"""

import sys
import os
import time
import struct
import socket
import queue
import threading
import argparse
import numpy as np
from collections import deque

# Agregar el directorio padre para importar el protocolo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdaq_udp_protocol import (
    DATA_PORT, CTRL_PORT, INFER_PORT, MAGIC,
    DEFAULT_FS, DEFAULT_SPR, DEFAULT_NCH,
    MSG_SYNC_REQ, MSG_SYNC_RSP, MSG_START, MSG_STOP, MSG_CFG,
    pack_data, pack_sync_rsp, ctrl_msg_type,
    unpack_sync_req, unpack_cfg, unpack_infer,
)

# ─── Intentar importar nidaqmx ──────────────────────────────────
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    HAS_NIDAQMX = True
except ImportError:
    HAS_NIDAQMX = False
    print("[WARN] nidaqmx no disponible – modo simulación forzado")

# ─── PyQt5 GUI (opcional, funciona headless si no está) ─────────
try:
    from PyQt5 import QtWidgets, QtCore
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
        QLabel, QPushButton, QGroupBox, QGridLayout, QDoubleSpinBox,
        QSpinBox, QLineEdit, QTextEdit, QCheckBox, QTabWidget, QSplitter,
    )
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont, QPalette, QColor
    import pyqtgraph as pg
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

# ═══════════════════════════════════════════════════════════════
# Hardware config defaults
# ═══════════════════════════════════════════════════════════════
FORCE_DEVICE   = "cDAQ1Mod1"
FORCE_CHANNEL  = "ai0"
FORCE_FS       = 2500        # Hz
FORCE_MIN_V    = -5.0
FORCE_MAX_V    = 5.0

ACCEL_DEVICE   = "cDAQ1Mod2"
ACCEL_CHANNEL  = "ai0"
ACCEL_FS       = 2500        # Hz
ACCEL_SENS_MV_G = 100.0      # mV/g  (PCB 352C33)

SAMPLES_PER_READ = 100


# ═══════════════════════════════════════════════════════════════
# Acquisition Thread
# ═══════════════════════════════════════════════════════════════
class AcquisitionThread(threading.Thread):
    """Lee NI 9205 (1ch) + NI 9234 (1ch) y pone bloques en data_queue."""

    def __init__(self, data_queue: queue.Queue, simulate: bool = False,
                 fs: int = FORCE_FS, spr: int = SAMPLES_PER_READ,
                 plot_queue=None):
        super().__init__(daemon=True)
        self.q = data_queue
        self.plot_q = plot_queue
        self.simulate = simulate or (not HAS_NIDAQMX)
        self.fs = fs
        self.spr = spr
        self.running = True

    def run(self):
        if self.simulate:
            self._run_sim()
        else:
            self._run_hw()

    # ── Hardware ──────────────────────────────────────────────
    def _run_hw(self):
        force_task = None
        accel_task = None
        try:
            # NI 9205 – voltaje diferencial
            force_task = nidaqmx.Task("force_sender")
            force_task.ai_channels.add_ai_voltage_chan(
                f"{FORCE_DEVICE}/{FORCE_CHANNEL}",
                terminal_config=TerminalConfiguration.DIFF,
                min_val=FORCE_MIN_V, max_val=FORCE_MAX_V,
            )
            force_task.timing.cfg_samp_clk_timing(
                rate=self.fs,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.spr * 10,
            )

            # NI 9234 – IEPE acelerómetro
            accel_task = nidaqmx.Task("accel_sender")
            accel_task.ai_channels.add_ai_accel_chan(
                f"{ACCEL_DEVICE}/{ACCEL_CHANNEL}",
                sensitivity=ACCEL_SENS_MV_G,
                min_val=-50.0, max_val=50.0,
                current_excit_val=0.004,
            )
            accel_task.timing.cfg_samp_clk_timing(
                rate=self.fs,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self.spr * 10,
            )

            force_task.start()
            accel_task.start()
            print(f"[ACQ] NI 9205 ({FORCE_DEVICE}/{FORCE_CHANNEL}) + "
                  f"NI 9234 ({ACCEL_DEVICE}/{ACCEL_CHANNEL}) @ {self.fs} Hz")

            while self.running:
                try:
                    f_data = force_task.read(number_of_samples_per_channel=self.spr)
                    a_data = accel_task.read(number_of_samples_per_channel=self.spr)
                    t_now = time.perf_counter()
                    force = np.asarray(f_data, dtype=np.float32)
                    accel = np.asarray(a_data, dtype=np.float32)
                    if not self.q.full():
                        self.q.put((t_now, force, accel))
                    if self.plot_q and not self.plot_q.full():
                        self.plot_q.put((t_now, force.copy(), accel.copy()))
                except Exception as e:
                    if self.running:
                        print(f"[ACQ] Error lectura: {e}")
                    break

        except Exception as e:
            print(f"[ACQ] Error init HW: {e}")
        finally:
            for t in (force_task, accel_task):
                if t:
                    try:
                        t.stop(); t.close()
                    except Exception:
                        pass

    # ── Simulación ────────────────────────────────────────────
    def _run_sim(self):
        print(f"[ACQ] Simulación @ {self.fs} Hz, {self.spr} samp/bloque")
        t = 0.0
        dt = self.spr / self.fs
        while self.running:
            t_arr = np.linspace(t, t + dt, self.spr, endpoint=False)
            force = (0.5 * np.sin(2 * np.pi * 40 * t_arr)
                     + 0.02 * np.random.randn(self.spr)).astype(np.float32)
            accel = (0.3 * np.sin(2 * np.pi * 40 * t_arr + 0.2)
                     + 0.01 * np.random.randn(self.spr)).astype(np.float32)
            t_now = time.perf_counter()
            if not self.q.full():
                self.q.put((t_now, force, accel))
            if self.plot_q and not self.plot_q.full():
                self.plot_q.put((t_now, force.copy(), accel.copy()))
            t += dt
            time.sleep(dt * 0.9)

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════
# UDP Sender Thread
# ═══════════════════════════════════════════════════════════════
class UDPSenderThread(threading.Thread):
    """Toma bloques de data_queue y los envía por UDP al Jetson."""

    def __init__(self, data_queue: queue.Queue, jetson_ip: str,
                 fs: int = FORCE_FS):
        super().__init__(daemon=True)
        self.q = data_queue
        self.jetson_ip = jetson_ip
        self.fs = fs
        self.running = True
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Estadísticas
        self.packets_sent = 0
        self.bytes_sent = 0

    def run(self):
        print(f"[UDP-TX] Enviando a {self.jetson_ip}:{DATA_PORT}")
        while self.running:
            try:
                t_now, force, accel = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            pkt = pack_data(self.seq, t_now, float(self.fs), force, accel)
            try:
                self.sock.sendto(pkt, (self.jetson_ip, DATA_PORT))
                self.packets_sent += 1
                self.bytes_sent += len(pkt)
                self.seq += 1
            except Exception as e:
                print(f"[UDP-TX] Error: {e}")

    def stop(self):
        self.running = False
        self.sock.close()


# ═══════════════════════════════════════════════════════════════
# Control Listener Thread  (escucha comandos del Jetson)
# ═══════════════════════════════════════════════════════════════
class ControlListenerThread(threading.Thread):
    """Escucha en CTRL_PORT comandos START/STOP/SYNC/CFG del Jetson."""

    def __init__(self, on_start, on_stop, on_cfg):
        super().__init__(daemon=True)
        self.running = True
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_cfg = on_cfg
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", CTRL_PORT))
        self.sock.settimeout(0.5)

    def run(self):
        print(f"[CTRL] Escuchando en :{CTRL_PORT}")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break

            mtype = ctrl_msg_type(data)
            if mtype == MSG_SYNC_REQ:
                t1 = unpack_sync_req(data)
                if t1 is not None:
                    t2 = time.perf_counter()
                    rsp = pack_sync_rsp(t1, t2, time.perf_counter())
                    self.sock.sendto(rsp, addr)
            elif mtype == MSG_START:
                print(f"[CTRL] START recibido de {addr}")
                self.on_start()
            elif mtype == MSG_STOP:
                print(f"[CTRL] STOP recibido de {addr}")
                self.on_stop()
            elif mtype == MSG_CFG:
                result = unpack_cfg(data)
                if result:
                    fs, spr = result
                    print(f"[CTRL] CFG recibido: fs={fs}, spr={spr}")
                    self.on_cfg(fs, spr)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Inference Listener Thread  (recibe resultados de inferencia del Jetson)
# ═══════════════════════════════════════════════════════════════
class InferenceListenerThread(threading.Thread):
    """Escucha en INFER_PORT resultados de inferencia del Jetson."""

    def __init__(self, on_infer=None):
        super().__init__(daemon=True)
        self.running = True
        self.on_infer = on_infer   # callback(seq, t_s, values)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", INFER_PORT))
        self.sock.settimeout(0.5)
        # Últimos resultados (accesibles desde GUI)
        self.last_seq = -1
        self.last_values = None
        self.count = 0

    def run(self):
        print(f"[INFER-RX] Escuchando inferencia en :{INFER_PORT}")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            result = unpack_infer(data)
            if result is None:
                continue
            seq, t_s, values = result
            self.last_seq = seq
            self.last_values = values
            self.count += 1
            if self.on_infer:
                self.on_infer(seq, t_s, values)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# GUI (PyQt5 + PyQtGraph)
# ═══════════════════════════════════════════════════════════════
if HAS_GUI:
    class SenderWindow(QMainWindow):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.setWindowTitle("cDAQ UDP Sender + Monitor")
            self.setGeometry(100, 100, 1400, 900)

            self.data_queue = queue.Queue(maxsize=50)
            self.plot_queue = queue.Queue(maxsize=200)
            self.acq_thread = None
            self.udp_thread = None
            self.ctrl_thread = None
            self.infer_thread = None
            self.streaming = False

            # Plot buffers (5 seconds)
            self._buf_maxlen = int(FORCE_FS * 5)
            self.force_buf = np.zeros(0, dtype=np.float32)
            self.accel_buf = np.zeros(0, dtype=np.float32)
            self._fft_counter = 0
            self._max_plot_points = 2000  # diezmar a esto para dibujar

            self._apply_dark_theme()
            self._build_ui()
            self._start_ctrl_listener()
            self._start_infer_listener()

            # Plot timer (~30 FPS)
            self.plot_timer = QTimer()
            self.plot_timer.timeout.connect(self._update_plots)

            # Stats timer
            self.stats_timer = QTimer()
            self.stats_timer.timeout.connect(self._update_stats)
            self.stats_timer.start(500)

        # ── Tema oscuro (estilo caracterizacion_sensor_fuerza) ────
        def _apply_dark_theme(self):
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(53, 53, 53))
            palette.setColor(QPalette.WindowText, Qt.white)
            palette.setColor(QPalette.Base, QColor(35, 35, 35))
            palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            palette.setColor(QPalette.Text, Qt.white)
            palette.setColor(QPalette.Button, QColor(53, 53, 53))
            palette.setColor(QPalette.ButtonText, Qt.white)
            palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            self.setPalette(palette)
            pg.setConfigOption('background', '#2b2b2b')
            pg.setConfigOption('foreground', 'w')

        # ── UI ────────────────────────────────────────────────────
        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QVBoxLayout(central)

            # ── Config (compacta, horizontal) ──
            cfg_box = QGroupBox("Configuración")
            cfg_grid = QGridLayout(cfg_box)
            cfg_grid.addWidget(QLabel("Jetson IP:"), 0, 0)
            self.ip_edit = QLineEdit(self.args.jetson_ip)
            cfg_grid.addWidget(self.ip_edit, 0, 1)
            cfg_grid.addWidget(QLabel("Fs (Hz):"), 0, 2)
            self.fs_spin = QSpinBox()
            self.fs_spin.setRange(100, 51200)
            self.fs_spin.setValue(self.args.fs)
            cfg_grid.addWidget(self.fs_spin, 0, 3)
            cfg_grid.addWidget(QLabel("SPR:"), 0, 4)
            self.spr_spin = QSpinBox()
            self.spr_spin.setRange(10, 5000)
            self.spr_spin.setValue(self.args.spr)
            cfg_grid.addWidget(self.spr_spin, 0, 5)
            self.sim_check = QCheckBox("Simulación")
            self.sim_check.setChecked(self.args.simulate)
            cfg_grid.addWidget(self.sim_check, 0, 6)
            self.btn_start = QPushButton("▶ Iniciar Streaming")
            self.btn_start.setStyleSheet(
                "background-color: #27ae60; color: white; "
                "font-weight: bold; padding: 8px 16px;")
            self.btn_start.clicked.connect(self._on_start)
            cfg_grid.addWidget(self.btn_start, 0, 7)
            self.btn_stop = QPushButton("⏹ Detener")
            self.btn_stop.setStyleSheet(
                "background-color: #e74c3c; color: white; "
                "font-weight: bold; padding: 8px 16px;")
            self.btn_stop.clicked.connect(self._on_stop)
            self.btn_stop.setEnabled(False)
            cfg_grid.addWidget(self.btn_stop, 0, 8)
            main_layout.addWidget(cfg_box)

            # ── Mediciones en tiempo real ──
            info_box = QGroupBox("Mediciones en Tiempo Real")
            info_grid = QGridLayout(info_box)
            lbl_font = QFont("Consolas", 10)

            info_grid.addWidget(QLabel("⚡ FUERZA:"), 0, 0)
            self.force_val_label = QLabel("0.0000 V")
            self.force_val_label.setFont(lbl_font)
            info_grid.addWidget(self.force_val_label, 0, 1)
            self.force_rms_label = QLabel("RMS: 0.0000 V")
            self.force_rms_label.setFont(lbl_font)
            info_grid.addWidget(self.force_rms_label, 0, 2)
            self.force_pp_label = QLabel("Pk-Pk: 0.0000 V")
            self.force_pp_label.setFont(lbl_font)
            info_grid.addWidget(self.force_pp_label, 0, 3)

            info_grid.addWidget(QLabel("📡 ACELERACIÓN:"), 0, 4)
            self.accel_val_label = QLabel("0.0000 g")
            self.accel_val_label.setFont(lbl_font)
            info_grid.addWidget(self.accel_val_label, 0, 5)
            self.accel_rms_label = QLabel("RMS: 0.0000 g")
            self.accel_rms_label.setFont(lbl_font)
            info_grid.addWidget(self.accel_rms_label, 0, 6)
            self.accel_pp_label = QLabel("Pk-Pk: 0.0000 g")
            self.accel_pp_label.setFont(lbl_font)
            info_grid.addWidget(self.accel_pp_label, 0, 7)

            info_grid.addWidget(QLabel("🧠 INFERENCIA (Jetson):"), 1, 0)
            self.infer_label = QLabel("Sin datos de inferencia")
            self.infer_label.setFont(lbl_font)
            self.infer_label.setStyleSheet("color: #2ecc71; font-weight: bold;")
            self.infer_label.setWordWrap(True)
            info_grid.addWidget(self.infer_label, 1, 1, 1, 7)
            main_layout.addWidget(info_box)

            # ── Tabs ──
            self.tabs = QTabWidget()
            self.tabs.setStyleSheet("""
                QTabBar::tab {
                    background: #3a3a3a; color: white;
                    padding: 8px 20px; font-weight: bold;
                }
                QTabBar::tab:selected { background: #2980b9; }
            """)
            self._setup_signals_tab()
            self._setup_fft_tab()
            self._setup_log_tab()
            main_layout.addWidget(self.tabs)

            # ── Status bar ──
            self.status_label = QLabel("Estado: Detenido")
            self.status_label.setFont(QFont("Consolas", 10))
            self.statusBar().addWidget(self.status_label)

        # ── Tab: Señales en tiempo ────────────────────────────────
        def _setup_signals_tab(self):
            tab = QWidget()
            layout = QVBoxLayout(tab)
            splitter = QSplitter(Qt.Vertical)

            pg.setConfigOptions(antialias=False)

            self.force_plot = pg.PlotWidget(
                title="⚡ Fuerza / Voltaje (NI 9205)")
            self.force_plot.setLabel('left', 'Voltaje', 'V')
            self.force_plot.setLabel('bottom', 'Tiempo', 's')
            self.force_plot.showGrid(x=True, y=True, alpha=0.3)
            self.force_curve = self.force_plot.plot(
                pen=pg.mkPen('#3498db', width=2))
            splitter.addWidget(self.force_plot)

            self.accel_plot = pg.PlotWidget(
                title="📡 Aceleración (NI 9234)")
            self.accel_plot.setLabel('left', 'Aceleración', 'g')
            self.accel_plot.setLabel('bottom', 'Tiempo', 's')
            self.accel_plot.showGrid(x=True, y=True, alpha=0.3)
            self.accel_curve = self.accel_plot.plot(
                pen=pg.mkPen('#e74c3c', width=2))
            splitter.addWidget(self.accel_plot)

            # Superposición normalizada
            self.overlay_plot = pg.PlotWidget(
                title="🔀 Superposición Normalizada")
            self.overlay_plot.setLabel('left', 'Amplitud', 'norm')
            self.overlay_plot.setLabel('bottom', 'Tiempo', 's')
            self.overlay_plot.showGrid(x=True, y=True, alpha=0.3)
            self.overlay_force_curve = self.overlay_plot.plot(
                pen=pg.mkPen('#3498db', width=2), name='Fuerza')
            self.overlay_accel_curve = self.overlay_plot.plot(
                pen=pg.mkPen('#e74c3c', width=2), name='Aceleración')
            self.overlay_plot.addLegend()
            splitter.addWidget(self.overlay_plot)

            layout.addWidget(splitter)
            self.tabs.addTab(tab, "📈 Señales")

        # ── Tab: FFT ──────────────────────────────────────────────
        def _setup_fft_tab(self):
            tab = QWidget()
            layout = QVBoxLayout(tab)
            splitter = QSplitter(Qt.Vertical)

            self.fft_force_plot = pg.PlotWidget(title="FFT Fuerza")
            self.fft_force_plot.setLabel('left', 'Magnitud')
            self.fft_force_plot.setLabel('bottom', 'Frecuencia', 'Hz')
            self.fft_force_plot.showGrid(x=True, y=True, alpha=0.3)
            self.fft_force_curve = self.fft_force_plot.plot(
                pen=pg.mkPen('#3498db', width=2))
            self.fft_force_peak = pg.TextItem(anchor=(0, 1), color='#3498db')
            self.fft_force_peak.setFont(QFont('Arial', 11, QFont.Bold))
            self.fft_force_plot.addItem(self.fft_force_peak)
            splitter.addWidget(self.fft_force_plot)

            self.fft_accel_plot = pg.PlotWidget(title="FFT Aceleración")
            self.fft_accel_plot.setLabel('left', 'Magnitud')
            self.fft_accel_plot.setLabel('bottom', 'Frecuencia', 'Hz')
            self.fft_accel_plot.showGrid(x=True, y=True, alpha=0.3)
            self.fft_accel_curve = self.fft_accel_plot.plot(
                pen=pg.mkPen('#e74c3c', width=2))
            self.fft_accel_peak = pg.TextItem(anchor=(0, 1), color='#e74c3c')
            self.fft_accel_peak.setFont(QFont('Arial', 11, QFont.Bold))
            self.fft_accel_plot.addItem(self.fft_accel_peak)
            splitter.addWidget(self.fft_accel_plot)

            layout.addWidget(splitter)
            self.tabs.addTab(tab, "📊 FFT")

        # ── Tab: Log ──────────────────────────────────────────────
        def _setup_log_tab(self):
            tab = QWidget()
            layout = QVBoxLayout(tab)
            self.log_text = QTextEdit()
            self.log_text.setReadOnly(True)
            self.log_text.setFont(QFont("Consolas", 9))
            layout.addWidget(self.log_text)
            self.tabs.addTab(tab, "📋 Log")

        def _log(self, msg: str):
            self.log_text.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        # ── Listeners ─────────────────────────────────────────────
        def _start_ctrl_listener(self):
            self.ctrl_thread = ControlListenerThread(
                on_start=self._on_start_remote,
                on_stop=self._on_stop_remote,
                on_cfg=self._on_cfg_remote,
            )
            self.ctrl_thread.start()
            self._log("Control listener activo")

        def _start_infer_listener(self):
            self.infer_thread = InferenceListenerThread()
            self.infer_thread.start()
            self._log("Inference listener activo")

        # ── Callbacks de control remoto ──
        def _on_start_remote(self):
            QtCore.QMetaObject.invokeMethod(
                self, "_on_start", QtCore.Qt.QueuedConnection)

        def _on_stop_remote(self):
            QtCore.QMetaObject.invokeMethod(
                self, "_on_stop", QtCore.Qt.QueuedConnection)

        def _on_cfg_remote(self, fs, spr):
            self.fs_spin.setValue(fs)
            self.spr_spin.setValue(spr)
            self._log(f"Config remota: fs={fs}, spr={spr}")

        # ── Start / Stop ──────────────────────────────────────────
        @QtCore.pyqtSlot()
        def _on_start(self):
            if self.streaming:
                return
            fs = self.fs_spin.value()
            spr = self.spr_spin.value()
            sim = self.sim_check.isChecked()
            jetson_ip = self.ip_edit.text().strip()

            # Limpiar buffers y colas
            self.force_buf = np.zeros(0, dtype=np.float32)
            self.accel_buf = np.zeros(0, dtype=np.float32)
            for q in (self.data_queue, self.plot_queue):
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

            self.acq_thread = AcquisitionThread(
                self.data_queue, simulate=sim, fs=fs, spr=spr,
                plot_queue=self.plot_queue)
            self.udp_thread = UDPSenderThread(
                self.data_queue, jetson_ip, fs=fs)
            self.acq_thread.start()
            self.udp_thread.start()
            self.streaming = True
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.plot_timer.start(33)   # ~30 FPS
            self._log(f"Streaming → {jetson_ip}:{DATA_PORT}  "
                      f"fs={fs} spr={spr} sim={sim}")

        @QtCore.pyqtSlot()
        def _on_stop(self):
            if not self.streaming:
                return
            if self.acq_thread:
                self.acq_thread.stop()
            if self.udp_thread:
                self.udp_thread.stop()
            self.streaming = False
            self.plot_timer.stop()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._log("Streaming detenido")

        # ── Actualización de gráficas (~30 FPS) ──────────────────
        def _update_plots(self):
            # Drenar plot_queue → buffers numpy (sin .tolist())
            chunks_f = []
            chunks_a = []
            drained = 0
            while not self.plot_queue.empty():
                try:
                    t_now, force, accel = self.plot_queue.get_nowait()
                    chunks_f.append(force)
                    chunks_a.append(accel)
                    drained += 1
                except queue.Empty:
                    break

            if drained == 0 and len(self.force_buf) == 0:
                return

            # Concatenar nuevos chunks y recortar al máximo del buffer
            if drained > 0:
                new_f = np.concatenate(chunks_f)
                new_a = np.concatenate(chunks_a)
                self.force_buf = np.concatenate(
                    [self.force_buf, new_f])[-self._buf_maxlen:]
                self.accel_buf = np.concatenate(
                    [self.accel_buf, new_a])[-self._buf_maxlen:]
            elif len(self.force_buf) > 0:
                # Sin datos nuevos → no redibujar
                return

            force_arr = self.force_buf
            accel_arr = self.accel_buf
            n = min(len(force_arr), len(accel_arr))
            if n < 2:
                return

            force_arr = force_arr[-n:]
            accel_arr = accel_arr[-n:]
            fs = self.fs_spin.value()

            # Diezmar para dibujar (máx ~2000 puntos por curva)
            if n > self._max_plot_points:
                step = n // self._max_plot_points
                f_plot = force_arr[::step]
                a_plot = accel_arr[::step]
                t_plot = np.linspace(-n / fs, 0, len(f_plot))
            else:
                f_plot = force_arr
                a_plot = accel_arr
                t_plot = np.linspace(-n / fs, 0, n)

            # Series de tiempo
            self.force_curve.setData(t_plot, f_plot)
            self.accel_curve.setData(t_plot, a_plot)

            # Superposición normalizada (usar solo últimos 1000 pts para stats)
            tail = min(n, 1000)
            f_tail = force_arr[-tail:]
            a_tail = accel_arr[-tail:]
            f_std = np.std(f_tail)
            a_std = np.std(a_tail)
            if f_std > 1e-6 and a_std > 1e-6:
                f_norm = (f_plot - np.mean(f_tail)) / f_std
                a_norm = (a_plot - np.mean(a_tail)) / a_std
                self.overlay_force_curve.setData(t_plot, f_norm)
                self.overlay_accel_curve.setData(t_plot, a_norm)

            # Info labels (stats sobre la cola reciente, no todo el buffer)
            self.force_val_label.setText(f"{force_arr[-1]:.4f} V")
            self.force_rms_label.setText(
                f"RMS: {np.sqrt(np.mean(f_tail**2)):.4f} V")
            self.force_pp_label.setText(
                f"Pk-Pk: {np.ptp(f_tail):.4f} V")

            self.accel_val_label.setText(f"{accel_arr[-1]:.4f} g")
            self.accel_rms_label.setText(
                f"RMS: {np.sqrt(np.mean(a_tail**2)):.4f} g")
            self.accel_pp_label.setText(
                f"Pk-Pk: {np.ptp(a_tail):.4f} g")

            # FFT (cada 5 frames para no saturar CPU)
            self._fft_counter += 1
            if self._fft_counter % 5 == 0 and n >= 256:
                self._update_fft(force_arr, accel_arr, fs)

        def _update_fft(self, force_arr, accel_arr, fs):
            n_fft = min(len(force_arr), 4096)
            window = np.hanning(n_fft)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)

            # Fuerza
            f_fft = force_arr[-n_fft:] * window
            F_mag = np.abs(np.fft.rfft(f_fft)) * 2.0 / n_fft
            # Saltar DC (índice 0)
            self.fft_force_curve.setData(freqs[1:], F_mag[1:])
            if len(F_mag) > 1:
                pk = np.argmax(F_mag[1:]) + 1
                self.fft_force_peak.setText(
                    f"Peak: {freqs[pk]:.1f} Hz ({F_mag[pk]:.4f})")
                self.fft_force_peak.setPos(freqs[pk], F_mag[pk])

            # Aceleración
            a_fft = accel_arr[-n_fft:] * window
            A_mag = np.abs(np.fft.rfft(a_fft)) * 2.0 / n_fft
            self.fft_accel_curve.setData(freqs[1:], A_mag[1:])
            if len(A_mag) > 1:
                pk = np.argmax(A_mag[1:]) + 1
                self.fft_accel_peak.setText(
                    f"Peak: {freqs[pk]:.1f} Hz ({A_mag[pk]:.4f})")
                self.fft_accel_peak.setPos(freqs[pk], A_mag[pk])

        # ── Estadísticas + inferencia ─────────────────────────────
        def _update_stats(self):
            if self.streaming and self.udp_thread:
                pkts = self.udp_thread.packets_sent
                kb = self.udp_thread.bytes_sent / 1024
                infer_n = (self.infer_thread.count
                           if self.infer_thread else 0)
                self.status_label.setText(
                    f"Streaming  |  Pkts: {pkts}  |  "
                    f"TX: {kb:.1f} KB  |  Infer: {infer_n}")
            elif not self.streaming:
                self.status_label.setText("Detenido")

            # Inferencia del Jetson
            if (self.infer_thread
                    and self.infer_thread.last_values is not None):
                v = self.infer_thread.last_values
                seq = self.infer_thread.last_seq
                if len(v) >= 12:
                    # Envelope engine (Nivel 1)
                    cut = "🔴 CORTE" if v[7] > 0.5 else "⚪ reposo"
                    txt = (f"Seq:{seq}  {cut}  "
                           f"F_est={v[0]:.4f}V(pk:{v[1]:.4f})  "
                           f"Env={v[2]:.4f}g  "
                           f"f₀={v[3]:.1f}Hz  "
                           f"THD_a={v[5]:.1f}% THD_f={v[6]:.1f}%")
                elif len(v) >= 6:
                    txt = (f"Seq:{seq}  "
                           f"F: mean={v[0]:.4f} std={v[1]:.4f} "
                           f"rms={v[2]:.4f}  |  "
                           f"A: mean={v[3]:.4f} std={v[4]:.4f} "
                           f"rms={v[5]:.4f}")
                else:
                    txt = f"Seq:{seq}  {v}"
                self.infer_label.setText(txt)

        # ── Cierre ────────────────────────────────────────────────
        def closeEvent(self, event):
            self._on_stop()
            if self.ctrl_thread:
                self.ctrl_thread.stop()
            if self.infer_thread:
                self.infer_thread.stop()
            event.accept()


# ═══════════════════════════════════════════════════════════════
# Headless mode  (sin GUI)
# ═══════════════════════════════════════════════════════════════
def run_headless(args):
    """Ejecuta el sender sin GUI (útil en scripts o SSH)."""
    data_queue = queue.Queue(maxsize=50)
    streaming = False
    acq_thread = None
    udp_thread = None

    def start_streaming():
        nonlocal streaming, acq_thread, udp_thread
        if streaming:
            return
        while not data_queue.empty():
            try:
                data_queue.get_nowait()
            except queue.Empty:
                break
        acq_thread = AcquisitionThread(data_queue, simulate=args.simulate,
                                       fs=args.fs, spr=args.spr)
        udp_thread = UDPSenderThread(data_queue, args.jetson_ip, fs=args.fs)
        acq_thread.start()
        udp_thread.start()
        streaming = True
        print("[HEADLESS] Streaming iniciado")

    def stop_streaming():
        nonlocal streaming, acq_thread, udp_thread
        if not streaming:
            return
        if acq_thread:
            acq_thread.stop()
        if udp_thread:
            udp_thread.stop()
        streaming = False
        print("[HEADLESS] Streaming detenido")

    def on_cfg(fs, spr):
        args.fs = fs
        args.spr = spr
        print(f"[HEADLESS] Config actualizada: fs={fs}, spr={spr}")

    ctrl = ControlListenerThread(
        on_start=start_streaming, on_stop=stop_streaming, on_cfg=on_cfg)
    ctrl.start()

    if args.auto_start:
        start_streaming()

    print("[HEADLESS] Presiona Ctrl+C para salir. "
          "Esperando comandos en CTRL_PORT…")
    try:
        while True:
            time.sleep(1)
            if streaming and udp_thread:
                print(f"  pkts={udp_thread.packets_sent}  "
                      f"tx={udp_thread.bytes_sent/1024:.1f}KB", end="\r")
    except KeyboardInterrupt:
        print("\n[HEADLESS] Saliendo…")
        stop_streaming()
        ctrl.stop()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="cDAQ UDP Sender (Windows)")
    ap.add_argument("--jetson-ip", default="192.168.137.164",
                    help="IP del Jetson receiver")
    ap.add_argument("--fs", type=int, default=FORCE_FS,
                    help="Sample rate (Hz)")
    ap.add_argument("--spr", type=int, default=SAMPLES_PER_READ,
                    help="Samples per read / bloque")
    ap.add_argument("--simulate", action="store_true",
                    help="Modo simulación sin hardware NI")
    ap.add_argument("--headless", action="store_true",
                    help="Sin GUI")
    ap.add_argument("--auto-start", action="store_true",
                    help="Iniciar streaming automáticamente (headless)")
    args = ap.parse_args()

    if args.headless or not HAS_GUI:
        run_headless(args)
    else:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        win = SenderWindow(args)
        win.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()

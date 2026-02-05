#!/usr/bin/env python3
"""
GUI para caracterización de fuerza magnética con gráficas en tiempo real.
- Envía u(t) senoidal en lazo abierto al controlador C++
- Adquiere fuerza por AD1 (libdwf)
- Recibe corriente/voltaje por UDP
- Gráficas: Fuerza vs tiempo, Fuerza vs corriente (fase)
"""
import sys
import os
import time
import queue
import threading
import socket
import struct
import subprocess
import csv
from collections import deque
from dataclasses import dataclass

import numpy as np
import ctypes


if "--headless" in sys.argv:
    sys.argv.remove("--headless")
    from levitador_caracterizacion_headless import main as headless_main

    headless_main()
    raise SystemExit(0)


import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


UDP_IP = "0.0.0.0"
UDP_PORT = 9000


class DwfError(RuntimeError):
    pass


class Dwf:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary("libdwf.so")
        self._bind()

    def _bind(self):
        self.lib.FDwfGetLastErrorMsg.argtypes = [ctypes.c_char * 512]
        self.lib.FDwfGetLastErrorMsg.restype = ctypes.c_int
        self.lib.FDwfEnum.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.FDwfEnum.restype = ctypes.c_int
        self.lib.FDwfEnumDeviceName.argtypes = [ctypes.c_int, ctypes.c_char * 32]
        self.lib.FDwfEnumDeviceName.restype = ctypes.c_int
        self.lib.FDwfEnumSN.argtypes = [ctypes.c_int, ctypes.c_char * 32]
        self.lib.FDwfEnumSN.restype = ctypes.c_int
        self.lib.FDwfEnumDeviceIsOpened.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.FDwfEnumDeviceIsOpened.restype = ctypes.c_int
        self.lib.FDwfDeviceOpen.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self.lib.FDwfDeviceOpen.restype = ctypes.c_int
        self.lib.FDwfDeviceClose.argtypes = [ctypes.c_int]
        self.lib.FDwfDeviceClose.restype = ctypes.c_int
        self.lib.FDwfDeviceAutoConfigureSet.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.FDwfDeviceAutoConfigureSet.restype = ctypes.c_int
        self.lib.FDwfAnalogInReset.argtypes = [ctypes.c_int]
        self.lib.FDwfAnalogInReset.restype = ctypes.c_int
        self.lib.FDwfAnalogInAcquisitionModeSet.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.FDwfAnalogInAcquisitionModeSet.restype = ctypes.c_int
        self.lib.FDwfAnalogInFrequencySet.argtypes = [ctypes.c_int, ctypes.c_double]
        self.lib.FDwfAnalogInFrequencySet.restype = ctypes.c_int
        self.lib.FDwfAnalogInChannelEnableSet.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.FDwfAnalogInChannelEnableSet.restype = ctypes.c_int
        self.lib.FDwfAnalogInChannelRangeSet.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.lib.FDwfAnalogInChannelRangeSet.restype = ctypes.c_int
        self.lib.FDwfAnalogInChannelOffsetSet.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.lib.FDwfAnalogInChannelOffsetSet.restype = ctypes.c_int
        self.lib.FDwfAnalogInConfigure.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.FDwfAnalogInConfigure.restype = ctypes.c_int
        self.lib.FDwfAnalogInStatus.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte)]
        self.lib.FDwfAnalogInStatus.restype = ctypes.c_int
        self.lib.FDwfAnalogInStatusRecord.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.FDwfAnalogInStatusRecord.restype = ctypes.c_int
        self.lib.FDwfAnalogInRecordLengthSet.argtypes = [ctypes.c_int, ctypes.c_double]
        self.lib.FDwfAnalogInRecordLengthSet.restype = ctypes.c_int
        self.lib.FDwfAnalogInStatusData.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ]
        self.lib.FDwfAnalogInStatusData.restype = ctypes.c_int

    def _last_error(self) -> str:
        buf = (ctypes.c_char * 512)()
        self.lib.FDwfGetLastErrorMsg(buf)
        return buf.value.decode("utf-8", errors="replace")

    def _check(self, ok: int, where: str):
        if ok == 0:
            raise DwfError(f"{where}: {self._last_error()}")

    def enum_devices(self):
        c = ctypes.c_int(0)
        self._check(self.lib.FDwfEnum(0, ctypes.byref(c)), "FDwfEnum")
        devices = []
        for idx in range(c.value):
            name = (ctypes.c_char * 32)()
            sn = (ctypes.c_char * 32)()
            opened = ctypes.c_int(0)
            self._check(self.lib.FDwfEnumDeviceName(idx, name), "FDwfEnumDeviceName")
            self._check(self.lib.FDwfEnumSN(idx, sn), "FDwfEnumSN")
            self._check(self.lib.FDwfEnumDeviceIsOpened(idx, ctypes.byref(opened)), "FDwfEnumDeviceIsOpened")
            devices.append({
                "index": idx,
                "name": name.value.decode("utf-8", errors="replace"),
                "sn": sn.value.decode("utf-8", errors="replace"),
                "opened": bool(opened.value),
            })
        return devices

    def open(self, idx: int) -> int:
        c = ctypes.c_int(0)
        self._check(self.lib.FDwfEnum(0, ctypes.byref(c)), "FDwfEnum")
        if idx >= c.value:
            raise DwfError(f"Device index {idx} out of range (found {c.value} devices)")
        hdwf = ctypes.c_int(0)
        self._check(self.lib.FDwfDeviceOpen(idx, ctypes.byref(hdwf)), "FDwfDeviceOpen")
        self._check(self.lib.FDwfDeviceAutoConfigureSet(hdwf.value, 0), "FDwfDeviceAutoConfigureSet")
        return hdwf.value

    def close(self, hdwf: int):
        if hdwf:
            self.lib.FDwfDeviceClose(hdwf)


@dataclass
class Ad1Config:
    device_index: int = 0
    channel: int = 1
    sample_rate_hz: float = 2500.0
    volts_range: float = 5.0
    volts_offset: float = 0.0


class TelemetryThread(QThread):
    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.settimeout(0.1)

    def run(self):
        packet_fmt = "<8f1i1f"
        packet_size = struct.calcsize(packet_fmt)
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                if len(data) != packet_size:
                    continue
                unpacked = struct.unpack(packet_fmt, data)
                telemetry = {
                    "t": unpacked[0],
                    "y_ref": unpacked[1],
                    "y_sensor": unpacked[2],
                    "y_est": unpacked[3],
                    "phi_est": unpacked[4],
                    "current": unpacked[5],
                    "voltage": unpacked[6],
                    "error_mm": unpacked[7],
                    "mode": unpacked[8],
                    "alpha": unpacked[9],
                }
                self.data_received.emit(telemetry)
            except socket.timeout:
                pass
            except Exception:
                time.sleep(0.05)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass


class Ad1AcquisitionThread(threading.Thread):
    def __init__(self, cfg: Ad1Config, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.out_queue = out_queue
        self.running = True
        self.dwf = Dwf()
        self.hdwf = 0
        self.sample_index = 0

    def stop(self):
        self.running = False

    def _push_chunk(self, t: np.ndarray, v: np.ndarray):
        try:
            self.out_queue.put_nowait((t, v))
        except queue.Full:
            pass

    def run(self):
        try:
            self.hdwf = self.dwf.open(self.cfg.device_index)
            acqmode_record = 3
            self.dwf._check(self.dwf.lib.FDwfAnalogInReset(self.hdwf), "FDwfAnalogInReset")
            self.dwf._check(self.dwf.lib.FDwfAnalogInAcquisitionModeSet(self.hdwf, acqmode_record), "FDwfAnalogInAcquisitionModeSet")
            self.dwf._check(self.dwf.lib.FDwfAnalogInRecordLengthSet(self.hdwf, 0.0), "FDwfAnalogInRecordLengthSet")
            self.dwf._check(self.dwf.lib.FDwfAnalogInFrequencySet(self.hdwf, float(self.cfg.sample_rate_hz)), "FDwfAnalogInFrequencySet")

            for ch in (0, 1):
                enable = 1 if (ch == (self.cfg.channel - 1)) else 0
                self.dwf._check(self.dwf.lib.FDwfAnalogInChannelEnableSet(self.hdwf, ch, enable), "FDwfAnalogInChannelEnableSet")

            idx = self.cfg.channel - 1
            self.dwf._check(self.dwf.lib.FDwfAnalogInChannelRangeSet(self.hdwf, idx, float(self.cfg.volts_range)), "FDwfAnalogInChannelRangeSet")
            self.dwf._check(self.dwf.lib.FDwfAnalogInChannelOffsetSet(self.hdwf, idx, float(self.cfg.volts_offset)), "FDwfAnalogInChannelOffsetSet")
            self.dwf._check(self.dwf.lib.FDwfAnalogInConfigure(self.hdwf, 1, 1), "FDwfAnalogInConfigure")

            st = ctypes.c_ubyte(0)
            available = ctypes.c_int(0)
            lost = ctypes.c_int(0)
            corrupt = ctypes.c_int(0)

            while self.running:
                self.dwf._check(self.dwf.lib.FDwfAnalogInStatus(self.hdwf, 1, ctypes.byref(st)), "FDwfAnalogInStatus")
                self.dwf._check(
                    self.dwf.lib.FDwfAnalogInStatusRecord(self.hdwf, ctypes.byref(available), ctypes.byref(lost), ctypes.byref(corrupt)),
                    "FDwfAnalogInStatusRecord",
                )
                n = int(available.value)
                if n > 0:
                    buf = (ctypes.c_double * n)()
                    self.dwf._check(self.dwf.lib.FDwfAnalogInStatusData(self.hdwf, idx, buf, n), "FDwfAnalogInStatusData")
                    v = np.ctypeslib.as_array(buf).astype(np.float64, copy=True)
                    t = (self.sample_index + np.arange(n, dtype=np.int64)) / float(self.cfg.sample_rate_hz)
                    self.sample_index += n
                    self._push_chunk(t, v)
                time.sleep(0.001)

        except Exception as e:
            try:
                self.out_queue.put_nowait(("__error__", str(e)))
            except Exception:
                pass
        finally:
            try:
                if self.hdwf:
                    self.dwf.lib.FDwfAnalogInConfigure(self.hdwf, 0, 0)
            except Exception:
                pass
            self.dwf.close(self.hdwf)


class LogThread(QThread):
    log_received = pyqtSignal(str)

    def __init__(self, stream, prefix: str):
        super().__init__()
        self.stream = stream
        self.prefix = prefix
        self.running = True

    def run(self):
        try:
            while self.running:
                line = self.stream.readline()
                if not line:
                    break
                try:
                    s = line.decode("utf-8", errors="replace")
                except Exception:
                    s = str(line)
                self.log_received.emit(self.prefix + s.rstrip())
        except Exception:
            pass

    def stop(self):
        self.running = False
        try:
            self.wait(500)
        except Exception:
            pass


class CaracterizacionFuerzaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caracterización Fuerza Magnética - Test u(t) Senoidal")
        self.setGeometry(50, 50, 1400, 800)

        self.cfg = Ad1Config()
        self._dwf = Dwf()

        self.acquiring = False
        self.recording = False
        self.ad_thread: Ad1AcquisitionThread | None = None
        self.ad_queue: queue.Queue = queue.Queue(maxsize=20)

        self.cpp_process = None
        self.log_threads = []

        self.uol_enabled_state = False
        self.uol_amp_state = 2.0
        self.uol_freq_state = 0.5
        self.uol_offset_state = 0.0
        self.test_running = False
        self.test_timer = QTimer()
        self.test_timer.setSingleShot(True)
        self.test_timer.timeout.connect(self._stop_test)

        self.telemetry = None
        self.last_udp_monotonic = None
        self.telem_thread = TelemetryThread()
        self.telem_thread.data_received.connect(self._on_telem)
        self.telem_thread.start()

        self.buffer_seconds = 30.0
        self.max_buffer = int(self.cfg.sample_rate_hz * self.buffer_seconds)
        self.t_buf = deque(maxlen=self.max_buffer)
        self.v_buf = deque(maxlen=self.max_buffer)
        self.f_buf = deque(maxlen=self.max_buffer)
        self.i_buf = deque(maxlen=self.max_buffer)
        self.u_buf = deque(maxlen=self.max_buffer)
        self.yref_buf = deque(maxlen=self.max_buffer)
        self.ysensor_buf = deque(maxlen=self.max_buffer)
        self.yest_buf = deque(maxlen=self.max_buffer)
        self.phi_buf = deque(maxlen=self.max_buffer)
        self.errmm_buf = deque(maxlen=self.max_buffer)
        self.mode_buf = deque(maxlen=self.max_buffer)
        self.alpha_buf = deque(maxlen=self.max_buffer)

        self.all_rows = []

        self._apply_dark_theme()
        self._setup_ui()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update)
        self.update_timer.start(50)

        self._refresh_devices()
        self._refresh_ports()

    def _apply_dark_theme(self):
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(30, 30, 30))
        pal.setColor(QPalette.WindowText, Qt.white)
        pal.setColor(QPalette.Base, QColor(20, 20, 20))
        pal.setColor(QPalette.AlternateBase, QColor(30, 30, 30))
        pal.setColor(QPalette.Text, Qt.white)
        pal.setColor(QPalette.Button, QColor(50, 50, 50))
        pal.setColor(QPalette.ButtonText, Qt.white)
        self.setPalette(pal)
        pg.setConfigOption("background", "#151515")
        pg.setConfigOption("foreground", "w")

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        left = QWidget()
        left.setMaximumWidth(380)
        left_layout = QVBoxLayout(left)

        # AD1
        grp_ad1 = QGroupBox("AD1 (Fuerza)")
        g1 = QGridLayout(grp_ad1)
        g1.addWidget(QLabel("Dispositivo:"), 0, 0)
        self.combo_dev = QComboBox()
        g1.addWidget(self.combo_dev, 0, 1, 1, 2)
        self.btn_refresh_dev = QPushButton("🔄")
        self.btn_refresh_dev.clicked.connect(self._refresh_devices)
        g1.addWidget(self.btn_refresh_dev, 0, 3)

        g1.addWidget(QLabel("Canal:"), 1, 0)
        self.spin_ch = QSpinBox()
        self.spin_ch.setRange(1, 2)
        self.spin_ch.setValue(1)
        g1.addWidget(self.spin_ch, 1, 1)

        g1.addWidget(QLabel("Fs (Hz):"), 1, 2)
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(10, 100000)
        self.spin_fs.setValue(2500)
        self.spin_fs.setDecimals(0)
        g1.addWidget(self.spin_fs, 1, 3)

        g1.addWidget(QLabel("N/V:"), 2, 0)
        self.spin_n_per_v = QDoubleSpinBox()
        self.spin_n_per_v.setRange(-1000, 1000)
        self.spin_n_per_v.setValue(1.0)
        self.spin_n_per_v.setDecimals(4)
        g1.addWidget(self.spin_n_per_v, 2, 1)

        g1.addWidget(QLabel("V0:"), 2, 2)
        self.spin_v0 = QDoubleSpinBox()
        self.spin_v0.setRange(-10, 10)
        self.spin_v0.setValue(0.0)
        self.spin_v0.setDecimals(4)
        g1.addWidget(self.spin_v0, 2, 3)

        self.btn_tare = QPushButton("Tare (V0 = actual)")
        self.btn_tare.clicked.connect(self._tare)
        g1.addWidget(self.btn_tare, 3, 0, 1, 2)

        self.btn_start_acq = QPushButton("▶ Iniciar AD1")
        self.btn_start_acq.clicked.connect(self._toggle_acq)
        g1.addWidget(self.btn_start_acq, 3, 2, 1, 2)

        left_layout.addWidget(grp_ad1)

        # Controlador
        grp_ctrl = QGroupBox("Controlador C++")
        g2 = QGridLayout(grp_ctrl)
        g2.addWidget(QLabel("Puerto:"), 0, 0)
        self.combo_ports = QComboBox()
        g2.addWidget(self.combo_ports, 0, 1, 1, 2)
        self.btn_refresh_ports = QPushButton("🔄")
        self.btn_refresh_ports.clicked.connect(self._refresh_ports)
        g2.addWidget(self.btn_refresh_ports, 0, 3)

        self.btn_run = QPushButton("▶ Iniciar")
        self.btn_run.clicked.connect(self._run_controller)
        g2.addWidget(self.btn_run, 1, 0, 1, 2)
        self.btn_stop = QPushButton("⏹ Detener")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_controller)
        g2.addWidget(self.btn_stop, 1, 2, 1, 2)

        left_layout.addWidget(grp_ctrl)

        # Test U-OL
        grp_test = QGroupBox("Test u(t) Senoidal (Lazo Abierto)")
        g3 = QGridLayout(grp_test)

        g3.addWidget(QLabel("Amp (V):"), 0, 0)
        self.spin_amp = QDoubleSpinBox()
        self.spin_amp.setRange(0, 10)
        self.spin_amp.setValue(2.0)
        self.spin_amp.setDecimals(2)
        g3.addWidget(self.spin_amp, 0, 1)

        g3.addWidget(QLabel("Freq (Hz):"), 0, 2)
        self.spin_freq = QDoubleSpinBox()
        self.spin_freq.setRange(0.05, 20)
        self.spin_freq.setValue(0.5)
        self.spin_freq.setDecimals(2)
        g3.addWidget(self.spin_freq, 0, 3)

        g3.addWidget(QLabel("Offset (V):"), 1, 0)
        self.spin_offset = QDoubleSpinBox()
        self.spin_offset.setRange(0, 10)
        self.spin_offset.setValue(0.0)
        self.spin_offset.setDecimals(2)
        g3.addWidget(self.spin_offset, 1, 1)

        g3.addWidget(QLabel("Duración (s):"), 1, 2)
        self.spin_dur = QDoubleSpinBox()
        self.spin_dur.setRange(1, 600)
        self.spin_dur.setValue(20)
        self.spin_dur.setDecimals(1)
        g3.addWidget(self.spin_dur, 1, 3)

        self.btn_test_start = QPushButton("▶ Iniciar Test")
        self.btn_test_start.setStyleSheet("background-color: #27ae60; font-weight: bold;")
        self.btn_test_start.clicked.connect(self._start_test)
        g3.addWidget(self.btn_test_start, 2, 0, 1, 2)

        self.btn_test_stop = QPushButton("⏹ Detener Test")
        self.btn_test_stop.setEnabled(False)
        self.btn_test_stop.clicked.connect(self._stop_test)
        g3.addWidget(self.btn_test_stop, 2, 2, 1, 2)

        left_layout.addWidget(grp_test)

        # Guardar
        grp_save = QGroupBox("Guardar")
        g4 = QGridLayout(grp_save)
        self.btn_save = QPushButton("💾 Guardar CSV")
        self.btn_save.clicked.connect(self._save_csv)
        g4.addWidget(self.btn_save, 0, 0)
        self.btn_clear = QPushButton("🗑 Limpiar buffers")
        self.btn_clear.clicked.connect(self._clear_buffers)
        g4.addWidget(self.btn_clear, 0, 1)
        left_layout.addWidget(grp_save)

        # Stats
        grp_stats = QGroupBox("Stats")
        g5 = QGridLayout(grp_stats)
        self.lbl_f = QLabel("F(N): --")
        self.lbl_i = QLabel("I(A): --")
        self.lbl_u = QLabel("U(V): --")
        self.lbl_y = QLabel("Y(mm): --")
        self.lbl_phi = QLabel("Phi: --")
        self.lbl_udp = QLabel("UDP age (s): --")
        self.lbl_samples = QLabel("Samples: 0")
        g5.addWidget(self.lbl_f, 0, 0)
        g5.addWidget(self.lbl_i, 0, 1)
        g5.addWidget(self.lbl_u, 1, 0)
        g5.addWidget(self.lbl_y, 1, 1)
        g5.addWidget(self.lbl_phi, 2, 0)
        g5.addWidget(self.lbl_udp, 2, 1)
        g5.addWidget(self.lbl_samples, 3, 0, 1, 2)
        left_layout.addWidget(grp_stats)

        # Log
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(120)
        self.txt_log.setStyleSheet("background-color:#101010; color:#00ff00; font-family:Consolas;")
        left_layout.addWidget(self.txt_log)

        left_layout.addStretch()
        main.addWidget(left)

        # Gráficas
        right = QWidget()
        right_layout = QVBoxLayout(right)

        splitter = QSplitter(Qt.Vertical)

        # Fuerza vs tiempo
        self.plot_f_t = pg.PlotWidget(title="Fuerza vs Tiempo")
        self.plot_f_t.setLabel("left", "Fuerza", units="N")
        self.plot_f_t.setLabel("bottom", "Tiempo", units="s")
        self.plot_f_t.showGrid(x=True, y=True, alpha=0.3)
        self.curve_f = self.plot_f_t.plot(pen=pg.mkPen("#e74c3c", width=2))
        splitter.addWidget(self.plot_f_t)

        # Fuerza vs corriente (fase)
        self.plot_f_i = pg.PlotWidget(title="Fuerza vs Corriente (Fase)")
        self.plot_f_i.setLabel("left", "Fuerza", units="N")
        self.plot_f_i.setLabel("bottom", "Corriente", units="A")
        self.plot_f_i.showGrid(x=True, y=True, alpha=0.3)
        self.scatter_fi = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush("#3498db"))
        self.plot_f_i.addItem(self.scatter_fi)
        splitter.addWidget(self.plot_f_i)

        # Fuerza vs posición (fase)
        self.plot_f_y = pg.PlotWidget(title="Fuerza vs Posición (Fase)")
        self.plot_f_y.setLabel("left", "Fuerza", units="N")
        self.plot_f_y.setLabel("bottom", "Posición", units="mm")
        self.plot_f_y.showGrid(x=True, y=True, alpha=0.3)
        self.scatter_fy = pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush("#9b59b6"))
        self.plot_f_y.addItem(self.scatter_fy)
        splitter.addWidget(self.plot_f_y)

        # Posición vs tiempo
        self.plot_y_t = pg.PlotWidget(title="Posición vs Tiempo")
        self.plot_y_t.setLabel("left", "Posición", units="mm")
        self.plot_y_t.setLabel("bottom", "Tiempo", units="s")
        self.plot_y_t.showGrid(x=True, y=True, alpha=0.3)
        self.plot_y_t.addLegend()
        self.curve_yref = self.plot_y_t.plot(pen=pg.mkPen("#ecf0f1", width=1), name="y_ref")
        self.curve_ys = self.plot_y_t.plot(pen=pg.mkPen("#00bcd4", width=2), name="y_sensor")
        self.curve_ye = self.plot_y_t.plot(pen=pg.mkPen("#e67e22", width=2), name="y_est")
        splitter.addWidget(self.plot_y_t)

        # Corriente y Voltaje vs tiempo
        self.plot_i_u = pg.PlotWidget(title="Corriente / Voltaje vs Tiempo")
        self.plot_i_u.setLabel("left", "I (A) / U (V)")
        self.plot_i_u.setLabel("bottom", "Tiempo", units="s")
        self.plot_i_u.showGrid(x=True, y=True, alpha=0.3)
        self.plot_i_u.addLegend()
        self.curve_i = self.plot_i_u.plot(pen=pg.mkPen("#2ecc71", width=2), name="I (A)")
        self.curve_u = self.plot_i_u.plot(pen=pg.mkPen("#f39c12", width=2), name="U (V)")
        splitter.addWidget(self.plot_i_u)

        right_layout.addWidget(splitter)
        main.addWidget(right, stretch=1)

    def _log(self, msg: str):
        self.txt_log.append(msg)

    def _refresh_devices(self):
        self.combo_dev.clear()
        try:
            devs = self._dwf.enum_devices()
            for d in devs:
                self.combo_dev.addItem(f"{d['index']}: {d['name']} ({d['sn']})", d["index"])
        except Exception as e:
            self._log(f"Error enumerando AD1: {e}")

    def _refresh_ports(self):
        self.combo_ports.clear()
        import glob
        ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        for p in ports:
            self.combo_ports.addItem(p)
        if not ports:
            self.combo_ports.addItem("/dev/ttyUSB0")

    def _tare(self):
        if len(self.v_buf) > 0:
            v0 = float(np.mean(list(self.v_buf)[-min(2000, len(self.v_buf)) :]))
            self.spin_v0.setValue(v0)
            self._log(f"Tare: V0 = {v0:.4f}")

    def _toggle_acq(self):
        if not self.acquiring:
            self._start_acq()
        else:
            self._stop_acq()

    def _start_acq(self):
        if self.acquiring:
            return
        idx = self.combo_dev.currentData()
        if idx is None:
            self._log("Selecciona dispositivo AD1")
            return
        self.cfg.device_index = idx
        self.cfg.channel = self.spin_ch.value()
        self.cfg.sample_rate_hz = self.spin_fs.value()
        self.max_buffer = int(self.cfg.sample_rate_hz * self.buffer_seconds)
        self.t_buf = deque(maxlen=self.max_buffer)
        self.v_buf = deque(maxlen=self.max_buffer)
        self.f_buf = deque(maxlen=self.max_buffer)
        self.i_buf = deque(maxlen=self.max_buffer)
        self.u_buf = deque(maxlen=self.max_buffer)
        self.yref_buf = deque(maxlen=self.max_buffer)
        self.ysensor_buf = deque(maxlen=self.max_buffer)
        self.yest_buf = deque(maxlen=self.max_buffer)
        self.phi_buf = deque(maxlen=self.max_buffer)
        self.errmm_buf = deque(maxlen=self.max_buffer)
        self.mode_buf = deque(maxlen=self.max_buffer)
        self.alpha_buf = deque(maxlen=self.max_buffer)

        self.ad_thread = Ad1AcquisitionThread(self.cfg, self.ad_queue)
        self.ad_thread.start()
        self.acquiring = True
        self.btn_start_acq.setText("⏹ Detener AD1")
        self._log("AD1 adquisición iniciada")

    def _stop_acq(self):
        if not self.acquiring:
            return
        if self.ad_thread:
            self.ad_thread.stop()
            self.ad_thread.join(timeout=2)
            self.ad_thread = None
        self.acquiring = False
        self.btn_start_acq.setText("▶ Iniciar AD1")
        self._log("AD1 adquisición detenida")

    def _on_telem(self, data: dict):
        self.telemetry = data
        self.last_udp_monotonic = time.monotonic()

    def _run_controller(self):
        if self.cpp_process is not None:
            return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exe_candidates = ["levitador_sensorless_kan_tmp", "levitador_sensorless_kan"]
        exe_path = None
        for c in exe_candidates:
            p = os.path.join(script_dir, c)
            if os.path.exists(p) and os.access(p, os.X_OK):
                exe_path = p
                break
        if not exe_path:
            self._log("No se encuentra ejecutable del controlador")
            return
        port = self.combo_ports.currentText()
        if not port:
            self._log("Selecciona puerto serial")
            return
        try:
            self.cpp_process = subprocess.Popen(
                [exe_path, port],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.log_threads = []
            if self.cpp_process.stdout is not None:
                t_out = LogThread(self.cpp_process.stdout, "[C++] ")
                t_out.log_received.connect(self._log)
                t_out.start()
                self.log_threads.append(t_out)
            if self.cpp_process.stderr is not None:
                t_err = LogThread(self.cpp_process.stderr, "[ERR] ")
                t_err.log_received.connect(self._log)
                t_err.start()
                self.log_threads.append(t_err)

            self.uol_enabled_state = False
            self.uol_amp_state = 2.0
            self.uol_freq_state = 0.5
            self.uol_offset_state = 0.0
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self._log(f"Controlador iniciado: {os.path.basename(exe_path)}")
        except Exception as e:
            self._log(f"Error iniciando controlador: {e}")

    def _stop_controller(self):
        if self.cpp_process:
            try:
                self.cpp_process.terminate()
            except Exception:
                pass
            self.cpp_process = None
        for t in list(self.log_threads):
            try:
                t.stop()
            except Exception:
                pass
        self.log_threads = []
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log("Controlador detenido")

    def _send_key(self, key: str) -> bool:
        if not self.cpp_process or not self.cpp_process.stdin:
            return False
        if self.cpp_process.poll() is not None:
            return False
        try:
            self.cpp_process.stdin.write(key.encode("utf-8"))
            self.cpp_process.stdin.flush()
            return True
        except Exception:
            return False

    def _set_uol_params(self, amp_v: float, freq_hz: float, offset_v: float):
        amp_step, freq_step, off_step = 0.2, 0.05, 0.2
        key_delay = 0.015  # 15ms entre teclas para que el C++ las procese

        n_off = int(round((offset_v - self.uol_offset_state) / off_step))
        for _ in range(abs(n_off)):
            self._send_key("p" if n_off > 0 else "o")
            time.sleep(key_delay)
        self.uol_offset_state += n_off * off_step

        n_amp = int(round((amp_v - self.uol_amp_state) / amp_step))
        for _ in range(abs(n_amp)):
            self._send_key("k" if n_amp > 0 else "j")
            time.sleep(key_delay)
        self.uol_amp_state += n_amp * amp_step

        n_freq = int(round((freq_hz - self.uol_freq_state) / freq_step))
        for _ in range(abs(n_freq)):
            self._send_key("." if n_freq > 0 else ",")
            time.sleep(key_delay)
        self.uol_freq_state += n_freq * freq_step
        
        self._log(f"Params set: off={self.uol_offset_state:.1f}V, amp={self.uol_amp_state:.1f}V, f={self.uol_freq_state:.2f}Hz")

    def _start_test(self):
        if self.test_running:
            return
        if self.cpp_process is None:
            self._log("Inicia el controlador primero")
            return
        if not self.acquiring:
            self._log("Inicia adquisición AD1 primero")
            return

        amp = self.spin_amp.value()
        freq = self.spin_freq.value()
        offset = self.spin_offset.value()
        dur = self.spin_dur.value()

        if not self.uol_enabled_state:
            if not self._send_key("u"):
                self._log("No se pudo activar U-OL (controlador no responde)")
                return
            self.uol_enabled_state = True

        self._set_uol_params(amp, freq, offset)

        self.recording = True
        self.all_rows = []
        self.test_running = True
        self.btn_test_start.setEnabled(False)
        self.btn_test_stop.setEnabled(True)
        self._log(f"Test iniciado: A={self.uol_amp_state:.2f}V, f={self.uol_freq_state:.2f}Hz, off={self.uol_offset_state:.2f}V, dur={dur:.1f}s")
        self.test_timer.start(int(dur * 1000))

    def _stop_test(self):
        if not self.test_running:
            return
        if self.uol_enabled_state:
            self._send_key("u")
            self.uol_enabled_state = False
        self.recording = False
        self.test_running = False
        self.btn_test_start.setEnabled(True)
        self.btn_test_stop.setEnabled(False)
        self._log(f"Test detenido. Samples grabados: {len(self.all_rows)}")

    def _save_csv(self):
        if not self.all_rows:
            self._log("No hay datos para guardar")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar CSV", "caracterizacion_fuerza.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "t_s",
                    "force_N",
                    "current_A",
                    "voltage_V",
                    "y_ref_mm",
                    "y_sensor_mm",
                    "y_est_mm",
                    "phi_est",
                    "error_mm",
                    "mode",
                    "alpha",
                    "udp_age_s",
                ])
                w.writerows(self.all_rows)
            self._log(f"Guardado: {path} ({len(self.all_rows)} filas)")
        except Exception as e:
            self._log(f"Error guardando: {e}")

    def _clear_buffers(self):
        self.t_buf.clear()
        self.v_buf.clear()
        self.f_buf.clear()
        self.i_buf.clear()
        self.u_buf.clear()
        self.yref_buf.clear()
        self.ysensor_buf.clear()
        self.yest_buf.clear()
        self.phi_buf.clear()
        self.errmm_buf.clear()
        self.mode_buf.clear()
        self.alpha_buf.clear()
        self.all_rows = []
        self._log("Buffers limpiados")

    def _update(self):
        if self.cpp_process is not None and self.cpp_process.poll() is not None:
            self._log("Controlador terminó")
            self._stop_test()
            self._stop_controller()

        while True:
            try:
                item = self.ad_queue.get_nowait()
            except queue.Empty:
                break

            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and item[0] == "__error__":
                self._log(f"AD1 error: {item[1]}")
                continue

            t, v = item
            if v.size == 1 and np.isnan(v[0]):
                continue

            n_per_v = float(self.spin_n_per_v.value())
            v0 = float(self.spin_v0.value())
            f_n = (v - v0) * n_per_v

            i_a = float(self.telemetry.get("current", np.nan)) if self.telemetry else np.nan
            u_v = float(self.telemetry.get("voltage", np.nan)) if self.telemetry else np.nan
            y_ref = float(self.telemetry.get("y_ref", np.nan)) if self.telemetry else np.nan
            y_sensor = float(self.telemetry.get("y_sensor", np.nan)) if self.telemetry else np.nan
            y_est = float(self.telemetry.get("y_est", np.nan)) if self.telemetry else np.nan
            phi_est = float(self.telemetry.get("phi_est", np.nan)) if self.telemetry else np.nan
            err_mm = float(self.telemetry.get("error_mm", np.nan)) if self.telemetry else np.nan
            mode = int(self.telemetry.get("mode", -1)) if self.telemetry else -1
            alpha = float(self.telemetry.get("alpha", np.nan)) if self.telemetry else np.nan
            udp_age_s = np.nan
            if self.last_udp_monotonic is not None:
                udp_age_s = float(time.monotonic() - self.last_udp_monotonic)

            for i in range(len(v)):
                self.t_buf.append(float(t[i]))
                self.v_buf.append(float(v[i]))
                self.f_buf.append(float(f_n[i]))
                self.i_buf.append(i_a)
                self.u_buf.append(u_v)
                self.yref_buf.append(y_ref)
                self.ysensor_buf.append(y_sensor)
                self.yest_buf.append(y_est)
                self.phi_buf.append(phi_est)
                self.errmm_buf.append(err_mm)
                self.mode_buf.append(mode)
                self.alpha_buf.append(alpha)

                if self.recording:
                    self.all_rows.append([
                        float(t[i]),
                        float(f_n[i]),
                        i_a,
                        u_v,
                        y_ref,
                        y_sensor,
                        y_est,
                        phi_est,
                        err_mm,
                        mode,
                        alpha,
                        udp_age_s,
                    ])

        if len(self.t_buf) < 2:
            return

        t_np = np.array(self.t_buf)
        f_np = np.array(self.f_buf)
        i_np = np.array(self.i_buf)
        u_np = np.array(self.u_buf)
        yref_np = np.array(self.yref_buf)
        ys_np = np.array(self.ysensor_buf)
        ye_np = np.array(self.yest_buf)
        phi_np = np.array(self.phi_buf)

        self.curve_f.setData(t_np, f_np)

        valid_i = ~np.isnan(i_np)
        if np.any(valid_i):
            pts = np.column_stack([i_np[valid_i], f_np[valid_i]])
            if len(pts) > 5000:
                pts = pts[-5000:]
            self.scatter_fi.setData(pts[:, 0], pts[:, 1])

            self.curve_i.setData(t_np[valid_i], i_np[valid_i])
            self.curve_u.setData(t_np[valid_i], u_np[valid_i])

        valid_y = ~np.isnan(ys_np)
        if np.any(valid_y):
            pts = np.column_stack([ys_np[valid_y], f_np[valid_y]])
            if len(pts) > 5000:
                pts = pts[-5000:]
            self.scatter_fy.setData(pts[:, 0], pts[:, 1])

        if len(t_np) > 0:
            self.curve_yref.setData(t_np, yref_np)
            self.curve_ys.setData(t_np, ys_np)
            self.curve_ye.setData(t_np, ye_np)

        f_mean = float(np.mean(f_np[-500:])) if len(f_np) > 0 else 0
        i_mean = np.nan
        u_mean = np.nan
        y_mean = np.nan
        phi_mean = np.nan
        if len(i_np) > 0:
            w = i_np[-min(500, len(i_np)) :]
            fin = np.isfinite(w)
            if np.any(fin):
                i_mean = float(np.mean(w[fin]))
        if len(u_np) > 0:
            w = u_np[-min(500, len(u_np)) :]
            fin = np.isfinite(w)
            if np.any(fin):
                u_mean = float(np.mean(w[fin]))
        if len(ys_np) > 0:
            w = ys_np[-min(500, len(ys_np)) :]
            fin = np.isfinite(w)
            if np.any(fin):
                y_mean = float(np.mean(w[fin]))
        if len(phi_np) > 0:
            w = phi_np[-min(500, len(phi_np)) :]
            fin = np.isfinite(w)
            if np.any(fin):
                phi_mean = float(np.mean(w[fin]))

        self.lbl_f.setText(f"F(N): {f_mean:+.4f}")
        self.lbl_i.setText(f"I(A): {i_mean:.3f}" if np.isfinite(i_mean) else "I(A): --")
        self.lbl_u.setText(f"U(V): {u_mean:.2f}" if np.isfinite(u_mean) else "U(V): --")
        self.lbl_y.setText(f"Y(mm): {y_mean:.3f}" if np.isfinite(y_mean) else "Y(mm): --")
        self.lbl_phi.setText(f"Phi: {phi_mean:+.4f}" if np.isfinite(phi_mean) else "Phi: --")

        if self.last_udp_monotonic is None:
            self.lbl_udp.setText("UDP age (s): --")
        else:
            self.lbl_udp.setText(f"UDP age (s): {time.monotonic() - self.last_udp_monotonic:.2f}")
        self.lbl_samples.setText(f"Samples: {len(self.all_rows)}")

    def closeEvent(self, event):
        try:
            self._stop_test()
        except Exception:
            pass
        try:
            self._stop_acq()
        except Exception:
            pass
        try:
            self.telem_thread.stop()
        except Exception:
            pass
        try:
            self._stop_controller()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CaracterizacionFuerzaGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

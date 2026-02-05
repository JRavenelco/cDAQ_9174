from __future__ import annotations

import sys
import queue
from typing import List, Optional

import numpy as np

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


def parse_channels(ch_str: str) -> List[int]:
    try:
        ch = [int(x.strip()) for x in ch_str.split(',') if x.strip() != '']
        ch = sorted(set(ch))
        return ch if ch else [0]
    except Exception:
        return [0]


class DigilentForceGUI(QtWidgets.QMainWindow):
    """Standalone GUI to acquire and plot force data using Digilent AnalogIn - FIXED VERSION."""

    def __init__(
        self,
        default_channels: List[int] = [0, 1],
        sample_rate: float = 20000.0,
        channel_range: float = 5.0,
        time_window: float = 5.0,
        queue_size: int = 32,
        device_buffer: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Digilent Force (AnalogIn) - Fixed")

        # Acquisition params
        self.channels: List[int] = list(default_channels)
        self.sample_rate: float = float(sample_rate)
        self.channel_range: float = float(channel_range)
        self.time_window: float = float(time_window)
        self.queue_size: int = int(queue_size)
        self.device_buffer: Optional[int] = int(device_buffer) if device_buffer else None

        # Runtime state
        self.queue: Optional[queue.Queue] = None
        self.th = None  # DigilentAnalogInThread instance

        # Data buffers for plotting (fixed size channels, no dynamic rebuild)
        self.max_channels = 8  # Support up to 8 channels statically
        self.buffers: List[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(self.max_channels)]
        self.nan_counts: List[int] = [0 for _ in range(self.max_channels)]

        # UI
        self._setup_ui()

        # Timer to poll data and refresh plots
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)  # 20 FPS
        self.timer.timeout.connect(self._on_timer)

    # --------------------- UI ---------------------
    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Controls
        ctrl = QtWidgets.QHBoxLayout()
        root.addLayout(ctrl)

        ctrl.addWidget(QtWidgets.QLabel("Canales:"))
        self.edit_channels = QtWidgets.QLineEdit(
            ",".join(str(c) for c in self.channels)
        )
        ctrl.addWidget(self.edit_channels)

        ctrl.addWidget(QtWidgets.QLabel("SR:"))
        self.spin_sr = QtWidgets.QDoubleSpinBox()
        self.spin_sr.setRange(1000.0, 100000.0)
        self.spin_sr.setDecimals(0)
        self.spin_sr.setValue(self.sample_rate)
        ctrl.addWidget(self.spin_sr)

        ctrl.addWidget(QtWidgets.QLabel("Rango:"))
        self.spin_range = QtWidgets.QDoubleSpinBox()
        self.spin_range.setRange(0.2, 25.0)
        self.spin_range.setDecimals(1)
        self.spin_range.setValue(self.channel_range)
        ctrl.addWidget(self.spin_range)

        ctrl.addWidget(QtWidgets.QLabel("Ventana:"))
        self.spin_window = QtWidgets.QDoubleSpinBox()
        self.spin_window.setRange(1.0, 30.0)
        self.spin_window.setDecimals(1)
        self.spin_window.setValue(self.time_window)
        ctrl.addWidget(self.spin_window)

        self.btn_start = QtWidgets.QPushButton("Iniciar")
        self.btn_stop = QtWidgets.QPushButton("Detener")
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)

        self.lbl_status = QtWidgets.QLabel("Detenido")
        ctrl.addWidget(self.lbl_status)

        self.btn_start.clicked.connect(self._start_acq)
        self.btn_stop.clicked.connect(self._stop_acq)

        # Info area (simple text instead of complex grid)
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        root.addWidget(self.info_text)

        # Fixed plots for first 4 channels
        plots_widget = QtWidgets.QWidget()
        plots_layout = QtWidgets.QGridLayout(plots_widget)
        root.addWidget(plots_widget)

        self.plot_widgets: List[pg.PlotWidget] = []
        self.curves: List[pg.PlotDataItem] = []

        for i in range(4):  # Fixed 4 plots
            w = pg.PlotWidget(title=f"Canal {i}")
            w.setLabel('left', 'Voltaje', units='V')
            w.setLabel('bottom', 'Tiempo', units='s')
            w.showGrid(x=True, y=True)
            w.setYRange(-self.channel_range, self.channel_range)
            w.setXRange(-self.time_window, 0.0)

            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            pen = pg.mkPen(colors[i], width=1)
            curve = w.plot(pen=pen)

            self.plot_widgets.append(w)
            self.curves.append(curve)

            row = i // 2
            col = i % 2
            plots_layout.addWidget(w, row, col)

    # --------------------- Acquisition control ---------------------
    def _start_acq(self) -> None:
        print("[DEBUG] Starting acquisition...")
        
        # Parse current parameters
        self.channels = parse_channels(self.edit_channels.text())
        self.sample_rate = float(self.spin_sr.value())
        self.channel_range = float(self.spin_range.value())
        self.time_window = float(self.spin_window.value())
        
        print(f"[DEBUG] Channels: {self.channels}, SR: {self.sample_rate}")

        # Reset data
        for i in range(self.max_channels):
            self.buffers[i] = np.empty(0, dtype=np.float32)
            self.nan_counts[i] = 0

        # Update plot ranges
        for w in self.plot_widgets:
            w.setYRange(-self.channel_range, self.channel_range)
            w.setXRange(-self.time_window, 0.0)

        # Create queue and thread
        self.queue = queue.Queue(maxsize=self.queue_size)
        
        try:
            from digilent_analogin_thread import DigilentAnalogInThread
            print("[DEBUG] DigilentAnalogInThread imported")
        except Exception as e:
            print(f"[DEBUG] Import failed: {e}")
            self.lbl_status.setText(f"Error: {e}")
            return

        try:
            print("[DEBUG] Creating thread...")
            self.th = DigilentAnalogInThread(
                data_queue=self.queue,
                channels=self.channels,
                sample_rate=self.sample_rate,
                channel_range=self.channel_range,
                buffer_size=None,  # Let thread use optimal default based on sample rate
                poll_interval_s=0.01,  # Reduced from 0.002 to prevent data loss
                verbose=True,  # Enable verbose for debugging
            )
            
            print("[DEBUG] Starting thread...")
            self.th.start()
            
            print("[DEBUG] Updating UI...")
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("Adquiriendo...")
            
            print("[DEBUG] Starting timer...")
            self.timer.start()
            print("[DEBUG] Acquisition started!")
            
        except Exception as e:
            print(f"[DEBUG] Thread error: {e}")
            import traceback
            traceback.print_exc()
            self.lbl_status.setText(f"Error: {e}")
            self.queue = None
            self.th = None

    def _stop_acq(self) -> None:
        print("[DEBUG] Stopping...")
        self.timer.stop()
        
        if self.th is not None:
            try:
                self.th.stop()
                self.th.join(timeout=2.0)
            except Exception as e:
                print(f"[DEBUG] Stop error: {e}")
                
        if self.th and hasattr(self.th, 'exception') and self.th.exception:
            self.lbl_status.setText(f"Error: {self.th.exception}")
        else:
            self.lbl_status.setText("Detenido")
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.th = None
        self.queue = None
        print("[DEBUG] Stopped")

    def closeEvent(self, ev):  # noqa: N802
        self._stop_acq()
        super().closeEvent(ev)

    # --------------------- Data handling ---------------------
    def _on_timer(self) -> None:
        if self.queue is None or self.th is None:
            return

        # Check thread health
        if not self.th.is_alive():
            print("[DEBUG] Thread died")
            if hasattr(self.th, 'exception') and self.th.exception:
                print(f"[DEBUG] Exception: {self.th.exception}")
            self._stop_acq()
            return

        blocks_processed = 0
        updated = False
        
        # Process available blocks (limited to avoid GUI freeze)
        while blocks_processed < 5:  # Max 5 blocks per timer tick
            try:
                blk = self.queue.get_nowait()
                blocks_processed += 1
            except queue.Empty:
                break

            if not isinstance(blk, np.ndarray) or blk.ndim != 2:
                continue

            n_ch, n_samp = blk.shape
            
            # Process only channels we care about
            for i in range(min(n_ch, len(self.channels))):
                if i >= self.max_channels:
                    break
                    
                arr = blk[i, :]
                
                # NaN filtering
                nan_mask = np.isnan(arr)
                if nan_mask.any():
                    self.nan_counts[i] += int(nan_mask.sum())
                    arr = arr[~nan_mask]
                    
                if arr.size == 0:
                    continue

                # Append new data
                self.buffers[i] = np.concatenate([self.buffers[i], arr])
                
                # Trim to time window
                max_len = int(self.time_window * self.sample_rate)
                if self.buffers[i].size > max_len:
                    self.buffers[i] = self.buffers[i][-max_len:]

            updated = True

        if updated:
            self._update_plots_and_info()

    def _update_plots_and_info(self) -> None:
        # Update info text
        info_lines = []
        for i, ch_idx in enumerate(self.channels):
            if i >= self.max_channels:
                break
            y = self.buffers[i]
            if y.size > 0:
                mn = float(np.min(y))
                mx = float(np.max(y))
                rms = float(np.sqrt(np.mean(y * y)))
                info_lines.append(f"Ch{ch_idx}: min={mn:.3f}V, max={mx:.3f}V, RMS={rms:.3f}V, NaNs={self.nan_counts[i]}, N={y.size}")
            else:
                info_lines.append(f"Ch{ch_idx}: sin datos")
        
        self.info_text.setPlainText('\n'.join(info_lines))

        # Update plots
        for i, ch_idx in enumerate(self.channels):
            if i >= len(self.plot_widgets):
                break
                
            y = self.buffers[i]
            if y.size > 0:
                # Time axis: newest sample at t=0, older samples negative
                t = (np.arange(y.size) - y.size) / self.sample_rate
                self.curves[i].setData(t, y)
                
                # Update plot title
                self.plot_widgets[i].setTitle(f"Canal {ch_idx} ({y.size} muestras)")
            else:
                self.curves[i].setData([], [])


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="GUI Digilent Force - FIXED")
    parser.add_argument("--channels", "-c", type=str, default="0,1", help="Canales")
    parser.add_argument("--sample_rate", "--sr", type=float, default=10000.0, help="Sample rate (Hz)")
    parser.add_argument("--range", "-r", dest="channel_range", type=float, default=5.0, help="Range (±V)")
    parser.add_argument("--window", "-w", type=float, default=5.0, help="Time window (s)")
    parser.add_argument("--queue_size", type=int, default=16, help="Queue size")
    parser.add_argument("--buffer_size", type=int, default=0, help="Device buffer size")
    args = parser.parse_args()

    ch = parse_channels(args.channels)
    buf = None if int(args.buffer_size) == 0 else int(args.buffer_size)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    
    print("Creating GUI...")
    win = DigilentForceGUI(
        default_channels=ch,
        sample_rate=float(args.sample_rate),
        channel_range=float(args.channel_range),
        time_window=float(args.window),
        queue_size=int(args.queue_size),
        device_buffer=buf,
    )
    win.resize(1000, 700)
    win.show()
    print("GUI shown, starting event loop...")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

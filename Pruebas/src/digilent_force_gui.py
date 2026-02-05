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
    """Standalone GUI to acquire and plot force data using Digilent AnalogIn.

    - Uses DigilentAnalogInThread (pydwf) to stream continuous voltage samples.
    - Real-time plots with pyqtgraph, matching the style of the existing force tab:
      left axis: Voltaje (V), bottom: Tiempo (s), grid on, Y range = ±range, X range = [-time_window, 0].
    - Displays basic data quality metrics per channel: min, max, RMS, NaNs (counted if detected by GUI), sample count.
    """

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
        self.setWindowTitle("Digilent Force (AnalogIn)")

        # Acquisition params
        self.channels: List[int] = list(default_channels)
        self.sample_rate: float = float(sample_rate)
        self.channel_range: float = float(channel_range)
        self.time_window: float = float(time_window)
        self.queue_size: int = int(queue_size)
        self.device_buffer: Optional[int] = int(device_buffer) if device_buffer else None

        # Runtime state
        self.queue: Optional[queue.Queue] = None
        self.th = None  # DigilentAnalogInThread instance (created lazily)

        self.buffers: List[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in self.channels]
        self.nan_counts: List[int] = [0 for _ in self.channels]

        # UI
        self._setup_ui()

        # Timer to poll data and refresh plots
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS
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
        self.edit_channels.setToolTip("Lista de canales, ej: 0,1,2")
        ctrl.addWidget(self.edit_channels)

        ctrl.addWidget(QtWidgets.QLabel("SR (Hz):"))
        self.spin_sr = QtWidgets.QDoubleSpinBox()
        self.spin_sr.setRange(1.0, 10_000_000.0)
        self.spin_sr.setDecimals(1)
        self.spin_sr.setValue(self.sample_rate)
        self.spin_sr.setSingleStep(100.0)
        ctrl.addWidget(self.spin_sr)

        ctrl.addWidget(QtWidgets.QLabel("Rango (±V):"))
        self.spin_range = QtWidgets.QDoubleSpinBox()
        self.spin_range.setRange(0.2, 50.0)
        self.spin_range.setDecimals(2)
        self.spin_range.setValue(self.channel_range)
        self.spin_range.setSingleStep(0.5)
        ctrl.addWidget(self.spin_range)

        ctrl.addWidget(QtWidgets.QLabel("Ventana (s):"))
        self.spin_window = QtWidgets.QDoubleSpinBox()
        self.spin_window.setRange(0.2, 60.0)
        self.spin_window.setDecimals(1)
        self.spin_window.setValue(self.time_window)
        self.spin_window.setSingleStep(0.5)
        self.spin_window.valueChanged.connect(self._on_window_changed)
        ctrl.addWidget(self.spin_window)

        self.btn_start = QtWidgets.QPushButton("Iniciar")
        self.btn_stop = QtWidgets.QPushButton("Detener")
        self.btn_stop.setEnabled(False)
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)

        self.lbl_status = QtWidgets.QLabel("Detenido")
        self.lbl_status.setMinimumWidth(200)
        ctrl.addWidget(self.lbl_status)
        ctrl.addStretch(1)

        self.btn_start.clicked.connect(self._start_acq)
        self.btn_stop.clicked.connect(self._stop_acq)

        # Info grid (metrics)
        self.info_grid = QtWidgets.QGridLayout()
        root.addLayout(self.info_grid)
        self._build_info_header()

        # Plots grid
        self.plots_grid = QtWidgets.QGridLayout()
        root.addLayout(self.plots_grid)

        # Create channel-dependent views
        self._rebuild_channel_views()

    def _build_info_header(self) -> None:
        labels = ["Canal", "Min (V)", "Max (V)", "RMS (V)", "NaNs", "Muestras"]
        for j, txt in enumerate(labels):
            lab = QtWidgets.QLabel(f"<b>{txt}</b>")
            self.info_grid.addWidget(lab, 0, j)

    def _clear_layout(self, lay: QtWidgets.QLayout) -> None:
        print(f"[DEBUG] Clearing layout with {lay.count()} items")
        while lay.count():
            item = lay.takeAt(0)
            if item is None:
                break
            w = item.widget()
            if w is not None:
                print(f"[DEBUG] Removing widget: {w}")
                w.setParent(None)
                w.deleteLater()
            # For nested layouts
            child_lay = item.layout()
            if child_lay is not None:
                print(f"[DEBUG] Recursively clearing child layout")
                self._clear_layout(child_lay)
        print(f"[DEBUG] Layout cleared")

    def _rebuild_channel_views(self) -> None:
        # Clear dynamic rows of info grid (keep header at row 0)
        while self.info_grid.rowCount() > 1:
            # Remove widgets from bottom row upward
            row = self.info_grid.rowCount() - 1
            for col in range(self.info_grid.columnCount()):
                item = self.info_grid.itemAtPosition(row, col)
                if item:
                    w = item.widget()
                    if w:
                        w.setParent(None)
                        w.deleteLater()

        # Clear plots grid completely
        self._clear_layout(self.plots_grid)

        # Parse channels from edit and reset buffers/metrics
        self.channels = parse_channels(self.edit_channels.text())
        self.buffers = [np.empty(0, dtype=np.float32) for _ in self.channels]
        self.nan_counts = [0 for _ in self.channels]

        # Create metrics labels and plots per channel
        self.lab_min: List[QtWidgets.QLabel] = []
        self.lab_max: List[QtWidgets.QLabel] = []
        self.lab_rms: List[QtWidgets.QLabel] = []
        self.lab_nans: List[QtWidgets.QLabel] = []
        self.lab_count: List[QtWidgets.QLabel] = []

        self.plot_widgets: List[pg.PlotWidget] = []
        self.curves: List[pg.PlotDataItem] = []

        # Info rows
        for i, ch in enumerate(self.channels, start=1):
            self.info_grid.addWidget(QtWidgets.QLabel(str(ch)), i, 0)

            lab_min = QtWidgets.QLabel("-")
            lab_max = QtWidgets.QLabel("-")
            lab_rms = QtWidgets.QLabel("-")
            lab_nans = QtWidgets.QLabel("0")
            lab_cnt = QtWidgets.QLabel("0")

            self.lab_min.append(lab_min)
            self.lab_max.append(lab_max)
            self.lab_rms.append(lab_rms)
            self.lab_nans.append(lab_nans)
            self.lab_count.append(lab_cnt)

            self.info_grid.addWidget(lab_min, i, 1)
            self.info_grid.addWidget(lab_max, i, 2)
            self.info_grid.addWidget(lab_rms, i, 3)
            self.info_grid.addWidget(lab_nans, i, 4)
            self.info_grid.addWidget(lab_cnt, i, 5)

        # Plots: arrange in two columns similar to existing GUI
        for idx, ch in enumerate(self.channels):
            w = pg.PlotWidget(title=f"Canal {ch}")
            w.setLabel('left', 'Voltaje', units='V')
            w.setLabel('bottom', 'Tiempo', units='s')
            w.showGrid(x=True, y=True)
            w.setYRange(-self.spin_range.value(), self.spin_range.value())
            w.setXRange(-self.spin_window.value(), 0.0)

            pen = pg.mkPen(self._color_for(idx), width=1)
            curve = w.plot(pen=pen)

            self.plot_widgets.append(w)
            self.curves.append(curve)

            r = idx // 2
            c = idx % 2
            self.plots_grid.addWidget(w, r, c)

    # --------------------- Acquisition control ---------------------
    def _on_window_changed(self) -> None:
        # Update plot X ranges immediately
        self.time_window = float(self.spin_window.value())
        for w in self.plot_widgets:
            w.setXRange(-self.time_window, 0.0, padding=0)

    def _start_acq(self) -> None:
        print("[DEBUG] _start_acq called")
        # Validate/parse params
        self.channels = parse_channels(self.edit_channels.text())
        self.sample_rate = float(self.spin_sr.value())
        self.channel_range = float(self.spin_range.value())
        self.time_window = float(self.spin_window.value())
        print(f"[DEBUG] Parsed params: channels={self.channels}, sr={self.sample_rate}, range={self.channel_range}")

        # Rebuild views, as channels might have changed
        print("[DEBUG] Rebuilding channel views...")
        self._rebuild_channel_views()
        print("[DEBUG] Channel views rebuilt")

        # Create queue and start thread (lazy import to avoid hard dependency at startup)
        self.queue = queue.Queue(maxsize=self.queue_size)
        print(f"[DEBUG] Queue created with maxsize={self.queue_size}")
        
        try:
            from digilent_analogin_thread import DigilentAnalogInThread
            print("[DEBUG] DigilentAnalogInThread imported successfully")
        except Exception as e:
            print(f"[DEBUG] Import error: {e}")
            self.lbl_status.setText(f"Error importando pydwf/digilent: {e}")
            QtWidgets.QMessageBox.critical(
                self,
                "Error",
                "No se pudo importar DigilentAnalogInThread. Asegúrate de tener pydwf instalado (pip install pydwf) y el archivo digilent_analogin_thread.py disponible.",
            )
            self.queue = None
            return

        try:
            print("[DEBUG] Creating DigilentAnalogInThread...")
            self.th = DigilentAnalogInThread(
                data_queue=self.queue,
                channels=self.channels,
                sample_rate=self.sample_rate,
                channel_range=self.channel_range,
                buffer_size=self.device_buffer,
                poll_interval_s=0.002,
                verbose=False,  # Reduce console spam to avoid UI freeze
            )
            print("[DEBUG] Starting thread...")
            self.th.start()
            print("[DEBUG] Thread started successfully")
        except Exception as e:
            print(f"[DEBUG] Thread creation/start error: {e}")
            import traceback
            traceback.print_exc()
            self.lbl_status.setText(f"Error iniciando adquisición: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo iniciar la adquisición:\n{e}")
            self.queue = None
            self.th = None
            return

        print("[DEBUG] Updating UI state...")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText(
            f"Adquiriendo: canales={self.channels}, SR={self.sample_rate:.0f} Hz, rango=±{self.channel_range:.2f} V"
        )
        print("[DEBUG] Starting timer...")
        self.timer.start()
        print("[DEBUG] _start_acq completed")

    def _stop_acq(self) -> None:
        print("[DEBUG] _stop_acq called")
        self.timer.stop()
        print("[DEBUG] Timer stopped")
        
        if self.th is not None:
            try:
                print("[DEBUG] Stopping thread...")
                self.th.stop()
                print("[DEBUG] Joining thread...")
                self.th.join(timeout=2.0)
                print("[DEBUG] Thread joined")
            except Exception as e:
                print(f"[DEBUG] Exception stopping thread: {e}")
                
        if self.th is not None and getattr(self.th, "exception", None):
            print(f"[DEBUG] Thread had exception: {self.th.exception}")
            self.lbl_status.setText(f"Error hilo: {self.th.exception}")
            QtWidgets.QMessageBox.warning(self, "Hilo", f"Excepción en adquisición:\n{self.th.exception}")
        else:
            self.lbl_status.setText("Detenido")
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.th = None
        self.queue = None
        print("[DEBUG] _stop_acq completed")

    def closeEvent(self, ev):  # noqa: N802
        try:
            self._stop_acq()
        finally:
            super().closeEvent(ev)

    # --------------------- Data handling ---------------------
    def _on_timer(self) -> None:
        try:
            # If thread failed, stop and report
            if self.th is not None and getattr(self.th, "exception", None):
                print(f"[DEBUG] Thread exception detected: {self.th.exception}")
                self._stop_acq()
                return

            if self.queue is None:
                print("[DEBUG] Queue is None, timer returning")
                return

            # Check if thread is still alive
            if self.th is not None and not self.th.is_alive():
                print("[DEBUG] Thread is not alive anymore")
                if hasattr(self.th, 'exception') and self.th.exception:
                    print(f"[DEBUG] Thread died with exception: {self.th.exception}")
                self._stop_acq()
                return

            blocks_processed = 0
            max_blocks_per_tick = 6  # limit to keep UI responsive
            updated = False
            while blocks_processed < max_blocks_per_tick:
                try:
                    blk = self.queue.get_nowait()
                    blocks_processed += 1
                except queue.Empty:
                    break

                # Expect blk shape = (n_channels, n_samples)
                if not isinstance(blk, np.ndarray) or blk.ndim != 2:
                    print(f"[DEBUG] Invalid block: type={type(blk)}, ndim={getattr(blk, 'ndim', None)}")
                    continue

                n_ch, n_samp = blk.shape
                print(f"[DEBUG] Processing block: shape=({n_ch}, {n_samp})")
                
                for i in range(min(n_ch, len(self.channels))):
                    arr = blk[i, :]
                    # Count NaNs in GUI (thread already drops blocks with NaN, but defensive)
                    nan_mask = np.isnan(arr)
                    if nan_mask.any():
                        nan_count = int(nan_mask.sum())
                        self.nan_counts[i] += nan_count
                        print(f"[DEBUG] Channel {i}: {nan_count} NaNs detected")
                        arr = arr[~nan_mask]
                    if arr.size == 0:
                        print(f"[DEBUG] Channel {i}: no valid samples after NaN filtering")
                        continue

                    # Append and trim to window
                    old_size = self.buffers[i].size
                    self.buffers[i] = np.concatenate([self.buffers[i], arr])
                    self._trim_buffer(i)
                    new_size = self.buffers[i].size
                    print(f"[DEBUG] Channel {i}: buffer {old_size} -> {new_size} samples")

                updated = True

            if blocks_processed > 0:
                print(f"[DEBUG] Timer processed {blocks_processed} blocks")

            if updated:
                print("[DEBUG] Updating plots and labels...")
                self._update_plots_and_labels()
                print("[DEBUG] Plots and labels updated")
            
        except Exception as e:
            print(f"[DEBUG] Timer exception: {e}")
            import traceback
            traceback.print_exc()
            self._stop_acq()

    def _trim_buffer(self, i: int) -> None:
        max_len = max(1, int(self.time_window * max(1.0, self.sample_rate)))
        buf = self.buffers[i]
        if buf.size > max_len:
            self.buffers[i] = buf[-max_len:]

    def _update_plots_and_labels(self) -> None:
        sr = max(1.0, float(self.sample_rate))
        tw = float(self.time_window)

        for i, _ch in enumerate(self.channels):
            y = self.buffers[i]
            n = y.size
            if n > 0:
                # Metrics over current window
                y64 = y.astype(np.float64, copy=False)
                rms = float(np.sqrt(np.mean(y64 * y64)))
                mn = float(np.min(y64))
                mx = float(np.max(y64))
                self.lab_min[i].setText(f"{mn:.3f}")
                self.lab_max[i].setText(f"{mx:.3f}")
                self.lab_rms[i].setText(f"{rms:.3f}")
                self.lab_nans[i].setText(str(self.nan_counts[i]))
                self.lab_count[i].setText(str(n))

                # Time axis: [-tw, 0]
                t = (np.arange(n, dtype=np.float32) - n) / sr
                self.curves[i].setData(t, y)
            else:
                self.curves[i].setData([], [])
                self.lab_count[i].setText("0")

        # Keep ranges updated
        for w in self.plot_widgets:
            w.setXRange(-tw, 0.0, padding=0)
            w.setYRange(-self.channel_range, self.channel_range, padding=0)

    # --------------------- Utils ---------------------
    @staticmethod
    def _color_for(i: int):
        palette = [
            (200, 0, 0),
            (0, 160, 0),
            (0, 80, 200),
            (200, 120, 0),
            (160, 0, 160),
            (0, 150, 150),
        ]
        return palette[i % len(palette)]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="GUI adquisición Digilent (Fuerza)")
    parser.add_argument("--channels", "-c", type=str, default="0,1", help="Canales, ej: 0,1")
    parser.add_argument("--sample_rate", "--sr", type=float, default=20000.0, help="Frecuencia de muestreo (Hz)")
    parser.add_argument("--range", "-r", dest="channel_range", type=float, default=5.0, help="Rango del canal (±V)")
    parser.add_argument("--window", "-w", type=float, default=5.0, help="Ventana de tiempo (s)")
    parser.add_argument("--queue_size", type=int, default=32, help="Tamaño de la cola hilo->GUI")
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=0,
        help="Tamaño del buffer del dispositivo (0=por defecto del dispositivo)",
    )
    args = parser.parse_args()

    ch = parse_channels(args.channels)
    buf = None if int(args.buffer_size) == 0 else int(args.buffer_size)

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    win = DigilentForceGUI(
        default_channels=ch,
        sample_rate=float(args.sample_rate),
        channel_range=float(args.channel_range),
        time_window=float(args.window),
        queue_size=int(args.queue_size),
        device_buffer=buf,
    )
    win.resize(1200, 800)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

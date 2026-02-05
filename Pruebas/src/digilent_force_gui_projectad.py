"""
Digilent Force GUI usando implementación DWF basada completamente en projectAD.

Reemplaza pydwf wrapper con llamadas directas a DWF C library via ctypes,
exactamente como hace projectAD para evitar pérdida de datos.
"""

import sys
import queue
import numpy as np
from collections import deque
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from digilent_dwf_thread import DigilentDWFThread


class DigilentForceGUIProjectAD(QtWidgets.QMainWindow):
    """GUI para adquisición de fuerza usando Digilent con implementación projectAD."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digilent Force Acquisition (ProjectAD Implementation)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Parameters
        self.channels = [0, 1]  # Default channels
        self.sample_rate = 20000.0
        self.channel_range = 5.0
        self.time_window = 5.0  # seconds to display
        self.acquisition_time = 0.05  # 50ms per block
        
        # Data storage
        self.max_points = int(self.sample_rate * self.time_window)
        self.data_buffers = {ch: deque(maxlen=self.max_points) for ch in self.channels}
        self.time_buffer = deque(maxlen=self.max_points)
        self.time_offset = 0.0
        
        # Threading
        self.queue = queue.Queue(maxsize=20)
        self.th = None
        
        # Statistics
        self.stats = {ch: {'min': 0, 'max': 0, 'rms': 0, 'nans': 0, 'samples': 0} for ch in self.channels}
        
        self._setup_ui()
        self._setup_timer()
        
    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        control_layout = QtWidgets.QHBoxLayout()
        
        # Parameters
        control_layout.addWidget(QtWidgets.QLabel("Channels:"))
        self.le_channels = QtWidgets.QLineEdit("0,1")
        control_layout.addWidget(self.le_channels)
        
        control_layout.addWidget(QtWidgets.QLabel("Sample Rate:"))
        self.le_sample_rate = QtWidgets.QLineEdit("20000")
        control_layout.addWidget(self.le_sample_rate)
        
        control_layout.addWidget(QtWidgets.QLabel("Range (V):"))
        self.le_range = QtWidgets.QLineEdit("5.0")
        control_layout.addWidget(self.le_range)
        
        control_layout.addWidget(QtWidgets.QLabel("Time Window (s):"))
        self.le_time_window = QtWidgets.QLineEdit("5.0")
        control_layout.addWidget(self.le_time_window)
        
        # Buttons
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        
        self.btn_start.clicked.connect(self._start_acq)
        self.btn_stop.clicked.connect(self._stop_acq)
        
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        
        # Status
        self.lbl_status = QtWidgets.QLabel("Ready")
        control_layout.addWidget(self.lbl_status)
        
        main_layout.addLayout(control_layout)
        
        # Splitter for plots and info
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Voltage', units='V')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(True, True)
        
        # Plot curves for each channel
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        self.plot_curves = {}
        for i, ch in enumerate(self.channels):
            color = colors[i % len(colors)]
            self.plot_curves[ch] = self.plot_widget.plot([], [], pen=color, name=f'CH{ch}')
        
        splitter.addWidget(self.plot_widget)
        
        # Info panel
        self.info_text = QtWidgets.QTextEdit()
        self.info_text.setMaximumWidth(300)
        self.info_text.setReadOnly(True)
        splitter.addWidget(self.info_text)
        
        # Set splitter sizes
        splitter.setSizes([800, 300])
        
    def _setup_timer(self):
        """Setup timer for data processing."""
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_timer)
        
    def _get_parameters(self):
        """Get parameters from UI."""
        try:
            channels_str = self.le_channels.text()
            self.channels = [int(x.strip()) for x in channels_str.split(',')]
            
            self.sample_rate = float(self.le_sample_rate.text())
            self.channel_range = float(self.le_range.text())
            self.time_window = float(self.le_time_window.text())
            
            # Recalculate buffers
            self.max_points = int(self.sample_rate * self.time_window)
            self.data_buffers = {ch: deque(maxlen=self.max_points) for ch in self.channels}
            self.time_buffer = deque(maxlen=self.max_points)
            
            # Reset statistics
            self.stats = {ch: {'min': 0, 'max': 0, 'rms': 0, 'nans': 0, 'samples': 0} for ch in self.channels}
            
            return True
        except Exception as e:
            self.lbl_status.setText(f"Parameter error: {e}")
            return False
    
    def _start_acq(self):
        """Start data acquisition."""
        print("[DEBUG] _start_acq called")
        
        if not self._get_parameters():
            return
        
        # Clear queue
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        
        # Reset data
        self.time_offset = 0.0
        for ch in self.channels:
            self.data_buffers[ch].clear()
        self.time_buffer.clear()
        
        try:
            print("[DEBUG] Creating DWF thread...")
            self.th = DigilentDWFThread(
                data_queue=self.queue,
                channels=self.channels,
                sample_rate=self.sample_rate,
                channel_range=self.channel_range,
                acquisition_time=self.acquisition_time,
                verbose=True,
            )
            
            print("[DEBUG] Starting thread...")
            self.th.start()
            
            print("[DEBUG] Starting timer...")
            self.timer.start(50)  # 20 FPS
            
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("Acquiring... (ProjectAD Implementation)")
            
            print("[DEBUG] Acquisition started")
            
        except Exception as e:
            print(f"[DEBUG] Start error: {e}")
            import traceback
            traceback.print_exc()
            self.lbl_status.setText(f"Start error: {e}")
    
    def _stop_acq(self):
        """Stop data acquisition."""
        print("[DEBUG] _stop_acq called")
        
        self.timer.stop()
        
        if self.th is not None:
            print("[DEBUG] Stopping thread...")
            self.th.stop()
            self.th.join(timeout=3.0)
            self.th = None
            print("[DEBUG] Thread stopped")
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopped")
        
        print("[DEBUG] Acquisition stopped")
    
    def _on_timer(self):
        """Timer callback to process data."""
        try:
            # Check for thread exception
            if self.th is not None and getattr(self.th, "exception", None):
                print(f"[DEBUG] Thread exception detected: {self.th.exception}")
                self._stop_acq()
                return
            
            # Process available data
            blocks_processed = 0
            updated = False
            
            while blocks_processed < 5:  # Limit blocks per timer tick
                try:
                    blk = self.queue.get_nowait()
                    self._process_data_block(blk)
                    updated = True
                    blocks_processed += 1
                except queue.Empty:
                    break
            
            if updated:
                self._update_plots()
                self._update_info()
                
        except Exception as e:
            print(f"[DEBUG] Timer exception: {e}")
            import traceback
            traceback.print_exc()
            self._stop_acq()
    
    def _process_data_block(self, block: np.ndarray):
        """Process a single data block."""
        n_channels, n_samples = block.shape
        
        # Generate time stamps for this block
        dt = 1.0 / self.sample_rate
        block_times = np.arange(n_samples) * dt + self.time_offset
        self.time_offset += n_samples * dt
        
        # Add to buffers
        for t in block_times:
            self.time_buffer.append(t)
        
        for i, ch in enumerate(self.channels[:n_channels]):
            channel_data = block[i]
            
            # Add to buffer
            for sample in channel_data:
                self.data_buffers[ch].append(sample)
            
            # Update statistics
            if len(channel_data) > 0:
                valid_data = channel_data[~np.isnan(channel_data)]
                if len(valid_data) > 0:
                    self.stats[ch]['min'] = float(np.min(valid_data))
                    self.stats[ch]['max'] = float(np.max(valid_data))
                    self.stats[ch]['rms'] = float(np.sqrt(np.mean(valid_data**2)))
                
                self.stats[ch]['nans'] = int(np.sum(np.isnan(channel_data)))
                self.stats[ch]['samples'] += len(channel_data)
    
    def _update_plots(self):
        """Update plot curves."""
        if len(self.time_buffer) == 0:
            return
        
        time_array = np.array(list(self.time_buffer))
        
        for ch in self.channels:
            if ch in self.data_buffers and len(self.data_buffers[ch]) > 0:
                data_array = np.array(list(self.data_buffers[ch]))
                
                # Make sure arrays have same length
                min_len = min(len(time_array), len(data_array))
                if min_len > 0:
                    self.plot_curves[ch].setData(time_array[-min_len:], data_array[-min_len:])
    
    def _update_info(self):
        """Update information panel."""
        info_text = "=== Channel Statistics ===\n\n"
        
        for ch in self.channels:
            stats = self.stats[ch]
            info_text += f"Channel {ch}:\n"
            info_text += f"  Min: {stats['min']:.4f} V\n"
            info_text += f"  Max: {stats['max']:.4f} V\n"
            info_text += f"  RMS: {stats['rms']:.4f} V\n"
            info_text += f"  NaNs: {stats['nans']}\n"
            info_text += f"  Samples: {stats['samples']}\n\n"
        
        info_text += f"=== Acquisition Info ===\n"
        info_text += f"Sample Rate: {self.sample_rate:.0f} Hz\n"
        info_text += f"Channels: {self.channels}\n"
        info_text += f"Range: ±{self.channel_range} V\n"
        info_text += f"Time Window: {self.time_window} s\n"
        info_text += f"Acquisition Block: {self.acquisition_time*1000:.1f} ms\n"
        
        self.info_text.setPlainText(info_text)
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.th is not None:
            self._stop_acq()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = DigilentForceGUIProjectAD()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

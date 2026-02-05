"""
GUI simplificado para depurar problemas con DigilentAnalogInThread.
"""
import sys
import queue
import time
from typing import List

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


class SimpleDigilentGUI(QtWidgets.QMainWindow):
    """GUI mínimo para depurar DigilentAnalogInThread."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Digilent Debug")
        self.resize(800, 600)
        
        # State
        self.queue = None
        self.th = None
        self.data_count = 0
        
        self._setup_ui()
        
        # Timer más lento para debug
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(500)  # 2 Hz - muy lento para debug
        self.timer.timeout.connect(self._on_timer)
        
    def _setup_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.lbl_status = QtWidgets.QLabel("Ready")
        
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.lbl_status)
        controls.addStretch()
        layout.addLayout(controls)
        
        # Simple plot
        self.plot = pg.PlotWidget(title="Channel 0")
        self.plot.setLabel('left', 'Voltage', units='V')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen='r')
        layout.addWidget(self.plot)
        
        # Log area
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
        
        # Connect signals
        self.btn_start.clicked.connect(self._start_acq)
        self.btn_stop.clicked.connect(self._stop_acq)
        
    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg)  # Also print to console
        self.log_text.append(full_msg)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def _start_acq(self):
        self._log("Starting acquisition...")
        
        try:
            self._log("Creating queue...")
            self.queue = queue.Queue(maxsize=4)  # Small queue for debug
            
            self._log("Importing DigilentAnalogInThread...")
            from digilent_analogin_thread import DigilentAnalogInThread
            
            self._log("Creating thread...")
            self.th = DigilentAnalogInThread(
                data_queue=self.queue,
                channels=[0],  # Single channel for simplicity
                sample_rate=5000.0,  # Lower sample rate
                channel_range=5.0,
                buffer_size=10000,  # Smaller buffer
                poll_interval_s=0.01,  # Slower polling
                verbose=True,
            )
            
            self._log("Starting thread...")
            self.th.start()
            
            self._log("Thread started, updating UI...")
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("Acquiring...")
            
            self._log("Starting timer...")
            self.timer.start()
            self._log("Acquisition started successfully!")
            
        except Exception as e:
            self._log(f"ERROR starting acquisition: {e}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
            self.queue = None
            self.th = None
            
    def _stop_acq(self):
        self._log("Stopping acquisition...")
        
        self.timer.stop()
        self._log("Timer stopped")
        
        if self.th is not None:
            try:
                self._log("Stopping thread...")
                self.th.stop()
                self._log("Joining thread...")
                self.th.join(timeout=3.0)
                if self.th.is_alive():
                    self._log("WARNING: Thread did not join within timeout")
                else:
                    self._log("Thread joined successfully")
            except Exception as e:
                self._log(f"Error stopping thread: {e}")
                
        if self.th is not None and hasattr(self.th, 'exception') and self.th.exception:
            self._log(f"Thread had exception: {self.th.exception}")
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopped")
        self.th = None
        self.queue = None
        self._log("Stop completed")
        
    def _on_timer(self):
        try:
            if self.queue is None:
                self._log("Timer: queue is None")
                return
                
            if self.th is not None and not self.th.is_alive():
                self._log("Timer: thread died")
                if hasattr(self.th, 'exception') and self.th.exception:
                    self._log(f"Timer: thread exception: {self.th.exception}")
                self._stop_acq()
                return
                
            blocks_got = 0
            total_samples = 0
            
            # Process all available blocks
            while True:
                try:
                    blk = self.queue.get_nowait()
                    blocks_got += 1
                    
                    if isinstance(blk, np.ndarray) and blk.ndim == 2:
                        n_ch, n_samp = blk.shape
                        total_samples += n_samp
                        self.data_count += n_samp
                        
                        # Simple plot update - just use channel 0
                        if n_ch > 0:
                            y = blk[0, :]
                            if y.size > 0:
                                # Create simple time axis
                                t = np.arange(y.size) / 5000.0  # Assuming 5kHz
                                self.curve.setData(t, y)
                                
                except queue.Empty:
                    break
                except Exception as e:
                    self._log(f"Error processing block: {e}")
                    break
                    
            if blocks_got > 0:
                self._log(f"Timer: got {blocks_got} blocks, {total_samples} samples, total: {self.data_count}")
                self.lbl_status.setText(f"Acquiring... ({self.data_count} samples)")
                
        except Exception as e:
            self._log(f"Timer exception: {e}")
            import traceback
            self._log(f"Timer traceback: {traceback.format_exc()}")
            self._stop_acq()
            
    def closeEvent(self, event):
        if self.th is not None:
            self._stop_acq()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    
    print("Creating Simple Digilent GUI...")
    gui = SimpleDigilentGUI()
    gui.show()
    
    print("GUI created and shown. Starting event loop...")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

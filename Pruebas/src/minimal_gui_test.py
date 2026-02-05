"""
Test mínimo para identificar dónde se traba el GUI.
"""
import sys
import queue
import time

print("Starting minimal GUI test...")

# Test 1: Basic imports
try:
    print("Testing PyQt5 import...")
    from PyQt5 import QtCore, QtWidgets
    print("✓ PyQt5 imported")
    
    print("Testing pyqtgraph import...")
    import pyqtgraph as pg
    print("✓ pyqtgraph imported")
    
    print("Testing numpy import...")
    import numpy as np
    print("✓ numpy imported")
    
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: QApplication creation
try:
    print("Creating QApplication...")
    app = QtWidgets.QApplication(sys.argv)
    print("✓ QApplication created")
except Exception as e:
    print(f"✗ QApplication error: {e}")
    sys.exit(1)

# Test 3: Simple window
try:
    print("Creating simple window...")
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Test Window")
    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)
    
    btn = QtWidgets.QPushButton("Test Button")
    layout.addWidget(btn)
    
    print("✓ Simple window created")
    
    print("Showing window...")
    win.show()
    print("✓ Window shown")
    
except Exception as e:
    print(f"✗ Window creation error: {e}")
    sys.exit(1)

# Test 4: pyqtgraph widget
try:
    print("Creating pyqtgraph plot...")
    plot = pg.PlotWidget(title="Test Plot")
    plot.setLabel('left', 'Y')
    plot.setLabel('bottom', 'X')
    plot.showGrid(x=True, y=True)
    layout.addWidget(plot)
    print("✓ PyQtGraph plot created")
    
except Exception as e:
    print(f"✗ PyQtGraph error: {e}")
    sys.exit(1)

# Test 5: Simple timer
class TestTimer(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timeout)
        
    def on_timeout(self):
        self.count += 1
        print(f"Timer tick {self.count}")
        if self.count >= 3:
            print("Timer test complete")
            QtWidgets.QApplication.quit()

try:
    print("Testing timer...")
    timer_test = TestTimer()
    timer_test.timer.start(1000)  # 1 second intervals
    print("✓ Timer started")
    
except Exception as e:
    print(f"✗ Timer error: {e}")
    sys.exit(1)

print("All tests passed, starting event loop...")
print("Window should be visible, timer will quit after 3 seconds...")
sys.exit(app.exec_())

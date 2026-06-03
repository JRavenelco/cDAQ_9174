import sys
import os
import time
import json
import queue
import threading
import re
from datetime import datetime
from collections import deque

import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QTextEdit, QMessageBox, QCheckBox, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

try:
    import serial
    from serial.tools import list_ports
    HAS_PYSERIAL = True
except ImportError:
    HAS_PYSERIAL = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(BASE_DIR, "caracterizacion_fuerza", "rugosimetro_tmr200")
os.makedirs(DATOS_DIR, exist_ok=True)

METRIC_NAMES = [
    "Ra", "Rq", "Rz", "Rt", "Rp", "Rv", "Ry", "Rmax", "Rc",
    # Parametros adicionales que emite el TR200 (rugosidad + estadisticos + Abbott)
    "R3z", "RS", "RSm", "RSk", "Rku", "Rmr", "Rk", "Rpk", "Rvk",
    "Sm", "S", "Wa", "Wq", "Wt", "Pa", "Pq", "Pz"
]


class TMR200SerialThread(threading.Thread):
    def __init__(self, port, baudrate, timeout, xonxoff, result_queue):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.xonxoff = xonxoff
        self.result_queue = result_queue
        self.command_queue = queue.Queue()
        self.running = True
        self.ser = None

    def run(self):
        if not HAS_PYSERIAL:
            self.result_queue.put({
                "type": "error",
                "message": "pyserial no está instalado"
            })
            return

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                xonxoff=self.xonxoff,
            )
            self.result_queue.put({
                "type": "status",
                "message": f"Conectado a {self.port} @ {self.baudrate} bps"
            })
        except Exception as e:
            self.result_queue.put({
                "type": "error",
                "message": f"No se pudo abrir {self.port}: {e}"
            })
            return

        while self.running:
            try:
                try:
                    item = self.command_queue.get(timeout=0.1)
                except queue.Empty:
                    item = None

                if item is not None:
                    if item["type"] == "command":
                        self._handle_command(item)
                    elif item["type"] == "close":
                        break

                if self.ser and self.ser.in_waiting:
                    payload = self._read_until_idle(max_wait=1.2, idle_gap=0.18)
                    if payload:
                        self.result_queue.put({
                            "type": "response",
                            "timestamp": datetime.now().isoformat(),
                            "raw_bytes": payload,
                            "raw_text": self._decode_payload(payload),
                            "source": "stream"
                        })
            except Exception as e:
                self.result_queue.put({
                    "type": "error",
                    "message": f"Error de comunicación: {e}"
                })
                break

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.result_queue.put({
            "type": "status",
            "message": "Puerto serial cerrado"
        })

    def _handle_command(self, item):
        command = item.get("command", "")
        source = item.get("source", "manual")
        if not self.ser or not self.ser.is_open:
            self.result_queue.put({
                "type": "error",
                "message": "Puerto no disponible"
            })
            return

        payload = command.encode("ascii", errors="ignore")
        self.ser.reset_input_buffer()
        self.ser.write(payload)
        self.ser.flush()
        time.sleep(item.get("settle_s", 0.15))
        response = self._read_until_idle(
            max_wait=item.get("max_wait", 2.5),
            idle_gap=item.get("idle_gap", 0.2)
        )
        self.result_queue.put({
            "type": "response",
            "timestamp": datetime.now().isoformat(),
            "raw_bytes": response,
            "raw_text": self._decode_payload(response),
            "source": source,
            "command": command
        })

    def _read_until_idle(self, max_wait=2.5, idle_gap=0.2):
        start = time.time()
        last_rx = None
        data = bytearray()
        while self.running and (time.time() - start) < max_wait:
            waiting = self.ser.in_waiting if self.ser else 0
            if waiting:
                chunk = self.ser.read(waiting)
                if chunk:
                    data.extend(chunk)
                    last_rx = time.time()
            else:
                if last_rx is not None and (time.time() - last_rx) >= idle_gap:
                    break
                time.sleep(0.03)
        return bytes(data)

    @staticmethod
    def _decode_payload(payload):
        if not payload:
            return ""
        for encoding in ("ascii", "utf-8", "latin-1"):
            try:
                return payload.decode(encoding, errors="replace").strip()
            except Exception:
                continue
        return repr(payload)

    def enqueue_command(self, command, source="manual", max_wait=2.5, idle_gap=0.2, settle_s=0.15):
        self.command_queue.put({
            "type": "command",
            "command": command,
            "source": source,
            "max_wait": max_wait,
            "idle_gap": idle_gap,
            "settle_s": settle_s,
        })

    def stop(self):
        self.running = False
        try:
            self.command_queue.put_nowait({"type": "close"})
        except Exception:
            pass


class CaracterizacionRugosimetroGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rugosímetro TMR200 - Interfaz de Adquisición")
        self.setGeometry(120, 120, 1500, 900)

        self.serial_thread = None
        self.serial_queue = queue.Queue()
        self.measurements = []
        self.metric_history = {name: deque(maxlen=200) for name in METRIC_NAMES}
        self.current_metrics = {}
        self.polling = False

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.process_serial_events)
        self.update_timer.start(100)

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self.request_data)

        self.apply_dark_theme()
        self._setup_ui()
        self.refresh_ports()
        self._update_dependency_status()

    def apply_dark_theme(self):
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
        if HAS_PYQTGRAPH:
            pg.setConfigOption('background', '#2b2b2b')
            pg.setConfigOption('foreground', 'w')

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        config_group = QGroupBox("Configuración de comunicación")
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("Puerto COM:"), 0, 0)
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        config_layout.addWidget(self.port_combo, 0, 1)

        self.refresh_btn = QPushButton("Actualizar puertos")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        config_layout.addWidget(self.refresh_btn, 0, 2)

        config_layout.addWidget(QLabel("Baudrate:"), 0, 3)
        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(1200, 115200)
        self.baud_spin.setValue(115200)
        config_layout.addWidget(self.baud_spin, 0, 4)

        config_layout.addWidget(QLabel("Timeout (s):"), 0, 5)
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 10.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setValue(2.0)
        config_layout.addWidget(self.timeout_spin, 0, 6)

        self.xonxoff_cb = QCheckBox("Usar XON/XOFF")
        self.xonxoff_cb.setChecked(False)
        config_layout.addWidget(self.xonxoff_cb, 0, 7)

        self.connect_btn = QPushButton("Conectar")
        self.connect_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 8px;")
        self.connect_btn.clicked.connect(self.connect_serial)
        config_layout.addWidget(self.connect_btn, 1, 0)

        self.disconnect_btn = QPushButton("Desconectar")
        self.disconnect_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 8px;")
        self.disconnect_btn.clicked.connect(self.disconnect_serial)
        self.disconnect_btn.setEnabled(False)
        config_layout.addWidget(self.disconnect_btn, 1, 1)

        config_layout.addWidget(QLabel("Comando lectura:"), 1, 2)
        self.read_cmd_edit = QLineEdit("GET_DATA\\r\\n")
        config_layout.addWidget(self.read_cmd_edit, 1, 3, 1, 2)

        config_layout.addWidget(QLabel("Comando estado:"), 1, 5)
        self.status_cmd_edit = QLineEdit("STAT\\r\\n")
        config_layout.addWidget(self.status_cmd_edit, 1, 6)

        self.status_btn = QPushButton("Solicitar estado")
        self.status_btn.clicked.connect(self.request_status)
        config_layout.addWidget(self.status_btn, 1, 7)

        config_layout.addWidget(QLabel("Comando personalizado:"), 2, 0)
        self.custom_cmd_edit = QLineEdit("DATA?\\r\\n")
        config_layout.addWidget(self.custom_cmd_edit, 2, 1, 1, 3)

        self.send_custom_btn = QPushButton("Enviar comando")
        self.send_custom_btn.clicked.connect(self.send_custom_command)
        config_layout.addWidget(self.send_custom_btn, 2, 4)

        config_layout.addWidget(QLabel("Periodo polling (ms):"), 2, 5)
        self.poll_spin = QSpinBox()
        self.poll_spin.setRange(200, 10000)
        self.poll_spin.setValue(1500)
        config_layout.addWidget(self.poll_spin, 2, 6)

        self.poll_btn = QPushButton("Iniciar polling")
        self.poll_btn.clicked.connect(self.toggle_polling)
        config_layout.addWidget(self.poll_btn, 2, 7)

        main_layout.addWidget(config_group)

        meta_group = QGroupBox("Registro de medición")
        meta_layout = QGridLayout(meta_group)

        meta_layout.addWidget(QLabel("Pieza / ID:"), 0, 0)
        self.part_edit = QLineEdit()
        meta_layout.addWidget(self.part_edit, 0, 1)

        meta_layout.addWidget(QLabel("Operador:"), 0, 2)
        self.operator_edit = QLineEdit()
        meta_layout.addWidget(self.operator_edit, 0, 3)

        meta_layout.addWidget(QLabel("Material:"), 0, 4)
        self.material_edit = QLineEdit()
        meta_layout.addWidget(self.material_edit, 0, 5)

        meta_layout.addWidget(QLabel("Notas:"), 1, 0)
        self.notes_edit = QLineEdit()
        meta_layout.addWidget(self.notes_edit, 1, 1, 1, 5)

        self.capture_btn = QPushButton("Capturar medición actual")
        self.capture_btn.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 8px;")
        self.capture_btn.clicked.connect(self.capture_measurement)
        meta_layout.addWidget(self.capture_btn, 1, 6)

        self.save_btn = QPushButton("Guardar historial")
        self.save_btn.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; padding: 8px;")
        self.save_btn.clicked.connect(self.save_history)
        meta_layout.addWidget(self.save_btn, 0, 6)

        main_layout.addWidget(meta_group)

        info_group = QGroupBox("Estado y valores actuales")
        info_layout = QGridLayout(info_group)

        self.connection_label = QLabel("Desconectado")
        self.connection_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        info_layout.addWidget(QLabel("Conexión:"), 0, 0)
        info_layout.addWidget(self.connection_label, 0, 1)

        self.libs_label = QLabel("")
        info_layout.addWidget(QLabel("Dependencias:"), 0, 2)
        info_layout.addWidget(self.libs_label, 0, 3, 1, 4)

        self.last_command_label = QLabel("-")
        info_layout.addWidget(QLabel("Último comando:"), 1, 0)
        info_layout.addWidget(self.last_command_label, 1, 1)

        self.last_timestamp_label = QLabel("-")
        info_layout.addWidget(QLabel("Última respuesta:"), 1, 2)
        info_layout.addWidget(self.last_timestamp_label, 1, 3)

        self.metric_summary_label = QLabel("Sin métricas detectadas")
        self.metric_summary_label.setStyleSheet("font-weight: bold; color: #2ecc71;")
        self.metric_summary_label.setWordWrap(True)
        info_layout.addWidget(QLabel("Resumen:"), 2, 0)
        info_layout.addWidget(self.metric_summary_label, 2, 1, 1, 6)

        main_layout.addWidget(info_group)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab { background: #3a3a3a; color: white; padding: 8px 20px; font-weight: bold; } QTabBar::tab:selected { background: #2980b9; }")
        self._setup_metrics_tab()
        self._setup_history_tab()
        self._setup_log_tab()
        main_layout.addWidget(self.tabs)

        self.status_label = QLabel(f"Datos en: {DATOS_DIR}")
        self.statusBar().addWidget(self.status_label)

    def _setup_metrics_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor", "Unidad"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.metrics_table, 1)

        right_layout = QVBoxLayout()

        self.metric_selector = QComboBox()
        self.metric_selector.addItems(METRIC_NAMES)
        self.metric_selector.currentTextChanged.connect(self.update_plot)
        right_layout.addWidget(self.metric_selector)

        if HAS_PYQTGRAPH:
            self.metric_plot = pg.PlotWidget(title="Historial de métrica")
            self.metric_plot.showGrid(x=True, y=True, alpha=0.3)
            self.metric_curve = self.metric_plot.plot(pen=pg.mkPen('#3498db', width=2))
            right_layout.addWidget(self.metric_plot, 1)
        else:
            self.metric_plot = None
            fallback = QLabel("pyqtgraph no disponible")
            fallback.setAlignment(Qt.AlignCenter)
            right_layout.addWidget(fallback, 1)

        layout.addLayout(right_layout, 2)
        self.tabs.addTab(tab, "Métricas")

    def _setup_history_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels([
            "Timestamp", "Pieza", "Operador", "Material", "Ra", "Rq", "Rz", "Origen"
        ])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.history_table)

        self.tabs.addTab(tab, "Historial")

    def _setup_log_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.tabs.addTab(tab, "Log")

    def _update_dependency_status(self):
        deps = []
        deps.append("PySerial OK" if HAS_PYSERIAL else "PySerial faltante")
        deps.append("PyQtGraph OK" if HAS_PYQTGRAPH else "PyQtGraph faltante")
        deps.append("Pandas OK")
        self.libs_label.setText(" | ".join(deps))

    def refresh_ports(self):
        current = self.port_combo.currentText().strip()
        self.port_combo.clear()
        if not HAS_PYSERIAL:
            self.port_combo.addItem("COM3")
            return
        ports = [p.device for p in list_ports.comports()]
        if not ports:
            ports = ["COM3"]
        self.port_combo.addItems(ports)
        if current:
            idx = self.port_combo.findText(current)
            if idx >= 0:
                self.port_combo.setCurrentIndex(idx)
            else:
                self.port_combo.setEditText(current)
        else:
            # Preseleccionar COM21 (TR200 por Bluetooth) si esta disponible
            idx_tr = self.port_combo.findText("COM21")
            if idx_tr >= 0:
                self.port_combo.setCurrentIndex(idx_tr)

    def connect_serial(self):
        port = self.port_combo.currentText().strip()
        if not port:
            QMessageBox.warning(self, "Puerto requerido", "Especifica un puerto COM válido.")
            return
        self.disconnect_serial(silent=True)
        self.serial_thread = TMR200SerialThread(
            port=port,
            baudrate=self.baud_spin.value(),
            timeout=self.timeout_spin.value(),
            xonxoff=self.xonxoff_cb.isChecked(),
            result_queue=self.serial_queue,
        )
        self.serial_thread.start()
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self.connection_label.setText(f"Conectando a {port}...")
        self.connection_label.setStyleSheet("font-weight: bold; color: #f39c12;")

    def disconnect_serial(self, silent=False):
        self.poll_timer.stop()
        self.polling = False
        self.poll_btn.setText("Iniciar polling")
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.join(timeout=2)
            self.serial_thread = None
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.connection_label.setText("Desconectado")
        self.connection_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        if not silent:
            self.status_label.setText("Desconectado")

    def process_serial_events(self):
        while not self.serial_queue.empty():
            item = self.serial_queue.get_nowait()
            if item["type"] == "status":
                self._append_log(item["message"])
                if "Conectado" in item["message"]:
                    self.connection_label.setText(item["message"])
                    self.connection_label.setStyleSheet("font-weight: bold; color: #27ae60;")
                self.status_label.setText(item["message"])
            elif item["type"] == "error":
                self._append_log(item["message"])
                self.connection_label.setText("Error")
                self.connection_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
                self.status_label.setText(item["message"])
                self.connect_btn.setEnabled(True)
                self.disconnect_btn.setEnabled(False)
            elif item["type"] == "response":
                self._handle_response(item)

    def _handle_response(self, item):
        raw_text = item.get("raw_text", "")
        source = item.get("source", "manual")
        timestamp = item.get("timestamp", datetime.now().isoformat())
        command = item.get("command", "")

        if command:
            self.last_command_label.setText(command.replace("\r", "\\r").replace("\n", "\\n"))
        self.last_timestamp_label.setText(timestamp)
        self._append_log(f"[{source}] {raw_text if raw_text else '<sin respuesta>'}")

        metrics = self.extract_metrics(raw_text)
        if metrics:
            self.current_metrics = metrics
            self.metric_summary_label.setText(self.build_metric_summary(metrics))
            self.populate_metrics_table(metrics)
            self.push_metric_history(metrics)
            self.update_plot()
            self.status_label.setText(f"Respuesta recibida con {len(metrics)} métricas")
        else:
            self.metric_summary_label.setText("Sin métricas detectadas en la última respuesta")
            self.status_label.setText("Respuesta recibida sin métricas parseables")

    def send_command(self, command, source="manual"):
        if not self.serial_thread:
            QMessageBox.warning(self, "Sin conexión", "Conecta primero el rugosímetro.")
            return
        resolved = self.decode_command_text(command)
        self.serial_thread.enqueue_command(resolved, source=source)

    def request_data(self):
        self.send_command(self.read_cmd_edit.text(), source="lectura")

    def request_status(self):
        self.send_command(self.status_cmd_edit.text(), source="estado")

    def send_custom_command(self):
        self.send_command(self.custom_cmd_edit.text(), source="custom")

    def toggle_polling(self):
        if not self.polling:
            if not self.serial_thread:
                QMessageBox.warning(self, "Sin conexión", "Conecta primero el rugosímetro.")
                return
            self.poll_timer.start(self.poll_spin.value())
            self.polling = True
            self.poll_btn.setText("Detener polling")
            self.status_label.setText("Polling activo")
        else:
            self.poll_timer.stop()
            self.polling = False
            self.poll_btn.setText("Iniciar polling")
            self.status_label.setText("Polling detenido")

    def capture_measurement(self):
        if not self.current_metrics:
            QMessageBox.warning(self, "Sin datos", "No hay métricas actuales para capturar.")
            return
        record = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "pieza": self.part_edit.text().strip(),
            "operador": self.operator_edit.text().strip(),
            "material": self.material_edit.text().strip(),
            "notas": self.notes_edit.text().strip(),
            "puerto": self.port_combo.currentText().strip(),
            "baudrate": self.baud_spin.value(),
            "xonxoff": self.xonxoff_cb.isChecked(),
            "metrics": dict(self.current_metrics),
            "raw_response": self.log_text.toPlainText().splitlines()[-1] if self.log_text.toPlainText().splitlines() else "",
            "source": self.last_command_label.text(),
        }
        self.measurements.append(record)
        self.append_history_row(record)
        self.status_label.setText(f"Medición capturada: {len(self.measurements)} registros")

    def save_history(self):
        if not self.measurements:
            QMessageBox.warning(self, "Sin historial", "No hay mediciones capturadas para guardar.")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rows = []
        for item in self.measurements:
            row = {
                "timestamp": item["timestamp"],
                "pieza": item["pieza"],
                "operador": item["operador"],
                "material": item["material"],
                "notas": item["notas"],
                "puerto": item["puerto"],
                "baudrate": item["baudrate"],
                "xonxoff": item["xonxoff"],
                "source": item["source"],
                "raw_response": item["raw_response"],
            }
            for name in METRIC_NAMES:
                metric = item["metrics"].get(name, {})
                row[name] = metric.get("value")
                row[f"{name}_unit"] = metric.get("unit", "")
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = os.path.join(DATOS_DIR, f"rugosimetro_{timestamp}.csv")
        json_path = os.path.join(DATOS_DIR, f"rugosimetro_{timestamp}_meta.json")
        df.to_csv(csv_path, index=False, sep='\t')

        meta = {
            "timestamp": timestamp,
            "n_mediciones": len(self.measurements),
            "equipo": "TMR200",
            "configuracion_serial": {
                "port": self.port_combo.currentText().strip(),
                "baudrate": self.baud_spin.value(),
                "timeout_s": self.timeout_spin.value(),
                "xonxoff": self.xonxoff_cb.isChecked(),
            },
            "comandos": {
                "lectura": self.read_cmd_edit.text(),
                "estado": self.status_cmd_edit.text(),
            },
            "dependencias": {
                "pyserial": HAS_PYSERIAL,
                "pyqtgraph": HAS_PYQTGRAPH,
                "pandas": True,
            }
        }
        with open(json_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

        QMessageBox.information(self, "Guardado", f"Archivos guardados:\n{csv_path}\n{json_path}")
        self.status_label.setText(f"Historial guardado: {csv_path}")

    def populate_metrics_table(self, metrics):
        ordered = sorted(metrics.items(), key=lambda kv: kv[0])
        self.metrics_table.setRowCount(len(ordered))
        for row, (name, metric) in enumerate(ordered):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(str(metric.get("value", ""))))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(metric.get("unit", "")))

    def append_history_row(self, record):
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        metrics = record.get("metrics", {})
        values = [
            record.get("timestamp", ""),
            record.get("pieza", ""),
            record.get("operador", ""),
            record.get("material", ""),
            self.metric_value_text(metrics, "Ra"),
            self.metric_value_text(metrics, "Rq"),
            self.metric_value_text(metrics, "Rz"),
            record.get("source", ""),
        ]
        for col, value in enumerate(values):
            self.history_table.setItem(row, col, QTableWidgetItem(value))

    def push_metric_history(self, metrics):
        for name in METRIC_NAMES:
            metric = metrics.get(name)
            if metric is not None:
                self.metric_history[name].append(metric.get("value"))

    def update_plot(self):
        if not HAS_PYQTGRAPH or self.metric_plot is None:
            return
        metric_name = self.metric_selector.currentText()
        values = [v for v in self.metric_history.get(metric_name, []) if v is not None]
        if not values:
            self.metric_curve.setData([], [])
            return
        x = list(range(1, len(values) + 1))
        self.metric_curve.setData(x, values)
        self.metric_plot.setTitle(f"Historial de {metric_name}")

    def build_metric_summary(self, metrics):
        preferred = [name for name in ("Ra", "Rq", "Rz", "Rt") if name in metrics]
        if not preferred:
            preferred = list(metrics.keys())[:4]
        parts = []
        for name in preferred:
            metric = metrics[name]
            unit = f" {metric['unit']}" if metric.get("unit") else ""
            parts.append(f"{name}={metric['value']}{unit}")
        return " | ".join(parts)

    def metric_value_text(self, metrics, name):
        metric = metrics.get(name)
        if not metric:
            return ""
        unit = metric.get("unit", "")
        return f"{metric.get('value', '')} {unit}".strip()

    def extract_metrics(self, text):
        metrics = {}
        if not text:
            return metrics
        clean = text.replace('\r', '\n')
        # Patron construido desde METRIC_NAMES, ordenado por longitud
        # (RSm antes que RS antes que S) para evitar matches parciales.
        _names_alt = '|'.join(re.escape(n) for n in sorted(METRIC_NAMES, key=len, reverse=True))
        pattern = re.compile(r'\b(' + _names_alt + r')\b\s*[:=]?\s*([-+]?\d+(?:[\.,]\d+)?)\s*([a-zA-Zµμumin\/%]*)', re.IGNORECASE)
        for match in pattern.finditer(clean):
            name = match.group(1)
            value = float(match.group(2).replace(',', '.'))
            unit = match.group(3).strip()
            metrics[name] = {"value": value, "unit": unit}
        if metrics:
            return metrics
        lines = [ln.strip() for ln in clean.split('\n') if ln.strip()]
        for line in lines:
            parts = re.split(r'[:=]', line, maxsplit=1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            value_match = re.search(r'([-+]?\d+(?:[\.,]\d+)?)\s*([a-zA-Zµμumin\/%]*)', parts[1].strip())
            if not value_match:
                continue
            if key in METRIC_NAMES:
                metrics[key] = {
                    "value": float(value_match.group(1).replace(',', '.')),
                    "unit": value_match.group(2).strip(),
                }
        return metrics

    def decode_command_text(self, text):
        return text.replace('\\r', '\r').replace('\\n', '\n').replace('\\t', '\t')

    def _append_log(self, message):
        stamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{stamp}] {message}")

    def closeEvent(self, event):
        self.disconnect_serial(silent=True)
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CaracterizacionRugosimetroGUI()
    window.show()
    sys.exit(app.exec_())

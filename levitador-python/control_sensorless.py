"""
Control SENSORLESS del Levitador Magnético

El PID usa la estimación del observador en vez del sensor físico.
El sensor solo se lee para comparación/logging.

Modos:
1. SEMI-SENSORLESS: Luenberger corrige con sensor, PID usa y_est
2. FULL-SENSORLESS: Luenberger NO usa sensor, PID usa y_est (solo corriente)

Autor: José de Jesús Santana Ramírez
"""

import time
import threading
import queue
import ctypes
from ctypes import wintypes
import math
import sys

# =============================================================================
# PARÁMETROS DEL MODELO
# =============================================================================
K0 = 0.0704
K = 0.0327
A = 0.0052
R = 2.72
M = 0.018
G = 9.81

# =============================================================================
# PARÁMETROS PID
# =============================================================================
Ts = 0.01
kp = 100
ki = 50
kd = 1.5
kpi = 12.0
kii = 3000.0
Vref = 9.86
Iref = 0.827
Rs = 2.2

# =============================================================================
# FUNCIONES DEL MODELO
# =============================================================================

def inductancia(y):
    y = max(0.0001, y)
    return K0 + K / (1 + y / A)

def dL_dy(y):
    y = max(0.0001, y)
    return -K / (A * (1 + y / A)**2)

def aceleracion(y, i):
    dL = dL_dy(y)
    return (1/(2*M)) * dL * i**2 + G

# =============================================================================
# OBSERVADOR DE LUENBERGER (con opción sensorless)
# =============================================================================

class ObservadorLuenberger:
    """
    Observador de Luenberger con modo sensorless.
    
    use_sensor=True:  Corrige con medición (semi-sensorless)
    use_sensor=False: Solo modelo físico (full-sensorless)
    """
    
    def __init__(self, polo1=-80, polo2=-80, dt=Ts, use_sensor=True):
        self.dt = dt
        self.use_sensor = use_sensor
        
        # Ganancias (solo se usan si use_sensor=True)
        self.L1 = -(polo1 + polo2)
        self.L2 = polo1 * polo2
        
        self.y_est = 0.005
        self.v_est = 0.0
        self.initialized = False
        
        modo = "CON sensor" if use_sensor else "SIN sensor (full sensorless)"
        print(f"📐 Luenberger {modo}")
    
    def estimar(self, y_medido, i):
        """
        Estima [y, v]. Si use_sensor=False, ignora y_medido.
        """
        if not self.initialized:
            if self.use_sensor:
                self.y_est = y_medido
            else:
                # Sin sensor: inicializar desde modelo de equilibrio
                self.y_est = self._estimar_equilibrio(i) or 0.005
            self.v_est = 0.0
            self.initialized = True
            return self.y_est, self.v_est
        
        # Modelo físico
        a = aceleracion(self.y_est, i)
        
        if self.use_sensor:
            # Con corrección del sensor
            e = y_medido - self.y_est
            dy_dt = self.v_est + self.L1 * e
            dv_dt = a + self.L2 * e
        else:
            # Sin sensor: solo modelo
            dy_dt = self.v_est
            dv_dt = a
        
        # Integrar
        self.y_est += dy_dt * self.dt
        self.v_est += dv_dt * self.dt
        
        # Limitar a rango físico
        self.y_est = max(0.0005, min(0.022, self.y_est))
        self.v_est = max(-1.5, min(1.5, self.v_est))
        
        return self.y_est, self.v_est
    
    def _estimar_equilibrio(self, i):
        """Estima posición inicial desde corriente de equilibrio."""
        if i < 0.05:
            return 0.010  # Posición por defecto
        try:
            dL_abs = 2 * M * G / (i**2)
            factor = K / (A * dL_abs)
            y = A * (math.sqrt(factor) - 1)
            return max(0.002, min(0.015, y))
        except:
            return 0.005
    
    def reset(self):
        self.y_est = 0.005
        self.v_est = 0.0
        self.initialized = False


# =============================================================================
# SERIAL
# =============================================================================

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

class Win32Serial:
    def __init__(self, port, baudrate):
        self.handle = kernel32.CreateFileW(
            f"\\\\.\\{port}", 0xC0000000, 0, None, 3, 0, None
        )
        if self.handle == -1:
            raise Exception(f"No se pudo abrir {port}")
        
        class DCB(ctypes.Structure):
            _fields_ = [
                ("DCBlength", wintypes.DWORD), ("BaudRate", wintypes.DWORD),
                ("fBinary", wintypes.DWORD, 1), ("fParity", wintypes.DWORD, 1),
                ("fOutxCtsFlow", wintypes.DWORD, 1), ("fOutxDsrFlow", wintypes.DWORD, 1),
                ("fDtrControl", wintypes.DWORD, 2), ("fDsrSensitivity", wintypes.DWORD, 1),
                ("fTXContinueOnXoff", wintypes.DWORD, 1), ("fOutX", wintypes.DWORD, 1),
                ("fInX", wintypes.DWORD, 1), ("fErrorChar", wintypes.DWORD, 1),
                ("fNull", wintypes.DWORD, 1), ("fRtsControl", wintypes.DWORD, 2),
                ("fAbortOnError", wintypes.DWORD, 1), ("fDummy2", wintypes.DWORD, 17),
                ("wReserved", wintypes.WORD), ("XonLim", wintypes.WORD),
                ("XoffLim", wintypes.WORD), ("ByteSize", wintypes.BYTE),
                ("Parity", wintypes.BYTE), ("StopBits", wintypes.BYTE),
                ("XonChar", ctypes.c_char), ("XoffChar", ctypes.c_char),
                ("ErrorChar", ctypes.c_char), ("EofChar", ctypes.c_char),
                ("EvtChar", ctypes.c_char), ("wReserved1", wintypes.WORD),
            ]
        
        dcb = DCB()
        dcb.DCBlength = ctypes.sizeof(DCB)
        kernel32.GetCommState(self.handle, ctypes.byref(dcb))
        dcb.BaudRate = baudrate
        dcb.ByteSize = 8
        dcb.Parity = 0
        dcb.StopBits = 0
        kernel32.SetCommState(self.handle, ctypes.byref(dcb))
        
        class COMMTIMEOUTS(ctypes.Structure):
            _fields_ = [
                ("ReadIntervalTimeout", wintypes.DWORD),
                ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
                ("ReadTotalTimeoutConstant", wintypes.DWORD),
                ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
                ("WriteTotalTimeoutConstant", wintypes.DWORD),
            ]
        
        timeouts = COMMTIMEOUTS()
        timeouts.ReadIntervalTimeout = 50
        timeouts.ReadTotalTimeoutConstant = 1000
        kernel32.SetCommTimeouts(self.handle, ctypes.byref(timeouts))
        self.is_open = True
    
    def read(self, size):
        buf = ctypes.create_string_buffer(size)
        read_count = wintypes.DWORD()
        kernel32.ReadFile(self.handle, buf, size, ctypes.byref(read_count), None)
        return buf.raw[:read_count.value]
    
    def write(self, data):
        written = wintypes.DWORD()
        kernel32.WriteFile(self.handle, data, len(data), ctypes.byref(written), None)
        return written.value
    
    def close(self):
        if self.is_open:
            kernel32.CloseHandle(self.handle)
            self.is_open = False


# =============================================================================
# CONTROL SENSORLESS
# =============================================================================

data_queue = queue.Queue()
exit_event = threading.Event()


def control_sensorless(port='COM1', baudrate=115200, full_sensorless=False):
    """
    Control PID usando observador en vez de sensor.
    
    full_sensorless=False: Luenberger corrige con sensor (semi)
    full_sensorless=True:  Luenberger NO usa sensor (full)
    """
    
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    iTs = 1 / Ts
    
    # Observador
    observador = ObservadorLuenberger(
        polo1=-80, polo2=-80, 
        use_sensor=(not full_sensorless)
    )
    
    # Variables PID
    t = 0
    yd = 0.005
    ef, ef_1 = 0, 0
    integral, intei = 0, 0
    u = 0
    y_ctrl_prev = 0.005
    
    modo_str = "FULL SENSORLESS" if full_sensorless else "SEMI-SENSORLESS"
    
    ser = None
    try:
        ser = Win32Serial(port, baudrate)
        print(f"✅ Puerto {port} abierto.")
        print(f"🎮 Modo: {modo_str}")
        print(f"   PID usa: OBSERVADOR (no sensor directo)")
        
        print("⏳ Esperando switch (0xAA)...")
        while not exit_event.is_set():
            recibido = ser.read(1)
            if recibido == b'\xAA':
                print("✅ ¡Switch activado! Iniciando control sensorless...")
                break
        else:
            return
        
        filename = f"MONIT_sensorless_{'full' if full_sensorless else 'semi'}.txt"
        with open(filename, "w+") as fp:
            fp.write("# t\tyd\ty_sensor\ty_est\tv_est\tie\tu\n")
            
            flagcom = 0
            pv = 0
            i_raw = 0
            
            while not exit_event.is_set():
                b = ser.read(1)
                if not b:
                    continue
                
                recibido = b[0]
                
                if flagcom != 0:
                    flagcom += 1
                
                if (recibido == 0xAA) and (flagcom == 0):
                    pv = 0
                    i_raw = 0
                    flagcom = 1
                    continue
                
                if flagcom == 2:
                    pv = recibido << 8
                    continue
                
                if flagcom == 3:
                    pv = pv + recibido
                    continue
                
                if flagcom == 4:
                    i_raw = recibido << 8
                    continue
                
                if flagcom == 5:
                    i_raw = i_raw + recibido
                    
                    if pv > 1023 or i_raw > 1023:
                        flagcom = 0
                        continue
                    
                    # Mediciones (sensor solo para logging)
                    y_sensor = esc * pv
                    ie = esci * i_raw
                    
                    # === OBSERVADOR ===
                    y_est, v_est = observador.estimar(y_sensor, ie)
                    
                    # === PID USA OBSERVADOR (no sensor!) ===
                    y_ctrl = y_est  # <-- CLAVE: usamos estimación
                    
                    ef_1 = ef
                    ef = yd - y_ctrl
                    
                    proporcional = kp * ef
                    derivativa = kd * (ef - ef_1) * iTs
                    
                    if -Iref < integral < Iref:
                        integral += ki * Ts * ef
                    else:
                        integral = 0.95 * Iref if integral >= Iref else -0.95 * Iref
                    
                    id_val = proporcional + integral + derivativa
                    if id_val > 0:
                        id_val = 0
                    if id_val <= -Iref:
                        id_val = -Iref
                    
                    ied = -id_val
                    ei = ied - ie
                    propi = kpi * ei
                    
                    if -Vref < intei < Vref:
                        intei += kii * Ts * ei
                    else:
                        intei = 0.95 * Vref if intei >= Vref else -0.95 * Vref
                    
                    u = propi + intei
                    if u > Vref:
                        u = Vref
                    if u <= 0:
                        u = 0
                    
                    # Enviar PWM
                    pwm = int(escs * u)
                    pwm = min(254, max(0, pwm))
                    ser.write(bytes([pwm]))
                    
                    y_ctrl_prev = y_ctrl
                    
                    # Log
                    data_queue.put((t, yd, y_sensor, y_est, v_est, ie, u))
                    fp.write(f"{t:.4f}\t{yd:.6f}\t{y_sensor:.6f}\t{y_est:.6f}\t{v_est:.6f}\t{ie:.4f}\t{u:.4f}\n")
                    fp.flush()
                    
                    flagcom = 0
                    t += Ts
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ser and ser.is_open:
            ser.close()
        print("🔌 Hilo finalizado.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Argumento: 'full' para full sensorless, otro para semi
    full_sensorless = len(sys.argv) > 1 and sys.argv[1].lower() == 'full'
    
    print("="*60)
    if full_sensorless:
        print("⚡ CONTROL FULL SENSORLESS")
        print("   El observador NO usa el sensor de posición")
        print("   Solo usa: corriente (i) + modelo físico")
    else:
        print("⚡ CONTROL SEMI-SENSORLESS")
        print("   El observador SÍ corrige con sensor")
        print("   Pero el PID usa la estimación, no el sensor directo")
    print("="*60)
    
    control = threading.Thread(
        target=control_sensorless, 
        args=('COM1', 115200, full_sensorless),
        daemon=True
    )
    control.start()
    
    try:
        duration = 30
        start = time.time()
        print(f"🚀 Corriendo {duration}s...")
        
        while time.time() - start < duration:
            if not data_queue.empty():
                while not data_queue.empty():
                    t_val, yd, y_s, y_e, v_e, ie, u = data_queue.get()
                
                error = abs(y_s - yd) * 1000
                print(f"t={t_val:.1f}s | Sensor={y_s*1000:.1f}mm | Est={y_e*1000:.1f}mm | Error={error:.1f}mm | u={u:.1f}V")
            
            time.sleep(0.5)
            
            if not control.is_alive():
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ Detenido")
    finally:
        exit_event.set()
        control.join(timeout=2)
        print("✅ Finalizado.")


if __name__ == '__main__':
    main()

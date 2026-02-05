"""
Control PID Base + Comparación de Observadores (Luenberger vs Clásico)

El PID controla usando el sensor físico (funciona excelente).
Ambos observadores estiman en paralelo para comparar su precisión.

Autor: José de Jesús Santana Ramírez
"""

import time
import threading
import queue
import ctypes
from ctypes import wintypes
import math

# =============================================================================
# PARÁMETROS DEL MODELO
# =============================================================================
K0 = 0.0704    # H - Inductancia base
K = 0.0327     # H - Variación de inductancia
A = 0.0052     # m - Parámetro de forma
R = 2.72       # Ω - Resistencia de la bobina
M = 0.018      # kg - Masa de la esfera
G = 9.81       # m/s² - Gravedad

# =============================================================================
# PARÁMETROS DE CONTROL PID (los que funcionan)
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
    """Aceleración física: ÿ = (1/2m) * dL/dy * i² + g"""
    dL = dL_dy(y)
    return (1/(2*M)) * dL * i**2 + G

# =============================================================================
# OBSERVADOR DE LUENBERGER
# =============================================================================

class ObservadorLuenberger:
    """
    Observador de Luenberger para estimar [y, ẏ].
    Usa el modelo físico y corrige con la medición del sensor.
    """
    
    def __init__(self, polo1=-80, polo2=-80, dt=Ts):
        self.dt = dt
        self.L1 = -(polo1 + polo2)
        self.L2 = polo1 * polo2
        
        self.y_est = 0.005
        self.v_est = 0.0
        self.initialized = False
    
    def estimar(self, y_medido, i):
        if not self.initialized:
            self.y_est = y_medido
            self.v_est = 0.0
            self.initialized = True
            return self.y_est, self.v_est
        
        # Error
        e = y_medido - self.y_est
        
        # Modelo
        a = aceleracion(self.y_est, i)
        
        # Observador
        dy_dt = self.v_est + self.L1 * e
        dv_dt = a + self.L2 * e
        
        # Integrar
        self.y_est += dy_dt * self.dt
        self.v_est += dv_dt * self.dt
        
        # Limitar
        self.y_est = max(0.0001, min(0.025, self.y_est))
        self.v_est = max(-2.0, min(2.0, self.v_est))
        
        return self.y_est, self.v_est
    
    def reset(self):
        self.y_est = 0.005
        self.v_est = 0.0
        self.initialized = False


# =============================================================================
# OBSERVADOR CLÁSICO (basado en inductancia)
# =============================================================================

class ObservadorClasico:
    """
    Observador basado en integración del flujo: φ = ∫(u - R*i)dt
    Luego L = φ/i y se despeja y desde L(y).
    """
    
    def __init__(self, dt=Ts):
        self.dt = dt
        self.phi = 0.0
        self.i_prev = 0.0
        self.u_prev = 0.0
        self.y_est = 0.005
        self.y_est_prev = 0.005
        self.v_est = 0.0
        self.L_est = K0 + K
        self.initialized = False
        self.alpha = 0.25
        self.y_buffer = []
        self.buffer_size = 5
    
    def estimar(self, i, u):
        if not self.initialized:
            if i > 0.01:
                self.initialized = True
                y_init = self._estimar_equilibrio(i)
                self.y_est = y_init if y_init else 0.005
                L0 = inductancia(self.y_est)
                self.phi = L0 * i
            self.i_prev = i
            self.u_prev = u
            return self.y_est, self.v_est
        
        # Modo caída (corriente baja)
        if i < 0.03:
            self.y_est += 0.03 * self.dt
            self.y_est = min(self.y_est, 0.022)
            self.v_est = 0.03
            self.i_prev = i
            self.u_prev = u
            return self.y_est, self.v_est
        
        # Integrar flujo (trapecio)
        dphi_now = u - R * i
        dphi_prev = self.u_prev - R * self.i_prev
        dphi = 0.5 * (dphi_now + dphi_prev) * self.dt
        self.phi += dphi
        
        # Calcular posición desde inductancia
        y_nuevo = None
        if i > 0.05:
            self.L_est = self.phi / i
            L_min = K0 * 0.5
            L_max = (K0 + K) * 2.0
            
            if L_min < self.L_est < L_max:
                y_nuevo = self._posicion_desde_L(self.L_est)
        
        # También usar modelo de equilibrio
        y_eq = self._estimar_equilibrio(i)
        
        # Fusionar estimaciones
        if y_nuevo is not None and y_eq is not None:
            di = abs(i - self.i_prev) / self.dt
            peso_eq = 1.0 / (1.0 + di * 100)
            y_fusion = peso_eq * y_eq + (1 - peso_eq) * y_nuevo
        elif y_eq is not None:
            y_fusion = y_eq
        elif y_nuevo is not None:
            y_fusion = y_nuevo
        else:
            y_fusion = None
        
        # Actualizar con filtro
        if y_fusion is not None:
            self.y_buffer.append(y_fusion)
            if len(self.y_buffer) > self.buffer_size:
                self.y_buffer.pop(0)
            
            y_prom = sum(self.y_buffer) / len(self.y_buffer)
            self.y_est = self.alpha * y_prom + (1 - self.alpha) * self.y_est
            
            # Corregir drift del integrador
            L_esperada = inductancia(self.y_est)
            self.phi = 0.9 * self.phi + 0.1 * L_esperada * i
        
        # Velocidad
        self.v_est = (self.y_est - self.y_est_prev) / self.dt
        self.y_est_prev = self.y_est
        
        # Limitar
        self.y_est = max(0.0001, min(0.025, self.y_est))
        
        self.i_prev = i
        self.u_prev = u
        
        return self.y_est, self.v_est
    
    def _posicion_desde_L(self, L):
        denom = L - K0
        if denom > 0.001:
            y = A * (K / denom - 1)
            return max(0.001, min(0.020, y))
        return self.y_est
    
    def _estimar_equilibrio(self, i):
        if i < 0.05:
            return None
        try:
            dL_abs = 2 * M * G / (i**2)
            factor = K / (A * dL_abs)
            y = A * (math.sqrt(factor) - 1)
            return max(0.001, min(0.015, y))
        except:
            return 0.005
    
    def reset(self):
        self.phi = 0.0
        self.y_est = 0.005
        self.v_est = 0.0
        self.initialized = False


# =============================================================================
# COMUNICACIÓN SERIAL
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
# HILO DE CONTROL PID + OBSERVADORES
# =============================================================================

data_queue = queue.Queue()
exit_event = threading.Event()


def control_thread_pid_obs(port='COM1', baudrate=115200):
    """Control PID con ambos observadores corriendo en paralelo."""
    
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    iTs = 1 / Ts
    
    # Observadores
    obs_luenberger = ObservadorLuenberger(polo1=-80, polo2=-80)
    obs_clasico = ObservadorClasico()
    
    # Variables PID
    t = 0
    yd = 0.005
    ef, ef_1, y_prev = 0, 0, 0
    integral, intei = 0, 0
    u = 0
    
    ser = None
    try:
        ser = Win32Serial(port, baudrate)
        print(f"✅ Puerto {port} abierto.")
        print("🎮 Control: PID BASE (sensor)")
        print("👁️ Observadores: Luenberger + Clásico (comparación)")
        
        print("⏳ Esperando switch (0xAA)...")
        while not exit_event.is_set():
            recibido = ser.read(1)
            if recibido == b'\xAA':
                print("✅ ¡Switch activado!")
                break
        else:
            return
        
        with open("MONIT_pid_obs.txt", "w+") as fp:
            fp.write("# t\tyd\ty_sensor\ty_luenberger\tv_luenberger\ty_clasico\tv_clasico\tie\tu\n")
            
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
                    
                    # Mediciones
                    y_sensor = esc * pv
                    ie = esci * i_raw
                    
                    # === OBSERVADORES (en paralelo) ===
                    y_luen, v_luen = obs_luenberger.estimar(y_sensor, ie)
                    y_clas, v_clas = obs_clasico.estimar(ie, u)
                    
                    # === CONTROL PID (usa sensor, no observador) ===
                    ef_1 = ef
                    y_1 = y_prev
                    y = y_sensor  # PID usa el sensor real
                    
                    ef = yd - y
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
                    
                    y_prev = y
                    
                    # Log
                    data_queue.put((t, yd, y_sensor, y_luen, v_luen, y_clas, v_clas, ie, u))
                    fp.write(f"{t:.4f}\t{yd:.6f}\t{y_sensor:.6f}\t{y_luen:.6f}\t{v_luen:.6f}\t{y_clas:.6f}\t{v_clas:.6f}\t{ie:.4f}\t{u:.4f}\n")
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
    print("="*60)
    print("🎮 CONTROL PID + COMPARACIÓN DE OBSERVADORES")
    print("="*60)
    print("Control: PID Base (MAE=0.09mm demostrado)")
    print("Observadores: Luenberger vs Clásico (estimando en paralelo)")
    print("="*60)
    
    control = threading.Thread(target=control_thread_pid_obs, daemon=True)
    control.start()
    
    try:
        duration = 30
        start = time.time()
        print(f"🚀 Corriendo {duration}s...")
        
        while time.time() - start < duration:
            if not data_queue.empty():
                while not data_queue.empty():
                    t_val, yd, y_s, y_l, v_l, y_c, v_c, ie, u = data_queue.get()
                
                e_l = abs(y_s - y_l) * 1000
                e_c = abs(y_s - y_c) * 1000
                print(f"t={t_val:.1f}s | y={y_s*1000:.1f}mm | Luen={y_l*1000:.1f}mm(e={e_l:.1f}) | Clas={y_c*1000:.1f}mm(e={e_c:.1f}) | u={u:.1f}V")
            
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

"""
Control No Lineal por Linealización (Lie) con Observador de Luenberger
para el Levitador Magnético.

Este script implementa:
1. Observador de Luenberger para estimar [y, ẏ] desde mediciones de [y_sensor, i]
2. Observador Clásico (basado en inductancia) para comparación
3. Control por Linealización usando derivadas de Lie

Autor: José de Jesús Santana Ramírez
"""

import time
import threading
import queue
import ctypes
from ctypes import wintypes
import math

# =============================================================================
# PARÁMETROS DEL MODELO FÍSICO (identificados)
# =============================================================================
K0 = 0.0704    # H - Inductancia base
K = 0.0327     # H - Variación de inductancia
A = 0.0052     # m - Parámetro de forma
R = 2.72       # Ω - Resistencia de la bobina
M = 0.018      # kg - Masa de la esfera
G = 9.81       # m/s² - Gravedad

# =============================================================================
# PARÁMETROS DE CONTROL Y OBSERVADOR
# =============================================================================
Ts = 0.01      # s - Periodo de muestreo
Vref = 9.86    # V - Voltaje máximo
Rs_sense = 2.2 # Ω - Resistencia de sensado

# Especificaciones del controlador Lie (conservadoras)
TR = 0.2       # s - Tiempo de subida
MP = 15        # % - Sobrepaso máximo

# =============================================================================
# FUNCIONES DEL MODELO
# =============================================================================

def inductancia(y):
    """L(y) = k0 + k/(1 + y/a)"""
    y = max(0.0001, y)
    return K0 + K / (1 + y / A)

def dL_dy(y):
    """dL/dy = -k / (a · (1 + y/a)²)"""
    y = max(0.0001, y)
    return -K / (A * (1 + y / A)**2)

def d2L_dy2(y):
    """d²L/dy² = 2k / (a² · (1 + y/a)³)"""
    y = max(0.0001, y)
    return 2 * K / (A**2 * (1 + y / A)**3)

def aceleracion(y, i):
    """Aceleración de la esfera: ÿ = (1/2m) * dL/dy * i² + g"""
    dL = dL_dy(y)
    return (1/(2*M)) * dL * i**2 + G

# =============================================================================
# OBSERVADOR DE LUENBERGER
# =============================================================================

class ObservadorLuenberger:
    """
    Observador de Luenberger para estimar estados [y, ẏ] del levitador.
    
    Modelo:
        ẏ = v
        v̇ = (1/2m) * dL/dy * i² + g
        
    Observador:
        ŷ_dot = v̂ + L1*(y_med - ŷ)
        v̂_dot = f(ŷ, i) + L2*(y_med - ŷ)
    
    Las ganancias L1, L2 se eligen para ubicar los polos del observador.
    """
    
    def __init__(self, polo1=-100, polo2=-100, dt=Ts):
        """
        Args:
            polo1, polo2: Polos deseados del observador (negativos, rápidos)
            dt: Periodo de muestreo
        """
        self.dt = dt
        
        # Ganancias del observador desde ubicación de polos
        # Para sistema de 2do orden: det(sI - (A - LC)) = (s - p1)(s - p2)
        # L = [L1, L2]^T
        # A - LC tiene polinomio característico s² + L1*s + L2
        # Queremos (s - p1)(s - p2) = s² - (p1+p2)s + p1*p2
        self.L1 = -(polo1 + polo2)      # = 200 para polos en -100
        self.L2 = polo1 * polo2          # = 10000 para polos en -100
        
        # Estados estimados
        self.y_est = 0.005   # Posición estimada [m]
        self.v_est = 0.0     # Velocidad estimada [m/s]
        
        self.initialized = False
        
        print(f"📐 Observador Luenberger: L1={self.L1:.1f}, L2={self.L2:.1f}")
        print(f"   Polos en s = {polo1}, {polo2}")
    
    def estimar(self, y_medido, i):
        """
        Actualiza la estimación de estados usando medición de posición y corriente.
        
        Args:
            y_medido: Posición medida por el sensor [m]
            i: Corriente medida [A]
            
        Returns:
            (y_est, v_est): Estados estimados
        """
        # Inicialización
        if not self.initialized:
            self.y_est = y_medido
            self.v_est = 0.0
            self.initialized = True
            return self.y_est, self.v_est
        
        # Error de estimación
        e = y_medido - self.y_est
        
        # Modelo: aceleración física
        a = aceleracion(self.y_est, i)
        
        # Ecuaciones del observador (Euler)
        # ŷ_dot = v̂ + L1*e
        # v̂_dot = a + L2*e
        dy_dt = self.v_est + self.L1 * e
        dv_dt = a + self.L2 * e
        
        # Integración
        self.y_est += dy_dt * self.dt
        self.v_est += dv_dt * self.dt
        
        # Limitar a rango físico
        self.y_est = max(0.0001, min(0.025, self.y_est))
        self.v_est = max(-2.0, min(2.0, self.v_est))  # ±2 m/s máx
        
        return self.y_est, self.v_est
    
    def reset(self):
        self.y_est = 0.005
        self.v_est = 0.0
        self.initialized = False


# =============================================================================
# OBSERVADOR CLÁSICO (basado en inductancia - el tuyo)
# =============================================================================

class ObservadorClasico:
    """
    Observador de posición basado en el modelo de inductancia.
    Estima y desde L = φ/i, donde φ se integra de u - R*i.
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
        self.alpha = 0.3
    
    def estimar(self, i, u):
        """
        Estima posición y velocidad desde corriente y voltaje.
        """
        if not self.initialized:
            if i > 0.05:
                self.initialized = True
                # Estimar desde equilibrio
                y_init = self._estimar_equilibrio(i)
                self.y_est = y_init if y_init else 0.005
                self.phi = inductancia(self.y_est) * i
            self.i_prev = i
            self.u_prev = u
            return self.y_est, self.v_est
        
        # Integrar flujo
        dphi = (u - R * i) * self.dt
        self.phi += dphi
        
        # Calcular L y posición
        if i > 0.05:
            self.L_est = self.phi / i
            L_min, L_max = K0 * 0.5, (K0 + K) * 1.5
            
            if L_min < self.L_est < L_max:
                y_nuevo = self._posicion_desde_L(self.L_est)
                self.y_est = self.alpha * y_nuevo + (1 - self.alpha) * self.y_est
                
                # Corregir drift
                L_esperada = inductancia(self.y_est)
                self.phi = 0.95 * self.phi + 0.05 * L_esperada * i
        
        # Velocidad por derivada
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
        """Estima y desde corriente de equilibrio."""
        if i < 0.05:
            return None
        # En equilibrio: F_mag = m*g
        # (1/2m) * |dL/dy| * i² = g
        # |dL/dy| = 2*m*g / i²
        dL_abs = 2 * M * G / (i**2)
        # dL/dy = -k / (a * (1+y/a)²)
        # Despejando y (aproximación)
        try:
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
# CONTROLADOR LIE CON OBSERVADOR
# =============================================================================

# Calcular ganancias del controlador
zeta = -math.log(MP/100) / math.sqrt(math.log(MP/100)**2 + math.pi**2)
omega_d = (math.pi - math.atan(math.sqrt(1 - zeta**2) / zeta)) / TR
omega_n = omega_d / math.sqrt(1 - zeta**2)
p3 = -30  # Polo rápido (conservador)

K1_ctrl = omega_n**2 * abs(p3)
K2_ctrl = omega_n**2 + 2 * zeta * omega_n * abs(p3)
K3_ctrl = 2 * zeta * omega_n + abs(p3)

print(f"Ganancias Control Lie: K1={K1_ctrl:.1f}, K2={K2_ctrl:.1f}, K3={K3_ctrl:.1f}")


class ControladorLieConObservador:
    """
    Control por linealización usando estados del observador.
    """
    
    def __init__(self, observador_tipo='luenberger'):
        self.k1 = K1_ctrl
        self.k2 = K2_ctrl
        self.k3 = K3_ctrl
        
        # Seleccionar observador
        if observador_tipo == 'luenberger':
            self.observador = ObservadorLuenberger(polo1=-80, polo2=-80)
            self.obs_tipo = 'Luenberger'
        else:
            self.observador = ObservadorClasico()
            self.obs_tipo = 'Clásico'
        
        # Para estimar aceleración
        self.v_prev = 0.0
        self.integral_backup = 0.0
    
    def calcular_control(self, y_sensor, i, u_prev, yd, dt=Ts):
        """
        Calcula señal de control usando observador para estimar estados.
        """
        # Obtener estados estimados del observador
        if self.obs_tipo == 'Luenberger':
            y_est, v_est = self.observador.estimar(y_sensor, i)
        else:
            y_est, v_est = self.observador.estimar(i, u_prev)
        
        # Protección corriente baja
        if i < 0.02:
            e = yd - y_sensor
            u = 30.0 * e
            return max(0, min(Vref, u)), y_est, v_est
        
        # Estimar aceleración desde velocidad
        a_est = (v_est - self.v_prev) / dt
        a_est = max(-50, min(50, a_est))  # Limitar
        self.v_prev = v_est
        
        # Usar posición estimada para el control
        y = y_est
        dy = v_est
        ddy = a_est
        
        # Error
        e = y - yd
        de = dy
        
        # Términos del modelo
        L = inductancia(y)
        dL = dL_dy(y)
        ddL = d2L_dy2(y)
        
        # f(x)
        f1 = (1/L) * (-R * i - dL * dy * i)
        f2 = dy
        f3 = (1/(2*M)) * dL * i**2 + G
        
        # g(x)
        g1 = 1/L
        
        # Derivadas de Lie
        Lf3h = (1/M) * i * dL * f1 + (1/(2*M)) * (i**2) * ddL * f2
        LgLf2h = (1/M) * i * dL * g1
        
        # Controlador estabilizante
        v = -self.k1 * e - self.k2 * de - self.k3 * ddy
        
        # Ley de control
        if abs(LgLf2h) > 1e-6:
            u = (-Lf3h + v) / LgLf2h
        else:
            # Backup PID
            u = 30 * (yd - y) + self.integral_backup
            self.integral_backup += 5 * (yd - y) * dt
            self.integral_backup = max(-Vref, min(Vref, self.integral_backup))
        
        u = max(0, min(Vref, u))
        
        return u, y_est, v_est


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
# HILO DE CONTROL
# =============================================================================

data_queue = queue.Queue()
exit_event = threading.Event()


def control_thread(port='COM1', baudrate=115200, observador_tipo='luenberger'):
    """Hilo de control con observador seleccionable."""
    
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs_sense * 1023.0)
    escs = 254.0 / Vref
    
    controlador = ControladorLieConObservador(observador_tipo)
    
    t = 0
    yd = 0.005
    u_prev = 0
    
    ser = None
    try:
        ser = Win32Serial(port, baudrate)
        print(f"✅ Puerto {port} abierto.")
        print(f"🔬 Usando observador: {controlador.obs_tipo}")
        
        print("⏳ Esperando switch (byte 0xAA)...")
        while not exit_event.is_set():
            recibido = ser.read(1)
            if recibido == b'\xAA':
                print("✅ ¡Switch activado!")
                break
        else:
            return
        
        filename = f"MONIT_lie_{observador_tipo}.txt"
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
                    
                    y_sensor = esc * pv
                    ie = esci * i_raw
                    
                    # Control con observador
                    u, y_est, v_est = controlador.calcular_control(
                        y_sensor, ie, u_prev, yd, Ts
                    )
                    u_prev = u
                    
                    # Enviar PWM
                    pwm = int(escs * u)
                    pwm = min(254, max(0, pwm))
                    ser.write(bytes([pwm]))
                    
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
    import sys
    
    # Seleccionar observador por argumento
    obs_tipo = 'luenberger'
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['clasico', 'classic', 'c']:
            obs_tipo = 'clasico'
    
    print("="*60)
    print("⚡ CONTROL LIE + OBSERVADOR")
    print("="*60)
    print(f"Observador: {obs_tipo.upper()}")
    print(f"Modelo: k0={K0:.4f}H, k={K:.4f}H, a={A:.4f}m")
    print("="*60)
    
    control = threading.Thread(
        target=control_thread, 
        args=('COM1', 115200, obs_tipo),
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
                
                print(f"t={t_val:.1f}s | Ref={yd*1000:.1f} | Sensor={y_s*1000:.1f} | Est={y_e*1000:.1f} | v={v_e:.3f} | u={u:.1f}V")
            
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

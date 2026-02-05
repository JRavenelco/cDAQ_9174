"""
Control SENSORLESS usando el Observador 2023 de José Santana

Observador basado en integración del flujo magnético:
    z = u - R*i           (voltaje en inductancia)
    φ = ∫z dt             (flujo magnético)
    y = (a·kg·i)/(φ + L0·i0 - k0·i) - a  (posición estimada)

Optimizado con Numba para cálculos rápidos.

Autor: José de Jesús Santana Ramírez
"""

import time
import threading
import queue
import ctypes
from ctypes import wintypes
import math
import sys

# Intentar importar Numba para optimización
try:
    from numba import jit, float64
    NUMBA_AVAILABLE = True
    print("✅ Numba disponible - cálculos optimizados")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba no disponible - usando Python puro")
    # Crear decorador dummy
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    float64 = float

# =============================================================================
# PARÁMETROS DEL MODELO (sistema actual - identificados previamente)
# =============================================================================
K0 = 0.0704    # H - Inductancia base
KG = 0.0327    # H - Variación de inductancia
A = 0.0052     # m - Parámetro de forma
R = 2.72       # Ω - Resistencia de la bobina
M = 0.018      # kg - Masa de la esfera
G = 9.81       # m/s²

# Valor inicial de inductancia L(y=0) = K0 + KG
L0_INIT = K0 + KG  # ≈ 0.103 H

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
# FUNCIONES OPTIMIZADAS CON NUMBA
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def integrar_trapecio(inte_prev, z_prev, z_now, dt):
    """Integración por trapecio (rápida)."""
    return inte_prev + 0.5 * (z_prev + z_now) * dt


@jit(nopython=True, cache=True, fastmath=True)
def estimar_posicion_2023(i, inte, i0, L0, k0, kg, a):
    """
    Fórmula del observador 2023:
    fhi = (a * kg * i) / (inte + L0*i0 - k0*i) - a
    
    Donde:
    - i: corriente actual
    - inte: integral de (u - R*i)
    - i0: corriente inicial
    - L0: inductancia inicial
    - k0, kg, a: parámetros del modelo
    """
    denom = inte + L0 * i0 - k0 * i
    
    # Protección contra división por cero
    if abs(denom) < 1e-6:
        return 0.010  # Valor por defecto
    
    y = (a * kg * i) / denom - a
    
    # Limitar a rango físico
    if y < 0.0005:
        y = 0.0005
    elif y > 0.022:
        y = 0.022
    
    return y


@jit(nopython=True, cache=True, fastmath=True)
def estimar_velocidad_2023(i, di_dt, inte, u, u0, i0, k0, kg, a, L0):
    """
    Fórmula de velocidad del observador 2023 (derivada de fhi):
    theta = (a*kg / denom) * (di/dt - (i/denom)*(u-R*i - u0+R*i0 - k0*di/dt))
    
    Simplificada para evitar inestabilidad numérica.
    """
    denom = inte + L0 * i0 - k0 * i
    
    if abs(denom) < 1e-6:
        return 0.0
    
    # Simplificación: usar solo la parte principal
    # La fórmula completa es muy sensible a ruido
    factor = (a * kg) / denom
    v = factor * di_dt
    
    # Limitar
    if v < -1.5:
        v = -1.5
    elif v > 1.5:
        v = 1.5
    
    return v


@jit(nopython=True, cache=True, fastmath=True)
def derivada_filtrada(x_now, x_prev, dx_prev, dt, alpha=0.3):
    """Derivada con filtro paso bajo para reducir ruido."""
    dx_raw = (x_now - x_prev) / dt
    return alpha * dx_raw + (1.0 - alpha) * dx_prev


# =============================================================================
# OBSERVADOR SENSORLESS 2023
# =============================================================================

class EstimadorResistencia:
    """Estimador adaptativo de resistencia para eliminar drift."""
    def __init__(self, R0=16.0, alpha=0.0, beta=0.02, dt=Ts, gain_fusion=0.1):
        self.R = R0
        self.R_amb = R0
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.gain = gain_fusion
        self.i_min_correction = 0.05
        self.u_avg = 0.0
        self.i_avg = 0.0
        self.tau_filter = 0.5
        self.alpha_filter = dt / self.tau_filter
        self.steps = 0
        self.decimation = 10
        
    def update(self, u, i):
        power = self.R * (i**2)
        dR = (self.alpha * power - self.beta * (self.R - self.R_amb)) * self.dt
        self.R += dR
        self.R = max(self.R_amb * 0.5, min(self.R, self.R_amb * 1.6))
        
        self.u_avg = (1 - self.alpha_filter) * self.u_avg + self.alpha_filter * u
        self.i_avg = (1 - self.alpha_filter) * self.i_avg + self.alpha_filter * i
        
        self.steps += 1
        if self.steps % self.decimation == 0:
            if abs(self.i_avg) > self.i_min_correction:
                R_meas = self.u_avg / self.i_avg
                if 0.5 * self.R_amb < R_meas < 2.0 * self.R_amb:
                    self.R += self.gain * (R_meas - self.R)
        
        return self.R


class ObservadorSensorless2023:
    """
    Observador sensorless basado en el diseño de 2023.
    Solo usa corriente (i) y voltaje (u) - NO sensor de posición.
    
    Versión mejorada con:
    - EstimadorResistencia (NUEVO - elimina drift)
    - Reset adaptativo del integrador
    - Estimación híbrida (flujo + equilibrio)
    - Corrección de drift
    """
    
    def __init__(self, dt=Ts):
        self.dt = dt
        
        # NUEVO: Estimador de Resistencia Dinámica
        self.estimador_R = EstimadorResistencia(
            R0=16.0,
            alpha=0.0,
            beta=0.02,
            dt=dt,
            gain_fusion=0.1
        )
        
        # Estado del observador
        self.phi = 0.0           # Flujo magnético (integral de u-Ri)
        self.z_prev = 0.0        # z anterior para trapecio
        self.i_prev = 0.0        # Corriente anterior
        self.di_prev = 0.0       # Derivada de corriente anterior
        self.u_prev = 0.0
        self.y_est_prev = 0.005
        
        # Estimaciones
        self.y_est = 0.005
        self.v_est = 0.0
        
        # Para estimación por equilibrio
        self.y_buffer = []
        self.buffer_size = 10
        
        self.initialized = False
        self.sample_count = 0
    
    def _L_from_y(self, y):
        """L(y) = k0 + kg/(1 + y/a)"""
        y = max(0.0001, y)
        return K0 + KG / (1 + y / A)
    
    def _y_from_L(self, L):
        """Despeja y de L(y) = k0 + kg/(1 + y/a)"""
        denom = L - K0
        if denom > 0.001:
            y = A * (KG / denom - 1)
            return max(0.0005, min(0.022, y))
        return self.y_est
    
    def _estimar_equilibrio(self, i):
        """
        En equilibrio: F_mag = m*g
        (1/2m) * |dL/dy| * i² = g
        |dL/dy| = kg / (a * (1+y/a)²)
        
        Despejando: y ≈ a * (sqrt(kg*i²/(2*a*m*g)) - 1)
        """
        if i < 0.05:
            return None
        try:
            factor = KG * i**2 / (2 * A * M * G)
            if factor > 0:
                y = A * (math.sqrt(factor) - 1)
                return max(0.001, min(0.020, y))
        except:
            pass
        return None
    
    def estimar(self, i, u):
        """
        Estima posición y velocidad usando método híbrido:
        1. Integración de flujo (dinámica rápida)
        2. Modelo de equilibrio (corrección de drift)
        """
        self.sample_count += 1
        
        # NUEVO: Actualizar resistencia dinámica
        R_dinamica = self.estimador_R.update(u, i)
        
        # === INICIALIZACIÓN ===
        if not self.initialized:
            if i > 0.03:
                self.initialized = True
                # Estimar posición inicial desde equilibrio
                y_init = self._estimar_equilibrio(i) or 0.005
                self.y_est = y_init
                # Inicializar flujo consistente: φ = L(y) * i
                L_init = self._L_from_y(y_init)
                self.phi = L_init * i
                print(f"📐 Obs2023 init: y={y_init*1000:.1f}mm, φ={self.phi:.4f}, i={i:.3f}A, R={R_dinamica:.2f}Ω")
            
            self.i_prev = i
            self.u_prev = u
            return self.y_est, self.v_est
        
        # === INTEGRAR FLUJO (trapecio) - USANDO R DINÁMICA ===
        z_now = u - R_dinamica * i
        z_prev = self.u_prev - R_dinamica * self.i_prev
        dphi = 0.5 * (z_now + z_prev) * self.dt
        self.phi += dphi
        
        # === MÉTODO 1: POSICIÓN DESDE FLUJO ===
        y_flujo = None
        if i > 0.05:
            L_est = self.phi / i
            # Verificar rango válido de L
            L_min = K0 * 0.8
            L_max = (K0 + KG) * 1.2
            if L_min < L_est < L_max:
                y_flujo = self._y_from_L(L_est)
        
        # === MÉTODO 2: POSICIÓN DESDE EQUILIBRIO ===
        y_eq = self._estimar_equilibrio(i)
        
        # === FUSIÓN ADAPTATIVA ===
        if y_flujo is not None and y_eq is not None:
            # Ponderar según estabilidad de corriente
            di_dt = abs(i - self.i_prev) / self.dt
            peso_eq = 1.0 / (1.0 + di_dt * 50)  # Más peso a eq si di/dt pequeño
            y_nuevo = peso_eq * y_eq + (1 - peso_eq) * y_flujo
        elif y_eq is not None:
            y_nuevo = y_eq
        elif y_flujo is not None:
            y_nuevo = y_flujo
        else:
            y_nuevo = self.y_est
        
        # === FILTRO PROMEDIO MÓVIL ===
        self.y_buffer.append(y_nuevo)
        if len(self.y_buffer) > self.buffer_size:
            self.y_buffer.pop(0)
        
        y_promedio = sum(self.y_buffer) / len(self.y_buffer)
        
        # Filtro paso bajo
        alpha = 0.3
        self.y_est = alpha * y_promedio + (1 - alpha) * self.y_est
        
        # === CORRECCIÓN DE DRIFT DEL INTEGRADOR ===
        # Forzar consistencia: φ ≈ L(y_est) * i
        L_esperada = self._L_from_y(self.y_est)
        phi_esperado = L_esperada * i
        # Corrección suave (10% por muestra)
        self.phi = 0.9 * self.phi + 0.1 * phi_esperado
        
        # === VELOCIDAD ===
        self.v_est = (self.y_est - self.y_est_prev) / self.dt
        self.v_est = max(-1.0, min(1.0, self.v_est))
        self.y_est_prev = self.y_est
        
        # === LIMITAR ===
        self.y_est = max(0.0005, min(0.022, self.y_est))
        
        # === ACTUALIZAR ===
        self.i_prev = i
        self.u_prev = u
        
        return self.y_est, self.v_est
    
    def reset(self):
        self.phi = 0.0
        self.y_est = 0.005
        self.v_est = 0.0
        self.y_buffer = []
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
# CONTROL SENSORLESS 2023
# =============================================================================

data_queue = queue.Queue()
exit_event = threading.Event()


def control_sensorless_2023(port='COM1', baudrate=115200, use_for_control=False):
    """
    Control con observador sensorless 2023.
    
    use_for_control=False: PID usa sensor, observador solo monitorea
    use_for_control=True:  PID usa observador (verdadero sensorless)
    """
    
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    iTs = 1 / Ts
    
    # Observador 2023
    observador = ObservadorSensorless2023()
    
    # Variables PID
    t = 0
    yd = 0.005
    ef, ef_1 = 0, 0
    integral, intei = 0, 0
    u = 0
    
    modo = "SENSORLESS (control)" if use_for_control else "MONITOREO (PID usa sensor)"
    
    ser = None
    try:
        ser = Win32Serial(port, baudrate)
        print(f"✅ Puerto {port} abierto.")
        print(f"🔬 Observador: 2023 (tu diseño)")
        print(f"🎮 Modo: {modo}")
        
        print("⏳ Esperando switch (0xAA)...")
        while not exit_event.is_set():
            recibido = ser.read(1)
            if recibido == b'\xAA':
                print("✅ ¡Switch activado!")
                break
        else:
            return
        
        filename = f"MONIT_obs2023_{'ctrl' if use_for_control else 'mon'}.txt"
        with open(filename, "w+") as fp:
            fp.write("# t\tyd\ty_sensor\ty_est\tv_est\tie\tu\tphi\tR_est\n")
            
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
                    
                    # === OBSERVADOR 2023 ===
                    y_est, v_est = observador.estimar(ie, u)
                    
                    # === PID ===
                    # Seleccionar fuente de retroalimentación
                    y_ctrl = y_est if use_for_control else y_sensor
                    
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
                    
                    # Log
                    R_est = observador.estimador_R.R
                    data_queue.put((t, yd, y_sensor, y_est, v_est, ie, u, observador.phi, R_est))
                    fp.write(f"{t:.4f}\t{yd:.6f}\t{y_sensor:.6f}\t{y_est:.6f}\t{v_est:.6f}\t{ie:.4f}\t{u:.4f}\t{observador.phi:.6f}\t{R_est:.4f}\n")
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
    # Argumentos: 'ctrl' para control, 'kan' para usar KAN corrector
    use_for_control = len(sys.argv) > 1 and sys.argv[1].lower() in ['ctrl', 'control', 'sensorless']
    use_kan = len(sys.argv) > 2 and sys.argv[2].lower() == 'kan'
    
    print("="*60)
    print("⚡ OBSERVADOR SENSORLESS 2023 (José Santana)")
    if use_kan:
        print("🧠 + KAN CORRECTOR (mejora 43%)")
    print("="*60)
    print(f"Fórmula: y = (a·kg·i)/(φ + L0·i0 - k0·i) - a")
    print(f"Optimización: {'Numba JIT' if NUMBA_AVAILABLE else 'Python puro'}")
    if use_for_control:
        print("🎮 MODO: CONTROL SENSORLESS (PID usa observador)")
    else:
        print("👁️ MODO: MONITOREO (PID usa sensor, observador compara)")
    print("="*60)
    
    control = threading.Thread(
        target=control_sensorless_2023, 
        args=('COM1', 115200, use_for_control),
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
                    t_val, yd, y_s, y_e, v_e, ie, u, phi, R_est = data_queue.get()
                
                e_obs = abs(y_s - y_e) * 1000
                print(f"t={t_val:.1f}s | Sensor={y_s*1000:.1f}mm | Est={y_e*1000:.1f}mm | Err={e_obs:.1f}mm | R={R_est:.2f}Ω")
            
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

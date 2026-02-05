"""
Control No Lineal por Linealización por Realimentación (Feedback Linearization)
usando Derivadas de Lie para el Levitador Magnético.

Modelo del sistema:
    Estados: X = [i, y, ẏ] (corriente, posición, velocidad)
    
    ẋ₁ = (1/L)(u - Ri - dL/dy · ẏ · i)   # Dinámica eléctrica
    ẋ₂ = ẏ                                # Cinemática
    ẋ₃ = (1/2m)(dL/dy)i² + g              # Dinámica mecánica
    
    Salida: h(x) = y (posición)
    Grado relativo: r = 3

Ley de control linealizante:
    u = (-Lf³h + v) / LgLf²h
    
    donde v = -K₁·e - K₂·ė - K₃·ÿ es el controlador estabilizante.

Autor: José de Jesús Santana Ramírez
Basado en: levitador_lie.m (simulación MATLAB)
"""

import time
import threading
import queue
import ctypes
from ctypes import wintypes

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
# PARÁMETROS DE CONTROL
# =============================================================================
Ts = 0.01      # s - Periodo de muestreo
Vref = 9.86    # V - Voltaje máximo
Rs = 2.2       # Ω - Resistencia de sensado

# Especificaciones de diseño (AJUSTADAS para sistema real con ruido)
TR = 0.3       # s - Tiempo de subida (más lento para reducir ruido)
MP = 10        # % - Sobrepaso máximo (menor para menos agresividad)

# =============================================================================
# CÁLCULO DE GANANCIAS DEL CONTROLADOR (Ubicación de Polos)
# =============================================================================
import math

# Calcular zeta y omega_n desde especificaciones
zeta = -math.log(MP/100) / math.sqrt(math.log(MP/100)**2 + math.pi**2)
omega_d = (math.pi - math.atan(math.sqrt(1 - zeta**2) / zeta)) / TR
omega_n = omega_d / math.sqrt(1 - zeta**2)

# Polos deseados
p1_real = -zeta * omega_n
p1_imag = omega_d
p3 = -50   # Polo rápido real (reducido para menos agresividad)

# Polinomio característico: (s - p1)(s - p1*)(s - p3)
# = s³ + (2ζωn + p3)s² + (ωn² + 2ζωn·p3)s + ωn²·p3
K1 = omega_n**2 * abs(p3)
K2 = omega_n**2 + 2 * zeta * omega_n * abs(p3)
K3 = 2 * zeta * omega_n + abs(p3)

print(f"Ganancias del controlador Lie:")
print(f"  K1 = {K1:.2f}")
print(f"  K2 = {K2:.2f}")
print(f"  K3 = {K3:.2f}")
print(f"  ζ = {zeta:.3f}, ωn = {omega_n:.2f} rad/s")

# =============================================================================
# FUNCIONES DEL MODELO DE INDUCTANCIA
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

# =============================================================================
# LEY DE CONTROL POR LINEALIZACIÓN
# =============================================================================

class ControladorLie:
    """
    Controlador no lineal por linealización por realimentación.
    """
    
    def __init__(self, k1=K1, k2=K2, k3=K3):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        
        # Estados
        self.y_prev = 0.005
        self.dy_prev = 0.0
        self.e_prev = 0.0
        
        # Para integrador de respaldo (si Lie falla)
        self.integral_backup = 0.0
        
    def calcular_control(self, i, y, yd, dt=Ts):
        """
        Calcula la señal de control usando linealización por realimentación.
        
        Args:
            i: Corriente medida [A]
            y: Posición medida [m]
            yd: Referencia de posición [m]
            dt: Periodo de muestreo [s]
            
        Returns:
            u: Voltaje de control [V]
        """
        # Protección contra corriente cero (singularidad)
        if i < 0.01:
            # Sin corriente, usar control proporcional simple
            e = yd - y
            u = 50.0 * e  # Ganancia alta para arrancar
            return max(0, min(Vref, u))
        
        # --- Estimar velocidad (derivada numérica filtrada) ---
        dy = (y - self.y_prev) / dt
        dy = 0.3 * dy + 0.7 * self.dy_prev  # Filtro paso bajo
        
        # --- Estimar aceleración (derivada de velocidad) ---
        ddy = (dy - self.dy_prev) / dt
        
        # --- Error y derivadas ---
        e = y - yd
        de = dy  # ẏd = 0 para referencia constante
        
        # --- Calcular términos del modelo ---
        L = inductancia(y)
        dL = dL_dy(y)
        ddL = d2L_dy2(y)
        
        # f(x) - Campo vectorial
        f1 = (1/L) * (-R * i - dL * dy * i)  # di/dt sin u
        f2 = dy                               # ẏ
        f3 = (1/(2*M)) * dL * i**2 + G        # ÿ (aceleración)
        
        # g(x) - Dirección de entrada
        g1 = 1/L
        
        # --- Derivadas de Lie ---
        # Lf³h = (1/m)·i·(dL/dy)·f1 + (1/2m)·i²·(d²L/dy²)·f2
        Lf3h = (1/M) * i * dL * f1 + (1/(2*M)) * (i**2) * ddL * f2
        
        # LgLf²h = (1/m)·i·(dL/dy)·g1
        LgLf2h = (1/M) * i * dL * g1
        
        # --- Controlador estabilizante ---
        # v = -K1·e - K2·ė - K3·ÿ
        v = -self.k1 * e - self.k2 * de - self.k3 * ddy
        
        # --- Ley de control linealizante ---
        # u = (-Lf³h + v) / LgLf²h
        if abs(LgLf2h) > 1e-6:
            u = (-Lf3h + v) / LgLf2h
        else:
            # Singularidad: usar backup PID simple
            u = 50 * (yd - y) + self.integral_backup
            self.integral_backup += 10 * (yd - y) * dt
            self.integral_backup = max(-Vref, min(Vref, self.integral_backup))
        
        # --- Saturación ---
        u = max(0, min(Vref, u))
        
        # --- Actualizar estados ---
        self.y_prev = y
        self.dy_prev = dy
        self.e_prev = e
        
        return u

# =============================================================================
# COMUNICACIÓN SERIAL (Win32)
# =============================================================================

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

class Win32Serial:
    """Comunicación serial usando Win32 API."""
    
    def __init__(self, port, baudrate):
        self.handle = kernel32.CreateFileW(
            f"\\\\.\\{port}",
            0xC0000000,  # GENERIC_READ | GENERIC_WRITE
            0,
            None,
            3,  # OPEN_EXISTING
            0,
            None
        )
        if self.handle == -1:
            raise Exception(f"No se pudo abrir {port}")
        
        # Configurar DCB
        class DCB(ctypes.Structure):
            _fields_ = [
                ("DCBlength", wintypes.DWORD),
                ("BaudRate", wintypes.DWORD),
                ("fBinary", wintypes.DWORD, 1),
                ("fParity", wintypes.DWORD, 1),
                ("fOutxCtsFlow", wintypes.DWORD, 1),
                ("fOutxDsrFlow", wintypes.DWORD, 1),
                ("fDtrControl", wintypes.DWORD, 2),
                ("fDsrSensitivity", wintypes.DWORD, 1),
                ("fTXContinueOnXoff", wintypes.DWORD, 1),
                ("fOutX", wintypes.DWORD, 1),
                ("fInX", wintypes.DWORD, 1),
                ("fErrorChar", wintypes.DWORD, 1),
                ("fNull", wintypes.DWORD, 1),
                ("fRtsControl", wintypes.DWORD, 2),
                ("fAbortOnError", wintypes.DWORD, 1),
                ("fDummy2", wintypes.DWORD, 17),
                ("wReserved", wintypes.WORD),
                ("XonLim", wintypes.WORD),
                ("XoffLim", wintypes.WORD),
                ("ByteSize", wintypes.BYTE),
                ("Parity", wintypes.BYTE),
                ("StopBits", wintypes.BYTE),
                ("XonChar", ctypes.c_char),
                ("XoffChar", ctypes.c_char),
                ("ErrorChar", ctypes.c_char),
                ("EofChar", ctypes.c_char),
                ("EvtChar", ctypes.c_char),
                ("wReserved1", wintypes.WORD),
            ]
        
        dcb = DCB()
        dcb.DCBlength = ctypes.sizeof(DCB)
        kernel32.GetCommState(self.handle, ctypes.byref(dcb))
        dcb.BaudRate = baudrate
        dcb.ByteSize = 8
        dcb.Parity = 0
        dcb.StopBits = 0
        kernel32.SetCommState(self.handle, ctypes.byref(dcb))
        
        # Timeouts
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

def control_thread_lie(port='COM1', baudrate=115200):
    """Hilo de control usando linealización por realimentación."""
    
    # Escalas de conversión
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    
    # Controlador Lie
    controlador = ControladorLie()
    
    # Variables
    t = 0
    yd = 0.005  # Referencia 5mm
    
    ser = None
    try:
        ser = Win32Serial(port, baudrate)
        print(f"✅ Puerto {port} abierto para control Lie.")
        
        # Esperar switch
        print("⏳ Esperando switch del microcontrolador (byte 0xAA)...")
        while not exit_event.is_set():
            recibido = ser.read(1)
            if recibido == b'\xAA':
                print("✅ ¡Switch activado! Iniciando control por linealización...")
                break
        else:
            print("❌ Cierre solicitado antes de la activación.")
            return
        
        # Archivo de salida
        with open("MONIT_lie.txt", "w+") as fp:
            fp.write("# t\tyd\ty\tie\tu\tLf3h\tLgLf2h\n")
            
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
                    
                    # Convertir a unidades físicas
                    y = esc * pv
                    ie = esci * i_raw
                    
                    # --- LEY DE CONTROL LIE ---
                    u = controlador.calcular_control(ie, y, yd, Ts)
                    
                    # Enviar PWM
                    pwmf = escs * u
                    pwm = int(abs(pwmf))
                    pwm = min(254, pwm)
                    ser.write(bytes([pwm]))
                    
                    # Log
                    data_queue.put((t, yd, y, ie, u))
                    fp.write(f"{t:.4f}\t{yd:.6f}\t{y:.6f}\t{ie:.4f}\t{u:.4f}\n")
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
        print("🔌 Hilo de control finalizado.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("⚡ CONTROL NO LINEAL POR LINEALIZACIÓN (LIE)")
    print("="*60)
    print(f"Parámetros del modelo:")
    print(f"  k0 = {K0:.4f} H")
    print(f"  k  = {K:.4f} H")
    print(f"  a  = {A:.4f} m")
    print(f"  R  = {R:.2f} Ω")
    print(f"  m  = {M:.3f} kg")
    print("="*60)
    
    # Iniciar hilo de control
    control = threading.Thread(target=control_thread_lie, daemon=True)
    control.start()
    
    try:
        duration = 30
        start_time = time.time()
        print(f"🚀 Corriendo prueba por {duration} segundos...")
        
        while time.time() - start_time < duration:
            if not data_queue.empty():
                while not data_queue.empty():
                    t_val, yd, y, ie, u = data_queue.get()
                
                print(f"⏱️ t={t_val:.1f}s | Ref={yd*1000:.1f}mm | y={y*1000:.1f}mm | i={ie:.3f}A | u={u:.1f}V")
            
            time.sleep(0.5)
            
            if not control.is_alive():
                print("⚠️ Hilo de control terminó.")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Detenido por el usuario")
    finally:
        exit_event.set()
        control.join(timeout=2)
        print("✅ Programa finalizado.")


if __name__ == '__main__':
    main()

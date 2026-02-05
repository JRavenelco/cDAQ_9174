"""
Replicación exacta del código C++ levitador.cpp en Python
Usa Win32 API directamente (CreateFile, ReadFile, WriteFile)
"""
import time
import ctypes
from ctypes import wintypes
import argparse

# Constantes Win32 (idénticas a C++)
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = -1

# Cargar kernel32.dll
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Definir funciones Win32 (idénticas a C++)
CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
]
CreateFileW.restype = wintypes.HANDLE

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

ReadFile = kernel32.ReadFile
ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
ReadFile.restype = wintypes.BOOL

WriteFile = kernel32.WriteFile
WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
WriteFile.restype = wintypes.BOOL

# ============ PARÁMETROS DE CONTROL (del C++) ============
Ts = 0.01
kp = 100
ki = 50
kd = 1.5
kpi = 12.0
kii = 3000.0
Vref = 9.86
Iref = 0.827
Rs = 2.2

# ============ VARIABLES DE ESTADO (del C++) ============
pv = 0
i = 0
y = 0.0
y_1 = 0.0
ef = 0.0
ef_1 = 0.0
u = 0.0
t = 0.0
esc = 0.05 / 1023.0
esci = 5.0 / (Rs * 1023.0)
escs = 254.0 / Vref
iTs = 1 / Ts
pwmf = 0.0
yd = 0.005
proporcional = 0.0
derivativa = 0.0
ie = 0.0
ied = 0.0
id = 0.0
ei = 0.0
propi = 0.0
intei = 0.0
integral = 0.0
flagcom = 0
pwm = 0

# Argumentos CLI
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--port', '-p', default='COM1')
args, _unknown = parser.parse_known_args()

# ============ ABRIR PUERTO (IDÉNTICO A C++) ============
print(f"Abriendo puerto {args.port}...")
h = CreateFileW(
    args.port,
    GENERIC_READ | GENERIC_WRITE,
    0,  # Sin compartir (acceso exclusivo)
    None,
    OPEN_EXISTING,
    0,
    None
)

if h == INVALID_HANDLE_VALUE:
    error_code = ctypes.get_last_error()
    raise SystemExit(f"Error al abrir {args.port}: {ctypes.FormatError(error_code)}")

print(f"✅ Puerto {args.port} abierto exitosamente")
print("\n" + "="*80)
print("⏳ ESPERANDO SWITCH DEL MICROCONTROLADOR...")
print("="*80)
print("Presiona el switch en el hardware para comenzar el envío de datos")
print("(El script esperará indefinidamente hasta recibir el primer byte 0xAA)")
print("Presiona Ctrl+C para cancelar\n")

# ============ ESPERAR SINCRONIZACIÓN INICIAL ============
sync_found = False
while not sync_found:
    buffer = ctypes.create_string_buffer(1)
    bytes_read = wintypes.DWORD()
    
    if ReadFile(h, buffer, 1, ctypes.byref(bytes_read), None):
        if bytes_read.value > 0:
            recibido = buffer.raw[0]
            if recibido == 0xAA:
                print("✅ ¡Switch activado! Byte de sincronización 0xAA recibido")
                print("Iniciando control de levitación...\n")
                sync_found = True
                flagcom = 1  # Ya recibimos el 0xAA
                break

print("Tiempo\t\tyd\t\ty\t\tied\t\tie\t\tu\t\tPWM")
print("-" * 80)

# ============ BUCLE PRINCIPAL (IDÉNTICO A C++) ============
with open("MONIT.txt", "w+") as fp:
    try:
        while True:
            # Leer 1 byte (idéntico a ReadFile en C++)
            buffer = ctypes.create_string_buffer(1)
            bytes_read = wintypes.DWORD()
            
            if not ReadFile(h, buffer, 1, ctypes.byref(bytes_read), None):
                print("Error al leer del puerto")
                break
            
            if bytes_read.value == 0:
                continue
            
            recibido = buffer.raw[0]  # Obtener byte como int (0-255)
            
            # ============ PROTOCOLO DEL PIC (del C++) ============
            if flagcom != 0:
                flagcom += 1
            
            # Byte de sincronización 0xAA
            if (recibido == 0xAA) and (flagcom == 0):
                pv = 0
                i = 0
                flagcom = 1
            
            # Byte alto de posición
            if flagcom == 2:
                pv = recibido
                pv = pv << 8
            
            # Byte bajo de posición
            if flagcom == 3:
                pv = pv + recibido
            
            # Byte alto de corriente
            if flagcom == 4:
                i = recibido
                i = i << 8
            
            # Byte bajo de corriente + CÁLCULO DE CONTROL
            if flagcom == 5:
                if t > 10:
                    yd = 0.0045
                if t > 15:
                    yd = 0.005
                
                ef_1 = ef
                y_1 = y
                
                i = i + recibido
                y = esc * pv
                ie = esci * i
                ef = yd - y
                proporcional = kp * ef
                derivativa = kd * (ef - ef_1) * iTs
                
                # Integral con saturación
                if (integral < Iref) and (integral > -Iref):
                    integral = integral + ki * Ts * ef
                else:
                    if integral >= Iref:
                        integral = 0.95 * Iref
                    if integral <= -Iref:
                        integral = -0.95 * Iref
                
                id = proporcional + integral + derivativa
                
                # Solo aplicamos corriente negativa (fuerza magnética es atracción)
                if id > 0:
                    id = 0
                if id <= -Iref:
                    id = -Iref
                
                ied = -id  # La fuerza magnética va en sentido contrario
                ei = ied - ie
                propi = kpi * ei
                
                # Integral de voltaje con saturación
                if (intei < Vref) and (intei > -Vref):
                    intei = intei + kii * Ts * ei
                else:
                    if intei >= Vref:
                        intei = 0.95 * Vref
                    if intei <= -Vref:
                        intei = -0.95 * Vref
                
                u = propi + intei
                
                # Saturación de voltaje
                if u > Vref:
                    u = Vref
                if u <= 0:
                    u = 0
                
                pwmf = u
                pwmf = escs * pwmf  # Escalamiento de salida
                pwm = int(abs(pwmf))
                
                # ============ ENVIAR PWM AL PIC (IDÉNTICO A C++) ============
                enviar = bytes([pwm])
                bytes_written = wintypes.DWORD()
                
                if not WriteFile(h, enviar, 1, ctypes.byref(bytes_written), None):
                    print("Error al escribir al puerto")
                    break
                
                # Mostrar datos en pantalla
                print(f"{t:.4f}\t\t{yd:.4f}\t\t{y:.4f}\t\t{ied:.4f}\t\t{ie:.4f}\t\t{u:.4f}\t\t{pwm}")
                
                # Escribir en archivo
                fp.write(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}\n")
                fp.flush()
                
                flagcom = 0
                t = t + Ts
    
    except KeyboardInterrupt:
        print(f"\n\nInterrumpido por el usuario")
        print(f"Tiempo total: {t:.2f}s")

# ============ CERRAR PUERTO ============
CloseHandle(h)
print("\nPuerto cerrado. Fin del programa.")

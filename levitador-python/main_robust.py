import time
import ctypes
from ctypes import wintypes
import argparse

# Constantes Win32
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = -1

# Cargar kernel32.dll
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Definir funciones Win32
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

# Configurar timeouts
class COMMTIMEOUTS(ctypes.Structure):
    _fields_ = [
        ('ReadIntervalTimeout', wintypes.DWORD),
        ('ReadTotalTimeoutMultiplier', wintypes.DWORD),
        ('ReadTotalTimeoutConstant', wintypes.DWORD),
        ('WriteTotalTimeoutMultiplier', wintypes.DWORD),
        ('WriteTotalTimeoutConstant', wintypes.DWORD),
    ]

SetCommTimeouts = kernel32.SetCommTimeouts
SetCommTimeouts.argtypes = [wintypes.HANDLE, ctypes.POINTER(COMMTIMEOUTS)]
SetCommTimeouts.restype = wintypes.BOOL

# Parámetros de control
Ts = 0.01
kp = 100
ki = 50
kd = 1.5
kpi = 12.0
kii = 3000.0
Vref = 9.86
Iref = 0.827
Rs = 2.2

# Variables de estado y control
pv = 0
i = 0
y = 0
y_1 = 0
ef = 0
ef_1 = 0
u = 0
t = 0
esc = 0.05 / 1023.0
esci = 5.0 / (Rs * 1023.0)
escs = 254.0 / Vref
iTs = 1 / Ts
pwmf = 0
yd = 0.005
proporcional = 0
derivativa = 0
ie = 0
ied = 0
id = 0
ei = 0
propi = 0
intei = 0
integral = 0
flagcom = 0

# Argumentos CLI
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--port', '-p', default='COM1')
args, _unknown = parser.parse_known_args()

# Abrir puerto serie con Win32 API
print(f"🔌 Abriendo {args.port} con Win32 API...")
h = CreateFileW(
    args.port,
    GENERIC_READ | GENERIC_WRITE,
    0,
    None,
    OPEN_EXISTING,
    0,
    None
)

if h == INVALID_HANDLE_VALUE:
    error_code = ctypes.get_last_error()
    raise SystemExit(f"❌ No se pudo abrir {args.port}: Error {error_code} - {ctypes.FormatError(error_code)}")

print(f"✅ Puerto {args.port} abierto exitosamente")

# Nota: Sin configurar timeouts explícitos (usa default del puerto)
print("\n" + "="*60)
print("VERIFICANDO COMUNICACIÓN CON EL MICROCONTROLADOR (PIC16F877A)")
print("="*60)
print("\n🔍 Esperando byte de sincronización (0xAA)...")
print("   NOTA: El pin PC0 del PIC debe estar en HIGH (modo PC)")
print("   Si no hay datos en 5 segundos, verifica:")
print("   - Microcontrolador encendido y programado")
print("   - Pin PC0 conectado a +5V (modo control PC)")
print("   - Cable TX del PIC conectado a RX de COM1\n")

# Buscar sincronización inicial
sync_found = False
start_time = time.time()
timeout_seconds = 5

while not sync_found and (time.time() - start_time) < timeout_seconds:
    buffer = ctypes.create_string_buffer(1)
    bytes_read = wintypes.DWORD()
    
    if ReadFile(h, buffer, 1, ctypes.byref(bytes_read), None):
        if bytes_read.value > 0:
            recibido = buffer.raw[0]
            if recibido == 0xAA:
                print(f"✅ Sincronización encontrada! (0xAA después de {time.time()-start_time:.2f}s)")
                sync_found = True
            else:
                print(f"   Byte recibido: 0x{recibido:02X} (esperando 0xAA...)")

if not sync_found:
    CloseHandle(h)
    raise SystemExit(f"\n❌ TIMEOUT: No se recibió 0xAA en {timeout_seconds}s\n"
                     "   El microcontrolador NO está enviando datos.\n"
                     "   Verifica que el pin PC0 esté en HIGH (modo PC)")

print("\n" + "="*60)
print("INICIANDO BUCLE DE CONTROL")
print("="*60)
print("Presiona Ctrl+C para salir\n")

# Apertura del archivo de registro
with open("MONIT.txt", "w+") as fp:
    exit_requested = False
    flagcom = 1  # Ya recibimos el 0xAA
    bytes_received = 0
    last_print_time = time.time()
    
    try:
        while not exit_requested:
            # Leer 1 byte
            buffer = ctypes.create_string_buffer(1)
            bytes_read = wintypes.DWORD()
            
            if not ReadFile(h, buffer, 1, ctypes.byref(bytes_read), None):
                error_code = ctypes.get_last_error()
                print(f"\n❌ Error al leer: {ctypes.FormatError(error_code)}")
                break
            
            if bytes_read.value == 0:
                # Timeout - micro dejó de enviar
                if time.time() - last_print_time > 2.0:
                    print("\n⚠️  Sin datos del micro por >2s. ¿Se desconectó o cambió a modo PIC?")
                    last_print_time = time.time()
                continue
            
            recibido = buffer.raw[0:1]
            bytes_received += 1
            
            if flagcom != 0:
                flagcom += 1
            
            if recibido == b'\xAA' and flagcom == 0:
                pv = 0
                i = 0
                flagcom = 1
            
            if flagcom == 2:
                pv = (pv << 8) + recibido[0]
            
            if flagcom == 3:
                pv = (pv << 8) + recibido[0]
            
            if flagcom == 4:
                i = (i << 8) + recibido[0]
            
            if flagcom == 5:
                if t > 10:
                    yd = 0.0045
                if t > 15:
                    yd = 0.005
                
                ef_1 = ef
                y_1 = y
                
                i = (i << 8) + recibido[0]
                y = esc * pv
                ie = esci * i
                ef = yd - y
                proporcional = kp * ef
                derivativa = kd * (ef - ef_1) * iTs
                
                if -Iref < integral < Iref:
                    integral = integral + ki * Ts * ef
                else:
                    if integral >= Iref:
                        integral = 0.95 * Iref
                    if integral <= -Iref:
                        integral = -0.95 * Iref
                
                id = proporcional + integral + derivativa
                
                if id > 0:
                    id = 0
                if id <= -Iref:
                    id = -Iref
                
                ied = -id
                ei = ied - ie
                propi = kpi * ei
                
                if -Vref < intei < Vref:
                    intei = intei + kii * Ts * ei
                else:
                    if intei >= Vref:
                        intei = 0.95 * Vref
                    if intei <= -Vref:
                        intei = -0.95 * Vref
                
                u = propi + intei
                
                if u > Vref:
                    u = Vref
                if u <= 0:
                    u = 0
                
                pwmf = u
                pwmf = escs * pwmf
                pwm = int(abs(pwmf))
                
                # Enviar PWM
                enviar = bytes([pwm])
                bytes_written = wintypes.DWORD()
                
                if not WriteFile(h, enviar, 1, ctypes.byref(bytes_written), None):
                    print("\n❌ Error al escribir PWM al puerto")
                    break
                
                print(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}\tPWM={pwm}")
                
                # Escribir algunos datos en el archivo
                fp.write(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}\n")
                fp.flush()  # Forzar escritura inmediata
                
                flagcom = 0
                t = t + Ts
                last_print_time = time.time()
                
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Interrumpido por el usuario (Ctrl+C)")
        print(f"   Bytes totales recibidos: {bytes_received}")
        print(f"   Tiempo de ejecución: {t:.2f}s")

# Cierra el puerto serie al salir del bucle
CloseHandle(h)
print("\n✅ Puerto cerrado. Fin del programa.")

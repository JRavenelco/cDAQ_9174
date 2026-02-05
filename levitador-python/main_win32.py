import time
import ctypes
from ctypes import wintypes
import msvcrt
import argparse

# Constantes Win32
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = -1
NOPARITY = 0
ONESTOPBIT = 0

# Cargar kernel32.dll
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Definir CreateFileW
CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
]
CreateFileW.restype = wintypes.HANDLE

# Definir GetCommState y SetCommState
GetCommState = kernel32.GetCommState
GetCommState.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
GetCommState.restype = wintypes.BOOL

SetCommState = kernel32.SetCommState
SetCommState.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
SetCommState.restype = wintypes.BOOL

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [wintypes.HANDLE]
CloseHandle.restype = wintypes.BOOL

ReadFile = kernel32.ReadFile
ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
ReadFile.restype = wintypes.BOOL

WriteFile = kernel32.WriteFile
WriteFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
WriteFile.restype = wintypes.BOOL

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
parser.add_argument('--baud', '-b', type=int, default=115200)
args, _unknown = parser.parse_known_args()

# Abrir puerto serie con Win32 API (como C++)
print(f"Abriendo {args.port} @ {args.baud} baud con Win32 API...")
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
    raise SystemExit(f"No se pudo abrir {args.port}: Error {error_code} - {ctypes.FormatError(error_code)}")

print(f"✅ Puerto {args.port} abierto exitosamente")

# Configurar DCB (estructura completa de Windows)
class DCB(ctypes.Structure):
    _fields_ = [
        ('DCBlength', wintypes.DWORD),
        ('BaudRate', wintypes.DWORD),
        ('flags', wintypes.DWORD),
        ('wReserved', wintypes.WORD),
        ('XonLim', wintypes.WORD),
        ('XoffLim', wintypes.WORD),
        ('ByteSize', wintypes.BYTE),
        ('Parity', wintypes.BYTE),
        ('StopBits', wintypes.BYTE),
        ('XonChar', wintypes.BYTE),
        ('XoffChar', wintypes.BYTE),
        ('ErrorChar', wintypes.BYTE),
        ('EofChar', wintypes.BYTE),
        ('EvtChar', wintypes.BYTE),
        ('wReserved1', wintypes.WORD),
    ]

dcb = DCB()
dcb.DCBlength = ctypes.sizeof(DCB)

if not GetCommState(h, ctypes.byref(dcb)):
    CloseHandle(h)
    raise SystemExit("No se pudo obtener configuración del puerto")

dcb.BaudRate = args.baud
dcb.ByteSize = 8
dcb.Parity = NOPARITY
dcb.StopBits = ONESTOPBIT
dcb.flags = 0x00000003

if not SetCommState(h, ctypes.byref(dcb)):
    CloseHandle(h)
    raise SystemExit("No se pudo configurar el puerto")

print("Puerto configurado: 115200 8N1")

# Apertura del archivo de registro
with open("MONIT.txt", "w+") as fp:
    exit_requested = False
    
    print("Iniciando bucle de control... (Presiona Ctrl+C para salir)")
    
    while not exit_requested:
        # Leer 1 byte
        buffer = ctypes.create_string_buffer(1)
        bytes_read = wintypes.DWORD()
        
        if not ReadFile(h, buffer, 1, ctypes.byref(bytes_read), None):
            print("Error al leer del puerto")
            break
        
        if bytes_read.value == 0:
            continue
        
        recibido = buffer.raw[0:1]
        
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
                print("Error al escribir al puerto")
                break
            
            print(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}")
            
            # Escribir algunos datos en el archivo
            #fp.write(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}\n")
            
            flagcom = 0
            t = t + Ts

# Cierra el puerto serie al salir del bucle
CloseHandle(h)
print("\nPuerto cerrado. Fin del programa.")

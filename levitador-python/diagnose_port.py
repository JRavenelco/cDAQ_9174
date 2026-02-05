"""
Script de diagnóstico para verificar si hay datos en COM1
"""
import time
import ctypes
from ctypes import wintypes

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

# Abrir puerto
print("🔌 Abriendo COM1...")
h = CreateFileW(
    "COM1",
    GENERIC_READ | GENERIC_WRITE,
    0,
    None,
    OPEN_EXISTING,
    0,
    None
)

if h == INVALID_HANDLE_VALUE:
    error_code = ctypes.get_last_error()
    raise SystemExit(f"❌ Error al abrir COM1: {ctypes.FormatError(error_code)}")

print("✅ COM1 abierto")
print("\n📊 DIAGNÓSTICO: Leyendo datos por 10 segundos...\n")

start_time = time.time()
bytes_received = 0
sync_bytes = 0
data = []

while time.time() - start_time < 10:
    buffer = ctypes.create_string_buffer(1)
    bytes_read = wintypes.DWORD()
    
    if ReadFile(h, buffer, 1, ctypes.byref(bytes_read), None):
        if bytes_read.value > 0:
            byte_val = buffer.raw[0]
            bytes_received += 1
            data.append(byte_val)
            
            if byte_val == 0xAA:
                sync_bytes += 1
                print(f"✅ Byte de sincronización 0xAA encontrado (#{sync_bytes})")
            else:
                print(f"   Byte: 0x{byte_val:02X} ({byte_val:3d})")
    else:
        error_code = ctypes.get_last_error()
        print(f"❌ ReadFile error: {ctypes.FormatError(error_code)}")
        break

elapsed = time.time() - start_time
CloseHandle(h)

print("\n" + "="*60)
print("RESULTADOS:")
print("="*60)
print(f"Tiempo transcurrido: {elapsed:.2f}s")
print(f"Bytes totales recibidos: {bytes_received}")
print(f"Bytes de sincronización (0xAA): {sync_bytes}")

if bytes_received == 0:
    print("\n❌ NO SE RECIBIÓ NINGÚN DATO")
    print("\nPosibles causas:")
    print("  1. Microcontrolador no está encendido")
    print("  2. Pin PC0 del PIC está en LOW (debe estar en HIGH)")
    print("  3. Cable TX del PIC no está conectado a RX de COM1")
    print("  4. Microcontrolador está en reset o bloqueado")
    print("  5. Baudrate incorrecto (debe ser 115200)")
elif sync_bytes == 0:
    print("\n⚠️  Se recibieron datos pero NO se encontró sincronización (0xAA)")
    print(f"Primeros 20 bytes: {[f'0x{b:02X}' for b in data[:20]]}")
else:
    print(f"\n✅ COMUNICACIÓN CORRECTA")
    print(f"Frecuencia de sincronización: ~{sync_bytes/elapsed:.1f} Hz (esperado: ~100 Hz = 10ms)")

print("\nPuerto cerrado.")

import ctypes
from ctypes import wintypes

# Constantes Win32
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = -1

# Cargar kernel32.dll
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Definir CreateFileW
CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    wintypes.LPCWSTR,  # lpFileName
    wintypes.DWORD,    # dwDesiredAccess
    wintypes.DWORD,    # dwShareMode
    wintypes.LPVOID,   # lpSecurityAttributes
    wintypes.DWORD,    # dwCreationDisposition
    wintypes.DWORD,    # dwFlagsAndAttributes
    wintypes.HANDLE    # hTemplateFile
]
CreateFileW.restype = wintypes.HANDLE

print("Intentando abrir COM1 con Win32 API (como C++)...")
print("Formato 1: COM1")
handle = CreateFileW(
    "COM1",
    GENERIC_READ | GENERIC_WRITE,
    0,  # Sin compartir (acceso exclusivo)
    None,
    OPEN_EXISTING,
    0,
    None
)

if handle == INVALID_HANDLE_VALUE:
    error_code = ctypes.get_last_error()
    print(f"  ❌ Falló con error {error_code}: {ctypes.FormatError(error_code)}")
    
    print("\nFormato 2: \\\\.\\COM1")
    handle = CreateFileW(
        "\\\\.\\COM1",
        GENERIC_READ | GENERIC_WRITE,
        0,
        None,
        OPEN_EXISTING,
        0,
        None
    )
    
    if handle == INVALID_HANDLE_VALUE:
        error_code = ctypes.get_last_error()
        print(f"  ❌ Falló con error {error_code}: {ctypes.FormatError(error_code)}")
        
        print("\n🔍 DIAGNÓSTICO:")
        print("  - Error 2 (FileNotFound): El puerto COM1 no existe o driver deshabilitado")
        print("  - Error 5 (AccessDenied): Otro programa tiene COM1 abierto")
        print("  - Error 31 (DeviceNotReady): Dispositivo desconectado")
        
        print("\n📋 SOLUCIONES:")
        print("  1. Abre Device Manager (devmgmt.msc)")
        print("  2. Busca 'Puertos (COM y LPT)'")
        print("  3. Verifica que COM1 (Prolific USB-to-Serial) esté:")
        print("     - Sin símbolo de advertencia (triángulo amarillo)")
        print("     - Habilitado (clic derecho > Habilitar)")
        print("  4. Si tiene advertencia, actualiza el driver:")
        print("     - Clic derecho > Actualizar controlador")
        print("     - O desinstala y reconecta el USB")
    else:
        print(f"  ✅ Abierto exitosamente con formato \\\\.\\COM1")
        kernel32.CloseHandle(handle)
else:
    print(f"  ✅ Abierto exitosamente con formato COM1")
    kernel32.CloseHandle(handle)

print("\n" + "="*60)
print("Probando todos los puertos COM1-COM20...")
print("="*60)

for i in range(1, 21):
    port = f"\\\\.\\COM{i}"
    handle = CreateFileW(
        port,
        GENERIC_READ | GENERIC_WRITE,
        0,
        None,
        OPEN_EXISTING,
        0,
        None
    )
    
    if handle != INVALID_HANDLE_VALUE:
        print(f"✅ COM{i} - DISPONIBLE y se puede abrir")
        kernel32.CloseHandle(handle)
    else:
        error_code = ctypes.get_last_error()
        if error_code == 2:
            pass  # No existe, no mostrar
        elif error_code == 5:
            print(f"⚠️  COM{i} - EXISTE pero está EN USO por otra aplicación")
        else:
            print(f"❌ COM{i} - Error {error_code}: {ctypes.FormatError(error_code)}")

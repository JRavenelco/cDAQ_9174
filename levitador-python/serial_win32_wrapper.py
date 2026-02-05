"""
Wrapper Win32 para puerto serie que replica el comportamiento de CreateFile en C++
"""
import ctypes
from ctypes import wintypes
import time

# Constantes Win32
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = -1

# DCB constants
NOPARITY = 0
ONESTOPBIT = 0

# Cargar kernel32.dll
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

# Definir CreateFileW
CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE
]
CreateFileW.restype = wintypes.HANDLE

class Win32Serial:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.handle = None
        
    def open(self):
        """Abre el puerto serie usando Win32 API (como C++)"""
        self.handle = CreateFileW(
            self.port,
            GENERIC_READ | GENERIC_WRITE,
            0,  # Sin compartir
            None,
            OPEN_EXISTING,
            0,
            None
        )
        
        if self.handle == INVALID_HANDLE_VALUE:
            error_code = ctypes.get_last_error()
            raise IOError(f"No se pudo abrir {self.port}: Error {error_code} - {ctypes.FormatError(error_code)}")
        
        # Configurar DCB (como en C++)
        class DCB(ctypes.Structure):
            _fields_ = [
                ('DCBlength', wintypes.DWORD),
                ('BaudRate', wintypes.DWORD),
                ('flags', wintypes.DWORD),  # Todos los flags en un solo DWORD
                ('wReserved', wintypes.WORD),
                ('XonLim', wintypes.WORD),
                ('XoffLim', wintypes.WORD),
                ('ByteSize', wintypes.BYTE),
                ('Parity', wintypes.BYTE),
                ('StopBits', wintypes.BYTE),
                ('XonChar', ctypes.c_char),
                ('XoffChar', ctypes.c_char),
                ('ErrorChar', ctypes.c_char),
                ('EofChar', ctypes.c_char),
                ('EvtChar', ctypes.c_char),
                ('wReserved1', wintypes.WORD),
            ]
        
        dcb = DCB()
        dcb.DCBlength = ctypes.sizeof(DCB)
        
        if not kernel32.GetCommState(self.handle, ctypes.byref(dcb)):
            self.close()
            raise IOError("No se pudo obtener configuración del puerto")
        
        # Configurar como en C++
        dcb.BaudRate = self.baudrate
        dcb.ByteSize = 8
        dcb.Parity = NOPARITY
        dcb.StopBits = ONESTOPBIT
        dcb.flags = 0x00000003  # fBinary=1, fParity=1
        
        if not kernel32.SetCommState(self.handle, ctypes.byref(dcb)):
            self.close()
            raise IOError("No se pudo configurar el puerto")
        
        print(f"✅ Puerto {self.port} abierto exitosamente con Win32 API")
    
    def read(self, size=1):
        """Lee bytes del puerto"""
        buffer = ctypes.create_string_buffer(size)
        bytes_read = wintypes.DWORD()
        
        if not kernel32.ReadFile(
            self.handle,
            buffer,
            size,
            ctypes.byref(bytes_read),
            None
        ):
            raise IOError("Error al leer del puerto")
        
        return buffer.raw[:bytes_read.value]
    
    def write(self, data):
        """Escribe bytes al puerto"""
        if isinstance(data, (bytes, bytearray)):
            buffer = data
        else:
            buffer = bytes(data)
        
        bytes_written = wintypes.DWORD()
        
        if not kernel32.WriteFile(
            self.handle,
            buffer,
            len(buffer),
            ctypes.byref(bytes_written),
            None
        ):
            raise IOError("Error al escribir al puerto")
        
        return bytes_written.value
    
    def close(self):
        """Cierra el puerto"""
        if self.handle and self.handle != INVALID_HANDLE_VALUE:
            kernel32.CloseHandle(self.handle)
            self.handle = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()

import ctypes
from ctypes import wintypes

# Constantes de la API de Win32
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = -1

class DCB(ctypes.Structure):
    _fields_ = [("DCBlength", wintypes.DWORD),
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
                ("wReserved1", wintypes.WORD)]

class Win32Serial:
    def __init__(self, port, baudrate):
        port_name = f"\\\\.\\{port}"
        self.handle = ctypes.windll.kernel32.CreateFileW(
            port_name,
            GENERIC_READ | GENERIC_WRITE,
            0, 0,
            OPEN_EXISTING,
            0, 0
        )
        if self.handle == INVALID_HANDLE_VALUE:
            raise ctypes.WinError()

        dcb = DCB()
        dcb.DCBlength = ctypes.sizeof(dcb)
        ctypes.windll.kernel32.GetCommState(self.handle, ctypes.byref(dcb))
        dcb.BaudRate = baudrate
        dcb.ByteSize = 8
        dcb.Parity = 0
        dcb.StopBits = 0 # 1 stop bit
        if not ctypes.windll.kernel32.SetCommState(self.handle, ctypes.byref(dcb)):
            raise ctypes.WinError()

        # Configurar COMMTIMEOUTS (crítico para lectura no bloqueante)
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
        if not ctypes.windll.kernel32.SetCommTimeouts(self.handle, ctypes.byref(timeouts)):
            raise ctypes.WinError()


    def read(self, size=1):
        buffer = ctypes.create_string_buffer(size)
        bytes_read = wintypes.DWORD(0)
        if not ctypes.windll.kernel32.ReadFile(self.handle, buffer, size, ctypes.byref(bytes_read), None):
            raise ctypes.WinError()
        return buffer.raw[:bytes_read.value]

    def write(self, data):
        bytes_written = wintypes.DWORD(0)
        if not ctypes.windll.kernel32.WriteFile(self.handle, data, len(data), ctypes.byref(bytes_written), None):
            raise ctypes.WinError()
        return bytes_written.value

    def close(self):
        if self.handle != INVALID_HANDLE_VALUE:
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = INVALID_HANDLE_VALUE

    @property
    def is_open(self):
        return self.handle != INVALID_HANDLE_VALUE

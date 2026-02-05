"""
Test rápido del observador híbrido Obs2023 + KAN
"""

import time
import threading
import queue
import ctypes
from ctypes import wintypes

# Importar el observador híbrido
from kan_corrector_sensorless import ObservadorHibridoKAN

# Parámetros
Ts = 0.01
Vref = 9.86
Iref = 0.827
Rs = 2.2
kp, ki, kd = 100, 50, 1.5
kpi, kii = 12.0, 3000.0

# Serial
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

class Win32Serial:
    def __init__(self, port, baudrate):
        self.handle = kernel32.CreateFileW(f"\\\\.\\{port}", 0xC0000000, 0, None, 3, 0, None)
        if self.handle == -1:
            raise Exception(f"No se pudo abrir {port}")
        
        class DCB(ctypes.Structure):
            _fields_ = [
                ("DCBlength", ctypes.wintypes.DWORD), ("BaudRate", ctypes.wintypes.DWORD),
                ("fBinary", ctypes.wintypes.DWORD, 1), ("fParity", ctypes.wintypes.DWORD, 1),
                ("fOutxCtsFlow", ctypes.wintypes.DWORD, 1), ("fOutxDsrFlow", ctypes.wintypes.DWORD, 1),
                ("fDtrControl", ctypes.wintypes.DWORD, 2), ("fDsrSensitivity", ctypes.wintypes.DWORD, 1),
                ("fTXContinueOnXoff", ctypes.wintypes.DWORD, 1), ("fOutX", ctypes.wintypes.DWORD, 1),
                ("fInX", ctypes.wintypes.DWORD, 1), ("fErrorChar", ctypes.wintypes.DWORD, 1),
                ("fNull", ctypes.wintypes.DWORD, 1), ("fRtsControl", ctypes.wintypes.DWORD, 2),
                ("fAbortOnError", ctypes.wintypes.DWORD, 1), ("fDummy2", ctypes.wintypes.DWORD, 17),
                ("wReserved", ctypes.wintypes.WORD), ("XonLim", ctypes.wintypes.WORD),
                ("XoffLim", ctypes.wintypes.WORD), ("ByteSize", ctypes.wintypes.BYTE),
                ("Parity", ctypes.wintypes.BYTE), ("StopBits", ctypes.wintypes.BYTE),
                ("XonChar", ctypes.c_char), ("XoffChar", ctypes.c_char),
                ("ErrorChar", ctypes.c_char), ("EofChar", ctypes.c_char),
                ("EvtChar", ctypes.c_char), ("wReserved1", ctypes.wintypes.WORD),
            ]
        dcb = DCB()
        dcb.DCBlength = ctypes.sizeof(DCB)
        kernel32.GetCommState(self.handle, ctypes.byref(dcb))
        dcb.BaudRate = baudrate
        dcb.ByteSize = 8
        kernel32.SetCommState(self.handle, ctypes.byref(dcb))
        
        class COMMTIMEOUTS(ctypes.Structure):
            _fields_ = [("ReadIntervalTimeout", ctypes.wintypes.DWORD),
                       ("ReadTotalTimeoutMultiplier", ctypes.wintypes.DWORD),
                       ("ReadTotalTimeoutConstant", ctypes.wintypes.DWORD),
                       ("WriteTotalTimeoutMultiplier", ctypes.wintypes.DWORD),
                       ("WriteTotalTimeoutConstant", ctypes.wintypes.DWORD)]
        timeouts = COMMTIMEOUTS()
        timeouts.ReadIntervalTimeout = 50
        timeouts.ReadTotalTimeoutConstant = 1000
        kernel32.SetCommTimeouts(self.handle, ctypes.byref(timeouts))
        self.is_open = True
    
    def read(self, size):
        buf = ctypes.create_string_buffer(size)
        n = ctypes.wintypes.DWORD()
        kernel32.ReadFile(self.handle, buf, size, ctypes.byref(n), None)
        return buf.raw[:n.value]
    
    def write(self, data):
        w = ctypes.wintypes.DWORD()
        kernel32.WriteFile(self.handle, data, len(data), ctypes.byref(w), None)
        return w.value
    
    def close(self):
        if self.is_open:
            kernel32.CloseHandle(self.handle)
            self.is_open = False

data_queue = queue.Queue()
exit_event = threading.Event()

def control_thread():
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    iTs = 1 / Ts
    
    # Observador híbrido
    obs = ObservadorHibridoKAN()
    
    # PID
    t = 0
    yd = 0.005
    ef, ef_1 = 0, 0
    integral, intei = 0, 0
    u = 0
    
    try:
        ser = Win32Serial('COM1', 115200)
        print("✅ Puerto abierto")
        print("⏳ Esperando switch...")
        
        while not exit_event.is_set():
            if ser.read(1) == b'\xAA':
                print("✅ Switch activado!")
                break
        
        with open("MONIT_kan_hibrido.txt", "w") as fp:
            fp.write("# t\ty_s\ty_kan\tdelta\tie\tu\n")
            
            flagcom = 0
            pv = i_raw = 0
            
            while not exit_event.is_set():
                b = ser.read(1)
                if not b:
                    continue
                
                recibido = b[0]
                if flagcom != 0:
                    flagcom += 1
                
                if recibido == 0xAA and flagcom == 0:
                    pv = i_raw = 0
                    flagcom = 1
                    continue
                
                if flagcom == 2:
                    pv = recibido << 8
                elif flagcom == 3:
                    pv += recibido
                elif flagcom == 4:
                    i_raw = recibido << 8
                elif flagcom == 5:
                    i_raw += recibido
                    
                    if pv > 1023 or i_raw > 1023:
                        flagcom = 0
                        continue
                    
                    y_sensor = esc * pv
                    ie = esci * i_raw
                    
                    # Observador híbrido
                    y_kan, delta = obs.estimar(ie, u)
                    
                    # PID usa sensor (monitoreo)
                    y = y_sensor
                    
                    ef_1 = ef
                    ef = yd - y
                    prop = kp * ef
                    deriv = kd * (ef - ef_1) * iTs
                    
                    if -Iref < integral < Iref:
                        integral += ki * Ts * ef
                    else:
                        integral = 0.95 * Iref if integral >= Iref else -0.95 * Iref
                    
                    id_val = prop + integral + deriv
                    if id_val > 0: id_val = 0
                    if id_val <= -Iref: id_val = -Iref
                    
                    ei = -id_val - ie
                    propi = kpi * ei
                    
                    if -Vref < intei < Vref:
                        intei += kii * Ts * ei
                    else:
                        intei = 0.95 * Vref if intei >= Vref else -0.95 * Vref
                    
                    u = propi + intei
                    u = max(0, min(Vref, u))
                    
                    pwm = int(escs * u)
                    pwm = max(0, min(254, pwm))
                    ser.write(bytes([pwm]))
                    
                    data_queue.put((t, y_sensor, y_kan, delta, ie, u))
                    fp.write(f"{t:.4f}\t{y_sensor:.6f}\t{y_kan:.6f}\t{delta:.6f}\t{ie:.4f}\t{u:.4f}\n")
                    fp.flush()
                    
                    flagcom = 0
                    t += Ts
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ser' in dir() and ser.is_open:
            ser.close()
        print("🔌 Hilo finalizado")

def main():
    print("="*60)
    print("🧠 TEST OBSERVADOR HÍBRIDO: Obs2023 + KAN")
    print("="*60)
    
    th = threading.Thread(target=control_thread, daemon=True)
    th.start()
    
    try:
        start = time.time()
        while time.time() - start < 30:
            if not data_queue.empty():
                while not data_queue.empty():
                    t, y_s, y_k, delta, ie, u = data_queue.get()
                
                err = abs(y_s - y_k) * 1000
                print(f"t={t:.1f}s | Sensor={y_s*1000:.1f}mm | KAN={y_k*1000:.1f}mm | Δ={delta*1000:.2f}mm | Err={err:.1f}mm")
            
            time.sleep(0.5)
            if not th.is_alive():
                break
    except KeyboardInterrupt:
        print("\n⏹️ Detenido")
    finally:
        exit_event.set()
        th.join(timeout=2)
        print("✅ Finalizado")

if __name__ == '__main__':
    main()

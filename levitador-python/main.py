import time
import math
import msvcrt
import argparse
from serial.tools import list_ports

from serial_win32_wrapper import Win32Serial

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

# Configuración de la comunicación serie
def open_serial(desired_port=None, baud=115200):
    if desired_port:
        try:
            print(f"Abriendo {desired_port} @ {baud} baud con Win32 API...")
            ser = Win32Serial(desired_port, baud)
            ser.open()
            return ser
        except Exception as e:
            raise SystemExit(f"No se pudo abrir {desired_port}: {e}")
    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit("No se encontraron puertos serie. Conecta el dispositivo o usa --port COM1")
    def is_usb(p):
        d = (p.description or '')
        dev = (p.device or '')
        return ('USB' in d) or ('CH340' in d) or ('CP210' in d) or ('Arduino' in d) or ('ACM' in dev) or ('PL2303' in d) or ('Prolific' in d)
    ordered = sorted(ports, key=lambda p: (0 if is_usb(p) else 1, p.device))
    tried = []
    for p in ordered:
        try:
            print(f"Intentando {p.device} ({p.description})")
            ser = Win32Serial(p.device, baud)
            ser.open()
            return ser
        except Exception as e:
            print(f"Falló {p.device}: {e}")
            tried.append(f"{p.device} ({p.description})")
            continue
    listado = ", ".join(tried) if tried else ""
    raise SystemExit(f"No se pudo abrir ningún puerto. Intentados: {listado}. Usa --port COM1")

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--port', '-p', default=None)
parser.add_argument('--baud', '-b', type=int, default=115200)
args, _unknown = parser.parse_known_args()
ser = open_serial(args.port, args.baud)

# Apertura del archivo de registro
with open("MONIT.txt", "w+") as fp:
    exit_requested = False

    while not exit_requested:
        recibido = ser.read(1)

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

            enviar = bytes([pwm])
            ser.write(enviar)

            print(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}")

            # Escribir algunos datos en el archivo
            #fp.write(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}\n")

            flagcom = 0
            t = t + Ts


# Cierra el puerto serie al salir del bucle
ser.close()

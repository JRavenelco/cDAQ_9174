import time
import math
import serial
import asyncio
import threading
import msvcrt
import matplotlib.pyplot as plt

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
ser = serial.Serial('COM1', 115200, timeout=None)

# Apertura del archivo de registro
with open("MONIT.txt", "w+") as fp:
    exit_requested = False

    # Datos para la gráfica
    time_points = []
    y_values = []
    u_values = []

    async def plot_data():
        plt.ion()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        while not exit_requested:
            if len(time_points) > 0 and len(y_values) > 0 and len(u_values) > 0:
                ax1.clear()
                ax2.clear()

                ax1.plot(time_points, y_values, 'b', label='y')
                ax2.plot(time_points, u_values, 'r', label='u')

                ax1.set_xlabel('Tiempo')
                ax1.set_ylabel('y', color='b')
                ax2.set_ylabel('u', color='r')

                plt.pause(1)  # Actualizar el gráfico cada segundo

    async def main():
        global exit_requested
        global t, ef, ef_1, y, y_1, i, ie, id, integral, intei, pwmf, yd, pv

        t = 0
        ef = 0
        ef_1 = 0
        y = 0
        y_1 = 0
        i = 0
        ie = 0
        id = 0
        integral = 0
        intei = 0
        pwmf = 0
        yd = 0

        flagcom = 0

        # Apertura del archivo de registro dentro de la función main
        with open("MONIT.txt", "w+") as fp:

            while not exit_requested:
                recibido = ser.read(1)

                if flagcom != 0:
                    flagcom += 1

                if recibido == b'\xAA' and flagcom == 0:
                    pv = 0
                    i = 0
                    flagcom = 1

                if flagcom == 2:
                    pv = (pv << 8) + ord(recibido)

                if flagcom == 3:
                    pv = (pv << 8) + ord(recibido)

                if flagcom == 4:
                    i = (i << 8) + ord(recibido)

                if flagcom == 5:
                    if t > 10:
                        yd = 0.0045
                    if t > 15:
                        yd = 0.005

                    ef_1 = ef
                    y_1 = y

                    i = (i << 8) + ord(recibido)
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

                    time_points.append(t)
                    y_values.append(y)
                    u_values.append(u)

                    # Escribir algunos datos en el archivo
                    fp.write(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}\n")

                    flagcom = 0
                    t = t + Ts

        ser.close()

    def acquire_data():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())

    if __name__ == "__main__":
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        plot_thread = threading.Thread(target=lambda: loop.run_until_complete(plot_data()))
        plot_thread.daemon = True
        plot_thread.start()

        asyncio_thread = threading.Thread(target=acquire_data)
        asyncio_thread.start()

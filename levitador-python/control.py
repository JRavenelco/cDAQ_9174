import time
import math
from serial_win32 import Win32Serial
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Parámetros y Variables Globales ---
# Parámetros de control (CORREGIDOS)
Ts = 0.01
kp = 100
ki = 50
kd = 1.5
kpi = 12.0
kii = 3000.0
Vref = 9.86
Iref = 0.827
Rs = 2.2

USE_IMPROVED_PID = True
BETA_P = 0.8
KAW_I = 200.0
KAW_U = 2000.0

# Cola para comunicación entre hilos
data_queue = queue.Queue()
exit_event = threading.Event()

# --- Hilo de Control (Tiempo Real) ---
def control_thread(port='COM1', baudrate=115200):
    # Variables locales del hilo de control
    pv, i, y, y_1, ef, ef_1, u, t = 0, 0, 0, 0, 0, 0, 0, 0
    
    # Parámetros de escalado y offset (CORREGIDOS)
    esc = 0.05 / 1023.0  # Original
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    iTs = 1 / Ts
    pv_offset = 0  # Original
    pwmf, yd, proporcional, derivativa, ie, ied, id, ei, propi, intei, integral = 0, 0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0
    y_prev = 0.005

    ser = None
    try:
        ser = Win32Serial(port, baudrate)
        print(f"Puerto {port} abierto para control.")

        # Esperar switch de activación
        print("Esperando switch del microcontrolador (byte 0xAA)...")
        while not exit_event.is_set():
            recibido = ser.read(1)
            if recibido == b'\xAA':
                print(f"¡Switch activado! Usando offset fijo: {pv_offset}")
                break
        else:
            print("Cierre solicitado antes de la activación.")
            return

        # El bucle de control principal se ejecuta después de la sincronización
        with open("MONIT.txt", "w+") as fp:
            # Estado idéntico al C++
            flagcom = 0
            pv = 0
            i = 0
            last_byte_time = time.time()
            while not exit_event.is_set():
                b = ser.read(1)
                if not b:
                    # Timeout de byte, evitar bloqueo del hilo
                    if time.time() - last_byte_time > 1.0:
                        print("Timeout esperando byte.")
                        last_byte_time = time.time()
                    continue
                last_byte_time = time.time()
                recibido = b[0]

                if flagcom != 0:
                    flagcom += 1

                if (recibido == 0xAA) and (flagcom == 0):
                    pv = 0
                    i = 0
                    flagcom = 1
                    continue

                if flagcom == 2:
                    pv = recibido
                    pv = pv << 8
                    continue

                if flagcom == 3:
                    pv = pv + recibido
                    continue

                if flagcom == 4:
                    i = recibido
                    i = i << 8
                    continue

                if flagcom == 5:
                    i = i + recibido
                    # Guardia de integridad de frame
                    if pv > 1023 or i > 1023:
                        # Trama inválida: descartar y resincronizar
                        flagcom = 0
                        continue

                    # --- Ley de Control ---
                    if t > 10:
                        yd = 0.0045
                    if t > 15:
                        yd = 0.005

                    ef_1, y_1 = ef, y
                    y = esc * (pv - pv_offset)
                    ie = esci * i

                    if USE_IMPROVED_PID:
                        e_i = yd - y
                        e_p = (BETA_P * yd) - y
                        ef = e_i
                        d_y = (y - y_prev) * iTs
                        derivativa = -kd * d_y

                        id_unsat = (kp * e_p) + integral + derivativa
                        if id_unsat > 0:
                            id = 0.0
                        elif id_unsat <= -Iref:
                            id = -Iref
                        else:
                            id = id_unsat

                        integral = integral + (ki * Ts * e_i) + (KAW_I * (id - id_unsat) * Ts)
                        if integral > Iref:
                            integral = Iref
                        if integral < -Iref:
                            integral = -Iref

                        y_prev = y
                    else:
                        ef = yd - y
                        proporcional = kp * ef
                        derivativa = kd * (ef - ef_1) * iTs
                        if -Iref < integral < Iref:
                            integral += ki * Ts * ef
                        else:
                            integral = 0.95 * Iref if integral >= Iref else -0.95 * Iref
                        id = proporcional + integral + derivativa
                        if id > 0:
                            id = 0
                        if id <= -Iref:
                            id = -Iref

                    ied = -id
                    ei = ied - ie
                    propi = kpi * ei

                    if USE_IMPROVED_PID:
                        u_unsat = propi + intei
                        if u_unsat > Vref:
                            u = Vref
                        elif u_unsat <= 0:
                            u = 0.0
                        else:
                            u = u_unsat

                        intei = intei + (kii * Ts * ei) + (KAW_U * (u - u_unsat) * Ts)
                        if intei > Vref:
                            intei = Vref
                        if intei < -Vref:
                            intei = -Vref
                    else:
                        if -Vref < intei < Vref:
                            intei += kii * Ts * ei
                        else:
                            intei = 0.95 * Vref if intei >= Vref else -0.95 * Vref
                        u = propi + intei
                        if u > Vref:
                            u = Vref
                        if u <= 0:
                            u = 0
                    pwmf = escs * u
                    pwm = int(abs(pwmf))
                    ser.write(bytes([pwm]))

                    data_queue.put((t, y, u))
                    fp.write(f"{t:.4f}\t{yd:.4f}\t{y:.4f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}\n")
                    fp.flush()
                    flagcom = 0
                    t += Ts

                # (Control solo se actualiza en flagcom==5, igual que en C++)

    except Exception as e:
        print(f"Error en el hilo de control: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
        print("Hilo de control finalizado.")

# --- Hilo Principal (Graficación) ---
if __name__ == "__main__":
    control = threading.Thread(target=control_thread, daemon=True)
    control.start()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('Tiempo (s)')
    ax1.set_ylabel('Posición (y)', color='b')
    ax2.set_ylabel('Control (u)', color='r')
    line1, = ax1.plot([], [], 'b-', label='Posición (y)')
    line2, = ax2.plot([], [], 'r-', label='Control (u)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    time_points, y_values, u_values = [], [], []

    def update_plot(frame):
        while not data_queue.empty():
            t, y, u = data_queue.get_nowait()
            time_points.append(t)
            y_values.append(y)
            u_values.append(u)

        if time_points:
            line1.set_data(time_points, y_values)
            line2.set_data(time_points, u_values)
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()
        return line1, line2

    ani = FuncAnimation(fig, update_plot, blit=True, interval=100, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Cerrando programa...")
    finally:
        exit_event.set()
        control.join(timeout=2)
        print("Programa finalizado.")


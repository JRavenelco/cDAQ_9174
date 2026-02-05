import asyncio
import serial
import msvcrt

class DataAcquisition:
    def __init__(self):
        self.exit_requested = False
        self.flagcom = 0

        # Parámetros de control
        self.Ts = 0.01
        self.kp = 100
        self.ki = 50
        self.kd = 1.5
        self.kpi = 12.0
        self.kii = 3000.0
        self.Vref = 9.86
        self.Iref = 0.827
        self.Rs = 2.2

        # Variables de estado y control
        self.pv = 0
        self.i = 0
        self.y = 0
        self.y_1 = 0
        self.ef = 0
        self.ef_1 = 0
        self.u = 0
        self.t = 0
        self.esc = 0.05 / 1023.0
        self.esci = 5.0 / (self.Rs * 1023.0)
        self.escs = 254.0 / self.Vref
        self.iTs = 1 / self.Ts
        self.pwmf = 0
        self.yd = 0.005
        self.proporcional = 0
        self.derivativa = 0
        self.ie = 0
        self.ied = 0
        self.id = 0
        self.ei = 0
        self.propi = 0
        self.intei = 0
        self.integral = 0

    async def acquire_data(self):
        ser = serial.Serial('COM1', 115200, timeout=None)

        while not self.exit_requested:
            received = ser.read(1)

            if self.flagcom != 0:
                self.flagcom += 1

            if received == b'\xAA' and self.flagcom == 0:
                self.pv = 0
                self.i = 0
                self.flagcom = 1

            if self.flagcom == 2:
                self.pv = (self.pv << 8) + ord(received)

            if self.flagcom == 3:
                self.pv = (self.pv << 8) + ord(received)

            if self.flagcom == 4:
                self.i = (self.i << 8) + ord(received)

            if self.flagcom == 5:
                if self.t > 10:
                    self.yd = 0.0045
                if self.t > 15:
                    self.yd = 0.005

                self.ef_1 = self.ef
                self.y_1 = self.y

                self.i = (self.i << 8) + ord(received)
                self.y = self.esc * self.pv
                self.ie = self.esci * self.i
                self.ef = self.yd - self.y
                self.proporcional = self.kp * self.ef
                self.derivativa = self.kd * (self.ef - self.ef_1) * self.iTs

                if -self.Iref < self.integral < self.Iref:
                    self.integral = self.integral + self.ki * self.Ts * self.ef
                else:
                    if self.integral >= self.Iref:
                        self.integral = 0.95 * self.Iref
                    if self.integral <= -self.Iref:
                        self.integral = -0.95 * self.Iref

                self.id = self.proporcional + self.integral + self.derivativa

                if self.id > 0:
                    self.id = 0
                if self.id <= -self.Iref:
                    self.id = -self.Iref

                self.ied = -self.id
                self.ei = self.ied - self.ie
                self.propi = self.kpi * self.ei

                if -self.Vref < self.intei < self.Vref:
                    self.intei = self.intei + self.kii * self.Ts * self.ei
                else:
                    if self.intei >= self.Vref:
                        self.intei = 0.95 * self.Vref
                    if self.intei <= -self.Vref:
                        self.intei = -0.95 * self.Vref

                self.u = self.propi + self.intei

                if self.u > self.Vref:
                    self.u = self.Vref
                if self.u <= 0:
                    self.u = 0

                self.pwmf = self.u
                self.pwmf = self.escs * self.pwmf
                self.pwm = int(abs(self.pwmf))

                # Realiza las operaciones necesarias con los datos adquiridos
                # Luego, puedes almacenarlos en una lista o realizar otras acciones

            self.flagcom = 0
            self.t = self.t + self.Ts

    async def main(self):
        # Crea una tarea para la adquisición de datos
        acquisition_task = asyncio.create_task(self.acquire_data())

        while not self.exit_requested:
            # Tu código principal (control) continúa aquí...

            # Comprueba si una tecla está disponible para salir
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'q':
                    self.exit_requested = True  # Establece la bandera de salida

            await asyncio.sleep(self.Ts)

        # Espera a que termine la tarea de adquisición de datos
        await acquisition_task

if __name__ == '__main__':
    data_acquisition = DataAcquisition()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(data_acquisition.main())

import argparse
import sys
import time
from ctypes import byref, c_double, c_int, c_ubyte, cdll

import numpy as np

FORCE_I2C_SCL = 0
FORCE_I2C_SDA = 1
ADS1219_ADDR8 = 0x40 << 1
ADS1219_VREF = 2.048
ADS1219_GAIN = 4
ADS1219_SPS = 1000
ADS1219_CONFIG = (0b00 << 5) | ((1 if ADS1219_GAIN == 4 else 0) << 4) | (0b11 << 2) | (1 << 1) | 0


def load_dwf():
    try:
        return cdll.dwf if sys.platform.startswith("win") else cdll.LoadLibrary("libdwf.so")
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar WaveForms SDK/dwf: {exc}") from exc


class ADS1219Monitor:
    def __init__(self):
        self.dwf = load_dwf()
        self.hdwf = c_int()

    def open(self):
        self.dwf.FDwfDeviceOpen(c_int(-1), byref(self.hdwf))
        if self.hdwf.value == 0:
            raise RuntimeError("No se pudo abrir el Analog Discovery. Revisa que WaveForms no lo tenga ocupado.")

        self.dwf.FDwfDigitalI2cReset(self.hdwf)
        self.dwf.FDwfDigitalI2cRateSet(self.hdwf, c_double(400e3))
        self.dwf.FDwfDigitalI2cSclSet(self.hdwf, c_int(FORCE_I2C_SCL))
        self.dwf.FDwfDigitalI2cSdaSet(self.hdwf, c_int(FORCE_I2C_SDA))

        free = c_int()
        ok_clear = self.dwf.FDwfDigitalI2cClear(self.hdwf, byref(free))
        if not ok_clear:
            raise RuntimeError("No se pudo inicializar/limpiar el bus I2C.")
        if not free.value:
            raise RuntimeError("Bus I2C ocupado. Revisa 3.3V, GND común y pull-ups SDA/SCL.")

        self.i2c_write(0x06)
        time.sleep(0.001)
        self.i2c_write(0x40, ADS1219_CONFIG)

        cfg = self.i2c_write_read(0x20, 1)[0]
        if cfg != ADS1219_CONFIG:
            raise RuntimeError(f"Config ADS1219 leída 0x{cfg:02X}, esperada 0x{ADS1219_CONFIG:02X}.")

        self.i2c_write(0x08)
        t0 = time.time()
        while not self.data_ready():
            if time.time() - t0 > 2.0:
                raise RuntimeError("ADS1219 configurado, pero no entrega DRDY después de 2 s.")
            time.sleep(0.0002)

    def close(self):
        if self.hdwf.value != 0:
            try:
                self.i2c_write(0x02)
            except Exception:
                pass
            try:
                self.dwf.FDwfDeviceClose(self.hdwf)
            except Exception:
                pass
            self.hdwf = c_int()

    def i2c_write(self, *data):
        tx = (c_ubyte * len(data))(*data)
        nak = c_int()
        ok = self.dwf.FDwfDigitalI2cWrite(self.hdwf, c_int(ADS1219_ADDR8), tx, c_int(len(data)), byref(nak))
        if not ok or nak.value != 0:
            raise RuntimeError(f"ADS1219 no responde en 0x{ADS1219_ADDR8 >> 1:02X}; NAK={nak.value}.")

    def i2c_write_read(self, cmd, n):
        tx = (c_ubyte * 1)(cmd)
        rx = (c_ubyte * n)()
        nak = c_int()
        ok = self.dwf.FDwfDigitalI2cWriteRead(self.hdwf, c_int(ADS1219_ADDR8), tx, c_int(1), rx, c_int(n), byref(nak))
        if not ok or nak.value != 0:
            raise RuntimeError(f"ADS1219 no responde al leer 0x{cmd:02X}; NAK={nak.value}.")
        return rx

    def data_ready(self):
        return (self.i2c_write_read(0x24, 1)[0] & 0x80) != 0

    def read_volts(self):
        rx = self.i2c_write_read(0x10, 3)
        raw = (rx[0] << 16) | (rx[1] << 8) | rx[2]
        if raw & 0x800000:
            raw -= 1 << 24
        volts = raw / (1 << 23) * (ADS1219_VREF / ADS1219_GAIN)
        return raw, volts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--fast-rdata", action="store_true")
    parser.add_argument("--target-fs", type=float, default=0, help="Frecuencia de lectura objetivo en Hz (0 = sin límite)")
    args = parser.parse_args()

    monitor = ADS1219Monitor()
    try:
        monitor.open()
        print("ADS1219 en vivo")
        print(f"I2C addr=0x{ADS1219_ADDR8 >> 1:02X}, config=0x{ADS1219_CONFIG:02X}, Vref={ADS1219_VREF} V, gain={ADS1219_GAIN}, SPS={ADS1219_SPS}")
        print(f"Modo lectura: {'solo RDATA, sin DRDY por muestra' if args.fast_rdata else 'DRDY + RDATA'}")
        print("Ctrl+C para salir")

        start_time = time.perf_counter()
        block_start = start_time
        last_print = start_time
        next_deadline = start_time
        total_samples = 0
        raw_values = []
        volt_values = []
        target_interval = (1.0 / args.target_fs) if args.target_fs > 0 else 0.0

        while True:
            if args.fast_rdata or monitor.data_ready():
                # Pacing: deadline absoluto, espaciado exacto, sin deriva.
                # time.sleep en Windows redondea hacia arriba en sub-ms, asi que
                # dormimos el grueso y afinamos el ultimo tramo con busy-wait.
                if target_interval > 0:
                    now = time.perf_counter()
                    wait = next_deadline - now
                    if wait > 0:
                        if wait > 0.0015:
                            time.sleep(wait - 0.0012)
                        while time.perf_counter() < next_deadline:
                            pass
                        next_deadline += target_interval
                    else:
                        # Vamos retrasados: re-anclar para no acumular rafaga
                        next_deadline = now + target_interval

                raw, volts = monitor.read_volts()
                raw_values.append(raw)
                volt_values.append(volts)

                now = time.perf_counter()
                if len(volt_values) >= args.samples or now - last_print >= args.interval:
                    arr = np.asarray(volt_values, dtype=float)
                    raw_arr = np.asarray(raw_values, dtype=float)
                    block_dt = now - block_start
                    total_samples += len(arr)
                    total_dt = now - start_time
                    fs_eff = len(arr) / block_dt if block_dt > 0 else 0.0
                    fs_avg = total_samples / total_dt if total_dt > 0 else 0.0
                    print(
                        f"n={len(arr):4d} | "
                        f"dt={block_dt:.3f} s | "
                        f"fs_eff={fs_eff:7.1f} Hz | "
                        f"fs_prom={fs_avg:7.1f} Hz | "
                        f"actual={arr[-1]:+.9f} V | "
                        f"media={np.mean(arr):+.9f} V | "
                        f"rms={np.sqrt(np.mean(arr**2)):.9f} V | "
                        f"min={np.min(arr):+.9f} V | "
                        f"max={np.max(arr):+.9f} V | "
                        f"raw={int(raw_arr[-1])}"
                    )
                    raw_values.clear()
                    volt_values.clear()
                    block_start = now
                    last_print = now
            else:
                time.sleep(0.0002)
    except KeyboardInterrupt:
        print("\nDetenido por usuario.")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

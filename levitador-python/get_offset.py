import asyncio
import serial_asyncio
import time

async def main():
    print("Este script leerá los valores crudos del sensor de posición (pv).")
    print("Asegúrate de que la bola esté en la posición más baja.")
    print("Presiona Ctrl+C para detener después de unos 5-10 segundos.\n")

    try:
        reader, _ = await serial_asyncio.open_serial_connection(url='COM1', baudrate=115200)
        print("Puerto COM1 abierto. Esperando datos...")

        # Esperar el primer byte de sincronización
        while True:
            sync_byte = await reader.read(1)
            if sync_byte == b'\xAA':
                break
        
        print("Sincronización encontrada. Leyendo valores de 'pv':\n")
        start_time = time.time()
        
        while time.time() - start_time < 10: # Leer por 10 segundos
            # Leer paquete de 4 bytes (pvH, pvL, iH, iL)
            packet = await reader.read(4)
            if len(packet) < 4:
                continue

            # Reconstruir pv
            pv = (packet[0] << 8) + packet[1]
            print(f"Valor crudo de pv: {pv}")

            # Esperar el siguiente byte de sincronización
            next_sync = await reader.read(1)
            if next_sync != b'\xAA':
                print("Pérdida de sincronización...")
                while True:
                    sync_byte = await reader.read(1)
                    if sync_byte == b'\xAA':
                        break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\nScript finalizado. Anota el valor promedio de 'pv'.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCerrando...")

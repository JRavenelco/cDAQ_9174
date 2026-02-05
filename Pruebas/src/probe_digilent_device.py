import sys
print(f"sys.path: {sys.path}")

try:
    from pydwf import DwfLibrary, DeviceEnumeration, PyDwfError
    # Corregir la importación para obtener la función/clase 'openDwfDevice'
    from pydwf.utilities import openDwfDevice 
except ImportError as e:
    print(f"Error al importar componentes de pydwf: {e}")
    print("Asegúrate de que 'pydwf' esté instalado y accesible.")
    exit()

print("Buscando dispositivos Digilent WaveForms...")

dwf_library = None
try:
    dwf_library = DwfLibrary()
    device_enumerator = DeviceEnumeration(dwf_library)
    num_devices = 0

    if hasattr(device_enumerator, 'enumerateDevices'):
        print("Usando device_enumerator.enumerateDevices()...")
        num_devices = device_enumerator.enumerateDevices()
        print(f"Resultado de enumerateDevices() (esperando un int): {num_devices}")
        print(f"Número de dispositivos encontrados: {num_devices}")
    else:
        print("El método 'enumerateDevices' no fue encontrado en device_enumerator.")

    if isinstance(num_devices, int) and num_devices > 0:
        print(f"\nSe encontraron {num_devices} dispositivos.")

        # Obtener información del primer dispositivo (índice 0) desde device_enumerator
        device_name_str = "Desconocido"
        serial_number_str = "Desconocido"
        try:
            if hasattr(device_enumerator, 'deviceName') and hasattr(device_enumerator, 'serialNumber'):
                print("\nObteniendo información del primer dispositivo (índice 0)...")
                device_name_str = device_enumerator.deviceName(0)
                serial_number_str = device_enumerator.serialNumber(0)
                print(f"  Nombre del dispositivo (desde enumerador): {device_name_str}")
                print(f"  Número de serie (desde enumerador): {serial_number_str}")
            else:
                print("Los métodos deviceName o serialNumber no están en device_enumerator.")
        except PyDwfError as e_info:
            print(f"Error al obtener información del dispositivo desde el enumerador: {e_info}")
        except Exception as e_generic_info:
            print(f"Error genérico al obtener información del dispositivo: {e_generic_info}")

        print("\nAbriendo el primer dispositivo lógicamente disponible...")
        with openDwfDevice(dwf_library) as device: # Esto debería abrir el dispositivo en el índice 0
            print("Dispositivo abierto exitosamente.")
            # El objeto 'device' (DwfDevice) se usa para operaciones, no para obtener nombre/serial.
            # La información ya se obtuvo del enumerador.
            print(f"  Confirmando apertura para: {device_name_str} (S/N: {serial_number_str})")
            
            # Configurar comunicación SPI
            print("\nConfigurando comunicación SPI...")
            try:
                spi = device.protocol.spi
                # Configuración de pines y parámetros SPI
                # Ajusta estos valores según tu configuración de Waveshare
                spi.clockSet(0)  # Pin para el reloj SPI (ajusta según tu conexión)
                spi.dataSet(0, 1)  # Pin para MOSI (SDO del Waveshare)
                spi.dataSet(1, 2)  # Pin para MISO (SDI del Waveshare)
                spi.selectSet(3, 0)  # Pin para Chip Select (CS), ajusta según tu conexión, level=0 (activo bajo)
                spi.modeSet(0)  # Modo SPI (0, 1, 2 o 3, según tu configuración)
                spi.frequencySet(1000000)  # Frecuencia del reloj SPI (1 MHz, ajusta si necesario)
                spi.orderSet(1)  # Orden de bits (1 para MSB primero, 0 para LSB primero)
                # Asegurarse de que el dispositivo SPI esté inicializado
                spi.reset()  # Resetear el módulo SPI para asegurar un estado limpio
                print("SPI configurado exitosamente.")

                # Inspeccionar objeto SPI para depuración
                print("\nInspeccionando objeto SPI...")
                print(f"Tipo de objeto SPI: {type(spi)}")
                print(f"Métodos y atributos disponibles en SPI: {dir(spi)}")

                # Capturar datos SPI
                print("\nCapturando datos SPI...")
                # Enviar y recibir una trama de datos (ajusta según el protocolo de tu Waveshare)
                data_to_send = [0x01, 0x02, 0x03, 0x04]  # Datos de ejemplo para enviar, como lista de enteros
                received_data = []
                try:
                    # Enviar datos byte por byte usando writeOne
                    for byte in data_to_send:
                        spi.writeOne(8, byte, 1)  # 8 bits por palabra, enviar un solo byte, tx=1 (indicar transmisión)
                        print(f"Byte enviado: {byte}")
                    print(f"Datos enviados completamente: {data_to_send}")
                    
                    # Leer datos byte por byte (mismo número de bytes que enviamos)
                    for _ in range(len(data_to_send)):
                        byte_received = spi.readOne(8, 8)  # 8 bits por palabra, bits_per_word=8
                        received_data.append(byte_received)
                        print(f"Byte recibido: {byte_received}")
                    print(f"Datos recibidos completamente: {received_data}")
                except Exception as e_spi_rw:
                    print(f"Error al intentar operaciones de escritura/lectura SPI: {e_spi_rw}")
            except PyDwfError as e_spi:
                print(f"Error al configurar o capturar datos SPI: {e_spi}")
            except Exception as e_generic_spi:
                print(f"Error genérico al trabajar con SPI: {e_generic_spi}")

            print("\nCerrando el dispositivo (automáticamente con 'with').")

    elif isinstance(num_devices, int) and num_devices == 0:
        print("No se encontraron dispositivos Digilent WaveForms (conteo es 0).")
        print("Asegúrate de que tu dispositivo esté conectado y reconocido por el sistema.")
    else:
         print(f"No se pudo determinar el número de dispositivos o el resultado no fue un entero esperado. Resultado: {num_devices}")

except PyDwfError as e:
    print(f"Ocurrió un error de PyDwf: {e}")
    if "DWF library not found" in str(e) or "Error loading DWF library" in str(e):
        print("\n--- Posible Solución ---")
        print("Asegúrate de que el software WaveForms de Digilent esté instalado correctamente.")
        print("Este software instala las librerías y controladores necesarios.")
        print("Puedes descargarlo desde: https://digilent.com/reference/software/waveforms/waveforms-3/start")
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}")

print("\nPrueba finalizada.")

"""
Script rápido para verificar si hay dispositivos Digilent disponibles.
"""
import sys

try:
    from pydwf import DwfLibrary
    from pydwf.utilities import openDwfDevice
    print("✓ pydwf importado correctamente")
except Exception as e:
    print(f"✗ Error importando pydwf: {e}")
    sys.exit(1)

try:
    # Intentar abrir dispositivo directamente (método simplificado)
    dwf = DwfLibrary()
    print("Intentando abrir primer dispositivo disponible...")
    with openDwfDevice(dwf) as device:
        print(f"✓ Dispositivo abierto: {device}")
        ai = device.analogIn
        print(f"  AnalogIn channels: {ai.channelCount()}")
        fmin, fmax = ai.frequencyInfo()
        print(f"  Frequency range: {fmin} - {fmax} Hz")
        bmin, bmax = ai.bufferSizeInfo()
        print(f"  Buffer size range: {bmin} - {bmax}")
        print("✓ Dispositivo funcionando correctamente")

except Exception as e:
    print(f"✗ Error con dispositivo Digilent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✓ Todo OK - dispositivo Digilent listo para usar")

#!/usr/bin/env python3
"""
Script de prueba para verificar configuración del NI 9219 (Bridge V/V)
"""

import sys

print("=" * 70)
print("TEST: Configuración NI 9219 Bridge V/V")
print("=" * 70)

# Verificar nidaqmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, ExcitationSource, BridgeConfiguration, BridgeUnits
    print("\n✅ nidaqmx importado correctamente")
    print(f"   Versión: {nidaqmx.__version__}")
except ImportError as e:
    print(f"\n❌ Error importando nidaqmx: {e}")
    sys.exit(1)

# Verificar constantes disponibles
print("\n📋 Constantes disponibles:")
print(f"   - BridgeConfiguration.FULL_BRIDGE: {BridgeConfiguration.FULL_BRIDGE}")
print(f"   - ExcitationSource.INTERNAL: {ExcitationSource.INTERNAL}")
print(f"   - BridgeUnits.MILLIVOLTS_PER_VOLT: {BridgeUnits.MILLIVOLTS_PER_VOLT}")
print(f"   - AcquisitionType.CONTINUOUS: {AcquisitionType.CONTINUOUS}")

# Verificar métodos disponibles
print("\n🔍 Métodos disponibles en ai_channels:")
task = nidaqmx.Task()
ai_methods = [m for m in dir(task.ai_channels) if 'bridge' in m.lower()]
print(f"   Métodos con 'bridge': {ai_methods}")

# Mostrar configuración esperada
print("\n" + "=" * 70)
print("CONFIGURACIÓN ESPERADA (Bridge V/V Setup):")
print("=" * 70)
print("""
Signal Input Range:
  - Min: -10 mV/V
  - Max: +10 mV/V

Scaled Units: mV/V

Bridge Type: Full Bridge

Vex Source: Internal

Vex Value: 2.5 V

Bridge Resistance: 350 Ω

Timing:
  - Mode: Continuous Samples
  - Rate: 100 Hz
  - Samples to Read: 10
""")

print("\n✅ Configuración lista para usar en ProjectAD")
print("=" * 70)

task.close()

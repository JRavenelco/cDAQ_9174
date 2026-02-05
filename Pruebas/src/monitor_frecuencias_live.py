"""
Monitor de Frecuencias EN VIVO
Muestra FFT cada segundo mientras el DAQ principal está corriendo
"""
import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
from scipy.fft import rfft, rfftfreq
import time

# Configuración
DEVICE = "cDAQ1Mod2"
CHANNELS = ["ai0", "ai1"]  # Prensa, Pieza
FORCE_DEVICE = "cDAQ1Mod1"
FORCE_CHANNEL = "ai0"
FS = 2500
BLOCK = 2500  # 1 segundo de datos

print("=" * 60)
print("🎯 MONITOR DE FRECUENCIAS EN VIVO")
print("=" * 60)
print(f"  ai0 = Prensa | ai1 = Pieza | Fuerza")
print("  Ctrl+C para detener")
print("=" * 60)

try:
    while True:
        # Leer aceleración (1 segundo)
        with nidaqmx.Task() as task_a:
            for ch in CHANNELS:
                task_a.ai_channels.add_ai_accel_chan(
                    f"{DEVICE}/{ch}",
                    sensitivity=98.3,
                    min_val=-50, max_val=50,
                    current_excit_val=0.004
                )
            task_a.timing.cfg_samp_clk_timing(FS, sample_mode=AcquisitionType.FINITE, samps_per_chan=BLOCK)
            data_a = np.array(task_a.read(number_of_samples_per_channel=BLOCK))
        
        # Leer fuerza
        with nidaqmx.Task() as task_f:
            task_f.ai_channels.add_ai_voltage_chan(f"{FORCE_DEVICE}/{FORCE_CHANNEL}", min_val=-5, max_val=5)
            task_f.timing.cfg_samp_clk_timing(FS, sample_mode=AcquisitionType.FINITE, samps_per_chan=BLOCK)
            data_f = np.array(task_f.read(number_of_samples_per_channel=BLOCK))
        
        # FFT
        freq = rfftfreq(BLOCK, 1/FS)
        
        # Prensa (ai0)
        fft_prensa = np.abs(rfft(data_a[0] - np.mean(data_a[0])))
        idx_prensa = np.argmax(fft_prensa[freq > 10])
        f_prensa = freq[freq > 10][idx_prensa]
        
        # Pieza (ai1)
        fft_pieza = np.abs(rfft(data_a[1] - np.mean(data_a[1])))
        idx_pieza = np.argmax(fft_pieza[freq > 10])
        f_pieza = freq[freq > 10][idx_pieza]
        
        # Fuerza
        fft_fuerza = np.abs(rfft(data_f - np.mean(data_f)))
        idx_fuerza = np.argmax(fft_fuerza[freq > 5])
        f_fuerza = freq[freq > 5][idx_fuerza]
        
        # RMS
        rms_prensa = np.std(data_a[0])
        rms_pieza = np.std(data_a[1])
        rms_fuerza = np.std(data_f)
        
        # Mostrar
        print(f"\r🔊 Prensa: {f_prensa:6.1f} Hz ({rms_prensa:.3f}g) | "
              f"Pieza: {f_pieza:6.1f} Hz ({rms_pieza:.3f}g) | "
              f"Fuerza: {f_fuerza:6.1f} Hz ({rms_fuerza:.4f}V)", end="", flush=True)
        
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\n✅ Monitor detenido")
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Nota: Si el DAQ principal está usando los canales, ciérralo primero.")

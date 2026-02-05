"""
Prueba standalone de adquisición con Digilent WaveForms (AnalogIn) usando DigilentAnalogInThread.

- Lee bloques de muestras durante unos segundos y muestra métricas de calidad de datos.
- Requiere pydwf==1.1.19 y un dispositivo Digilent compatible (p.ej. Analog Discovery).

Uso:
  python test_digilent_acq.py --duration 6 --channels 0,1 --sample_rate 20000 --range 5.0
"""
from __future__ import annotations

import sys
import time
import math
import queue
import argparse
import numpy as np


def parse_channels(ch_str: str) -> list[int]:
    return [int(x.strip()) for x in ch_str.split(',') if x.strip() != '']


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba de adquisición Digilent AnalogIn")
    parser.add_argument("--duration", "-d", type=float, default=6.0, help="Duración de la prueba en segundos")
    parser.add_argument("--channels", "-c", type=str, default="0,1", help="Lista de canales, ej: 0,1")
    parser.add_argument("--sample_rate", "--sr", type=float, default=20000.0, help="Frecuencia de muestreo (Hz)")
    parser.add_argument("--range", "-r", dest="channel_range", type=float, default=5.0, help="Rango de canal (V)")
    parser.add_argument("--verbose", action="store_true", help="Mensajes detallados")
    parser.add_argument("--print_every", type=int, default=0, help="Imprime un bloque cada N (0=solo resumen final)")
    parser.add_argument("--queue_size", type=int, default=8, help="Tamaño de la cola hilo->main")
    parser.add_argument("--buffer_size", type=int, default=0, help="Tamaño del buffer del dispositivo (0 = usar máximo disponible)")
    args = parser.parse_args()

    try:
        from digilent_analogin_thread import DigilentAnalogInThread
    except Exception as e:
        print("[ERROR] No se pudo importar DigilentAnalogInThread. Asegúrate de tener pydwf instalado (pip install pydwf) y el archivo digilent_analogin_thread.py disponible.", file=sys.stderr)
        print(f"Detalle: {e}", file=sys.stderr)
        return 2

    ch_list = parse_channels(args.channels)
    if not ch_list:
        print("[ERROR] Debes especificar al menos un canal, p.ej. --channels 0,1", file=sys.stderr)
        return 2

    q: queue.Queue = queue.Queue(maxsize=int(args.queue_size))
    th = DigilentAnalogInThread(
        data_queue=q,
        channels=ch_list,
        sample_rate=float(args.sample_rate),
        channel_range=float(args.channel_range),
        buffer_size=(1000000 if int(args.buffer_size) == 0 else int(args.buffer_size)),
        poll_interval_s=0.002,
        verbose=args.verbose,
    )

    print(f"Iniciando adquisición Digilent WF: canales={ch_list}, SR={args.sample_rate} Hz, rango=±{args.channel_range} V, duración={args.duration}s")
    t0 = time.time()
    th.start()

    blk_count = 0
    n_ch = len(ch_list)
    # Acumuladores para métricas agregadas
    ch_min = np.full(n_ch, np.inf, dtype=np.float64)
    ch_max = np.full(n_ch, -np.inf, dtype=np.float64)
    ch_sum2 = np.zeros(n_ch, dtype=np.float64)
    ch_count = np.zeros(n_ch, dtype=np.int64)
    ch_nans = np.zeros(n_ch, dtype=np.int64)
    try:
        while time.time() - t0 < args.duration:
            try:
                blk = q.get(timeout=0.75)
            except queue.Empty:
                # No llegó bloque en timeout; continuar esperando
                continue

            blk_count += 1
            if not isinstance(blk, np.ndarray) or blk.ndim != 2:
                print(f"[WARN] Bloque inesperado: tipo={type(blk)}, ndim={getattr(blk, 'ndim', None)}")
                continue

            n_ch, n_samp = blk.shape
            # Acumular métricas y NaNs
            for ch in range(n_ch):
                arr = blk[ch, :].astype(np.float64, copy=False)
                nan_mask = np.isnan(arr)
                ch_nans[ch] += int(nan_mask.sum())
                arr_clean = arr[~nan_mask]
                if arr_clean.size:
                    # min/max agregados
                    vmin = float(arr_clean.min())
                    vmax = float(arr_clean.max())
                    if vmin < ch_min[ch]:
                        ch_min[ch] = vmin
                    if vmax > ch_max[ch]:
                        ch_max[ch] = vmax
                    # RMS agregado por suma de cuadrados
                    ch_sum2[ch] += float(np.dot(arr_clean, arr_clean))
                    ch_count[ch] += int(arr_clean.size)

            # Salida periódica reducida
            if args.print_every > 0 and (blk_count % args.print_every == 0):
                print(f"Bloque #{blk_count}: shape=({n_ch},{n_samp})")
    except KeyboardInterrupt:
        print("Interrumpido por usuario.")
    finally:
        th.stop()
        th.join(timeout=2.0)
        if getattr(th, 'exception', None):
            print(f"[THREAD ERROR] {th.exception}", file=sys.stderr)

    # Resumen final
    print("Resumen de la prueba:")
    print(f"  Bloques procesados: {blk_count}")
    for ch in range(n_ch):
        if ch_count[ch] > 0:
            rms = math.sqrt(ch_sum2[ch] / float(ch_count[ch]))
            nans = int(ch_nans[ch])
            mn = float(ch_min[ch]) if math.isfinite(ch_min[ch]) else float('nan')
            mx = float(ch_max[ch]) if math.isfinite(ch_max[ch]) else float('nan')
            expected = int(args.sample_rate * args.duration)
            delivered = int(ch_count[ch])
            missing = max(0, expected - delivered)
            perc = (missing / expected * 100.0) if expected > 0 else 0.0
            print(f"  ch{ch}: min={mn:.4f} V, max={mx:.4f} V, RMS={rms:.4f} V, NaNs={nans}, muestras={delivered}, esperadas≈{expected}, falta≈{missing} ({perc:.1f}%)")
        else:
            print(f"  ch{ch}: sin muestras válidas (posibles NaNs)")
    print("Prueba finalizada.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador offline de sesiones guardadas en 'experimentos_mesa/'

Genera un reporte de:
- Hallazgos globales (conteo de sesiones, SR, canales guardados, rangos de duración, notas)
- Resumen por sesión (RMS, pico de frecuencia, etc. por Vib ai0 y Fuerza ai0)
- Interpretación técnica y recomendaciones básicas

Uso:
    python analisis_offline_experimentos.py [ruta_directorio_experimentos]

Si no se pasa ruta, toma la carpeta 'experimentos_mesa' al lado de este script.

Salida:
- Imprime el reporte en consola
- Guarda un archivo 'reporte_analisis.txt' y 'reporte_analisis.md' dentro de la carpeta analizada
"""
from __future__ import annotations
import os
import sys
import csv
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Dependencias opcionales
try:
    import numpy as np
except Exception:
    np = None  # Si falta, los gráficos y FFT no estarán disponibles
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

@dataclass
class SessionMeta:
    base: str
    t_start: str
    t_end: str
    duration_s: float
    sr_vib: float
    sr_force: float
    frecuencia: Optional[float] = None
    notas: str = ""
    canales_vib: List[str] = field(default_factory=list)
    canales_force: List[str] = field(default_factory=list)

@dataclass
class ChannelFeatures:
    RMS: Optional[float] = None
    MaxAbs: Optional[float] = None
    Mean: Optional[float] = None
    Std: Optional[float] = None
    PeakFreq: Optional[float] = None
    SpectralCentroid: Optional[float] = None
    WaveletEnergy: Optional[float] = None

@dataclass
class SessionSummary:
    meta: SessionMeta
    vib_ai0: ChannelFeatures = field(default_factory=ChannelFeatures)
    force_ai0: ChannelFeatures = field(default_factory=ChannelFeatures)


def _float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_log(log_path: str) -> List[SessionMeta]:
    if not os.path.exists(log_path):
        return []
    out: List[SessionMeta] = []
    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                base = row.get('Base Archivo', '')
                out.append(SessionMeta(
                    base=base,
                    t_start=row.get('Timestamp Inicio', ''),
                    t_end=row.get('Timestamp Fin', ''),
                    duration_s=_float(row.get('Duracion (s)', '') or '0') or 0.0,
                    sr_vib=_float(row.get('SR Vib (Hz)', '') or '0') or 0.0,
                    sr_force=_float(row.get('SR Fuerza (Hz)', '') or '0') or 0.0,
                    frecuencia=_float(row.get('Frecuencia (Hz)', '') or ''),
                    notas=row.get('Notas', '') or ''
                ))
            except Exception:
                continue
    return out


def read_session_csv(path: str) -> Tuple[SessionMeta, Dict[Tuple[str,str], ChannelFeatures]]:
    """Lee metadatos del encabezado '#' y el bloque de FEATURES al final.
    Retorna: (SessionMeta parcial, features por (Tipo, canal))
    """
    meta = SessionMeta(base=os.path.splitext(os.path.basename(path))[0], t_start='', t_end='',
                       duration_s=0.0, sr_vib=0.0, sr_force=0.0)
    features: Dict[Tuple[str,str], ChannelFeatures] = {}

    re_key_val = re.compile(r'^#\s*([^:]+):\s*(.*)$')

    with open(path, 'r', encoding='utf-8') as f:
        header_zone = True
        in_feat = False
        for raw in f:
            line = raw.rstrip('\n')
            if header_zone and line.startswith('#'):
                m = re_key_val.match(line)
                if m:
                    k = m.group(1).strip().lower()
                    v = m.group(2).strip()
                    if k.startswith('frecuencia'):
                        meta.frecuencia = _float(v.split()[0]) if v else None
                    elif k.startswith('notas'):
                        meta.notas = v
                    elif 'canales vibración guardados' in k:
                        meta.canales_vib = [s.strip() for s in v.split(',') if s.strip()]
                    elif 'canales fuerza guardados' in k:
                        meta.canales_force = [s.strip() for s in v.split(',') if s.strip()]
                    elif k.startswith('sr vib'):
                        meta.sr_vib = _float(v.split()[0]) or 0.0
                    elif k.startswith('sr fuerza'):
                        meta.sr_force = _float(v.split()[0]) or 0.0
                    elif k.startswith('duración') or k.startswith('duracion'):
                        meta.duration_s = _float(v.split()[0]) or meta.duration_s
                continue
            else:
                header_zone = False

            if line.startswith('# FEATURES SUMMARY'):
                in_feat = True
                continue
            if in_feat:
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    canal, tipo = parts[0], parts[1]
                    feats = ChannelFeatures(
                        RMS=_float(parts[2]),
                        MaxAbs=_float(parts[3]),
                        Mean=_float(parts[4]),
                        Std=_float(parts[5]),
                        PeakFreq=_float(parts[6]),
                        SpectralCentroid=_float(parts[7]),
                        WaveletEnergy=_float(parts[8])
                    )
                    features[(tipo.upper(), canal)] = feats
        # fin for
    return meta, features


def collect_summaries(dir_path: str) -> List[SessionSummary]:
    log_path = os.path.join(dir_path, 'experimentos_log.csv')
    metas = parse_log(log_path)
    # Ordenar por timestamp inicio si posible, si no, por nombre base
    metas_sorted = sorted(metas, key=lambda m: m.t_start or m.base)

    # Map meta by base to merge data
    meta_by_base: Dict[str, SessionMeta] = {m.base: m for m in metas_sorted if m.base}

    # Recorrer CSVs de sesión
    session_summaries: List[SessionSummary] = []
    for base, meta in meta_by_base.items():
        csv_path = os.path.join(dir_path, base + '.csv')
        if not os.path.exists(csv_path):
            continue
        file_meta, feats = read_session_csv(csv_path)
        # Merge file meta fields if present
        if not meta.frecuencia and file_meta.frecuencia:
            meta.frecuencia = file_meta.frecuencia
        if not meta.notas and file_meta.notas:
            meta.notas = file_meta.notas
        if file_meta.canales_vib:
            meta.canales_vib = file_meta.canales_vib
        if file_meta.canales_force:
            meta.canales_force = file_meta.canales_force
        if file_meta.duration_s:
            meta.duration_s = file_meta.duration_s
        if file_meta.sr_vib:
            meta.sr_vib = file_meta.sr_vib
        if file_meta.sr_force:
            meta.sr_force = file_meta.sr_force

        vib_ai0 = feats.get(('VIB', 'ai0'), ChannelFeatures())
        force_ai0 = feats.get(('FORCE', 'ai0'), ChannelFeatures())
        session_summaries.append(SessionSummary(meta=meta, vib_ai0=vib_ai0, force_ai0=force_ai0))
    return session_summaries


def build_report(dir_path: str) -> str:
    session_summaries = collect_summaries(dir_path)

    # Hallazgos globales
    n = len(session_summaries)
    bases = [s.meta.base for s in session_summaries]
    sr_v_set = sorted({int(s.meta.sr_vib) for s in session_summaries if s.meta.sr_vib})
    sr_f_set = sorted({int(s.meta.sr_force) for s in session_summaries if s.meta.sr_force})
    durs = [s.meta.duration_s for s in session_summaries if s.meta.duration_s]
    dmin, dmax = (min(durs), max(durs)) if durs else (0.0, 0.0)

    # Canales guardados predominantes
    vib_sets = [tuple(s.meta.canales_vib) for s in session_summaries if s.meta.canales_vib]
    force_sets = [tuple(s.meta.canales_force) for s in session_summaries if s.meta.canales_force]
    vib_common = ', '.join(sorted(set(v for t in vib_sets for v in t))) if vib_sets else ''
    force_common = ', '.join(sorted(set(v for t in force_sets for v in t))) if force_sets else ''

    last_notes = session_summaries[-1].meta.notas if session_summaries else ''

    def fmt(x: Optional[float], nd=2) -> str:
        return (f"{x:.{nd}f}" if x is not None else "-")

    # Construcción del reporte
    lines: List[str] = []
    lines.append("Hallazgos globales")
    if n:
        lines.append(f"[Sesiones] {n} archivos: {bases[0]}…{bases[-1]}.csv")
    else:
        lines.append("[Sesiones] 0 archivos")
    lines.append(f"[SR] Vib {sr_v_set[0] if sr_v_set else '-'} Hz; Fuerza {sr_f_set[0] if sr_f_set else '-'} Hz (experimentos_log.csv)")
    lines.append(f"[Canales guardados] Vib: {vib_common or '-'}; Fuerza: {force_common or '-'}")
    lines.append(f"[Duraciones] Entre ~{fmt(dmin)} s y ~{fmt(dmax)} s")
    if last_notes:
        lines.append(f"[Notas] Última sesión: {last_notes}")
    lines.append("")

    # Resumen por sesión
    lines.append("Resumen por sesión")
    for s in session_summaries:
        meta = s.meta
        lines.append(f"[{meta.base}]")
        freq_txt = f"{fmt(meta.frecuencia,0)} Hz" if meta.frecuencia is not None else "-"
        lines.append(f"Frecuencia: {freq_txt}. Duración: {fmt(meta.duration_s,2)} s.")
        # Vibración ai0
        vib = s.vib_ai0
        vib_rms = fmt(vib.RMS,3)
        vib_peak = fmt(vib.PeakFreq,2)
        lines.append(f"Vib ai0: RMS {vib_rms} g, Pico {vib_peak} Hz.")
        # Fuerza ai0
        fr = s.force_ai0
        fr_mean = fmt(fr.Mean,3)
        fr_rms = fmt(fr.RMS,3)
        fr_peak = fmt(fr.PeakFreq,2)
        extra = []
        if fr.Std is not None:
            extra.append(f"Std {fmt(fr.Std,4)} V")
        if fr.MaxAbs is not None:
            extra.append(f"MaxAbs {fmt(fr.MaxAbs,2)} V")
        extra_txt = (", " + ", ".join(extra)) if extra else ""
        lines.append(f"Fuerza ai0: Mean {fr_mean} V, RMS {fr_rms} V, Pico {fr_peak} Hz{extra_txt}.")
        # Detalles extra de vib si hay centroid/WE
        if s.vib_ai0.SpectralCentroid is not None or s.vib_ai0.WaveletEnergy is not None:
            cen = f"Centroid {fmt(s.vib_ai0.SpectralCentroid,1)} Hz" if s.vib_ai0.SpectralCentroid is not None else None
            we = f"WaveletEnergy {fmt(s.vib_ai0.WaveletEnergy,1)}" if s.vib_ai0.WaveletEnergy is not None else None
            detail = ", ".join([t for t in [cen, we] if t])
            if detail:
                lines.append(f"Vib ai0: {detail}.")
        lines.append("")

    # Interpretación técnica (básica automática)
    lines.append("Interpretación técnica")
    # Seguimiento de consigna en vibración
    within2 = 0
    total_v = 0
    for s in session_summaries:
        if s.meta.frecuencia and s.vib_ai0.PeakFreq is not None:
            total_v += 1
            if abs(s.vib_ai0.PeakFreq - s.meta.frecuencia) <= 2.0:
                within2 += 1
    if total_v:
        lines.append(f"[Vibración] Picos cercanos a la consigna (±2 Hz) en {within2}/{total_v} sesiones.")
    # DC en fuerza
    means = [abs(s.force_ai0.Mean) for s in session_summaries if s.force_ai0.Mean is not None]
    if means:
        med_mean = sorted(means)[len(means)//2]
        lines.append(f"[Fuerza] Componente DC promedio |mean| ≈ {fmt(med_mean,3)} V. Considere HPF o remoción de offset.")
    # Transitorios
    maxabs = [s.force_ai0.MaxAbs for s in session_summaries if s.force_ai0.MaxAbs is not None]
    if maxabs and max(maxabs) > 1.0:
        lines.append("[Fuerza] Se detectan picos significativos (MaxAbs > 1 V) en alguna sesión; posibles transitorios o saturación.")

    # Recomendaciones
    lines.append("Recomendaciones")
    lines.append("[Fuerza: offset] Aplicar HPF (1–2 Hz) o remover offset para resaltar componente dinámica.")
    lines.append("[Vibración: armónicos] Agregar detección de 2×/3× y marcadores en FFT/espectrograma si se requiere.")
    if session_summaries:
        last = session_summaries[-1]
        if last.meta.notas:
            lines.append("[Sesión final] Analizar trayectoria de frecuencia con espectrograma si hubo barrido.")
    lines.append("")

    report = "\n".join(lines)
    return report


def save_report(dir_path: str, report: str) -> None:
    txt_path = os.path.join(dir_path, 'reporte_analisis.txt')
    md_path = os.path.join(dir_path, 'reporte_analisis.md')
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report)
        with open(md_path, 'w', encoding='utf-8') as f:
            # Simple conversión a markdown (añadir encabezados)
            md = []
            for line in report.splitlines():
                if line.strip() in ("Hallazgos globales", "Resumen por sesión", "Interpretación técnica", "Recomendaciones"):
                    md.append(f"## {line.strip()}")
                else:
                    md.append(line)
            f.write("\n".join(md))
        print(f"✅ Reporte guardado en:\n  - {txt_path}\n  - {md_path}")
    except Exception as e:
        print(f"⚠️ No se pudo guardar el reporte: {e}")


def parse_numeric_data(csv_path: str) -> Dict[str, List[float]]:
    """Extrae columnas numéricas del bloque principal (tiempo, vib ai0, fuerza ai0).
    Devuelve un dict con keys: 't', 'vib_ai0', 'force_ai0'. Si alguna no existe, lista vacía.
    """
    data: Dict[str, List[float]] = {'t': [], 'vib_ai0': [], 'force_ai0': []}
    if not os.path.exists(csv_path):
        return data
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    # Encontrar header de datos (no comentarios) y rango de datos
    header_idx = None
    for i, line in enumerate(lines):
        if not line.startswith('#') and (',' in line) and 'Tiempo' in line:
            header_idx = i
            break
    if header_idx is None:
        return data
    headers = [h.strip() for h in lines[header_idx].split(',')]
    # Mapeo de columnas
    col_time = next((idx for idx, h in enumerate(headers) if h.lower().startswith('tiempo')), None)
    col_vib = next((idx for idx, h in enumerate(headers) if 'acel_ai0' in h.lower()), None)
    col_force = next((idx for idx, h in enumerate(headers) if 'volt_ai0' in h.lower()), None)
    # Leer hasta línea vacía o nueva sección
    for j in range(header_idx + 1, len(lines)):
        line = lines[j].strip()
        if not line or line.startswith('#'):
            break
        parts = [p.strip() for p in line.split(',')]
        try:
            if col_time is not None and col_time < len(parts):
                data['t'].append(float(parts[col_time]))
            if col_vib is not None and col_vib < len(parts):
                data['vib_ai0'].append(float(parts[col_vib]))
            if col_force is not None and col_force < len(parts):
                data['force_ai0'].append(float(parts[col_force]))
        except Exception:
            continue
    return data


def compute_fft_db(x: List[float], fs: float) -> Tuple[List[float], List[float]]:
    if np is None or len(x) < 8:
        return [], []
    x = np.asarray(x, dtype=float)
    nfft = int(2 ** np.floor(np.log2(min(4096, max(128, len(x))))))
    win = np.hanning(nfft)
    if len(x) < nfft:
        xz = np.zeros(nfft)
        xz[:len(x)] = x
        x = xz
    else:
        x = x[-nfft:]
    x = x - np.mean(x)
    X = np.fft.rfft(x * win)
    mag = np.abs(X)
    mag = mag + 1e-12
    mag_db = 20.0 * np.log10(mag / np.max(mag))
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs.tolist(), mag_db.tolist()


def compute_fft_mag(x: List[float], fs: float, nfft: Optional[int] = None) -> Tuple[List[float], List[float]]:
    if np is None or len(x) < 8:
        return [], []
    x = np.asarray(x, dtype=float)
    if nfft is None:
        nfft = int(2 ** np.floor(np.log2(min(4096, max(128, len(x))))))
    win = np.hanning(nfft)
    if len(x) < nfft:
        xz = np.zeros(nfft)
        xz[:len(x)] = x
        x = xz
    else:
        x = x[-nfft:]
    x = x - np.mean(x)
    X = np.fft.rfft(x * win)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs.tolist(), mag.tolist()


def generate_plots(dir_path: str, summaries: List[SessionSummary], show: bool = False, remove_dc: bool = True, frf_only: bool = False) -> None:
    if not HAS_MPL or np is None:
        print("⚠️ matplotlib o numpy no disponibles; no se generarán gráficos.")
        return
    for s in summaries:
        base = s.meta.base
        csv_path = os.path.join(dir_path, base + '.csv')
        data = parse_numeric_data(csv_path)
        t = data['t']
        vib = data['vib_ai0']
        force = data['force_ai0']
        if not t or (not vib and not force):
            continue
        # Remover DC si aplica
        if remove_dc:
            if vib:
                m = float(np.mean(vib))
                vib = (np.asarray(vib) - m).tolist()
            if force:
                m = float(np.mean(force))
                force = (np.asarray(force) - m).tolist()

        if not frf_only:
            # Figura tiempo (2 subplots)
            fig1, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            fig1.suptitle(f"{base} - Señales en tiempo" + (" (DC removido)" if remove_dc else ""))
            if vib:
                ax[0].plot(t, vib, color='#FFD700')
                ax[0].set_ylabel('Acel (g)')
                ax[0].grid(True, alpha=0.3)
            if force:
                ax[1].plot(t, force, color='#E91E63')
                ax[1].set_ylabel('Fza (V)')
                ax[1].grid(True, alpha=0.3)
            ax[1].set_xlabel('Tiempo (s)')
            fig1.tight_layout()
            out1 = os.path.join(dir_path, f"{base}_time.png")
            fig1.savefig(out1, dpi=130)
            if not show:
                plt.close(fig1)

        if not frf_only:
            # Figura FFT (overlay vib/force)
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
            fig2.suptitle(f"{base} - FFT normalizada (0 dB = pico)" + (" (DC removido)" if remove_dc else ""))
            if vib and s.meta.sr_vib:
                fv, dv = compute_fft_db(vib, s.meta.sr_vib)
                if fv:
                    ax2.plot(fv, dv, label='Vib ai0', color='#FFD700')
            if force and s.meta.sr_force:
                ff, df = compute_fft_db(force, s.meta.sr_force)
                if ff:
                    ax2.plot(ff, df, label='Fza ai0', color='#E91E63')
            ax2.set_xlabel('Frecuencia (Hz)')
            ax2.set_ylabel('Amplitud (dB)')
            ax2.set_ylim(-80, 0)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            fig2.tight_layout()
            out2 = os.path.join(dir_path, f"{base}_fft.png")
            fig2.savefig(out2, dpi=130)
            if not show:
                plt.close(fig2)

        # Figura FRF (Accel/Force)
        if vib and force and s.meta.sr_vib and s.meta.sr_force:
            # NFFT común para misma resolución
            nfft_common = int(2 ** np.floor(np.log2(min(4096, max(128, len(vib), len(force))))))
            fv, mv = compute_fft_mag(vib, s.meta.sr_vib, nfft=nfft_common)
            ff, mf = compute_fft_mag(force, s.meta.sr_force, nfft=nfft_common)
            if fv and ff:
                m = min(len(mv), len(mf))
                eps = 1e-12
                frf_db = 20.0 * np.log10((np.asarray(mv[:m]) + eps) / (np.asarray(mf[:m]) + eps))
                fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4))
                fig3.suptitle(f"{base} - FRF Accel/Force (dB)" + (" (DC removido)" if remove_dc else ""))
                ax3.plot(fv[:m], frf_db, color='#03A9F4', label='|A/F| dB (g/V)')
                ax3.set_xlabel('Frecuencia (Hz)')
                ax3.set_ylabel('FRF (dB)')
                ax3.set_ylim(-80, 80)
                ax3.grid(True, alpha=0.3)
                ax3.legend()
                fig3.tight_layout()
                out3 = os.path.join(dir_path, f"{base}_frf.png")
                fig3.savefig(out3, dpi=130)
                if not show:
                    plt.close(fig3)

    # Si se solicitó mostrar, mantener las ventanas hasta que el usuario las cierre
    if show:
        plt.show()


def main(argv: List[str]) -> int:
    # Flags simples
    show = False
    make_plots = False
    remove_dc = True
    frf_only = False
    args = []
    for a in argv[1:]:
        if a == '--show':
            show = True
        elif a == '--plots':
            make_plots = True
        elif a == '--no-remove-dc':
            remove_dc = False
        elif a == '--frf-only':
            frf_only = True
        else:
            args.append(a)
    if len(args) >= 1:
        dir_path = args[0]
    else:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experimentos_mesa')
    if not os.path.isdir(dir_path):
        print(f"❌ Carpeta no encontrada: {dir_path}")
        return 2

    summaries = collect_summaries(dir_path)
    report = build_report(dir_path)
    print("\n" + report + "\n")
    save_report(dir_path, report)
    if make_plots:
        generate_plots(dir_path, summaries, show=show, remove_dc=remove_dc, frf_only=frf_only)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

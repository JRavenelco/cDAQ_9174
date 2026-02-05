"""
Control Avanzado del Levitador Magnético - Experimentos de Identificación
==========================================================================

Este script permite ejecutar diferentes perfiles de referencia para:
1. Identificar parámetros físicos (k0, k, a)
2. Explorar la zona no lineal (acercar sin pegar)
3. Generar datos para el benchmark

Uso:
    python control_experimentos.py --experimento escalon
    python control_experimentos.py --experimento approach --y_min 0.002
    python control_experimentos.py --experimento senoidal --freq 0.5

Autor: Jesús (Doctorado UAQ)
"""

import time
import math
from serial_win32 import Win32Serial
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

# =============================================================================
# PARÁMETROS DEL SISTEMA (del benchmark)
# =============================================================================
# Parámetros físicos identificados
K0_NOMINAL = 0.0363  # H - Inductancia base
K_NOMINAL = 0.0035   # H - Variación de inductancia
A_NOMINAL = 0.0052   # m - Parámetro de forma

# Límites de operación segura
Y_MIN_SEGURO = 0.002  # 2mm - Mínimo seguro (evitar pegado)
Y_MAX = 0.008         # 8mm - Máximo (límite físico)
Y_EQUILIBRIO = 0.005  # 5mm - Punto de equilibrio nominal

# =============================================================================
# PARÁMETROS DE CONTROL PID
# =============================================================================
Ts = 0.01
kp = 100
ki = 50
kd = 1.5
kpi = 12.0
kii = 3000.0
Vref = 9.86
Iref = 0.827
Rs = 2.2

# =============================================================================
# PERFILES DE REFERENCIA PARA EXPERIMENTOS
# =============================================================================

class ReferenciaEscalon:
    """Escalón simple para respuesta transitoria."""
    def __init__(self, y0=0.005, y1=0.004, t_cambio=5.0):
        self.y0 = y0
        self.y1 = y1
        self.t_cambio = t_cambio
    
    def __call__(self, t):
        return self.y0 if t < self.t_cambio else self.y1


class ReferenciaSenoidal:
    """Senoidal para análisis de frecuencia."""
    def __init__(self, y_centro=0.005, amplitud=0.001, frecuencia=0.5, t_inicio=3.0):
        self.y_centro = y_centro
        self.amplitud = amplitud
        self.frecuencia = frecuencia
        self.t_inicio = t_inicio
    
    def __call__(self, t):
        if t < self.t_inicio:
            return self.y_centro
        return self.y_centro + self.amplitud * math.sin(2 * math.pi * self.frecuencia * (t - self.t_inicio))


class ReferenciaApproach:
    """
    Aproximación gradual al electroimán para explorar zona no lineal.
    Útil para identificar el punto crítico antes del pegado.
    """
    def __init__(self, y_inicio=0.006, y_min=0.003, velocidad=0.0002, t_inicio=3.0):
        self.y_inicio = y_inicio
        self.y_min = max(y_min, Y_MIN_SEGURO)  # Límite de seguridad
        self.velocidad = velocidad  # m/s de descenso
        self.t_inicio = t_inicio
    
    def __call__(self, t):
        if t < self.t_inicio:
            return self.y_inicio
        
        t_rel = t - self.t_inicio
        y = self.y_inicio - self.velocidad * t_rel
        return max(y, self.y_min)  # No bajar del mínimo


class ReferenciaChirp:
    """Barrido de frecuencia (chirp) para análisis espectral completo."""
    def __init__(self, y_centro=0.005, amplitud=0.0008, f_inicio=0.1, f_fin=2.0, duracion=30.0, t_inicio=3.0):
        self.y_centro = y_centro
        self.amplitud = amplitud
        self.f_inicio = f_inicio
        self.f_fin = f_fin
        self.duracion = duracion
        self.t_inicio = t_inicio
    
    def __call__(self, t):
        if t < self.t_inicio:
            return self.y_centro
        
        t_rel = t - self.t_inicio
        if t_rel > self.duracion:
            return self.y_centro
        
        # Frecuencia instantánea (chirp lineal)
        f_inst = self.f_inicio + (self.f_fin - self.f_inicio) * t_rel / self.duracion
        fase = 2 * math.pi * (self.f_inicio * t_rel + 0.5 * (self.f_fin - self.f_inicio) * t_rel**2 / self.duracion)
        
        return self.y_centro + self.amplitud * math.sin(fase)


class ReferenciaMultiEscalon:
    """Múltiples escalones para explorar diferentes puntos de operación."""
    def __init__(self, niveles=[0.006, 0.005, 0.004, 0.003], t_por_nivel=10.0, t_inicio=3.0):
        self.niveles = niveles
        self.t_por_nivel = t_por_nivel
        self.t_inicio = t_inicio
    
    def __call__(self, t):
        if t < self.t_inicio:
            return self.niveles[0]
        
        t_rel = t - self.t_inicio
        idx = int(t_rel / self.t_por_nivel)
        idx = min(idx, len(self.niveles) - 1)
        return self.niveles[idx]


class ReferenciaPRBS:
    """Señal pseudo-aleatoria binaria para identificación de sistemas."""
    def __init__(self, y_bajo=0.004, y_alto=0.006, periodo=0.5, seed=42, t_inicio=3.0):
        self.y_bajo = y_bajo
        self.y_alto = y_alto
        self.periodo = periodo
        self.seed = seed
        self.t_inicio = t_inicio
        np.random.seed(seed)
        self._secuencia = np.random.randint(0, 2, size=1000)
    
    def __call__(self, t):
        if t < self.t_inicio:
            return (self.y_bajo + self.y_alto) / 2
        
        t_rel = t - self.t_inicio
        idx = int(t_rel / self.periodo) % len(self._secuencia)
        return self.y_alto if self._secuencia[idx] else self.y_bajo


# =============================================================================
# CONTROLADOR Y ADQUISICIÓN
# =============================================================================

# Cola para comunicación entre hilos
data_queue = queue.Queue()
exit_event = threading.Event()
perfil_referencia = None  # Se configura según el experimento


def control_thread(port='COM1', baudrate=115200, duracion=60.0, guardar_archivo=None):
    """Hilo de control con perfil de referencia configurable."""
    global perfil_referencia
    
    # Variables locales
    pv, i, y, y_1, ef, ef_1, u, t = 0, 0, 0, 0, 0, 0, 0, 0
    esc = 0.05 / 1023.0
    esci = 5.0 / (Rs * 1023.0)
    escs = 254.0 / Vref
    iTs = 1 / Ts
    pwmf, yd, proporcional, derivativa, ie, ied, id_val, ei, propi, intei, integral = 0, 0.005, 0, 0, 0, 0, 0, 0, 0, 0, 0

    ser = None
    fp = None
    
    try:
        ser = Win32Serial(port, baudrate)
        print(f"✅ Puerto {port} abierto para control.")
        
        # Esperar switch de activación
        print("⏳ Esperando switch del microcontrolador (byte 0xAA)...")
        while not exit_event.is_set():
            recibido = ser.read(1)
            if recibido == b'\xAA':
                print("✅ ¡Switch activado! Iniciando control...")
                break
        else:
            print("❌ Cierre solicitado antes de la activación.")
            return

        # Preparar archivo de salida
        if guardar_archivo:
            fp = open(guardar_archivo, "w+")
            fp.write("# t\tyd\ty\tid\tie\tu\n")
        
        # Bucle de control principal
        flagcom = 0
        pv = 0
        i = 0
        t_inicio = time.time()
        
        while not exit_event.is_set():
            # Verificar duración
            if time.time() - t_inicio > duracion:
                print(f"\n✅ Experimento completado ({duracion}s)")
                break
            
            b = ser.read(1)
            if not b:
                continue
            
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
                
                # Validar trama
                if pv > 1023 or i > 1023:
                    flagcom = 0
                    continue
                
                # Obtener referencia del perfil
                if perfil_referencia:
                    yd = perfil_referencia(t)
                else:
                    yd = 0.005  # Default
                
                # Límites de seguridad
                yd = max(Y_MIN_SEGURO, min(yd, Y_MAX))
                
                # --- Ley de Control PID ---
                ef_1, y_1 = ef, y
                y = esc * pv
                ie = esci * i
                ef = yd - y
                proporcional = kp * ef
                derivativa = kd * (ef - ef_1) * iTs
                
                if -Iref < integral < Iref:
                    integral += ki * Ts * ef
                else:
                    integral = 0.95 * Iref if integral >= Iref else -0.95 * Iref
                
                id_val = proporcional + integral + derivativa
                if id_val > 0:
                    id_val = 0
                if id_val <= -Iref:
                    id_val = -Iref
                
                ied = -id_val
                ei = ied - ie
                propi = kpi * ei
                
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
                
                # Enviar datos a la gráfica
                data_queue.put((t, y, yd, u, ie))
                
                # Guardar en archivo
                if fp:
                    fp.write(f"{t:.4f}\t{yd:.6f}\t{y:.6f}\t{ied:.4f}\t{ie:.4f}\t{u:.4f}\n")
                    fp.flush()
                
                flagcom = 0
                t += Ts

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
        if fp:
            fp.close()
        print("🔌 Hilo de control finalizado.")


# =============================================================================
# VISUALIZACIÓN EN TIEMPO REAL
# =============================================================================

def crear_grafica():
    """Crea la gráfica de tiempo real con 3 subplots."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Posición
    axes[0].set_ylabel('Posición [mm]')
    axes[0].set_title('Control del Levitador - Experimento')
    line_y, = axes[0].plot([], [], 'b-', label='y medida', linewidth=1)
    line_yd, = axes[0].plot([], [], 'r--', label='yd referencia', linewidth=1)
    axes[0].axhline(y=Y_MIN_SEGURO*1000, color='orange', linestyle=':', label=f'Límite seguro ({Y_MIN_SEGURO*1000}mm)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Control
    axes[1].set_ylabel('Voltaje [V]')
    line_u, = axes[1].plot([], [], 'g-', linewidth=1)
    axes[1].grid(True, alpha=0.3)
    
    # Corriente
    axes[2].set_ylabel('Corriente [A]')
    axes[2].set_xlabel('Tiempo [s]')
    line_i, = axes[2].plot([], [], 'm-', linewidth=1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, axes, (line_y, line_yd, line_u, line_i)


def main():
    global perfil_referencia
    
    parser = argparse.ArgumentParser(description='Experimentos de Control del Levitador')
    parser.add_argument('--experimento', '-e', default='escalon',
                        choices=['escalon', 'senoidal', 'approach', 'chirp', 'multiescalon', 'prbs'],
                        help='Tipo de experimento')
    parser.add_argument('--duracion', '-d', type=float, default=30.0, help='Duración [s]')
    parser.add_argument('--y_min', type=float, default=0.003, help='Posición mínima [m] para approach')
    parser.add_argument('--freq', type=float, default=0.5, help='Frecuencia [Hz] para senoidal')
    parser.add_argument('--puerto', '-p', default='COM1', help='Puerto serie')
    parser.add_argument('--guardar', '-g', action='store_true', help='Guardar datos automáticamente')
    args = parser.parse_args()
    
    # Configurar perfil de referencia según experimento
    print(f"\n{'='*60}")
    print(f"🔬 EXPERIMENTO: {args.experimento.upper()}")
    print(f"{'='*60}")
    
    if args.experimento == 'escalon':
        perfil_referencia = ReferenciaEscalon(y0=0.005, y1=0.004, t_cambio=5.0)
        print("📊 Escalón: 5mm → 4mm en t=5s")
        
    elif args.experimento == 'senoidal':
        perfil_referencia = ReferenciaSenoidal(y_centro=0.005, amplitud=0.001, frecuencia=args.freq)
        print(f"📊 Senoidal: centro=5mm, amplitud=1mm, f={args.freq}Hz")
        
    elif args.experimento == 'approach':
        perfil_referencia = ReferenciaApproach(y_inicio=0.006, y_min=args.y_min, velocidad=0.0002)
        print(f"📊 Approach: 6mm → {args.y_min*1000:.1f}mm (gradual)")
        print(f"⚠️  Límite de seguridad: {Y_MIN_SEGURO*1000}mm")
        
    elif args.experimento == 'chirp':
        perfil_referencia = ReferenciaChirp(y_centro=0.005, amplitud=0.0008, f_inicio=0.1, f_fin=2.0)
        print("📊 Chirp: 0.1Hz → 2Hz, amplitud=0.8mm")
        
    elif args.experimento == 'multiescalon':
        perfil_referencia = ReferenciaMultiEscalon(niveles=[0.006, 0.005, 0.004, 0.0035], t_por_nivel=8.0)
        print("📊 Multi-escalón: 6mm → 5mm → 4mm → 3.5mm")
        
    elif args.experimento == 'prbs':
        perfil_referencia = ReferenciaPRBS(y_bajo=0.004, y_alto=0.006, periodo=0.5)
        print("📊 PRBS: alternando 4mm ↔ 6mm")
    
    print(f"⏱️  Duración: {args.duracion}s")
    print(f"{'='*60}\n")
    
    # Archivo de salida
    guardar_archivo = None
    if args.guardar:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).parent / 'experimentos'
        output_dir.mkdir(exist_ok=True)
        guardar_archivo = str(output_dir / f"exp_{args.experimento}_{timestamp}.txt")
        print(f"💾 Guardando en: {guardar_archivo}")
    
    # Iniciar hilo de control
    control = threading.Thread(
        target=control_thread,
        args=(args.puerto, 115200, args.duracion, guardar_archivo),
        daemon=True
    )
    control.start()
    
    # Crear gráfica
    fig, axes, lines = crear_grafica()
    line_y, line_yd, line_u, line_i = lines
    
    # Datos para la gráfica
    time_points, y_vals, yd_vals, u_vals, i_vals = [], [], [], [], []
    
    def update_plot(frame):
        while not data_queue.empty():
            t, y, yd, u, i = data_queue.get_nowait()
            time_points.append(t)
            y_vals.append(y * 1000)  # Convertir a mm
            yd_vals.append(yd * 1000)
            u_vals.append(u)
            i_vals.append(i)
        
        if time_points:
            line_y.set_data(time_points, y_vals)
            line_yd.set_data(time_points, yd_vals)
            line_u.set_data(time_points, u_vals)
            line_i.set_data(time_points, i_vals)
            
            for ax in axes:
                ax.relim()
                ax.autoscale_view()
        
        return line_y, line_yd, line_u, line_i
    
    ani = FuncAnimation(fig, update_plot, interval=100, blit=True, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n⏹️  Detenido por el usuario")
    finally:
        exit_event.set()
        control.join(timeout=2)
        print("✅ Programa finalizado.")


if __name__ == '__main__':
    main()

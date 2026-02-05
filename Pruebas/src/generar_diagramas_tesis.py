"""
GENERADOR DE DIAGRAMAS TECNICOS PARA TESIS
==========================================
Crea 3 visualizaciones de alto nivel:
1. Arquitectura KAN-PINN
2. Aceleracion por GPU
3. Proceso de Limpieza de Friccion
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Configuracion
OUTPUT_DIR = "diagramas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colores UAQ
AZUL_UAQ = "#00274c"
DORADO_UAQ = "#c5911e"
VERDE = "#38a169"
ROJO = "#B30000"
GRIS = "#7a7a7a"
BLANCO = "#ffffff"

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'


def diagrama_1_arquitectura_kan_pinn():
    """
    Diagrama 1: Arquitectura KAN-PINN (El 'Cerebro')
    Muestra la red neuronal con funciones de activacion aprendibles
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Titulo
    ax.text(7, 9.5, 'Arquitectura KAN-PINN', fontsize=20, fontweight='bold', 
            ha='center', color=AZUL_UAQ)
    ax.text(7, 9.0, '(Kolmogorov-Arnold Network + Physics-Informed Neural Network)', 
            fontsize=12, ha='center', color=GRIS, style='italic')
    
    # === ENTRADA ===
    entrada_box = FancyBboxPatch((0.5, 4), 1.5, 2, boxstyle="round,pad=0.05",
                                  facecolor=AZUL_UAQ, edgecolor='black', linewidth=2)
    ax.add_patch(entrada_box)
    ax.text(1.25, 5, '$t$\n(Tiempo)', fontsize=12, ha='center', va='center', 
            color=BLANCO, fontweight='bold')
    
    # === CAPA KAN 1 ===
    # Nodos con mini-graficas de splines
    for i, y in enumerate([6.5, 5, 3.5]):
        # Nodo
        circle = Circle((3.5, y), 0.4, facecolor=DORADO_UAQ, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        # Mini spline dentro
        xs = np.linspace(-0.3, 0.3, 20)
        ys = 0.15 * np.sin(8*xs + i) + y
        ax.plot(xs + 3.5, ys, color=BLANCO, linewidth=2)
    
    ax.text(3.5, 7.5, 'KAN Layer 1', fontsize=10, ha='center', color=DORADO_UAQ, fontweight='bold')
    ax.text(3.5, 2.5, r'$\phi_1(x)$ aprendible', fontsize=9, ha='center', color=GRIS, style='italic')
    
    # === CAPA KAN 2 ===
    for i, y in enumerate([6, 4]):
        circle = Circle((5.5, y), 0.4, facecolor=DORADO_UAQ, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        xs = np.linspace(-0.3, 0.3, 20)
        ys = 0.15 * np.tanh(5*xs + i*2) + y
        ax.plot(xs + 5.5, ys, color=BLANCO, linewidth=2)
    
    ax.text(5.5, 7.5, 'KAN Layer 2', fontsize=10, ha='center', color=DORADO_UAQ, fontweight='bold')
    
    # === SALIDAS DE LA RED ===
    # Posicion
    salida_x = FancyBboxPatch((7, 5.5), 1.5, 1, boxstyle="round,pad=0.05",
                               facecolor=VERDE, edgecolor='black', linewidth=2)
    ax.add_patch(salida_x)
    ax.text(7.75, 6, r'$\hat{x}$', fontsize=14, ha='center', va='center', 
            color=BLANCO, fontweight='bold')
    
    # Velocidad
    salida_v = FancyBboxPatch((7, 3.5), 1.5, 1, boxstyle="round,pad=0.05",
                               facecolor=VERDE, edgecolor='black', linewidth=2)
    ax.add_patch(salida_v)
    ax.text(7.75, 4, r'$\hat{v}$', fontsize=14, ha='center', va='center', 
            color=BLANCO, fontweight='bold')
    
    ax.text(7.75, 7, 'Salidas Red', fontsize=10, ha='center', color=VERDE, fontweight='bold')
    
    # === BLOQUE DE FISICA ===
    fisica_box = FancyBboxPatch((9.5, 2.5), 3.5, 5, boxstyle="round,pad=0.1",
                                 facecolor='#fff3cd', edgecolor=ROJO, linewidth=3)
    ax.add_patch(fisica_box)
    ax.text(11.25, 7, 'Physics Loss', fontsize=12, ha='center', color=ROJO, fontweight='bold')
    
    # Ecuacion de fisica
    ax.text(11.25, 5.8, r'$\mathcal{L}_{physics} =$', fontsize=11, ha='center', color=AZUL_UAQ)
    ax.text(11.25, 5.0, r'$\left( m\ddot{x} + c\dot{x} + kx \right)$', fontsize=11, ha='center', color=AZUL_UAQ)
    ax.text(11.25, 4.2, r'$- F_{medida}$', fontsize=11, ha='center', color=ROJO)
    
    # Parametros identificables
    ax.text(11.25, 3.2, 'Parametros:', fontsize=9, ha='center', color=GRIS)
    ax.text(11.25, 2.8, r'$m, k, c, \alpha$', fontsize=10, ha='center', color=DORADO_UAQ, fontweight='bold')
    
    # === FLECHAS DE CONEXION ===
    # Entrada -> KAN1
    ax.annotate('', xy=(3.1, 5), xytext=(2, 5),
                arrowprops=dict(arrowstyle='->', color=AZUL_UAQ, lw=2))
    
    # KAN1 -> KAN2 (multiples)
    for y1 in [6.5, 5, 3.5]:
        for y2 in [6, 4]:
            ax.plot([3.9, 5.1], [y1, y2], color=GRIS, lw=0.5, alpha=0.5)
    
    # KAN2 -> Salidas
    ax.annotate('', xy=(7, 6), xytext=(5.9, 6),
                arrowprops=dict(arrowstyle='->', color=VERDE, lw=2))
    ax.annotate('', xy=(7, 4), xytext=(5.9, 4),
                arrowprops=dict(arrowstyle='->', color=VERDE, lw=2))
    
    # Salidas -> Fisica
    ax.annotate('', xy=(9.5, 5), xytext=(8.5, 5.5),
                arrowprops=dict(arrowstyle='->', color=ROJO, lw=2))
    ax.annotate('', xy=(9.5, 4.5), xytext=(8.5, 4),
                arrowprops=dict(arrowstyle='->', color=ROJO, lw=2))
    
    # === BACKPROPAGATION ===
    ax.annotate('', xy=(3.5, 2), xytext=(11.25, 2),
                arrowprops=dict(arrowstyle='<-', color=ROJO, lw=2, 
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(7.5, 1.2, 'Backpropagation', fontsize=11, ha='center', color=ROJO, fontweight='bold')
    ax.text(7.5, 0.7, '(Ajuste de pesos y parametros fisicos)', fontsize=9, ha='center', color=GRIS)
    
    plt.tight_layout()
    output = os.path.join(OUTPUT_DIR, "arquitectura_kan_pinn.png")
    fig.savefig(output, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f" Guardado: {output}")


def diagrama_2_aceleracion_gpu():
    """
    Diagrama 2: Aceleracion por GPU (El 'Musculo')
    Muestra el flujo de datos paralelo
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Titulo
    ax.text(7, 7.5, 'Aceleracion por GPU: Flujo de Datos Paralelo', fontsize=18, 
            fontweight='bold', ha='center', color=AZUL_UAQ)
    
    # === ARCHIVOS CSV (IZQUIERDA) ===
    ax.text(1.5, 6.5, 'Archivos de Datos', fontsize=11, ha='center', 
            color=GRIS, fontweight='bold')
    
    for i, (nombre, color) in enumerate([('fuerza.csv', ROJO), ('vibracion.csv', VERDE)]):
        y = 5.5 - i * 1.5
        rect = FancyBboxPatch((0.5, y - 0.4), 2, 0.8, boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor='black', alpha=0.7)
        ax.add_patch(rect)
        ax.text(1.5, y, nombre, fontsize=9, ha='center', va='center', color=BLANCO, fontweight='bold')
    
    # === FLECHA CARGA ===
    ax.annotate('', xy=(3.5, 4.5), xytext=(2.7, 4.5),
                arrowprops=dict(arrowstyle='->', color=AZUL_UAQ, lw=3))
    ax.text(3.1, 5.2, 'DataLoader', fontsize=9, ha='center', color=GRIS)
    
    # === CPU (BATCHING) ===
    cpu_box = FancyBboxPatch((3.5, 3.5), 2.5, 2, boxstyle="round,pad=0.1",
                              facecolor='#e0e0e0', edgecolor=AZUL_UAQ, linewidth=2)
    ax.add_patch(cpu_box)
    ax.text(4.75, 5.2, 'CPU', fontsize=12, ha='center', color=AZUL_UAQ, fontweight='bold')
    ax.text(4.75, 4.5, 'Batching', fontsize=10, ha='center', color=GRIS)
    ax.text(4.75, 4.0, r'$N = 1024$', fontsize=10, ha='center', color=AZUL_UAQ)
    
    # === FLECHA A GPU ===
    ax.annotate('', xy=(7, 4.5), xytext=(6, 4.5),
                arrowprops=dict(arrowstyle='->', color=VERDE, lw=4))
    ax.text(6.5, 5.2, '.to(cuda:0)', fontsize=9, ha='center', color=VERDE, fontweight='bold')
    
    # === GPU (GRANDE) ===
    gpu_box = FancyBboxPatch((7, 1.5), 5.5, 5, boxstyle="round,pad=0.1",
                              facecolor='#d4edda', edgecolor=VERDE, linewidth=3)
    ax.add_patch(gpu_box)
    ax.text(9.75, 6.2, 'GPU (CUDA)', fontsize=14, ha='center', color=VERDE, fontweight='bold')
    
    # Nucleos paralelos
    ax.text(9.75, 5.5, 'Procesamiento Masivamente Paralelo', fontsize=10, ha='center', color=GRIS)
    
    # Grid de nucleos
    for i in range(5):
        for j in range(3):
            rect = Rectangle((7.5 + i*1, 4.2 - j*0.7), 0.8, 0.5, 
                            facecolor=DORADO_UAQ, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
    ax.text(9.75, 2.3, f'1024 hilos simultaneos', fontsize=9, ha='center', color=AZUL_UAQ)
    
    # Ecuacion de autograd
    ax.text(9.75, 1.8, r'Autograd: $\frac{\partial^2 \hat{x}}{\partial t^2} = \ddot{x}$', 
            fontsize=11, ha='center', color=ROJO, fontweight='bold')
    
    # === COMPARACION TIEMPOS ===
    ax.text(1.5, 1.5, 'Speedup:', fontsize=11, ha='left', color=AZUL_UAQ, fontweight='bold')
    ax.text(1.5, 1.0, 'CPU: ~1800s (30 min)', fontsize=10, ha='left', color=GRIS)
    ax.text(1.5, 0.5, 'GPU: ~4s', fontsize=10, ha='left', color=VERDE, fontweight='bold')
    ax.text(1.5, 0.0, r'$\approx 450\times$ mas rapido', fontsize=11, ha='left', color=DORADO_UAQ, fontweight='bold')
    
    plt.tight_layout()
    output = os.path.join(OUTPUT_DIR, "flujo_gpu.png")
    fig.savefig(output, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f" Guardado: {output}")


def diagrama_3_proceso_friccion():
    """
    Diagrama 3: Proceso de 'Limpieza' de Friccion
    Grafico de senales apiladas
    CORREGIDO: La senal medida (roja) debe mostrar distorsion por friccion
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Proceso de Separacion de Friccion (KAN-PINN)', 
                 fontsize=16, fontweight='bold', color=AZUL_UAQ, y=0.98)
    
    # Generar datos sinteticos para ilustracion
    t = np.linspace(0, 2, 1000)
    freq = 20  # Hz
    
    # Senal "real" con friccion
    x = 0.001 * np.sin(2 * np.pi * freq * t)  # Desplazamiento
    v = 0.001 * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)  # Velocidad
    a = -0.001 * (2 * np.pi * freq)**2 * np.sin(2 * np.pi * freq * t)  # Aceleracion
    
    # Parametros "fisicos"
    m, k, c = 40, 400000, 400
    
    # Fuerza inercial + elastica (LINEAL - sinusoidal perfecta)
    F_lineal = m * a + k * x + c * v
    
    # Friccion simulada - AGRESIVA para que se note la distorsion
    # Modelo Bouc-Wen con histeresis visible
    F_friccion = 40 * np.tanh(8000 * v) + 15 * np.sign(v) + np.random.normal(0, 3, len(t))
    
    # Fuerza total "medida" = lineal + friccion (DISTORSIONADA!)
    F_total = F_lineal + F_friccion
    
    # === PANEL 1: Fuerza Total (Sucia) ===
    axes[0].plot(t, F_total, color=ROJO, lw=1.5, alpha=0.8)
    axes[0].fill_between(t, F_total, alpha=0.2, color=ROJO)
    axes[0].set_ylabel('Fuerza Total\n(Medida)', fontsize=11, color=ROJO)
    axes[0].set_title('Senal Original: Contiene inercia + rigidez + friccion', fontsize=11, color=GRIS)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', lw=0.5)
    
    # Anotacion
    axes[0].text(0.1, max(F_total)*0.8, r'$F_{medida} = ?$', fontsize=12, color=ROJO, fontweight='bold')
    
    # === PANEL 2: Componente Lineal (Identificada) ===
    axes[1].plot(t, F_lineal, color=AZUL_UAQ, lw=1.5)
    axes[1].fill_between(t, F_lineal, alpha=0.2, color=AZUL_UAQ)
    axes[1].set_ylabel('Componente Lineal\n(Identificada)', fontsize=11, color=AZUL_UAQ)
    axes[1].set_title(r'KAN-PINN identifica: $F_{lineal} = m\ddot{x} + c\dot{x} + kx$', fontsize=11, color=GRIS)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='black', lw=0.5)
    
    # Parametros identificados
    axes[1].text(0.1, max(F_lineal)*0.8, f'm={m} kg, k={k/1000:.0f} kN/m', fontsize=10, 
                color=AZUL_UAQ, fontweight='bold')
    
    # === PANEL 3: Friccion Pura (Residuo) ===
    F_friccion_estimada = F_total - F_lineal
    axes[2].plot(t, F_friccion_estimada, color=VERDE, lw=1.5)
    axes[2].fill_between(t, F_friccion_estimada, alpha=0.2, color=VERDE)
    axes[2].set_ylabel('Friccion Pura\n(Residuo)', fontsize=11, color=VERDE)
    axes[2].set_xlabel('Tiempo (s)', fontsize=12)
    axes[2].set_title(r'Resta: $F_{friccion} = F_{medida} - F_{lineal}$', fontsize=11, color=GRIS)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='black', lw=0.5)
    
    # Anotacion de histeresis
    axes[2].text(1.5, max(F_friccion_estimada)*0.6, 'Lazo de\nHisteresis', fontsize=10, 
                color=VERDE, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=VERDE))
    
    # Ajustar espaciado
    plt.tight_layout()
    
    # Agregar subplot para lazo de histeresis
    # Crear axes inset en panel 3
    ax_inset = axes[2].inset_axes([0.65, 0.15, 0.3, 0.7])
    ax_inset.plot(x * 1000, F_friccion_estimada, color=VERDE, lw=1, alpha=0.7)
    ax_inset.set_xlabel('x (mm)', fontsize=8)
    ax_inset.set_ylabel('F (N)', fontsize=8)
    ax_inset.set_title('Lazo F-x', fontsize=9, color=VERDE)
    ax_inset.grid(True, alpha=0.2)
    ax_inset.tick_params(labelsize=7)
    
    output = os.path.join(OUTPUT_DIR, "proceso_friccion.png")
    fig.savefig(output, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f" Guardado: {output}")


if __name__ == "__main__":
    print("Generando diagramas tecnicos para tesis...")
    print("=" * 50)
    
    diagrama_1_arquitectura_kan_pinn()
    diagrama_2_aceleracion_gpu()
    diagrama_3_proceso_friccion()
    
    print("=" * 50)
    print(" Todos los diagramas generados en src/diagramas/")

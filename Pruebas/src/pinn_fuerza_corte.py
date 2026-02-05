import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import detrend, butter, filtfilt
import os
from glob import glob
import time

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. PREPROCESAMIENTO DE DATOS
# ==========================================
def cargar_y_procesar_datos(filepath):
    print(f"📂 Cargando: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath)
    
    t = df['tiempo_s'].values
    fuerza = df['fuerza_V'].values
    acel_g = df['aceleracion_g'].values
    
    # Conversión a unidades SI (aproximada para el modelo)
    # Asumimos que el voltaje ya es proporcional a la fuerza
    acel_ms2 = acel_g * 9.81
    
    # Filtrado para integración (High-pass para evitar drift)
    fs = 1 / (t[1] - t[0])
    b, a = butter(2, 5.0, btype='high', fs=fs) # Corte en 5Hz
    acel_filt = filtfilt(b, a, acel_ms2)
    
    # Integración numérica: a -> v -> x
    vel = cumulative_trapezoid(acel_filt, t, initial=0)
    vel = detrend(vel) # Remover tendencia lineal
    
    disp = cumulative_trapezoid(vel, t, initial=0)
    disp = detrend(disp)
    
    # Normalización Min-Max a [-1, 1] (Crucial para PINNs)
    scale_f = np.max(np.abs(fuerza))
    scale_x = np.max(np.abs(disp))
    scale_v = np.max(np.abs(vel))
    
    f_norm = fuerza / scale_f
    x_norm = disp / scale_x
    v_norm = vel / scale_v
    
    print(f"   Datos procesados: {len(t)} muestras")
    print(f"   Escalas: F_max={scale_f:.2f}, X_max={scale_x:.2e}, V_max={scale_v:.2e}")
    
    return t, f_norm, x_norm, v_norm, scale_f, scale_x, scale_v

# ==========================================
# 2. DEFINICIÓN DE LA PINN
# ==========================================
class BoucWenPINN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Red Neuronal para estimar z(t) (variable histéresis)
        # Input: [t, x, v] -> Output: [z]
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Salida: z
        )
        
        # Parámetros Físicos Entrenables (Physics Parameters)
        # Inicializamos con valores razonables log-transformados para asegurar positividad
        self.log_A = nn.Parameter(torch.tensor(0.0))      # A approx 1.0
        self.log_beta = nn.Parameter(torch.tensor(0.0))   # beta approx 1.0
        self.log_gamma = nn.Parameter(torch.tensor(-1.0)) # gamma approx 0.3
        self.log_n = nn.Parameter(torch.tensor(0.0))      # n approx 1.0
        self.log_k = nn.Parameter(torch.tensor(0.0))      # k (rigidez)
        self.alpha = nn.Parameter(torch.tensor(0.5))      # Ratio (0-1)

    def forward(self, inputs):
        # inputs: [t, x, v]
        z_pred = self.net(inputs)
        return z_pred

    def get_params(self):
        return {
            'A': torch.exp(self.log_A),
            'beta': torch.exp(self.log_beta),
            'gamma': torch.exp(self.log_gamma),
            'n': torch.exp(self.log_n) + 1.0, # n > 1 usualmente
            'k': torch.exp(self.log_k),
            'alpha': torch.sigmoid(self.alpha) # Entre 0 y 1
        }

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
def train_pinn(filepath, epochs=5000):
    # 1. Preparar datos
    t, f_real, x, v, s_f, s_x, s_v = cargar_y_procesar_datos(filepath)
    
    # Convertir a tensores
    t_tens = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device)
    x_tens = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
    v_tens = torch.tensor(v, dtype=torch.float32).view(-1, 1).to(device)
    f_tens = torch.tensor(f_real, dtype=torch.float32).view(-1, 1).to(device)
    
    # Inputs concatenados para la red
    inputs = torch.cat([t_tens, x_tens, v_tens], dim=1)
    
    # Modelo
    model = BoucWenPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_history = []
    
    print("\n🚀 Iniciando entrenamiento PINN...")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Predicción de z
        # Necesitamos calcular dz/dt para la loss física
        # Usamos autodiff respecto al tiempo NO es directo porque t es input numérico
        # Usaremos diferencias finitas para dz/dt ya que t es discreto y denso
        
        z_pred = model(inputs)
        
        # Calcular derivada temporal de z (dz/dt) numéricamente
        # dz/dt ≈ (z[i+1] - z[i-1]) / (2*dt)
        # Para simplificar en PINN con datos densos, usamos gradientes simples
        dt = t[1] - t[0]
        dz_dt = torch.gradient(z_pred.flatten(), spacing=(dt,))[0].view(-1, 1)
        
        # Obtener parámetros actuales
        params = model.get_params()
        A, beta, gamma, n, k, alpha = params.values()
        
        # --- PHYSICS LOSS (Ecuación Diferencial Bouc-Wen) ---
        # dz/dt = A*v - beta*|v|*z - gamma*v*|z|^n
        # Ajustamos escalas: v real = v_norm * s_v
        v_real = v_tens # Trabajamos en espacio normalizado para estabilidad
        
        physics_res = dz_dt - (A * v_real - beta * torch.abs(v_real) * z_pred - gamma * v_real * torch.abs(z_pred)**n)
        loss_physics = torch.mean(physics_res**2)
        
        # --- DATA LOSS (Fuerza) ---
        # F = alpha*k*x + (1-alpha)*k*z
        f_pred = alpha * k * x_tens + (1 - alpha) * k * z_pred
        loss_data = torch.mean((f_pred - f_tens)**2)
        
        # Loss Total
        loss = loss_data + 0.1 * loss_physics
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.6f} (Data={loss_data.item():.6f}, Phys={loss_physics.item():.6f})")
            print(f"   Params: k={k.item():.2f}, n={n.item():.2f}, alpha={alpha.item():.2f}")
            
    print(f"✅ Entrenamiento finalizado en {time.time()-start_time:.1f}s")
    
    return model, loss_history, t, f_real, x, v, z_pred.detach().cpu().numpy(), f_pred.detach().cpu().numpy()

# ==========================================
# 4. MAIN Y VISUALIZACIÓN
# ==========================================
if __name__ == "__main__":
    # Buscar el archivo con mayor señal (probablemente 50Hz o 70Hz)
    csv_files = sorted(glob(os.path.join("caracterizacion_fuerza", "*_50Hz.csv")))
    if not csv_files:
         csv_files = sorted(glob(os.path.join("caracterizacion_fuerza", "*.csv")))
    
    target_file = csv_files[-1] # Usar el más reciente
    
    model, history, t, f_real, x, v, z_est, f_est = train_pinn(target_file, epochs=3000)
    
    # --- GRÁFICAS ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Entrenamiento
    axes[0,0].plot(history)
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Convergencia de Loss (PINN)')
    axes[0,0].set_xlabel('Epochs')
    axes[0,0].set_ylabel('Loss Total')
    
    # 2. Comparación Tiempo (Zoom)
    zoom = slice(1000, 1500) # Ver un par de ciclos
    axes[0,1].plot(t[zoom], f_real[zoom], 'b-', label='Fuerza Real', alpha=0.7)
    axes[0,1].plot(t[zoom], f_est[zoom], 'r--', label='PINN (Bouc-Wen)', linewidth=2)
    axes[0,1].set_title('Comparación Temporal (Zoom)')
    axes[0,1].legend()
    
    # 3. Ciclo de Histéresis (F vs Desplazamiento/Entrada)
    # Usamos 'v' o 'f' como entrada proxy similar a Lissajous
    axes[1,0].plot(v[zoom], f_real[zoom], 'b-', label='Real', alpha=0.5)
    axes[1,0].plot(v[zoom], f_est[zoom], 'r--', label='PINN', linewidth=2)
    axes[1,0].set_title('Ciclo de Histéresis (Fuerza vs Velocidad)')
    axes[1,0].set_xlabel('Velocidad (norm)')
    axes[1,0].set_ylabel('Fuerza (norm)')
    axes[1,0].legend()
    
    # 4. Variable Oculta Z
    axes[1,1].plot(t[zoom], z_est[zoom], 'g-')
    axes[1,1].set_title('Variable Latente Z (Histéresis aprendida)')
    axes[1,1].set_xlabel('Tiempo')
    
    plt.tight_layout()
    plt.savefig('resultado_pinn_fuerza.png')
    print("💾 Gráfica guardada: resultado_pinn_fuerza.png")
    
    # Mostrar parámetros finales
    p = model.get_params()
    print("\n🧩 PARÁMETROS IDENTIFICADOS POR LA PINN:")
    for k, v in p.items():
        print(f"   {k} = {v.item():.4f}")
    
    plt.show()

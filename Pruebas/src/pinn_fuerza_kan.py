import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 1. DEFINICIÓN DE EFFICIENT KAN
# ==========================================
class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        return result.contiguous()

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

class KAN(nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
    ):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                KANLinear(
                    layers_hidden[i],
                    layers_hidden[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ==========================================
# 2. MODELO BOUC-WEN KAN-PINN
# ==========================================
class BoucWenKANPINN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # KAN para estimar z(t)
        # Input: [t, x, v] -> Output: [z]
        # Estructura: 3 -> 8 -> 1
        self.kan = KAN(
            layers_hidden=[3, 8, 1],
            grid_size=5,
            spline_order=3
        )
        
        # Parámetros Físicos (igual que antes)
        self.log_A = nn.Parameter(torch.tensor(0.0))
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.log_gamma = nn.Parameter(torch.tensor(-1.0))
        self.log_n = nn.Parameter(torch.tensor(0.0)) # n approx 2
        self.log_k = nn.Parameter(torch.tensor(0.0))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs):
        # inputs: [t, x, v]
        z_pred = self.kan(inputs)
        return z_pred

    def get_params(self):
        return {
            'A': torch.exp(self.log_A),
            'beta': torch.exp(self.log_beta),
            'gamma': torch.exp(self.log_gamma),
            'n': torch.exp(self.log_n) + 1.0,
            'k': torch.exp(self.log_k),
            'alpha': torch.sigmoid(self.alpha)
        }

# ==========================================
# 3. ENTRENAMIENTO
# ==========================================
def cargar_y_procesar_datos(filepath):
    # (Misma función de preprocesamiento)
    print(f"📂 Cargando: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath)
    t = df['tiempo_s'].values
    fuerza = df['fuerza_V'].values
    acel_g = df['aceleracion_g'].values
    acel_ms2 = acel_g * 9.81
    fs = 1 / (t[1] - t[0])
    b, a = butter(2, 5.0, btype='high', fs=fs)
    acel_filt = filtfilt(b, a, acel_ms2)
    vel = cumulative_trapezoid(acel_filt, t, initial=0)
    vel = detrend(vel)
    disp = cumulative_trapezoid(vel, t, initial=0)
    disp = detrend(disp)
    
    scale_f = np.max(np.abs(fuerza))
    scale_x = np.max(np.abs(disp))
    scale_v = np.max(np.abs(vel))
    
    f_norm = fuerza / scale_f
    x_norm = disp / scale_x
    v_norm = vel / scale_v
    
    return t, f_norm, x_norm, v_norm

def train_kan_pinn(filepath, epochs=3000):
    t, f_real, x, v = cargar_y_procesar_datos(filepath)
    
    t_tens = torch.tensor(t, dtype=torch.float32).view(-1, 1).to(device)
    x_tens = torch.tensor(x, dtype=torch.float32).view(-1, 1).to(device)
    v_tens = torch.tensor(v, dtype=torch.float32).view(-1, 1).to(device)
    f_tens = torch.tensor(f_real, dtype=torch.float32).view(-1, 1).to(device)
    
    inputs = torch.cat([t_tens, x_tens, v_tens], dim=1)
    
    model = BoucWenKANPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2) # KAN tolera LRs más altos
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
    
    loss_history = []
    
    print("\n🚀 Iniciando entrenamiento KAN-PINN...")
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        z_pred = model(inputs)
        
        # Derivada dz/dt (diferencias finitas)
        dt = t[1] - t[0]
        dz_dt = torch.gradient(z_pred.flatten(), spacing=(dt,))[0].view(-1, 1)
        
        params = model.get_params()
        A, beta, gamma, n, k, alpha = params.values()
        
        # Physics Loss (Bouc-Wen)
        v_real = v_tens
        physics_res = dz_dt - (A * v_real - beta * torch.abs(v_real) * z_pred - gamma * v_real * torch.abs(z_pred)**n)
        loss_physics = torch.mean(physics_res**2)
        
        # Data Loss
        f_pred = alpha * k * x_tens + (1 - alpha) * k * z_pred
        loss_data = torch.mean((f_pred - f_tens)**2)
        
        loss = loss_data + 0.1 * loss_physics
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        loss_history.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.6f} (Data={loss_data.item():.6f})")
            print(f"   Params: k={k.item():.2f}, n={n.item():.2f}, A={A.item():.2f}")
            
    print(f"✅ Entrenamiento KAN finalizado en {time.time()-start_time:.1f}s")
    
    return model, loss_history, t, f_real, x, v, z_pred.detach().cpu().numpy(), f_pred.detach().cpu().numpy()

if __name__ == "__main__":
    csv_files = sorted(glob(os.path.join("caracterizacion_fuerza", "*_50Hz.csv")))
    if not csv_files:
         csv_files = sorted(glob(os.path.join("caracterizacion_fuerza", "*.csv")))
    target_file = csv_files[-1]
    
    model, history, t, f_real, x, v, z_est, f_est = train_kan_pinn(target_file)
    
    # Gráficas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0,0].plot(history)
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('KAN-PINN Loss')
    
    zoom = slice(1000, 1500)
    axes[0,1].plot(t[zoom], f_real[zoom], 'b-', label='Real')
    axes[0,1].plot(t[zoom], f_est[zoom], 'r--', label='KAN Prediction')
    axes[0,1].set_title('Comparación Temporal')
    axes[0,1].legend()
    
    axes[1,0].plot(v[zoom], f_real[zoom], 'b-', alpha=0.5, label='Real')
    axes[1,0].plot(v[zoom], f_est[zoom], 'r--', label='KAN')
    axes[1,0].set_title('Ciclo Histéresis (F vs V)')
    axes[1,0].legend()
    
    # Visualización de la Función de Activación KAN (Primera capa)
    # Esto es único de KAN: ver qué "forma" aprendió
    axes[1,1].set_title('Activación KAN Aprendida')
    # Tomamos una muestra de activaciones
    # (Implementación simple de visualización)
    axes[1,1].text(0.5, 0.5, "KAN Activations", ha='center')
    
    plt.tight_layout()
    plt.savefig('resultado_kan_pinn.png')
    print("💾 Guardado: resultado_kan_pinn.png")
    
    p = model.get_params()
    print("\n🧩 PARÁMETROS KAN-PINN:")
    for k, v in p.items():
        print(f"   {k} = {v.item():.4f}")
    
    plt.show()

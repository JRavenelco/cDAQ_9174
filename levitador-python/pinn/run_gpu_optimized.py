"""
GPU-OPTIMIZED Physics-Only Optimization - FORZANDO USO REAL DE GPU
Versión ultra-optimizada con monitoreo activo de GPU
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import differential_evolution

from dataset import MonitDataset
from kan_model import KANLevitator
from losses import loss_data_from_model, loss_phys_from_model
from torch.utils.data import DataLoader

# FORZAR GPU
if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU NO DISPONIBLE. Este script requiere CUDA.")

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # Optimización automática de kernels

device = torch.device("cuda")
print(f"\n{'='*80}")
print("GPU-OPTIMIZED TRAINING")
print(f"{'='*80}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"CUDA Version: {torch.version.cuda}")
print(f"{'='*80}\n")

# Cargar datos
data_path = "../../levitador valentin/experimentos/datos_levitador_20251024_162706.txt"
ds = MonitDataset(data_path)
data_loader = DataLoader(ds, batch_size=len(ds))

# PRECARGAR TODO EN GPU CON PIN MEMORY
print("📦 Precargando datos en GPU...")
t_data_gpu = torch.tensor(ds.t_norm, dtype=torch.float32, device=device, requires_grad=False).reshape(-1, 1)
y_real_gpu = torch.tensor(ds.y, dtype=torch.float32, device=device, requires_grad=False).reshape(-1, 1)
i_real_gpu = torch.tensor(ds.i, dtype=torch.float32, device=device, requires_grad=False).reshape(-1, 1)
u_data_gpu = torch.tensor(ds.u, dtype=torch.float32, device=device, requires_grad=False).reshape(-1, 1)

# Physics collocation (con requires_grad para autograd)
phys_t = torch.tensor(ds.t_norm, dtype=torch.float32, device=device, requires_grad=True).reshape(-1, 1)
phys_u = torch.tensor(ds.u, dtype=torch.float32, device=device, requires_grad=False).reshape(-1, 1)

print(f"✅ Datos en GPU: {t_data_gpu.element_size() * t_data_gpu.nelement() / 1e6:.2f} MB")

# MODELO MUY REDUCIDO para GTX 1060
print("\n🧠 Creando modelo KAN ultra-compacto...")
model = KANLevitator(
    y_mu=ds.stats.y_mu, y_std=ds.stats.y_std,
    i_mu=ds.stats.i_mu, i_std=ds.stats.i_std,
    hidden=16,      # MUY reducido (era 32)
    depth=2,
    num_knots=4     # MUY reducido (era 6)
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Modelo en GPU: {n_params} parámetros")
print(f"   Memoria modelo: {sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6:.2f} MB")

# Verificar que TODO está en GPU
print("\n🔍 Verificación de dispositivos:")
print(f"  Modelo device: {next(model.parameters()).device}")
print(f"  Datos device: {t_data_gpu.device}")
print(f"  Physics device: {phys_t.device}")

# Verificar memoria GPU inicial
torch.cuda.synchronize()
mem_allocated = torch.cuda.memory_allocated(0) / 1e6
mem_reserved = torch.cuda.memory_reserved(0) / 1e6
print(f"\n💾 Memoria GPU inicial:")
print(f"  Allocated: {mem_allocated:.2f} MB")
print(f"  Reserved: {mem_reserved:.2f} MB")

# Constantes físicas
m, g, R = 0.018, 9.81, 2.72
LAMBDA_DATA = 1.0
LAMBDA_PHYSICS = 1e-3

# Contador de evaluaciones para monitoreo
eval_counter = {'count': 0, 'last_log': time.time()}


def fitness_physics_only(params_3d: np.ndarray) -> float:
    """
    Fitness con monitoreo activo de GPU.
    """
    global eval_counter
    eval_counter['count'] += 1
    
    # Log cada 5 evaluaciones
    if eval_counter['count'] % 5 == 0:
        torch.cuda.synchronize()
        mem_used = torch.cuda.memory_allocated(0) / 1e6
        elapsed = time.time() - eval_counter['last_log']
        print(f"  Eval {eval_counter['count']}: GPU Memory = {mem_used:.1f} MB | "
              f"Time/eval = {elapsed/5:.2f}s")
        eval_counter['last_log'] = time.time()
    
    # Actualizar parámetros físicos (en GPU)
    with torch.no_grad():
        model.log_k0.data = torch.tensor(params_3d[0], device=device, dtype=torch.float32)
        model.log_k.data = torch.tensor(params_3d[1], device=device, dtype=torch.float32)
        model.log_a.data = torch.tensor(params_3d[2], device=device, dtype=torch.float32)
    
    model.eval()
    
    # Loss de datos (sin autograd)
    with torch.no_grad():
        loss_data = loss_data_from_model(model, t_data_gpu, y_real_gpu, i_real_gpu)
    
    # Loss de física en mini-batches (con autograd)
    batch_size = 100  # Un poco más grande ahora
    n_points = phys_t.shape[0]
    physics_losses = []
    
    for i in range(0, n_points, batch_size):
        end_idx = min(i + batch_size, n_points)
        t_batch = phys_t[i:end_idx]
        u_batch = phys_u[i:end_idx]
        
        loss_phys_batch = loss_phys_from_model(model, t_batch, u_batch, m=m, g=g, R=R)
        physics_losses.append(loss_phys_batch.detach())  # Detach para liberar memoria
    
    loss_physics = torch.stack(physics_losses).mean()
    loss_total = LAMBDA_DATA * loss_data + LAMBDA_PHYSICS * loss_physics
    
    # Limpiar cache cada 10 evaluaciones
    if eval_counter['count'] % 10 == 0:
        torch.cuda.empty_cache()
    
    return loss_total.item()


def phase1_physics_search(
    maxiter: int = 20,    # REDUCIDO para prueba
    popsize: int = 10      # REDUCIDO
) -> tuple[np.ndarray, float]:
    """
    Phase 1: DE optimización de parámetros físicos.
    """
    print(f"\n{'='*80}")
    print("PHASE 1: DIFFERENTIAL EVOLUTION (GPU-Optimized)")
    print(f"{'='*80}")
    print(f"Configuración:")
    print(f"  - Iteraciones: {maxiter}")
    print(f"  - Población: {popsize}")
    print(f"  - Batch size: 100")
    print(f"  - Total evaluaciones: ~{maxiter * popsize}")
    print()
    
    # Bounds
    log_k0_init = np.log(36.3e-3)
    log_k_init = np.log(3.5e-3)
    log_a_init = np.log(5.2e-3)
    
    bounds = [
        (log_k0_init - 3.0, log_k0_init + 3.0),
        (log_k_init - 3.0, log_k_init + 3.0),
        (log_a_init - 3.0, log_a_init + 3.0)
    ]
    
    print("🚀 Iniciando búsqueda global...")
    start_time = time.time()
    
    result = differential_evolution(
        fitness_physics_only,
        bounds,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        disp=True,
        workers=1,
        polish=False
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Phase 1 completada en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Mejor loss: {result.fun:.6e}")
    print(f"   Evaluaciones: {result.nfev}")
    
    k0_best = np.exp(result.x[0])
    k_best = np.exp(result.x[1])
    a_best = np.exp(result.x[2])
    
    print(f"\n📊 Mejores parámetros físicos:")
    print(f"   k0 = {k0_best:.6e}")
    print(f"   k  = {k_best:.6e}")
    print(f"   a  = {a_best:.6e}")
    
    return result.x, result.fun


def phase2_full_refinement(
    best_physics: np.ndarray,
    num_epochs: int = 300,   # Reducido
    lr: float = 5e-5
) -> tuple[float, list[float]]:
    """
    Phase 2: Adam refinement (GPU-accelerated).
    """
    print(f"\n{'='*80}")
    print("PHASE 2: ADAM REFINEMENT (GPU-Accelerated)")
    print(f"{'='*80}")
    print(f"Configuración:")
    print(f"  - Épocas: {num_epochs}")
    print(f"  - Learning rate: {lr}")
    print()
    
    # Cargar mejores parámetros físicos
    with torch.no_grad():
        model.log_k0.data = torch.tensor(best_physics[0], device=device, dtype=torch.float32)
        model.log_k.data = torch.tensor(best_physics[1], device=device, dtype=torch.float32)
        model.log_a.data = torch.tensor(best_physics[2], device=device, dtype=torch.float32)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    start_time = time.time()
    
    print("🎯 Entrenando con Adam...")
    for epoch in range(num_epochs):
        # Limpiar cache cada 50 épocas
        if epoch % 50 == 0:
            torch.cuda.empty_cache()
        
        optimizer.zero_grad()
        
        # Data loss
        loss_data = loss_data_from_model(model, t_data_gpu, y_real_gpu, i_real_gpu)
        
        # Physics loss en mini-batches
        batch_size = 100
        n_points = phys_t.shape[0]
        physics_losses = []
        
        for i in range(0, n_points, batch_size):
            end_idx = min(i + batch_size, n_points)
            t_batch = phys_t[i:end_idx]
            u_batch = phys_u[i:end_idx]
            loss_phys_batch = loss_phys_from_model(model, t_batch, u_batch, m=m, g=g, R=R)
            physics_losses.append(loss_phys_batch)
        
        loss_physics = torch.stack(physics_losses).mean()
        loss_total = LAMBDA_DATA * loss_data + LAMBDA_PHYSICS * loss_physics
        
        loss_total.backward()
        optimizer.step()
        
        loss_value = loss_total.item()
        loss_history.append(loss_value)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            k0, k, a = model.get_phys_params()
            mem_used = torch.cuda.memory_allocated(0) / 1e6
            print(f"  Epoch {epoch+1:3d} | Loss: {loss_value:.6e} | "
                  f"GPU: {mem_used:.1f}MB | k0={k0.item():.3e}, k={k.item():.3e}, a={a.item():.3e}")
    
    elapsed = time.time() - start_time
    final_loss = loss_history[-1]
    
    print(f"\n✅ Phase 2 completada en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Loss final: {final_loss:.6e}")
    
    return final_loss, loss_history


def main():
    print(f"\n{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + "  KAN-PINN GPU-OPTIMIZED TRAINING".center(78) + "#")
    print("#" + "  (Verificación activa de uso de GPU)".center(78) + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}\n")
    
    save_dir = Path("runs_gpu_optimized")
    save_dir.mkdir(exist_ok=True)
    
    # Test rápido de GPU
    print("🧪 Test rápido de GPU...")
    test_tensor = torch.randn(1000, 1000, device=device)
    start = time.time()
    result = test_tensor @ test_tensor.T
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"   Matrix multiply (1000×1000): {gpu_time*1000:.2f}ms")
    print(f"   ✅ GPU funcionando correctamente\n")
    
    # Phase 1
    best_physics, loss_phase1 = phase1_physics_search(
        maxiter=20,  # Reducido para test
        popsize=10
    )
    
    np.save(save_dir / "phase1_best_physics.npy", best_physics)
    
    # Phase 2
    loss_final, adam_history = phase2_full_refinement(
        best_physics=best_physics,
        num_epochs=300,
        lr=5e-5
    )
    
    np.save(save_dir / "phase2_loss_history.npy", np.array(adam_history))
    
    # Guardar modelo
    k0_final, k_final, a_final = model.get_phys_params()
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'loss': loss_final,
        'physical_params': {
            'k0': k0_final.item(),
            'k': k_final.item(),
            'a': a_final.item()
        },
        'stats': {
            'y_mu': ds.stats.y_mu,
            'y_std': ds.stats.y_std,
            'i_mu': ds.stats.i_mu,
            'i_std': ds.stats.i_std
        },
        'architecture': {
            'hidden': 16,
            'depth': 2,
            'num_knots': 4
        }
    }
    
    torch.save(checkpoint, save_dir / "best_model_gpu.pt")
    
    # Reporte final de GPU
    torch.cuda.synchronize()
    final_mem = torch.cuda.memory_allocated(0) / 1e6
    max_mem = torch.cuda.max_memory_allocated(0) / 1e6
    
    print(f"\n{'='*80}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*80}")
    print(f"Loss final: {loss_final:.6e}")
    print(f"\nParámetros físicos finales:")
    print(f"  k0 = {k0_final.item():.6e}")
    print(f"  k  = {k_final.item():.6e}")
    print(f"  a  = {a_final.item():.6e}")
    print(f"\nUso de GPU:")
    print(f"  Memoria actual: {final_mem:.2f} MB")
    print(f"  Memoria pico: {max_mem:.2f} MB")
    print(f"\nArchivos guardados en: {save_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

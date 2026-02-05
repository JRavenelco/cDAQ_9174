"""
Visualizador de Arquitectura KAN (Kolmogorov-Arnold Network).
Crea un mapa/grafo mostrando todas las conexiones y funciones aprendidas.
"""
from __future__ import annotations

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from kan_model import KANLevitator


def visualize_kan_architecture(model: KANLevitator, save_path: str):
    """
    Crea una visualización completa de la arquitectura KAN.
    Muestra:
    - Topología de la red (nodos y conexiones)
    - Funciones B-spline aprendidas en cada edge
    - Parámetros físicos
    """
    
    # Get model info
    kan_net = model.kan_net
    num_layers = len(kan_net)
    
    print(f"\n{'='*80}")
    print("KAN NETWORK ARCHITECTURE")
    print(f"{'='*80}")
    print(f"Total layers: {num_layers}")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ============================================
    # SUBPLOT 1: Network Topology (Graph View)
    # ============================================
    ax_topology = plt.subplot(2, 3, (1, 4))
    ax_topology.set_title("KAN Topology: Learnable Activations on Edges", fontsize=14, fontweight='bold')
    ax_topology.axis('off')
    
    # Calculate layer positions
    layer_sizes = []
    layer_positions = []
    x_spacing = 3.0
    
    for i, layer in enumerate(kan_net):
        if hasattr(layer, 'in_features'):
            in_f = layer.in_features
            out_f = layer.out_features
            
            if i == 0:
                layer_sizes.append(in_f)
            layer_sizes.append(out_f)
    
    max_nodes = max(layer_sizes)
    
    # Draw nodes and edges
    for layer_idx in range(len(layer_sizes)):
        x = layer_idx * x_spacing
        n_nodes = layer_sizes[layer_idx]
        y_spacing = 8.0 / max(n_nodes, 1)
        y_offset = (max_nodes - n_nodes) * y_spacing / 2
        
        positions = []
        for node_idx in range(n_nodes):
            y = y_offset + node_idx * y_spacing
            positions.append((x, y))
            
            # Draw node
            circle = plt.Circle((x, y), 0.2, color='steelblue', ec='black', zorder=3)
            ax_topology.add_patch(circle)
            
            # Label
            if layer_idx == 0:
                label = 't'
            elif layer_idx == len(layer_sizes) - 1:
                label = 'y' if node_idx == 0 else 'i'
            else:
                label = f'h{node_idx}'
            
            ax_topology.text(x, y, label, ha='center', va='center', 
                           fontsize=10, fontweight='bold', color='white', zorder=4)
        
        layer_positions.append(positions)
        
        # Draw edges with learnable activations
        if layer_idx < len(layer_sizes) - 1:
            next_positions = []
            n_next = layer_sizes[layer_idx + 1]
            y_spacing_next = 8.0 / max(n_next, 1)
            y_offset_next = (max_nodes - n_next) * y_spacing_next / 2
            
            for node_idx in range(n_next):
                y_next = y_offset_next + node_idx * y_spacing_next
                next_positions.append((x + x_spacing, y_next))
            
            # Draw edges
            for i, (x1, y1) in enumerate(positions):
                for j, (x2, y2) in enumerate(next_positions):
                    # Edge with annotation
                    ax_topology.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=1, zorder=1)
                    
                    # Add small marker indicating learnable activation
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax_topology.plot(mid_x, mid_y, 'o', color='orange', markersize=3, 
                                   alpha=0.6, zorder=2)
    
    # Add layer labels
    for i, x in enumerate([i * x_spacing for i in range(len(layer_sizes))]):
        if i == 0:
            label_text = "Input\n(t)"
        elif i == len(layer_sizes) - 1:
            label_text = "Output\n(y, i)"
        else:
            label_text = f"Hidden {i}"
        
        ax_topology.text(x, -1, label_text, ha='center', va='top', 
                        fontsize=10, style='italic')
    
    ax_topology.set_xlim(-1, (len(layer_sizes) - 1) * x_spacing + 1)
    ax_topology.set_ylim(-2, 9)
    ax_topology.set_aspect('equal')
    
    # ============================================
    # SUBPLOT 2-4: Sample B-Spline Functions
    # ============================================
    x_eval = np.linspace(-1, 1, 200)
    x_tensor = torch.tensor(x_eval, dtype=torch.float32).reshape(-1, 1)
    
    # Sample 3 different edges and show their learned activations
    sample_count = 0
    for layer_idx, layer in enumerate(kan_net):
        if hasattr(layer, 'activations') and sample_count < 3:
            ax_spline = plt.subplot(2, 3, 2 + sample_count)
            
            # Get a few activation functions from this layer
            n_in = layer.in_features
            n_out = layer.out_features
            
            # Sample max 5 edges
            n_samples = min(5, n_in * n_out)
            
            for _ in range(n_samples):
                i = np.random.randint(0, n_in)
                j = np.random.randint(0, n_out)
                
                with torch.no_grad():
                    activation_func = layer.activations[i][j]
                    y_eval = activation_func(x_tensor).numpy().flatten()
                
                ax_spline.plot(x_eval, y_eval, alpha=0.7, linewidth=2)
            
            ax_spline.set_xlabel('Input')
            ax_spline.set_ylabel('Output')
            ax_spline.set_title(f'Layer {layer_idx}: Learned B-Splines (sample)', fontsize=10)
            ax_spline.grid(True, alpha=0.3)
            ax_spline.axhline(0, color='black', linewidth=0.5)
            ax_spline.axvline(0, color='black', linewidth=0.5)
            
            sample_count += 1
    
    # ============================================
    # SUBPLOT 5: Physical Parameters
    # ============================================
    ax_params = plt.subplot(2, 3, 5)
    ax_params.axis('off')
    
    k0, k, a = model.get_phys_params()
    
    param_text = "Physical Parameters (Learned)\n" + "="*35 + "\n\n"
    param_text += f"k₀ = {k0.item():.6e}\n"
    param_text += f"k  = {k.item():.6e}\n"
    param_text += f"a  = {a.item():.6e}\n\n"
    param_text += "Magnetic Inductance:\n"
    param_text += "L(y) = k₀ + k/(1 + y/a)\n\n"
    param_text += f"Total trainable params: {sum(p.numel() for p in model.parameters())}"
    
    ax_params.text(0.1, 0.5, param_text, fontsize=11, family='monospace',
                  verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # ============================================
    # SUBPLOT 6: Network Statistics
    # ============================================
    ax_stats = plt.subplot(2, 3, 6)
    ax_stats.axis('off')
    
    # Count parameters per layer
    stats_text = "Network Statistics\n" + "="*35 + "\n\n"
    stats_text += f"Architecture: {' → '.join(map(str, layer_sizes))}\n\n"
    
    total_params = 0
    for layer_idx, layer in enumerate(kan_net):
        if hasattr(layer, 'activations'):
            layer_params = sum(p.numel() for p in layer.parameters())
            total_params += layer_params
            stats_text += f"Layer {layer_idx}: {layer_params:,} params\n"
    
    stats_text += f"\nPhysical params: 3\n"
    stats_text += f"Total: {total_params + 3:,} parameters"
    
    ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Main title
    fig.suptitle("Kolmogorov-Arnold Network (KAN) Architecture\n"
                 "Each edge has a learnable B-spline activation function",
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Network visualization saved to: {save_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize KAN network architecture')
    parser.add_argument('--ckpt', type=str, default='runs_physics_safe/best_model_safe.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--save-path', type=str, default='runs_physics_safe/kan_architecture.png',
                       help='Where to save the visualization')
    args = parser.parse_args()
    
    print(f"\nLoading model from: {args.ckpt}")
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    arch = ckpt['architecture']
    stats = ckpt['stats']
    
    # Recreate model
    model = KANLevitator(
        y_mu=stats['y_mu'],
        y_std=stats['y_std'],
        i_mu=stats['i_mu'],
        i_std=stats['i_std'],
        hidden=arch['hidden'],
        depth=arch['depth'],
        num_knots=arch['num_knots']
    )
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Architecture: hidden={arch['hidden']}, depth={arch['depth']}, knots={arch['num_knots']}")
    
    # Visualize
    visualize_kan_architecture(model, args.save_path)


if __name__ == "__main__":
    main()

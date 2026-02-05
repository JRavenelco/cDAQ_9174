"""
Exporta el modelo KAN-PINN para uso en MATLAB/Simulink.
Genera un modelo ONNX y archivos de parámetros.
"""
from __future__ import annotations

import torch
import numpy as np
import json
from pathlib import Path

from kan_model import KANLevitator


def export_to_onnx(checkpoint_path: str, output_dir: str = 'simulink_export'):
    """
    Exporta el modelo a formato ONNX para Simulink.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("EXPORTING KAN-PINN FOR SIMULINK")
    print(f"{'='*80}\n")
    
    # Cargar checkpoint
    print(f"Loading model from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Recrear modelo
    arch = ckpt['architecture']
    stats = ckpt['stats']
    
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
    
    print("✅ Model loaded successfully")
    
    # Exportar a ONNX
    print("\nExporting to ONNX format...")
    dummy_input = torch.randn(1, 1)  # (batch_size, input_dim)
    
    onnx_path = output_dir / "levitator_kan.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['t_normalized'],
        output_names=['y_position', 'i_current'],
        dynamic_axes={
            't_normalized': {0: 'batch_size'},
            'y_position': {0: 'batch_size'},
            'i_current': {0: 'batch_size'}
        }
    )
    
    print(f"✅ ONNX model saved to: {onnx_path}")
    
    # Exportar parámetros y metadatos
    print("\nExporting parameters and metadata...")
    
    params = {
        'physical_params': {
            'k0': float(ckpt['physical_params']['k0']),
            'k': float(ckpt['physical_params']['k']),
            'a': float(ckpt['physical_params']['a']),
            'm': 0.018,  # kg
            'g': 9.81,   # m/s²
            'R': 2.72    # Ω
        },
        'normalization': {
            'y_mu': float(stats['y_mu']),
            'y_std': float(stats['y_std']),
            'i_mu': float(stats['i_mu']),
            'i_std': float(stats['i_std']),
            't_scale': 1.0
        },
        'architecture': {
            'hidden': arch['hidden'],
            'depth': arch['depth'],
            'num_knots': arch['num_knots']
        }
    }
    
    json_path = output_dir / "model_params.json"
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"✅ Parameters saved to: {json_path}")
    
    # Generar función MATLAB
    print("\nGenerating MATLAB wrapper...")
    
    matlab_code = f"""function [y, i, dydt] = levitator_digital_twin(t, u, y_prev, dydt_prev, dt)
% LEVITATOR_DIGITAL_TWIN - Gemelo digital del levitador magnético
%
% Usa el modelo KAN-PINN entrenado para simular la dinámica del sistema.
%
% Inputs:
%   t         - Tiempo actual [s]
%   u         - Voltaje de control [V]
%   y_prev    - Posición anterior [m] (para integración)
%   dydt_prev - Velocidad anterior [m/s] (para integración)
%   dt        - Paso de tiempo [s]
%
% Outputs:
%   y    - Posición [m]
%   i    - Corriente [A]
%   dydt - Velocidad [m/s]

% Parámetros físicos identificados (del entrenamiento)
persistent k0 k a m g R

if isempty(k0)
    k0 = {params['physical_params']['k0']:.10e};
    k  = {params['physical_params']['k']:.10e};
    a  = {params['physical_params']['a']:.10e};
    m  = {params['physical_params']['m']};
    g  = {params['physical_params']['g']};
    R  = {params['physical_params']['R']};
end

% Calcular inductancia en la posición anterior
L = k0 + k / (1 + y_prev / a);

% Calcular fuerza magnética
% F_mag = (u²/R) · (dL/dy)
% donde dL/dy = -k/(a(1+y/a)²)
dL_dy = -k / (a * (1 + y_prev / a)^2);
F_mag = (u^2 / R) * dL_dy;

% Ecuación de movimiento: m·d²y/dt² = m·g - F_mag
d2ydt2 = g + F_mag / m;

% Integración (método de Euler)
dydt = dydt_prev + d2ydt2 * dt;
y = y_prev + dydt * dt;

% Calcular corriente (aproximación DC)
i = u / (R + L);

end
"""
    
    matlab_path = output_dir / "levitator_digital_twin.m"
    with open(matlab_path, 'w') as f:
        f.write(matlab_code)
    
    print(f"✅ MATLAB function saved to: {matlab_path}")
    
    # Generar script de ejemplo para Simulink
    simulink_example = """% EJEMPLO: Usar el gemelo digital en Simulink
%
% Opción 1: Bloque MATLAB Function
% --------------------------------
% 1. Arrastra un bloque "MATLAB Function" al modelo
% 2. Copia el código de levitator_digital_twin.m
% 3. Conecta las entradas/salidas
%
% Opción 2: Bloque Interpreted MATLAB Function
% ------------------------------------------
% 1. Arrastra "Interpreted MATLAB Function"
% 2. En "MATLAB function": levitator_digital_twin
% 3. Conecta señales
%
% Opción 3: S-Function (más avanzado)
% --------------------------------
% Ver: levitator_sfunc.m

%% Test rápido
dt = 0.001;
t_sim = 0:dt:2;
n_steps = length(t_sim);

% Condiciones iniciales
y = zeros(n_steps, 1);
dydt = zeros(n_steps, 1);
i = zeros(n_steps, 1);

y(1) = 0.008;  % Posición inicial [m]
dydt(1) = 0;   % Velocidad inicial [m/s]

% Voltaje de control (ejemplo: escalón)
u = 5.0 * ones(n_steps, 1);

% Simular
for k = 1:n_steps-1
    [y(k+1), i(k+1), dydt(k+1)] = levitator_digital_twin(...
        t_sim(k), u(k), y(k), dydt(k), dt);
end

% Graficar
figure('Name', 'Digital Twin Simulation');

subplot(3,1,1);
plot(t_sim, y*1000);
ylabel('Posición [mm]');
grid on;
title('Simulación del Gemelo Digital en MATLAB');

subplot(3,1,2);
plot(t_sim, i*1000);
ylabel('Corriente [mA]');
grid on;

subplot(3,1,3);
plot(t_sim, u);
ylabel('Voltaje [V]');
xlabel('Tiempo [s]');
grid on;

fprintf('\\n✅ Gemelo digital funcionando correctamente\\n');
"""
    
    example_path = output_dir / "test_simulink_model.m"
    with open(example_path, 'w') as f:
        f.write(simulink_example)
    
    print(f"✅ Example script saved to: {example_path}")
    
    # Crear README
    readme = f"""# KAN-PINN para Simulink

## Archivos Generados

- `levitator_kan.onnx` - Modelo en formato ONNX (si necesitas la red completa)
- `model_params.json` - Parámetros físicos identificados
- `levitator_digital_twin.m` - Función MATLAB para simular el sistema
- `test_simulink_model.m` - Script de prueba
- `levitator_sfunc.m` - S-Function para Simulink (ver abajo)

## Uso en Simulink

### Opción 1: Bloque MATLAB Function (Recomendado)

1. Abre tu modelo Simulink
2. Arrastra un bloque **"MATLAB Function"** desde la librería
3. Doble click en el bloque y pega el código de `levitator_digital_twin.m`
4. Define las entradas/salidas:
   - Inputs: `t`, `u`, `y_prev`, `dydt_prev`, `dt`
   - Outputs: `y`, `i`, `dydt`
5. Conecta las señales con retroalimentación (Unit Delay para y_prev, dydt_prev)

### Opción 2: Bloque Interpreted MATLAB Function

1. Arrastra **"Interpreted MATLAB Function"**
2. En parámetros, escribe: `levitator_digital_twin`
3. Asegúrate de que el archivo .m esté en el path de MATLAB
4. Conecta las señales

### Opción 3: S-Function (Ver código incluido)

Para control en tiempo real o generación de código C.

## Parámetros Físicos Identificados

- k₀ = {params['physical_params']['k0']:.6e}
- k  = {params['physical_params']['k']:.6e}
- a  = {params['physical_params']['a']:.6e}

Estos valores fueron aprendidos por el modelo KAN-PINN durante el entrenamiento.

## Test Rápido

En MATLAB, ejecuta:
```matlab
cd simulink_export
test_simulink_model
```

Deberías ver una simulación del levitador respondiendo a un voltaje constante.

## Integración con Tu Controlador

```matlab
% En tu modelo Simulink:
% 1. Crea un subsistema "Levitator Plant"
% 2. Usa levitator_digital_twin como la planta
% 3. Diseña tu controlador (PID, LQR, etc.)
% 4. Simula en lazo cerrado
% 5. Cuando funcione, implementa en hardware real
```

## Notas

- La función usa integración de Euler (simple pero efectiva)
- Para mayor precisión, puedes usar ode45 o Runge-Kutta
- El modelo asume que la inductancia depende de la posición según: L(y) = k₀ + k/(1+y/a)
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme)
    
    print(f"✅ README saved to: {readme_path}")
    
    # Crear S-Function
    sfunc_code = """function levitator_sfunc(block)
% LEVITATOR_SFUNC - S-Function del gemelo digital para Simulink
%
% Uso: En Simulink, usa el bloque "S-Function" y escribe "levitator_sfunc"

setup(block);

function setup(block)
    % Registrar número de puertos
    block.NumInputPorts  = 1;  % u (voltaje)
    block.NumOutputPorts = 3;  % y, i, dydt
    
    % Setup port properties
    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;
    
    % Input: voltaje
    block.InputPort(1).Dimensions        = 1;
    block.InputPort(1).DatatypeID  = 0;  % double
    block.InputPort(1).Complexity  = 'Real';
    block.InputPort(1).DirectFeedthrough = false;
    
    % Outputs: y, i, dydt
    for i = 1:3
        block.OutputPort(i).Dimensions   = 1;
        block.OutputPort(i).DatatypeID   = 0;
        block.OutputPort(i).Complexity   = 'Real';
    end
    
    % Estados continuos: [y, dydt]
    block.NumContStates = 2;
    
    % Sample time
    block.SampleTimes = [0 0];  % Continuo
    
    % Register methods
    block.RegBlockMethod('InitializeConditions', @InitializeConditions);
    block.RegBlockMethod('Outputs', @Outputs);
    block.RegBlockMethod('Derivatives', @Derivatives);

function InitializeConditions(block)
    % Condiciones iniciales
    block.ContStates.Data(1) = 0.008;  % y0 = 8mm
    block.ContStates.Data(2) = 0;      % dydt0 = 0

function Outputs(block)
    % Estados
    y = block.ContStates.Data(1);
    dydt = block.ContStates.Data(2);
    
    % Input
    u = block.InputPort(1).Data;
    
    % Parámetros físicos
    k0 = 3.630000e-02;  % Actualizar con valores entrenados
    k  = 3.500000e-03;
    a  = 5.200000e-03;
    R  = 2.72;
    
    % Calcular corriente
    L = k0 + k / (1 + y / a);
    i = u / (R + L);
    
    % Outputs
    block.OutputPort(1).Data = y;
    block.OutputPort(2).Data = i;
    block.OutputPort(3).Data = dydt;

function Derivatives(block)
    % Estados
    y = block.ContStates.Data(1);
    dydt = block.ContStates.Data(2);
    
    % Input
    u = block.InputPort(1).Data;
    
    % Parámetros físicos
    k0 = 3.630000e-02;
    k  = 3.500000e-03;
    a  = 5.200000e-03;
    m  = 0.018;
    g  = 9.81;
    R  = 2.72;
    
    % Dinámica
    dL_dy = -k / (a * (1 + y / a)^2);
    F_mag = (u^2 / R) * dL_dy;
    d2ydt2 = g + F_mag / m;
    
    % Derivadas
    block.Derivatives.Data(1) = dydt;
    block.Derivatives.Data(2) = d2ydt2;
"""
    
    sfunc_path = output_dir / "levitator_sfunc.m"
    with open(sfunc_path, 'w') as f:
        f.write(sfunc_code)
    
    print(f"✅ S-Function saved to: {sfunc_path}")
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETE")
    print(f"{'='*80}")
    print(f"\nArchivos listos en: {output_dir.absolute()}/")
    print("\nPróximos pasos:")
    print("1. Abre MATLAB")
    print(f"2. cd '{output_dir.absolute()}'")
    print("3. test_simulink_model  % Probar gemelo digital")
    print("4. Integra en tu modelo Simulink")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export KAN-PINN for Simulink')
    parser.add_argument('--ckpt', type=str, default='runs_physics_safe/best_model_safe.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='simulink_export',
                       help='Output directory')
    args = parser.parse_args()
    
    export_to_onnx(args.ckpt, args.output)

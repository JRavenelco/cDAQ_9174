function out = ajustar_kan_pinn_envolvente(t, fuerza, aceleracion, opts)
% AJUSTAR_KAN_PINN_ENVOLVENTE  Capa PROFUNDA (opcion "dos capas").
%
% Ajusta el modelo fisico KAN-PINN/Bouc-Wen con ENVOLVENTE a una VENTANA de
% corte y devuelve sus parametros como caracterizacion del caso. Es el port a
% MATLAB de scripts_modelos/test_kan_pinn_simple.py (genera la grafica
% graficas_bouc_wen/23_kan_pinn_con_envolvente.png).
%
% MODELO:
%   F = m*a + k*x + c*v + alpha*k*E + (1-alpha)*k*z
%   dz/dt = v*(A - |z|^n*(B*sign(v*z) + C))            (Bouc-Wen)
% con:
%   E    = envolvente de aceleracion (Hilbert + LP Butterworth orden 2, fc=15Hz)
%   v,x  = integrales filtradas (HP Butterworth orden 2, fc=5Hz) de a
%   z    = variable interna de histeresis
%
% alpha in [0,1] es el DESCRIPTOR de histeresis (alpha=1 -> lineal/envolvente;
% alpha<1 -> histeresis) y R2 la bondad de ajuste. Se ajusta el vector
% [m,c,k,alpha,A,B,C,n] minimizando el MSE con Evolucion Diferencial (DE) en
% MATLAB puro (sin Global Optimization Toolbox), equivalente a
% scipy.optimize.differential_evolution.
%
% Pensado para una VENTANA (~0.5 s); si la senal es larga se submuestrea para
% acotar el coste de la optimizacion. Se ejecuta en Interpreted execution
% (capa lenta/offline); no es para el retrieve en tiempo real.
%
% ENTRADAS
%   t, fuerza, aceleracion  vectores de la ventana (igual longitud)
%   opts (struct opcional):
%     .fc_env(15) .fc_int(5) .max_muestras(250) .maxiter(80) .popsize(30) .seed(42)
%
% SALIDA (struct out)
%   .params   [m c k alpha A B C n]
%   .m .c .k .alpha .A .B .C .n
%   .R2       bondad de ajuste (sobre fuerza normalizada)
%   .fs_sub   frecuencia tras submuestreo
%   .F_real .F_model .z .E   senales (normalizadas, submuestreadas)
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    if nargin < 4; opts = struct(); end
    fc_env       = getf(opts,'fc_env',15);
    fc_int       = getf(opts,'fc_int',5);
    max_muestras = getf(opts,'max_muestras',250);
    maxiter      = getf(opts,'maxiter',80);
    popsize      = getf(opts,'popsize',30);
    seed         = getf(opts,'seed',42);

    t = double(t(:)); fuerza = double(fuerza(:)); aceleracion = double(aceleracion(:));
    n0 = min([numel(t), numel(fuerza), numel(aceleracion)]);
    t = t(1:n0); fuerza = fuerza(1:n0); aceleracion = aceleracion(1:n0);
    v = isfinite(t) & isfinite(fuerza) & isfinite(aceleracion);
    t = t(v); fuerza = fuerza(v); aceleracion = aceleracion(v);
    if numel(t) < 16; error('ajustar_kan_pinn_envolvente: pocas muestras (%d).', numel(t)); end

    t = t - t(1);
    fs = infer_fs(t);

    % ── Envolvente sobre la senal CRUDA, antes de submuestrear ──────────────
    E_full = calcular_envolvente(aceleracion, fs, fc_env);

    % ── Submuestreo para acotar el coste de la DE (como el step de Python) ──
    step = max(1, round(numel(t) / max_muestras));
    idx  = 1:step:numel(t);
    fs_sub = fs / step;

    a = normalizar(aceleracion(idx));
    F = normalizar(fuerza(idx));
    E = normalizar(E_full(idx));
    [vel, pos] = integrar(a, fs_sub, fc_int);
    vel = normalizar(vel); pos = normalizar(pos);

    % ── Optimizacion DE de [m c k alpha A B C n] ────────────────────────────
    lb = [0.01 0.0 0.01 0.0 0.1 0.0 0.0 0.5];
    ub = [2.0  2.0 2.0  1.0 5.0 2.0 2.0 3.0];
    loss = @(p) mse_modelo(p, a, vel, pos, E, F, fs_sub);
    [best, ~] = de_min(loss, lb, ub, popsize, maxiter, seed, 0.7, 0.9);

    [F_model, z] = modelo_kan_pinn(a, vel, pos, E, best, fs_sub);
    R2 = 1 - sum((F - F_model).^2) / max(sum((F - mean(F)).^2), eps);

    out = struct();
    out.params = best;
    out.m = best(1); out.c = best(2); out.k = best(3); out.alpha = best(4);
    out.A = best(5); out.B = best(6); out.C = best(7); out.n = best(8);
    out.R2 = R2;
    out.fs_sub = fs_sub;
    out.F_real = F; out.F_model = F_model; out.z = z; out.E = E;
end


% ===========================================================================
% Modelo y perdida
% ===========================================================================
function [F, z] = modelo_kan_pinn(a, v, x, E, p, fs)
    m=p(1); c=p(2); k=p(3); alpha=p(4); A=p(5); B=p(6); C=p(7); n=p(8);
    dt = 1.0/fs;
    z = zeros(size(x));
    for i = 2:numel(x)
        sgn = sign(v(i)*z(i-1) + 1e-10);
        az  = abs(z(i-1)) + 1e-8;
        dz  = v(i) * (A - (az^n)*(B*sgn + C));
        z(i) = min(max(z(i-1) + dz*dt, -1), 1);
    end
    F = m*a + k*x + c*v + alpha*k*E + (1-alpha)*k*z;
end

function e = mse_modelo(p, a, v, x, E, F_real, fs)
    try
        [Fm, ~] = modelo_kan_pinn(a, v, x, E, p, fs);
        e = mean((F_real - Fm).^2);
        if ~isfinite(e); e = 1e10; end
    catch
        e = 1e10;
    end
end

% ===========================================================================
% Evolucion Diferencial (rand/1/bin) en MATLAB puro
% ===========================================================================
function [best, fbest] = de_min(loss, lb, ub, popsize, maxiter, seed, F, CR)
    rng(seed);
    D = numel(lb); lb = lb(:)'; ub = ub(:)';
    P = lb + rand(popsize, D) .* (ub - lb);
    fit = zeros(popsize, 1);
    for i = 1:popsize; fit(i) = loss(P(i,:)); end
    for it = 1:maxiter
        for i = 1:popsize
            r = randperm(popsize); r(r==i) = []; r = r(1:3);
            mut = P(r(1),:) + F*(P(r(2),:) - P(r(3),:));
            mut = min(max(mut, lb), ub);
            cross = rand(1, D) < CR; cross(randi(D)) = true;
            trial = P(i,:); trial(cross) = mut(cross);
            ft = loss(trial);
            if ft < fit(i); P(i,:) = trial; fit(i) = ft; end
        end
    end
    [fbest, bi] = min(fit); best = P(bi,:);
end

% ===========================================================================
% Helpers de senal (port de test_kan_pinn_simple.py)
% ===========================================================================
function E = calcular_envolvente(senal, fs, fc)
    ac = senal - mean(senal, 'omitnan');
    env_raw = abs(hilbert(ac));
    if isfinite(fs) && fc > 0 && fc < fs/2
        [b, a] = butter(2, fc/(fs/2), 'low');
        E = filtfilt(b, a, env_raw);
    else
        E = env_raw;
    end
    E = max(E, 0);
end

function [vel, pos] = integrar(a, fs, fc)
    dt = 1.0/fs;
    [b, af] = butter(2, fc/(fs/2), 'high');
    vel = filtfilt(b, af, cumtrapz(a)*dt);
    pos = filtfilt(b, af, cumtrapz(vel)*dt);
end

function y = normalizar(x)
    c = x - mean(x, 'omitnan'); s = max(abs(c)) + 1e-10;
    y = c / s;
end

function fs = infer_fs(t)
    if numel(t) < 3; fs = 2500; return; end
    dt = diff(t); dt = dt(isfinite(dt) & dt > 0);
    if isempty(dt); fs = 2500; else; fs = 1.0/median(dt); end
end

function val = getf(s, name, def)
    if isfield(s, name) && ~isempty(s.(name)); val = s.(name); else; val = def; end
end

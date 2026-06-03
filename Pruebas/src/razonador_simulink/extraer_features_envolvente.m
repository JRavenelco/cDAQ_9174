function [features, info] = extraer_features_envolvente(t, fuerza, aceleracion, fc_env)
% EXTRAER_FEATURES_ENVOLVENTE  Capa rapida F-E (opcion "dos capas").
%
% Gemelo CORREGIDO de extraer_features_ventana.m: calcula los descriptores de
% histeresis sobre la ENVOLVENTE de la aceleracion (metodologia de la tesis),
% NO sobre la aceleracion cruda. Replica migrar_graficas_envolvente_matlab.m
% y calcular_envolvente() de scripts_modelos/test_kan_pinn_simple.py:
%
%   E            = LP_Butterworth4(fc=15Hz) de |Hilbert(a - mean(a))|
%   loop_area_FE = shoelace(env_norm, f_center)      (area del lazo F-E)
%   corr_FE      = corr(fuerza, E)                    (la tesis: corr(F,E) > corr(F,A))
%
% El resto de features (RMS/pico de fuerza y de aceleracion, duracion) se
% mantienen igual que en el razonador. El vector resultante conserva el ORDEN
% FIJO del CBR, pero con la histeresis bien calculada:
%   [force_rms, force_peak_abs, input_rms, input_peak_abs,
%    corr_FE, loop_area_FE, duration_s]
%
% MOTIVO: loop_area/corr sobre la aceleracion CRUDA anti-correlacionan con la
% verdad F-E (Spearman -0.47, 14/19 casos cambian de clase). Ver memoria del
% proyecto "modelo-histeresis-envolvente".
%
% Requiere Signal Processing Toolbox (hilbert, butter, filtfilt).
% Se ejecuta en Interpreted execution (no codegen): es la CAPA LENTA de la
% arquitectura; el retrieve rapido solo recibe el vector ya calculado.
%
% ENTRADAS
%   t, fuerza, aceleracion  vectores de igual longitud (senal del corte)
%   fc_env                  frecuencia de corte del LP de la envolvente (def 15 Hz)
%
% SALIDAS
%   features [1x7]  vector en el orden fijo (con corr_FE y loop_area_FE)
%   info     struct con fs, envolvente E, env_norm, f_center, loop_area_cruda
%            (para comparar) y corr_cruda.
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    if nargin < 4 || isempty(fc_env); fc_env = 15; end

    t = double(t(:)); fuerza = double(fuerza(:)); aceleracion = double(aceleracion(:));
    n = min([numel(t), numel(fuerza), numel(aceleracion)]);
    t = t(1:n); fuerza = fuerza(1:n); aceleracion = aceleracion(1:n);
    valid = isfinite(t) & isfinite(fuerza) & isfinite(aceleracion);
    t = t(valid); fuerza = fuerza(valid); aceleracion = aceleracion(valid);
    if numel(t) < 16
        error('extraer_features_envolvente: muy pocas muestras validas (%d).', numel(t));
    end

    t = t - t(1);
    fs = infer_fs(t);

    % ── Envolvente de la aceleracion (Hilbert + LP Butterworth 4o, fc=15Hz) ──
    E = calcular_envolvente(aceleracion, fs, fc_env);

    % ── Features de NIVEL (sobre senal cruda, sin cambios) ──────────────────
    force_rms      = rms_(fuerza);
    force_peak_abs = max(abs(fuerza));
    input_rms      = rms_(aceleracion);
    input_peak_abs = max(abs(aceleracion));
    duration_s     = t(end) - t(1);

    % ── Features de HISTERESIS (sobre la ENVOLVENTE: F-E) ───────────────────
    f_center = fuerza - mean(fuerza, 'omitnan');
    env_norm = norm_sig(E);

    % corr(F, E): correlacion fuerza-envolvente (la tesis muestra |corr_FE|>|corr_FA|)
    if numel(E) > 2
        c = corrcoef(fuerza, E);
        corr_FE = c(1, 2);
    else
        corr_FE = NaN;
    end
    if ~isfinite(corr_FE); corr_FE = 0.0; end

    % area del lazo F-E por formula del poligono (shoelace), como en graficas_envolvente
    loop_area_FE = shoelace(env_norm, f_center);

    features = [force_rms, force_peak_abs, input_rms, input_peak_abs, ...
                corr_FE, loop_area_FE, duration_s];

    % ── info: envolvente + comparativa con el metodo crudo ──────────────────
    info = struct();
    info.fs           = fs;
    info.n_validos    = numel(t);
    info.E            = E;
    info.env_norm     = env_norm;
    info.f_center     = f_center;
    info.corr_FE      = corr_FE;
    info.loop_area_FE = loop_area_FE;
    % Metodo crudo (para comparar lado a lado con el CBR actual)
    a_norm = norm_sig(aceleracion);
    info.corr_cruda      = corr_safe(fuerza, aceleracion);
    info.loop_area_cruda = loop_area_cruda(a_norm, norm_sig(fuerza));
end


% ===========================================================================
% Envolvente: Hilbert + LP Butterworth 4o orden (migrar_graficas_envolvente)
% ===========================================================================
function E = calcular_envolvente(senal, fs, fc)
    ac = senal - mean(senal, 'omitnan');
    env_raw = abs(hilbert(ac));
    if isfinite(fs) && fc > 0 && fc < fs/2
        [b, a] = butter(4, fc/(fs/2), 'low');
        E = filtfilt(b, a, env_raw);
    else
        E = env_raw;
    end
    E = max(E, 0);
end

% ===========================================================================
% Helpers
% ===========================================================================
function y = norm_sig(x)
    c = x - mean(x, 'omitnan'); s = max(abs(c));
    if ~isfinite(s) || s <= 1e-12; y = zeros(size(c)); else; y = c / s; end
end

function area = shoelace(x, y)
    x = x(:); y = y(:); n = min(numel(x), numel(y));
    if n < 3; area = NaN; return; end
    x = x(1:n); y = y(1:n);
    area = 0.5 * abs(sum(x(1:end-1).*y(2:end) - x(2:end).*y(1:end-1)));
end

function a = loop_area_cruda(input_norm, force_norm)
    if numel(input_norm) < 3; a = NaN; return; end
    dx = gradient(input_norm);
    area = abs(sum(force_norm .* dx, 'omitnan'));
    rect = (max(input_norm)-min(input_norm)) * (max(force_norm)-min(force_norm));
    if ~isfinite(rect) || rect <= 1e-12; a = 0.0; else; a = area / rect; end
end

function r = corr_safe(x, y)
    x = x(:); y = y(:); m = isfinite(x) & isfinite(y); x = x(m); y = y(m);
    if numel(x) < 3 || std(x) <= 0 || std(y) <= 0; r = 0.0; return; end
    C = corrcoef(x, y); r = C(1, 2);
end

function fs = infer_fs(t)
    if numel(t) < 3; fs = 2500; return; end
    dt = diff(t); dt = dt(isfinite(dt) & dt > 0);
    if isempty(dt); fs = 2500; else; fs = 1.0 / median(dt); end
end

function r = rms_(x)
    r = sqrt(mean(double(x(:)).^2, 'omitnan'));
end

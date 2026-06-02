function [features, info] = extraer_features_ventana(t, fuerza, entrada, window_seconds)
% EXTRAER_FEATURES_VENTANA  Port de extract_features de construir_casos.py.
%
% Calcula el vector de 7 features (orden FIJO del razonador) a partir de una
% senal cruda de corte (tiempo, fuerza, entrada/aceleracion). Replica
% EXACTAMENTE construir_casos.py:
%   - normalize_signal: (x - mean) / max(|x - mean|)   (NO z-score)
%   - corr y loop_area_norm sobre la senal NORMALIZADA COMPLETA
%   - force_rms/peak, input_rms/peak sobre la senal CRUDA COMPLETA
%   - duration_s = t(end) - t(1)
%   (La ventana de energia rodante se reporta en info, pero las features
%    replican el calculo de Python que usa la senal completa.)
%
% Se ejecuta en Interpreted execution (lectura de senal). Marcar con
% coder.extrinsic si se invoca desde un bloque codegen.
%
% ENTRADAS
%   t, fuerza, entrada  vectores columna/fila de igual longitud
%   window_seconds      duracion de la ventana de energia (def 0.5)
%
% SALIDAS
%   features [1x7] = [force_rms, force_peak_abs, input_rms, input_peak_abs,
%                     corr_force_input, loop_area_norm, duration_s]
%   info     struct con fs, window_start_s, window_end_s, n_validos
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    if nargin < 4 || isempty(window_seconds); window_seconds = 0.5; end

    t = double(t(:)); fuerza = double(fuerza(:)); entrada = double(entrada(:));

    % ── Validos y recorte a longitud comun ─────────────────────────────────
    n = min([numel(t), numel(fuerza), numel(entrada)]);
    t = t(1:n); fuerza = fuerza(1:n); entrada = entrada(1:n);
    valid = isfinite(t) & isfinite(fuerza) & isfinite(entrada);
    t = t(valid); fuerza = fuerza(valid); entrada = entrada(valid);
    if numel(t) < 16
        error('extraer_features_ventana: muy pocas muestras validas (%d).', numel(t));
    end

    t = t - t(1);
    fs = infer_fs(t);

    force_norm = normalize_signal(fuerza);
    input_norm = normalize_signal(entrada);

    % ── Features sobre senal completa (como en construir_casos.py) ─────────
    force_rms      = rms_(fuerza);
    force_peak_abs = max(abs(fuerza));
    input_rms      = rms_(entrada);
    input_peak_abs = max(abs(entrada));

    if numel(force_norm) > 2
        c = corrcoef(force_norm, input_norm);
        corr_fi = c(1, 2);
    else
        corr_fi = NaN;
    end
    if ~isfinite(corr_fi); corr_fi = 0.0; end

    loop_area_norm = loop_area(input_norm, force_norm);
    duration_s = t(end) - t(1);

    features = [force_rms, force_peak_abs, input_rms, input_peak_abs, ...
                corr_fi, loop_area_norm, duration_s];

    % ── Ventana de energia rodante (reporte/contexto) ───────────────────────
    [start_i, end_i] = rolling_energy_window(force_norm, fs, window_seconds);
    info = struct();
    info.fs = fs;
    info.n_validos = numel(t);
    if end_i > start_i
        info.window_start_s = t(min(start_i+1, numel(t)));
        info.window_end_s   = t(min(end_i, numel(t)));
    else
        info.window_start_s = 0;
        info.window_end_s   = duration_s;
    end
end


% ===========================================================================
% Helpers (port 1:1 de construir_casos.py)
% ===========================================================================
function y = normalize_signal(x)
    x = double(x(:));
    centered = x - mean(x, 'omitnan');
    scale = max(abs(centered), [], 'omitnan');
    if ~isfinite(scale) || scale <= 1e-12
        y = zeros(size(centered));
    else
        y = centered / scale;
    end
end

function fs = infer_fs(t)
    if numel(t) < 3; fs = NaN; return; end
    dt = diff(t);
    dt = dt(isfinite(dt) & dt > 0);
    if isempty(dt); fs = NaN; return; end
    fs = 1.0 / median(dt);
end

function a = loop_area(input_norm, force_norm)
    if numel(input_norm) < 3; a = NaN; return; end
    dx = gradient(input_norm);
    area = abs(sum(force_norm .* dx, 'omitnan'));
    rect = (max(input_norm) - min(input_norm)) * (max(force_norm) - min(force_norm));
    if ~isfinite(rect) || rect <= 1e-12; a = 0.0; return; end
    a = area / rect;
end

function r = rms_(x)
    r = sqrt(mean(double(x(:)).^2, 'omitnan'));
end

function [start_i, end_i] = rolling_energy_window(force_norm, fs, seconds)
    n = numel(force_norm);
    if n == 0; start_i = 0; end_i = 0; return; end
    if ~isfinite(fs) || fs <= 0; fs = 2500.0; end
    window = max(16, round(seconds * fs));
    if n <= window; start_i = 0; end_i = n; return; end
    kernel = ones(window, 1) / window;
    energy = conv(force_norm.^2, kernel, 'valid');
    [~, imax] = max(energy);
    start_i = imax - 1;            % indice base-0 como en Python
    end_i = start_i + window;
end

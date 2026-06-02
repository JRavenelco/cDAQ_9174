function [class_id, min_dist, best_score, best_idx] = cbr_reasoner(features, threshold)
%#codegen
% CBR_REASONER  Parte 2: bloque MATLAB Function para Simulink.
%
% Razonador Basado en Casos (retrieval) que replica razonador_local.py:
%   - Normalizacion robusta por feature: (x - centro) / escala.
%   - Distancia euclidiana ponderada:  dist = norm((q-c).*w) / sqrt(D).
%   - Similitud:                        score = 1 / (1 + dist).
%
% La base de casos se embebe en tiempo de COMPILACION con coder.load, por lo
% que este bloque es apto para generacion de codigo C/C++ (MATLAB Coder) y
% ejecucion en tiempo real / HIL.
%
% ENTRADAS
%   features  [1 x 7] vector de features del caso actual, en este ORDEN:
%             [force_rms, force_peak_abs, input_rms, input_peak_abs,
%              corr_force_input, loop_area_norm, duration_s]
%   threshold (escalar) distancia maxima admitida. Si la mejor distancia es
%             >= threshold, el caso se RECHAZA (class_id = 0).
%             Usa un valor grande (p.ej. 1e6) para desactivar el rechazo.
%
% SALIDAS
%   class_id    etiqueta de histeresis del caso mas cercano:
%               1=lineal, 2=moderada, 3=marcada; 0=rechazado (fuera de umbral)
%   min_dist    distancia ponderada al caso mas cercano
%   best_score  similitud 1/(1+min_dist) del caso mas cercano
%   best_idx    indice (fila) del caso recuperado en la base; 0 si rechazado
%
% Genera primero la base con:  >> exportar_base_casos
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    % ── Cargar base de casos de forma estatica (una sola vez) ──────────────
    persistent S
    if isempty(S)
        S = coder.load('datos/base_de_casos_cbr.mat');
    end

    case_base   = S.case_base;      % [N x 7]
    feat_center = S.feat_center;    % [1 x 7]
    feat_scale  = S.feat_scale;     % [1 x 7]
    feat_weights= S.feat_weights;   % [1 x 7]
    hist_label  = S.hist_label;     % [N x 1]

    num_cases = size(case_base, 1);
    D = size(case_base, 2);

    % ── Normalizar el query (robust scaling) ───────────────────────────────
    q_norm = (features - feat_center) ./ feat_scale;   % [1 x 7]

    % ── Busqueda del caso mas cercano (Retrieve) ───────────────────────────
    min_dist = inf;
    best_idx = 0;
    for i = 1:num_cases
        c_norm = (case_base(i, :) - feat_center) ./ feat_scale;
        diff_vec = (q_norm - c_norm) .* feat_weights;
        dist = norm(diff_vec) / sqrt(D);
        if dist < min_dist
            min_dist = dist;
            best_idx = i;
        end
    end

    % ── Similitud y etiqueta ────────────────────────────────────────────────
    best_score = 1 / (1 + min_dist);

    if best_idx >= 1
        class_id = hist_label(best_idx);
    else
        class_id = 0;
    end

    % ── Aplicar umbral de rechazo ───────────────────────────────────────────
    if min_dist >= threshold
        class_id = 0;
        best_idx = 0;
    end
end

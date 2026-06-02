function [idx, dist, score, clase] = cbr_retrieve_topk(features, K)
%#codegen
% CBR_RETRIEVE_TOPK  Retrieve (4R) de los K casos mas cercanos.
%
% Generaliza cbr_reasoner.m: en vez de devolver solo el vecino mas cercano,
% devuelve los K casos mas similares ordenados por distancia ascendente.
% Es la base del ciclo 4R orquestado por el VLM (alimenta cbr_confianza.m
% y orquestador_cbr_vlm.m).
%
% Replica la logica de razonador_local.py:
%   - Normalizacion robusta por feature: (x - centro) / escala.
%   - Distancia euclidiana ponderada:  dist = norm((q-c).*w) / sqrt(D).
%   - Similitud:                        score = 1 / (1 + dist).
%
% La base de casos se embebe con coder.load -> apto para codegen (Hailo-8L).
%
% ENTRADAS
%   features  vector de 7 features (cualquier orientacion; se aplana a [1x7]):
%             [force_rms, force_peak_abs, input_rms, input_peak_abs,
%              corr_force_input, loop_area_norm, duration_s]
%   K         (escalar, constante en codegen) numero de vecinos a devolver.
%
% SALIDAS (todas [1 x K], ordenadas por distancia ascendente)
%   idx       indices (fila) de los K casos en la base
%   dist      distancias ponderadas
%   score     similitudes 1/(1+dist)
%   clase     etiqueta de histeresis de cada caso (1=lineal,2=moderada,3=marcada)
%
% Genera primero la base con:  >> exportar_base_casos
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    % ── Cargar base de casos de forma estatica (una sola vez) ──────────────
    persistent S
    if isempty(S)
        S = coder.load('datos/base_de_casos_cbr.mat');
    end

    case_base    = S.case_base;      % [N x D]
    feat_center  = S.feat_center;    % [1 x D]
    feat_scale   = S.feat_scale;     % [1 x D]
    feat_weights = S.feat_weights;   % [1 x D]
    hist_label   = S.hist_label;     % [N x 1]

    num_cases = size(case_base, 1);
    D = size(case_base, 2);

    % Aplanar features a fila [1 x D]
    q = reshape(features, 1, []);

    % ── Distancia ponderada a todos los casos ──────────────────────────────
    q_norm = (q - feat_center) ./ feat_scale;     % [1 x D]
    all_dist = zeros(num_cases, 1);
    for i = 1:num_cases
        c_norm = (case_base(i, :) - feat_center) ./ feat_scale;
        diff_vec = (q_norm - c_norm) .* feat_weights;
        all_dist(i) = norm(diff_vec) / sqrt(D);
    end

    % ── Ordenar ascendente y tomar top-K ────────────────────────────────────
    [sorted_dist, order] = sort(all_dist, 'ascend');

    Keff = min(K, num_cases);

    idx   = zeros(1, K);
    dist  = inf(1, K);
    score = zeros(1, K);
    clase = zeros(1, K);

    for j = 1:Keff
        ii = order(j);
        idx(j)   = ii;
        dist(j)  = sorted_dist(j);
        score(j) = 1 / (1 + sorted_dist(j));
        clase(j) = hist_label(ii);
    end
end

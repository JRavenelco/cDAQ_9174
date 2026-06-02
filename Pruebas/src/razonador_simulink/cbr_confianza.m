function [estado, conf, motivo] = cbr_confianza(dist, score, clase, threshold, tau_score, tau_margen)
%#codegen
% CBR_CONFIANZA  Clasifica la confianza del retrieval para la logica hibrida.
%
% A partir del resultado top-K de cbr_retrieve_topk.m, decide en cual de los
% tres estados esta el diagnostico, lo que determina si entra el VLM (System 2):
%
%   estado = 0  -> ALTA confianza        -> CBR directo (no se llama al VLM)
%   estado = 1  -> BAJA, NO OOD          -> VLM AUDITA (manda el CBR)
%   estado = 2  -> OOD (novedad)         -> VLM DECIDE (manda el VLM)
%
% Criterios (Perner 2008 para OOD; reject option / selective prediction para
% la baja confianza — ver prompts/defensa_vlm_orquestador.md):
%   - OOD:   best_dist >= threshold  (el caso esta fuera de la distribucion).
%   - BAJA:  best_score < tau_score        (el vecino no se parece lo bastante), o
%            margen top1-top2 < tau_margen  (ambiguo entre dos vecinos), o
%            clases divergentes en el top-2 (clase(1) ~= clase(2)).
%   - ALTA:  ninguna de las anteriores.
%
% ENTRADAS (dist, score, clase son [1 x K] de cbr_retrieve_topk)
%   dist        distancias ponderadas ordenadas ascendente
%   score       similitudes 1/(1+dist)
%   clase       etiquetas de histeresis de los top-K
%   threshold   umbral de rechazo OOD (misma escala que dist; p.ej. 1.0)
%   tau_score   similitud minima para considerar ALTA confianza (p.ej. 0.5)
%   tau_margen  margen minimo dist(2)-dist(1) para no ser ambiguo (p.ej. 0.05)
%
% SALIDAS
%   estado   0=alta, 1=baja_no_OOD, 2=OOD
%   conf     confianza continua en [0,1] (= score del mejor vecino)
%   motivo   codigo del disparo: 0=alta, 1=score_bajo, 2=margen_estrecho,
%            3=clases_divergentes, 4=OOD
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    best_dist  = dist(1);
    best_score = score(1);
    conf = best_score;

    % ── 1) OOD tiene prioridad (Perner 2008): el CBR ya no es fiable ────────
    if best_dist >= threshold
        estado = 2;
        motivo = 4;
        return;
    end

    % ── 2) Baja confianza (no OOD): tres disparadores ───────────────────────
    K = numel(dist);

    % (a) similitud insuficiente
    if best_score < tau_score
        estado = 1; motivo = 1; return;
    end

    if K >= 2
        % (b) margen estrecho entre los dos mejores (ambiguedad)
        margen = dist(2) - dist(1);
        if margen < tau_margen
            estado = 1; motivo = 2; return;
        end
        % (c) los dos mejores votan clases distintas
        if clase(1) ~= clase(2)
            estado = 1; motivo = 3; return;
        end
    end

    % ── 3) Alta confianza: CBR directo ──────────────────────────────────────
    estado = 0;
    motivo = 0;
end

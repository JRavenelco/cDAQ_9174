function probar_cbr()
% PROBAR_CBR  Parte 4a: prueba del Razonador CBR fuera de Simulink.
%
% Recorre todos los casos de la base, los pasa por cbr_reasoner() (el mismo
% codigo del bloque MATLAB Function) y reporta el caso mas cercano de cada
% uno. Sirve para validar que la traduccion MATLAB coincide con el razonador
% de Python (razonador_local.py) antes de cablear el modelo en Simulink.
%
% Uso:
%   >> exportar_base_casos   % (si aun no generaste el .mat)
%   >> probar_cbr

    here = fileparts(mfilename('fullpath'));
    mat_path = fullfile(here, 'datos', 'base_de_casos_cbr.mat');
    if ~isfile(mat_path)
        fprintf('No existe la base. Ejecutando exportar_base_casos...\n');
        exportar_base_casos();
    end
    S = load(mat_path);

    % Umbral grande para NO rechazar (ver vecino mas cercano siempre)
    threshold = 1e6;

    labels = {'lineal','moderada','marcada'};
    fprintf('\n%-38s %-10s %-9s %-8s %s\n', ...
            'CASO', 'DIST', 'SCORE', 'HIST', 'VECINO MAS CERCANO');
    fprintf('%s\n', repmat('-', 1, 110));

    n = size(S.case_base, 1);
    for k = 1:n
        % El query es el propio caso, pero excluimos el self para ver el vecino
        feats = S.case_base(k, :);
        [class_id, min_dist, best_score, best_idx] = ...
            cbr_reasoner_excluir(feats, threshold, k, S);

        if best_idx >= 1
            vecino = char(S.case_ids(best_idx));
            etiqueta = labels{class_id};
        else
            vecino = '(ninguno)';
            etiqueta = 'rechazado';
        end

        fprintf('%-38s %-10.4f %-9.4f %-8s %s\n', ...
                char(S.case_ids(k)), min_dist, best_score, etiqueta, vecino);
    end
    fprintf('\nOK. Si los vecinos coinciden con el razonador de Python, la traduccion es correcta.\n\n');
end


function [class_id, min_dist, best_score, best_idx] = cbr_reasoner_excluir(features, threshold, self_idx, S)
% Variante de cbr_reasoner que excluye un indice (para validacion 1-vs-resto).
    case_base    = S.case_base;
    feat_center  = S.feat_center;
    feat_scale   = S.feat_scale;
    feat_weights = S.feat_weights;
    hist_label   = S.hist_label;

    num_cases = size(case_base, 1);
    D = size(case_base, 2);
    q_norm = (features - feat_center) ./ feat_scale;

    min_dist = inf; best_idx = 0;
    for i = 1:num_cases
        if i == self_idx; continue; end
        c_norm = (case_base(i, :) - feat_center) ./ feat_scale;
        diff_vec = (q_norm - c_norm) .* feat_weights;
        dist = norm(diff_vec) / sqrt(D);
        if dist < min_dist
            min_dist = dist;
            best_idx = i;
        end
    end

    best_score = 1 / (1 + min_dist);
    if best_idx >= 1
        class_id = hist_label(best_idx);
    else
        class_id = 0;
    end
    if min_dist >= threshold
        class_id = 0; best_idx = 0;
    end
end

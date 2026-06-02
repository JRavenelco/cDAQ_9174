function ok = cbr_retain(features, clase, case_id, retain_flag)
% CBR_RETAIN  Cierra el ciclo 4R (Retain): anade un caso validado a la base.
%
% Si retain_flag es true, hace append del nuevo caso a
% datos/base_de_casos_cbr.mat (case_base, hist_label, case_ids). NO recalcula
% los centros/escalas/pesos (se mantienen los embebidos offline); esto preserva
% reproducibilidad y evita drift por un solo caso. Regenera con
% exportar_base_casos si quieres recomputar estadisticas con el caso incluido.
%
% Se ejecuta en Interpreted execution. Marcar coder.extrinsic si se llama desde
% un bloque codegen:  coder.extrinsic('cbr_retain');
%
% ENTRADAS
%   features    [1x7] vector de features del caso (orden FIJO del razonador).
%   clase       etiqueta de histeresis (1/2/3) validada por el VLM/usuario.
%   case_id     (char) identificador del nuevo caso.
%   retain_flag (logical) si false, no hace nada (devuelve false).
%
% SALIDA
%   ok   true si el caso fue anadido y guardado; false en caso contrario.
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    ok = false;
    if nargin < 4 || ~retain_flag
        return;
    end
    if nargin < 3 || isempty(case_id)
        case_id = sprintf('retain_%s', datestr(now, 'yyyymmdd_HHMMSS'));
    end

    here = fileparts(mfilename('fullpath'));
    mat_path = fullfile(here, 'datos', 'base_de_casos_cbr.mat');
    if ~isfile(mat_path)
        warning('cbr_retain: no existe %s. Ejecuta exportar_base_casos.', mat_path);
        return;
    end

    S = load(mat_path);

    feats = reshape(double(features), 1, []);
    D = size(S.case_base, 2);
    if numel(feats) ~= D
        warning('cbr_retain: features tiene %d elementos; se esperaban %d.', numel(feats), D);
        return;
    end

    % Evitar duplicar un case_id ya existente
    if any(strcmp(string(S.case_ids), string(case_id)))
        warning('cbr_retain: case_id "%s" ya existe; no se anade.', case_id);
        return;
    end

    % ── Append ───────────────────────────────────────────────────────────────
    S.case_base  = [S.case_base; feats];
    S.hist_label = [S.hist_label; double(clase)];
    S.case_ids   = [string(S.case_ids(:)); string(case_id)];

    % loop_area y rpm para mantener consistencia de campos (si existen)
    if isfield(S, 'loop_area')
        la = feats(strcmp_idx(S, 'loop_area_norm'));
        S.loop_area = [S.loop_area(:); la];
    end
    if isfield(S, 'rpm_est')
        S.rpm_est = [S.rpm_est(:); -1];
    end

    % Guardar (sobrescribe el .mat con N+1 casos)
    save(mat_path, '-struct', 'S');
    ok = true;
    fprintf('cbr_retain: caso "%s" anadido. Base: %d -> %d casos.\n', ...
            case_id, size(S.case_base,1)-1, size(S.case_base,1));
end


function j = strcmp_idx(S, name)
% Indice de 'name' en el orden de features; loop_area_norm es la posicion 6.
    j = 6;
    if isfield(S, 'feat_names')
        k = find(strcmp(S.feat_names, name), 1);
        if ~isempty(k); j = k; end
    end
end

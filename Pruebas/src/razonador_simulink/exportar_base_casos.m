function exportar_base_casos()
% EXPORTAR_BASE_CASOS  Parte 1 del Razonador CBR para Simulink.
%
% Lee la base de casos historicos (casos_historicos.csv) generada por el
% pipeline de Python (construir_casos.py) y la exporta a un archivo .mat
% que los bloques de Simulink (MATLAB Function / MATLAB System) cargan de
% forma estatica.
%
% Replica EXACTAMENTE la logica de retrieval de razonador_local.py:
%   - 7 features con pesos.
%   - Normalizacion robusta: (valor - mediana) / IQR  (fallback a std, luego 1).
%   - Las estadisticas (centro/escala) se calculan OFFLINE sobre la libreria
%     y se embeben, para que el bloque Simulink sea apto para tiempo real.
%
% Salida (base_de_casos_cbr.mat):
%   case_base    [N x 7]  valores crudos de las 7 features por caso
%   feat_names   {1 x 7}  nombres de las features (orden fijo)
%   feat_center  [1 x 7]  mediana por feature (para normalizar)
%   feat_scale   [1 x 7]  IQR por feature (para normalizar)
%   feat_weights [1 x 7]  pesos por feature
%   case_ids     {N x 1}  identificadores de caso (string)
%   loop_area    [N x 1]  loop_area_norm (para etiqueta de histeresis)
%   rpm_est      [N x 1]  rpm estimada
%   hist_label   [N x 1]  1=lineal, 2=moderada, 3=marcada (segun loop_area)
%
% Uso:
%   >> exportar_base_casos
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    % ── Orden FIJO de features y pesos (igual que razonador_local.py) ──────
    feat_names = {'force_rms','force_peak_abs','input_rms', ...
                  'input_peak_abs','corr_force_input','loop_area_norm', ...
                  'duration_s'};
    feat_weights = [1.0, 0.8, 1.0, 0.8, 0.7, 1.4, 0.3];

    % ── Localizar el CSV de casos ──────────────────────────────────────────
    % Carpeta standalone: prioriza la copia local en datos/, con fallback a la
    % ubicacion original del pipeline de Python (razonador_casos).
    here = fileparts(mfilename('fullpath'));
    csv_candidates = { ...
        fullfile(here, 'datos', 'casos_historicos.csv'), ...
        fullfile(here, '..', 'caracterizacion_fuerza', 'razonador_casos', ...
                 'salidas', 'casos_dataset', 'casos_historicos.csv') };
    csv_path = '';
    for ci = 1:numel(csv_candidates)
        if isfile(csv_candidates{ci})
            csv_path = csv_candidates{ci};
            break;
        end
    end
    if isempty(csv_path)
        error(['No se encontro casos_historicos.csv en:\n  %s\n  %s\n' ...
               'Copia el CSV a datos/ o ejecuta el pipeline de Python.'], ...
               csv_candidates{1}, csv_candidates{2});
    end

    % ── Leer CSV preservando nombres de columna ─────────────────────────────
    opts = detectImportOptions(csv_path, 'VariableNamingRule', 'preserve');
    T = readtable(csv_path, opts);

    n_feat = numel(feat_names);
    n_rows = height(T);

    % ── Construir matriz de features (N x 7) ────────────────────────────────
    case_base = NaN(n_rows, n_feat);
    for j = 1:n_feat
        col = feat_names{j};
        if ~ismember(col, T.Properties.VariableNames)
            error('Falta la columna "%s" en el CSV.', col);
        end
        v = T.(col);
        if ~isnumeric(v)
            v = str2double(string(v));   % por si viene como texto
        end
        case_base(:, j) = v;
    end

    % ── Descartar filas con features no finitas (NaN/Inf) ──────────────────
    valid = all(isfinite(case_base), 2);
    if any(~valid)
        fprintf('Aviso: se descartan %d caso(s) con features invalidas.\n', sum(~valid));
    end
    case_base = case_base(valid, :);

    % case_ids
    case_ids = string(T.case_id(valid));

    % loop_area y rpm (para etiquetas / contexto)
    loop_area = case_base(:, strcmp(feat_names, 'loop_area_norm'));
    if ismember('rpm_estimada', T.Properties.VariableNames)
        rpm_raw = T.rpm_estimada(valid);
        if ~isnumeric(rpm_raw); rpm_raw = str2double(string(rpm_raw)); end
        rpm_est = rpm_raw;
        rpm_est(~isfinite(rpm_est)) = -1;
    else
        rpm_est = -ones(size(loop_area));
    end

    % ── Estadisticas robustas por feature (centro = mediana, escala = IQR) ──
    feat_center = zeros(1, n_feat);
    feat_scale  = ones(1, n_feat);
    for j = 1:n_feat
        x = case_base(:, j);
        feat_center(j) = median(x);
        % IQR con percentiles estilo numpy (interpolacion lineal)
        q1 = np_percentile(x, 25);
        q3 = np_percentile(x, 75);
        iqr_val = q3 - q1;
        if ~isfinite(iqr_val) || iqr_val <= 1e-12
            iqr_val = std(x);
        end
        if ~isfinite(iqr_val) || iqr_val <= 1e-12
            iqr_val = 1.0;
        end
        feat_scale(j) = iqr_val;
    end

    % ── Etiqueta de histeresis segun loop_area_norm ────────────────────────
    %   < 0.3  -> 1 (lineal / casi sin histeresis)
    %   0.3-1.0-> 2 (histeresis moderada)
    %   > 1.0  -> 3 (histeresis marcada)
    hist_label = ones(size(loop_area));
    hist_label(loop_area >= 0.3 & loop_area < 1.0) = 2;
    hist_label(loop_area >= 1.0) = 3;

    % ── Guardar .mat ────────────────────────────────────────────────────────
    out_dir = fullfile(here, 'datos');
    if ~isfolder(out_dir); mkdir(out_dir); end
    out_path = fullfile(out_dir, 'base_de_casos_cbr.mat');
    save(out_path, 'case_base', 'feat_names', 'feat_center', 'feat_scale', ...
         'feat_weights', 'case_ids', 'loop_area', 'rpm_est', 'hist_label');

    % ── Resumen ──────────────────────────────────────────────────────────────
    fprintf('\n=== Base de casos CBR exportada ===\n');
    fprintf('  Casos validos : %d\n', size(case_base, 1));
    fprintf('  Features      : %d  (%s)\n', n_feat, strjoin(feat_names, ', '));
    fprintf('  Etiquetas     : lineal=%d, moderada=%d, marcada=%d\n', ...
            sum(hist_label==1), sum(hist_label==2), sum(hist_label==3));
    fprintf('  Centro (med.) : [%s]\n', num2str(feat_center, '% .4g'));
    fprintf('  Escala (IQR)  : [%s]\n', num2str(feat_scale, '% .4g'));
    fprintf('  Guardado en   : %s\n\n', out_path);
end


function p = np_percentile(x, q)
% NP_PERCENTILE  Percentil estilo numpy (metodo 'linear'), para igualar Python.
    x = sort(x(:));
    n = numel(x);
    if n == 0; p = NaN; return; end
    if n == 1; p = x(1); return; end
    pos = (q/100) * (n - 1);     % indice base-0
    lo = floor(pos);
    hi = ceil(pos);
    frac = pos - lo;
    p = x(lo+1) + frac * (x(hi+1) - x(lo+1));
end

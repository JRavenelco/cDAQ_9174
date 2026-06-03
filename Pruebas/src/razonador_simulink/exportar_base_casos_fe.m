function exportar_base_casos_fe()
% EXPORTAR_BASE_CASOS_FE  Regenera la base del CBR con descriptores F-E.
%
% Reemplaza las 2 features de histeresis CRUDAS por sus versiones F-E
% (envolvente), que anti-correlacionaban con la metodologia de la tesis:
%   corr_force_input -> corr_FE        (correlacion fuerza-envolvente)
%   loop_area_norm   -> loop_area_FE   (area del lazo F-E por shoelace)
% calculadas sobre la VENTANA 0.5s de cada caso (escala de la metodologia).
% Las 5 features de nivel (RMS/pico de fuerza y aceleracion, duracion) se
% toman del CSV sin cambio.
%
% La exploracion mostro que alpha/R2 del modelo KAN-PINN son ruidosos
% (modelo Bouc-Wen simple con R2<=0.28); por eso se guardan como METADATO
% (alpha_meta, R2_meta, k_meta, c_meta) y NO entran en la distancia del CBR.
%
% Pesos: corr_FE es ahora el descriptor de histeresis dominante (1.4);
% loop_area_FE pasa a 0.7 (poco discriminante en ventana 0.5s).
% Etiqueta lineal/moderada/marcada por TERCILES de loop_area_FE.
%
% Respalda la base cruda (base_de_casos_cbr_CRUDA.mat) y pisa
% base_de_casos_cbr.mat para ACTIVAR la correccion en el orquestador.
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    here = fileparts(mfilename('fullpath'));
    csv = fullfile(here,'datos','casos_historicos.csv');
    if ~isfile(csv)
        csv = fullfile(here,'..','caracterizacion_fuerza','razonador_casos', ...
                       'salidas','casos_dataset','casos_historicos.csv');
    end
    vdir = fullfile(here,'..','caracterizacion_fuerza','razonador_casos', ...
                    'salidas','casos_dataset','ventanas');

    T = readtable(csv, detectImportOptions(csv,'VariableNamingRule','preserve'));
    ids = string(T.case_id);
    N = numel(ids);

    feat_names   = {'force_rms','force_peak_abs','input_rms','input_peak_abs', ...
                    'corr_FE','loop_area_FE','duration_s'};
    feat_weights = [1.0, 0.8, 1.0, 0.8, 1.4, 0.7, 0.3];

    case_base  = nan(N, 7);
    alpha_meta = nan(N,1); R2_meta = nan(N,1); k_meta = nan(N,1); c_meta = nan(N,1);
    keep = false(N,1);

    fprintf('Recalculando features F-E (envolvente) por caso...\n');
    for i = 1:N
        wp = fullfile(vdir, ids(i) + "_ventana_0.5s.txt");
        if ~isfile(wp); fprintf('  (sin ventana) %s\n', ids(i)); continue; end
        M = readmatrix(wp,'FileType','text','NumHeaderLines',1);
        if size(M,2) < 3; continue; end
        t = M(:,1); f = M(:,2); a = M(:,3);     % tiempo_s, fuerza_V, entrada_g

        [~, info] = extraer_features_envolvente(t, f, a);
        mdl = ajustar_kan_pinn_envolvente(t, f, a, struct('maxiter',60,'popsize',25));

        fr = num(T,'force_rms',i); fp = num(T,'force_peak_abs',i);
        ir = num(T,'input_rms',i); ip = num(T,'input_peak_abs',i);
        dur = num(T,'duration_s',i);

        case_base(i,:) = [fr, fp, ir, ip, info.corr_FE, info.loop_area_FE, dur];
        alpha_meta(i)=mdl.alpha; R2_meta(i)=mdl.R2; k_meta(i)=mdl.k; c_meta(i)=mdl.c;
        keep(i) = all(isfinite(case_base(i,:)));
    end

    case_base = case_base(keep,:);
    case_ids  = ids(keep);
    alpha_meta=alpha_meta(keep); R2_meta=R2_meta(keep);
    k_meta=k_meta(keep); c_meta=c_meta(keep);
    rpm_est = getrpm(T, keep);

    % ── Estadisticas robustas (mediana / IQR estilo numpy) ──────────────────
    n_feat = 7;
    feat_center = zeros(1,n_feat); feat_scale = ones(1,n_feat);
    for j = 1:n_feat
        x = case_base(:,j);
        feat_center(j) = median(x);
        iqr_val = np_percentile(x,75) - np_percentile(x,25);
        if ~isfinite(iqr_val) || iqr_val <= 1e-12; iqr_val = std(x); end
        if ~isfinite(iqr_val) || iqr_val <= 1e-12; iqr_val = 1.0; end
        feat_scale(j) = iqr_val;
    end

    % ── Etiquetas por TERCILES de loop_area_FE (col 6) ──────────────────────
    loop_area = case_base(:,6);
    q = [np_percentile(loop_area,100/3), np_percentile(loop_area,200/3)];
    hist_label = ones(size(loop_area));
    hist_label(loop_area > q(1) & loop_area <= q(2)) = 2;
    hist_label(loop_area > q(2)) = 3;

    % ── Respaldar cruda y guardar F-E ───────────────────────────────────────
    out = fullfile(here,'datos','base_de_casos_cbr.mat');
    if isfile(out)
        copyfile(out, fullfile(here,'datos','base_de_casos_cbr_CRUDA.mat'));
    end
    save(out, 'case_base','feat_names','feat_center','feat_scale','feat_weights', ...
         'case_ids','loop_area','rpm_est','hist_label', ...
         'alpha_meta','R2_meta','k_meta','c_meta');

    fprintf('\n=== Base CBR regenerada a F-E ===\n');
    fprintf('  Casos validos : %d\n', size(case_base,1));
    fprintf('  Features      : %s\n', strjoin(feat_names, ', '));
    fprintf('  Pesos         : [%s]\n', num2str(feat_weights,'% .2g'));
    fprintf('  Umbrales (terciles loop_area_FE): t1=%.4f t2=%.4f\n', q(1), q(2));
    fprintf('  Etiquetas     : lineal=%d, moderada=%d, marcada=%d\n', ...
            sum(hist_label==1), sum(hist_label==2), sum(hist_label==3));
    fprintf('  Metadato      : alpha_meta, R2_meta, k_meta, c_meta (NO en distancia)\n');
    fprintf('  Respaldo cruda: base_de_casos_cbr_CRUDA.mat\n');
    fprintf('  Guardado      : %s\n\n', out);
end

function v = num(T, col, i)
    if ismember(col, T.Properties.VariableNames)
        x = T.(col); if ~isnumeric(x); x = str2double(string(x)); end
        v = x(i);
    else
        v = NaN;
    end
end

function rpm = getrpm(T, keep)
    if ismember('rpm_estimada', T.Properties.VariableNames)
        r = T.rpm_estimada; if ~isnumeric(r); r = str2double(string(r)); end
        rpm = r(keep); rpm(~isfinite(rpm)) = -1;
    else
        rpm = -ones(sum(keep),1);
    end
end

function p = np_percentile(x, q)
    x = sort(x(:)); n = numel(x);
    if n == 0; p = NaN; return; end
    if n == 1; p = x(1); return; end
    pos = (q/100)*(n-1); lo = floor(pos); hi = ceil(pos); frac = pos - lo;
    p = x(lo+1) + frac*(x(hi+1) - x(lo+1));
end

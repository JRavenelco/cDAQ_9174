function caso = correr_caso(entrada, opts)
% CORRER_CASO  Pipeline de UNA CORRIDA: integra las DOS CAPAS en el orquestador.
%
% Encadena, sobre un corte real, las dos capas de la arquitectura y el
% diagnostico CBR+VLM, devolviendo el "caso" completo:
%
%   senal (t,fuerza,aceleracion)
%     -> CAPA RAPIDA  : extraer_features_envolvente  (7 features F-E)
%     -> CAPA PROFUNDA: ajustar_kan_pinn_envolvente  (alpha, R2, k, c)  [METADATO]
%     -> (opcional)     generar_lazo_png              (imagen del lazo F-E para el VLM)
%     -> ORQUESTADOR  : orquestador_cbr_vlm(features, params con .meta=alpha/R2)
%     -> (opcional)     cbr_retain                    (cierra el ciclo 4R)
%
% El alpha/R2 de la capa profunda viajan como METADATO al VLM (contexto), pero
% NO entran en la distancia del CBR (la exploracion mostro que son ruidosos:
% modelo Bouc-Wen simple con R2<=0.28). La histeresis se mide con corr_FE/
% loop_area_FE (capa rapida F-E), que es lo que discrimina.
%
% ENTRADAS
%   entrada  o bien:
%            - ruta a un .txt de ventana (cols: tiempo_s, fuerza_V, entrada_g[, ...])
%            - matriz [N x >=3] con columnas [t, fuerza, aceleracion]
%   opts     struct opcional:
%            .usar_vlm   (def false)  activa la capa deliberativa (requiere VLM)
%            .modelo     (def 'cloud')
%            .con_imagen (def igual a usar_vlm) genera el lazo F-E en base64
%            .case_id    (def nombre del archivo)  id para Retain
%            .params     struct para sobreescribir umbrales del orquestador
%            .kan        struct opts de ajustar_kan_pinn_envolvente (maxiter/popsize)
%
% SALIDA (struct caso)
%   .case_id .features .feat_names .info_FE .modelo_kan (alpha/R2/k/c)
%   .diagnostico (salida completa de orquestador_cbr_vlm)
%   .img_b64 (si con_imagen)
%
% Uso:
%   >> caso = correr_caso('..\caracterizacion_fuerza\datos_txt_ventanas\corte2_mejor_0.5s.txt')
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    if nargin < 2; opts = struct(); end
    opts = completar_opts(opts);

    here = fileparts(mfilename('fullpath'));
    if ~isfile(fullfile(here,'datos','base_de_casos_cbr.mat'))
        fprintf('Base no encontrada; regenerando a F-E...\n');
        exportar_base_casos_fe();
    end

    % ── Leer la senal ───────────────────────────────────────────────────────
    [t, f, a, case_id] = leer_senal(entrada);
    if isempty(opts.case_id); opts.case_id = case_id; end

    fprintf('\n========= CORRER_CASO: %s (%d muestras) =========\n', opts.case_id, numel(t));

    % ── CAPA RAPIDA: features F-E ───────────────────────────────────────────
    [features, info_FE] = extraer_features_envolvente(t, f, a);
    fprintf('[1] Capa rapida F-E : corr_FE=%.3f  loop_area_FE=%.4f\n', ...
        info_FE.corr_FE, info_FE.loop_area_FE);

    % ── CAPA PROFUNDA: modelo KAN-PINN (metadato) ───────────────────────────
    mdl = ajustar_kan_pinn_envolvente(t, f, a, opts.kan);
    meta = struct('alpha', mdl.alpha, 'R2', mdl.R2, 'k', mdl.k, 'c', mdl.c);
    fprintf('[2] Capa profunda   : alpha=%.3f  R2=%.3f  (metadato, no en distancia)\n', ...
        mdl.alpha, mdl.R2);

    % ── Imagen del lazo F-E (opcional, para el VLM) ─────────────────────────
    img_b64 = '';
    if opts.con_imagen
        img_b64 = generar_lazo_png(f, info_FE.E, sprintf('Lazo F-E: %s', opts.case_id));
    end

    % ── ORQUESTADOR (CBR + VLM por excepcion) ───────────────────────────────
    params = opts.params;
    params.usar_vlm = opts.usar_vlm;
    params.modelo   = opts.modelo;
    params.img_b64  = img_b64;
    params.meta     = meta;
    out = orquestador_cbr_vlm(features, params);

    labels = {'lineal','moderada','marcada'};
    cl = '(rechazado)'; if out.clase_final >= 1 && out.clase_final <= 3; cl = labels{out.clase_final}; end
    fprintf('[3] Diagnostico CBR : clase=%d %s | caso_sel=%d | fuente=%s | conf=%.3f | estado=%d\n', ...
        out.clase_final, cl, out.caso_sel, out.fuente, out.confianza, out.estado);
    fprintf('    accion: %s\n', out.accion);
    if out.flag_discrepancia
        fprintf('    [!] VLM discrepa del CBR (ver motivo).\n');
    end

    % ── Empaquetar el caso ───────────────────────────────────────────────────
    caso = struct();
    caso.case_id      = opts.case_id;
    caso.features     = features;
    caso.feat_names   = {'force_rms','force_peak_abs','input_rms','input_peak_abs', ...
                         'corr_FE','loop_area_FE','duration_s'};
    caso.info_FE      = info_FE;
    caso.modelo_kan   = meta;
    caso.diagnostico  = out;
    caso.img_b64      = img_b64;

    fprintf('======================================================\n\n');
end


% ===========================================================================
% Helpers
% ===========================================================================
function o = completar_opts(o)
    d = struct('usar_vlm',false, 'modelo','cloud', 'con_imagen',[], ...
               'case_id','', 'params',struct(), ...
               'kan',struct('maxiter',60,'popsize',25));
    f = fieldnames(d);
    for i = 1:numel(f)
        if ~isfield(o, f{i}); o.(f{i}) = d.(f{i}); end
    end
    if isempty(o.con_imagen); o.con_imagen = o.usar_vlm; end
end

function [t, f, a, case_id] = leer_senal(entrada)
    if (ischar(entrada) || isstring(entrada)) && isfile(entrada)
        M = readmatrix(char(entrada), 'FileType','text', 'NumHeaderLines',1);
        [~, nm] = fileparts(char(entrada));
        case_id = regexprep(nm, '_ventana.*$', '');
    elseif isnumeric(entrada) && size(entrada,2) >= 3
        M = entrada;
        case_id = sprintf('caso_%s', datestr(now,'yyyymmdd_HHMMSS'));
    else
        error('correr_caso: entrada debe ser ruta a .txt o matriz [N x >=3].');
    end
    if size(M,2) < 3
        error('correr_caso: se requieren al menos 3 columnas [t, fuerza, aceleracion].');
    end
    t = M(:,1); f = M(:,2); a = M(:,3);
end

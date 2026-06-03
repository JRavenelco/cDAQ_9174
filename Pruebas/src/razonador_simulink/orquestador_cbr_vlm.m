function out = orquestador_cbr_vlm(features, params)
% ORQUESTADOR_CBR_VLM  Orquestador hibrido del ciclo 4R (CBR + VLM por excepcion).
%
% MATLAB Function principal de la capa deliberativa. Aplica la AUTORIDAD HIBRIDA:
%   - ALTA confianza        -> Reuse directo: manda el CBR (no se llama al VLM).
%   - BAJA, NO OOD          -> VLM AUDITA: la clase oficial sigue siendo la del CBR;
%                              si el VLM discrepa, marca flag_discrepancia y guarda
%                              su explicacion (Revise supervisado, reproducible).
%   - OOD (min_dist>=thr)   -> VLM DECIDE: la clase final la da el VLM (el CBR ya no
%                              es fiable por diseno).
% Si el VLM falla (ok=false), DEGRADA a CBR + aviso.
%
% Se ejecuta en Interpreted execution (usa vlm_cliente via coder.extrinsic).
%
% ENTRADAS
%   features  vector de 7 features (se aplana a [1x7]).
%   params    struct de configuracion (usar valores por defecto con params_default()):
%             .K          numero de vecinos a recuperar         (def 3)
%             .threshold  umbral de rechazo OOD                  (def 1.0)
%             .tau_score  similitud minima para ALTA confianza   (def 0.5)
%             .tau_margen margen minimo top1-top2                (def 0.05)
%             .modelo     preferencia VLM ('cloud'/'qwen3:8b'/...) (def 'cloud')
%             .usar_vlm   habilita la capa deliberativa          (def true)
%             .img_b64    imagen del lazo F-x en base64 (opc)    (def '')
%             .meta       struct metadato de la capa profunda (KAN-PINN):
%                         .alpha .R2 .k .c  -> se PASAN al VLM como contexto,
%                         pero NO entran en la distancia del CBR (son ruidosos).
%
% SALIDA (struct out)
%   .clase_final       clase de histeresis final (1/2/3/0)
%   .caso_sel          indice del caso aplicado
%   .accion            (char) accion recomendada
%   .confianza         (double) confianza del diagnostico
%   .fuente            (char) 'CBR' | 'CBR+audit' | 'VLM' | 'CBR(degradado)'
%   .flag_discrepancia (logical) VLM auditor discrepa del CBR
%   .retain_flag       (logical) caso a retener (Retain)
%   .motivo            (char) explicacion del camino tomado
%   .estado            0=alta,1=baja_no_OOD,2=OOD
%   .topk              struct con idx/dist/score/clase del retrieval
%   .meta              struct metadato KAN-PINN propagado (alpha/R2/k/c)
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    coder.extrinsic('vlm_cliente');

    if nargin < 2 || isempty(params); params = params_default(); end
    params = completar_params(params);

    % ── RETRIEVE (4R) ───────────────────────────────────────────────────────
    [idx, dist, score, clase] = cbr_retrieve_topk(features, params.K);

    % ── CONFIANZA -> estado {alta, baja_no_OOD, OOD} ────────────────────────
    [estado, conf, motivo_conf] = cbr_confianza(dist, score, clase, ...
        params.threshold, params.tau_score, params.tau_margen);

    % Diagnostico base del CBR (vecino mas cercano)
    clase_cbr = clase(1);
    caso_cbr  = idx(1);

    % Valores por defecto de salida (camino CBR)
    out = struct();
    out.clase_final       = clase_cbr;
    out.caso_sel          = caso_cbr;
    out.accion            = '';
    out.confianza         = conf;
    out.fuente            = 'CBR';
    out.flag_discrepancia = false;
    out.retain_flag       = false;
    out.motivo            = motivo_texto(motivo_conf);
    out.estado            = estado;
    out.topk              = struct('idx', idx, 'dist', dist, 'score', score, 'clase', clase);
    out.meta              = params.meta;

    % ── ALTA confianza -> CBR directo (no VLM) ──────────────────────────────
    if estado == 0 || ~params.usar_vlm
        out.accion = accion_por_clase(clase_cbr);
        return;
    end

    % ── BAJA o OOD -> llamar al VLM (System 2) ──────────────────────────────
    % Veredicto vacio por defecto (struct estable para codegen interpretado)
    v = struct('clase',0,'caso_sel',0,'accion','','retain',false, ...
               'confianza',0,'justificacion','','ok',false, ...
               'fuente_modelo','none','raw','');

    user_prompt = construir_prompt_usuario(features, idx, dist, score, clase, estado, params.meta);
    system_prompt = cargar_system_prompt();

    v = vlm_cliente(system_prompt, user_prompt, params.img_b64, params.modelo);

    if ~v.ok
        % Fallback: el VLM no respondio -> degradar a CBR + aviso
        out.fuente = 'CBR(degradado)';
        out.accion = accion_por_clase(clase_cbr);
        out.motivo = ['VLM no disponible; se mantiene CBR. ' v.justificacion];
        return;
    end

    if estado == 2
        % ── OOD -> el VLM DECIDE ─────────────────────────────────────────────
        out.clase_final = v.clase;
        out.fuente      = 'VLM';
        out.accion      = v.accion;
        out.confianza   = v.confianza;
        out.retain_flag = v.retain;
        if v.caso_sel >= 1; out.caso_sel = v.caso_sel; end
        out.motivo = ['OOD: VLM decide. ' v.justificacion];
    else
        % ── BAJA no-OOD -> el VLM AUDITA (manda el CBR) ─────────────────────
        out.fuente = 'CBR+audit';
        out.accion = accion_por_clase(clase_cbr);
        out.flag_discrepancia = (v.clase ~= 0) && (v.clase ~= clase_cbr);
        if v.caso_sel >= 1; out.caso_sel = v.caso_sel; end
        if out.flag_discrepancia
            out.motivo = sprintf('Baja confianza: VLM discrepa (CBR=%d, VLM=%d). %s', ...
                clase_cbr, v.clase, v.justificacion);
        else
            out.motivo = ['Baja confianza: VLM confirma CBR. ' v.justificacion];
        end
    end
end


% ===========================================================================
% Helpers
% ===========================================================================
function p = params_default()
    % Umbrales recalibrados a la base F-E (1-vs-resto sobre 19 casos):
    %   dist in-distribution: mediana=0.093, p75=0.180, p95=p100=0.486
    %   -> threshold (OOD) 0.6 (sobre el max in-distribution, con margen)
    %   -> tau_score 0.85 (score en p75 de dist; por debajo => baja confianza)
    p = struct('K',3, 'threshold',0.6, 'tau_score',0.85, 'tau_margen',0.05, ...
               'modelo','cloud', 'usar_vlm',true, 'img_b64','', 'meta',struct());
end

function p = completar_params(p)
    d = params_default();
    f = fieldnames(d);
    for i = 1:numel(f)
        if ~isfield(p, f{i}) || isempty(p.(f{i}))
            p.(f{i}) = d.(f{i});
        end
    end
end

function a = accion_por_clase(clase)
    switch clase
        case 1; a = 'Comportamiento lineal: corte nominal, sin accion correctiva.';
        case 2; a = 'Histeresis moderada: vigilar avance/RPM; considerar ajuste Bouc-Wen.';
        case 3; a = 'Histeresis marcada: ajuste Bouc-Wen recomendado; revisar desgaste y avance/RPM.';
        otherwise; a = 'Clase desconocida: revisar manualmente la senal.';
    end
end

function s = motivo_texto(motivo)
    switch motivo
        case 0; s = 'Alta confianza (CBR directo).';
        case 1; s = 'Baja confianza: similitud insuficiente.';
        case 2; s = 'Baja confianza: margen estrecho entre vecinos.';
        case 3; s = 'Baja confianza: clases divergentes en el top-2.';
        case 4; s = 'OOD: distancia >= umbral de rechazo.';
        otherwise; s = 'Estado no especificado.';
    end
end

function up = construir_prompt_usuario(features, idx, dist, score, clase, estado, meta)
    if nargin < 7; meta = struct(); end
    q = reshape(features, 1, []);
    names = {'force_rms','force_peak_abs','input_rms','input_peak_abs', ...
             'corr_FE','loop_area_FE','duration_s'};
    lineas = sprintf('MODO: %s\n\nFEATURES DEL CORTE ACTUAL (histeresis sobre envolvente F-E):\n', modo_texto(estado));
    for i = 1:numel(names)
        lineas = [lineas sprintf('  %-18s = %.5g\n', names{i}, q(i))]; %#ok<AGROW>
    end
    % Metadato de la capa profunda (modelo KAN-PINN): contexto, NO usado en distancia
    if isstruct(meta) && ~isempty(fieldnames(meta))
        lineas = [lineas sprintf('\nMETADATO MODELO KAN-PINN (capa profunda, contexto):\n')];
        if isfield(meta,'alpha'); lineas = [lineas sprintf('  alpha = %.4g  (1=lineal, 0=histeresis)\n', meta.alpha)]; end %#ok<AGROW>
        if isfield(meta,'R2');    lineas = [lineas sprintf('  R2    = %.4g  (calidad del ajuste; bajo => poco fiable)\n', meta.R2)]; end %#ok<AGROW>
        if isfield(meta,'k');     lineas = [lineas sprintf('  k     = %.4g\n', meta.k)]; end %#ok<AGROW>
        if isfield(meta,'c');     lineas = [lineas sprintf('  c     = %.4g\n', meta.c)]; end %#ok<AGROW>
    end
    lineas = [lineas sprintf('\nTOP-%d CASOS RECUPERADOS (CBR):\n', numel(idx))];
    for j = 1:numel(idx)
        lineas = [lineas sprintf('  #%d -> caso_sel=%d | dist=%.4f | score=%.4f | clase=%d\n', ...
            j, idx(j), dist(j), score(j), clase(j))]; %#ok<AGROW>
    end
    lineas = [lineas sprintf(['\nResponde SOLO con el JSON especificado ' ...
        '(clase, caso_sel, accion, retain, confianza, justificacion).'])];
    up = lineas;
end

function s = modo_texto(estado)
    if estado == 2; s = 'decidir'; else; s = 'auditar'; end
end

function sp = cargar_system_prompt()
    here = fileparts(mfilename('fullpath'));
    f = fullfile(here, 'prompts', 'orquestador_vlm.md');
    if isfile(f)
        sp = fileread(f);
    else
        sp = ['Eres un orquestador de un sistema CBR de histeresis en fresado. ' ...
              'Responde SOLO con un JSON: {"clase":int,"caso_sel":int,"accion":str,' ...
              '"retain":bool,"confianza":num,"justificacion":str}.'];
    end
end

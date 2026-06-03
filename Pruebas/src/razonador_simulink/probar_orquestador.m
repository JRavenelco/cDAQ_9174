function probar_orquestador()
% PROBAR_ORQUESTADOR  Valida los 3 caminos del orquestador hibrido SIN Simulink.
%
% Camino (a) ALTA confianza   -> fuente=CBR, sin llamar al VLM.
% Camino (b) BAJA no-OOD       -> fuente=CBR+audit (si el VLM responde) o CBR(degradado).
% Camino (c) OOD               -> fuente=VLM (si responde) o CBR(degradado).
%
% Util para probar la logica antes de cablear el modelo y para verificar el
% fallback del VLM (Ollama apagado -> OpenRouter; sin red -> degradado a CBR).
%
% Uso:
%   >> exportar_base_casos    % si aun no existe el .mat
%   >> probar_orquestador

    here = fileparts(mfilename('fullpath'));
    if ~isfile(fullfile(here,'datos','base_de_casos_cbr.mat'))
        exportar_base_casos_fe();
    end

    % Para aislar la LOGICA del orquestador del VLM (que requiere red/modelo),
    % por defecto se desactiva el VLM y se fuerza cada estado con los umbrales.
    fprintf('\n================ PRUEBA ORQUESTADOR CBR+VLM ================\n');

    % Vectores derivados de la BASE F-E ACTIVA (auto-consistentes con la escala
    % vigente). feat_real = un caso real de la base (alta confianza, dist=0);
    % feat_baja = punto intermedio entre dos casos (ambiguo, baja confianza);
    % feat_ood = caso real escalado fuera de la distribucion (OOD).
    S = load(fullfile(here,'datos','base_de_casos_cbr.mat'));
    cb = S.case_base;
    feat_real = cb(1,:);                 % caso real -> coincide consigo mismo
    feat_baja = 0.5*(cb(1,:) + cb(end,:));   % mezcla de dos casos -> ambiguo
    feat_ood  = cb(1,:) + 8*(S.feat_scale); % desplazado ~8 IQR -> OOD claro

    % ── (a) ALTA confianza: VLM off, umbrales laxos -> CBR directo ──────────
    p = struct('K',3,'threshold',1e6,'tau_score',0.0,'tau_margen',0.0, ...
               'modelo','cloud','usar_vlm',false,'img_b64','');
    mostrar('(a) ALTA confianza (VLM off)', orquestador_cbr_vlm(feat_real, p));

    % ── (b) BAJA no-OOD: feature INTERMEDIO (mezcla lineal+marcada) -> estado=1 ─
    %     Un punto ambiguo entre clases NO es identico a ningun caso (score<1) y
    %     queda dentro de la memoria (dist<threshold): cae en baja_no_OOD. Con el
    %     VLM apagado la clase la mantiene el CBR, pero el estado=1 demuestra que
    %     la deteccion del camino auditor funciona (con usar_vlm=true, el VLM auditaria).
    p = struct('K',3,'threshold',0.6,'tau_score',0.85,'tau_margen',0.05, ...
               'modelo','cloud','usar_vlm',false,'img_b64','');
    mostrar('(b) BAJA no-OOD: feature intermedio (VLM off)', orquestador_cbr_vlm(feat_baja, p));

    % ── (c) OOD: umbrales F-E por defecto, VLM off -> degradado a CBR ───────
    p = struct('K',3,'threshold',0.6,'tau_score',0.85,'tau_margen',0.05, ...
               'modelo','cloud','usar_vlm',false,'img_b64','');
    mostrar('(c) OOD (VLM off)', orquestador_cbr_vlm(feat_ood, p));

    fprintf('\n--- Con VLM activado (requiere Ollama o OPENROUTER_API_KEY) ---\n');
    fprintf('Para probar el VLM real, repite con usar_vlm=true, p.ej.:\n');
    fprintf('  p = struct(''K'',3,''threshold'',0.01,''tau_score'',0.5,''tau_margen'',0.05,...\n');
    fprintf('             ''modelo'',''cloud'',''usar_vlm'',true,''img_b64'','''');\n');
    fprintf('  orquestador_cbr_vlm([%s], p)\n', num2str(feat_ood));
    fprintf('===========================================================\n\n');
end


function mostrar(titulo, out)
    fprintf('\n%s\n', titulo);
    fprintf('  estado=%d | clase_final=%d | caso_sel=%d | confianza=%.3f\n', ...
            out.estado, out.clase_final, out.caso_sel, out.confianza);
    fprintf('  fuente=%s | flag_discrepancia=%d | retain=%d\n', ...
            out.fuente, out.flag_discrepancia, out.retain_flag);
    fprintf('  accion: %s\n', out.accion);
    fprintf('  motivo: %s\n', out.motivo);
end

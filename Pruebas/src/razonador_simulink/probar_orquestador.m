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
        exportar_base_casos();
    end

    % Para aislar la LOGICA del orquestador del VLM (que requiere red/modelo),
    % por defecto se desactiva el VLM y se fuerza cada estado con los umbrales.
    fprintf('\n================ PRUEBA ORQUESTADOR CBR+VLM ================\n');

    % Caso real de la base: histeresis marcada (loop_area_norm=2.08)
    feat_marcada = [0.8922 1.2031 0.01753 0.04359 -0.002637 2.0806 30.9996];
    % Caso real: lineal (loop_area_norm ~ 0.01)
    feat_lineal  = [0.6747 1.0902 0.01734 0.04373 0.001664 0.0101 49.44];
    % Caso sintetico claramente OOD (features fuera de rango)
    feat_ood     = [50 80 5 9 0.9 25 0.5];

    % ── (a) ALTA confianza: VLM off, umbrales laxos -> CBR directo ──────────
    p = struct('K',3,'threshold',1e6,'tau_score',0.0,'tau_margen',0.0, ...
               'modelo','cloud','usar_vlm',false,'img_b64','');
    mostrar('(a) ALTA confianza (VLM off)', orquestador_cbr_vlm(feat_marcada, p));

    % ── (b) BAJA no-OOD: forzar baja por tau_score alto, VLM off (degradado/CBR) ─
    p = struct('K',3,'threshold',1e6,'tau_score',0.99,'tau_margen',0.0, ...
               'modelo','cloud','usar_vlm',false,'img_b64','');
    mostrar('(b) BAJA no-OOD (VLM off -> CBR)', orquestador_cbr_vlm(feat_lineal, p));

    % ── (c) OOD: threshold bajo, VLM off -> degradado a CBR ─────────────────
    p = struct('K',3,'threshold',0.01,'tau_score',0.5,'tau_margen',0.05, ...
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

function [clase_final, confianza, estado, flag_discrep, fuente_cod] = wrap_orquestador(features)
%#codegen
% Envoltura escalar del orquestador hibrido para el bloque MATLAB Function.
% Corre en Interpreted execution (orquestador_cbr_vlm usa coder.extrinsic).
    coder.extrinsic('orquestador_cbr_vlm');
    out = struct('clase_final',0,'caso_sel',0,'accion','','confianza',0,'fuente','CBR','flag_discrepancia',false,'retain_flag',false,'motivo','','estado',0,'topk',struct());
    params = struct('K',3,'threshold',0.6,'tau_score',0.85,'tau_margen',0.05,'modelo','cloud','usar_vlm',false,'img_b64','','meta',struct());
    out = orquestador_cbr_vlm(features, params);
    clase_final = double(out.clase_final);
    confianza   = double(out.confianza);
    estado      = double(out.estado);
    flag_discrep= double(out.flag_discrepancia);
    fuente_cod  = fuente_a_cod(out.fuente);
end
function c = fuente_a_cod(f)
%#codegen
    c = 0;
    coder.varsize('f');
    if strcmp(f,'CBR'); c = 1;
    elseif strcmp(f,'CBR+audit'); c = 2;
    elseif strcmp(f,'VLM'); c = 3;
    elseif strcmp(f,'CBR(degradado)'); c = 4; end
end

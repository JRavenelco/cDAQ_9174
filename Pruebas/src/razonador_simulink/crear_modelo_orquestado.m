function crear_modelo_orquestado()
% CREAR_MODELO_ORQUESTADO  Builder del modelo Simulink CBR+VLM orquestado.
%
% Construye programaticamente 'modelo_cbr_vlm_orquestado.slx':
%   Constant (features 1x7) -> MATLAB Function (envoltura del orquestador) -> Displays
%
% El bloque MATLAB Function llama a orquestador_cbr_vlm (que a su vez usa
% vlm_cliente via coder.extrinsic), por lo que el modelo corre en
% INTERPRETED EXECUTION (load/webwrite/fileread no son codegen-friendly).
%
% Requisitos:
%   - Simulink instalado.
%   - >> exportar_base_casos   (genera datos/base_de_casos_cbr.mat)
%   - VLM opcional: Ollama local y/o OPENROUTER_API_KEY en .env.
%
% Uso:
%   >> crear_modelo_orquestado

    here = fileparts(mfilename('fullpath'));
    cd(here);

    if ~isfile(fullfile(here, 'datos', 'base_de_casos_cbr.mat'))
        fprintf('Generando base de casos primero...\n');
        exportar_base_casos();
    end

    % Crear (o regenerar) la funcion de envoltura para el bloque MATLAB Function
    escribir_envoltura(here);

    mdl = 'modelo_cbr_vlm_orquestado';
    if bdIsLoaded(mdl); close_system(mdl, 0); end
    new_system(mdl);
    open_system(mdl);

    % ── Constant: features de ejemplo (caso histeresis marcada) ─────────────
    feats = '[0.8922 1.2031 0.01753 0.04359 -0.002637 2.0806 30.9996]';
    add_block('simulink/Sources/Constant', [mdl '/Features'], ...
        'Value', feats, 'Position', [40 120 150 160]);

    % ── MATLAB Function con la envoltura del orquestador ────────────────────
    fcn = [mdl '/Orquestador CBR+VLM'];
    add_block('simulink/User-Defined Functions/MATLAB Function', fcn, ...
        'Position', [220 80 400 220]);
    % Inyectar el codigo de la envoltura en el bloque
    set_fcn_block_code(mdl, 'Orquestador CBR+VLM', fileread(fullfile(here, 'wrap_orquestador.m')));

    % ── Displays de las salidas escalares ───────────────────────────────────
    outs = {'clase_final','confianza','estado','flag_discrep','fuente_cod'};
    ypos = 40;
    for i = 1:numel(outs)
        d = [mdl '/' outs{i}];
        add_block('simulink/Sinks/Display', d, 'Position', [470 ypos 590 ypos+30]);
        add_line(mdl, sprintf('Orquestador CBR+VLM/%d', i), [outs{i} '/1'], 'autorouting','on');
        ypos = ypos + 55;
    end
    add_line(mdl, 'Features/1', 'Orquestador CBR+VLM/1', 'autorouting','on');

    set_param(mdl, 'SolverType','Fixed-step', 'Solver','FixedStepDiscrete', ...
              'FixedStep','1', 'StopTime','0');

    save_system(mdl, fullfile(here, [mdl '.slx']));
    fprintf('\nModelo creado: %s.slx\n', mdl);
    fprintf('Pulsa Run. Nota: si el VLM esta disponible, el caso de ejemplo (alta confianza)\n');
    fprintf('resuelve por CBR sin llamar al VLM (fuente_cod=1).\n\n');
end


function set_fcn_block_code(mdl, blockName, code)
% Inyecta codigo MATLAB en un bloque MATLAB Function via Stateflow API.
    rt = sfroot();
    chart = rt.find('-isa','Stateflow.EMChart','Path',[mdl '/' blockName]);
    chart.Script = code;
end


function escribir_envoltura(here)
% Genera wrap_orquestador.m: envoltura con I/O escalar para el bloque Simulink.
% Codifica 'fuente' (char) a un codigo numerico para poder mostrarlo en Display.
    code = [ ...
"function [clase_final, confianza, estado, flag_discrep, fuente_cod] = wrap_orquestador(features)" newline ...
"%#codegen" newline ...
"% Envoltura escalar del orquestador hibrido para el bloque MATLAB Function." newline ...
"% Corre en Interpreted execution (orquestador_cbr_vlm usa coder.extrinsic)." newline ...
"    coder.extrinsic('orquestador_cbr_vlm');" newline ...
"    out = struct('clase_final',0,'caso_sel',0,'accion','','confianza',0," ...
    + "'fuente','CBR','flag_discrepancia',false,'retain_flag',false," ...
    + "'motivo','','estado',0,'topk',struct());" newline ...
"    params = struct('K',3,'threshold',1.0,'tau_score',0.5,'tau_margen',0.05," ...
    + "'modelo','cloud','usar_vlm',true,'img_b64','');" newline ...
"    out = orquestador_cbr_vlm(features, params);" newline ...
"    clase_final = double(out.clase_final);" newline ...
"    confianza   = double(out.confianza);" newline ...
"    estado      = double(out.estado);" newline ...
"    flag_discrep= double(out.flag_discrepancia);" newline ...
"    fuente_cod  = fuente_a_cod(out.fuente);" newline ...
"end" newline ...
"function c = fuente_a_cod(f)" newline ...
"%#codegen" newline ...
"    c = 0;" newline ...
"    coder.varsize('f');" newline ...
"    if strcmp(f,'CBR'); c = 1;" newline ...
"    elseif strcmp(f,'CBR+audit'); c = 2;" newline ...
"    elseif strcmp(f,'VLM'); c = 3;" newline ...
"    elseif strcmp(f,'CBR(degradado)'); c = 4; end" newline ...
"end" newline ];
    fid = fopen(fullfile(here, 'wrap_orquestador.m'), 'w');
    fwrite(fid, char(join(code, "")));
    fclose(fid);
end

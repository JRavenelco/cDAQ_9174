function crear_modelo_simulink()
% CREAR_MODELO_SIMULINK  Parte 4b: arma un modelo Simulink de demo del CBR.
%
% Construye programaticamente un modelo con:
%   Constant (features 1x7)  ->  MATLAB System (CBRHysteresisSystem)  ->  Displays
%
% Asi no tienes que cablear nada a mano. Tras ejecutar, el modelo
% 'modelo_cbr_hysteresis.slx' queda abierto y listo para Run.
%
% Requisitos:
%   - Simulink instalado.
%   - Haber ejecutado  >> exportar_base_casos  (genera datos/base_de_casos_cbr.mat)
%
% Uso:
%   >> crear_modelo_simulink

    here = fileparts(mfilename('fullpath'));
    cd(here);   % para que las rutas relativas del .mat funcionen

    if ~isfile(fullfile(here, 'datos', 'base_de_casos_cbr.mat'))
        fprintf('Generando base de casos primero...\n');
        exportar_base_casos();
    end

    mdl = 'modelo_cbr_hysteresis';
    if bdIsLoaded(mdl); close_system(mdl, 0); end
    new_system(mdl);
    open_system(mdl);

    % ── Bloque Constant con un caso de ejemplo (features 1x7) ───────────────
    % Ejemplo: caso 'corte_20260311_131921_600_40' (histeresis marcada).
    %   [force_rms, force_peak_abs, input_rms, input_peak_abs,
    %    corr_force_input, loop_area_norm, duration_s]
    feats = '[0.8922 1.2031 0.01753 0.04359 -0.002637 2.0806 30.9996]';
    add_block('simulink/Sources/Constant', [mdl '/Features'], ...
        'Value', feats, 'Position', [40 100 140 140]);

    % ── Bloque MATLAB System con el Razonador CBR ───────────────────────────
    cbr = [mdl '/CBR Reasoner'];
    add_block('simulink/User-Defined Functions/MATLAB System', cbr, ...
        'System', 'CBRHysteresisSystem', ...
        'SimulateUsing', 'Interpreted execution', ...
        'Position', [220 80 360 200]);

    % ── Displays para las 4 salidas ─────────────────────────────────────────
    outs = {'class_id','min_dist','best_score','best_idx'};
    ypos = 40;
    for i = 1:numel(outs)
        d = [mdl '/' outs{i}];
        add_block('simulink/Sinks/Display', d, ...
            'Position', [440 ypos 540 ypos+30]);
        add_line(mdl, sprintf('CBR Reasoner/%d', i), [outs{i} '/1'], ...
                 'autorouting', 'on');
        ypos = ypos + 60;
    end

    % Conectar Constant -> CBR
    add_line(mdl, 'Features/1', 'CBR Reasoner/1', 'autorouting', 'on');

    % ── Configuracion de solver (discreto, 1 paso) ──────────────────────────
    set_param(mdl, 'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete', ...
              'FixedStep', '1', 'StopTime', '0');

    save_system(mdl, fullfile(here, [mdl '.slx']));
    fprintf('\nModelo creado: %s.slx\n', mdl);
    fprintf('Pulsa Run (o sim(''%s'')) para ver el diagnostico del caso de ejemplo.\n\n', mdl);
end

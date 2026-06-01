%% Ver y graficar resultados cDAQ 20260224_* (Live Editor)
% Abre/visualiza las figuras generadas en matlab_out/<timestamp>/
% Opción: re-ejecutar el análisis antes de visualizar.

clear; clc;

thisDir = fileparts(mfilename('fullpath'));
logsDir = fullfile(thisDir, 'logs');
outDir = fullfile(thisDir, 'matlab_out');

if ~isfolder(outDir)
    mkdir(outDir);
end

%% (Opcional) Re-ejecutar análisis
rerunAnalysis = false;
if usejava('desktop')
    a = questdlg('¿Quieres re-ejecutar el análisis antes de visualizar?', ...
        'Re-ejecutar análisis', 'Sí', 'No', 'No');
    rerunAnalysis = strcmp(a, 'Sí');
end

if rerunAnalysis
    if ~isfolder(logsDir)
        error('No existe logsDir: %s', logsDir);
    end
    run(fullfile(thisDir, 'analyze_cdaq_logs_20260224.m'));
end

%% Detectar timestamps disponibles
pngCandidates = dir(fullfile(outDir, '20260224_*', '*.png'));
figCandidates = dir(fullfile(outDir, '20260224_*', '*.fig'));

% Si todavía no hay outputs, intenta leer logs y sugerir ejecutar análisis
if isempty(pngCandidates) && isempty(figCandidates)
    fprintf('No hay figuras en %s\n', outDir);
    fprintf('Sugerencia: ejecuta analyze_cdaq_logs_20260224_live.m primero.\n');
end

tsFromOut = {};
for i = 1:numel(pngCandidates)
    tsFromOut{end+1} = string(pngCandidates(i).folder); %#ok<SAGROW>
end
for i = 1:numel(figCandidates)
    tsFromOut{end+1} = string(figCandidates(i).folder); %#ok<SAGROW>
end

timestamps = unique(extractAfter(string(tsFromOut), outDir + filesep), 'stable');
timestamps = unique(extractBefore(timestamps, filesep), 'stable');

timestamps(timestamps == "") = [];
if isempty(timestamps)
    % fallback: buscar por nombres en logs
    if isfolder(logsDir)
        rawCandidates = dir(fullfile(logsDir, 'cdaq_remote_20260224_*.csv'));
        rawCandidates = rawCandidates(~contains({rawCandidates.name}, '_inference.csv'));
        for i = 1:numel(rawCandidates)
            tok = regexp(rawCandidates(i).name, '^cdaq_remote_(\d{8}_\d{6})\.csv$', 'tokens', 'once');
            if ~isempty(tok)
                timestamps{end+1} = string(tok{1}); %#ok<SAGROW>
            end
        end
        timestamps = unique([timestamps{:}], 'stable');
    end
end

fprintf('Timestamps detectados (%d):\n', numel(timestamps));
for i = 1:numel(timestamps)
    fprintf('  %s\n', timestamps(i));
end

if isempty(timestamps)
    return;
end

%% Seleccionar timestamp a visualizar
selTs = timestamps(1);
if usejava('desktop')
    [idx, ok] = listdlg('PromptString','Selecciona timestamp:', ...
        'SelectionMode','single', 'ListString', cellstr(timestamps));
    if ok
        selTs = timestamps(idx);
    end
end

tsDir = fullfile(outDir, selTs);
if ~isfolder(tsDir)
    error('No existe carpeta de salida: %s', tsDir);
end

fprintf('\nVisualizando: %s\n', tsDir);

%% Abrir figuras .fig (interactivas)
figFiles = {
    'raw_full.fig'
    'raw_processing_20251210.fig'
    'raw_rms_per_block.fig'
    'raw_segment.fig'
    'raw_hist.fig'
    'inference_timeseries.fig'
    'inference_cutting.fig'
    'inference_features.fig'
};

for i = 1:numel(figFiles)
    p = fullfile(tsDir, figFiles{i});
    if isfile(p)
        try
            openfig(p, 'new', 'visible');
        catch
        end
    end
end

%% Mostrar PNG inline (para Live Editor)
% (en Live Editor verás las imágenes en el output del script)

pngFiles = {
    'raw_full.png'
    'raw_processing_20251210.png'
    'raw_rms_per_block.png'
    'raw_segment.png'
    'raw_hist.png'
    'inference_timeseries.png'
    'inference_cutting.png'
    'inference_features.png'
};

for i = 1:numel(pngFiles)
    p = fullfile(tsDir, pngFiles{i});
    if isfile(p)
        fprintf('\n%s\n', pngFiles{i});
        try
            I = imread(p);
            figure('Name', pngFiles{i});
            imshow(I);
            title(pngFiles{i}, 'Interpreter', 'none');
        catch
            fprintf('No se pudo mostrar %s (pero existe).\n', p);
        end
    end
end

%% Cargar summary y mostrar fila del timestamp
summaryMat = fullfile(outDir, 'summary.mat');
summaryCsv = fullfile(outDir, 'summary.csv');

if isfile(summaryCsv)
    try
        T = readtable(summaryCsv);
        row = T(strcmp(string(T.timestamp), string(selTs)), :);
        fprintf('\nFila summary.csv para %s:\n', selTs);
        disp(row);
    catch
    end
end

if isfile(summaryMat)
    try
        S = load(summaryMat);
        summary = S.summary;
        idx = find(strcmp(string({summary.items.timestamp}), string(selTs)), 1);
        if ~isempty(idx)
            fprintf('\nEstructura summary.items(%d):\n', idx);
            disp(summary.items(idx));
        end
    catch
    end
end

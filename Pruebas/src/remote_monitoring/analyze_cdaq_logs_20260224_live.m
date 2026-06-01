%% Live analysis: cDAQ logs 20260224_* (local o Jetson por Ethernet)
clear; clc;

thisDir = fileparts(mfilename('fullpath'));
logsDir = fullfile(thisDir, 'logs');
outDir = fullfile(thisDir, 'matlab_out');

if ~isfolder(logsDir)
    mkdir(logsDir);
end
if ~isfolder(outDir)
    mkdir(outDir);
end

%% Selección de modo
mode = "local";

if ~usejava('desktop')
    mode = "local";
else
    a = questdlg('¿De dónde quieres obtener los logs?', 'Logs 20260224_*', ...
        'Local (ya descargados)', 'Descargar desde Jetson (scp)', 'Local (ya descargados)');
    if strcmp(a, 'Descargar desde Jetson (scp)')
        mode = "jetson_scp";
    else
        mode = "local";
    end
end

%% (Opcional) Descargar desde Jetson usando scp
if mode == "jetson_scp"
    jetsonUser = "ubuntu";
    jetsonHost = "192.168.1.10";
    remoteLogsDir = "/home/ubuntu/cDAQ_9174/Pruebas/src/remote_monitoring/logs";

    if usejava('desktop')
        answ = inputdlg({"Jetson user","Jetson host/IP","Remote logs dir"}, ...
            "Config scp", [1 120], {char(jetsonUser), char(jetsonHost), char(remoteLogsDir)});
        if isempty(answ)
            error('Cancelado por el usuario.');
        end
        jetsonUser = string(answ{1});
        jetsonHost = string(answ{2});
        remoteLogsDir = string(answ{3});
    end

    pattern = "cdaq_remote_20260224_*";

    localDirArg = sprintf('"%s\\"', logsDir);
    remoteSpec = sprintf('%s@%s:%s/%s', jetsonUser, jetsonHost, remoteLogsDir, pattern);

    cmd = sprintf('scp %s %s', remoteSpec, localDirArg);
    fprintf('Ejecutando: %s\n', cmd);
    [st, out] = system(cmd);
    fprintf('%s\n', out);
    if st ~= 0
        error('Falló scp. Revisa conectividad/SSH keys y vuelve a intentar.');
    end
end

%% Verificar inventario local
raw = dir(fullfile(logsDir, 'cdaq_remote_20260224_*.csv'));
raw = raw(~contains({raw.name}, '_inference.csv'));
inf = dir(fullfile(logsDir, 'cdaq_remote_20260224_*_inference.csv'));
cases = dir(fullfile(logsDir, 'cdaq_remote_20260224_*_cases.jsonl'));

fprintf('\nInventario en: %s\n', logsDir);
fprintf('  raw: %d\n', numel(raw));
fprintf('  inference: %d\n', numel(inf));
fprintf('  cases: %d\n', numel(cases));

if isempty(raw) && isempty(inf) && isempty(cases)
    error('No hay archivos 20260224_* en logsDir.');
end

%% Ejecutar análisis principal
cd(thisDir);
run(fullfile(thisDir, 'analyze_cdaq_logs_20260224.m'));

%% Abrir outputs principales
summaryCsv = fullfile(outDir, 'summary.csv');
summaryMat = fullfile(outDir, 'summary.mat');

if isfile(summaryCsv)
    fprintf('\nOK: %s\n', summaryCsv);
end
if isfile(summaryMat)
    fprintf('OK: %s\n', summaryMat);
end

if usejava('desktop') && isfile(summaryCsv)
    try
        T = readtable(summaryCsv);
        disp(T);
    catch
    end
end

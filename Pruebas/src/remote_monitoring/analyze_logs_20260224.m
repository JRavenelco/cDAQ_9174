%% cDAQ logs 20260224_* - loader + plots

close all; clear; clc;

%% Config
scriptDir = fileparts(mfilename('fullpath'));
logDir = fullfile(scriptDir, 'logs');

patternRaw = fullfile(logDir, 'cdaq_remote_20260224_*.csv');
rawFiles = dir(patternRaw);
rawFiles = rawFiles(~endsWith({rawFiles.name}, '_inference.csv'));
rawFiles = rawFiles(~endsWith({rawFiles.name}, '_cases.jsonl'));
rawFiles = rawFiles(~contains({rawFiles.name}, '_inference.csv'));
rawFiles = rawFiles(~contains({rawFiles.name}, '_cases.jsonl'));

if isempty(rawFiles)
    error('No se encontraron raw CSVs con patrón: %s', patternRaw);
end

skipInferenceTs = "20260224_125616";  % inferencia degenerada conocida
maxPlotPoints = 200000;

%% Utilidades
getTs = @(fname) extractBetween(string(fname), "cdaq_remote_", ".csv");

readTableWithComments = @(p) readtable(p, detectImportOptions(p, 'CommentStyle', '#'));

function out = readCasesJsonl(casesPath)
    out = struct('items', [], 'jsonErrors', 0, 'n', 0);
    if ~isfile(casesPath)
        return;
    end

    lines = readlines(casesPath);
    items = cell(0,1);
    jsonErr = 0;

    for i = 1:numel(lines)
        s = strtrim(lines(i));
        if strlength(s) == 0
            continue;
        end
        try
            obj = jsondecode(s);
        catch
            jsonErr = jsonErr + 1;
            continue;
        end

        if isfield(obj, 'llm') && isstruct(obj.llm) && isfield(obj.llm, 'response') && ischar(obj.llm.response)
            try
                obj.llm.response_json = jsondecode(string(obj.llm.response));
            catch
                obj.llm.response_json = [];
            end
        end

        items{end+1,1} = obj; %#ok<AGROW>
    end

    out.items = items;
    out.jsonErrors = jsonErr;
    out.n = numel(items);
end

function tf = isInferenceDegenerate(T)
    tf = false;
    if isempty(T) || height(T) < 1
        return;
    end

    keys = ["F_est_mean_V","envelope_rms_g","freq_dom_Hz","THD_accel_pct","force_rms_V","accel_rms_g","R2_block"]; 
    keys = keys(ismember(keys, string(T.Properties.VariableNames)));
    if numel(keys) < 3
        return;
    end

    x = T{1, keys};
    x = x(:);
    if all(isfinite(x)) && std(double(x)) < 1e-12
        tf = true;
    end
end

function idx = downsampleIdx(n, maxPts)
    if n <= maxPts
        idx = 1:n;
        return;
    end
    stride = max(1, floor(n / maxPts));
    idx = 1:stride:n;
end

%% Cargar + graficar
results = struct();

for k = 1:numel(rawFiles)
    rawPath = fullfile(rawFiles(k).folder, rawFiles(k).name);
    ts = getTs(rawFiles(k).name);
    if isempty(ts)
        continue;
    end
    ts = string(ts(1));

    infPath = fullfile(logDir, "cdaq_remote_" + ts + "_inference.csv");
    casesPath = fullfile(logDir, "cdaq_remote_" + ts + "_cases.jsonl");

    fprintf('\n=== %s ===\n', ts);

    Traw = readTableWithComments(rawPath);
    fprintf('raw: rows=%d\n', height(Traw));

    Tinf = table();
    fs = NaN;
    nSamples = NaN;

    if isfile(infPath)
        TinfAll = readTableWithComments(infPath);
        if ismember('fs_hz', TinfAll.Properties.VariableNames)
            fs = mode(TinfAll.fs_hz);
        end
        if ismember('n_samples', TinfAll.Properties.VariableNames)
            nSamples = mode(TinfAll.n_samples);
        end

        deg = (ts == skipInferenceTs) || isInferenceDegenerate(TinfAll);
        if deg
            fprintf('infer: OMITIDA (degenerada). Se conserva solo meta fs_hz/n_samples.\n');
        else
            Tinf = TinfAll;
            fprintf('infer: rows=%d  fs=%.1f  n_samples=%d\n', height(Tinf), fs, nSamples);
        end
    else
        fprintf('infer: (no existe)\n');
    end

    cases = readCasesJsonl(casesPath);
    if isfile(casesPath)
        fprintf('cases: lines=%d  json_errors=%d\n', cases.n, cases.jsonErrors);
    else
        fprintf('cases: (no existe)\n');
    end

    if ~ismember('t_local_s', Traw.Properties.VariableNames) || ~ismember('sample_idx', Traw.Properties.VariableNames)
        warning('raw no tiene t_local_s/sample_idx; se omite plot raw.');
        tRaw = [];
    else
        if ~isfinite(fs)
            fs = 2000.0;
        end
        tRaw = Traw.t_local_s + double(Traw.sample_idx) ./ double(fs);
        tRaw = tRaw - tRaw(1);
    end

    if ~isempty(Traw) && ismember('force_v', Traw.Properties.VariableNames) && ismember('accel_g', Traw.Properties.VariableNames) && ~isempty(tRaw)
        idxRaw = downsampleIdx(height(Traw), maxPlotPoints);

        figure('Name', "RAW " + ts);
        tiledlayout(2,1);

        nexttile;
        plot(tRaw(idxRaw), Traw.force_v(idxRaw));
        grid on;
        xlabel('t (s)'); ylabel('force_v');
        title("Raw force_v - " + ts);

        nexttile;
        plot(tRaw(idxRaw), Traw.accel_g(idxRaw));
        grid on;
        xlabel('t (s)'); ylabel('accel_g');
        title("Raw accel_g - " + ts);
    end

    if ~isempty(Tinf) && ismember('t_local_s', Tinf.Properties.VariableNames)
        tInf = Tinf.t_local_s - Tinf.t_local_s(1);

        figure('Name', "INFER " + ts);
        tiledlayout(3,2);

        defPlot = @(name) (ismember(name, Tinf.Properties.VariableNames) && ~all(isnan(Tinf.(name))));

        nexttile; if defPlot('F_est_mean_V'), plot(tInf, Tinf.F_est_mean_V); grid on; title('F\_est\_mean\_V'); xlabel('t (s)'); end
        nexttile; if defPlot('envelope_rms_g'), plot(tInf, Tinf.envelope_rms_g); grid on; title('envelope\_rms\_g'); xlabel('t (s)'); end
        nexttile; if defPlot('freq_dom_Hz'), plot(tInf, Tinf.freq_dom_Hz); grid on; title('freq\_dom\_Hz'); xlabel('t (s)'); end
        nexttile; if defPlot('THD_accel_pct'), plot(tInf, Tinf.THD_accel_pct); grid on; title('THD\_accel\_pct'); xlabel('t (s)'); end
        nexttile; if defPlot('accel_rms_g'), plot(tInf, Tinf.accel_rms_g); grid on; title('accel\_rms\_g'); xlabel('t (s)'); end
        nexttile; if defPlot('force_rms_V'), plot(tInf, Tinf.force_rms_V); grid on; title('force\_rms\_V'); xlabel('t (s)'); end
    end

    results.(matlab.lang.makeValidName(char(ts))) = struct(
        'ts', ts,
        'rawPath', rawPath,
        'infPath', infPath,
        'casesPath', casesPath,
        'fs_hz', fs,
        'n_samples', nSamples,
        'raw', Traw,
        'infer', Tinf,
        'cases', cases);
end

%% Guardar resumen (opcional)
outDir = fullfile(scriptDir, 'matlab_out');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
save(fullfile(outDir, 'results_20260224.mat'), 'results', '-v7.3');
fprintf('\nGuardado: %s\n', fullfile(outDir, 'results_20260224.mat'));

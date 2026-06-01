clear; clc;

thisDir = fileparts(mfilename('fullpath'));
logsDir = fullfile(thisDir, 'logs');
if ~isfolder(logsDir)
    error('No existe la carpeta de logs: %s', logsDir);
end

outDir = fullfile(thisDir, 'matlab_out');
if ~isfolder(outDir)
    mkdir(outDir);
end

rawCandidates = dir(fullfile(logsDir, 'cdaq_remote_20260224_*.csv'));
rawCandidates = rawCandidates(~contains({rawCandidates.name}, '_inference.csv'));

infCandidates = dir(fullfile(logsDir, 'cdaq_remote_20260224_*_inference.csv'));
casesCandidates = dir(fullfile(logsDir, 'cdaq_remote_20260224_*_cases.jsonl'));

timestamps = {};
for i = 1:numel(rawCandidates)
    tok = regexp(rawCandidates(i).name, '^cdaq_remote_(\d{8}_\d{6})\.csv$', 'tokens', 'once');
    if ~isempty(tok)
        timestamps{end+1} = tok{1}; %#ok<SAGROW>
    end
end

for i = 1:numel(infCandidates)
    tok = regexp(infCandidates(i).name, '^cdaq_remote_(\d{8}_\d{6})_inference\.csv$', 'tokens', 'once');
    if ~isempty(tok)
        timestamps{end+1} = tok{1}; %#ok<SAGROW>
    end
end

for i = 1:numel(casesCandidates)
    tok = regexp(casesCandidates(i).name, '^cdaq_remote_(\d{8}_\d{6})_cases\.jsonl$', 'tokens', 'once');
    if ~isempty(tok)
        timestamps{end+1} = tok{1}; %#ok<SAGROW>
    end
end

timestamps = unique(timestamps, 'stable');
if isempty(timestamps)
    fprintf('No se encontraron timestamps 20260224_* en: %s\n', logsDir);
    return;
end

fprintf('Timestamps encontrados (%d):\n', numel(timestamps));
for i = 1:numel(timestamps)
    fprintf('  %s\n', timestamps{i});
end

summary = struct();
summary.logsDir = logsDir;
summary.outDir = outDir;
summary.timestamps = timestamps;

itemTmpl = struct();
itemTmpl.timestamp = "";
itemTmpl.paths = struct('raw', "", 'inference', "", 'cases', "");
itemTmpl.inventory = struct();
itemTmpl.quality = struct();
itemTmpl.features = struct();
itemTmpl.cases = struct('count', 0);

summary.items = repmat(itemTmpl, 0, 1);

for k = 1:numel(timestamps)
    ts = timestamps{k};
    fprintf('\n=== %s ===\n', ts);

    rawPath = fullfile(logsDir, sprintf('cdaq_remote_%s.csv', ts));
    infPath = fullfile(logsDir, sprintf('cdaq_remote_%s_inference.csv', ts));
    casesPath = fullfile(logsDir, sprintf('cdaq_remote_%s_cases.jsonl', ts));

    item = itemTmpl;
    item.timestamp = ts;
    item.paths = struct('raw', rawPath, 'inference', infPath, 'cases', casesPath);

    item.inventory = inventory_files(rawPath, infPath, casesPath);

    infer = table();
    if isfile(infPath)
        infer = read_csv_with_comments(infPath);
        item.inventory.inference_rows = height(infer);
    end

    cases = struct();
    if isfile(casesPath)
        [cases, caseInfo] = read_cases_jsonl(casesPath);
        item.inventory.cases_lines = caseInfo.nLines;
        item.inventory.cases_bad_lines = caseInfo.nBad;
    end

    item.quality = struct();
    item.quality.seq_gaps = [];
    item.quality.seq_min = NaN;
    item.quality.seq_max = NaN;
    item.quality.duration_s_est = NaN;

    if ~isempty(infer)
        [qualityInfer, seqMeta] = quality_inference(infer);
        item.quality = merge_struct(item.quality, qualityInfer);
    else
        seqMeta = struct('seqMin', NaN, 'seqMax', NaN, 'seqRangeN', 0, 'expectedNSamp', [], 'expectedFs', [], 'tBlock', []);
    end

    rawStats = struct();
    if isfile(rawPath) && ~isempty(infer)
        rawStats = validate_raw_vs_infer(rawPath, seqMeta);
        item.quality = merge_struct(item.quality, rawStats.quality);
        item.inventory.raw_rows = rawStats.quality.raw_rows;
    elseif isfile(rawPath)
        item.quality.raw_note = 'Inference CSV no disponible: se omiten checks por bloque (n_samples/fs_hz).';
        item.inventory.raw_rows = count_csv_data_rows(rawPath);
    end

    item.features = struct();
    if ~isempty(infer)
        item.features = feature_checks(infer);
        if isfield(item.features, 'degenerate')
            nDeg = height(item.features.degenerate);
            fprintf('Features degeneradas (var < thr): %d\n', nDeg);
            if nDeg > 0
                disp(item.features.degenerate(1:min(10, nDeg), :));
            end
        end
    end

    outTsDir = fullfile(outDir, ts);
    if ~isfolder(outTsDir)
        mkdir(outTsDir);
    end

    if ~isempty(infer)
        make_plots_inference(infer, outTsDir);
    end
    if isfield(rawStats, 'plot')
        make_plots_raw(rawStats.plot, outTsDir);
    end
    if isfield(rawStats, 'plot_full')
        make_plots_raw_full(rawStats.plot_full, outTsDir);
    end
    if isfield(rawStats, 'full')
        make_plots_processing_20251210(rawStats.full, outTsDir);
    end
    if isfield(rawStats, 'rms') && ~isempty(infer)
        make_plots_raw_rms(rawStats.rms, seqMeta, outTsDir);
    end

    item.cases = struct();
    item.cases.count = numel(cases);

    summary.items(end+1, 1) = item; %#ok<SAGROW>
end

save(fullfile(outDir, 'summary.mat'), 'summary', '-v7.3');
summaryTable = summary_to_table(summary);
writetable(summaryTable, fullfile(outDir, 'summary.csv'));

fprintf('\nOK. Salida en: %s\n', outDir);
fprintf('  - summary.mat\n');
fprintf('  - summary.csv\n');


function inv = inventory_files(rawPath, infPath, casesPath)
inv = struct();
inv.raw_exists = isfile(rawPath);
inv.inference_exists = isfile(infPath);
inv.cases_exists = isfile(casesPath);

inv.raw_bytes = file_size(rawPath);
inv.inference_bytes = file_size(infPath);
inv.cases_bytes = file_size(casesPath);

inv.raw_rows = NaN;
inv.inference_rows = NaN;
inv.cases_lines = NaN;
inv.cases_bad_lines = NaN;
end


function n = count_csv_data_rows(path)
if ~isfile(path)
    n = NaN;
    return;
end
ds = tabularTextDatastore(path, 'Delimiter', ',', 'CommentStyle', '#');
ds.ReadSize = 300000;
n = 0;
while hasdata(ds)
    T = read(ds);
    n = n + height(T);
end
end


function bytes = file_size(p)
if ~isfile(p)
    bytes = NaN;
    return;
end
d = dir(p);
bytes = d.bytes;
end


function T = read_csv_with_comments(path)
opts = detectImportOptions(path, 'CommentStyle', '#');
vars = opts.VariableNames;

try
    types = opts.VariableTypes;
catch
    types = repmat({''}, size(vars));
end

isText = false(size(vars));
for i = 1:numel(vars)
    t = '';
    if i <= numel(types)
        t = lower(string(types{i}));
    end
    isText(i) = any(t == ["char" "string" "categorical"]);
end

textVars = vars(isText);

if ~isempty(textVars)
    try
        opts = setvaropts(opts, textVars, 'WhitespaceRule', 'preserve');
    catch
    end
    try
        opts = setvaropts(opts, textVars, 'EmptyFieldRule', 'auto');
    catch
    end
end
T = readtable(path, opts);
end


function [cases, info] = read_cases_jsonl(path)
lines = readlines(path);
lines = lines(strlength(strtrim(lines)) > 0);

tmpl = struct();
tmpl.t_local_s = NaN;
tmpl.seq = NaN;
tmpl.meta = struct();
tmpl.features = struct();
tmpl.llm = struct('raw', [], 'response_parsed', []);
tmpl.raw = struct();

cases = repmat(tmpl, 0, 1);
info = struct('nLines', numel(lines), 'nBad', 0);

for i = 1:numel(lines)
    s = strtrim(lines(i));
    try
        obj = jsondecode(s);
    catch
        info.nBad = info.nBad + 1;
        continue;
    end

    row = tmpl;
    row.raw = obj;
    if isfield(obj, 't_local_s')
        row.t_local_s = double(obj.t_local_s);
    end
    if isfield(obj, 'seq')
        row.seq = double(obj.seq);
    end
    if isfield(obj, 'meta')
        row.meta = obj.meta;
    end
    if isfield(obj, 'features')
        row.features = obj.features;
    end

    if isfield(obj, 'llm')
        row.llm.raw = obj.llm;
        llm = obj.llm;
        if isstruct(llm) && isfield(llm, 'response')
            if ischar(llm.response) || (isstring(llm.response) && isscalar(llm.response))
                txt = char(llm.response);
                txt = strtrim(txt);
                if ~isempty(txt)
                    try
                        row.llm.response_parsed = jsondecode(txt);
                    catch
                        row.llm.response_parsed = txt;
                    end
                end
            end
        end
    end

    cases(end+1, 1) = row; %#ok<SAGROW>
end
end


function [q, seqMeta] = quality_inference(infer)
q = struct();
seq = double(infer.seq);
[seqSorted, idx] = sort(seq);

q.seq_min = min(seqSorted);
q.seq_max = max(seqSorted);

dseq = diff(seqSorted);
ix = find(dseq > 1);
gaps = [];
for i = 1:numel(ix)
    gaps(end+1, :) = [seqSorted(ix(i)), seqSorted(ix(i)+1), dseq(ix(i))]; %#ok<SAGROW>
end
q.seq_gaps = gaps;

fs = double(infer.fs_hz);
ns = double(infer.n_samples);
valid = (fs > 0) & (ns > 0);
if any(valid)
    q.duration_s_est = sum(ns(valid) ./ fs(valid));
else
    q.duration_s_est = NaN;
end

seqMin = q.seq_min;
seqMax = q.seq_max;
seqRangeN = seqMax - seqMin + 1;
expectedNSamp = nan(seqRangeN, 1);
expectedFs = nan(seqRangeN, 1);
tBlock = nan(seqRangeN, 1);

for i = 1:height(infer)
    s = double(infer.seq(i));
    j = s - seqMin + 1;
    expectedNSamp(j) = double(infer.n_samples(i));
    expectedFs(j) = double(infer.fs_hz(i));
    if ismember('t_local_s', infer.Properties.VariableNames)
        tBlock(j) = double(infer.t_local_s(i));
    end
end

seqMeta = struct('seqMin', seqMin, 'seqMax', seqMax, 'seqRangeN', seqRangeN, ...
    'expectedNSamp', expectedNSamp, 'expectedFs', expectedFs, 'tBlock', tBlock);
end


function rawStats = validate_raw_vs_infer(rawPath, seqMeta)
seqMin = seqMeta.seqMin;
seqMax = seqMeta.seqMax;
seqRangeN = seqMeta.seqRangeN;
expectedNSamp = seqMeta.expectedNSamp;
expectedFs = seqMeta.expectedFs;
tBlock = seqMeta.tBlock;

rawCount = zeros(seqRangeN, 1);
rawMinIdx = inf(seqRangeN, 1);
rawMaxIdx = -inf(seqRangeN, 1);
rawSumIdx = zeros(seqRangeN, 1);
rawSumSqIdx = zeros(seqRangeN, 1);
rawSumSqForce = zeros(seqRangeN, 1);
rawSumSqAccel = zeros(seqRangeN, 1);

maxPlotSamples = 200000;
plot_t = zeros(0, 1);
plot_force = zeros(0, 1);
plot_accel = zeros(0, 1);

desiredPlotPoints = 300000;
totalExpected = nansum(expectedNSamp);
stride = max(1, floor(double(totalExpected) / desiredPlotPoints));

maxFullSamples = 3000000;
keepFull = isfinite(totalExpected) && totalExpected > 0 && totalExpected <= maxFullSamples;
if keepFull
    full_force = nan(totalExpected, 1);
    full_accel = nan(totalExpected, 1);
    full_t = nan(totalExpected, 1);

    expectedMask = ~isnan(expectedNSamp) & expectedNSamp > 0;
    ns0 = expectedNSamp;
    ns0(~expectedMask) = 0;
    seqStart = nan(seqRangeN, 1);
    seqStart(expectedMask) = 1 + [0; cumsum(ns0(1:end-1))];
else
    seqStart = [];
end

plot_full_t = zeros(0, 1);
plot_full_force = zeros(0, 1);
plot_full_accel = zeros(0, 1);
globalCtr = 0;

ds = tabularTextDatastore(rawPath, 'Delimiter', ',', 'CommentStyle', '#');
ds.SelectedVariableNames = intersect(ds.VariableNames, {'seq','t_sender_s','t_local_s','sample_idx','force_v','accel_g'}, 'stable');
ds.ReadSize = 250000;

while hasdata(ds)
    T = read(ds);
    if isempty(T)
        continue;
    end

    s = double(T.seq);
    jAll = s - seqMin + 1;
    ok = (jAll >= 1) & (jAll <= seqRangeN);
    if ~any(ok)
        globalCtr = globalCtr + height(T);
        continue;
    end

    j = jAll(ok);
    samp = double(T.sample_idx(ok));
    fv = double(T.force_v(ok));
    ag = double(T.accel_g(ok));
    if ismember('t_local_s', T.Properties.VariableNames)
        t0 = double(T.t_local_s(ok));
    else
        t0 = double(T.t_sender_s(ok));
    end

    if keepFull
        start = seqStart(j);
        fsRow = expectedFs(j);
        t0Row = tBlock(j);
        okFull = ~isnan(start) & ~isnan(fsRow) & (fsRow > 0) & ~isnan(t0Row) & ~isnan(samp);
        if any(okFull)
            startK = start(okFull);
            fsK = fsRow(okFull);
            t0K = t0Row(okFull);
            sampK2 = samp(okFull);
            fvK2 = fv(okFull);
            agK2 = ag(okFull);

            idx = startK + sampK2;
            idx = idx(:);
            validIdx = isfinite(idx) & idx >= 1 & idx <= totalExpected;
            if any(validIdx)
                idx = idx(validIdx);
                full_force(idx) = fvK2(validIdx);
                full_accel(idx) = agK2(validIdx);
                full_t(idx) = t0K(validIdx) + sampK2(validIdx) ./ fsK(validIdx);
            end
        end
    end

    nChunk = height(T);
    idxAll = (globalCtr + (1:nChunk))';
    globalCtr = globalCtr + nChunk;
    keep = false(nChunk, 1);
    keep(ok) = mod(idxAll(ok), stride) == 0;
    if any(keep)
        sampK = double(T.sample_idx(keep));
        fvK = double(T.force_v(keep));
        agK = double(T.accel_g(keep));
        sK = double(T.seq(keep));
        jK = sK - seqMin + 1;
        if ismember('t_local_s', T.Properties.VariableNames)
            t0K = double(T.t_local_s(keep));
        else
            t0K = double(T.t_sender_s(keep));
        end
        fsK = expectedFs(jK);
        goodFs = fsK > 0;
        tSampK = nan(numel(sampK), 1);
        tSampK(goodFs) = t0K(goodFs) + sampK(goodFs) ./ fsK(goodFs);
        plot_full_t = [plot_full_t; tSampK]; %#ok<AGROW>
        plot_full_force = [plot_full_force; fvK]; %#ok<AGROW>
        plot_full_accel = [plot_full_accel; agK]; %#ok<AGROW>
    end

    for u = unique(j)'
        mask = (j == u);
        rawCount(u) = rawCount(u) + sum(mask);
        rawMinIdx(u) = min(rawMinIdx(u), min(samp(mask)));
        rawMaxIdx(u) = max(rawMaxIdx(u), max(samp(mask)));
        rawSumIdx(u) = rawSumIdx(u) + sum(samp(mask));
        rawSumSqIdx(u) = rawSumSqIdx(u) + sum(samp(mask).^2);
        rawSumSqForce(u) = rawSumSqForce(u) + sum(fv(mask).^2);
        rawSumSqAccel(u) = rawSumSqAccel(u) + sum(ag(mask).^2);
    end

    if numel(plot_force) < maxPlotSamples
        nRemain = maxPlotSamples - numel(plot_force);
        take = min(nRemain, numel(samp));
        sampTake = samp(1:take);
        jTake = j(1:take);
        tTake = t0(1:take);
        fsTake = expectedFs(jTake);
        goodFs = fsTake > 0;
        tSample = nan(take, 1);
        tSample(goodFs) = tTake(goodFs) + sampTake(goodFs) ./ fsTake(goodFs);
        plot_t = [plot_t; tSample]; %#ok<AGROW>
        plot_force = [plot_force; fv(1:take)]; %#ok<AGROW>
        plot_accel = [plot_accel; ag(1:take)]; %#ok<AGROW>
    end
end

expectedMask = ~isnan(expectedNSamp) & expectedNSamp > 0;
expN = expectedNSamp;

refSum = expN .* (expN - 1) / 2;
refSumSq = expN .* (expN - 1) .* (2 * expN - 1) / 6;

okCount = (rawCount == expN);
okMinMax = (rawMinIdx == 0) & (rawMaxIdx == expN - 1);
okSum = abs(rawSumIdx - refSum) < 1e-6;
okSumSq = abs(rawSumSqIdx - refSumSq) < 1e-3;

blockOk = expectedMask & okCount & okMinMax & okSum & okSumSq;
blockBad = expectedMask & ~blockOk;

rawStats = struct();
rawStats.quality = struct();
rawStats.quality.raw_rows = sum(rawCount);
rawStats.quality.raw_blocks_expected = sum(expectedMask);
rawStats.quality.raw_blocks_ok = sum(blockOk);
rawStats.quality.raw_blocks_bad = sum(blockBad);

badSeqIdx = find(blockBad);
rawStats.quality.raw_bad_seq = (badSeqIdx(:) + seqMin - 1)';

rawRmsForce = nan(seqRangeN, 1);
rawRmsAccel = nan(seqRangeN, 1);
nonzero = rawCount > 0;
rawRmsForce(nonzero) = sqrt(rawSumSqForce(nonzero) ./ rawCount(nonzero));
rawRmsAccel(nonzero) = sqrt(rawSumSqAccel(nonzero) ./ rawCount(nonzero));
rawStats.rms = struct('force_v', rawRmsForce, 'accel_g', rawRmsAccel);

pt = plot_t;
if ~isempty(pt)
    pt0 = pt(find(~isnan(pt), 1, 'first'));
    pt = pt - pt0;
end
rawStats.plot = struct('t_s', pt, 'force_v', plot_force, 'accel_g', plot_accel);

ptf = plot_full_t;
if ~isempty(ptf)
    ptf0 = ptf(find(~isnan(ptf), 1, 'first'));
    ptf = ptf - ptf0;
end
rawStats.plot_full = struct('t_s', ptf, 'force_v', plot_full_force, 'accel_g', plot_full_accel, ...
    'stride', stride, 'n_expected', double(totalExpected));

if keepFull
    tf = full_t;
    if ~isempty(tf)
        tf0 = tf(find(~isnan(tf), 1, 'first'));
        tf = tf - tf0;
    end
    rawStats.full = struct('t_s', tf, 'force_v', full_force, 'accel_g', full_accel, ...
        'fs_hz_median', median(expectedFs(~isnan(expectedFs) & expectedFs > 0)));
end
end


function f = feature_checks(infer)
vars = infer.Properties.VariableNames;
numMask = varfun(@isnumeric, infer, 'OutputFormat', 'uniform');
numVars = vars(numMask);

stats = table('Size', [numel(numVars) 3], 'VariableTypes', {'string','double','double'}, ...
    'VariableNames', {'name','std','var'});

for i = 1:numel(numVars)
    x = double(infer.(numVars{i}));
    stats.name(i) = string(numVars{i});
    stats.std(i) = std(x, 0, 'omitnan');
    stats.var(i) = var(x, 0, 'omitnan');
end

thr = 1e-8;
deg = stats(stats.var < thr, :);

corrs = struct();
if all(ismember({'envelope_rms_g','accel_rms_g'}, vars))
    corrs.envelope_vs_accel_rms = corr(double(infer.envelope_rms_g), double(infer.accel_rms_g), 'Rows', 'complete');
end

f = struct();
f.numeric_stats = stats;
f.degenerate = deg;
f.correlations = corrs;
end


function make_plots_inference(infer, outTsDir)
vars = infer.Properties.VariableNames;

if ismember('t_local_s', vars)
    t = double(infer.t_local_s);
    t = t - t(1);
else
    t = (0:height(infer)-1)';
end

plot_series(t, infer, outTsDir, {'F_est_mean_V','envelope_rms_g','freq_dom_Hz','THD_accel_pct'}, 'inference_features');
plot_series(t, infer, outTsDir, {'cutting_flag','cut_id'}, 'inference_cutting');
end


function plot_series(t, infer, outTsDir, names, fname)
vars = infer.Properties.VariableNames;
keep = names(ismember(names, vars));
if isempty(keep)
    return;
end

fig = figure('Visible', 'off');
ax = axes(fig);
hold(ax, 'on');

for i = 1:numel(keep)
    y = double(infer.(keep{i}));
    plot(ax, t, y, 'DisplayName', keep{i});
end

grid(ax, 'on');
xlabel(ax, 't (s, relativo)');
legend(ax, 'Location', 'best');

exportgraphics(fig, fullfile(outTsDir, fname + ".png"), 'Resolution', 200);
savefig(fig, fullfile(outTsDir, fname + ".fig"));
close(fig);
end


function make_plots_raw(rawPlot, outTsDir)
mask = ~isnan(rawPlot.t_s);
t = rawPlot.t_s(mask);
fv = rawPlot.force_v(mask);
ag = rawPlot.accel_g(mask);

if isempty(t)
    return;
end

fig = figure('Visible', 'off');
subplot(2,1,1);
plot(t, fv);
grid on;
xlabel('t (s, relativo)');
ylabel('force\_v');
title('Raw force\_v (segmento)');

subplot(2,1,2);
plot(t, ag);
grid on;
xlabel('t (s, relativo)');
ylabel('accel\_g');
title('Raw accel\_g (segmento)');

exportgraphics(fig, fullfile(outTsDir, 'raw_segment.png'), 'Resolution', 200);
savefig(fig, fullfile(outTsDir, 'raw_segment.fig'));
close(fig);

fig = figure('Visible', 'off');
subplot(1,2,1);
histogram(fv, 200);
grid on;
xlabel('force\_v');
title('Hist force\_v (segmento)');
subplot(1,2,2);
histogram(ag, 200);
grid on;
xlabel('accel\_g');
title('Hist accel\_g (segmento)');

exportgraphics(fig, fullfile(outTsDir, 'raw_hist.png'), 'Resolution', 200);
savefig(fig, fullfile(outTsDir, 'raw_hist.fig'));
close(fig);
end


function make_plots_raw_full(rawPlot, outTsDir)
mask = ~isnan(rawPlot.t_s);
t = rawPlot.t_s(mask);
fv = rawPlot.force_v(mask);
ag = rawPlot.accel_g(mask);

if isempty(t)
    return;
end

fig = figure('Visible', 'off');
subplot(2,1,1);
plot(t, fv);
grid on;
xlabel('t (s, relativo)');
ylabel('force\_v');
ttl = 'Raw force\_v (decimado)';
if isfield(rawPlot, 'stride')
    ttl = sprintf('Raw force\\_v (decimado, stride=%d)', rawPlot.stride);
end
title(ttl);

subplot(2,1,2);
plot(t, ag);
grid on;
xlabel('t (s, relativo)');
ylabel('accel\_g');
ttl = 'Raw accel\_g (decimado)';
if isfield(rawPlot, 'stride')
    ttl = sprintf('Raw accel\\_g (decimado, stride=%d)', rawPlot.stride);
end
title(ttl);

exportgraphics(fig, fullfile(outTsDir, 'raw_full.png'), 'Resolution', 200);
savefig(fig, fullfile(outTsDir, 'raw_full.fig'));
close(fig);
end


function make_plots_processing_20251210(fullSig, outTsDir)
% Procesamiento inspirado en trabajo 9-12 Dic 2025:
% - Envolvente (Hilbert) + lowpass (fc=15 Hz)
% - FFT + frecuencia dominante + THD
% - Estimación lineal de fuerza: F_est = K_ENV * envelope + B_ENV

K_ENV = 0.0669;
B_ENV = 0.8837;
fc_env = 15.0;

mask = ~isnan(fullSig.t_s) & ~isnan(fullSig.accel_g) & ~isnan(fullSig.force_v);
t = fullSig.t_s(mask);
a = fullSig.accel_g(mask);
f = fullSig.force_v(mask);

if isempty(t)
    return;
end

fs = NaN;
if isfield(fullSig, 'fs_hz_median')
    fs = fullSig.fs_hz_median;
end
if ~isfinite(fs) || fs <= 0
    fs = 2000;
end

% Envolvente
haveHilbert = exist('hilbert', 'file') == 2;
if haveHilbert
    env_raw = abs(hilbert(a));
else
    env_raw = abs(a);
end

env = env_raw;
try
    [b, a_f] = butter(2, fc_env / (fs / 2), 'low');
    env = filtfilt(b, a_f, env_raw);
catch
end

F_est = K_ENV * env + B_ENV;

[thd_f, f0_f, freqs_f, spec_f] = calc_thd_matlab(f, fs);
[thd_a, f0_a, freqs_a, spec_a] = calc_thd_matlab(a, fs);

% Decimar para graficar (por rendimiento)
maxPts = 250000;
stride = max(1, floor(numel(t) / maxPts));
tt = t(1:stride:end);
ff = f(1:stride:end);
aa = a(1:stride:end);
ee = env(1:stride:end);
Fe = F_est(1:stride:end);

fig = figure('Visible', 'off');
tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(tt, ff, 'k-');
grid on;
xlabel('t (s, relativo)');
ylabel('force\_v');
title('Fuerza (raw, decimado)');

nexttile;
plot(tt, aa, 'b-');
grid on;
xlabel('t (s, relativo)');
ylabel('accel\_g');
title('Aceleración (raw, decimado)');

nexttile;
plot(tt, ee, 'm-');
grid on;
xlabel('t (s, relativo)');
ylabel('env\_g');
title(sprintf('Envolvente (Hilbert + LP %.1f Hz)', fc_env));

nexttile;
plot(tt, Fe, 'r-');
grid on;
xlabel('t (s, relativo)');
ylabel('F\_est (V)');
title(sprintf('F\\_est = %.4f*env + %.4f', K_ENV, B_ENV));

nexttile;
semilogy(freqs_f, spec_f, 'k-');
grid on;
xlabel('f (Hz)');
ylabel('|FFT|');
title(sprintf('FFT fuerza: f0=%.1f Hz, THD=%.1f%%', f0_f, thd_f));
xlim([0 min(500, fs/2)]);

nexttile;
semilogy(freqs_a, spec_a, 'b-');
grid on;
xlabel('f (Hz)');
ylabel('|FFT|');
title(sprintf('FFT acel: f0=%.1f Hz, THD=%.1f%%', f0_a, thd_a));
xlim([0 min(500, fs/2)]);

exportgraphics(fig, fullfile(outTsDir, 'raw_processing_20251210.png'), 'Resolution', 200);
savefig(fig, fullfile(outTsDir, 'raw_processing_20251210.fig'));
close(fig);

make_plots_hysteresis_20251205(fullSig, outTsDir);
end


function [thdPct, f0, freqs, mag] = calc_thd_matlab(x, fs)
x = double(x(:));
if isempty(x)
    thdPct = 0; f0 = 0; freqs = 0; mag = 0;
    return;
end

n = numel(x);
n_fft = min(n, 4096);
xx = x(end-n_fft+1:end);
xx = xx - mean(xx, 'omitnan');

try
    w = hann(n_fft);
catch
    w = hanning(n_fft);
end
xx = xx .* w;

X = abs(rfft(xx)) * 2.0 / n_fft;
freqs = (0:numel(X)-1)' * (fs / n_fft);

% Quitar DC para buscar fundamental
if numel(X) > 1
    Xn = X(2:end);
    fn = freqs(2:end);
else
    Xn = X;
    fn = freqs;
end

if isempty(Xn)
    thdPct = 0; f0 = 0; mag = X;
    return;
end

[A0, pk] = max(Xn);
f0 = fn(pk);

if A0 < 1e-10
    thdPct = 0;
else
    harm_sum_sq = 0;
    for h = 2:6
        f_h = f0 * h;
        if f_h >= fs/2
            break;
        end
        [~, idx] = min(abs(fn - f_h));
        harm_sum_sq = harm_sum_sq + (Xn(idx) ^ 2);
    end
    thdPct = sqrt(harm_sum_sq) / A0 * 100;
end

mag = X;
end


function make_plots_hysteresis_20251205(fullSig, outTsDir)
mask = ~isnan(fullSig.accel_g) & ~isnan(fullSig.force_v);
a_g = double(fullSig.accel_g(mask));
f_v = double(fullSig.force_v(mask));

if numel(a_g) < 256
    return;
end

fs = NaN;
if isfield(fullSig, 'fs_hz_median')
    fs = fullSig.fs_hz_median;
end
if ~isfinite(fs) || fs <= 0
    fs = 2000;
end

t = (0:numel(a_g)-1)' / fs;
a_ms2 = a_g * 9.81;

dt = 1 / fs;

try
    fc_hp = 0.5;
    [b, a_f] = butter(2, fc_hp / (fs / 2), 'high');
    a_filt = filtfilt(b, a_f, a_ms2 - mean(a_ms2));
catch
    a_filt = a_ms2 - mean(a_ms2);
end

v = cumtrapz(a_filt) * dt;
try
    v = detrend(v);
catch
end

x = cumtrapz(v) * dt;
try
    x = detrend(x);
catch
end

f0 = f_v - mean(f_v);

area = 0;
try
    area = 0.5 * abs(sum(x(1:end-1) .* f0(2:end) - x(2:end) .* f0(1:end-1)));
catch
end

disp_range = max(x) - min(x);
force_range = max(f0) - min(f0);

alpha_est = 1;
k_app = 0;
if disp_range > 1e-12
    k_app = force_range / disp_range;
    energia_elastica = 0.5 * k_app * (disp_range/2)^2;
    if energia_elastica > 0
        ratio_energia = area / energia_elastica;
        alpha_est = max(0, min(1, 1 - ratio_energia/4));
    end
end

[thd_f, f0_f, freqs_f, spec_f] = calc_thd_matlab(f0, fs);
[thd_a, f0_a, freqs_a, spec_a] = calc_thd_matlab(a_filt, fs);

n_ciclos = 5000;
f_use = max([f0_f, f0_a, 0]);
if isfinite(f_use) && f_use > 0.1
    n_ciclos = round(5 * fs / f_use);
end
n_ciclos = min(n_ciclos, numel(x));

segN = min(5000, numel(t));

fig = figure('Visible', 'off');
tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(t(1:segN), f0(1:segN), 'r-', 'LineWidth', 0.5);
grid on;
xlabel('t (s)');
ylabel('force (V, demean)');
title('Fuerza vs tiempo');

nexttile;
semilogy(freqs_f, spec_f, 'r-');
grid on;
xlabel('f (Hz)');
ylabel('|FFT|');
title(sprintf('FFT fuerza (THD=%.1f%%)', thd_f));
xlim([0 min(500, fs/2)]);

nexttile;
semilogy(freqs_a, spec_a, 'b-');
grid on;
xlabel('f (Hz)');
ylabel('|FFT|');
title(sprintf('FFT acel (THD=%.1f%%)', thd_a));
xlim([0 min(500, fs/2)]);

nexttile;
plot(x(1:n_ciclos) * 1000, f0(1:n_ciclos), 'k-', 'LineWidth', 0.5);
grid on;
xlabel('x (mm, rel)');
ylabel('force (V, demean)');
title(sprintf('Histéresis F-x (alpha≈%.2f)', alpha_est));

nexttile;
plot(a_filt(1:n_ciclos), f0(1:n_ciclos), 'g-', 'LineWidth', 0.5);
grid on;
xlabel('a (m/s^2)');
ylabel('force (V, demean)');
title('Lissajous F-a');

nexttile;
plot(t(1:segN), a_filt(1:segN), 'b-', 'LineWidth', 0.5);
grid on;
xlabel('t (s)');
ylabel('a (m/s^2, hp)');
title(sprintf('Aceleración HP (area=%.3g)', area));

exportgraphics(fig, fullfile(outTsDir, 'raw_hysteresis_20251205.png'), 'Resolution', 200);
savefig(fig, fullfile(outTsDir, 'raw_hysteresis_20251205.fig'));
close(fig);
end


function X = rfft(x)
% rfft mínimo (para compatibilidad sin toolboxes)
n = numel(x);
Xc = fft(x);
X = Xc(1:floor(n/2)+1);
X = abs(X);
end


function make_plots_raw_rms(rms, seqMeta, outTsDir)
tBlock = seqMeta.tBlock;
mask = ~isnan(tBlock);
if ~any(mask)
    return;
end
t = tBlock(mask);
t = t - t(1);

fv = rms.force_v(mask);
ag = rms.accel_g(mask);

fig = figure('Visible', 'off');
subplot(2,1,1);
plot(t, fv);
grid on;
xlabel('t (s, relativo)');
ylabel('RMS force\_v');
title('RMS por bloque (force\_v)');

subplot(2,1,2);
plot(t, ag);
grid on;
xlabel('t (s, relativo)');
ylabel('RMS accel\_g');
title('RMS por bloque (accel\_g)');

exportgraphics(fig, fullfile(outTsDir, 'raw_rms_per_block.png'), 'Resolution', 200);
savefig(fig, fullfile(outTsDir, 'raw_rms_per_block.fig'));
close(fig);
end


function S = merge_struct(A, B)
S = A;
f = fieldnames(B);
for i = 1:numel(f)
    S.(f{i}) = B.(f{i});
end
end


function T = summary_to_table(summary)
rows = numel(summary.items);
if rows == 0
    T = table();
    return;
end

T = table('Size', [rows 12], ...
    'VariableTypes', {'string','double','double','double','double','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'timestamp','raw_bytes','inference_bytes','cases_bytes', ...
    'raw_rows','inference_rows','cases_lines','cases_bad_lines', ...
    'seq_min','seq_max','duration_s_est','raw_blocks_bad'});

for i = 1:rows
    it = summary.items(i);
    T.timestamp(i) = string(it.timestamp);
    T.raw_bytes(i) = it.inventory.raw_bytes;
    T.inference_bytes(i) = it.inventory.inference_bytes;
    T.cases_bytes(i) = it.inventory.cases_bytes;

    if isfield(it.inventory, 'raw_rows')
        T.raw_rows(i) = it.inventory.raw_rows;
    else
        T.raw_rows(i) = NaN;
    end
    if isfield(it.inventory, 'inference_rows')
        T.inference_rows(i) = it.inventory.inference_rows;
    else
        T.inference_rows(i) = NaN;
    end
    if isfield(it.inventory, 'cases_lines')
        T.cases_lines(i) = it.inventory.cases_lines;
    else
        T.cases_lines(i) = NaN;
    end
    if isfield(it.inventory, 'cases_bad_lines')
        T.cases_bad_lines(i) = it.inventory.cases_bad_lines;
    else
        T.cases_bad_lines(i) = NaN;
    end

    T.seq_min(i) = get_field(it.quality, 'seq_min');
    T.seq_max(i) = get_field(it.quality, 'seq_max');
    T.duration_s_est(i) = get_field(it.quality, 'duration_s_est');
    T.raw_blocks_bad(i) = get_field(it.quality, 'raw_blocks_bad');
end
end


function v = get_field(S, name)
if isfield(S, name)
    v = S.(name);
else
    v = NaN;
end
end

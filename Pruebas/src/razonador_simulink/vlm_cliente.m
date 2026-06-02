function veredicto = vlm_cliente(system_prompt, user_prompt, img_b64, modelo)
% VLM_CLIENTE  Cliente VLM dual (Ollama local <-> OpenRouter nube) con fallback.
%
% Nucleo de la capa deliberativa (System 2). Envia al VLM un prompt multimodal
% (texto + imagen del lazo F-x en base64) y devuelve un veredicto estructurado.
% Replica el enrutamiento y la config de .env de razonador_local.py, pero en
% MATLAB con webwrite (SIN Python).
%
% Marcar con coder.extrinsic en los bloques codegen:
%   coder.extrinsic('vlm_cliente');
%
% ENTRADAS
%   system_prompt  (char) system prompt (cargar de prompts/orquestador_vlm.md)
%   user_prompt    (char) contexto: 7 features + top-K casos + pregunta
%   img_b64        (char) imagen del lazo F-x en base64 SIN cabecera data:.
%                  Puede ir vacia ('') para llamada solo-texto.
%   modelo         (char) preferencia de proveedor/modelo:
%                    'cloud'/'openrouter'/con '/'  -> intenta OpenRouter primero
%                    otro (qwen3:8b, llava, ...)   -> intenta Ollama primero
%                  Si el preferido falla, hace FALLBACK al otro proveedor.
%
% SALIDA (struct veredicto)
%   .clase         1=lineal, 2=moderada, 3=marcada, 0=desconocido
%   .caso_sel      indice del caso top-K seleccionado por el VLM (0 si ninguno)
%   .accion        (char) accion recomendada
%   .retain        (logical) si el caso merece anadirse a la base (Retain)
%   .confianza     (double) confianza declarada por el VLM en [0,1]
%   .justificacion (char) explicacion del VLM
%   .ok            (logical) true si alguna llamada tuvo exito y se parseo JSON
%   .fuente_modelo (char) proveedor:modelo que respondio (o 'none')
%   .raw           (char) respuesta cruda del VLM (trazabilidad/registro)
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    if nargin < 3 || isempty(img_b64); img_b64 = ''; end
    if nargin < 4 || isempty(modelo);  modelo = 'cloud'; end

    veredicto = veredicto_vacio();

    % ── Config desde .env (combinando todos los .env del arbol) ─────────────
    env = leer_env();
    or_key   = get_env(env, 'OPENROUTER_API_KEY', '');
    or_base  = get_env(env, 'OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1');
    or_model = get_env(env, 'OPENROUTER_MODEL', 'qwen/qwen2.5-vl-72b-instruct');
    ollama_url = get_env(env, 'OLLAMA_URL', 'http://localhost:11434');

    % ── Orden de proveedores (preferido + fallback) ─────────────────────────
    if prefiere_openrouter(modelo)
        orden = {'openrouter', 'ollama'};
    else
        orden = {'ollama', 'openrouter'};
    end

    temperature = 0.2;   % baja -> reproducibilidad doctoral

    for p = 1:numel(orden)
        prov = orden{p};
        try
            if strcmp(prov, 'openrouter')
                if isempty(or_key)
                    continue;   % sin API key no se intenta la nube
                end
                if prefiere_openrouter(modelo) && ~contains(modelo, '/')
                    use_model = or_model;   % 'cloud' -> modelo del .env
                else
                    use_model = modelo;
                    if ~contains(modelo, '/'); use_model = or_model; end
                end
                raw = llamar_openrouter(or_base, or_key, use_model, ...
                                        system_prompt, user_prompt, img_b64, temperature);
                fuente = ['openrouter:' use_model];
            else
                % Ollama: si el modelo es 'cloud'/openrouter, usar un VLM local por defecto
                if prefiere_openrouter(modelo)
                    use_model = 'llava';   % VLM local por defecto
                else
                    use_model = modelo;
                end
                raw = llamar_ollama(ollama_url, use_model, ...
                                    system_prompt, user_prompt, img_b64, temperature);
                fuente = ['ollama:' use_model];
            end

            v = extraer_json_veredicto(raw);
            if v.ok
                v.fuente_modelo = fuente;
                v.raw = raw;
                veredicto = v;
                return;   % exito: no se intenta el fallback
            end
        catch ME
            % Falla de red / proveedor: se intenta el siguiente (fallback)
            veredicto.justificacion = sprintf('Fallo %s: %s', prov, ME.message);
        end
    end
    % Si llega aqui, ningun proveedor respondio con JSON valido -> ok=false
end


% ===========================================================================
% Llamada a OpenRouter (API compatible OpenAI, multimodal)
% ===========================================================================
function raw = llamar_openrouter(base_url, api_key, model, system_prompt, user_prompt, img_b64, temperature)
    url = [regexprep(base_url, '/+$', '') '/chat/completions'];

    % Contenido del usuario: texto (+ imagen si la hay)
    if isempty(img_b64)
        user_content = user_prompt;   % string simple
    else
        data_uri = ['data:image/png;base64,' img_b64];
        user_content = { ...
            struct('type','text','text',user_prompt), ...
            struct('type','image_url','image_url',struct('url',data_uri)) };
    end

    messages = { ...
        struct('role','system','content',system_prompt), ...
        struct('role','user','content',user_content) };

    payload = struct();
    payload.model = model;
    payload.messages = messages;
    payload.temperature = temperature;

    opts = weboptions('MediaType','application/json', ...
        'RequestMethod','post', 'Timeout', 120, ...
        'HeaderFields', { ...
            'Authorization', ['Bearer ' api_key]; ...
            'HTTP-Referer', 'https://github.com/JRavenelco/cDAQ_9174'; ...
            'X-Title', 'Razonador CBR Simulink - Histeresis' });

    resp = webwrite(url, payload, opts);

    raw = '';
    if isfield(resp, 'choices') && ~isempty(resp.choices)
        ch = resp.choices;
        if iscell(ch); ch1 = ch{1}; else; ch1 = ch(1); end
        if isfield(ch1, 'message') && isfield(ch1.message, 'content')
            raw = ch1.message.content;
        end
    end
end


% ===========================================================================
% Llamada a Ollama (local, /api/generate con images base64)
% ===========================================================================
function raw = llamar_ollama(ollama_url, model, system_prompt, user_prompt, img_b64, temperature)
    url = [regexprep(ollama_url, '/+$', '') '/api/generate'];

    payload = struct();
    payload.model  = model;
    payload.prompt = user_prompt;
    payload.system = system_prompt;
    payload.stream = false;
    payload.options = struct('temperature', temperature, 'num_predict', 2048);
    if ~isempty(img_b64)
        payload.images = {img_b64};   % array JSON de un elemento
    end

    opts = weboptions('MediaType','application/json', ...
        'RequestMethod','post', 'Timeout', 300);

    resp = webwrite(url, payload, opts);

    raw = '';
    if isfield(resp, 'response'); raw = resp.response; end
end


% ===========================================================================
% Extraccion robusta del JSON del veredicto desde el texto del VLM
% ===========================================================================
function v = extraer_json_veredicto(raw)
    v = veredicto_vacio();
    if isempty(raw) || ~ischar(raw) && ~isstring(raw)
        return;
    end
    txt = char(raw);

    % Quitar fences ```json ... ``` si los hubiera
    txt = regexprep(txt, '```json', '');
    txt = regexprep(txt, '```', '');

    % Tomar desde el primer { hasta el ultimo }
    i1 = find(txt == '{', 1, 'first');
    i2 = find(txt == '}', 1, 'last');
    if isempty(i1) || isempty(i2) || i2 <= i1
        return;
    end
    jtxt = txt(i1:i2);

    try
        s = jsondecode(jtxt);
    catch
        return;
    end

    % Mapear campos con tolerancia a faltantes
    v.clase         = campo_num(s, 'clase', 0);
    v.caso_sel      = campo_num(s, 'caso_sel', 0);
    v.confianza     = campo_num(s, 'confianza', 0);
    v.accion        = campo_str(s, 'accion', '');
    v.justificacion = campo_str(s, 'justificacion', '');
    v.retain        = campo_num(s, 'retain', 0) >= 1 || ...
                      strcmpi(campo_str(s, 'retain', 'false'), 'true');
    v.ok = true;
end


% ===========================================================================
% Utilidades
% ===========================================================================
function v = veredicto_vacio()
    v = struct('clase', 0, 'caso_sel', 0, 'accion', '', 'retain', false, ...
               'confianza', 0, 'justificacion', '', 'ok', false, ...
               'fuente_modelo', 'none', 'raw', '');
end

function tf = prefiere_openrouter(modelo)
    m = lower(strtrim(modelo));
    tf = strcmp(m,'cloud') || strcmp(m,'openrouter') || contains(m, '/');
end

function val = campo_num(s, name, def)
    val = def;
    if isfield(s, name)
        x = s.(name);
        if isnumeric(x) && ~isempty(x); val = double(x(1));
        elseif (ischar(x) || isstring(x)); d = str2double(x); if ~isnan(d); val = d; end
        end
    end
end

function val = campo_str(s, name, def)
    val = def;
    if isfield(s, name)
        x = s.(name);
        if ischar(x); val = x;
        elseif isstring(x); val = char(x);
        elseif isnumeric(x); val = num2str(x);
        elseif islogical(x); if x; val = 'true'; else; val = 'false'; end
        end
    end
end

function val = get_env(env, key, def)
    val = def;
    if isfield(env, key) && ~isempty(env.(key)); val = env.(key); end
end


% ===========================================================================
% Lector de .env (busca de la carpeta del script hacia la raiz, combinando)
% ===========================================================================
function env = leer_env()
    env = struct();
    here = fileparts(mfilename('fullpath'));
    % Construir lista de carpetas: here y todos sus padres
    dirs = {};
    d = here;
    while true
        dirs{end+1} = d; %#ok<AGROW>
        parent = fileparts(d);
        if isempty(parent) || strcmp(parent, d); break; end
        d = parent;
    end
    % Procesar de la raiz hacia abajo: los .env mas cercanos pisan a los lejanos
    for k = numel(dirs):-1:1
        f = fullfile(dirs{k}, '.env');
        if isfile(f)
            env = merge_env(env, parse_env_file(f));
        end
    end
    % Variables de entorno del sistema tienen prioridad final
    for key = {'OPENROUTER_API_KEY','OPENROUTER_BASE_URL','OPENROUTER_MODEL','OLLAMA_URL'}
        v = getenv(key{1});
        if ~isempty(v); env.(key{1}) = v; end
    end
end

function env = parse_env_file(path)
    env = struct();
    try
        lines = readlines(path);
    catch
        return;
    end
    for i = 1:numel(lines)
        ln = strtrim(char(lines(i)));
        if isempty(ln) || ln(1) == '#'; continue; end
        pos = find(ln == '=', 1, 'first');
        if isempty(pos); continue; end
        key = strtrim(ln(1:pos-1));
        val = strtrim(ln(pos+1:end));
        % quitar comillas envolventes
        if numel(val) >= 2 && ((val(1)=='"' && val(end)=='"') || (val(1)=='''' && val(end)==''''))
            val = val(2:end-1);
        end
        if ~isempty(key) && isvarname_relaxed(key)
            env.(matlab.lang.makeValidName(key)) = val;
        end
    end
end

function env = merge_env(env, extra)
    f = fieldnames(extra);
    for i = 1:numel(f)
        env.(f{i}) = extra.(f{i});
    end
end

function tf = isvarname_relaxed(key)
    tf = ~isempty(regexp(key, '^[A-Za-z_]\w*$', 'once'));
end

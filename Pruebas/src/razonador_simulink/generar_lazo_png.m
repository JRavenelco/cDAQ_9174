function img_b64 = generar_lazo_png(fuerza, entrada, titulo)
% GENERAR_LAZO_PNG  Renderiza el lazo Fuerza-desplazamiento (F-x) normalizado
% a un PNG temporal y lo devuelve en base64 (sin cabecera data:).
%
% El VLM (vlm_cliente.m) recibe esta imagen para "ver" la forma del lazo y
% auditar/decidir la clase de histeresis. Un lazo ancho/abierto => histeresis
% marcada; un lazo delgado/lineal => comportamiento lineal.
%
% Se ejecuta en Interpreted execution. Marcar coder.extrinsic si se llama desde
% un bloque codegen:  coder.extrinsic('generar_lazo_png');
%
% ENTRADAS
%   fuerza, entrada  vectores de la senal (cruda o normalizada) de igual longitud.
%                    Se normalizan internamente con (x-mean)/max(|x-mean|), igual
%                    que construir_casos.py, para que el eje sea comparable.
%   titulo           (char) titulo opcional de la figura.
%
% SALIDA
%   img_b64  (char) imagen PNG codificada en base64 (matlab.net.base64encode).
%            Cadena vacia '' si no se pudo generar.
%
% Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    if nargin < 3 || isempty(titulo); titulo = 'Lazo F-x normalizado'; end
    img_b64 = '';

    fuerza  = double(fuerza(:));
    entrada = double(entrada(:));
    n = min(numel(fuerza), numel(entrada));
    if n < 3; return; end
    fuerza = fuerza(1:n); entrada = entrada(1:n);

    fn = norm_sig(fuerza);
    xn = norm_sig(entrada);

    f = figure('Visible','off', 'Color','w', 'Position',[100 100 560 520]);
    try
        ax = axes('Parent', f);
        plot(ax, xn, fn, '-', 'LineWidth', 1.2, 'Color', [0.15 0.35 0.8]);
        hold(ax, 'on');
        plot(ax, xn(1), fn(1), 'go', 'MarkerFaceColor','g');   % inicio
        plot(ax, xn(end), fn(end), 'rs', 'MarkerFaceColor','r'); % fin
        grid(ax, 'on'); axis(ax, 'equal');
        xlabel(ax, 'Entrada normalizada (desplazamiento ~ x)');
        ylabel(ax, 'Fuerza normalizada (F)');
        title(ax, titulo);
        xlim(ax, [-1.1 1.1]); ylim(ax, [-1.1 1.1]);

        tmp = [tempname '.png'];
        exportgraphics(ax, tmp, 'Resolution', 110);

        bytes = leer_bytes(tmp);
        if ~isempty(bytes)
            img_b64 = char(matlab.net.base64encode(bytes));
        end
        if isfile(tmp); delete(tmp); end
    catch ME
        warning('generar_lazo_png: %s', ME.message);
    end
    close(f);
end


function y = norm_sig(x)
    centered = x - mean(x, 'omitnan');
    scale = max(abs(centered), [], 'omitnan');
    if ~isfinite(scale) || scale <= 1e-12
        y = zeros(size(centered));
    else
        y = centered / scale;
    end
end

function bytes = leer_bytes(path)
    bytes = uint8([]);
    fid = fopen(path, 'r');
    if fid < 0; return; end
    bytes = fread(fid, Inf, '*uint8');
    fclose(fid);
end

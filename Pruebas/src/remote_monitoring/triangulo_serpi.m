function triangulo_serpinsky()
% Genera un triángulo tipo Sierpinski mediante autómata binario en una matriz 100x100
E = zeros(100,100);
E(1,50) = 1;            % condición inicial
E(1,25) = 1;
E(1,75) = 1;
EA = E;                 % estado actual
for i = 2:100           % filas: actualizar de arriba hacia abajo
    EN = zeros(1,100);  % nueva fila temporal (solo se necesita una fila)
    for j = 2:99        % evitar bordes para acceder a j-1 y j+1
        % Regla típica para triángulo de Sierpinski: XOR de los vecinos superiores
        EN(j) = mean(EA(i-1,j-1) + EA(i-1,j+1), 2);
    end
    EA(i,:) = EN;       % asignar la fila calculada al estado actual
end

% Mostrar la evolución fila por fila con el "espacio" (lienzo) ya dibujado
figure;
h = imagesc(EA);                 % dibuja todo el lienzo desde el inicio
colormap nebula;
axis equal tight;
title('Triángulo de Sierpinski en construcción');
for k = 1:100
    % crear máscara que muestre solo las primeras k filas y deje el resto en blanco
    mask = false(size(EA));
    mask(1:k,:) = true;
    % actualizar datos: mantener valores originales donde mask=true, resto en 0
    D = zeros(size(EA));
    D(mask) = EA(mask);
    set(h, 'CData', D);
    drawnow;
    pause(0.01); % ajustar pausa a gusto
end

% Visualización alternativa usando surf y mesh del triángulo generado
figure;
[X,Y] = meshgrid(1:100,1:100);
Z = EA;                 % usar la matriz final EA como altura
% suavizar ligeramente para mejor apariencia en surf/mesh (opcional)
if exist('imgaussfilt','file')
    Zs = imgaussfilt(Z, 0.5);
else
    Zs = Z; % si no está Image Processing Toolbox, usar Z sin suavizar
end
subplot(1,2,1);
surf(X, Y, Zs, 'EdgeColor', 'none');
colormap(gray);
view(45,60);
axis tight; xlabel('x'); ylabel('y'); zlabel('z');
title('Surf del triángulo de Sierpinski');
camlight headlight; lighting gouraud;

subplot(1,2,2);
mesh(X, Y, Z);
colormap(gray);
view(45,60);
axis tight; xlabel('x'); ylabel('y'); zlabel('z');
title('Mesh del triángulo de Sierpinski');

% Además: reproducir la evolución fila por fila en una nueva figura,
% mostrando surf y mesh actualizados a medida que se añaden filas.
figure('Name','Evolución fila por fila','NumberTitle','off');
for k = 1:100
    Zk = zeros(size(Z));   % solo las primeras k filas tienen valores
    Zk(1:k,:) = Z(1:k,:);
    Zks = Zk;
    if exist('imgaussfilt','file')
        Zks = imgaussfilt(Zk,0.5);
    end

    subplot(1,2,1);
    surf(X, Y, Zks, 'EdgeColor','none');
    colormap winter
    %colormap(gray);
    view(45,60);
    axis tight; xlabel('x'); ylabel('y'); zlabel('z');
    title(sprintf('Surf - filas 1:%d', k));
    camlight headlight; lighting gouraud;

    subplot(1,2,2);
    mesh(X, Y, Zk);
    colormap winter
    %colormap(gray);
    view(45,60);
    axis tight; xlabel('x'); ylabel('y'); zlabel('z');
    title(sprintf('Mesh - filas 1:%d', k));

    drawnow;
    pause(0.005);
end
end
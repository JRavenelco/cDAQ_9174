classdef CBRHysteresisSystem < matlab.System & matlab.system.mixin.Propagates
    % CBRHysteresisSystem - Parte 3: bloque MATLAB System (Razonador CBR).
    %
    % Razonador Basado en Casos para diagnostico de histeresis en fresado.
    % Replica la logica de razonador_local.py (normalizacion robusta +
    % distancia euclidiana ponderada) y la encapsula como System Object,
    % ideal cuando se quiere un umbral AJUSTABLE EN CALIENTE desde la UI de
    % Simulink (slider / Dashboard) sin recompilar.
    %
    % ENTRADA
    %   features [1 x 7]  en el orden:
    %     [force_rms, force_peak_abs, input_rms, input_peak_abs,
    %      corr_force_input, loop_area_norm, duration_s]
    %
    % SALIDAS
    %   class_id    1=lineal, 2=moderada, 3=marcada; 0=rechazado
    %   min_dist    distancia ponderada al caso mas cercano
    %   best_score  similitud 1/(1+min_dist)
    %   best_idx    indice del caso recuperado (0 si rechazado)
    %
    % Uso en Simulink:
    %   1) Ejecuta  >> exportar_base_casos   (genera datos/base_de_casos_cbr.mat)
    %   2) Arrastra un bloque "MATLAB System" y escribe: CBRHysteresisSystem
    %   3) Ajusta Threshold desde el dialogo del bloque (tunable).
    %
    % Autor: Jesus Santana-Ramirez (tesis doctoral, histeresis en fresado)

    properties(Nontunable)
        % Archivo .mat con la base de casos (relativo al .m si no es absoluto)
        ModelFile (1,:) char = 'datos/base_de_casos_cbr.mat'
    end

    properties
        % Umbral de distancia maxima admitida (TUNABLE). >=1e6 desactiva rechazo.
        Threshold (1,1) double = 1.0
    end

    properties(Access = private)
        case_base    double = []
        feat_center  double = zeros(1,7)
        feat_scale   double = ones(1,7)
        feat_weights double = ones(1,7)
        hist_label   double = []
        model_loaded logical = false
    end

    methods(Access = protected)
        function setupImpl(obj)
            % Carga de la base de casos al iniciar la simulacion
            model_path = obj.ModelFile;
            if ~isfile(model_path)
                model_path = fullfile(fileparts(mfilename('fullpath')), obj.ModelFile);
            end

            if isfile(model_path)
                S = load(model_path);
                obj.case_base    = S.case_base;
                obj.feat_center  = S.feat_center;
                obj.feat_scale   = S.feat_scale;
                obj.feat_weights = S.feat_weights;
                obj.hist_label   = S.hist_label;
                obj.model_loaded = true;
            else
                obj.model_loaded = false;
                obj.case_base    = [];
            end
        end

        function [class_id, min_dist, best_score, best_idx] = stepImpl(obj, features)
            % Forzar fila [1 x D]: Simulink pasa la senal como columna [D x 1]
            % y el broadcasting con feat_center [1 x D] daria una matriz [D x D].
            features = reshape(features, 1, []);

            class_id   = 0;
            min_dist   = -1;
            best_score = 0;
            best_idx   = 0;

            if ~obj.model_loaded || isempty(obj.case_base)
                return;
            end

            num_cases = size(obj.case_base, 1);
            D = size(obj.case_base, 2);

            % Normalizar query (robust scaling)
            q_norm = (features - obj.feat_center) ./ obj.feat_scale;

            % Retrieve: caso mas cercano por distancia ponderada
            min_dist = inf;
            best_idx = 0;
            for i = 1:num_cases
                c_norm = (obj.case_base(i, :) - obj.feat_center) ./ obj.feat_scale;
                diff_vec = (q_norm - c_norm) .* obj.feat_weights;
                dist = norm(diff_vec) / sqrt(D);
                if dist < min_dist
                    min_dist = dist;
                    best_idx = i;
                end
            end

            best_score = 1 / (1 + min_dist);

            if best_idx >= 1
                class_id = obj.hist_label(best_idx);
            end

            % Umbral de rechazo
            if min_dist >= obj.Threshold
                class_id = 0;
                best_idx = 0;
            end
        end

        % ── Definicion de puertos para Simulink ─────────────────────────────
        function num = getNumInputsImpl(~)
            num = 1;
        end

        function num = getNumOutputsImpl(~)
            num = 4;
        end

        function n1 = getInputNamesImpl(~)
            n1 = 'features';
        end

        function [n1,n2,n3,n4] = getOutputNamesImpl(~)
            n1 = 'class_id';
            n2 = 'min_dist';
            n3 = 'best_score';
            n4 = 'best_idx';
        end

        function [sz1,sz2,sz3,sz4] = getOutputSizeImpl(~)
            sz1 = [1 1]; sz2 = [1 1]; sz3 = [1 1]; sz4 = [1 1];
        end

        function [dt1,dt2,dt3,dt4] = getOutputDataTypeImpl(~)
            dt1 = 'double'; dt2 = 'double'; dt3 = 'double'; dt4 = 'double';
        end

        function [c1,c2,c3,c4] = isOutputComplexImpl(~)
            c1 = false; c2 = false; c3 = false; c4 = false;
        end

        function [f1,f2,f3,f4] = isOutputFixedSizeImpl(~)
            f1 = true; f2 = true; f3 = true; f4 = true;
        end

        function icon = getIconImpl(~)
            icon = sprintf('CBR\nHysteresis\nReasoner');
        end
    end
end

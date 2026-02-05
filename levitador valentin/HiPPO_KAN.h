#ifndef HIPPO_KAN_H
#define HIPPO_KAN_H

#include <math.h>
#include <string.h>
#include "model_weights.h"

// Helpers Matemáticos
inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

class HiPPO_KAN {
private:
    // Estados de Memoria HiPPO (Vectores c)
    float c_u[N_HIPPO];
    float c_i[N_HIPPO];

    // Actualización recurrente HiPPO: c[k] = A*c[k-1] + B*u[k]
    void update_hippo_state(float* state, float input, const float* A, const float* B) {
        float next_state[N_HIPPO];

        for(int r=0; r<N_HIPPO; r++) {
            float sum = 0.0f;
            int row_offset = r * N_HIPPO;
            for(int c=0; c<N_HIPPO; c++) {
                sum += state[c] * A[row_offset + c];
            }
            sum += input * B[r];
            next_state[r] = sum;
        }

        for(int i=0; i<N_HIPPO; i++) state[i] = next_state[i];
    }

    // Evaluación recursiva de B-Spline (Cox-de Boor)
    float de_boor(int i, int k, float x, const float* t) {
        if (k == 0) {
            return (x >= t[i] && x < t[i+1]) ? 1.0f : 0.0f;
        }
        
        float val = 0.0f;
        float denom1 = t[i+k] - t[i];
        if (denom1 > 1e-6f) {
            val += (x - t[i]) / denom1 * de_boor(i, k-1, x, t);
        }
        
        float denom2 = t[i+k+1] - t[i+1];
        if (denom2 > 1e-6f) {
            val += (t[i+k+1] - x) / denom2 * de_boor(i+1, k-1, x, t);
        }
        
        return val;
    }

    void kan_layer_forward(const float* input, float* output, int in_dim, int out_dim,
                           const float* w_base, const float* w_spline, const float* grid_pts, int num_coeffs) {
        
        // Asumiendo k=3 constante
        const int k = 3; 

        for(int o=0; o<out_dim; o++) {
            float sum_node = 0.0f;
            for(int i=0; i<in_dim; i++) {
                float x = input[i];

                // 1. Base SiLU
                float silu_val = silu(x);
                sum_node += w_base[o * in_dim + i] * silu_val;

                // 2. Spline
                float spline_contrib = 0.0f;
                // Puntero a coeficientes para esta conexión. Stride es num_coeffs.
                int coeff_idx = (o * in_dim + i) * num_coeffs;
                const float* coeffs = &w_spline[coeff_idx];
                
                // Iterar sobre los coeficientes (bases)
                for(int j=0; j < num_coeffs; j++) {
                     float b_val = de_boor(j, k, x, grid_pts);
                     if (b_val > 0.0f) {
                        spline_contrib += coeffs[j] * b_val;
                     }
                }
                sum_node += spline_contrib;
            }
            output[o] = sum_node;
        }
    }

public:
    HiPPO_KAN() {
        reset_memory();
    }

    void reset_memory() {
        for(int i=0; i<N_HIPPO; i++) {
            c_u[i] = 0.0f;
            c_i[i] = 0.0f;
        }
    }

    float predict(float u_raw, float i_raw) {
        // 1. Normalizar
        float u_norm = (u_raw - U_MEAN) / (U_STD + 1e-6f);
        float i_norm = (i_raw - I_MEAN) / (I_STD + 1e-6f);
        
        // Clamp inputs to grid range [-2, 2]
        if (u_norm < -2.0f) u_norm = -2.0f;
        if (u_norm > 2.0f) u_norm = 2.0f;
        if (i_norm < -2.0f) i_norm = -2.0f;
        if (i_norm > 2.0f) i_norm = 2.0f;

        // 2. Actualizar HiPPO
        update_hippo_state(c_u, u_norm, HIPPO_AD, HIPPO_BD);
        update_hippo_state(c_i, i_norm, HIPPO_AD, HIPPO_BD);

        // 3. Preparar input para KAN -> [c_u, c_i]
        float kan_in[2 * N_HIPPO];
        for(int i=0; i<N_HIPPO; i++) {
            kan_in[i] = c_u[i];
            kan_in[N_HIPPO + i] = c_i[i];
        }

        // 4. KAN Layer 1
        float l1_out[16]; 
        // Pasamos L1_COEFFS explícitamente
        kan_layer_forward(kan_in, l1_out, L1_IN, L1_OUT, L1_BASE, L1_SPLINE, L1_GRID, L1_COEFFS);

        // 5. KAN Layer 2
        float l2_out[1];
        // Asumimos misma arquitectura para L2 (mismo grid y coeffs size)
        kan_layer_forward(l1_out, l2_out, L2_IN, L2_OUT, L2_BASE, L2_SPLINE, L1_GRID, L1_COEFFS);

        // 6. Desnormalizar salida
        float y_pred = l2_out[0] * Y_STD + Y_MEAN;

        return y_pred;
    }
};

#endif

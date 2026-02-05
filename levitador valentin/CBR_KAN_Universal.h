/*
 * CBR_KAN_Universal.h
 * Mapa de Control Universal: CBR + KAN para Levitador Sensorless
 * 
 * Arquitectura:
 * - CBR: Memoria de "Casos de Oro" discretos [u, i, phi, y, di/dt]
 * - KAN: Interpolador continuo entre casos (splines adaptativos)
 * - PINN: Restricción física u = R*i + dphi/dt
 * - HiPPO: Memoria temporal para capturar dinámica
 * 
 * Autor: Sistema CBR+KAN para Levitación Magnética
 * Fecha: Diciembre 2024
 */

#ifndef CBR_KAN_UNIVERSAL_H
#define CBR_KAN_UNIVERSAL_H

#include <math.h>

// ============================================================================
// CONSTANTES FÍSICAS - Usar las de CBR_InitPosition.h si ya están definidas
// ============================================================================
#ifndef MASS_KG
#define MASS_KG         0.009f
#endif
#ifndef GRAVITY
#define GRAVITY         9.81f
#endif
#define V_BUS           9.86f
#define R_NOMINAL       2.2f

// Parámetros del modelo - aliases a los de CBR_InitPosition.h
#define K0_L            K0_PARAM
#define K_L             K_PARAM
#define A_L             A_PARAM

// Límites - Ya definidos en CBR_InitPosition.h, no redefinir

// ============================================================================
// ESTRUCTURA DE "CASO DE ORO" (Golden Case)
// Representa un punto de operación perfectamente caracterizado
// ============================================================================
struct GoldenCase {
    // Posición objetivo
    float y_target;     // Posición de equilibrio [m]
    
    // Variables de estado en equilibrio
    float i_eq;         // Corriente de equilibrio [A]
    float u_eq;         // Voltaje de equilibrio [V]
    float phi_eq;       // Flujo magnético [Wb]
    float L_eq;         // Inductancia en ese punto [H]
    
    // Dinámica característica
    float didt_rise;    // di/dt durante subida [A/s]
    float didt_fall;    // di/dt durante bajada [A/s]
    
    // Parámetros de control óptimo
    float kp_local;     // Ganancia proporcional óptima
    float ki_local;     // Ganancia integral óptima
    
    // Indicadores de calidad
    float confidence;   // Confianza del caso [0-1]
    int samples;        // Número de muestras que lo validaron
};

// ============================================================================
// BASE DE CASOS DE ORO (Barrido 2mm - 20mm)
// Estos valores se calibran con el sensor y luego se usan sensorless
// ============================================================================
#define NUM_GOLDEN_CASES 10

// CASOS DE ORO CALIBRADOS CON DATOS EXPERIMENTALES (27-Dic-2025)
// Generados por calibrar_cbr_kan.py usando levitador_maestro.exe
static GoldenCase golden_cases[NUM_GOLDEN_CASES] = {
    // y[m],    i_eq[A], u_eq[V], phi[Wb],  L[H],     di/dt+,  di/dt-,  kp,    ki,    conf, n
    {0.002f,   0.230f,  4.51f,   0.0215f,  0.0937f,  105.2f, -94.7f,  350.0f, 80.0f, 0.68f, 68},
    {0.004f,   0.422f,  6.71f,   0.0369f,  0.0875f,  112.7f, -101.4f, 340.0f, 75.0f, 1.00f, 642},
    {0.006f,   0.500f,  7.32f,   0.0418f,  0.0835f,  118.0f, -106.2f, 330.0f, 70.0f, 1.00f, 443},
    {0.008f,   0.493f,  6.96f,   0.0399f,  0.0808f,  122.1f, -109.9f, 320.0f, 65.0f, 1.00f, 137},
    {0.010f,   0.496f,  6.81f,   0.0391f,  0.0788f,  125.2f, -112.7f, 310.0f, 60.0f, 0.76f, 76},
    {0.012f,   0.549f,  7.37f,   0.0424f,  0.0772f,  127.7f, -114.9f, 300.0f, 55.0f, 0.58f, 58},
    {0.014f,   0.585f,  7.58f,   0.0445f,  0.0760f,  129.7f, -116.7f, 290.0f, 50.0f, 0.68f, 68},
    {0.016f,   0.630f,  1.39f,   0.0473f,  0.0750f,  131.4f, -118.3f, 280.0f, 45.0f, 0.50f, 0},
    {0.018f,   0.690f,  1.52f,   0.0512f,  0.0742f,  132.9f, -119.6f, 270.0f, 40.0f, 0.50f, 0},
    {0.020f,   0.750f,  1.65f,   0.0552f,  0.0735f,  134.1f, -120.7f, 260.0f, 35.0f, 0.50f, 0},
};

// ============================================================================
// CLASE CBR_KAN: Mapa de Control Universal
// ============================================================================
class CBR_KAN_Map {
private:
    // Estado interno
    float R_estimated;      // Resistencia estimada adaptativa
    float y_current;        // Posición actual estimada
    float phi_current;      // Flujo actual
    
    // Historial para HiPPO (memoria temporal)
    static const int HISTORY_LEN = 16;
    float i_history[HISTORY_LEN];
    float u_history[HISTORY_LEN];
    int hist_idx;
    
    // Índice del caso activo
    int active_case_idx;
    
public:
    CBR_KAN_Map() {
        R_estimated = R_NOMINAL;
        y_current = Y_DEFAULT;
        phi_current = 0.0f;
        hist_idx = 0;
        active_case_idx = 4; // Centro (10mm)
        
        for (int i = 0; i < HISTORY_LEN; i++) {
            i_history[i] = 0.0f;
            u_history[i] = 0.0f;
        }
    }
    
    // ========================================================================
    // BÚSQUEDA CBR: Encontrar el caso más cercano
    // ========================================================================
    int find_nearest_case(float y_target) {
        int best_idx = 0;
        float min_dist = 1e9f;
        
        for (int i = 0; i < NUM_GOLDEN_CASES; i++) {
            float dist = fabsf(golden_cases[i].y_target - y_target);
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }
        return best_idx;
    }
    
    // ========================================================================
    // INTERPOLACIÓN KAN: Spline cúbico entre dos casos
    // ========================================================================
    float kan_interpolate(float y_query, int case_low, int case_high, 
                          float (GoldenCase::*field)) {
        if (case_low == case_high) {
            return golden_cases[case_low].*field;
        }
        
        float y_lo = golden_cases[case_low].y_target;
        float y_hi = golden_cases[case_high].y_target;
        float v_lo = golden_cases[case_low].*field;
        float v_hi = golden_cases[case_high].*field;
        
        // Interpolación normalizada [0,1]
        float t = (y_query - y_lo) / (y_hi - y_lo);
        t = fmaxf(0.0f, fminf(1.0f, t));
        
        // Spline Hermite para suavidad (aproximación simple)
        float t2 = t * t;
        float t3 = t2 * t;
        float h = 3.0f * t2 - 2.0f * t3;  // Smoothstep
        
        return v_lo * (1.0f - h) + v_hi * h;
    }
    
    // ========================================================================
    // ESTIMACIÓN DE POSICIÓN: CBR + KAN + Física
    // ========================================================================
    float estimate_position(float i_measured, float u_measured, float dt) {
        // Actualizar historial
        i_history[hist_idx] = i_measured;
        u_history[hist_idx] = u_measured;
        hist_idx = (hist_idx + 1) % HISTORY_LEN;
        
        // 1. Estimar R adaptativo
        if (i_measured > 0.05f && fabsf(u_measured) > 0.1f) {
            float R_inst = u_measured / i_measured;
            if (R_inst > 1.5f && R_inst < 5.0f) {
                R_estimated = 0.98f * R_estimated + 0.02f * R_inst;
            }
        }
        
        // 2. Calcular di/dt del historial
        int prev_idx = (hist_idx - 2 + HISTORY_LEN) % HISTORY_LEN;
        float di = i_measured - i_history[prev_idx];
        float didt = di / (2.0f * dt);
        
        // 3. Restricción PINN: u = R*i + L*di/dt + dphi/dt_extra
        // Despejando L aproximada: L ≈ (u - R*i) / di/dt
        float L_estimated = K0_L + K_L / 2.0f;  // Default
        if (fabsf(didt) > 1.0f) {
            float v_inductivo = u_measured - R_estimated * i_measured;
            L_estimated = v_inductivo / didt;
            L_estimated = fmaxf(0.04f, fminf(0.12f, L_estimated));
        }
        
        // 4. Invertir L(y) para obtener y
        // L = K0 + K/(1 + y/a)  =>  y = a * (K/(L-K0) - 1)
        float y_from_L = Y_DEFAULT;
        float denom = L_estimated - K0_L;
        if (fabsf(denom) > 0.001f) {
            y_from_L = A_L * (K_L / denom - 1.0f);
            y_from_L = fmaxf(Y_MIN, fminf(Y_MAX, y_from_L));
        }
        
        // 5. Búsqueda CBR para validación cruzada
        int idx = find_nearest_case(y_from_L);
        float y_cbr = golden_cases[idx].y_target;
        float i_expected = golden_cases[idx].i_eq;
        
        // 6. Fusión CBR + Física (KAN implícito en interpolación)
        // Si la corriente medida difiere mucho del caso, ajustar
        float i_error = fabsf(i_measured - i_expected) / (i_expected + 0.01f);
        float cbr_weight = 1.0f / (1.0f + i_error * 5.0f);  // Menos peso si hay error
        
        float y_fused = cbr_weight * y_cbr + (1.0f - cbr_weight) * y_from_L;
        
        // 7. Filtro paso bajo para estabilidad
        y_current = 0.7f * y_current + 0.3f * y_fused;
        
        active_case_idx = idx;
        return y_current;
    }
    
    // ========================================================================
    // OBTENER PARÁMETROS DE CONTROL INTERPOLADOS
    // ========================================================================
    void get_control_params(float y_target, float* kp_out, float* ki_out) {
        // Encontrar casos adyacentes
        int idx_lo = 0, idx_hi = NUM_GOLDEN_CASES - 1;
        
        for (int i = 0; i < NUM_GOLDEN_CASES - 1; i++) {
            if (golden_cases[i].y_target <= y_target && 
                golden_cases[i+1].y_target >= y_target) {
                idx_lo = i;
                idx_hi = i + 1;
                break;
            }
        }
        
        // Interpolación KAN para ganancias
        *kp_out = kan_interpolate(y_target, idx_lo, idx_hi, &GoldenCase::kp_local);
        *ki_out = kan_interpolate(y_target, idx_lo, idx_hi, &GoldenCase::ki_local);
    }
    
    // ========================================================================
    // OBTENER CORRIENTE DE EQUILIBRIO ESPERADA
    // ========================================================================
    float get_equilibrium_current(float y_target) {
        int idx_lo = 0, idx_hi = NUM_GOLDEN_CASES - 1;
        
        for (int i = 0; i < NUM_GOLDEN_CASES - 1; i++) {
            if (golden_cases[i].y_target <= y_target && 
                golden_cases[i+1].y_target >= y_target) {
                idx_lo = i;
                idx_hi = i + 1;
                break;
            }
        }
        
        return kan_interpolate(y_target, idx_lo, idx_hi, &GoldenCase::i_eq);
    }
    
    // ========================================================================
    // ACTUALIZAR CASO CON NUEVA OBSERVACIÓN (Aprendizaje Online)
    // ========================================================================
    void update_case(int idx, float i_measured, float u_measured, float y_actual) {
        if (idx < 0 || idx >= NUM_GOLDEN_CASES) return;
        
        GoldenCase& gc = golden_cases[idx];
        
        // Solo actualizar si la posición coincide aproximadamente
        if (fabsf(y_actual - gc.y_target) < 0.001f) {
            float alpha = 0.1f / (gc.samples + 1);  // Tasa de aprendizaje decreciente
            
            gc.i_eq = (1.0f - alpha) * gc.i_eq + alpha * i_measured;
            gc.u_eq = (1.0f - alpha) * gc.u_eq + alpha * u_measured;
            gc.samples++;
            
            // Actualizar confianza
            gc.confidence = fminf(1.0f, gc.confidence + 0.01f);
        }
    }
    
    // Getters
    float get_R_estimated() const { return R_estimated; }
    float get_current_position() const { return y_current; }
    int get_active_case() const { return active_case_idx; }
};

// Instancia global
static CBR_KAN_Map cbr_kan_map;

#endif // CBR_KAN_UNIVERSAL_H

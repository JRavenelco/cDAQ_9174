/*
 * CBR_InitPosition.h
 * Módulo de Razonamiento Basado en Casos (CBR) para estimación de posición inicial
 * 
 * Usa la relación física: L(y) = V_bus / (di/dt)
 * Donde L(y) = K0 + K / (1 + y/a) [Modelo de Santana 2023]
 * 
 * Despejando y: y = a * (K / (L - K0) - 1)
 */

#ifndef CBR_INIT_POSITION_H
#define CBR_INIT_POSITION_H

#include <math.h>

// ============================================================================
// CONSTANTES FÍSICAS DEL SISTEMA
// ============================================================================
#define MASS_KG         0.009f      // Masa del objeto: 9.0 g = 0.009 kg
#define GRAVITY         9.81f       // Aceleración gravitacional [m/s²]
#define V_BUS           9.86f       // Voltaje del bus [V] (igual a Vref)
#define R_SHUNT         2.2f        // Resistencia Shunt [Ω]

// Parámetros del modelo de inductancia (Identificados Fase 1 - Benchmark)
#define K0_PARAM        0.0363f     // Inductancia base [H]
#define K_PARAM         0.0035f     // Parámetro de posición [H]
#define A_PARAM         0.0052f     // Parámetro geométrico [m]

// Límites físicos
#define Y_MIN           0.0005f     // Posición mínima [m] = 0.5 mm
#define Y_MAX           0.022f      // Posición máxima [m] = 22 mm
#define Y_DEFAULT       0.015f      // Posición por defecto si CBR falla [m] = 15 mm

// ============================================================================
// BASE DE CASOS (Calibrados empíricamente o de MONIT.txt)
// Formato: {di/dt [A/s], y_0 [m]}
// Relación: di/dt alto -> baja inductancia -> posición alta (lejos)
//           di/dt bajo -> alta inductancia -> posición baja (cerca)
// ============================================================================
#define NUM_CASES 10

struct CBR_Case {
    float didt;     // Pendiente de corriente [A/s]
    float y0;       // Posición correspondiente [m]
};

// Casos iniciales calibrados usando L(y) = V_bus / (di/dt)
// y = a * (K / (L - K0) - 1)
// Estos valores se pueden refinar con datos reales de MONIT.txt
static const CBR_Case case_base[NUM_CASES] = {
    // di/dt [A/s],  y0 [m]
    {  50.0f,       0.020f },   // Muy lejos: L = 0.197 H
    {  75.0f,       0.015f },   // Lejos: L = 0.131 H
    { 100.0f,       0.012f },   // Medio-lejos: L = 0.0986 H
    { 125.0f,       0.010f },   // Medio: L = 0.0789 H
    { 150.0f,       0.008f },   // Medio-cerca: L = 0.0657 H
    { 175.0f,       0.007f },   // Cerca: L = 0.0563 H
    { 200.0f,       0.006f },   // Muy cerca: L = 0.0493 H
    { 250.0f,       0.005f },   // Muy muy cerca: L = 0.0394 H
    { 300.0f,       0.004f },   // Casi tocando: L = 0.0329 H
    { 400.0f,       0.003f },   // Contacto: L = 0.0247 H
};

// ============================================================================
// FUNCIONES DEL MÓDULO CBR
// ============================================================================

/**
 * Calcula la inductancia esperada para una posición dada
 * L(y) = K0 + K / (1 + y/a)
 */
inline float calcular_inductancia(float y) {
    return K0_PARAM + K_PARAM / (1.0f + y / A_PARAM);
}

/**
 * Calcula la posición esperada para una inductancia dada
 * y = a * (K / (L - K0) - 1)
 */
inline float calcular_posicion_desde_L(float L) {
    float denom = L - K0_PARAM;
    if (fabs(denom) < 1e-6f) return Y_DEFAULT;
    
    float y = A_PARAM * (K_PARAM / denom - 1.0f);
    
    // Clamping a rango físico
    if (y < Y_MIN) y = Y_MIN;
    if (y > Y_MAX) y = Y_MAX;
    
    return y;
}

/**
 * Estima la posición inicial usando la relación física directa
 * L = V_bus / (di/dt)  ->  y = f(L)
 * 
 * @param didt Pendiente de corriente medida [A/s]
 * @return Posición estimada [m]
 */
inline float cbr_fisica_directa(float didt) {
    if (didt < 10.0f) return Y_MAX;  // Corriente casi no sube -> muy lejos
    if (didt > 500.0f) return Y_MIN; // Corriente sube muy rápido -> muy cerca
    
    // L = V / (di/dt) [Simplificación ignorando R*i inicial]
    float L_estimada = V_BUS / didt;
    
    return calcular_posicion_desde_L(L_estimada);
}

/**
 * Búsqueda K-Nearest Neighbors (K=1) en la base de casos
 * Encuentra el caso más cercano por distancia euclidiana en di/dt
 * 
 * @param didt Pendiente de corriente medida [A/s]
 * @return Posición del caso más cercano [m]
 */
inline float cbr_knn_busqueda(float didt) {
    float min_dist = 1e9f;
    float y_mejor = Y_DEFAULT;
    
    for (int i = 0; i < NUM_CASES; i++) {
        float dist = fabs(didt - case_base[i].didt);
        if (dist < min_dist) {
            min_dist = dist;
            y_mejor = case_base[i].y0;
        }
    }
    
    return y_mejor;
}

/**
 * Interpolación lineal entre los dos casos más cercanos
 * Más preciso que K-NN simple
 * 
 * @param didt Pendiente de corriente medida [A/s]
 * @return Posición interpolada [m]
 */
inline float cbr_interpolacion(float didt) {
    // Encontrar los dos casos más cercanos (uno menor, uno mayor)
    int idx_menor = -1, idx_mayor = -1;
    float didt_menor = -1e9f, didt_mayor = 1e9f;
    
    for (int i = 0; i < NUM_CASES; i++) {
        if (case_base[i].didt <= didt && case_base[i].didt > didt_menor) {
            didt_menor = case_base[i].didt;
            idx_menor = i;
        }
        if (case_base[i].didt >= didt && case_base[i].didt < didt_mayor) {
            didt_mayor = case_base[i].didt;
            idx_mayor = i;
        }
    }
    
    // Casos extremos
    if (idx_menor < 0) return case_base[idx_mayor].y0;
    if (idx_mayor < 0) return case_base[idx_menor].y0;
    if (idx_menor == idx_mayor) return case_base[idx_menor].y0;
    
    // Interpolación lineal
    float t = (didt - didt_menor) / (didt_mayor - didt_menor);
    float y_interp = case_base[idx_menor].y0 * (1.0f - t) + case_base[idx_mayor].y0 * t;
    
    // Clamping
    if (y_interp < Y_MIN) y_interp = Y_MIN;
    if (y_interp > Y_MAX) y_interp = Y_MAX;
    
    return y_interp;
}

/**
 * Función principal de estimación CBR
 * Combina física directa con interpolación de casos para robustez
 * 
 * @param didt Pendiente de corriente medida [A/s]
 * @return Posición estimada [m]
 */
inline float cbr_estimar_posicion(float didt) {
    // Método 1: Física directa (L = V/di_dt)
    float y_fisica = cbr_fisica_directa(didt);
    
    // Método 2: Interpolación de casos
    float y_casos = cbr_interpolacion(didt);
    
    // Fusión: 70% física + 30% casos (la física es más confiable)
    float y_final = 0.7f * y_fisica + 0.3f * y_casos;
    
    // Clamping final
    if (y_final < Y_MIN) y_final = Y_MIN;
    if (y_final > Y_MAX) y_final = Y_MAX;
    
    return y_final;
}

/**
 * Calcula la corriente de equilibrio para una posición dada
 * Basado en el balance de fuerzas: F_mag = m*g
 * F_mag ≈ (1/2) * (dL/dy) * i²
 * 
 * @param y Posición [m]
 * @return Corriente de equilibrio [A]
 */
inline float calcular_corriente_equilibrio(float y) {
    // dL/dy = -K / (a * (1 + y/a)²) = -K*a / (a + y)²
    float dL_dy = -K_PARAM * A_PARAM / ((A_PARAM + y) * (A_PARAM + y));
    
    // F_mag = (1/2) * |dL/dy| * i² = m * g
    // i² = 2 * m * g / |dL/dy|
    float i_sq = 2.0f * MASS_KG * GRAVITY / fabs(dL_dy);
    
    if (i_sq < 0.0f) return 0.3f; // Default seguro
    
    return sqrtf(i_sq);
}

/**
 * Calcula el flujo inicial para una posición y corriente dadas
 * φ = L(y) * i
 * 
 * @param y Posición [m]
 * @param i Corriente [A]
 * @return Flujo magnético [Wb]
 */
inline float calcular_flujo_inicial(float y, float i) {
    float L = calcular_inductancia(y);
    return L * i;
}

#endif // CBR_INIT_POSITION_H

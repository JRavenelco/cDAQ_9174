/*
 * Levitador Magnético con Observador de Posición
 * ===============================================
 * 
 * Incluye:
 * - Control PID en cascada (posición -> corriente -> voltaje)
 * - Observador de posición basado en integración de flujo
 * - Estimador de resistencia adaptativo (elimina drift)
 * 
 * Autor: Basado en código original de Valentín
 * Modificado: 19-Dic-2025
 */

#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <math.h>

// ============================================================================
// PARÁMETROS DEL MODELO MAGNÉTICO (optimizados offline)
// ============================================================================
#define K0  0.0657f    // Inductancia base [H]
#define K   0.0393f    // Variación de inductancia [H]
#define A   0.00498f   // Parámetro geométrico [m]
#define M   0.018f     // Masa de la esfera [kg]
#define G   9.81f      // Gravedad [m/s²]

// ============================================================================
// PARÁMETROS DE CONTROL PID
// ============================================================================
#define Ts   0.01f     // Periodo de muestreo [s]
#define kp   100.0f    // Ganancia proporcional
#define ki   50.0f     // Ganancia integral
#define kd   1.5f      // Ganancia derivativa
#define kpi  12.0f     // Ganancia proporcional de corriente
#define kii  3000.0f   // Ganancia integral de corriente
#define Vref 9.86f     // Voltaje de referencia [V]
#define Iref 0.827f    // Corriente de referencia [A]
#define Rs   2.2f      // Resistencia de sensado [Ohm]

// ============================================================================
// PARÁMETROS DEL ESTIMADOR DE RESISTENCIA
// ============================================================================
#define R0_EST      16.0f    // Resistencia inicial (efectiva) [Ohm]
#define ALPHA_EST   0.0f     // Coef. calentamiento (desactivado)
#define BETA_EST    0.02f    // Coef. enfriamiento (dinámica lenta)
#define GAIN_FUSION 0.1f     // Ganancia de fusión
#define I_MIN_CORR  0.05f    // Corriente mínima para corrección [A]
#define DECIMATION  10       // Actualizar R cada N pasos

// ============================================================================
// ESTRUCTURA: ESTIMADOR DE RESISTENCIA
// ============================================================================
typedef struct {
    float R;           // Resistencia estimada [Ohm]
    float R_amb;       // Resistencia ambiente [Ohm]
    float alpha;       // Coef. calentamiento
    float beta;        // Coef. enfriamiento
    float gain;        // Ganancia de fusión
    float i_min;       // Corriente mínima
    float u_avg;       // Voltaje promediado
    float i_avg;       // Corriente promediada
    float alpha_filt;  // Filtro pasabajas
    int steps;         // Contador de pasos
    int decimation;    // Decimación
} EstimadorResistencia;

void EstimadorR_init(EstimadorResistencia *est) {
    est->R = R0_EST;
    est->R_amb = R0_EST;
    est->alpha = ALPHA_EST;
    est->beta = BETA_EST;
    est->gain = GAIN_FUSION;
    est->i_min = I_MIN_CORR;
    est->u_avg = 0.0f;
    est->i_avg = 0.0f;
    est->alpha_filt = Ts / 0.5f;  // tau = 0.5s
    est->steps = 0;
    est->decimation = DECIMATION;
}

float EstimadorR_update(EstimadorResistencia *est, float u, float i) {
    // 1. Predicción térmica (desactivada si alpha=0)
    float power = est->R * i * i;
    float dR = (est->alpha * power - est->beta * (est->R - est->R_amb)) * Ts;
    est->R += dR;
    
    // Clamp de seguridad
    if (est->R > est->R_amb * 1.6f) est->R = est->R_amb * 1.6f;
    if (est->R < est->R_amb * 0.5f) est->R = est->R_amb * 0.5f;
    
    // 2. Corrección con Ley de Ohm (lazo lento)
    est->u_avg = (1.0f - est->alpha_filt) * est->u_avg + est->alpha_filt * u;
    est->i_avg = (1.0f - est->alpha_filt) * est->i_avg + est->alpha_filt * i;
    
    est->steps++;
    if (est->steps % est->decimation == 0) {
        if (fabs(est->i_avg) > est->i_min) {
            float R_meas = est->u_avg / est->i_avg;
            
            // Validar coherencia
            if (R_meas > 0.5f * est->R_amb && R_meas < 2.0f * est->R_amb) {
                est->R += est->gain * (R_meas - est->R);
            }
        }
    }
    
    return est->R;
}

// ============================================================================
// ESTRUCTURA: OBSERVADOR DE POSICIÓN
// ============================================================================
typedef struct {
    float k0, k, a;      // Parámetros del modelo magnético
    float r;             // Resistencia (dinámica)
    float phi;           // Flujo magnético integrado
    float i_prev;        // Corriente previa
    float u_prev;        // Voltaje previo
    float y_est;         // Posición estimada
    float dy_est;        // Velocidad estimada
    float y_est_prev;    // Posición estimada previa
    float alpha_pos;     // Filtro de posición
    int initialized;     // Flag de inicialización
    EstimadorResistencia est_R;  // Estimador de resistencia
} ObservadorPosicion;

void Observador_init(ObservadorPosicion *obs) {
    obs->k0 = K0;
    obs->k = K;
    obs->a = A;
    obs->r = R0_EST;
    obs->phi = 0.0f;
    obs->i_prev = 0.0f;
    obs->u_prev = 0.0f;
    obs->y_est = 0.005f;
    obs->dy_est = 0.0f;
    obs->y_est_prev = 0.005f;
    obs->alpha_pos = 0.25f;
    obs->initialized = 0;
    
    EstimadorR_init(&obs->est_R);
}

float Observador_inductancia(ObservadorPosicion *obs, float y) {
    return obs->k0 + obs->k / (1.0f + y / obs->a);
}

float Observador_posicion_desde_L(ObservadorPosicion *obs, float L) {
    float denom = L - obs->k0;
    if (denom > 0.001f) {
        float y = obs->a * (obs->k / denom - 1.0f);
        if (y < 0.001f) y = 0.001f;
        if (y > 0.020f) y = 0.020f;
        return y;
    }
    return obs->y_est;
}

float Observador_estimar_equilibrio(ObservadorPosicion *obs, float i) {
    if (i < 0.05f) return 0.005f;
    
    // F_mag = m*g en equilibrio
    // (k*i²)/(2*a*(1+y/a)²) = m*g
    float termino = obs->k * i * i / (2.0f * obs->a * M * G);
    if (termino > 0.0f) {
        float raiz = sqrtf(termino);
        if (raiz > 1.0f) {
            float y = obs->a * (raiz - 1.0f);
            if (y > 0.001f && y < 0.020f) return y;
        }
    }
    return 0.005f;
}

void Observador_estimar(ObservadorPosicion *obs, float i, float u, float *y_out, float *dy_out) {
    // Actualizar resistencia dinámica
    obs->r = EstimadorR_update(&obs->est_R, u, i);
    
    // Inicialización
    if (!obs->initialized) {
        if (i > 0.03f) {
            obs->initialized = 1;
            float y_init = Observador_estimar_equilibrio(obs, i);
            obs->y_est = (y_init > 0.001f) ? y_init : 0.005f;
            float L_init = Observador_inductancia(obs, obs->y_est);
            obs->phi = L_init * i;
        }
        obs->i_prev = i;
        obs->u_prev = u;
        *y_out = obs->y_est;
        *dy_out = obs->dy_est;
        return;
    }
    
    // Integrar flujo (trapecio) - USANDO R DINÁMICA
    float z_now = u - obs->r * i;
    float z_prev = obs->u_prev - obs->r * obs->i_prev;
    float dphi = 0.5f * (z_now + z_prev) * Ts;
    obs->phi += dphi;
    
    // === MÉTODO 1: FÓRMULA 2023 (José Santana) ===
    // y = (a*kg*i)/(φ + L0*i0 - k0*i) - a
    // Esta es tu fórmula EXACTA de observador.m línea 168-169
    float y_2023 = 0.0f;
    float L0 = obs->k0 + obs->k;  // Inductancia inicial
    float denom = obs->phi + L0 * 0.0f - obs->k0 * i;  // i0=0 simplificado
    
    if (fabs(denom) > 1e-6f && i > 0.05f) {
        y_2023 = (obs->a * obs->k * i) / denom - obs->a;
        
        // Limitar rango físico
        if (y_2023 < 0.0005f) y_2023 = 0.0005f;
        if (y_2023 > 0.022f) y_2023 = 0.022f;
    }
    
    // === MÉTODO 2: EQUILIBRIO (respaldo) ===
    float y_eq = Observador_estimar_equilibrio(obs, i);
    
    // === FUSIÓN ===
    float y_nuevo;
    if (y_2023 > 0.001f && y_2023 < 0.020f) {
        // Usar fórmula 2023 si es válida
        if (y_eq > 0.001f && y_eq < 0.020f) {
            // Ponderar con equilibrio
            float di = fabs(i - obs->i_prev) / Ts;
            float peso_eq = 1.0f / (1.0f + di * 50.0f);
            y_nuevo = peso_eq * y_eq + (1.0f - peso_eq) * y_2023;
        } else {
            y_nuevo = y_2023;
        }
    } else if (y_eq > 0.001f) {
        y_nuevo = y_eq;
    } else {
        y_nuevo = obs->y_est;
    }
    
    // Filtro paso bajo (alpha=0.3 como en Python)
    obs->y_est = obs->alpha_pos * y_nuevo + (1.0f - obs->alpha_pos) * obs->y_est;
    
    // Corrección de drift del integrador
    float L_esperada = Observador_inductancia(obs, obs->y_est);
    obs->phi = 0.9f * obs->phi + 0.1f * L_esperada * i;
    
    // Velocidad
    obs->dy_est = (obs->y_est - obs->y_est_prev) / Ts;
    if (obs->dy_est < -1.0f) obs->dy_est = -1.0f;
    if (obs->dy_est > 1.0f) obs->dy_est = 1.0f;
    obs->y_est_prev = obs->y_est;
    
    // Limitar posición
    if (obs->y_est < 0.0005f) obs->y_est = 0.0005f;
    if (obs->y_est > 0.022f) obs->y_est = 0.022f;
    
    obs->i_prev = i;
    obs->u_prev = u;
    
    *y_out = obs->y_est;
    *dy_out = obs->dy_est;
}

// ============================================================================
// VARIABLES GLOBALES
// ============================================================================
float yd = 0.005f;
float ef = 0.0f, ef_1 = 0.0f, y_prev = 0.0f;
float integral = 0.0f, intei = 0.0f;
float u = 0.0f;
ObservadorPosicion observador;

int USE_OBSERVER_FOR_CONTROL = 0;  // 0=SENSOR, 1=OBSERVADOR

using namespace std;

int main()
{
    HANDLE h;
    DCB dcb;
    BOOL fSuccess;
    DWORD BytesWritten, BytesRead;
    COMMTIMEOUTS timeouts;
    
    float esc = 0.05f / 1023.0f;
    float esci = 5.0f / (Rs * 1023.0f);
    float escs = 254.0f / Vref;
    float iTs = 1.0f / Ts;
    
    unsigned char buffer[7];
    unsigned int pv = 0, icte = 0;
    float t = 0.0f;
    
    // Inicializar observador
    Observador_init(&observador);
    
    printf("========================================================\n");
    printf("  CONTROL CON OBSERVADOR DE POSICION\n");
    printf("========================================================\n");
    printf("Parametros del modelo:\n");
    printf("  K0 = %.4f H\n", K0);
    printf("  K  = %.4f H\n", K);
    printf("  A  = %.5f m\n", A);
    printf("  R0 = %.2f Ohm (estimador adaptativo)\n", R0_EST);
    printf("Control: %s\n", USE_OBSERVER_FOR_CONTROL ? "OBSERVADOR" : "SENSOR");
    printf("========================================================\n\n");
    
    // Configurar puerto serial
    h = CreateFile("COM1", GENERIC_READ | GENERIC_WRITE, 0, 0,
                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    
    if (h == INVALID_HANDLE_VALUE) {
        printf("Error abriendo COM1\n");
        return 1;
    }
    
    fSuccess = GetCommState(h, &dcb);
    if (!fSuccess) {
        printf("Error GetCommState\n");
        CloseHandle(h);
        return 1;
    }
    
    dcb.BaudRate = CBR_115200;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    
    fSuccess = SetCommState(h, &dcb);
    if (!fSuccess) {
        printf("Error SetCommState\n");
        CloseHandle(h);
        return 1;
    }
    
    // Configurar timeouts
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 1000;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant = 0;
    timeouts.WriteTotalTimeoutMultiplier = 0;
    SetCommTimeouts(h, &timeouts);
    
    printf("Puerto COM1 abierto.\n");
    printf("Esperando switch (0xAA)...\n");
    
    // Esperar activación
    while (1) {
        ReadFile(h, buffer, 1, &BytesRead, NULL);
        if (BytesRead > 0 && buffer[0] == 0xAA) {
            printf("Switch activado!\n\n");
            break;
        }
    }
    
    // Abrir archivo de log
    FILE *fp = fopen("MONIT_observador.txt", "w");
    fprintf(fp, "# t\tyd\ty_sensor\ty_obs\tdy_obs\tie\tu\tR_est\n");
    
    printf("t[s]\tSensor\tObs\tR_est\n");
    printf("--------------------------------------------\n");
    
    // Lazo de control
    while (!_kbhit()) {
        // Leer datos del microcontrolador
        ReadFile(h, buffer, 6, &BytesRead, NULL);
        
        if (BytesRead == 6 && buffer[0] == 0xAA) {
            pv = (buffer[1] << 8) + buffer[2];
            icte = (buffer[3] << 8) + buffer[4];
            
            if (pv <= 1023 && icte <= 1023) {
                // Mediciones
                float y_sensor = esc * pv;
                float ie = esci * icte;
                
                // === OBSERVADOR (solo monitoreo) ===
                // TEMPORALMENTE DESACTIVADO PARA DEBUG
                float y_obs = 0.005f, dy_obs = 0.0f;
                // Observador_estimar(&observador, ie, u, &y_obs, &dy_obs);
                
                // Seleccionar realimentación
                float y = y_sensor;  // FORZAR SENSOR (debug)
                
                // === CONTROL PID ===
                ef_1 = ef;
                ef = yd - y;
                float proporcional = kp * ef;
                float derivativa = kd * (ef - ef_1) * iTs;
                
                if (integral > -Iref && integral < Iref) {
                    integral += ki * Ts * ef;
                } else {
                    integral = (integral >= Iref) ? 0.95f * Iref : -0.95f * Iref;
                }
                
                float id_val = proporcional + integral + derivativa;
                if (id_val > 0.0f) id_val = 0.0f;
                if (id_val <= -Iref) id_val = -Iref;
                
                // Lazo de corriente
                float ied = -id_val;
                float ei = ied - ie;
                float propi = kpi * ei;
                
                if (intei > -Vref && intei < Vref) {
                    intei += kii * Ts * ei;
                } else {
                    intei = (intei >= Vref) ? 0.95f * Vref : -0.95f * Vref;
                }
                
                u = propi + intei;
                if (u > Vref) u = Vref;
                if (u <= 0.0f) u = 0.0f;
                
                // Enviar PWM
                unsigned char pwm = (unsigned char)(escs * u);
                if (pwm > 254) pwm = 254;
                WriteFile(h, &pwm, 1, &BytesWritten, NULL);
                
                y_prev = y;
                
                // Log
                fprintf(fp, "%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.4f\t%.4f\t%.4f\n",
                        t, yd, y_sensor, y_obs, dy_obs, ie, u, observador.r);
                
                // Consola (cada 0.5s)
                if ((int)(t * 10) % 5 == 0) {
                    printf("%.1f\t%.1f\t%.1f\t%.2f\n",
                           t, y_sensor * 1000, y_obs * 1000, observador.r);
                }
                
                t += Ts;
            }
        }
    }
    
    fclose(fp);
    CloseHandle(h);
    
    printf("\n\nPrograma finalizado.\n");
    printf("Datos guardados en MONIT_observador.txt\n");
    
    return 0;
}

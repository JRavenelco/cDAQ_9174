/*
 * Levitador Magnético CON Observador
 * Basado en levitador.cpp FUNCIONAL + Observador 2023
 */

#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <math.h>

// Parámetros modelo magnético
#define K0  0.0657f
#define K   0.0393f
#define A   0.00498f
#define M   0.018f
#define G   9.81f

// Parámetros PID (ORIGINALES de levitador.cpp)
#define Ts   0.01
#define kp   100
#define ki   50
#define kd   1.5
#define kpi  12.0
#define kii  3000.0
#define Vref 9.86
#define Iref 0.827
#define Rs   2.2

// Parámetros Estimador R
#define R0_EST 16.0f
#define ALPHA_EST 0.0f
#define BETA_EST 0.02f
#define GAIN_FUSION 0.1f

// ============================================================================
// ESTIMADOR DE RESISTENCIA
// ============================================================================
typedef struct {
    float R, R_amb, alpha, beta, gain, i_min;
    float u_avg, i_avg, alpha_filt;
    int steps, decimation;
} EstimadorResistencia;

void EstimadorR_init(EstimadorResistencia *est) {
    est->R = R0_EST;
    est->R_amb = R0_EST;
    est->alpha = ALPHA_EST;
    est->beta = BETA_EST;
    est->gain = GAIN_FUSION;
    est->i_min = 0.05f;
    est->u_avg = 0.0f;
    est->i_avg = 0.0f;
    est->alpha_filt = Ts / 0.5f;
    est->steps = 0;
    est->decimation = 10;
}

float EstimadorR_update(EstimadorResistencia *est, float u, float i) {
    float power = est->R * i * i;
    float dR = (est->alpha * power - est->beta * (est->R - est->R_amb)) * Ts;
    est->R += dR;
    
    // Clamp manual
    float R_min = est->R_amb * 0.5f;
    float R_max = est->R_amb * 1.6f;
    if (est->R < R_min) est->R = R_min;
    if (est->R > R_max) est->R = R_max;
    
    est->u_avg = (1.0f - est->alpha_filt) * est->u_avg + est->alpha_filt * u;
    est->i_avg = (1.0f - est->alpha_filt) * est->i_avg + est->alpha_filt * i;
    
    est->steps++;
    if (est->steps % est->decimation == 0 && fabs(est->i_avg) > est->i_min) {
        float R_meas = est->u_avg / est->i_avg;
        if (0.5f * est->R_amb < R_meas && R_meas < 2.0f * est->R_amb) {
            est->R += est->gain * (R_meas - est->R);
        }
    }
    return est->R;
}

// ============================================================================
// OBSERVADOR 2023
// ============================================================================
typedef struct {
    float k0, k, a;
    EstimadorResistencia est_R;
    float phi, i_prev, u_prev;
    float y_est, y_est_prev, dy_est;
    float alpha_pos;
    int initialized;
    float r;
} ObservadorPosicion;

void Observador_init(ObservadorPosicion *obs) {
    obs->k0 = K0;
    obs->k = K;
    obs->a = A;
    EstimadorR_init(&obs->est_R);
    obs->phi = 0.0f;
    obs->i_prev = 0.0f;
    obs->u_prev = 0.0f;
    obs->y_est = 0.005f;
    obs->y_est_prev = 0.005f;
    obs->dy_est = 0.0f;
    obs->alpha_pos = 0.3f;
    obs->initialized = 0;
    obs->r = R0_EST;
}

float Observador_estimar_equilibrio(ObservadorPosicion *obs, float i) {
    float mg = M * G;
    if (i > 0.01f) {
        float termino = obs->k * i * i / (2.0f * obs->a * mg);
        if (termino > 0.0f) {
            float raiz = sqrtf(termino);
            if (raiz > 1.0f) {
                float y = obs->a * (raiz - 1.0f);
                if (y > 0.001f && y < 0.020f) return y;
            }
        }
    }
    return 0.005f;
}

void Observador_estimar(ObservadorPosicion *obs, float i, float u, float *y_out, float *dy_out) {
    obs->r = EstimadorR_update(&obs->est_R, u, i);
    
    if (!obs->initialized) {
        if (i > 0.03f) {
            obs->initialized = 1;
            float y_init = Observador_estimar_equilibrio(obs, i);
            obs->y_est = (y_init > 0.001f) ? y_init : 0.005f;
            float L_init = obs->k0 + obs->k / (1.0f + obs->y_est / obs->a);
            obs->phi = L_init * i;
        }
        obs->i_prev = i;
        obs->u_prev = u;
        *y_out = obs->y_est;
        *dy_out = obs->dy_est;
        return;
    }
    
    // Integrar flujo (trapecio)
    float z_now = u - obs->r * i;
    float z_prev = obs->u_prev - obs->r * obs->i_prev;
    float dphi = 0.5f * (z_now + z_prev) * Ts;
    obs->phi += dphi;
    
    // Fórmula 2023: y = (a*k*i)/(φ + L0*i0 - k0*i) - a
    float L0 = obs->k0 + obs->k;
    float denom = obs->phi + L0 * 0.0f - obs->k0 * i;
    float y_2023 = 0.005f;
    
    if (fabs(denom) > 1e-6f && i > 0.05f) {
        y_2023 = (obs->a * obs->k * i) / denom - obs->a;
        if (y_2023 < 0.0005f) y_2023 = 0.0005f;
        if (y_2023 > 0.022f) y_2023 = 0.022f;
    }
    
    // Fusión con equilibrio
    float y_eq = Observador_estimar_equilibrio(obs, i);
    float y_nuevo = y_2023;
    
    if (y_2023 > 0.001f && y_2023 < 0.020f && y_eq > 0.001f && y_eq < 0.020f) {
        float di = fabs(i - obs->i_prev) / Ts;
        float peso_eq = 1.0f / (1.0f + di * 50.0f);
        y_nuevo = peso_eq * y_eq + (1.0f - peso_eq) * y_2023;
    }
    
    // Filtro
    obs->y_est = obs->alpha_pos * y_nuevo + (1.0f - obs->alpha_pos) * obs->y_est;
    
    // Corrección drift
    float L_esperada = obs->k0 + obs->k / (1.0f + obs->y_est / obs->a);
    obs->phi = 0.9f * obs->phi + 0.1f * L_esperada * i;
    
    // Velocidad
    obs->dy_est = (obs->y_est - obs->y_est_prev) / Ts;
    obs->y_est_prev = obs->y_est;
    
    obs->i_prev = i;
    obs->u_prev = u;
    
    *y_out = obs->y_est;
    *dy_out = obs->dy_est;
}

// ============================================================================
// VARIABLES GLOBALES (COMO levitador.cpp ORIGINAL)
// ============================================================================
float pwmf = 0, yd = 0.005;
float proporcional = 0, derivativa = 0, ie = 0, ied = 0, id_val = 0;
float ei = 0, propi = 0, intei = 0, integral = 0;
float y_prev = 0.005;

ObservadorPosicion observador;
int USE_OBSERVER = 0;  // 0=SENSOR (monitoreo), 1=OBSERVADOR (control)

using namespace std;

int main()
{
    HANDLE h;
    DCB dcb;
    BOOL fSuccess;
    char chRead;
    DWORD dwRead, BytesWritten;
    COMMTIMEOUTS timeouts;
    
    float esc, esci, escs, iTs;
    unsigned char dato[7];
    unsigned int pv = 0, icte = 0;
    int flagcom = 0, flag = 0;
    float t = 0, ef = 0, ef_1 = 0;
    
    esc = 0.05 / 1023.0;
    esci = 5.0 / (Rs * 1023.0);
    escs = 254.0 / Vref;
    iTs = 1 / Ts;
    
    // Inicializar observador
    Observador_init(&observador);
    
    printf("========================================================\n");
    printf("  LEVITADOR CON OBSERVADOR 2023\n");
    printf("========================================================\n");
    printf("Modo: %s\n", USE_OBSERVER ? "OBSERVADOR (sensorless)" : "SENSOR (monitoreo)");
    printf("========================================================\n\n");
    
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
    
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    SetCommTimeouts(h, &timeouts);
    
    printf("Puerto COM1 abierto\n");
    printf("Esperando switch (0xAA)...\n");
    
    while (1) {
        ReadFile(h, &chRead, 1, &dwRead, NULL);
        if (dwRead == 1 && chRead == (char)0xAA) {
            printf("Switch activado!\n\n");
            break;
        }
    }
    
    FILE *fp = fopen("MONIT_obs_funcional.txt", "w");
    fprintf(fp, "# t\tyd\ty_sensor\ty_obs\tdy_obs\tie\tu\tR_est\n");
    
    printf("t[s]\tSensor\tObs\tMAE\n");
    printf("----------------------------------------\n");
    
    // LOOP DE CONTROL (EXACTO de levitador.cpp)
    while (!_kbhit()) {
        fSuccess = ReadFile(h, &chRead, 1, &dwRead, NULL);
        
        if (dwRead == 1) {
            dato[flagcom] = chRead;
            
            if (flagcom == 0 && dato[0] != (char)0xAA) {
                flagcom = 0;
            } else {
                flagcom++;
                if (flagcom == 6) {
                    pv = ((int)dato[1] << 8) + (int)dato[2];
                    icte = ((int)dato[3] << 8) + (int)dato[4];
                    
                    if (pv <= 1023 && icte <= 1023) {
                        float y_sensor = esc * pv;
                        ie = esci * icte;
                        
                        // Observador (siempre corre para comparar)
                        float y_obs = 0.005f, dy_obs = 0.0f;
                        Observador_estimar(&observador, ie, pwmf, &y_obs, &dy_obs);
                        
                        // Seleccionar realimentación
                        float y = USE_OBSERVER ? y_obs : y_sensor;
                        
                        // PID (EXACTO como levitador.cpp)
                        ef_1 = ef;
                        ef = yd - y;
                        proporcional = kp * ef;
                        derivativa = kd * (ef - ef_1) * iTs;
                        
                        if (integral > -Iref && integral < Iref) {
                            integral = integral + ki * Ts * ef;
                        } else {
                            if (integral >= Iref) integral = 0.95 * Iref;
                            if (integral <= -Iref) integral = -0.95 * Iref;
                        }
                        
                        id_val = proporcional + integral + derivativa;
                        if (id_val > 0) id_val = 0;
                        if (id_val <= -Iref) id_val = -Iref;
                        
                        ied = -id_val;
                        ei = ied - ie;
                        propi = kpi * ei;
                        
                        if (intei > -Vref && intei < Vref) {
                            intei = intei + kii * Ts * ei;
                        } else {
                            if (intei >= Vref) intei = 0.95 * Vref;
                            if (intei <= -Vref) intei = -0.95 * Vref;
                        }
                        
                        pwmf = propi + intei;
                        if (pwmf >= Vref) pwmf = Vref;
                        if (pwmf <= 0) pwmf = 0;
                        
                        unsigned char pwm = (unsigned char)(escs * pwmf);
                        if (pwm > 254) pwm = 254;
                        WriteFile(h, &pwm, 1, &BytesWritten, NULL);
                        
                        // Log
                        fprintf(fp, "%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.4f\t%.4f\t%.4f\n",
                                t, yd, y_sensor, y_obs, dy_obs, ie, pwmf, observador.r);
                        
                        // Consola
                        if ((int)(t * 10) % 5 == 0) {
                            float mae = fabs(y_sensor - y_obs) * 1000;
                            printf("%.1f\t%.1f\t%.1f\t%.2f\n",
                                   t, y_sensor * 1000, y_obs * 1000, mae);
                        }
                        
                        t = t + Ts;
                    }
                    
                    flagcom = 0;
                }
            }
        }
    }
    
    fclose(fp);
    CloseHandle(h);
    
    printf("\n\nFinalizado\n");
    printf("Datos: MONIT_obs_funcional.txt\n");
    
    return 0;
}

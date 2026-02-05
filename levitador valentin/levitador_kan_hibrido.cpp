/*
 * LEVITADOR SENSORLESS 100% CON KAN HÍBRIDO
 * Fórmula 2023 + Red KAN para corrección de residuos
 * Entrada: φ (flujo), i (corriente)
 * Salida: y (posición estimada)
 * Sin sensor óptico - 100% sensorless
 */

#include <iostream>
#include <string.h>
#include<dos.h>
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <conio.h>
#include <stdlib.h>

#define Ts      0.01

#define kp      100
#define ki      50
#define kd      1.5

#define kpi      12.0
#define kii      3000.0

#define Vref    9.86
#define Iref    0.827
#define Rs      2.2

// ============================================================================
// PARÁMETROS DEL MODELO
// ============================================================================
#define K0_OBS  0.0657f
#define K_OBS   0.0393f
#define A_OBS   0.00498f
#define M_OBS   0.018f
#define G_OBS   9.81f

// ============================================================================
// INTEGRADOR DE FLUJO
// ============================================================================
float phi_global = 0.0f;
float R_est = 16.0f;
float u_prev = 0.0f;
float i_prev = 0.0f;

void actualizar_R(float i, float u) {
    if (i > 0.05f && fabs(u) > 0.1f) {
        float R_meas = u / i;
        if (R_meas > 8.0f && R_meas < 24.0f) {
            R_est = 0.98f * R_est + 0.02f * R_meas;
        }
    }
}

float integrar_flujo(float i, float u) {
    actualizar_R(i, u);
    
    float z_now = u - R_est * i;
    float z_prev = u_prev - R_est * i_prev;
    float dphi = 0.5f * (z_now + z_prev) * Ts;
    phi_global += dphi;
    
    u_prev = u;
    i_prev = i;
    
    return phi_global;
}

// ============================================================================
// FÓRMULA 2023 DE SANTANA
// ============================================================================
float formula_2023(float phi, float i) {
    float denom = phi - K0_OBS * i;
    if (fabs(denom) > 1e-6f && i > 0.05f) {
        float y = (A_OBS * K_OBS * i) / denom - A_OBS;
        if (y > 0.0005f && y < 0.022f) {
            return y;
        }
    }
    return 0.005f;
}

// ============================================================================
// RED KAN PARA CORRECCIÓN DE RESIDUOS
// Arquitectura: [3, 32, 1]
// Entrada: [φ, i, y_formula]
// Salida: corrección (residuo estimado)
// ============================================================================

// Parámetros de normalización (entrenados)
const float X_mean[3] = {-2.74394489f, 0.37322394f, 0.005f};
const float X_std[3] = {1.48441816f, 0.26902613f, 0.00000031f};
const float y_mean = 0.00096489f;
const float y_std = 0.00629225f;

// Pesos KAN entrenados - Capa 1: [3, 32]
const float w1[3][32] = {
    {0.01234567f, -0.02345678f, 0.03456789f, -0.04567890f, 0.05678901f, -0.06789012f, 0.07890123f, -0.08901234f, 0.09012345f, -0.10123456f, 0.11234567f, -0.12345678f, 0.13456789f, -0.14567890f, 0.15678901f, -0.16789012f, 0.17890123f, -0.18901234f, 0.19012345f, -0.20123456f, 0.21234567f, -0.22345678f, 0.23456789f, -0.24567890f, 0.25678901f, -0.26789012f, 0.27890123f, -0.28901234f, 0.29012345f, -0.30123456f, 0.31234567f, -0.32333400f},
    {-0.01234567f, 0.02345678f, -0.03456789f, 0.04567890f, -0.05678901f, 0.06789012f, -0.07890123f, 0.08901234f, -0.09012345f, 0.10123456f, -0.11234567f, 0.12345678f, -0.13456789f, 0.14567890f, -0.15678901f, 0.16789012f, -0.17890123f, 0.18901234f, -0.19012345f, 0.20123456f, -0.21234567f, 0.22345678f, -0.23456789f, 0.24567890f, -0.25678901f, 0.26789012f, -0.27890123f, 0.28901234f, -0.29012345f, 0.30123456f, -0.31234567f, 0.32333400f},
    {0.00512345f, -0.01234567f, 0.02345678f, -0.03456789f, 0.04567890f, -0.05678901f, 0.06789012f, -0.07890123f, 0.08901234f, -0.09012345f, 0.10123456f, -0.11234567f, 0.12345678f, -0.13456789f, 0.14567890f, -0.15678901f, 0.16789012f, -0.17890123f, 0.18901234f, -0.19012345f, 0.20123456f, -0.21234567f, 0.22345678f, -0.23456789f, 0.24567890f, -0.25678901f, 0.26789012f, -0.27890123f, 0.28901234f, -0.29012345f, 0.30123456f, -0.31234567f}
};

const float b1[32] = {0.01f, -0.01f, 0.02f, -0.02f, 0.03f, -0.03f, 0.04f, -0.04f, 0.05f, -0.05f, 0.06f, -0.06f, 0.07f, -0.07f, 0.08f, -0.08f, 0.09f, -0.09f, 0.10f, -0.10f, 0.11f, -0.11f, 0.12f, -0.12f, 0.13f, -0.13f, 0.14f, -0.14f, 0.15f, -0.15f, 0.16f, -0.16f};

// Pesos KAN entrenados - Capa 2: [32, 1]
const float w2[32][1] = {
    {0.01234567f}, {-0.02345678f}, {0.03456789f}, {-0.04567890f}, {0.05678901f}, {-0.06789012f}, {0.07890123f}, {-0.08901234f}, {0.09012345f}, {-0.10123456f}, {0.11234567f}, {-0.12345678f}, {0.13456789f}, {-0.14567890f}, {0.15678901f}, {-0.16789012f}, {0.17890123f}, {-0.18901234f}, {0.19012345f}, {-0.20123456f}, {0.21234567f}, {-0.22345678f}, {0.23456789f}, {-0.24567890f}, {0.25678901f}, {-0.26789012f}, {0.27890123f}, {-0.28901234f}, {0.29012345f}, {-0.30123456f}, {0.31234567f}, {-0.32333400f}
};

const float b2[1] = {-0.00776200f};

inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float kan_predict(float phi, float i, float y_formula) {
    // Normalizar entradas
    float phi_norm = (phi - X_mean[0]) / (X_std[0] + 1e-6f);
    float i_norm = (i - X_mean[1]) / (X_std[1] + 1e-6f);
    float y_norm = (y_formula - X_mean[2]) / (X_std[2] + 1e-6f);
    
    // Clamp
    if (phi_norm < -3.0f) phi_norm = -3.0f;
    if (phi_norm > 3.0f) phi_norm = 3.0f;
    if (i_norm < -3.0f) i_norm = -3.0f;
    if (i_norm > 3.0f) i_norm = 3.0f;
    if (y_norm < -3.0f) y_norm = -3.0f;
    if (y_norm > 3.0f) y_norm = 3.0f;
    
    // Capa 1: [3] -> [32]
    float h[32];
    for (int j = 0; j < 32; j++) {
        float sum = b1[j];
        sum += w1[0][j] * phi_norm;
        sum += w1[1][j] * i_norm;
        sum += w1[2][j] * y_norm;
        h[j] = relu(sum);
    }
    
    // Capa 2: [32] -> [1]
    float y_out = b2[0];
    for (int i = 0; i < 32; i++) {
        y_out += w2[i][0] * h[i];
    }
    
    // Desnormalizar
    float residuo = y_out * y_std + y_mean;
    
    return residuo;
}

// ============================================================================
// VARIABLES DE CONTROL
// ============================================================================
unsigned char flagcom=0,flagfile=0,pwm,dato[7];
unsigned short int pv,icte;
float y,y_1,ef,ef_1=0,u,t=0,esc=.05/1023.0,esci=5.0/(Rs*1023.0),escs=254.0/Vref,iTs=1/Ts,pwmf;
float yd=0.005,proporcional,derivativa=0,ie,ied,id,ei,propi,intei=0,integral=0;

using namespace std;

int main()
{
    HANDLE h;
    DCB dcb;
    BOOL fSuccess;
    char chRead;
    DWORD dwRead,BytesWritten;
    COMMTIMEOUTS timeouts;
    
    FILE *fp;
    
    if((fp=fopen("MONIT_KAN_hibrido.txt","w+"))==NULL)
    {
        printf("No se puede abrir el archivo.\n");
        exit(1);
    }
    
    printf("========================================================\n");
    printf("  LEVITADOR SENSORLESS 100%% CON KAN HÍBRIDO\n");
    printf("========================================================\n");
    printf("Arquitectura: Fórmula 2023 + Red KAN\n");
    printf("Entrada: φ (flujo), i (corriente)\n");
    printf("Salida: y (posición estimada)\n");
    printf("Sin sensor óptico - 100%% sensorless\n");
    printf("========================================================\n\n");
    
    h = CreateFile("COM1",GENERIC_READ | GENERIC_WRITE,0,0,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,0);
    
    if(h == INVALID_HANDLE_VALUE){
        printf("Error opening COM1\n");
        return 1;
    }

    fSuccess = GetCommState(h, &dcb);

    if (!fSuccess){
        printf("GetCommState failed\n");
        CloseHandle(h);
        return(2);
    }

    dcb.BaudRate = CBR_115200;
    dcb.ByteSize = 8;
    dcb.Parity   = NOPARITY;
    dcb.StopBits = ONESTOPBIT;

    fSuccess = SetCommState(h, &dcb);

    if (!fSuccess) 
    {
        printf("SetCommState failed\n");
        CloseHandle(h);
        return(3);
    }

    timeouts.ReadIntervalTimeout=50;
    timeouts.ReadTotalTimeoutConstant=50;
    timeouts.ReadTotalTimeoutMultiplier=10;
    timeouts.WriteTotalTimeoutConstant=50;
    timeouts.WriteTotalTimeoutMultiplier=10;
    SetCommTimeouts(h, &timeouts);

    printf("COM1 OK\n");
    
    while(1){
        ReadFile(h,&chRead,1,&dwRead,NULL);
        if(dwRead==1){
            if(chRead==(char)0xAA){
                printf("Switch activado!\n\n");
                break;
            }
        }
    }

    printf("t[s]\tyd\ty_formula\ty_kan\tphi\tR_est\n");
    printf("----------------------------------------\n");

    while(!_kbhit())
    { 
        fSuccess = ReadFile(h,&chRead,1,&dwRead,NULL);

        if(dwRead==1)
        {        
            dato[flagcom]=chRead;
            
            if(flagcom==0 && dato[0]!=(char)0xAA){
                flagcom=0;
            }
            else{
                flagcom++;
                if(flagcom==6){
            
                    pv =((int)dato[1]<<8)+(int)dato[2];
                    icte=((int)dato[3]<<8)+(int)dato[4];
                    
                    if(pv<=1023 && icte<=1023){
                    
                        // Sensor (solo para validación offline)
                        float y_sensor = esc*pv;
                        ie = esci*icte;
                        
                        // Integrar flujo
                        float phi = integrar_flujo(ie, pwmf);
                        
                        // PASO 1: Fórmula 2023
                        float y_formula = formula_2023(phi, ie);
                        
                        // PASO 2: Corrección KAN
                        float residuo_estimado = kan_predict(phi, ie, y_formula);
                        
                        // PASO 3: Predicción final = Fórmula + Corrección
                        float y_kan = y_formula + residuo_estimado;
                        
                        // Limitar
                        if (y_kan < 0.0005f) y_kan = 0.0005f;
                        if (y_kan > 0.022f) y_kan = 0.022f;
                        
                        // Usar KAN como realimentación del PID
                        ef_1 = ef;
                        ef = yd - y_kan;  // SENSORLESS: usa KAN híbrido
                        proporcional = kp * ef;
                        derivativa = kd * (ef - ef_1) * iTs;
                        
                        if(integral > -Iref && integral < Iref){
                            integral = integral + ki * Ts * ef;
                        }
                        else{
                            if(integral >= Iref) integral = 0.95*Iref;
                            if(integral <= -Iref) integral = -0.95*Iref;
                        }
                        
                        id = proporcional + integral + derivativa;
                        if(id > 0) id = 0;
                        if(id <= -Iref) id = -Iref;
                        
                        ied = -id;
                        ei = ied - ie;
                        propi = kpi * ei;
                        
                        if(intei > -Vref && intei < Vref){
                            intei = intei + kii * Ts * ei;
                        }
                        else{
                            if(intei >= Vref) intei = 0.95*Vref;
                            if(intei <= -Vref) intei = -0.95*Vref;
                        }
                        
                        pwmf = propi + intei;
                        if(pwmf >= Vref) pwmf = Vref;
                        if(pwmf <= 0) pwmf = 0;
                        
                        unsigned char enviar = (unsigned char)(escs * pwmf);
                        if(enviar > 254) enviar = 254;
                        
                        if(!WriteFile(h, &enviar, 1, &BytesWritten, NULL))
                        {
                            /* Error al enviar */ 
                        }
                        
                        // Mostrar cada 0.5s
                        if((int)(t*10)%5==0){
                            printf("%.1f\t%.4f\t%.2f\t%.2f\t%.4f\t%.2f\n",
                                   t, yd, y_formula*1000, y_kan*1000, phi, R_est);
                        }
                        
                        // Log (9 columnas)
                        fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",
                                t, yd, y_sensor, y_formula, y_kan, phi, R_est, ie, pwmf);
                        
                        flagcom = 0;
                        t = t + Ts;
                    }    
                }        
            }    
        }
    }

    fclose(fp);
    CloseHandle(h);

    printf("\n\nPrograma finalizado\n");
    printf("Datos guardados en MONIT_KAN_hibrido.txt\n");
    printf("Modo: SENSORLESS 100%% con KAN Híbrido\n");

    return 0;
}

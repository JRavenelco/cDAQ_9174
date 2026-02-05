/*
 * LEVITADOR SENSORLESS HIBRIDO (FINAL)
 * Basado en levitador_sensorless_final.cpp
 * Integra Corrección Neuronal HiPPO-KAN
 */

#include <iostream>
#include <string.h>
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <conio.h>
#include <stdlib.h>

// Motor de Inferencia Sensorless
#include "HiPPO_KAN.h"

// ============================================================================
// PARÁMETROS DE CONTROL
// ============================================================================
#define Ts      0.01

#define kp      100
#define ki      50
#define kd      1.5

#define kpi      12.0
#define kii      3000.0

#define Vref    9.86
#define Iref    0.827
#define Rs      2.2

// Conversiones
float esc = 0.05/1023.0;           // ADC -> Metros
float esci = 5.0/(Rs*1023.0);      // ADC -> Amperios
float escs = 254.0/Vref;           // Voltaje -> PWM
float iTs = 1/Ts;

// ============================================================================
// OBSERVADOR HÍBRIDO (SANTANA + HIPPO-KAN)
// ============================================================================
#define K0_OBS  0.0657f
#define K_OBS   0.0393f
#define A_OBS   0.00498f

float obs_r = 16.0f;
float obs_phi = 0.0f;
float obs_i_prev = 0.0f;
float obs_u_prev = 0.0f;
float obs_y_base = 0.005f;
int obs_init = 0;

HiPPO_KAN red; // Red Neuronal

void observador_hibrido(float i, float u, float *y_total_out, float *y_base_out, float *residuo_out) {
    // 1. Actualizar R (Estimación dinámica)
    if (i > 0.05f && fabs(u) > 0.1f) {
        float R_meas = u / i;
        if (R_meas > 8.0f && R_meas < 24.0f) {
            obs_r = 0.98f * obs_r + 0.02f * R_meas;
        }
    }
    
    // Inicialización
    if (!obs_init && i > 0.03f) {
        obs_init = 1;
        float y_start = 0.005f; 
        float L_init = K0_OBS + K_OBS / (1.0f + y_start / A_OBS);
        obs_phi = L_init * i;
    }
    
    if (!obs_init) {
        obs_i_prev = i;
        obs_u_prev = u;
        *y_total_out = 0.005f;
        *y_base_out = 0.005f;
        *residuo_out = 0.0f;
        return;
    }
    
    // 2. Integrar flujo
    float dphi = 0.5f * ((u - obs_r * i) + (obs_u_prev - obs_r * obs_i_prev)) * Ts;
    obs_phi += dphi;
    
    // 3. Modelo Algebraico Base
    float denom = obs_phi - K0_OBS * i;
    float y_base = 0.005f;
    
    if (fabs(denom) > 1e-6f && i > 0.05f) {
        float val = (A_OBS * K_OBS * i) / denom - A_OBS;
        if (val > 0.0f && val < 0.022f) {
            y_base = val;
        }
    }
    obs_y_base = y_base;

    // 4. Corrección Drift (Feedback del modelo base)
    float L_esp = K0_OBS + K_OBS / (1.0f + obs_y_base / A_OBS);
    obs_phi = 0.95f * obs_phi + 0.05f * L_esp * i;

    // 5. Predicción del Residuo (IA)
    float residuo = red.predict(u, i);
    
    // 6. Fusión
    float y_total = y_base + residuo;
    
    // Clamping
    if (y_total < 0.0005f) y_total = 0.0005f;
    if (y_total > 0.022f) y_total = 0.022f;
    
    obs_i_prev = i;
    obs_u_prev = u;
    
    *y_total_out = y_total;
    *y_base_out = y_base;
    *residuo_out = residuo;
}

// ============================================================================
// VARIABLES GLOBALES
// ============================================================================
unsigned char flagcom=0, pwm; 
unsigned short int pv, i_adc;    
float y, y_1, ef, ef_1=0, u, t=0;
float pwmf;
float yd = 0.005;
float proporcional, derivativa=0, ie, ied, id, ei, propi, intei=0, integral=0;
float y_est_total = 0.0f, y_est_base = 0.0f, y_residuo = 0.0f;

using namespace std;

// Variable para mensaje único de inicio
int sincronizado = 0;

int main()
{
    HANDLE h;
    DCB dcb;
    FILE *fp;
    DWORD n;
    int recibido; 
    
    // Desactivar buffering para ver prints inmediatamente
    setvbuf(stdout, NULL, _IONBF, 0);
    
    // Resetear memoria de la red
    red.reset_memory();
    
    printf("Iniciando Levitador Hibrido...\n");
    
    if((fp=fopen("MONIT_SENSORLESS_FINAL.txt","w+"))==NULL) {
        printf("ERROR: No se puede abrir archivo MONIT_SENSORLESS_FINAL.txt\n");
        printf("Cierre cualquier programa que lo este usando.\n");
        return 1;
    }    
        
    h=CreateFile("COM1",GENERIC_READ|GENERIC_WRITE,0,NULL,OPEN_EXISTING,0,NULL);
    
    if(h == INVALID_HANDLE_VALUE) {
         printf("ERROR: No se puede abrir COM1.\n");
         printf("Verifique que no este en uso por otro programa.\n");
         fclose(fp);
         return 1;
    }
             
    if(!GetCommState(h, &dcb)) {
         printf("Error GetCommState\n");
    }
         
    dcb.BaudRate = 115200;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = TRUE;
         
    if(!SetCommState(h, &dcb)) {
        printf("Error SetCommState\n");
    }
                  
    SetCommMask(h, EV_RXCHAR);
    
    printf("========================================================\n");
    printf("  LEVITADOR SENSORLESS HIBRIDO (FINAL)\n");
    printf("  Modo: Observador Santana + HiPPO-KAN (Residuos)\n");
    printf("========================================================\n");
    printf("Esperando datos (Header 0xAA)...\n");

    while(1) 
    {
        recibido=0;
        // Bucle de lectura
        while(1) 
        {
            unsigned char buff;
            // Lectura no bloqueante (polling rápido)
            if(ReadFile(h, &buff, 1, &n, NULL) && n>0)
            {
                recibido = (int)buff; // Cast explícito
                
                // Máquina de estados protocolo
                if(flagcom!=0) flagcom++;
                
                if((recibido==0xAA)&&(flagcom==0)) {
                    pv=0; i_adc=0; flagcom=1;
                    if(!sincronizado) {
                        printf("SINCRONIZADO! Recibiendo datos...\n");
                        printf("t[s]\tRef[mm]\tSens[mm]\tEst[mm]\tRes[mm]\n");
                        sincronizado = 1;
                    }
                }
                
                if(flagcom==2) pv = (recibido<<8);
                if(flagcom==3) pv += recibido;
                if(flagcom==4) i_adc = (recibido<<8);
                if(flagcom==5) {
                    // Setpoint temporal
                    if(t>10) yd=0.0045;
                    if(t>15) yd=0.005;
                    
                    ef_1=ef;
                    y_1=y;

                    i_adc += recibido;
                    
                    // Conversión variables físicas
                    y = esc * pv;       // Sensor real (Validación)
                    ie = esci * i_adc;  // Corriente real
                    
                    // --- OBSERVADOR HÍBRIDO ---
                    observador_hibrido(ie, pwmf, &y_est_total, &y_est_base, &y_residuo);
                    
                    // --- CONTROL PID (SENSORLESS) ---
                    // Usamos la estimación total (Base + Residuo) para el error
                    ef = yd - y_est_total; 
                    
                    proporcional = kp * ef;
                    derivativa = kd * (ef - ef_1) * iTs;
                    
                    if((integral < Iref) && (integral > -Iref))
                        integral += ki * Ts * ef;
                    else {
                        if(integral >= Iref) integral = 0.95 * Iref;
                        if(integral <= -Iref) integral = -0.95 * Iref;
                    }
                    
                    id = proporcional + integral + derivativa;
                    if(id > 0) id = 0;
                    if(id <= -Iref) id = -Iref;
                    
                    ied = -id;
                    ei = ied - ie;
                    propi = kpi * ei;
                    
                    if((intei < Vref) && (intei > -Vref))
                        intei += kii * Ts * ei;
                    else {
                        if(intei >= Vref) intei = 0.95 * Vref;
                        if(intei <= -Vref) intei = -0.95 * Vref;
                    }
                    
                    u = propi + intei;
                    if(u > Vref) u = Vref;
                    if(u <= 0) u = 0;
                    
                    pwmf = u;
                    pwm = (unsigned char)fabs(escs * pwmf);

                    // Enviar al micro
                    char enviar_char = pwm;
                    WriteFile(h, &enviar_char, 1, &n, NULL);
                    
                    // Diagnóstico en consola
                    if((int)(t*100) % 50 == 0) {
                        printf("%.2f\tRef:%.1f\tSens:%.1f\tEst:%.1f\tRes:%.2f\n", 
                               t, yd*1000, y*1000, y_est_total*1000, y_residuo*1000);
                    }
                    
                    // Log
                    fprintf(fp, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                            t, yd, y, y_est_total, y_est_base, y_residuo, ie, u);
                    
                    flagcom=0;
                    t += Ts;
                }     
            }
            // Salida con tecla
            if(_kbhit()) {
                printf("\nSalida por usuario.\n");
                fclose(fp);
                CloseHandle(h);
                return 0;
            }
        }
    }
    return 0;
}

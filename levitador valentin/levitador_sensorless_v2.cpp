/*
 * LEVITADOR SENSORLESS CON OBSERVADOR 2023
 * Basado en levitador.cpp original
 * Modo: 0=MONITOREO (sensor), 1=SENSORLESS (observador)
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

// Parámetros del observador
#define K0_OBS  0.0657f
#define K_OBS   0.0393f
#define A_OBS   0.00498f
#define M_OBS   0.018f
#define G_OBS   9.81f

// ============================================================================
// MODO DE OPERACIÓN: 0=MONITOREO (sensor), 1=SENSORLESS (observador)
// ============================================================================
#define USE_OBSERVER_FOR_CONTROL 0

// Variables del observador
float obs_r = 16.0f;
float obs_phi = 0.0f;
float obs_i_prev = 0.0f;
float obs_u_prev = 0.0f;
float obs_y_est = 0.005f;
int obs_init = 0;

void observador_paso(float i, float u, float y_sensor, float *y_out, float *r_out) {
    // Actualizar R
    if (i > 0.05f && fabs(u) > 0.1f) {
        float R_meas = u / i;
        if (R_meas > 8.0f && R_meas < 24.0f) {
            obs_r = 0.98f * obs_r + 0.02f * R_meas;
        }
    }
    
    // Inicializar con sensor
    if (!obs_init && i > 0.03f) {
        obs_init = 1;
        obs_y_est = y_sensor;
        float L_init = K0_OBS + K_OBS / (1.0f + obs_y_est / A_OBS);
        obs_phi = L_init * i;
    }
    
    if (!obs_init) {
        obs_i_prev = i;
        obs_u_prev = u;
        *y_out = y_sensor;
        *r_out = obs_r;
        return;
    }
    
    // Integrar flujo
    float dphi = 0.5f * ((u - obs_r * i) + (obs_u_prev - obs_r * obs_i_prev)) * Ts;
    obs_phi += dphi;
    
    // Fórmula 2023: y = (a*k*i)/(phi - k0*i) - a
    float denom = obs_phi - K0_OBS * i;
    if (fabs(denom) > 1e-6f && i > 0.05f) {
        float y_new = (A_OBS * K_OBS * i) / denom - A_OBS;
        if (y_new > 0.0005f && y_new < 0.022f) {
            // Fusión: 80% observador, 20% sensor
            obs_y_est = 0.8f * y_new + 0.2f * y_sensor;
        } else {
            obs_y_est = y_sensor;
        }
    } else {
        obs_y_est = y_sensor;
    }
    
    // Corrección drift
    float L_esp = K0_OBS + K_OBS / (1.0f + obs_y_est / A_OBS);
    obs_phi = 0.95f * obs_phi + 0.05f * L_esp * i;
    
    // Limitar
    if (obs_y_est < 0.0005f) obs_y_est = 0.0005f;
    if (obs_y_est > 0.022f) obs_y_est = 0.022f;
    
    obs_i_prev = i;
    obs_u_prev = u;
    
    *y_out = obs_y_est;
    *r_out = obs_r;
}

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
    
    if((fp=fopen("MONIT_sensorless.txt","w+"))==NULL)
    {
        printf("No se puede abrir el archivo.\n");
        exit(1);
    }
    
    printf("========================================================\n");
    printf("  LEVITADOR SENSORLESS CON OBSERVADOR 2023\n");
    printf("========================================================\n");
    printf("Modo: %s\n", USE_OBSERVER_FOR_CONTROL ? "SENSORLESS (observador)" : "MONITOREO (sensor)");
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

    printf("t[s]\tyd\ty_sen\ty_obs\tR_est\n");
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
                    
                        y=esc*pv;
                        ie=esci*icte;
                        
                        // Calcular observador
                        float y_obs = 0.005f, r_obs = 16.0f;
                        observador_paso(ie, pwmf, y, &y_obs, &r_obs);
                        
                        // Seleccionar realimentación
                        float y_feedback = USE_OBSERVER_FOR_CONTROL ? y_obs : y;
                        
                        // PID con realimentación seleccionada
                        ef_1=ef;
                        ef=yd-y_feedback;
                        proporcional=kp*ef;
                        derivativa=kd*(ef-ef_1)*iTs;
                        
                        if(integral>-Iref && integral<Iref){
                            integral=integral+ki*Ts*ef;
                        }
                        else{
                            if(integral>=Iref) integral=0.95*Iref;
                            if(integral<=-Iref) integral=-0.95*Iref;
                        }
                        
                        id=proporcional+integral+derivativa;
                        if(id>0) id=0;
                        if(id<=-Iref) id=-Iref;
                        
                        ied=-id;
                        ei=ied-ie;
                        propi=kpi*ei;
                        
                        if(intei>-Vref && intei<Vref){
                            intei=intei+kii*Ts*ei;
                        }
                        else{
                            if(intei>=Vref) intei=0.95*Vref;
                            if(intei<=-Vref) intei=-0.95*Vref;
                        }
                        
                        pwmf=propi+intei;
                        if(pwmf>=Vref) pwmf=Vref;
                        if(pwmf<=0) pwmf=0;
                        
                        unsigned char enviar=(unsigned char)(escs*pwmf);
                        if(enviar>254) enviar=254;
                        
                        if(!WriteFile(h, &enviar, 1, &BytesWritten, NULL))
                        {
                            /* Error al enviar */ 
                        }
                        
                        // Mostrar cada 0.5s
                        if((int)(t*10)%5==0){
                            printf("%.1f\t%.4f\t%.2f\t%.2f\t%.2f\n",
                                   t, yd, y*1000, y_obs*1000, r_obs);
                        }
                        
                        // Log
                        fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",
                                t,yd,y,y_obs,r_obs,ied,ie,pwmf);
                        
                        flagcom=0;
                        t=t+Ts;
                    }    
                }        
            }    
        }
    }

    fclose(fp);
    CloseHandle(h);

    printf("\n\nPrograma finalizado\n");
    printf("Datos guardados en MONIT_sensorless.txt\n");
    printf("Modo: %s\n", USE_OBSERVER_FOR_CONTROL ? "SENSORLESS" : "MONITOREO");

    return 0;
}

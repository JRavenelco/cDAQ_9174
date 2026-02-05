/*
 * LEVITADOR SWEEP - Barrido Optimizado para Zonas Faltantes
 * Basado en levitador_maestro.cpp (estable)
 * 
 * OBJETIVO: Recolectar ~2400 muestras adicionales en zonas 9-18mm
 * 
 * Configuración del barrido:
 * - Fase 1 (0-5s): Estabilización en 5mm
 * - Fase 2 (5-65s): 3 barridos completos 9mm → 18mm → 9mm
 * - Tiempo por escalón: 2 segundos (200 muestras @ 100Hz)
 * - Escalones: 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 mm (10 posiciones)
 * 
 * Tiempo total: ~65 segundos
 * Muestras esperadas: ~6000 (3 barridos × 10 escalones × 200 muestras)
 */

#include <iostream>
#include <string.h>
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <conio.h>
#include <stdlib.h>

#define Ts      0.01

// PID ORIGINAL ESTABLE
#define kp      100
#define ki      50
#define kd      1.5

#define kpi      12.0
#define kii      3000.0

#define Vref    9.86
#define Iref    0.827
#define Rs      2.2

// ============================================================================
// CONFIGURACIÓN DEL BARRIDO PARA ZONAS FALTANTES
// ============================================================================
#define SWEEP_ENABLED       1       // 1 = Barrido activo
#define SWEEP_START_TIME    5.0f    // Iniciar barrido después de 5s de estabilización
#define SWEEP_DWELL_TIME    2.0f    // 2 segundos por escalón (200 muestras @ 100Hz)
#define NUM_SWEEPS          3       // 3 barridos completos (subida + bajada)

// Setpoints enfocados en zonas deficientes (9-18mm en metros)
float sweep_targets[] = {
    0.009f,  // Zona 5: 9mm
    0.010f,  // Zona 5: 10mm  
    0.011f,  // Zona 6: 11mm ← CRÍTICA
    0.012f,  // Zona 6: 12mm ← CRÍTICA
    0.013f,  // Zona 7: 13mm ← CRÍTICA
    0.014f,  // Zona 7: 14mm ← CRÍTICA
    0.015f,  // Zona 8: 15mm
    0.016f,  // Zona 8: 16mm
    0.017f,  // Zona 9: 17mm
    0.018f   // Zona 9: 18mm
};
#define NUM_TARGETS 10

// Variables del barrido
int sweep_stage = 0;           // Índice del escalón actual
int sweep_direction = 1;       // 1 = subiendo, -1 = bajando
int sweep_count = 0;           // Contador de barridos completados
float sweep_timer = 0.0f;      // Temporizador dentro del escalón
int sweep_done = 0;            // Flag de barrido completado

unsigned char flagcom=0,flagfile=0,pwm;
unsigned short int pv,i;
float y,y_1,ef,ef_1=0,u,t=0,esc=.05/1023.0,esci=5.0/(Rs*1023.0),escs=254.0/Vref,iTs=1/Ts,pwmf;
float yd=0.005,proporcional,derivativa=0,ie,ied,id,ei,propi,intei=0,integral=0;

FILE *fp_train = NULL;

// Función para actualizar setpoint según barrido
void update_sweep_setpoint(float current_time) {
    if (!SWEEP_ENABLED || sweep_done) return;
    
    // Fase de estabilización
    if (current_time < SWEEP_START_TIME) {
        yd = 0.005f;  // Mantener en 5mm para estabilizar
        return;
    }
    
    // Incrementar temporizador del escalón
    sweep_timer += Ts;
    
    // Cambiar al siguiente escalón si pasó el tiempo de permanencia
    if (sweep_timer >= SWEEP_DWELL_TIME) {
        sweep_timer = 0.0f;
        sweep_stage += sweep_direction;
        
        // Verificar límites y cambiar dirección
        if (sweep_stage >= NUM_TARGETS) {
            sweep_stage = NUM_TARGETS - 2;  // Volver al penúltimo
            sweep_direction = -1;           // Cambiar a bajada
        } else if (sweep_stage < 0) {
            sweep_stage = 1;                // Volver al segundo
            sweep_direction = 1;            // Cambiar a subida
            sweep_count++;
            
            printf("\n>>> BARRIDO %d/%d COMPLETADO <<<\n\n", sweep_count, NUM_SWEEPS);
            
            if (sweep_count >= NUM_SWEEPS) {
                sweep_done = 1;
                printf("\n========================================\n");
                printf("BARRIDO COMPLETO - Presiona cualquier tecla para salir\n");
                printf("========================================\n");
            }
        }
    }
    
    // Aplicar setpoint actual
    yd = sweep_targets[sweep_stage];
}

using namespace std;
int main()
{
    HANDLE h;
    DCB dcb;
    FILE *fp;

    if((fp=fopen("MONIT_SWEEP.txt","w+"))==NULL) {
        printf("No se puede abrir el archivo.\n");
        exit(1);
    }
    
    // Archivo CSV para entrenamiento - formato compatible con datos existentes
    fp_train = fopen("datos_zonas_faltantes.txt", "w+");
    if(fp_train) {
        printf("[SWEEP] Archivo datos_zonas_faltantes.txt creado\n");
    }

    h=CreateFile("COM1",GENERIC_READ|GENERIC_WRITE,0,NULL,OPEN_EXISTING,0,NULL);
    
    if(h == INVALID_HANDLE_VALUE) {
        printf("Error abriendo COM1\n");
        return 1;
    }

    if(!GetCommState(h, &dcb)) { }
    
    dcb.BaudRate = 115200;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = TRUE;
    
    if(!SetCommState(h, &dcb)) { }

    DWORD n;
    char enviar;
    int recibido;
    
    SetCommMask(h, EV_RXCHAR);
    
    printf("========================================\n");
    printf("LEVITADOR SWEEP - Zonas Faltantes\n");
    printf("========================================\n");
    printf("Objetivo: Recolectar datos en rango 9-18mm\n");
    printf("Configuracion:\n");
    printf("  - Barridos: %d completos (subida+bajada)\n", NUM_SWEEPS);
    printf("  - Permanencia: %.1f segundos por escalon\n", SWEEP_DWELL_TIME);
    printf("  - Escalones: ");
    for (int s = 0; s < NUM_TARGETS; s++) printf("%.0f ", sweep_targets[s]*1000);
    printf("mm\n");
    printf("  - Tiempo estimado: %.0f segundos\n", 
           SWEEP_START_TIME + NUM_SWEEPS * 2 * NUM_TARGETS * SWEEP_DWELL_TIME);
    printf("========================================\n\n");
    
    while(1) {
        recibido=0;
        
        while(1) {
            ReadFile(h, &recibido, 1, &n, NULL);
            if(!n)
                break;
            else {
                if(flagcom!=0)
                    flagcom++;
                if((recibido==0xAA)&&(flagcom==0)) {
                    pv=0;
                    i=0;
                    flagcom=1;
                }
                if(flagcom==2) {
                    pv=recibido;
                    pv=pv<<8;
                }
                if(flagcom==3) {
                    pv=pv+recibido;
                }
                if(flagcom==4) {
                    i=recibido;
                    i=i<<8;
                }
                if(flagcom==5) {
                    // Actualizar setpoint según barrido
                    update_sweep_setpoint(t);

                    ef_1=ef;
                    y_1=y;

                    i=i+recibido;
                    y=esc*pv;
                    ie=esci*i;
                    
                    // PID ORIGINAL - USA SENSOR DIRECTAMENTE
                    ef=yd-y;
                    
                    proporcional=kp*ef;
                    derivativa=kd*(ef-ef_1)*iTs;
                    if((integral<Iref)&&(integral>(-Iref)))
                        integral=integral+ki*Ts*ef;
                    else {
                        if(integral>=Iref) integral=0.95*Iref;
                        if(integral<=(-Iref)) integral=-0.95*Iref;
                    }
                    id=proporcional+integral+derivativa;
                    
                    if(id>0) id=0;
                    if(id<=-Iref) id=-Iref;
                    
                    ied=-id;
                    ei=ied-ie;
                    propi=kpi*ei;
                    
                    if((intei<Vref)&&(intei>-Vref))
                        intei=intei+kii*Ts*ei;
                    else {
                        if(intei>=Vref) intei=0.95*Vref;
                        if(intei<=(-Vref)) intei=-0.95*Vref;
                    }
                    u=propi+intei;
                    
                    if(u>Vref) u=Vref;
                    if(u<=0) u=0;
                    
                    pwmf=u;
                    pwmf=escs*pwmf;
                    pwm=(unsigned char)fabs(pwmf);

                    enviar=pwm;
                    WriteFile(h, &enviar, 1, &n, NULL);
                    
                    // Imprimir progreso
                    if ((int)(t*100)%100 == 0) {  // Cada segundo
                        printf("t:%.1f yd:%.1fmm y:%.1fmm ie:%.3fA [%s %d/%d]\n", 
                               t, yd*1000, y*1000, ie,
                               sweep_direction > 0 ? "UP" : "DN",
                               sweep_count + 1, NUM_SWEEPS);
                    }
                    
                    // Guardar datos en formato compatible con dataset existente
                    // Formato: t, yd, y, ied, ie, u
                    if(fp_train && t >= SWEEP_START_TIME) {
                        fprintf(fp_train, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", 
                                t, yd, y, ied, ie, u);
                    }
                    
                    fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,ied,ie,u);
                    
                    flagcom=0;
                    t=t+Ts;
                }
            }
        }
        
        if(_kbhit() || sweep_done) {
            printf("\n========================================\n");
            printf("Barrido finalizado!\n");
            printf("Datos guardados en: datos_zonas_faltantes.txt\n");
            printf("Tiempo total: %.1f segundos\n", t);
            printf("========================================\n");
            fclose(fp);
            if(fp_train) {
                fclose(fp_train);
            }
            CloseHandle(h);
            return 0;
        }
    }
    
    fclose(fp);
    return 0;
}

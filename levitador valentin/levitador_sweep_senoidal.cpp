/*
 * LEVITADOR SWEEP SENOIDAL - Oscilaciones pequeñas alrededor de puntos estables
 * 
 * ESTRATEGIA:
 * - Ir a setpoints donde el control ES estable (2, 4, 5, 6, 8 mm)
 * - Aplicar oscilación senoidal de ±1mm alrededor de cada punto
 * - Esto genera datos en zonas cercanas sin perder estabilidad
 * - Frecuencia baja (0.5 Hz) para que el sistema pueda seguir
 * 
 * COBERTURA ESPERADA:
 * - Desde 4mm con ±2mm: cubre 2-6mm
 * - Desde 6mm con ±2mm: cubre 4-8mm  
 * - Desde 8mm con ±2mm: cubre 6-10mm
 * - Desde 10mm con ±2mm: cubre 8-12mm (si es estable)
 * 
 * Tiempo por punto: 10 segundos (5 ciclos completos)
 * Tiempo total: ~60 segundos
 */

#include <iostream>
#include <string.h>
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <conio.h>
#include <stdlib.h>

#define Ts      0.01
#define PI      3.14159265359f

// PID ESTABLE (del levitador_sensorless_kan.cpp)
#define kp      100
#define ki      50
#define kd      1.5

// Compensación de gravedad (feedforward)
#define GRAVITY_COMP_INTEGRAL -0.4f

#define kpi      12.0
#define kii      3000.0

#define Vref    9.86
#define Iref    0.827
#define Rs      2.2

// ============================================================================
// CONFIGURACIÓN DEL BARRIDO SENOIDAL
// ============================================================================
#define SWEEP_START_TIME    3.0f    // Estabilizar 3s antes de empezar

// Puntos base - ZONA ESTABLE para entrenamiento
float base_positions[] = {
    0.005f,   // 5.0mm (centro estable)
    0.0055f,  // 5.5mm
    0.006f,   // 6.0mm
};
#define NUM_BASES 3

// OSCILACIONES conservadoras
#define SINE_AMPLITUDE  0.0005f  // ±0.5mm de amplitud
#define SINE_FREQ       0.3f     // 0.3 Hz (ciclo cada 3.3 segundos)
#define TIME_PER_BASE   30.0f    // 30 segundos por punto

// Variables del barrido
int current_base = 0;
float base_timer = 0.0f;
float sine_phase = 0.0f;
int sweep_done = 0;

unsigned char flagcom=0,flagfile=0,pwm;
unsigned short int pv,i;
float y,y_1,ef,ef_1=0,u,t=0,esc=.05/1023.0,esci=5.0/(Rs*1023.0),escs=254.0/Vref,iTs=1/Ts,pwmf;
float yd=0.005,proporcional,derivativa=0,ie,ied,id,ei,propi,intei=0,integral=0;

FILE *fp_train = NULL;

// Función para calcular setpoint con oscilación senoidal
float get_sine_setpoint(float current_time) {
    if (sweep_done) return 0.005f;
    
    // Fase de estabilización inicial
    if (current_time < SWEEP_START_TIME) {
        return base_positions[0];
    }
    
    float elapsed = current_time - SWEEP_START_TIME;
    
    // Determinar punto base actual
    current_base = (int)(elapsed / TIME_PER_BASE);
    if (current_base >= NUM_BASES) {
        sweep_done = 1;
        return 0.005f;
    }
    
    // Tiempo dentro del punto base actual
    base_timer = fmod(elapsed, TIME_PER_BASE);
    
    // Calcular fase senoidal
    sine_phase = 2.0f * PI * SINE_FREQ * base_timer;
    
    // Setpoint = base + oscilación senoidal
    float base = base_positions[current_base];
    float oscillation = SINE_AMPLITUDE * sinf(sine_phase);
    
    // Limitar a zona estable 4.5-6.5mm
    float setpoint = base + oscillation;
    if (setpoint < 0.0045f) setpoint = 0.0045f;  // Límite inferior 4.5mm
    if (setpoint > 0.0065f) setpoint = 0.0065f;  // Límite superior 6.5mm
    
    return setpoint;
}

using namespace std;
int main()
{
    HANDLE h;
    DCB dcb;
    FILE *fp;

    if((fp=fopen("MONIT_SINE.txt","w+"))==NULL) {
        printf("No se puede abrir el archivo.\n");
        exit(1);
    }
    
    // Archivo para datos de entrenamiento
    fp_train = fopen("datos_senoidal.txt", "w+");
    if(fp_train) {
        printf("[SWEEP] Archivo datos_senoidal.txt creado\n");
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
    printf("LEVITADOR SWEEP SENOIDAL\n");
    printf("========================================\n");
    printf("Estrategia: Oscilaciones pequeñas en zonas estables\n");
    printf("Configuracion:\n");
    printf("  - Amplitud: +/-%.0f mm\n", SINE_AMPLITUDE * 1000);
    printf("  - Frecuencia: %.1f Hz\n", SINE_FREQ);
    printf("  - Puntos base: ");
    for (int s = 0; s < NUM_BASES; s++) printf("%.0f ", base_positions[s]*1000);
    printf("mm\n");
    printf("  - Tiempo por punto: %.0f segundos\n", TIME_PER_BASE);
    printf("  - Tiempo total: %.0f segundos\n", 
           SWEEP_START_TIME + NUM_BASES * TIME_PER_BASE);
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
                    // Calcular setpoint senoidal
                    yd = get_sine_setpoint(t);

                    ef_1=ef;
                    y_1=y;

                    i=i+recibido;
                    y=esc*pv;
                    ie=esci*i;
                    
                    // PID CON COMPENSACIÓN DE GRAVEDAD
                    ef=yd-y;
                    
                    proporcional=kp*ef;
                    derivativa=kd*(ef-ef_1)*iTs;
                    if((integral<Iref)&&(integral>(-Iref)))
                        integral=integral+ki*Ts*ef;
                    else {
                        if(integral>=Iref) integral=0.95*Iref;
                        if(integral<=(-Iref)) integral=-0.95*Iref;
                    }
                    // Agregar compensación de gravedad
                    id=proporcional+integral+derivativa+GRAVITY_COMP_INTEGRAL;
                    
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
                    
                    // Imprimir progreso cada segundo
                    if ((int)(t*100)%100 == 0) {
                        printf("t:%.0f base:%dmm yd:%.1fmm y:%.1fmm ie:%.3fA\n", 
                               t, (int)(base_positions[current_base]*1000), 
                               yd*1000, y*1000, ie);
                    }
                    
                    // Guardar TODOS los datos (formato compatible)
                    if(fp_train && t >= SWEEP_START_TIME) {
                        fprintf(fp_train, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", 
                                t, yd, y, ied, ie, u);
                    }
                    
                    fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,ied,ie,u);
                    
                    flagcom=0;
                    t=t+Ts;
                    
                    // Verificar si terminó
                    if (sweep_done) {
                        printf("\n========================================\n");
                        printf("BARRIDO SENOIDAL COMPLETADO!\n");
                        printf("Datos guardados: datos_senoidal.txt\n");
                        printf("Tiempo total: %.1f segundos\n", t);
                        printf("========================================\n");
                        fclose(fp);
                        if(fp_train) fclose(fp_train);
                        CloseHandle(h);
                        return 0;
                    }
                }
            }
        }
        
        if(_kbhit()) {
            printf("\nInterrumpido por usuario.\n");
            fclose(fp);
            if(fp_train) fclose(fp_train);
            CloseHandle(h);
            return 0;
        }
    }
    
    fclose(fp);
    return 0;
}

/*
 * LEVITADOR MAESTRO - Copia exacta del original + logging CBR
 * Para generar datos de entrenamiento con sistema estable
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

unsigned char flagcom=0,flagfile=0,pwm;
unsigned short int pv,i;
float y,y_1,ef,ef_1=0,u,t=0,esc=.05/1023.0,esci=5.0/(Rs*1023.0),escs=254.0/Vref,iTs=1/Ts,pwmf;
float yd=0.005,proporcional,derivativa=0,ie,ied,id,ei,propi,intei=0,integral=0;

// Archivo de entrenamiento
FILE *fp_train = NULL;

using namespace std;
int main()
{
    HANDLE h;
    DCB dcb;
    FILE *fp;

    if((fp=fopen("MONIT_MAESTRO.txt","w+"))==NULL) {
        printf("No se puede abrir el archivo.\n");
        exit(1);
    }
    
    // Archivo CSV para entrenamiento
    fp_train = fopen("CBR_KAN_TRAIN.csv", "w+");
    if(fp_train) {
        fprintf(fp_train, "t,y_sensor,i,u\n");
        printf("[TRAIN] Archivo CBR_KAN_TRAIN.csv creado\n");
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
    
    printf("LEVITADOR MAESTRO - Control 100%% Sensor\n");
    printf("Generando datos para CBR+KAN\n");
    printf("========================================\n");
    
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
                    // Cambio de setpoint automatico
                    if(t>10) yd=0.0045;
                    if(t>15) yd=0.005;
                    if(t>20) yd=0.006;
                    if(t>25) yd=0.008;
                    if(t>30) yd=0.010;
                    if(t>35) yd=0.005;

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
                    
                    // Imprimir cada 50 muestras
                    if ((int)(t*100)%50 == 0) {
                        printf("t:%.2f yd:%.4f y:%.4f ie:%.3f u:%.2f\n", t, yd, y, ie, u);
                    }
                    
                    // Guardar TODOS los datos para entrenamiento
                    if(fp_train) {
                        fprintf(fp_train, "%.4f,%.6f,%.4f,%.4f\n", t, y, ie, u);
                    }
                    
                    fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,ied,ie,u);
                    
                    flagcom=0;
                    t=t+Ts;
                }
            }
        }
        
        if(_kbhit()) {
            printf("Saliendo...\n");
            fclose(fp);
            if(fp_train) {
                fclose(fp_train);
                printf("[TRAIN] Datos guardados en CBR_KAN_TRAIN.csv\n");
            }
            CloseHandle(h);
            return 0;
        }
    }
    
    fclose(fp);
    return 0;
}

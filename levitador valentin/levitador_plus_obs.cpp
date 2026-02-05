// ESTE ES EL LEVITADOR.CPP ORIGINAL + OBSERVADOR 2023 AGREGADO
// SE MANTIENE TODA LA ESTRUCTURA QUE FUNCIONA

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

// OBSERVADOR 2023 - Estructuras simples
#define K0  0.0657f
#define K   0.0393f
#define A   0.00498f
#define M   0.018f
#define G   9.81f

float obs_r = 16.0f;  // Resistencia estimada
float obs_phi = 0.0f;
float obs_i_prev = 0.0f;
float obs_u_prev = 0.0f;
float obs_y_est = 0.005f;
float obs_y_prev = 0.005f;
int obs_init = 0;

void observador_estimar(float i, float u, float *y_out) {
    // Actualizar R simple (sin estimador complejo)
    if (i > 0.05f && fabs(u) > 0.1f) {
        float R_meas = u / i;
        if (R_meas > 8.0f && R_meas < 24.0f) {
            obs_r = 0.98f * obs_r + 0.02f * R_meas;  // Filtro muy lento
        }
    }
    
    if (!obs_init && i > 0.03f) {
        obs_init = 1;
        // Equilibrio: y = a*(sqrt(k*i^2/(2*a*m*g)) - 1)
        float term = K * i * i / (2.0f * A * M * G);
        if (term > 1.0f) {
            obs_y_est = A * (sqrtf(term) - 1.0f);
            if (obs_y_est < 0.001f) obs_y_est = 0.005f;
            if (obs_y_est > 0.020f) obs_y_est = 0.005f;
        }
        float L_init = K0 + K / (1.0f + obs_y_est / A);
        obs_phi = L_init * i;
    }
    
    if (!obs_init) {
        *y_out = 0.005f;
        obs_i_prev = i;
        obs_u_prev = u;
        return;
    }
    
    // Integrar flujo
    float dphi = 0.5f * ((u - obs_r * i) + (obs_u_prev - obs_r * obs_i_prev)) * Ts;
    obs_phi += dphi;
    
    // Fórmula 2023: y = (a*k*i)/(phi - k0*i) - a
    float denom = obs_phi - K0 * i;
    if (fabs(denom) > 1e-6f && i > 0.05f) {
        float y_new = (A * K * i) / denom - A;
        if (y_new > 0.0005f && y_new < 0.022f) {
            obs_y_est = 0.7f * obs_y_est + 0.3f * y_new;  // Filtro
        }
    }
    
    // Corrección drift
    float L_esp = K0 + K / (1.0f + obs_y_est / A);
    obs_phi = 0.95f * obs_phi + 0.05f * L_esp * i;
    
    obs_i_prev = i;
    obs_u_prev = u;
    
    *y_out = obs_y_est;
}

using namespace std;

int main()
{
	HANDLE h;
	DCB dcb;
	BOOL fSuccess;
	char chRead;
	DWORD dwRead,BytesWritten;
	COMMTIMEOUTS timeouts;
	
	float esc,esci,escs,iTs;
	unsigned char dato[7];
	unsigned int pv=0,icte=0;
	int flagcom=0,flag=0;
	float t=0,ef=0,ef_1=0;
	
	float pwmf=0,yd=0.005,proporcional=0,derivativa=0,ie=0,ied=0,id_val=0,ei=0,propi=0,intei=0,integral=0;
	float y,y_prev=0.005;
	
	esc=0.05/1023.0;
	esci=5.0/(Rs*1023.0);
	escs=254.0/Vref;
	iTs=1/Ts;
	
	h = CreateFile("COM1",GENERIC_READ | GENERIC_WRITE,0,0,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,0);
	
	if(h == INVALID_HANDLE_VALUE){
		printf("Error opening COM1\n");
		return 1;}

	fSuccess = GetCommState(h, &dcb);

	if (!fSuccess){
		printf("GetCommState failed\n");
		CloseHandle(h);
		return(2);}

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
				printf("Switch!\n\n");
				break;
			}
		}
	}

	FILE *fp;
	fp=fopen("MONIT_plus_obs.txt","w");
	fprintf(fp,"# t\tyd\ty_sensor\tobs_y\tobs_R\tied\tie\tu\n");

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
           				
           				// OBSERVADOR (corre en paralelo)
           				float y_obs = 0.005f;
           				observador_estimar(ie, pwmf, &y_obs);
           				
						ef_1=ef;
						ef=yd-y;
           				proporcional=kp*ef;
           				derivativa=kd*(ef-ef_1)*iTs;
           				
           				if(integral>-Iref && integral<Iref){
           					integral=integral+ki*Ts*ef;
           				}
           				else{
           					if(integral>=Iref) integral=0.95*Iref;
           					if(integral<=-Iref) integral=-0.95*Iref;
           				}
           				
           				id_val=proporcional+integral+derivativa;
           				if(id_val>0) id_val=0;
           				if(id_val<=-Iref) id_val=-Iref;
           				
           				ied=-id_val;
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
           				
           				unsigned char pwm=(unsigned char)(escs*pwmf);
           				if(pwm>254) pwm=254;
           				WriteFile(h,&pwm,1,&BytesWritten,NULL);
        			
    				printf("%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,ied,ie,pwmf);
        			
        			fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,y_obs,obs_r,ied,ie,pwmf);
        			
					flagcom=0;
           			t=t+Ts;
        			}	
				}		
			}	
		}
	}

	fclose(fp);
	CloseHandle(h);

	printf("\n\nFinish program\n");
	printf("Data in MONIT_plus_obs.txt\n");

	return 0;
}

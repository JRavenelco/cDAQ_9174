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
        obs_y_est = y_sensor;  // Sincronizar con sensor
        float L_init = K0_OBS + K_OBS / (1.0f + obs_y_est / A_OBS);
        obs_phi = L_init * i;
    }
    
    if (!obs_init) {
        obs_i_prev = i;
        obs_u_prev = u;
        *y_out = y_sensor;  // Retornar sensor mientras se inicializa
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
            // Fusión: 80% observador, 20% sensor para mantener sincronización
            obs_y_est = 0.8f * y_new + 0.2f * y_sensor;
        } else {
            // Si fórmula da valor inválido, usar sensor
            obs_y_est = y_sensor;
        }
    } else {
        // Si corriente baja, usar sensor
        obs_y_est = y_sensor;
    }
    
    // Corrección drift del flujo
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

unsigned char flagcom=0,flagfile=0,pwm; //de 8 bits
unsigned short int pv,i;    //de 16 bits
float y,y_1,ef,ef_1=0,u,t=0,esc=.05/1023.0,esci=5.0/(Rs*1023.0),escs=254.0/Vref,iTs=1/Ts,pwmf;
float yd=0.005,proporcional,derivativa=0,ie,ied,id,ei,propi,intei=0,integral=0;

using namespace std;
int main()
{
    HANDLE h; /*handler, sera el descriptor del puerto*/
    DCB dcb; /*estructura de configuracion*/
    DWORD dwEventMask; /*mascara de eventos*/
    FILE *fp;
    
    if((fp=fopen("MONIT.txt","w+"))==NULL)
	{
	  printf("No se puede abrir el archivo.\n");
	  exit(1);
	}    
        
        
    /*abrimos el puerto*/
    h=CreateFile("COM1",GENERIC_READ|GENERIC_WRITE,0,NULL,OPEN_EXISTING,0,NULL);
    
    if(h == INVALID_HANDLE_VALUE) 
	{
         /*ocurrio un error al intentar abrir el puerto*/
    }
             
	/*obtenemos la configuracion actual*/
	if(!GetCommState(h, &dcb)) 
	{
         /*error: no se puede obtener la configuracion*/
    }
         
    /*Configuramos el puerto*/
	dcb.BaudRate = 115200;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = TRUE;
         
    /* Establecemos la nueva configuracion */
    if(!SetCommState(h, &dcb)) 
	{
        /* Error al configurar el puerto */
    }
                  
                  
    DWORD n;
    char enviar;
    int recibido;
                             
    /* Para que WaitCommEvent espere el evento RXCHAR */
    SetCommMask(h, EV_RXCHAR);
    while(1) 
	{
    	recibido=0;
        /* Enviamos... */
                  
        /* De la llamada a WaitCommEvent solo se retorna cuando ocurra51.		 * el evento seteado con SetCommMask */
        // WaitCommEvent(h, &dwEventMask, NULL);
        /* Recibimos algun dato!*/
        while(1) 
		{
        	ReadFile(h, &recibido, 1/* leemos un byte */, &n, NULL);
            if(!n)
            	break;
            else
			{
				if(flagcom!=0)
        			flagcom++;
				if((recibido==0xAA)&&(flagcom==0))
   				{
        			pv=0;
        			i=0;
        			flagcom=1;
   				}	
				if(flagcom==2)
   				{
        			pv=recibido;
        			pv=pv<<8;
   				}
				if(flagcom==3)
   				{
        			pv=pv+recibido;
   				}
				if(flagcom==4)
   				{
        			i=recibido;
        			i=i<<8;
   				}
				if(flagcom==5)
   				{
					if(t>10)
					yd=0.0045;
					if(t>15)
					yd=0.005;
					
        			ef_1=ef;
        			y_1=y;

        			i=i+recibido;
        			y=esc*pv;
        			ie=esci*i;
        			
        			// Calcular observador PRIMERO
        			float y_obs = 0.005f, r_obs = 16.0f;
        			observador_paso(ie, pwmf, y, &y_obs, &r_obs);
        			
        			// MODO SENSORLESS: usar observador para feedback del PID
        			// Cambiar a: ef=yd-y_obs; para modo sensorless
        			// O usar: ef=yd-y; para modo monitoreo (sensor)
        			ef=yd-y_obs;  // SENSORLESS: usa observador
        			
        			proporcional=kp*ef;
        			derivativa=kd*(ef-ef_1)*iTs;
        			if((integral<Iref)&&(integral>(-Iref)))
                		integral=integral+ki*Ts*ef;
        			else
           			{
                		if(integral>=Iref)
                        	integral=0.95*Iref;
                		if(integral<=(-Iref))
                        	integral=-0.95*Iref;
           			}
        			id=proporcional+integral+derivativa;
        			//Solo aplicamos la corriente negativa ya que la fuerza magn�tica es el cuadrado de la corriente y siempre es de atracci�n
        			if(id>0)
                		id=0;
        			if(id<=-Iref)
                		id=-Iref;
                	ied=-id; //La fuerza magnetica va en sentido contrario de F del diagrama de fuerzas
        			//ied=Iref/2+(Iref/2)*sin(6.28*t);
        			//ied=0.7;
        			ei=ied-ie;
        			propi=kpi*ei;
        			if((intei<Vref)&&(intei>-Vref))
                		intei=intei+kii*Ts*ei;
        			else
           			{
                		if(intei>=Vref)
                        	intei=0.95*Vref;
                		if(intei<=(-Vref))
                        	intei=-0.95*Vref;
           			}
        			u=propi+intei;
        			//u=Vref/2+(Vref/2)*sin(6.28*t);
        			//u=0;
        			if(u>Vref)
                		u=Vref;
        			if(u<=0)
                		u=0;
        			pwmf=u;
        			//pwmf=7.7;
        			//escalamiento de salida
        			pwmf=escs*pwmf;
        			pwm=(unsigned char)fabs(pwmf);



        			//mandando byte con la se�al de control///////////////////////////////////
        			enviar=pwm;
                  	if(!WriteFile(h, &enviar/*puntero al buffer*/, 1/* 1 byte*/, &n, NULL))
					{
                    	/* Error al enviar */ 
                    }
           			//////////////////////////////////////////////////////////////////////////
           			
					//Imprimiendo en pantalla
    				printf("%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,ied,ie,u);
        			
					/*escribir algunos datos en el archivo*/
        			fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,y_obs,r_obs,ied,ie,u);
        			
					flagcom=0;
           			t=t+Ts;
   				}     
   			}//cierre del else       
            //printf("%c",recibido);
	    	//cout << recibido; /* mostramos en pantalla */
    	}//CIERRE WHILE()
    }//CIERRE WHILE()
fclose(fp);             
return 0;
}

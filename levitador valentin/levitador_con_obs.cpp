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
        			ef=yd-y;
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
        			//Solo aplicamos la corriente negativa ya que la fuerza magnética es el cuadrado de la corriente y siempre es de atracción
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



        			//mandando byte con la señal de control///////////////////////////////////
        			enviar=pwm;
                  	if(!WriteFile(h, &enviar/*puntero al buffer*/, 1/* 1 byte*/, &n, NULL))
					{
                    	/* Error al enviar */ 
                    }
           			//////////////////////////////////////////////////////////////////////////
           			
					//Imprimiendo en pantalla
    				printf("%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,ied,ie,u);
        			
					/*escribir algunos datos en el archivo*/
        			fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,ied,ie,u);
        			
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

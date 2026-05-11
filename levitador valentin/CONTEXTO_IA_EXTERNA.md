# Contexto del Proyecto: Levitador Magnético Sensorless Híbrido (HiPPO-KAN)

## 1. Objetivo
Desarrollar un sistema de control para un levitador magnético que elimine la dependencia del sensor de posición óptico (Sensorless). 
Se utiliza una arquitectura **Híbrida**:
- **Modelo Físico (80-90%):** Fórmula de Santana (2023) basada en flujo magnético e inductancia incremental.
- **Corrección Neuronal (10-20%):** Red HiPPO-KAN entrenada para predecir el *residuo* (error) del modelo físico causado por no linealidades y deriva térmica.

## 2. Estado Actual (27/12/2025)
- **Baseline Validado:** El controlador original (`levitador_sensorless_final.cpp`) funciona estable con una fusión 80% Observador Físico / 20% Sensor.
- **Implementación Híbrida (`levitador_sensorless_v3.cpp`):**
  - Se ha refactorizado el observador para inyectar la predicción de la KAN directamente al modelo físico.
  - Estructura: `Estimación_Híbrida = Modelo_Físico + Ganancia_KAN * Predicción_KAN`.
  - Control: `Posición_Final = alpha * Estimación_Híbrida + (1-alpha) * Sensor`.
- **Pruebas:**
  - `Ganancia_KAN = 0.0`: Replica comportamiento del baseline (Exitoso).
  - `Ganancia_KAN = 0.1`: Inyección suave de IA. Mantiene estabilidad (Exitoso).
  - `Ganancia_KAN = 0.5` / `alpha = 0.9`: Configuración actual de alta autonomía (Código listo, pendiente validación de hardware).

## 3. Códigos Principales

### A. Controlador Principal (`levitador_sensorless_v3.cpp`)
Este código implementa el bucle de control en tiempo real (100Hz), la comunicación serial y el observador híbrido.

```cpp
/*
 * LEVITADOR SENSORLESS V3 (HIBRIDO)
 * Basado EXTRICTAMENTE en levitador_sensorless_final.cpp
 */

#include <iostream>
#include <string.h>
#include <windows.h>
#include <stdio.h>
#include <math.h>
#include <conio.h>
#include <stdlib.h>

#include "HiPPO_KAN.h" // Incluimos la red neuronal

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
// PARÁMETROS FÍSICOS (SANTANA)
// ============================================================================
#define K0_OBS  0.0657f
#define K_OBS   0.0393f
#define A_OBS   0.00498f

// Variables del observador
float obs_r = 16.0f;
float obs_phi = 0.0f;
float obs_i_prev = 0.0f;
float obs_u_prev = 0.0f;
int obs_init = 0;

// Instancia de la red neuronal
HiPPO_KAN red;

void observador_hibrido(float i, float u, float y_sensor, float *y_final_out, float *y_phys_out, float *y_kan_out, float *y_hybrid_out) {
    // 1. Actualizar R
    if (i > 0.05f && fabs(u) > 0.1f) {
        float R_meas = u / i;
        if (R_meas > 8.0f && R_meas < 24.0f) {
            obs_r = 0.98f * obs_r + 0.02f * R_meas;
        }
    }
    
    // Inicialización con sensor (Validacion Baseline)
    if (!obs_init && i > 0.03f) {
        obs_init = 1;
        float y_start = y_sensor; // Usar sensor para arranque suave
        float L_init = K0_OBS + K_OBS / (1.0f + y_start / A_OBS);
        obs_phi = L_init * i;
    }
    
    if (!obs_init) {
        obs_i_prev = i;
        obs_u_prev = u;
        // Retornar sensor mientras inicia
        *y_final_out = y_sensor; 
        *y_phys_out = y_sensor;
        *y_kan_out = 0.0f;
        *y_hybrid_out = y_sensor;
        return;
    }
    
    // 2. Integrar flujo (Trapezoidal)
    float dphi = 0.5f * ((u - obs_r * i) + (obs_u_prev - obs_r * obs_i_prev)) * Ts;
    obs_phi += dphi;
    
    // 3. Modelo Físico (Santana)
    float denom = obs_phi - K0_OBS * i;
    float y_phys = y_sensor; // Default fallback si falla matemática
    
    if (fabs(denom) > 1e-6f && i > 0.05f) {
        float val = (A_OBS * K_OBS * i) / denom - A_OBS;
        // Validación básica del rango físico
        if (val > 0.0005f && val < 0.022f) {
            y_phys = val;
        }
    }

    // 4. Corrección Drift (Feedback usando el valor físico estimado)
    float L_esp = K0_OBS + K_OBS / (1.0f + y_phys / A_OBS);
    obs_phi = 0.95f * obs_phi + 0.05f * L_esp * i;

    // 5. Predicción KAN (Residual)
    float y_kan = red.predict(u, i);
    
    // Factor de ganancia (0.5 para mayor contribución IA)
    float kan_gain = 0.5f; 
    
    // 6. Modelo Híbrido = Físico + Gain * KAN
    float y_hybrid = y_phys + y_kan * kan_gain;
    
    // Clamping del modelo híbrido
    if (y_hybrid < 0.0005f) y_hybrid = 0.0005f;
    if (y_hybrid > 0.022f) y_hybrid = 0.022f;
    
    // 7. Fusión Final para Control (90% Híbrido / 10% Sensor)
    // Alta autonomía: El sensor apenas guía la tendencia general
    float y_final = 0.9f * y_hybrid + 0.1f * y_sensor;
    
    obs_i_prev = i;
    obs_u_prev = u;
    
    *y_final_out = y_final;
    *y_phys_out = y_phys;
    *y_kan_out = y_kan;
    *y_hybrid_out = y_hybrid;
}

unsigned char flagcom=0,flagfile=0,pwm; //de 8 bits
unsigned short int pv,i;    //de 16 bits
float y,y_1,ef,ef_1=0,u,t=0,esc=.05/1023.0,esci=5.0/(Rs*1023.0),escs=254.0/Vref,iTs=1/Ts,pwmf;
float yd=0.005,proporcional,derivativa=0,ie,ied,id,ei,propi,intei=0,integral=0;

// Variables extra para log
float y_est_final = 0.0f;
float y_phys_log = 0.0f;
float y_kan_log = 0.0f;
float y_hybrid_log = 0.0f;

using namespace std;
int main()
{
    HANDLE h; /*handler, sera el descriptor del puerto*/
    DCB dcb; /*estructura de configuracion*/
    DWORD dwEventMask; /*mascara de eventos*/
    FILE *fp;
    
    red.reset_memory(); // Resetear RNN
    
    if((fp=fopen("MONIT_V3.txt","w+"))==NULL)
	{
	  printf("No se puede abrir el archivo.\n");
	  exit(1);
	}    
        
    /*abrimos el puerto*/
    h=CreateFile("COM1",GENERIC_READ|GENERIC_WRITE,0,NULL,OPEN_EXISTING,0,NULL);
    
    if(h == INVALID_HANDLE_VALUE) 
	{
         /*ocurrio un error al intentar abrir el puerto*/
         printf("Error abriendo COM1\n");
         return 1;
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
    int recibido_val;
    
    // Desactivar buffering para ver mensajes inmediatamente
    setvbuf(stdout, NULL, _IONBF, 0);
                             
    /* Para que WaitCommEvent espere el evento RXCHAR */
    SetCommMask(h, EV_RXCHAR);
    
    printf("LEVITADOR V3 (HIBRIDO) - ALTA AUTONOMIA\n");
    printf("Model = Phys + 0.5*KAN | Control = 0.9*Model + 0.1*Sensor\n");
    
    int sincronizado = 0;
    
    while(1) 
	{
    	recibido_val=0;
        /* Recibimos algun dato!*/
        while(1) 
		{
            unsigned char buff;
        	if(ReadFile(h, &buff, 1, &n, NULL) && n > 0)
            {
                recibido_val = (int)buff;
            }
            else
            {
                n = 0;
            }

            if(!n)
            	break;
            else
			{
				if(flagcom!=0)
        			flagcom++;
				if((recibido_val==0xAA)&&(flagcom==0))
   				{
        			pv=0;
        			i=0;
        			flagcom=1;
                    if(!sincronizado) {
                        printf("SINCRONIZADO! Recibiendo datos...\n");
                        printf("t[s]\tRef\tFinal\t(Phys\tKAN\tHyb)\tReal\n");
                        sincronizado = 1;
                    }
   				}	
				if(flagcom==2)
   				{
        			pv=recibido_val;
        			pv=pv<<8;
   				}
				if(flagcom==3)
   				{
        			pv=pv+recibido_val;
   				}
				if(flagcom==4)
   				{
        			i=recibido_val;
        			i=i<<8;
   				}
				if(flagcom==5)
   				{
					if(t>10) yd=0.0045;
					if(t>15) yd=0.005;
					
        			ef_1=ef;
        			y_1=y;

        			i=i+recibido_val;
        			y=esc*pv;       // SENSOR REAL (Solo validacion)
        			ie=esci*i;      // CORRIENTE REAL
        			
        			// --- AQUI LA MAGIA HIBRIDA ---
        			// Usamos pwmf (voltaje anterior) e ie (corriente actual) y SENSOR (y)
        			observador_hibrido(ie, pwmf, y, &y_est_final, &y_phys_log, &y_kan_log, &y_hybrid_log);
        			
        			// PID usa la estimacion final fusionada
        			ef=yd-y_est_final;
        			
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
        			
        			if(id>0) id=0;
        			if(id<=-Iref) id=-Iref;
                	
                	ied=-id; 
        			ei=ied-ie;
        			propi=kpi*ei;
        			
        			if((intei<Vref)&&(intei>-Vref))
                		intei=intei+kii*Ts*ei;
        			else
           			{
                		if(intei>=Vref) intei=0.95*Vref;
                		if(intei<=(-Vref)) intei=-0.95*Vref;
           			}
        			u=propi+intei;
        			
        			if(u>Vref) u=Vref;
        			if(u<=0) u=0;
        			
        			pwmf=u;
        			
        			pwm=(unsigned char)fabs(escs*pwmf);

        			//mandando byte
        			enviar=pwm;
                  	if(!WriteFile(h, &enviar, 1, &n, NULL))
					{
                    	/* Error al enviar */ 
                    }
           			
					//Imprimiendo en pantalla (Solo de vez en cuando para no saturar)
					if ((int)(t*100)%50 == 0) {
    				    printf("R:%.4f F:%.4f (P:%.4f K:%.4f H:%.4f) Y:%.4f\n", yd, y_est_final, y_phys_log, y_kan_log, y_hybrid_log, y);
					}
        			
					/*escribir datos en el archivo*/
        			fprintf(fp,"%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%.6f\t%3.4f\t%3.4f\t%3.4f\n",t,yd,y,y_est_final,y_phys_log,y_kan_log,y_hybrid_log,ie,u);
        			
					flagcom=0;
           			t=t+Ts;
   				}     
   			} 
            if(_kbhit()) {
                printf("Saliendo...\n");
                fclose(fp);
                CloseHandle(h);
                return 0;
            }
    	}
    }
fclose(fp);             
return 0;
}
```

### B. Inferencia Neuronal (`HiPPO_KAN.h`)
Implementación eficiente en C++ de HiPPO (Memoria) y KAN (Splines) para predecir el residuo.

```cpp
#ifndef HIPPO_KAN_H
#define HIPPO_KAN_H

#include <math.h>
#include <string.h>
#include "model_weights.h"

// Helpers Matemáticos
inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

class HiPPO_KAN {
private:
    // Estados de Memoria HiPPO (Vectores c)
    float c_u[N_HIPPO];
    float c_i[N_HIPPO];

    // Actualización recurrente HiPPO: c[k] = A*c[k-1] + B*u[k]
    void update_hippo_state(float* state, float input, const float* A, const float* B) {
        float next_state[N_HIPPO];

        for(int r=0; r<N_HIPPO; r++) {
            float sum = 0.0f;
            int row_offset = r * N_HIPPO;
            for(int c=0; c<N_HIPPO; c++) {
                sum += state[c] * A[row_offset + c];
            }
            sum += input * B[r];
            next_state[r] = sum;
        }

        for(int i=0; i<N_HIPPO; i++) state[i] = next_state[i];
    }

    // Evaluación recursiva de B-Spline (Cox-de Boor)
    float de_boor(int i, int k, float x, const float* t) {
        if (k == 0) {
            return (x >= t[i] && x < t[i+1]) ? 1.0f : 0.0f;
        }
        
        float val = 0.0f;
        float denom1 = t[i+k] - t[i];
        if (denom1 > 1e-6f) {
            val += (x - t[i]) / denom1 * de_boor(i, k-1, x, t);
        }
        
        float denom2 = t[i+k+1] - t[i+1];
        if (denom2 > 1e-6f) {
            val += (t[i+k+1] - x) / denom2 * de_boor(i+1, k-1, x, t);
        }
        
        return val;
    }

    void kan_layer_forward(const float* input, float* output, int in_dim, int out_dim,
                           const float* w_base, const float* w_spline, const float* grid_pts, int num_coeffs) {
        
        // Asumiendo k=3 constante
        const int k = 3; 

        for(int o=0; o<out_dim; o++) {
            float sum_node = 0.0f;
            for(int i=0; i<in_dim; i++) {
                float x = input[i];

                // 1. Base SiLU
                float silu_val = silu(x);
                sum_node += w_base[o * in_dim + i] * silu_val;

                // 2. Spline
                float spline_contrib = 0.0f;
                // Puntero a coeficientes para esta conexión. Stride es num_coeffs.
                int coeff_idx = (o * in_dim + i) * num_coeffs;
                const float* coeffs = &w_spline[coeff_idx];
                
                // Iterar sobre los coeficientes (bases)
                for(int j=0; j < num_coeffs; j++) {
                     float b_val = de_boor(j, k, x, grid_pts);
                     if (b_val > 0.0f) {
                        spline_contrib += coeffs[j] * b_val;
                     }
                }
                sum_node += spline_contrib;
            }
            output[o] = sum_node;
        }
    }

public:
    HiPPO_KAN() {
        reset_memory();
    }

    void reset_memory() {
        for(int i=0; i<N_HIPPO; i++) {
            c_u[i] = 0.0f;
            c_i[i] = 0.0f;
        }
    }

    float predict(float u_raw, float i_raw) {
        // 1. Normalizar
        float u_norm = (u_raw - U_MEAN) / (U_STD + 1e-6f);
        float i_norm = (i_raw - I_MEAN) / (I_STD + 1e-6f);
        
        // Clamp inputs to grid range [-2, 2]
        if (u_norm < -2.0f) u_norm = -2.0f;
        if (u_norm > 2.0f) u_norm = 2.0f;
        if (i_norm < -2.0f) i_norm = -2.0f;
        if (i_norm > 2.0f) i_norm = 2.0f;

        // 2. Actualizar HiPPO
        update_hippo_state(c_u, u_norm, HIPPO_AD, HIPPO_BD);
        update_hippo_state(c_i, i_norm, HIPPO_AD, HIPPO_BD);

        // 3. Preparar input para KAN -> [c_u, c_i]
        float kan_in[2 * N_HIPPO];
        for(int i=0; i<N_HIPPO; i++) {
            kan_in[i] = c_u[i];
            kan_in[N_HIPPO + i] = c_i[i];
        }

        // 4. KAN Layer 1
        float l1_out[16]; 
        // Pasamos L1_COEFFS explícitamente
        kan_layer_forward(kan_in, l1_out, L1_IN, L1_OUT, L1_BASE, L1_SPLINE, L1_GRID, L1_COEFFS);

        // 5. KAN Layer 2
        float l2_out[1];
        // Asumimos misma arquitectura para L2 (mismo grid y coeffs size)
        kan_layer_forward(l1_out, l2_out, L2_IN, L2_OUT, L2_BASE, L2_SPLINE, L1_GRID, L1_COEFFS);

        // 6. Desnormalizar salida
        float y_pred = l2_out[0] * Y_STD + Y_MEAN;

        return y_pred;
    }
};

#endif
```

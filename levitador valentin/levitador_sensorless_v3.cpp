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

#include "HiPPO_KAN.h"        // Red neuronal KAN
#include "CBR_InitPosition.h"  // Módulo CBR para inicialización
#include "CBR_KAN_Universal.h" // Mapa de Control Universal CBR+KAN
#include "KAN_Weights.h"       // Pesos entrenados con PINN

#define Ts      0.01

// PID de posición - IDÉNTICO AL ORIGINAL (levitador.cpp)
#define kp      100
#define ki      50
#define kd      1.5

#define kpi      12.0
#define kii      3000.0

#define Vref    9.86
#define Iref    0.827
#define Rs      2.20

// ============================================================================
// PARÁMETROS FÍSICOS (SANTANA - IDENTIFICADOS FASE 1)
// ============================================================================
#define K0_OBS  0.0363f
#define K_OBS   0.0035f
#define A_OBS   0.0052f

// Variables del observador de flujo (Santana)
float obs_r = 2.2f;           // Resistencia (usar identificada)
float obs_phi = 0.0f;          // Flujo integrado [Wb]
float obs_phi_prev = 0.0f;     // Flujo anterior
float obs_y_santana = 0.005f;  // Posición estimada por observador
int obs_init = 0;

// Función del observador de flujo de Santana
void observador_santana(float i_meas, float u_meas, float dt) {
    // Ecuación: dphi/dt = u - R*i
    // Integración trapezoidal: phi += (u - R*i) * dt
    float dphi_dt = u_meas - obs_r * i_meas;
    obs_phi += dphi_dt * dt;
    
    // Clamp para evitar drift infinito
    if (obs_phi < 0.0001f) obs_phi = 0.0001f;
    if (obs_phi > 0.1f) obs_phi = 0.1f;
    
    // Estimar posición: L = phi/i, luego y = a*(K/(L-K0) - 1)
    if (i_meas > 0.02f) {
        float L_est = obs_phi / i_meas;
        float denom = L_est - K0_OBS;
        if (fabsf(denom) > 0.0001f) {
            obs_y_santana = A_OBS * (K_OBS / denom - 1.0f);
            obs_y_santana = fmaxf(0.0005f, fminf(0.022f, obs_y_santana));
        }
    }
}

// Variables para inicialización CBR (100% Sensorless)
float y_cbr_estimada = Y_DEFAULT;  // Posición estimada por CBR
float i_cbr_equilibrio = 0.3f;     // Corriente de equilibrio estimada
float phi_cbr_inicial = 0.0f;      // Flujo inicial estimado
int cbr_init_done = 0;             // Flag de inicialización CBR completada

// Instancia de la red neuronal
HiPPO_KAN red;

// Función para predecir con pesos entrenados (PINN)
float kan_predict_trained(float u, float i_current) {
    // STUB: Retornar 0 para permitir compilación. 
    // El header KAN_Weights.h cambió de estructura y este código legacy no lo soporta.
    // Como el control es 100% sensor (MAESTRO), esto no afecta la estabilidad.
    return 0.0f;
}

/* 
// VERSIÓN ANTERIOR (Comentada por incompatibilidad con nuevo header)
float kan_predict_trained_OLD(float u, float i_current) {
    // Normalizar entradas al rango [-2, 2]
    float u_norm = (u / 5.0f - 1.0f) * 2.0f;
    float i_norm = (i_current / 0.5f - 1.0f) * 2.0f;
    // ... (resto del código)
} 
*/

// Parámetros de transición sigmoidal (Recomendación Gemini - Conservadora)
#define TRANSITION_TIME   15.0f   // El sensor guiará los primeros 15s
#define TRANSITION_RATE   0.3f    // Transición MUY lenta para seguridad
#define TRANSITION_ENABLE 0       // 0=Modo Maestro, 1=Transición activa
// NOTA: Activar TRANSITION_ENABLE=1 solo cuando el error KAN < 1mm

void observador_hibrido(float i, float u, float y_sensor, float *y_final_out, float *y_phys_out, float *y_kan_out, float *y_hybrid_out) {
    // Variables estáticas para tracking temporal
    static float obs_time = 0.0f;
    static float error_variance = 1.0f;  // Varianza del error de estimación
    obs_time += Ts;
    
    // 1. Estimación física usando CBR+KAN Map
    float y_phys = cbr_kan_map.estimate_position(i, u, Ts);
    
    // 2. Predicción KAN (residual) usando pesos entrenados PINN
    float y_kan = kan_predict_trained(u, i);
    
    // 3. Modelo Híbrido = Físico + KAN residual
    float y_hybrid = y_phys + y_kan * 0.3f;
    if (y_hybrid < 0.0005f) y_hybrid = 0.0005f;
    if (y_hybrid > 0.022f) y_hybrid = 0.022f;
    
    // 4. FUSIÓN SIGMOIDAL (Recomendación Gemini)
    // alpha = 1 → 100% sensor (Modo Maestro)
    // alpha = 0 → 100% híbrido (Modo Sensorless)
    float alpha;
    #if TRANSITION_ENABLE
        // Transición suave basada en tiempo
        alpha = 1.0f / (1.0f + expf(TRANSITION_RATE * (obs_time - TRANSITION_TIME)));
        
        // Actualizar varianza del error (filtro exponencial)
        float err = y_sensor - y_hybrid;
        error_variance = 0.99f * error_variance + 0.01f * err * err;
    #else
        alpha = 1.0f;  // Modo Maestro: 100% sensor
    #endif
    
    float y_final = alpha * y_sensor + (1.0f - alpha) * y_hybrid;
    
    // DEBUG periódico
    static int diag_cnt = 0;
    if (++diag_cnt % 500 == 0) {
        float i_expected = cbr_kan_map.get_equilibrium_current(y_sensor);
        printf("[CBR-KAN] y=%.4f m (%.2f mm), i=%.3f A (exp:%.3f), case=%d, alpha=%.2f\n", 
               y_sensor, y_sensor*1000, i, i_expected, 
               cbr_kan_map.get_active_case(), alpha);
    }
    
    // Salidas
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

// Archivo de entrenamiento CBR+KAN
FILE *fp_train = NULL;

// Flag de Modo Barrido (Para entrenamiento)
int sweep_mode = 1;     // 1 = Barrido activo
float sweep_timer = 0;  
int sweep_stage = 0;    

// Puntos base ZONA ESTABLE (4.5-6.0mm) para no caerse
float sweep_targets[] = {0.005, 0.0055, 0.006, 0.0055, 0.005}; 
int num_sweep_stages = 5;

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
	
	// Archivo para entrenamiento con FLUJO DE SANTANA
	fp_train = fopen("CBR_KAN_TRAIN.csv", "w+");
	if(fp_train) {
	    fprintf(fp_train, "t,y_sensor,y_santana,phi,i,u\n");
	    printf("[TRAIN] Archivo CBR_KAN_TRAIN.csv creado (incluye flujo Santana)\n");
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
    
    printf("LEVITADOR V3 (HIBRIDO) - CBR ONLINE\n");
    printf("CBR captura di/dt en las primeras 5 muestras del control\n");
    printf("================================================\n");
    
    // Variables para CBR online (captura durante control)
    #define CBR_SAMPLES 8          // Más muestras para mejor estimación
    #define CBR_PULSE_PWM 200      // PWM alto para forzar transitorio
    float cbr_ie_buffer[CBR_SAMPLES];
    int cbr_sample_idx = 0;
    int cbr_pulse_active = 1;      // Flag para fase de pulso CBR
    
    printf("Esperando sincronizacion...\n");
    printf("[CBR] Fase de pulso activa (PWM=%d) para medir di/dt\n", CBR_PULSE_PWM);
    
    int sincronizado = 0;
    
    while(1) 
	{
    	recibido_val=0;
        /* Recibimos algun dato - IGUAL QUE ORIGINAL */
        while(1) 
		{
        	ReadFile(h, &recibido_val, 1, &n, NULL);
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
					// --- LÓGICA DE BARRIDO AUTOMÁTICO ---
					if (sweep_mode) {
					    sweep_timer += Ts;
					    if (sweep_timer > 5.0f) { // Cambiar cada 5 segundos
					        sweep_timer = 0;
					        sweep_stage++;
					        if (sweep_stage >= num_sweep_stages) sweep_stage = 0; // Ciclar
					        
					        yd = sweep_targets[sweep_stage];
					        printf("\n[SWEEP] Nuevo Setpoint: %.1f mm\n", yd * 1000.0f);
					    }
					}
					
        			ef_1=ef;
        			y_1=y;

        			i=i+recibido_val;
        			y=esc*pv;       // SENSOR REAL (Solo validacion)
        			ie=esci*i;      // CORRIENTE REAL
        			
        			// --- CBR ONLINE: Capturar primeras muestras para estimar y0 ---
        			if (!cbr_init_done && cbr_sample_idx < CBR_SAMPLES) {
        			    cbr_ie_buffer[cbr_sample_idx] = ie;
        			    printf("[CBR] Muestra %d: ie=%.4f A\n", cbr_sample_idx, ie);
        			    cbr_sample_idx++;
        			    
        			    // Cuando tengamos suficientes muestras, calcular di/dt
        			    if (cbr_sample_idx >= CBR_SAMPLES) {
        			        float didt_sum = 0.0f;
        			        int valid = 0;
        			        for (int k = 1; k < CBR_SAMPLES; k++) {
        			            float didt_k = (cbr_ie_buffer[k] - cbr_ie_buffer[k-1]) / Ts;
        			            printf("[CBR] di/dt[%d] = %.2f A/s\n", k, didt_k);
        			            if (fabs(didt_k) > 0.5f) { didt_sum += didt_k; valid++; }
        			        }
        			        float didt_avg = (valid > 0) ? didt_sum / valid : 50.0f;
        			        
        			        // Estimar posición inicial con CBR
        			        y_cbr_estimada = cbr_estimar_posicion(fabs(didt_avg));
        			        i_cbr_equilibrio = calcular_corriente_equilibrio(y_cbr_estimada);
        			        phi_cbr_inicial = calcular_flujo_inicial(y_cbr_estimada, ie);
        			        
        			        printf("[CBR] ==============================\n");
        			        printf("[CBR] di/dt_avg = %.2f A/s\n", didt_avg);
        			        printf("[CBR] y0 estimada = %.4f m (%.2f mm)\n", y_cbr_estimada, y_cbr_estimada*1000);
        			        printf("[CBR] i_eq = %.3f A, phi0 = %.6f Wb\n", i_cbr_equilibrio, phi_cbr_inicial);
        			        printf("[CBR] ==============================\n");
        			        cbr_init_done = 1;
        			    }
        			}
        			
        			// --- OBSERVADOR DE FLUJO SANTANA ---
        			observador_santana(ie, u, Ts);
        			
        			// --- OBSERVADOR HIBRIDO ACTIVO ---
        			// CBR+KAN observa mientras el sensor controla (Modo Maestro)
        			observador_hibrido(ie, pwmf, y, &y_est_final, &y_phys_log, &y_kan_log, &y_hybrid_log);
        			
        			// MODO MAESTRO: PID usa SENSOR para control estable
        			// ef=yd-y_est_final;  // <-- Desactivado durante entrenamiento
        			ef=yd-y;               // <-- Sensor controla, CBR+KAN observa
        			
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
        			
        			// MODO MAESTRO: Siempre usar el PWM calculado por PID (como el original)
        			// La fase CBR solo OBSERVA, no interfiere con el control
        			pwmf=escs*pwmf;
        			pwm=(unsigned char)fabs(pwmf);
        			
        			// Marcar fin de fase CBR si corresponde
        			if (cbr_pulse_active && cbr_sample_idx >= CBR_SAMPLES) {
        			    cbr_pulse_active = 0;
        			    printf("[CBR] Observacion completada, control PID activo\n");
        			}

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
        			
        			// Guardar datos para entrenamiento (con flujo Santana)
        			if(fp_train && cbr_init_done) {
        			    fprintf(fp_train, "%.4f,%.6f,%.6f,%.8f,%.4f,%.4f\n",
        			            t, y, obs_y_santana, obs_phi, ie, u);
        			}
        			
					flagcom=0;
           			t=t+Ts;
           			
           			// Auto-salida después de 60 segundos
           			if (t > 60.0f) {
           			    printf("\n[AUTO-EXIT] Tiempo limite alcanzado (60s). Guardando datos...\n");
           			    fclose(fp);
           			    if(fp_train) {
           			        fclose(fp_train);
           			        printf("[TRAIN] Datos guardados en CBR_KAN_TRAIN.csv\n");
           			    }
           			    CloseHandle(h);
           			    return 0;
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
    }
fclose(fp);             
return 0;
}

/*
 * LEVITADOR KAN SIMPLE (Linux)
 * Versión simplificada de levitador_sensorless_kan.cpp
 * Mantiene lógica de control PID + Observador HiPPO-KAN
 */

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>
#include <linux/serial.h>

#include "sensorless_hippo_kan_final.h"
#include "CBR_InitPosition.h"

// ============================================================================
// PARÁMETROS DE CONTROL (Idénticos al código de Windows)
// ============================================================================
#define Ts      0.01f

#define kp      100.0f
#define ki      50.0f
#define kd      1.5f

#define kpi     12.0f
#define kii     3000.0f

#define Vref    9.86f
#define Iref    0.827f
#define Rs      2.20f

// ============================================================================
// VARIABLES GLOBALES
// ============================================================================
SensorlessHiPPO::SensorlessPipeline kan_pipeline;

// Estado serial (Replicando Windows)
unsigned char flagcom = 0;
unsigned short pv_raw = 0;
unsigned short i_raw = 0;

// Variables de control
float y = 0, y_1 = 0;
float ef = 0, ef_1 = 0;
float integral = 0;
float intei = 0;

// Constantes de escala
const float esc = 0.05f / 1023.0f;
const float esci = 5.0f / (Rs * 1023.0f);
const float escs = 254.0f / Vref;
const float iTs = 1.0f / Ts;

// ============================================================================
// FUNCIONES AUXILIARES
// ============================================================================

// Estimador HiPPO-KAN (Se mantiene igual, solo pasivo por ahora)
float estimar_posicion_kan(float u, float i, float t_actual) {
    // ... (Lógica HiPPO mantenida para monitoreo) ...
    // Por brevedad, usamos una versión simplificada aquí que solo llama al pipeline
    // En la implementación real, deberíamos mantener la lógica completa si queremos usarlo después.
    // Para esta prueba de "copia de Windows", solo devolveremos algo seguro o ejecutaremos
    // el pipeline sin afectar el control.
    
    static bool init = false;
    if (!init) { kan_pipeline.reset(); init = true; }
    
    float y_est = kan_pipeline.estimate(u, i);
    
    // Filtro simple para visualización
    static float y_filt = 0.005f;
    y_filt = 0.8f * y_filt + 0.2f * y_est;
    return y_filt;
}

int open_serial(const char* port) {
    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) return -1;

    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) return -1;

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 1;
    tty.c_cc[VTIME] = 0; // Blocking read per byte
    
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) return -1;

    // --- LOW LATENCY MODE (Critical for Linux USB-Serial) ---
    struct serial_struct serial;
    if (ioctl(fd, TIOCGSERIAL, &serial) == 0) {
        serial.flags |= ASYNC_LOW_LATENCY;
        ioctl(fd, TIOCSSERIAL, &serial);
        printf("[INFO] Modo Low Latency activado en %s\n", port);
    } else {
        perror("[WARN] No se pudo activar Low Latency");
    }

    tcflush(fd, TCIFLUSH);
    return fd;
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Uso: %s <puerto_serial> [modo: 0=Sensor(default), 1=Sensorless]\n", argv[0]);
        return 1;
    }

    const char* port = argv[1];
    int mode = (argc > 2) ? atoi(argv[2]) : 0; 

    printf("LEVITADOR KAN SIMPLE (Replica Windows)\n");
    printf("Puerto: %s\n", port);
    printf("Modo: %s\n", mode == 0 ? "SENSOR (Maestro)" : "SENSORLESS (KAN)");
    printf("Kp=%.1f Ki=%.1f Kd=%.1f | Kpi=%.1f Kii=%.1f\n", kp, ki, kd, kpi, kii);
    printf("----------------------------------------\n");

    int fd = open_serial(port);
    if (fd < 0) {
        perror("Error abriendo puerto");
        return 1;
    }
    
    FILE *fp_log = fopen("KAN_SIMPLE_LOG.csv", "w");
    if (fp_log) fprintf(fp_log, "t,yd,y_sensor,y_est,u,i,pwm\n");

    unsigned char rx_byte;
    unsigned char pwm_byte = 0;
    
    float t = 0.0f;
    float yd = 0.005f; 
    float u = 0.0f;
    float u_prev = 0.0f;
    float i_a = 0.0f;
    float y_est = 0.005f;
    float y_est_offset = 0.0f;
    float alpha_sensor = 1.0f;
    const float I_KAN_MIN = 0.10f;
    const float I_RESET = 0.05f;
    const float U_RESET = 0.50f;

    printf("Esperando datos...\n");

    while (1) {
        int n = read(fd, &rx_byte, 1);
        if (n <= 0) continue;

        // Máquina de estados exacta de Windows
        if (flagcom != 0) flagcom++;

        if ((rx_byte == 0xAA) && (flagcom == 0)) {
            pv_raw = 0;
            i_raw = 0;
            flagcom = 1;
        }

        if (flagcom == 2) {
            pv_raw = rx_byte;
            pv_raw = pv_raw << 8;
        }

        if (flagcom == 3) {
            pv_raw = pv_raw + rx_byte;
        }

        if (flagcom == 4) {
            i_raw = rx_byte;
            i_raw = i_raw << 8;
        }

        if (flagcom == 5) {
            // Byte final recibido, procesar ciclo de control
            i_raw = i_raw + rx_byte;

            // Escalamiento (Igual a Windows)
            y = esc * pv_raw;
            float ie = esci * i_raw;
            i_a = ie; // Guardar para log

            float y_est_raw;
            if ((ie < I_RESET) && (u_prev < U_RESET)) {
                kan_pipeline.reset();
                y_est_raw = y;
            } else {
                y_est_raw = estimar_posicion_kan(u_prev, ie, t);
            }

            if (mode == 1) {
                if ((alpha_sensor > 0.0f) && (ie > I_KAN_MIN)) {
                    y_est_offset = 0.98f * y_est_offset + 0.02f * (y - y_est_raw);
                    if (y_est_offset > 0.015f) y_est_offset = 0.015f;
                    if (y_est_offset < -0.015f) y_est_offset = -0.015f;
                }
            }

            y_est = y_est_raw + y_est_offset;
            if (y_est < 0.001f) y_est = 0.001f;
            if (y_est > 0.020f) y_est = 0.020f;

            if (mode == 1) {
                float est_err = fabsf(y_est - y);
                float alpha_target = est_err / 0.004f;
                if (alpha_target < 0.0f) alpha_target = 0.0f;
                if (alpha_target > 1.0f) alpha_target = 1.0f;
                if (ie < I_KAN_MIN) alpha_target = 1.0f;
                alpha_sensor = 0.98f * alpha_sensor + 0.02f * alpha_target;
                if (alpha_sensor < 0.0f) alpha_sensor = 0.0f;
                if (alpha_sensor > 1.0f) alpha_sensor = 1.0f;
            }

            // Perfil de Setpoint (Igual a Windows)
            if (t > 10.0f) yd = 0.0045f;
            if (t > 15.0f) yd = 0.005f;

            // --- PID POSICION ---
            float y_fb;
            if (mode == 1) {
                y_fb = alpha_sensor * y + (1.0f - alpha_sensor) * y_est;
            } else {
                y_fb = y;
            }
            ef = yd - y_fb;
            
            // Si estuviéramos en modo sensorless real, y sería y_est
            // Pero mantenemos la lógica base de Windows primero
            
            float proporcional = kp * ef;
            float derivativa = kd * (ef - ef_1) * iTs;

            if ((integral < Iref) && (integral > -Iref)) {
                integral = integral + ki * Ts * ef;
            } else {
                if (integral >= Iref) integral = 0.95f * Iref;
                if (integral <= -Iref) integral = -0.95f * Iref;
            }

            float id = proporcional + integral + derivativa;

            if (id > 0) id = 0;
            if (id <= -Iref) id = -Iref;

            // --- PI CORRIENTE ---
            float ied = -id;
            float ei = ied - ie;
            
            float propi = kpi * ei;
            
            if ((intei < Vref) && (intei > -Vref)) {
                intei = intei + kii * Ts * ei;
            } else {
                if (intei >= Vref) intei = 0.95f * Vref;
                if (intei <= -Vref) intei = -0.95f * Vref;
            }

            u = propi + intei;

            if (u > Vref) u = Vref;
            if (u <= 0) u = 0;

            float pwmf = u;
            pwmf = escs * pwmf;
            pwm_byte = (unsigned char)fabs(pwmf);

            // Enviar control
            write(fd, &pwm_byte, 1);

            u_prev = u;

            // Actualizar estados pasados
            ef_1 = ef;
            y_1 = y;
            t = t + Ts;
            flagcom = 0; // Reiniciar máquina

            // Logs y HiPPO (Auxiliar)
            static int print_div = 0;
            if (++print_div >= 20) { // 200ms aprox
                printf("t=%.2f yd=%.1fmm y=%.1fmm y_est=%.1fmm a=%.2f off=%.1fmm i=%.3fA u=%.2fV PWM=%d\n",
                       t, yd*1000, y*1000, y_est*1000, alpha_sensor, y_est_offset*1000.0f, ie, u, pwm_byte);
                print_div = 0;
            }

            if (fp_log) fprintf(fp_log, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d\n",
                                t, yd, y, y_est, u, ie, pwm_byte);
        }
    }

    close(fd);
    if(fp_log) fclose(fp_log);
    return 0;
}

/*
 * LEVITADOR SENSORLESS V4 - HiPPO-KAN
 * Observador completo con HiPPO para flujo + KAN para posición
 */

#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _WIN32
 #include <winsock2.h>
 #include <ws2tcpip.h>
 #include <windows.h>
 #include <conio.h>
 #pragma comment(lib, "Ws2_32.lib")
#else
 #include <sys/socket.h>
 #include <netinet/in.h>
 #include <arpa/inet.h>
 #include <unistd.h>
 #include <fcntl.h>
 #include <termios.h>
 #include <errno.h>
 #include <sys/select.h>
 #include <time.h>
#endif

#include "sensorless_hippo_kan_final.h"  // Pipeline HiPPO-KAN completo (Final)
#include "CBR_InitPosition.h"      // Física instantánea para warm-up

#define Ts      0.01
#define KAN_OUTPUT_DELTA 0

// ============================================================================
// PROFILING (Jetson/Linux)
// ============================================================================
#define ENABLE_PROFILING 1
#define PROF_LOG_EVERY 1

static FILE* fp_profile = NULL;

#if ENABLE_PROFILING && !defined(_WIN32)
static inline uint64_t now_ns_monotonic_raw() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline double ns_to_us(uint64_t ns) {
    return (double)ns / 1000.0;
}
#endif

// UDP Telemetry Config
#define UDP_IP "127.0.0.1"
#define UDP_PORT 9000

#ifdef _WIN32
 SOCKET udp_socket = INVALID_SOCKET;
#else
 int udp_socket = -1;
 #ifndef INVALID_SOCKET
  #define INVALID_SOCKET (-1)
 #endif
#endif
struct sockaddr_in dest_addr;

#pragma pack(push, 1)
struct TelemetryPacket {
    float t;
    float y_ref;
    float y_sensor;
    float y_est;
    float phi_est;
    float current;
    float voltage;
    float error_mm;
    int mode;
    float fusion_alpha;
};
#pragma pack(pop)

void init_udp() {
#ifdef _WIN32
    WSADATA wsaData;
    int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (res != 0) {
        printf("WSAStartup failed: %d\n", res);
        return;
    }
    udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_socket == INVALID_SOCKET) {
        printf("Socket failed: %d\n", WSAGetLastError());
        return;
    }
#else
    udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_socket < 0) {
        printf("Socket failed: %d\n", errno);
        return;
    }
#endif

    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(UDP_PORT);
    dest_addr.sin_addr.s_addr = inet_addr(UDP_IP);

    printf("[UDP] Initialized. Sending to %s:%d\n", UDP_IP, UDP_PORT);
}

void send_telemetry(float t, float yd, float y_sens, float y_est, float phi, float i, float u, float err, int mode, float alpha) {
    static int pkt_count = 0;
    if (udp_socket == INVALID_SOCKET) {
        if (pkt_count == 0) printf("[UDP] Socket invalido, no se envian paquetes!\n");
        pkt_count++;
        return;
    }
    
    TelemetryPacket pkt;
    pkt.t = t;
    pkt.y_ref = yd;
    pkt.y_sensor = y_sens;
    pkt.y_est = y_est;
    pkt.phi_est = phi;
    pkt.current = i;
    pkt.voltage = u;
    pkt.error_mm = err;
    pkt.mode = mode;
    pkt.fusion_alpha = alpha;
    
    int sent = sendto(udp_socket, (char*)&pkt, sizeof(pkt), 0, (struct sockaddr*)&dest_addr, sizeof(dest_addr));
    pkt_count++;
    if (pkt_count == 1) {
        printf("[UDP] Primer paquete enviado (%d bytes)\n", sent);
    }
}


// PID de posición
#define kp      100    // Restaurado a valor base robusto
#define ki      50     // Restaurado a valor base
#define kd      1.5    // Restaurado a valor base

// Valor estimado de integral para sostener el peso del iman (Feedforward de gravedad)
// id = integral = -0.4 implica ied = 0.4A aprox
#define GRAVITY_COMP_INTEGRAL -0.4f

#define kpi      12.0
#define kii      3000.0

#define Vref    9.86
#define Iref    0.827
#define Rs      2.20

// ============================================================================
// VARIABLES GLOBALES (Movidas al inicio para acceso global)
// ============================================================================
using namespace std;
unsigned char flagcom=0, flagfile=0, pwm;
unsigned short int pv, i_raw;
float y, y_1, ef, ef_1=0, u, t=0;
float esc=.05/1023.0, esci=5.0/(Rs*1023.0), escs=254.0/Vref, iTs=1/Ts, pwmf;
float yd=0.005, proporcional, derivativa=0, ie, ied, id, ei, propi, intei=0, integral=0;

bool sine_enabled = false;
float yd_base = 0.005f;
float sine_amp = 0.0005f;
float sine_freq = 0.3f;

bool u_ol_enabled = false;
float u_ol_offset = 0.0f;
float u_ol_amp = 2.0f;
float u_ol_freq = 0.5f;

float compute_yd(float t_now) {
    float yd_out = yd_base;
    if (sine_enabled) {
        yd_out = yd_base + sine_amp * sinf(2.0f * 3.14159265359f * sine_freq * t_now);
    }
    if (yd_out < 0.001f) yd_out = 0.001f;
    if (yd_out > 0.020f) yd_out = 0.020f;
    return yd_out;
}

float compute_u_ol(float t_now) {
    float u_out = u_ol_offset;
    if (u_ol_enabled) {
        u_out = u_ol_offset + u_ol_amp * sinf(2.0f * 3.14159265359f * u_ol_freq * t_now);
    }
    if (u_out < 0.0f) u_out = 0.0f;
    if (u_out > Vref) u_out = Vref;
    return u_out;
}

float y_est_final = 0.0f;
FILE *fp_train = NULL;
bool ood_active = false;
float phi_phys = 0.0f;
bool phi_phys_initialized = false;
float kan_confidence = 1.0f;

// ============================================================================
// OBSERVADOR HiPPO-KAN GLOBAL
// ============================================================================
SensorlessHiPPO::SensorlessPipeline kan_pipeline;

// Modo de operación
#define MODE_MAESTRO    0   // 100% sensor
#define MODE_HIBRIDO    1   // Fusión
#define MODE_SENSORLESS 2   // 100% KAN

int operation_mode = MODE_MAESTRO;  // Inicia en modo sensor para pruebas seguras
float fusion_alpha = 1.0f;  // Inicia en 100% sensor

// ============================================================================
// FUSIÓN DINÁMICA - Variables de Estado
// ============================================================================
#define FUSION_ALPHA_INICIAL  1.0f    // 100% sensor al inicio
#define FUSION_ALPHA_OBJETIVO 0.5f    // 50% sensor / 50% KAN (conservador mientras se calibra)
#define FUSION_DECREMENTO     0.005f  // Reducción lenta por ciclo
#define ERROR_UMBRAL_ENTRY_MM 2.0f    // Umbral para ENTRAR a estabilidad (< 2.0mm)
#define ERROR_UMBRAL_EXIT_MM  3.0f    // Umbral para SALIR de estabilidad (> 3.0mm)
#define MUESTRAS_ESTABLES     50      // Muestras consecutivas para considerar estable

int contador_estable = 0;              // Contador de muestras estables consecutivas
bool estado_estable = false;           // Flag de estado estable alcanzado
bool fusion_dinamica_activa = true;    // Habilitar/deshabilitar fusión dinámica

// ============================================================================
// FUNCIÓN: actualizar_fusion_dinamica()
// Ajusta fusion_alpha dinámicamente con histéresis
// ============================================================================
void actualizar_fusion_dinamica(float error_mm) {
    if (!fusion_dinamica_activa) return;
    
    // Solo actúa en modo HIBRIDO
    if (operation_mode != MODE_HIBRIDO) {
        contador_estable = 0;
        estado_estable = false;
        return;
    }
    
    float abs_error = fabs(error_mm);

    if (ood_active || kan_confidence < 0.2f) {
        if (estado_estable) {
            estado_estable = false;
            contador_estable = 0;
            fusion_alpha = FUSION_ALPHA_INICIAL;
        } else {
            contador_estable = 0;
        }
        return;
    }

    // Lógica de Histéresis
    if (!estado_estable) {
        // Estamos buscando estabilidad
        if (abs_error < ERROR_UMBRAL_ENTRY_MM && kan_confidence > 0.6f) {
            contador_estable++;
            if (contador_estable >= MUESTRAS_ESTABLES) {
                estado_estable = true;
                printf("\n>> [FUSION] Estado ESTABLE alcanzado (Err < %.1fmm)! Iniciando transicion...\n", ERROR_UMBRAL_ENTRY_MM);
            }
        } else {
            contador_estable = 0; // Reiniciar si un solo pico supera el umbral de entrada
        }
    } else {
        // Ya estamos estables, verificamos si perdemos estabilidad
        if (abs_error > ERROR_UMBRAL_EXIT_MM) {
            printf("\n>> [FUSION] Inestabilidad detectada (Err > %.1fmm)! Aplicando Warm Reset...\n", ERROR_UMBRAL_EXIT_MM);
            estado_estable = false;
            contador_estable = 0;
            fusion_alpha = FUSION_ALPHA_INICIAL; // Reset inmediato
            // WARM RESET: Pre-carga de estados para mantener campo magnético
            integral = 0.08f;  // Offset para compensar gravedad
            intei = 0.45f;     // Pre-carga de voltaje para generar campo instantáneo
        }
    }
    
    // Reducir fusion_alpha linealmente si estamos en estado estable
    if (estado_estable && fusion_alpha > FUSION_ALPHA_OBJETIVO) {
        fusion_alpha -= FUSION_DECREMENTO;
        
        // Clamp al valor objetivo
        if (fusion_alpha < FUSION_ALPHA_OBJETIVO) {
            fusion_alpha = FUSION_ALPHA_OBJETIVO;
            printf("\n>> [FUSION] Transicion completa: %.0f%% sensor / %.0f%% KAN\n",
                   fusion_alpha * 100, (1 - fusion_alpha) * 100);
        }
    }
}

// Variables para logging
float y_kan_est = 0.0f;
float phi_est = 0.0f;
float y_cbr_warmup = 0.0f;  // Estimación CBR para debug

// Variables de estado para CBR Warm-up
float i_prev_sample = 0.0f;
#define WARMUP_TIME 0.3f    // 300ms de transición suave CBR -> HiPPO
#define CURRENT_TRIGGER 0.1f // Gatillo de arranque (100mA)

static bool hippo_active = false;
static float t_activation = 0.0f;

// ============================================================================
// FUNCIÓN DE ESTIMACIÓN HiPPO-KAN + SOFT-START
// ============================================================================
float estimar_posicion_kan(float u, float i, float y_sensor, float t_actual) {
    // 1. GATILLO DE ARRANQUE (Start Trigger)
    // Mantiene HiPPO apagado hasta que hay corriente física real
    if (!hippo_active) {
        if (i > CURRENT_TRIGGER) {
            hippo_active = true;
            t_activation = t_actual;
            kan_pipeline.reset(); // Reiniciar estados a 0
            printf("[HiPPO] ACTIVADO en t=%.3fs (i=%.3fA)\n", t_actual, i);
        } else {
            // Mientras esperamos, asumimos posición de reposo (14mm)
            // y mantenemos HiPPO en reset para no integrar ruido
            kan_pipeline.reset();
            phi_est = kan_pipeline.get_phi();
            i_prev_sample = i;
            return 0.014f; 
        }
    }

    // 2. ESTIMADOR HÍBRIDO: HiPPO-KAN con Criterio de Coherencia
    // Usamos la red KAN entrenada, pero con validación de coherencia
    
    // Ejecutar pipeline HiPPO-KAN completo
    float y_kan_raw = kan_pipeline.estimate(u, i);
    phi_est = kan_pipeline.get_phi();

    float u_z_raw = (u - SensorlessHiPPO::U_MEAN) / SensorlessHiPPO::U_STD;
    float i_z_raw = (i - SensorlessHiPPO::I_MEAN) / SensorlessHiPPO::I_STD;
    ood_active = (fabsf(u_z_raw) > 3.0f) || (fabsf(i_z_raw) > 3.0f) || (i < 0.05f);

    if (!phi_phys_initialized) {
        phi_phys = phi_est;
        phi_phys_initialized = true;
    }

    if (!ood_active) {
        phi_phys += (u - Rs * i) * Ts;
        if (phi_phys < 0.0001f) phi_phys = 0.0001f;
        if (phi_phys > 0.10f) phi_phys = 0.10f;
    }

    if (ood_active) {
        // En OOD evitamos integrar phi directo (u-Ri)*dt porque puede explotar
        // cuando u está saturado. En su lugar estimamos L instantánea usando di/dt.
        float didt = (i - i_prev_sample) / Ts;
        float y_phys = 0.014f;

        if (fabsf(didt) > 0.5f) {
            float L_inst = (u - Rs * i) / didt;
            // Clamp suave de L para evitar valores absurdos
            if (L_inst < 0.01f) L_inst = 0.01f;
            if (L_inst > 0.20f) L_inst = 0.20f;

            phi_phys = L_inst * i;
            y_phys = calcular_posicion_desde_L(L_inst);
        }
        // Si di/dt es muy pequeño, mantenemos phi_phys previo (evita ruido)

        // Clamp de seguridad para phi
        if (phi_phys < 0.0001f) phi_phys = 0.0001f;
        if (phi_phys > 0.10f) phi_phys = 0.10f;

        phi_est = phi_phys;
        y_kan_raw = y_phys;
    }

    float y_phys_base = 0.014f;
    if (i > 0.05f) {
        float L_phys = phi_phys / i;
        if (L_phys < 0.01f) L_phys = 0.01f;
        if (L_phys > 0.20f) L_phys = 0.20f;
        y_phys_base = calcular_posicion_desde_L(L_phys);
    }

#if KAN_OUTPUT_DELTA
    y_kan_raw = y_phys_base + y_kan_raw;
#endif

    static float phi_est_prev = 0.0f;
    static bool phi_est_prev_initialized = false;
    float dphi_dt = 0.0f;
    if (phi_est_prev_initialized) {
        dphi_dt = (phi_est - phi_est_prev) / Ts;
    } else {
        phi_est_prev_initialized = true;
    }
    phi_est_prev = phi_est;

    float phi_model = calcular_inductancia(y_kan_raw) * i;
    float res_const = phi_est - phi_model;
    float res_kirch = u - (Rs * i) - dphi_dt;

    const float sigma_phi = 0.003f;
    const float sigma_u = 2.0f;
    float z_phi = res_const / sigma_phi;
    float z_u = res_kirch / sigma_u;
    kan_confidence = 1.0f / (1.0f + z_phi * z_phi + z_u * z_u);
    if (kan_confidence < 0.0f) kan_confidence = 0.0f;
    if (kan_confidence > 1.0f) kan_confidence = 1.0f;
    if (ood_active) kan_confidence = 0.0f;

    float y_blend_raw = y_phys_base + kan_confidence * (y_kan_raw - y_phys_base);

    // Variable estática para filtro suavizador
    static float y_kan_filt = 0.005f;
    
    // Suavizado de la salida KAN (filtro de una polea)
    y_kan_filt = 0.7f * y_kan_filt + 0.3f * y_blend_raw;
    
    // Asignar al estimador principal
    y_kan_est = y_kan_filt;

    // 3. SOFT-START con CBR (Transición suave)
    // Durante los primeros ms tras la activación, confiamos en CBR
    float t_since_active = t_actual - t_activation;
    float y_estimada_final = y_kan_est;
    
    if (t_since_active < WARMUP_TIME) {
        // Calcular di/dt
        float didt = (i - i_prev_sample) / Ts;
        if (fabs(didt) < 0.1f) didt = 0.1f;
        
        // Estimar con física instantánea si hay voltaje suficiente
        if (u > 1.0f) {
            float L_inst = (u - Rs * i) / didt;
            y_cbr_warmup = calcular_posicion_desde_L(L_inst);
            
            // Fusión: 100% CBR al inicio -> 0% al final del warmup
            float w_cbr = 1.0f - (t_since_active / WARMUP_TIME);
            
            // Curva de transición suave (cosenoidal)
            w_cbr = 0.5f * (1.0f + cosf(3.14159f * t_since_active / WARMUP_TIME));
            
            y_estimada_final = w_cbr * y_cbr_warmup + (1.0f - w_cbr) * y_kan_est;
        } else {
            // Sin voltaje, mantenemos la última estimación o reposo
             y_estimada_final = (t_since_active < 0.05f) ? 0.014f : y_kan_est;
        }
    }
    
    // Actualizar estado anterior
    i_prev_sample = i;
    
    // 4. Selección de salida según modo
    float y_output;
    switch (operation_mode) {
        case MODE_MAESTRO:
            y_output = y_sensor;
            break;
        case MODE_SENSORLESS:
            y_output = y_estimada_final;
            break;
        case MODE_HIBRIDO:
        default:
            y_output = fusion_alpha * y_sensor + (1.0f - fusion_alpha) * y_estimada_final;
            break;
    }
    
    // Clamp de seguridad
    if (y_output < 0.001f) y_output = 0.001f;
    if (y_output > 0.020f) y_output = 0.020f;
    
    return y_output;
}

// Función para procesar comandos de teclado y pipe (GUI)
#ifdef _WIN32
void check_input(HANDLE h_serial, FILE* fp_log) {
    char key = 0;
    
    // 1. Check Pipe (Stdin) - Prioridad a comandos de GUI
    HANDLE hStdIn = GetStdHandle(STD_INPUT_HANDLE);
    DWORD dwAvail = 0;
    if (PeekNamedPipe(hStdIn, NULL, 0, NULL, &dwAvail, NULL) && dwAvail > 0) {
        char buf[1];
        DWORD dwRead;
        if (ReadFile(hStdIn, buf, 1, &dwRead, NULL) && dwRead > 0) {
            key = buf[0];
        }
    }
    // 2. Check Console (Direct keyboard) - Fallback manual
    else if (_kbhit()) {
        key = _getch();
    }

    if(key != 0 && key != '\n' && key != '\r') {
        switch(key) {
            case '1': yd_base = 0.004f; printf("\n>> Setpoint: 4mm\n"); break;
            case '2': yd_base = 0.005f; printf("\n>> Setpoint: 5mm\n"); break;
            case '3': yd_base = 0.006f; printf("\n>> Setpoint: 6mm\n"); break;
            case '4': yd_base = 0.007f; printf("\n>> Setpoint: 7mm\n"); break;
            case '5': yd_base = 0.008f; printf("\n>> Setpoint: 8mm\n"); break;

            case 'g': case 'G':
                sine_enabled = !sine_enabled;
                printf("\n>> SENO %s (base=%.1fmm, amp=%.2fmm, f=%.2fHz)\n", sine_enabled ? "ON" : "OFF", yd_base*1000.0f, sine_amp*1000.0f, sine_freq);
                break;

            case '[':
                sine_amp -= 0.0001f;
                if (sine_amp < 0.0f) sine_amp = 0.0f;
                printf("\n>> SENO amp=%.2fmm\n", sine_amp*1000.0f);
                break;
            case ']':
                sine_amp += 0.0001f;
                if (sine_amp > 0.003f) sine_amp = 0.003f;
                printf("\n>> SENO amp=%.2fmm\n", sine_amp*1000.0f);
                break;

            case '{':
                sine_freq -= 0.05f;
                if (sine_freq < 0.05f) sine_freq = 0.05f;
                printf("\n>> SENO f=%.2fHz\n", sine_freq);
                break;
            case '}':
                sine_freq += 0.05f;
                if (sine_freq > 5.0f) sine_freq = 5.0f;
                printf("\n>> SENO f=%.2fHz\n", sine_freq);
                break;

            case 'u': case 'U':
                u_ol_enabled = !u_ol_enabled;
                printf("\n>> U-OL %s (offset=%.2fV, amp=%.2fV, f=%.2fHz)\n", u_ol_enabled ? "ON" : "OFF", u_ol_offset, u_ol_amp, u_ol_freq);
                fflush(stdout);
                break;
            case 'j': case 'J':
                u_ol_amp -= 0.2f;
                if (u_ol_amp < 0.0f) u_ol_amp = 0.0f;
                printf("\n>> U-OL amp=%.2fV\n", u_ol_amp);
                fflush(stdout);
                break;
            case 'k': case 'K':
                u_ol_amp += 0.2f;
                if (u_ol_amp > Vref) u_ol_amp = Vref;
                printf("\n>> U-OL amp=%.2fV\n", u_ol_amp);
                fflush(stdout);
                break;
            case ',':
                u_ol_freq -= 0.05f;
                if (u_ol_freq < 0.05f) u_ol_freq = 0.05f;
                printf("\n>> U-OL f=%.2fHz\n", u_ol_freq);
                fflush(stdout);
                break;
            case '.':
                u_ol_freq += 0.05f;
                if (u_ol_freq > 20.0f) u_ol_freq = 20.0f;
                printf("\n>> U-OL f=%.2fHz\n", u_ol_freq);
                fflush(stdout);
                break;
            case 'o': case 'O':
                u_ol_offset -= 0.2f;
                if (u_ol_offset < 0.0f) u_ol_offset = 0.0f;
                printf("\n>> U-OL offset=%.2fV\n", u_ol_offset);
                fflush(stdout);
                break;
            case 'p': case 'P':
                u_ol_offset += 0.2f;
                if (u_ol_offset > Vref) u_ol_offset = Vref;
                printf("\n>> U-OL offset=%.2fV\n", u_ol_offset);
                fflush(stdout);
                break;
            
            case 'm': case 'M':
                operation_mode = MODE_MAESTRO;
                printf("\n>> MODO MAESTRO\n");
                break;
            case 'h': case 'H':
                operation_mode = MODE_HIBRIDO;
                printf("\n>> MODO HIBRIDO (%.0f%% sensor)\n", fusion_alpha*100);
                break;
            case 's': case 'S':
                operation_mode = MODE_SENSORLESS;
                printf("\n>> MODO SENSORLESS\n");
                break;
                
            case '+': case '=':
                fusion_alpha += 0.1f;
                if(fusion_alpha > 1.0f) fusion_alpha = 1.0f;
                printf("\n>> Fusion: %.0f%% sensor\n", fusion_alpha*100);
                break;
            case '-': case '_':
                fusion_alpha -= 0.1f;
                if(fusion_alpha < 0.0f) fusion_alpha = 0.0f;
                printf("\n>> Fusion: %.0f%% sensor\n", fusion_alpha*100);
                break;
                
            case 'r': case 'R':
                kan_pipeline.reset();
                printf("\n>> HiPPO reset\n");
                break;
                
            case 'd': case 'D':
                fusion_dinamica_activa = !fusion_dinamica_activa;
                if (fusion_dinamica_activa) {
                    fusion_alpha = FUSION_ALPHA_INICIAL;
                    contador_estable = 0;
                    estado_estable = false;
                    printf("\n>> Fusion Dinamica ACTIVADA (reset a 100%% sensor)\n");
                } else {
                    printf("\n>> Fusion Dinamica DESACTIVADA (alpha=%.0f%%)\n", fusion_alpha*100);
                }
                break;
                
            case 'q': case 'Q':
                printf("\nSaliendo...\n");
                if(fp_log) fclose(fp_log);
                if(fp_train) fclose(fp_train);
                if(fp_profile) fclose(fp_profile);
                CloseHandle(h_serial);
                exit(0);
        }
    }
}
#else
void check_input(int fd_serial, FILE* fp_log) {
    char key = 0;

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(0, &rfds);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    int ret = select(1, &rfds, NULL, NULL, &tv);
    if (ret > 0 && FD_ISSET(0, &rfds)) {
        char buf[1];
        ssize_t nread = read(0, buf, 1);
        if (nread > 0) {
            key = buf[0];
        }
    }

    if(key != 0 && key != '\n' && key != '\r') {
        switch(key) {
            case '1': yd_base = 0.004f; printf("\n>> Setpoint: 4mm\n"); break;
            case '2': yd_base = 0.005f; printf("\n>> Setpoint: 5mm\n"); break;
            case '3': yd_base = 0.006f; printf("\n>> Setpoint: 6mm\n"); break;
            case '4': yd_base = 0.007f; printf("\n>> Setpoint: 7mm\n"); break;
            case '5': yd_base = 0.008f; printf("\n>> Setpoint: 8mm\n"); break;

            case 'g': case 'G':
                sine_enabled = !sine_enabled;
                printf("\n>> SENO %s (base=%.1fmm, amp=%.2fmm, f=%.2fHz)\n", sine_enabled ? "ON" : "OFF", yd_base*1000.0f, sine_amp*1000.0f, sine_freq);
                break;

            case '[':
                sine_amp -= 0.0001f;
                if (sine_amp < 0.0f) sine_amp = 0.0f;
                printf("\n>> SENO amp=%.2fmm\n", sine_amp*1000.0f);
                break;
            case ']':
                sine_amp += 0.0001f;
                if (sine_amp > 0.003f) sine_amp = 0.003f;
                printf("\n>> SENO amp=%.2fmm\n", sine_amp*1000.0f);
                break;

            case '{':
                sine_freq -= 0.05f;
                if (sine_freq < 0.05f) sine_freq = 0.05f;
                printf("\n>> SENO f=%.2fHz\n", sine_freq);
                break;
            case '}':
                sine_freq += 0.05f;
                if (sine_freq > 5.0f) sine_freq = 5.0f;
                printf("\n>> SENO f=%.2fHz\n", sine_freq);
                break;

            case 'u': case 'U':
                u_ol_enabled = !u_ol_enabled;
                printf("\n>> U-OL %s (offset=%.2fV, amp=%.2fV, f=%.2fHz)\n", u_ol_enabled ? "ON" : "OFF", u_ol_offset, u_ol_amp, u_ol_freq);
                fflush(stdout);
                break;
            case 'j': case 'J':
                u_ol_amp -= 0.2f;
                if (u_ol_amp < 0.0f) u_ol_amp = 0.0f;
                printf("\n>> U-OL amp=%.2fV\n", u_ol_amp);
                fflush(stdout);
                break;
            case 'k': case 'K':
                u_ol_amp += 0.2f;
                if (u_ol_amp > Vref) u_ol_amp = Vref;
                printf("\n>> U-OL amp=%.2fV\n", u_ol_amp);
                fflush(stdout);
                break;
            case ',':
                u_ol_freq -= 0.05f;
                if (u_ol_freq < 0.05f) u_ol_freq = 0.05f;
                printf("\n>> U-OL f=%.2fHz\n", u_ol_freq);
                fflush(stdout);
                break;
            case '.':
                u_ol_freq += 0.05f;
                if (u_ol_freq > 20.0f) u_ol_freq = 20.0f;
                printf("\n>> U-OL f=%.2fHz\n", u_ol_freq);
                fflush(stdout);
                break;
            case 'o': case 'O':
                u_ol_offset -= 0.2f;
                if (u_ol_offset < 0.0f) u_ol_offset = 0.0f;
                printf("\n>> U-OL offset=%.2fV\n", u_ol_offset);
                fflush(stdout);
                break;
            case 'p': case 'P':
                u_ol_offset += 0.2f;
                if (u_ol_offset > Vref) u_ol_offset = Vref;
                printf("\n>> U-OL offset=%.2fV\n", u_ol_offset);
                fflush(stdout);
                break;

            case 'm': case 'M':
                operation_mode = MODE_MAESTRO;
                printf("\n>> MODO MAESTRO\n");
                break;
            case 'h': case 'H':
                operation_mode = MODE_HIBRIDO;
                printf("\n>> MODO HIBRIDO (%.0f%% sensor)\n", fusion_alpha*100);
                break;
            case 's': case 'S':
                operation_mode = MODE_SENSORLESS;
                printf("\n>> MODO SENSORLESS\n");
                break;

            case '+': case '=':
                fusion_alpha += 0.1f;
                if(fusion_alpha > 1.0f) fusion_alpha = 1.0f;
                printf("\n>> Fusion: %.0f%% sensor\n", fusion_alpha*100);
                break;
            case '-': case '_':
                fusion_alpha -= 0.1f;
                if(fusion_alpha < 0.0f) fusion_alpha = 0.0f;
                printf("\n>> Fusion: %.0f%% sensor\n", fusion_alpha*100);
                break;

            case 'r': case 'R':
                kan_pipeline.reset();
                printf("\n>> HiPPO reset\n");
                break;

            case 'd': case 'D':
                fusion_dinamica_activa = !fusion_dinamica_activa;
                if (fusion_dinamica_activa) {
                    fusion_alpha = FUSION_ALPHA_INICIAL;
                    contador_estable = 0;
                    estado_estable = false;
                    printf("\n>> Fusion Dinamica ACTIVADA (reset a 100%% sensor)\n");
                } else {
                    printf("\n>> Fusion Dinamica DESACTIVADA (alpha=%.0f%%)\n", fusion_alpha*100);
                }
                break;

            case 'q': case 'Q':
                printf("\nSaliendo...\n");
                if(fp_log) fclose(fp_log);
                if(fp_train) fclose(fp_train);
                if(fp_profile) fclose(fp_profile);
                close(fd_serial);
                if (udp_socket != INVALID_SOCKET) close(udp_socket);
                exit(0);
        }
    }
}
#endif


int main(int argc, char* argv[])
{
#ifdef _WIN32
    HANDLE h;
    DCB dcb;
#else
    int h;
    struct termios tty;
#endif
    FILE *fp;

    setvbuf(stdout, NULL, _IOLBF, 0);
    
    // Parse COM Port from arguments
#ifdef _WIN32
    char portName[32];
    if (argc > 1) {
        snprintf(portName, sizeof(portName), "\\\\.\\%s", argv[1]);
    } else {
        strcpy(portName, "\\\\.\\COM1"); // Default
    }
#else
    const char* portName = (argc > 1) ? argv[1] : "/dev/ttyUSB0";
#endif

    // Optional: auto-start U-OL from CLI for headless/testing
    // Usage: ./levitador_sensorless_kan <port> --uol <offset_V> <amp_V> <freq_Hz>
    bool uol_autostart = false;
    float uol_aut_offset = u_ol_offset;
    float uol_aut_amp = u_ol_amp;
    float uol_aut_freq = u_ol_freq;
    for (int ai = 2; ai < argc; ++ai) {
        if (strcmp(argv[ai], "--uol") == 0 && (ai + 3) < argc) {
            uol_autostart = true;
            uol_aut_offset = (float)atof(argv[ai + 1]);
            uol_aut_amp = (float)atof(argv[ai + 2]);
            uol_aut_freq = (float)atof(argv[ai + 3]);
            ai += 3;
        }
    }
    if (uol_autostart) {
        u_ol_offset = uol_aut_offset;
        u_ol_amp = uol_aut_amp;
        u_ol_freq = uol_aut_freq;
        u_ol_enabled = true;
        if (u_ol_offset < 0.0f) u_ol_offset = 0.0f;
        if (u_ol_offset > Vref) u_ol_offset = Vref;
        if (u_ol_amp < 0.0f) u_ol_amp = 0.0f;
        if (u_ol_amp > Vref) u_ol_amp = Vref;
        if (u_ol_freq < 0.05f) u_ol_freq = 0.05f;
        if (u_ol_freq > 20.0f) u_ol_freq = 20.0f;
    }

    // Inicializar pipeline HiPPO-KAN
    kan_pipeline.reset();
    printf("========================================\n");
    printf("LEVITADOR SENSORLESS V4 - HiPPO-KAN\n");
    printf("========================================\n");
    printf("Port: %s\n", portName);
    if (u_ol_enabled) {
        printf("U-OL: ON (offset=%.2fV, amp=%.2fV, f=%.2fHz)\n", u_ol_offset, u_ol_amp, u_ol_freq);
    }
    printf("Modo: %s\n", 
           operation_mode == MODE_MAESTRO ? "MAESTRO (100%% sensor)" :
           operation_mode == MODE_HIBRIDO ? "HIBRIDO (fusion)" :
           "SENSORLESS (100%% KAN)");
    printf("Fusion: %.0f%% sensor / %.0f%% KAN\n", 
           fusion_alpha * 100, (1-fusion_alpha) * 100);
    printf("========================================\n");
    
    if((fp=fopen("MONIT_KAN.txt","w+"))==NULL) {
        printf("No se puede abrir archivo.\n");
        exit(1);
    }
    
    fp_train = fopen("KAN_VALIDATION.csv", "w+");
    if(fp_train) {
        fprintf(fp_train, "t,y_sensor,y_kan,phi_est,i,u,error_mm\n");
        printf("[LOG] KAN_VALIDATION.csv creado\n");
    }

    fp_profile = fopen("PROFILE_TS.csv", "w+");
    if (fp_profile) {
        setvbuf(fp_profile, NULL, _IOFBF, 1 << 20);
        fprintf(fp_profile, "t,frame_dt_us,cycle_total_us,cycle_compute_us,infer_us,log_us\n");
        printf("[LOG] PROFILE_TS.csv creado\n");
    }
    
#ifdef _WIN32
    h = CreateFile(portName, GENERIC_READ|GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);
    
    if(h == INVALID_HANDLE_VALUE) {
        printf("Error abriendo %s (Code: %d)\n", portName, GetLastError());
        return 1;
    }
#else
    h = open(portName, O_RDWR | O_NOCTTY | O_SYNC);
    if (h < 0) {
        printf("Error abriendo %s (Code: %d)\n", portName, errno);
        return 1;
    }
    if (tcgetattr(h, &tty) != 0) {
        printf("Error abriendo %s (Code: %d)\n", portName, errno);
        return 1;
    }
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 1;
    tty.c_cc[VTIME] = 1;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;
    if (tcsetattr(h, TCSANOW, &tty) != 0) {
        printf("Error abriendo %s (Code: %d)\n", portName, errno);
        return 1;
    }
#endif
    
    // Initialize UDP after opening Serial to ensure system is ready
    init_udp();
    
#ifdef _WIN32
    GetCommState(h, &dcb);
    dcb.BaudRate = 115200;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fParity = TRUE;
    SetCommState(h, &dcb);
#endif

#ifdef _WIN32
    DWORD n;
#else
    ssize_t n;
#endif
    char enviar;
    unsigned char recibido_val;

    setvbuf(stdout, NULL, _IONBF, 0);
#ifdef _WIN32
    SetCommMask(h, EV_RXCHAR);
#endif
    
    printf("\nTeclas: 1-5=Setpoint | M/H/S=Modo | +/-=Fusion | D=Din | Q=Salir\n\n");
    printf("Ref y(t): G=ON/OFF | [ ]=Amp -/+ | { }=Freq -/+\n");
    printf("U-OL u(t): U=ON/OFF | J/K=Amp -/+ | ,/.=Freq -/+ | O/P=Offset -/+\n\n");
    printf("Esperando sincronizacion...\n");
    
    int sincronizado = 0;
    time_t last_serial_rx = time(NULL);
    time_t last_no_serial_warn = 0;
    
    while(1) {
        recibido_val = 0;
        
        while(1) {
            
#ifdef _WIN32
            ReadFile(h, &recibido_val, 1, &n, NULL);
            if(!n) break;
#else
            n = read(h, &recibido_val, 1);
            if(n <= 0) break;
#endif
            else {
                last_serial_rx = time(NULL);
                if(flagcom != 0) flagcom++;
                
                if((recibido_val == 0xAA) && (flagcom == 0)) {
                    pv = 0;
                    i_raw = 0;
                    flagcom = 1;
                    if(!sincronizado) {
                        printf("SINCRONIZADO!\n");
                        printf("t\tRef\tSensor\tKAN\tPhi\tErr\tMode\tAlpha\tU\tI\tUz\tIz\n");
                        sincronizado = 1;
                    }
                }
                
                if(flagcom == 2) { pv = recibido_val; pv = pv << 8; }
                if(flagcom == 3) { pv = pv + recibido_val; }
                if(flagcom == 4) { i_raw = recibido_val; i_raw = i_raw << 8; }
                
                if(flagcom == 5) {
                    #if ENABLE_PROFILING && !defined(_WIN32)
                    uint64_t t_cycle_start_ns = now_ns_monotonic_raw();
                    static uint64_t t_cycle_prev_start_ns = 0;
                    uint64_t frame_dt_ns = (t_cycle_prev_start_ns == 0) ? 0 : (t_cycle_start_ns - t_cycle_prev_start_ns);
                    t_cycle_prev_start_ns = t_cycle_start_ns;
                    uint64_t t_infer_start_ns = 0;
                    uint64_t t_infer_end_ns = 0;
                    uint64_t t_compute_end_ns = 0;
                    uint64_t t_cycle_end_ns = 0;
                    #endif

                    yd = compute_yd(t);
                    ef_1 = ef;
                    y_1 = y;
                    
                    i_raw = i_raw + recibido_val;
                    pv &= 0x03FF;
                    i_raw &= 0x03FF;
                    y = esc * pv;
                    ie = esci * i_raw;
                    
                    if (u_ol_enabled) {
                        // LAZO ABIERTO: u(t) senoidal (actualiza cada 10ms)
                        u = compute_u_ol(t);
                        // Actualizar estimador con el u real aplicado (para telemetría)
                        #if ENABLE_PROFILING && !defined(_WIN32)
                        t_infer_start_ns = now_ns_monotonic_raw();
                        #endif
                        y_est_final = estimar_posicion_kan(u, ie, y, t);
                        #if ENABLE_PROFILING && !defined(_WIN32)
                        t_infer_end_ns = now_ns_monotonic_raw();
                        #endif
                    } else {
                        // ESTIMACIÓN HiPPO-KAN
                        #if ENABLE_PROFILING && !defined(_WIN32)
                        t_infer_start_ns = now_ns_monotonic_raw();
                        #endif
                        y_est_final = estimar_posicion_kan(u, ie, y, t);
                        #if ENABLE_PROFILING && !defined(_WIN32)
                        t_infer_end_ns = now_ns_monotonic_raw();
                        #endif
                        
                        // CONTROL PID
                        ef = yd - y_est_final;
                        
                        proporcional = kp * ef;
                        derivativa = kd * (ef - ef_1) * iTs;
                        
                        if((integral < Iref) && (integral > (-Iref)))
                            integral = integral + ki * Ts * ef;
                        else {
                            if(integral >= Iref) integral = 0.95 * Iref;
                            if(integral <= (-Iref)) integral = -0.95 * Iref;
                        }
                        
                        id = proporcional + integral + derivativa;
                        if(id > 0) id = 0;
                        if(id <= -Iref) id = -Iref;
                        
                        ied = -id;
                        ei = ied - ie;
                        propi = kpi * ei;
                        
                        if((intei < Vref) && (intei > -Vref))
                            intei = intei + kii * Ts * ei;
                        else {
                            if(intei >= Vref) intei = 0.95 * Vref;
                            if(intei <= (-Vref)) intei = -0.95 * Vref;
                        }
                        
                        u = propi + intei;
                        if(u > Vref) u = Vref;
                        if(u <= 0) u = 0;
                    }
                    
                    pwmf = u;
                    pwmf = escs * pwmf;
                    pwm = (unsigned char)fabs(pwmf);
                    
                    enviar = pwm;
#ifdef _WIN32
                    WriteFile(h, &enviar, 1, &n, NULL);
#else
                    write(h, &enviar, 1);
#endif
                    
                    float error_mm = (y_kan_est - y) * 1000.0f;
                    
                    // Enviar Telemetría UDP
                    send_telemetry(t, yd, y, y_est_final, phi_est, ie, u, error_mm, operation_mode, fusion_alpha);
                    
                    // Actualizar fusión dinámica basada en error de control
                    float error_control_mm = (yd - y_est_final) * 1000.0f;
                    if (ood_active) {
                        fusion_alpha = 1.0f;
                        contador_estable = 0;
                        estado_estable = false;
                    }
                    actualizar_fusion_dinamica(error_control_mm);
                    
                    #if ENABLE_PROFILING && !defined(_WIN32)
                    // Fin de cómputo “duro” del ciclo (control + salida + telemetría + fusión)
                    // Todo lo que sigue (printf/fprintf/check_input/profile write) se considera "logging/overhead".
                    t_compute_end_ns = now_ns_monotonic_raw();
                    #endif
                    
                    // Imprimir cada 0.5s (evita problemas de redondeo con modulo)
                    static float last_print_t = -1e9f;
                    if ((t - last_print_t) >= 0.5f) {
                        last_print_t = t;
                        const char* mode_str = 
                            operation_mode == MODE_MAESTRO ? "M" :
                            operation_mode == MODE_HIBRIDO ? "H" : "S";
                        float u_z = (u - SensorlessHiPPO::U_MEAN) / SensorlessHiPPO::U_STD;
                        float i_z = (ie - SensorlessHiPPO::I_MEAN) / SensorlessHiPPO::I_STD;
                        if (u_z > SensorlessHiPPO::Z_CLAMP) u_z = SensorlessHiPPO::Z_CLAMP;
                        if (u_z < -SensorlessHiPPO::Z_CLAMP) u_z = -SensorlessHiPPO::Z_CLAMP;
                        if (i_z > SensorlessHiPPO::Z_CLAMP) i_z = SensorlessHiPPO::Z_CLAMP;
                        if (i_z < -SensorlessHiPPO::Z_CLAMP) i_z = -SensorlessHiPPO::Z_CLAMP;
                        printf("%.1f\t%.3f\t%.3f\t%.3f\t%.4f\t%+.1f\t%s\t%.0f%%\t%.2f\t%.3f\t%+.2f\t%+.2f\n", 
                               t, yd*1000, y*1000, y_kan_est*1000, phi_est, error_mm, mode_str, fusion_alpha*100, u, ie, u_z, i_z);
                        fflush(stdout);
                    }
                    
                    fprintf(fp, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                            t, yd, y, y_est_final, ie, u);
                    
                    if(fp_train) {
                        fprintf(fp_train, "%.4f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\n",
                                t, y, y_kan_est, phi_est, ie, u, error_mm);
                    }
                    
                    // Check input every control cycle for max responsiveness
                    check_input(h, fp);
                    
                    #if ENABLE_PROFILING && !defined(_WIN32)
                    t_cycle_end_ns = now_ns_monotonic_raw();
                    if (fp_profile) {
                        static uint64_t prof_k = 0;
                        prof_k++;
                        if ((PROF_LOG_EVERY <= 1) || (prof_k % (uint64_t)PROF_LOG_EVERY == 0)) {
                            uint64_t cycle_total_ns = t_cycle_end_ns - t_cycle_start_ns;
                            uint64_t cycle_compute_ns = (t_compute_end_ns > 0) ? (t_compute_end_ns - t_cycle_start_ns) : 0;
                            uint64_t infer_ns = (t_infer_end_ns > t_infer_start_ns) ? (t_infer_end_ns - t_infer_start_ns) : 0;
                            uint64_t log_ns = (t_cycle_end_ns > t_compute_end_ns) ? (t_cycle_end_ns - t_compute_end_ns) : 0;

                            fprintf(fp_profile, "%.4f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                                    t,
                                    ns_to_us(frame_dt_ns),
                                    ns_to_us(cycle_total_ns),
                                    ns_to_us(cycle_compute_ns),
                                    ns_to_us(infer_ns),
                                    ns_to_us(log_ns));
                        }
                    }
                    #endif
                    
                    flagcom = 0;
                    t = t + Ts;
                }
            }
        }
        
        // Check input even if no serial data is arriving
        check_input(h, fp);

        // Diagnóstico: si no llega serial, t no avanza y u(t) queda constante
        {
            time_t now_s = time(NULL);
            if (sincronizado && (now_s - last_serial_rx) >= 1) {
                if ((now_s - last_no_serial_warn) >= 1) {
                    printf("[WARN] Sin datos serial del PIC por %lds (t=%.2fs). Revisa modo PC0/serial.\n", (long)(now_s - last_serial_rx), t);
                    fflush(stdout);
                    last_no_serial_warn = now_s;
                }
            }
        }
    }
    
    fclose(fp);
    return 0;
}

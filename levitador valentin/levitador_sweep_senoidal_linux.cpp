/*
 * LEVITADOR SWEEP SENOIDAL - Versión Linux
 * 
 * ESTRATEGIA:
 * - Ir a setpoints donde el control ES estable (5, 5.5, 6 mm)
 * - Aplicar oscilación senoidal de ±0.5mm alrededor de cada punto
 * - Esto genera datos en zonas cercanas sin perder estabilidad
 * - Frecuencia baja (0.3 Hz) para que el sistema pueda seguir
 * 
 * COBERTURA ESPERADA:
 * - Zona estable 4.5-6.5mm con oscilaciones conservadoras
 * 
 * Tiempo por punto: 30 segundos (~9 ciclos completos)
 * Tiempo total: ~90 segundos + 3s estabilización
 * 
 * USO: ./levitador_sweep_senoidal_linux [puerto]
 *      Ej: ./levitador_sweep_senoidal_linux /dev/ttyUSB0
 */

#include <iostream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/select.h>
#include <sys/time.h>
#include <signal.h>

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

unsigned char flagcom = 0, pwm;
unsigned short int pv, i_raw;
float y, y_1, ef, ef_1 = 0, u, t = 0;
float esc = 0.05f / 1023.0f;
float esci = 5.0f / (Rs * 1023.0f);
float escs = 254.0f / Vref;
float iTs = 1.0f / Ts;
float pwmf;
float yd = 0.005f, proporcional, derivativa = 0, ie, ied, id, ei, propi, intei = 0, integral = 0;

FILE *fp_train = NULL;
volatile bool running = true;

// Signal handler para Ctrl+C
void signal_handler(int sig) {
    printf("\n[SIGNAL] Interrupción recibida, terminando...\n");
    running = false;
}

// Función para verificar si hay tecla disponible (Linux)
int kbhit_linux() {
    struct timeval tv;
    fd_set fds;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &fds);
}

// Configurar terminal para input sin bloqueo
struct termios orig_termios;
void setup_terminal() {
    tcgetattr(STDIN_FILENO, &orig_termios);
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

void restore_terminal() {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}

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

int main(int argc, char *argv[])
{
    const char *port = "/dev/ttyUSB0";
    if (argc > 1) {
        port = argv[1];
    }
    
    // Configurar signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    FILE *fp;
    int fd;
    
    // Abrir archivo de monitoreo
    if ((fp = fopen("MONIT_SINE.txt", "w+")) == NULL) {
        printf("No se puede abrir archivo MONIT_SINE.txt\n");
        return 1;
    }
    
    // Archivo para datos de entrenamiento
    fp_train = fopen("datos_senoidal.txt", "w+");
    if (fp_train) {
        printf("[SWEEP] Archivo datos_senoidal.txt creado\n");
    }
    
    // Abrir puerto serial
    fd = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        printf("Error abriendo %s\n", port);
        fclose(fp);
        if (fp_train) fclose(fp_train);
        return 1;
    }
    
    // Configurar puerto serial
    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    
    if (tcgetattr(fd, &tty) != 0) {
        printf("Error en tcgetattr\n");
        close(fd);
        fclose(fp);
        if (fp_train) fclose(fp_train);
        return 1;
    }
    
    // Configurar baudrate 115200
    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);
    
    // 8N1, no flow control
    tty.c_cflag &= ~PARENB;        // No parity
    tty.c_cflag &= ~CSTOPB;        // 1 stop bit
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;            // 8 bits
    tty.c_cflag &= ~CRTSCTS;       // No hardware flow control
    tty.c_cflag |= CREAD | CLOCAL; // Enable receiver, ignore modem control
    
    // Raw input
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    
    // Raw output
    tty.c_oflag &= ~OPOST;
    
    // Timeouts: non-blocking
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    
    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        printf("Error en tcsetattr\n");
        close(fd);
        fclose(fp);
        if (fp_train) fclose(fp_train);
        return 1;
    }
    
    // Flush buffers
    tcflush(fd, TCIOFLUSH);
    
    // Configurar terminal para teclado sin bloqueo
    setup_terminal();
    
    printf("========================================\n");
    printf("LEVITADOR SWEEP SENOIDAL (Linux)\n");
    printf("========================================\n");
    printf("Puerto: %s\n", port);
    printf("Estrategia: Oscilaciones pequeñas en zonas estables\n");
    printf("Configuracion:\n");
    printf("  - Amplitud: +/-%.1f mm\n", SINE_AMPLITUDE * 1000);
    printf("  - Frecuencia: %.1f Hz\n", SINE_FREQ);
    printf("  - Puntos base: ");
    for (int s = 0; s < NUM_BASES; s++) printf("%.1f ", base_positions[s] * 1000);
    printf("mm\n");
    printf("  - Tiempo por punto: %.0f segundos\n", TIME_PER_BASE);
    printf("  - Tiempo total: ~%.0f segundos\n", 
           SWEEP_START_TIME + NUM_BASES * TIME_PER_BASE);
    printf("========================================\n");
    printf("Presiona cualquier tecla para interrumpir\n\n");
    fflush(stdout);
    
    unsigned char recibido;
    ssize_t n;
    
    while (running) {
        // Leer bytes del serial
        n = read(fd, &recibido, 1);
        
        if (n > 0) {
            if (flagcom != 0)
                flagcom++;
            
            if ((recibido == 0xAA) && (flagcom == 0)) {
                pv = 0;
                i_raw = 0;
                flagcom = 1;
            }
            if (flagcom == 2) {
                pv = recibido;
                pv = pv << 8;
            }
            if (flagcom == 3) {
                pv = pv + recibido;
            }
            if (flagcom == 4) {
                i_raw = recibido;
                i_raw = i_raw << 8;
            }
            if (flagcom == 5) {
                // Calcular setpoint senoidal
                yd = get_sine_setpoint(t);

                ef_1 = ef;
                y_1 = y;

                i_raw = i_raw + recibido;
                y = esc * pv;
                ie = esci * i_raw;
                
                // PID CON COMPENSACIÓN DE GRAVEDAD
                ef = y - yd;
                
                proporcional = kp * ef;
                derivativa = kd * (ef - ef_1) * iTs;
                if ((integral < Iref) && (integral > (-Iref)))
                    integral = integral + ki * Ts * ef;
                else {
                    if (integral >= Iref) integral = 0.95f * Iref;
                    if (integral <= (-Iref)) integral = -0.95f * Iref;
                }
                // Agregar compensación de gravedad
                id = proporcional + integral + derivativa + GRAVITY_COMP_INTEGRAL;
                
                if (id > 0) id = 0;
                if (id <= -Iref) id = -Iref;
                
                ied = -id;
                ei = ied - ie;
                propi = kpi * ei;
                
                if ((intei < Vref) && (intei > -Vref))
                    intei = intei + kii * Ts * ei;
                else {
                    if (intei >= Vref) intei = 0.95f * Vref;
                    if (intei <= (-Vref)) intei = -0.95f * Vref;
                }
                u = propi + intei;
                
                if (u > Vref) u = Vref;
                if (u <= 0) u = 0;
                
                pwmf = u;
                pwmf = escs * pwmf;
                pwm = (unsigned char)fabs(pwmf);

                // Enviar PWM al micro
                write(fd, &pwm, 1);
                
                // Imprimir progreso cada segundo
                if ((int)(t * 100) % 100 == 0) {
                    printf("t:%.0f base:%.1fmm yd:%.2fmm y:%.2fmm ie:%.3fA u:%.2fV\n", 
                           t, base_positions[current_base] * 1000, 
                           yd * 1000, y * 1000, ie, u);
                    fflush(stdout);
                }
                
                // Guardar TODOS los datos (formato compatible)
                if (fp_train && t >= SWEEP_START_TIME) {
                    fprintf(fp_train, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", 
                            t, yd, y, ied, ie, u);
                }
                
                fprintf(fp, "%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n", t, yd, y, ied, ie, u);
                
                flagcom = 0;
                t = t + Ts;
                
                // Verificar si terminó
                if (sweep_done) {
                    printf("\n========================================\n");
                    printf("BARRIDO SENOIDAL COMPLETADO!\n");
                    printf("Datos guardados: datos_senoidal.txt\n");
                    printf("Tiempo total: %.1f segundos\n", t);
                    printf("========================================\n");
                    fflush(stdout);
                    break;
                }
            }
        }
        
        // Verificar teclado
        if (kbhit_linux()) {
            char c;
            read(STDIN_FILENO, &c, 1);
            printf("\nInterrumpido por usuario (tecla '%c').\n", c);
            break;
        }
        
        // Pequeño sleep para no saturar CPU cuando no hay datos
        usleep(100);  // 100us
    }
    
    // Cleanup
    restore_terminal();
    fclose(fp);
    if (fp_train) fclose(fp_train);
    close(fd);
    
    printf("Programa terminado.\n");
    return 0;
}

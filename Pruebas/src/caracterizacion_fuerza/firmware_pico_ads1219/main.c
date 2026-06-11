/*
 * Firmware Pico — Adquisición de fuerza con ADS1219 @ 1000 SPS, streaming binario por USB CDC.
 *
 * Arquitectura (dos cores, bare-metal, sin FreeRTOS):
 *   - core1: lee el ADS1219 (I2C, maestro local) a la tasa de DRDY (1000 SPS) y mete
 *            cada muestra (raw 24-bit + timestamp us del MCU) en una cola SPSC.
 *   - core0: vacía la cola en bloques y los envía por USB CDC en binario. Atiende
 *            comandos de 1 byte del PC (S=start, X=stop, P=ping).
 *
 * Por qué así: el Pico es el maestro I2C, así que NO hay round-trip USB por muestra
 * (lo que topaba al PC+Analog Discovery a ~487 Hz). El muestreo lo marca el reloj del
 * propio ADS1219; el USB sólo transporta bloques ya bufferizados.
 *
 * Cadena: celda DYMH-105 -> INA-4LC (G=601) -> ADS1219 AIN0/AIN1 -> Pico (I2C) -> USB -> PC
 *
 * Protocolo de bloque (little-endian) — lo parsea caracterizacion_fuerza_PicoADS1219.py:
 *   Header (16 bytes):
 *     uint8  magic0 = 0xA5
 *     uint8  magic1 = 0x5A
 *     uint16 n            // muestras en este bloque
 *     uint32 seq          // índice de la PRIMERA muestra del bloque (contador global entregado)
 *     uint32 t_us         // micros() del MCU de la primera muestra del bloque
 *     uint32 dropped      // muestras perdidas acumuladas por overflow de la cola
 *   Payload: n × int32 (LE) = lectura cruda del ADC (24-bit con signo extendido)
 *
 * El PC convierte:  volts = raw / 2^23 * (VREF / GAIN),  VREF = 2.048 V interno.
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "pico/stdlib.h"
#include "pico/stdio.h"
#include "pico/stdio_usb.h"
#include "pico/multicore.h"
#include "pico/util/queue.h"
#include "hardware/i2c.h"
#include "hardware/gpio.h"

/* ---- Configuración (override por -D en CMake) ---- */
#ifndef ADS_I2C_SDA_PIN
#define ADS_I2C_SDA_PIN 8
#endif
#ifndef ADS_I2C_SCL_PIN
#define ADS_I2C_SCL_PIN 9
#endif
#ifndef ADS_DRDY_PIN
#define ADS_DRDY_PIN 7
#endif
#ifndef ADS_USE_DRDY_PIN
#define ADS_USE_DRDY_PIN 0
#endif
#ifndef ADS_GAIN
#define ADS_GAIN 4
#endif

#define ADS_I2C_PORT  i2c0
#define ADS_I2C_HZ    (400 * 1000)
#define ADS_ADDR      0x40         /* 7-bit (SparkFun ADS1219 por defecto) */

/* Config reg0: MUX=AIN0-AIN1(000) | GAIN(bit4) | DR=1000SPS(11) | CM=continuo(1) | VREF=interno(0) */
#define ADS_CONFIG    ((0x00 << 5) | (((ADS_GAIN == 4) ? 1 : 0) << 4) | (0x03 << 2) | (1 << 1) | 0)

/* Comandos ADS1219 */
#define ADS_CMD_RESET     0x06
#define ADS_CMD_START     0x08
#define ADS_CMD_POWERDOWN 0x02
#define ADS_CMD_RDATA     0x10
#define ADS_CMD_RREG_CFG  0x20   /* RREG registro 0 (config) */
#define ADS_CMD_RREG_STAT 0x24   /* RREG registro 1 (status), bit7 = DRDY */

/* ---- Protocolo de bloque ---- */
#define BLK_MAGIC0   0xA5
#define BLK_MAGIC1   0x5A
#define BLK_HDR_SIZE 16
#define BLK_MAX      64           /* muestras máx por bloque */
#define QUEUE_CAP    512          /* ~0.5 s de buffer a 1000 Hz */

typedef struct {
    int32_t  raw;
    uint32_t t_us;
} samp_t;

static queue_t sample_q;
static volatile bool     g_streaming = false;
static volatile uint32_t g_dropped   = 0;
static volatile bool     g_ads_ok    = false;
static volatile bool     g_reinit    = true;   /* pide (re)configurar el ADS1219 en core1 */

/* ====================== ADS1219 (I2C) ====================== */

static bool ads_cmd(uint8_t c) {
    return i2c_write_blocking(ADS_I2C_PORT, ADS_ADDR, &c, 1, false) == 1;
}

static bool ads_wreg_config(uint8_t value) {
    uint8_t b[2] = { 0x40, value };   /* 0x40 = WREG registro 0 */
    return i2c_write_blocking(ADS_I2C_PORT, ADS_ADDR, b, 2, false) == 2;
}

static bool ads_rreg(uint8_t reg_cmd, uint8_t *out) {
    if (i2c_write_blocking(ADS_I2C_PORT, ADS_ADDR, &reg_cmd, 1, true) != 1) return false;
    return i2c_read_blocking(ADS_I2C_PORT, ADS_ADDR, out, 1, false) == 1;
}

static inline bool ads_data_ready(void) {
#if ADS_USE_DRDY_PIN
    return gpio_get(ADS_DRDY_PIN) == 0;   /* /DRDY activo en bajo */
#else
    uint8_t st;
    if (!ads_rreg(ADS_CMD_RREG_STAT, &st)) return false;
    return (st & 0x80) != 0;              /* bit7 = DRDY */
#endif
}

static bool ads_read_raw(int32_t *out) {
    uint8_t cmd = ADS_CMD_RDATA;
    uint8_t rx[3];
    if (i2c_write_blocking(ADS_I2C_PORT, ADS_ADDR, &cmd, 1, true) != 1) return false;
    if (i2c_read_blocking(ADS_I2C_PORT, ADS_ADDR, rx, 3, false) != 3) return false;
    int32_t raw = ((int32_t)rx[0] << 16) | ((int32_t)rx[1] << 8) | (int32_t)rx[2];
    if (raw & 0x800000) raw -= (1 << 24);   /* signo de 24 bits */
    *out = raw;
    return true;
}

static bool ads_init(void) {
    i2c_init(ADS_I2C_PORT, ADS_I2C_HZ);
    gpio_set_function(ADS_I2C_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(ADS_I2C_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(ADS_I2C_SDA_PIN);
    gpio_pull_up(ADS_I2C_SCL_PIN);
#if ADS_USE_DRDY_PIN
    gpio_init(ADS_DRDY_PIN);
    gpio_set_dir(ADS_DRDY_PIN, GPIO_IN);
    gpio_pull_up(ADS_DRDY_PIN);
#endif

    if (!ads_cmd(ADS_CMD_RESET)) return false;
    sleep_ms(2);
    if (!ads_wreg_config((uint8_t)ADS_CONFIG)) return false;

    uint8_t cfg = 0;
    if (!ads_rreg(ADS_CMD_RREG_CFG, &cfg)) return false;
    if (cfg != (uint8_t)ADS_CONFIG) return false;

    if (!ads_cmd(ADS_CMD_START)) return false;   /* conversión continua */

    /* Espera el primer DRDY (máx 2 s) */
    absolute_time_t t0 = get_absolute_time();
    while (!ads_data_ready()) {
        if (absolute_time_diff_us(t0, get_absolute_time()) > 2000000) return false;
        sleep_us(200);
    }
    return true;
}

/* ====================== core1: muestreo ====================== */

static void core1_main(void) {
    int32_t raw;
    for (;;) {
        if (g_reinit) {
            g_ads_ok = ads_init();   /* core1 es el dueño del bus I2C: reconfigura aquí */
            g_reinit = false;
        }
        if (ads_data_ready()) {
            uint32_t t = time_us_32();        /* instante (aprox) en que la muestra quedó lista */
            if (ads_read_raw(&raw)) {         /* leer RDATA limpia DRDY hasta la próxima conversión */
                if (g_streaming) {
                    samp_t s = { raw, t };
                    if (!queue_try_add(&sample_q, &s)) {
                        g_dropped++;          /* core0 no alcanzó a vaciar */
                    }
                }
            }
        } else {
#if !ADS_USE_DRDY_PIN
            busy_wait_us(60);                 /* evita martillar el bus mientras no hay dato */
#else
            tight_loop_contents();
#endif
        }
    }
}

/* ====================== core0: USB + envío ====================== */

static void send_status_line(void) {
    /* Sólo texto y sólo cuando NO se está enviando binario, para no romper el parser */
    char line[96];
    int len = snprintf(line, sizeof(line),
                       "INFO ads_ok=%d gain=%d config=0x%02X sps=1000 dropped=%lu\n",
                       g_ads_ok ? 1 : 0, ADS_GAIN, (unsigned)(uint8_t)ADS_CONFIG,
                       (unsigned long)g_dropped);
    if (len > 0) {
        fwrite(line, 1, (size_t)len, stdout);
        fflush(stdout);
    }
}

int main(void) {
    stdio_init_all();                 /* USB CDC */
    setvbuf(stdout, NULL, _IONBF, 0); /* sin buffer de stdio */
    stdio_set_translate_crlf(&stdio_usb, false);  /* binario: NO traducir 0x0A->CRLF */
    sleep_ms(100);

    queue_init(&sample_q, sizeof(samp_t), QUEUE_CAP);

    /* La (re)inicialización del ADS1219 la hace core1 (dueño del bus I2C) al ver g_reinit
       (true al arranque y en cada 'S'), así el orden de conexión del sensor no importa. */
    multicore_launch_core1(core1_main);

    uint32_t seq = 0;
    static uint8_t block[BLK_HDR_SIZE + 4 * BLK_MAX];
    samp_t s;

    for (;;) {
        /* --- comandos del PC (1 byte) --- */
        int c = getchar_timeout_us(0);
        while (c != PICO_ERROR_TIMEOUT) {
            switch (c) {
                case 'S': case 's':
                    g_reinit = true;          /* re-sondea/reconfigura el ADS1219 */
                    g_dropped = 0;
                    /* descarta backlog viejo antes de empezar */
                    while (queue_try_remove(&sample_q, &s)) { }
                    seq = 0;
                    g_streaming = true;
                    break;
                case 'X': case 'x':
                    g_streaming = false;
                    break;
                case 'P': case 'p':
                    if (!g_streaming) { fwrite("PONG\n", 1, 5, stdout); fflush(stdout); }
                    break;
                case 'I': case 'i':
                    if (!g_streaming) send_status_line();
                    break;
                default:
                    break;
            }
            c = getchar_timeout_us(0);
        }

        /* --- arma y envía un bloque --- */
        if (g_streaming) {
            uint16_t n = 0;
            uint32_t t_first = 0;
            while (n < BLK_MAX && queue_try_remove(&sample_q, &s)) {
                if (n == 0) t_first = s.t_us;
                memcpy(block + BLK_HDR_SIZE + 4 * n, &s.raw, 4);  /* RP2040 es LE */
                n++;
            }
            if (n > 0) {
                uint32_t dropped = g_dropped;
                uint16_t n16 = n;
                block[0] = BLK_MAGIC0;
                block[1] = BLK_MAGIC1;
                memcpy(block + 2,  &n16,     2);
                memcpy(block + 4,  &seq,     4);
                memcpy(block + 8,  &t_first, 4);
                memcpy(block + 12, &dropped, 4);
                fwrite(block, 1, (size_t)(BLK_HDR_SIZE + 4 * n), stdout);
                fflush(stdout);
                seq += n;
            }
        }

        sleep_ms(5);   /* deja acumular ~5 muestras/bloque; latencia ~5 ms */
    }
}

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

static int open_serial(const char* port) {
    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "Error abriendo %s (errno=%d)\n", port, errno);
        return -1;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd, &tty) != 0) {
        fprintf(stderr, "tcgetattr falló (errno=%d)\n", errno);
        close(fd);
        return -1;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        fprintf(stderr, "tcsetattr falló (errno=%d)\n", errno);
        close(fd);
        return -1;
    }

    tcflush(fd, TCIOFLUSH);
    return fd;
}

static int read_byte_timeout(int fd, uint8_t* out, int timeout_ms) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;

    int sel = select(fd + 1, &rfds, NULL, NULL, &tv);
    if (sel <= 0) return 0;

    uint8_t b = 0;
    ssize_t n = read(fd, &b, 1);
    if (n <= 0) return 0;

    *out = b;
    return 1;
}

static void usage(const char* argv0) {
    fprintf(stderr,
            "Uso: %s --port /dev/ttyUSB1 --out out.csv [--seconds 10] [--pwm 0..255] [--pwm-sine amp freq_hz offset] [--print] [--print-every N] [--no-csv]\n",
            argv0);
}

int main(int argc, char** argv) {
    const char* port = "/dev/ttyUSB0";
    const char* out_path = "serial_log.csv";
    double seconds = 10.0;

    int print_enabled = 0;
    int print_every = 1;
    int csv_enabled = 1;

    int pwm_const_enabled = 0;
    int pwm_const = 0;

    int pwm_sine_enabled = 0;
    double pwm_sine_amp = 0.0;
    double pwm_sine_freq = 1.0;
    double pwm_sine_offset = 0.0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
            seconds = atof(argv[++i]);
        } else if (strcmp(argv[i], "--print") == 0) {
            print_enabled = 1;
        } else if (strcmp(argv[i], "--print-every") == 0 && i + 1 < argc) {
            print_every = atoi(argv[++i]);
            if (print_every < 1) print_every = 1;
        } else if (strcmp(argv[i], "--no-csv") == 0) {
            csv_enabled = 0;
        } else if (strcmp(argv[i], "--pwm") == 0 && i + 1 < argc) {
            pwm_const_enabled = 1;
            pwm_const = atoi(argv[++i]);
            if (pwm_const < 0) pwm_const = 0;
            if (pwm_const > 255) pwm_const = 255;
        } else if (strcmp(argv[i], "--pwm-sine") == 0 && i + 3 < argc) {
            pwm_sine_enabled = 1;
            pwm_sine_amp = atof(argv[++i]);
            pwm_sine_freq = atof(argv[++i]);
            pwm_sine_offset = atof(argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    int fd = open_serial(port);
    if (fd < 0) return 1;

    FILE* fp = NULL;
    if (csv_enabled) {
        fp = fopen(out_path, "w");
        if (!fp) {
            fprintf(stderr, "No se pudo abrir %s\n", out_path);
            close(fd);
            return 1;
        }
        fprintf(fp, "t_host_s,pv_raw,i_raw,y_m,i_a,pwm_sent\n");
        fflush(fp);
    } else if (!print_enabled) {
        print_enabled = 1;
    }

    if (print_enabled) {
        printf("t_host_s\tpv_raw\ti_raw\ty_mm\ti_a\tpwm\n");
        fflush(stdout);
    }

    const double Rs = 2.20;
    const double Vref = 9.86;
    (void)Vref;

    double t0 = (double)clock() / (double)CLOCKS_PER_SEC;
    double t_end = t0 + seconds;

    int synced = 0;
    int frame_k = 0;

    while (((double)clock() / (double)CLOCKS_PER_SEC) < t_end) {
        uint8_t b = 0;
        if (!read_byte_timeout(fd, &b, 50)) {
            continue;
        }

        if (!synced) {
            if (b == 0xAA) {
                synced = 1;
                frame_k = 0;
            }
            continue;
        }

        uint8_t buf[4];
        buf[0] = 0;
        buf[1] = 0;
        buf[2] = 0;
        buf[3] = 0;

        int got = 0;
        while (got < 4) {
            uint8_t bb = 0;
            if (!read_byte_timeout(fd, &bb, 100)) {
                synced = 0;
                break;
            }
            buf[got++] = bb;
        }
        if (!synced) continue;

        uint16_t pv = (uint16_t)(((uint16_t)buf[0] << 8) | (uint16_t)buf[1]);
        uint16_t ir = (uint16_t)(((uint16_t)buf[2] << 8) | (uint16_t)buf[3]);
        pv &= 0x03FF;
        ir &= 0x03FF;

        double t_now = (double)clock() / (double)CLOCKS_PER_SEC;
        double t_rel = t_now - t0;

        int pwm_to_send = 0;
        if (pwm_sine_enabled) {
            double x = pwm_sine_offset + pwm_sine_amp * sin(2.0 * M_PI * pwm_sine_freq * t_rel);
            if (x < 0.0) x = 0.0;
            if (x > 255.0) x = 255.0;
            pwm_to_send = (int)lrint(x);
        } else if (pwm_const_enabled) {
            pwm_to_send = pwm_const;
        }

        if (pwm_const_enabled || pwm_sine_enabled) {
            uint8_t outb = (uint8_t)pwm_to_send;
            (void)write(fd, &outb, 1);
        }

        double y_m = 0.05 * ((double)pv / 1023.0);
        double i_a = (5.0 / (Rs * 1023.0)) * (double)ir;

        if (fp) {
            fprintf(fp, "%.6f,%u,%u,%.8f,%.6f,%d\n", t_rel, (unsigned)pv, (unsigned)ir, y_m, i_a, pwm_to_send);
        }

        if (print_enabled && ((frame_k % print_every) == 0)) {
            printf("%.3f\t%u\t%u\t%.3f\t%.4f\t%d\n", t_rel, (unsigned)pv, (unsigned)ir, y_m * 1000.0, i_a, pwm_to_send);
            fflush(stdout);
        }

        frame_k++;
        if (fp && (frame_k % 50) == 0) {
            fflush(fp);
        }
    }

    if (fp) {
        fflush(fp);
        fclose(fp);
    }
    close(fd);
    return 0;
}

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

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

extern "C" {
typedef int BOOL;
typedef int HDWF;

BOOL FDwfGetLastErrorMsg(char *szErrorMsg);
BOOL FDwfEnum(int enumfilter, int *pcDevices);
BOOL FDwfDeviceOpen(int idx, HDWF *phdwf);
BOOL FDwfDeviceClose(HDWF hdwf);
BOOL FDwfDeviceAutoConfigureSet(HDWF hdwf, int autoConfigure);
BOOL FDwfAnalogInReset(HDWF hdwf);
BOOL FDwfAnalogInAcquisitionModeSet(HDWF hdwf, int mode);
BOOL FDwfAnalogInRecordLengthSet(HDWF hdwf, double length);
BOOL FDwfAnalogInFrequencySet(HDWF hdwf, double hz);
BOOL FDwfAnalogInChannelEnableSet(HDWF hdwf, int idx, BOOL enable);
BOOL FDwfAnalogInChannelRangeSet(HDWF hdwf, int idx, double range);
BOOL FDwfAnalogInChannelOffsetSet(HDWF hdwf, int idx, double offset);
BOOL FDwfAnalogInConfigure(HDWF hdwf, BOOL fReconfigure, BOOL fStart);
BOOL FDwfAnalogInStatus(HDWF hdwf, BOOL fReadData, unsigned char *pst);
BOOL FDwfAnalogInStatusRecord(HDWF hdwf, int *pcAvailable, int *pcLost, int *pcCorrupt);
BOOL FDwfAnalogInStatusData(HDWF hdwf, int idx, double *rgdSamples, int cSamples);
}

static void dwf_check(BOOL ok, const char *where) {
    if (ok) return;
    char msg[512];
    memset(msg, 0, sizeof(msg));
    FDwfGetLastErrorMsg(msg);
    fprintf(stderr, "[DWF] %s failed: %s\n", where, msg);
    exit(1);
}

static double now_s() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int open_serial(const char *port) {
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

static int read_byte_timeout(int fd, uint8_t *out, int timeout_ms) {
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

struct Ad1Config {
    int device_index = 0;
    int channel = 1; // 1-based
    double sample_rate_hz = 2500.0;
    double volts_range = 5.0;
    double volts_offset = 0.0;
};

static HDWF configure_ad1(const Ad1Config &cfg) {
    int device_count = 0;
    dwf_check(FDwfEnum(0, &device_count), "FDwfEnum");
    if (cfg.device_index >= device_count) {
        fprintf(stderr, "[DWF] device_index %d fuera de rango (found %d)\n", cfg.device_index, device_count);
        exit(1);
    }

    HDWF hdwf = 0;
    dwf_check(FDwfDeviceOpen(cfg.device_index, &hdwf), "FDwfDeviceOpen");
    dwf_check(FDwfDeviceAutoConfigureSet(hdwf, 0), "FDwfDeviceAutoConfigureSet");

    const int acqmode_record = 3;
    dwf_check(FDwfAnalogInReset(hdwf), "FDwfAnalogInReset");
    dwf_check(FDwfAnalogInAcquisitionModeSet(hdwf, acqmode_record), "FDwfAnalogInAcquisitionModeSet");
    dwf_check(FDwfAnalogInRecordLengthSet(hdwf, 0.0), "FDwfAnalogInRecordLengthSet");
    dwf_check(FDwfAnalogInFrequencySet(hdwf, cfg.sample_rate_hz), "FDwfAnalogInFrequencySet");

    for (int ch = 0; ch < 2; ++ch) {
        int enable = (ch == (cfg.channel - 1)) ? 1 : 0;
        dwf_check(FDwfAnalogInChannelEnableSet(hdwf, ch, enable), "FDwfAnalogInChannelEnableSet");
    }

    int idx = cfg.channel - 1;
    dwf_check(FDwfAnalogInChannelRangeSet(hdwf, idx, cfg.volts_range), "FDwfAnalogInChannelRangeSet");
    dwf_check(FDwfAnalogInChannelOffsetSet(hdwf, idx, cfg.volts_offset), "FDwfAnalogInChannelOffsetSet");
    dwf_check(FDwfAnalogInConfigure(hdwf, 1, 1), "FDwfAnalogInConfigure");
    return hdwf;
}

static bool read_ad1_chunk(HDWF hdwf, int channel_idx, std::vector<double> *out) {
    if (!out) return false;
    unsigned char st = 0;
    int available = 0;
    int lost = 0;
    int corrupt = 0;

    dwf_check(FDwfAnalogInStatus(hdwf, 1, &st), "FDwfAnalogInStatus");
    dwf_check(FDwfAnalogInStatusRecord(hdwf, &available, &lost, &corrupt), "FDwfAnalogInStatusRecord");

    if (available <= 0) {
        return false;
    }

    out->assign((size_t)available, 0.0);
    dwf_check(FDwfAnalogInStatusData(hdwf, channel_idx, out->data(), available), "FDwfAnalogInStatusData");
    return true;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "Uso: %s --port /dev/ttyUSB1 --out out.csv [--seconds 20] [--amp 4.0] [--offset 4.5] [--freq 100]\n"
            "          [--ad1-channel 1] [--ad1-fs 2500] [--ad1-range 5.0] [--print-every 10]\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *port = "/dev/ttyUSB1";
    const char *out_path = "uol_ad1_monitor.csv";
    double seconds = 20.0;
    double amp_v = 4.0;
    double offset_v = 4.5;
    double freq_hz = 100.0;
    int print_every = 1;

    Ad1Config ad1_cfg;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = argv[++i];
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_path = argv[++i];
        } else if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) {
            seconds = atof(argv[++i]);
        } else if (strcmp(argv[i], "--amp") == 0 && i + 1 < argc) {
            amp_v = atof(argv[++i]);
        } else if (strcmp(argv[i], "--offset") == 0 && i + 1 < argc) {
            offset_v = atof(argv[++i]);
        } else if (strcmp(argv[i], "--freq") == 0 && i + 1 < argc) {
            freq_hz = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ad1-channel") == 0 && i + 1 < argc) {
            ad1_cfg.channel = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ad1-fs") == 0 && i + 1 < argc) {
            ad1_cfg.sample_rate_hz = atof(argv[++i]);
        } else if (strcmp(argv[i], "--ad1-range") == 0 && i + 1 < argc) {
            ad1_cfg.volts_range = atof(argv[++i]);
        } else if (strcmp(argv[i], "--print-every") == 0 && i + 1 < argc) {
            print_every = atoi(argv[++i]);
            if (print_every < 1) print_every = 1;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    int fd = open_serial(port);
    if (fd < 0) return 1;

    HDWF hdwf = configure_ad1(ad1_cfg);
    int ad1_idx = ad1_cfg.channel - 1;

    FILE *fp = fopen(out_path, "w");
    if (!fp) {
        fprintf(stderr, "No se pudo abrir %s\n", out_path);
        close(fd);
        FDwfDeviceClose(hdwf);
        return 1;
    }

    fprintf(fp,
            "t_host_s,t_ad_s,u_cmd_v,pwm_cmd,pv_raw,i_raw,y_mm,i_a,ad1_mean_v\n");
    fflush(fp);

    printf("[UOL] port=%s amp=%.2fV offset=%.2fV freq=%.2fHz seconds=%.1f\n", port, amp_v, offset_v,
           freq_hz, seconds);
    printf("[AD1] ch=%d fs=%.0fHz range=%.1fV\n", ad1_cfg.channel, ad1_cfg.sample_rate_hz, ad1_cfg.volts_range);
    printf("t\tu_cmd\tpwm\ty_mm\ti_a\tAD1_mean\tAD1_amp\n");

    const double Vref = 9.86;
    const double pwm_scale = 25.76; // mismo escs del PIC
    const double Rs = 2.20;

    double t0 = now_s();
    double t_end = t0 + seconds;

    bool synced = false;
    int frame_k = 0;
    double last_print = t0;

    double ad1_min = std::numeric_limits<double>::infinity();
    double ad1_max = -std::numeric_limits<double>::infinity();
    double ad1_mean = std::numeric_limits<double>::quiet_NaN();
    double ad1_t = std::numeric_limits<double>::quiet_NaN();
    size_t ad1_sample_index = 0;
    double y_sum_mm = 0.0;
    size_t y_count = 0;

    while (now_s() < t_end) {
        std::vector<double> chunk;
        if (read_ad1_chunk(hdwf, ad1_idx, &chunk)) {
            double sum = 0.0;
            for (double v : chunk) sum += v;
            ad1_mean = sum / (double)chunk.size();
            ad1_sample_index += chunk.size();
            ad1_t = ((double)ad1_sample_index - (double)chunk.size() / 2.0) / ad1_cfg.sample_rate_hz;
            ad1_min = std::min(ad1_min, ad1_mean);
            ad1_max = std::max(ad1_max, ad1_mean);
        }

        uint8_t b = 0;
        if (!read_byte_timeout(fd, &b, 5)) {
            continue;
        }

        if (!synced) {
            if (b == 0xAA) {
                synced = true;
            }
            continue;
        }

        uint8_t buf[4];
        int got = 0;
        while (got < 4) {
            uint8_t bb = 0;
            if (!read_byte_timeout(fd, &bb, 10)) {
                synced = false;
                break;
            }
            buf[got++] = bb;
        }
        if (!synced) continue;

        uint16_t pv = (uint16_t)(((uint16_t)buf[0] << 8) | (uint16_t)buf[1]);
        uint16_t ir = (uint16_t)(((uint16_t)buf[2] << 8) | (uint16_t)buf[3]);
        pv &= 0x03FF;
        ir &= 0x03FF;

        double t_now = now_s();
        double t_rel = t_now - t0;

        double u_cmd = offset_v + amp_v * sin(2.0 * M_PI * freq_hz * t_rel);
        if (u_cmd < 0.0) u_cmd = 0.0;
        if (u_cmd > Vref) u_cmd = Vref;

        int pwm = (int)lrint(u_cmd * pwm_scale);
        if (pwm < 0) pwm = 0;
        if (pwm > 255) pwm = 255;

        uint8_t outb = (uint8_t)pwm;
        (void)write(fd, &outb, 1);

        double y_m = 0.05 * ((double)pv / 1023.0);
        double i_a = (5.0 / (Rs * 1023.0)) * (double)ir;
        double y_mm = y_m * 1000.0;
        if (std::isfinite(y_mm)) {
            y_sum_mm += y_mm;
            y_count++;
        }

        fprintf(fp, "%.6f,%.6f,%.4f,%d,%u,%u,%.6f,%.6f,%.6f\n", t_rel, ad1_t, u_cmd, pwm,
                (unsigned)pv, (unsigned)ir, y_mm, i_a, ad1_mean);

        if ((frame_k % print_every) == 0) {
            double ad1_amp = 0.0;
            if (std::isfinite(ad1_min) && std::isfinite(ad1_max)) {
                ad1_amp = 0.5 * (ad1_max - ad1_min);
            }
            if ((t_now - last_print) >= 0.5) {
                printf("%.2f\t%.3f\t%d\t%.3f\t%.4f\t%.4f\t%.4f\n", t_rel, u_cmd, pwm, y_mm,
                       i_a, ad1_mean, ad1_amp);
                fflush(stdout);
                ad1_min = std::numeric_limits<double>::infinity();
                ad1_max = -std::numeric_limits<double>::infinity();
                last_print = t_now;
            }
        }

        frame_k++;
        if ((frame_k % 50) == 0) {
            fflush(fp);
        }
    }

    if (y_count > 0) {
        double y_avg_mm = y_sum_mm / (double)y_count;
        printf("[RESUMEN] y_mm promedio: %.4f (N=%zu)\n", y_avg_mm, y_count);
    } else {
        printf("[RESUMEN] y_mm promedio: -- (sin muestras)\n");
    }

    fflush(fp);
    fclose(fp);
    close(fd);
    FDwfDeviceClose(hdwf);
    return 0;
}

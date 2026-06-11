# Firmware Pico — Fuerza ADS1219 @ 1000 SPS por USB CDC

Lee el **ADS1219** (celda DYMH-105 → INA-4LC G=601) con una **Raspberry Pi Pico / Pico 2 W**
como maestro I2C local y transmite las muestras al PC en **bloques binarios por USB CDC**.

Sustituye al Analog Discovery como maestro I2C: al muestrear localmente (sin round-trip USB
por muestra) alcanza los **1000 SPS reales** del ADS1219, que el camino PC+AD topaba a ~487 Hz.

- **core1**: lee el ADS1219 a la tasa de DRDY y encola cada muestra (raw 24-bit + timestamp µs).
- **core0**: vacía la cola en bloques y los envía por USB; atiende comandos de 1 byte.

GUI del PC que lo consume: [`../caracterizacion_fuerza_PicoADS1219.py`](../caracterizacion_fuerza_PicoADS1219.py)

## Cableado (ADS1219 → Pico)

| ADS1219 | Pico | Nota |
|---|---|---|
| SDA | **GP8** | pull-up 10 kΩ a 3V3 |
| SCL | **GP9** | pull-up 10 kΩ a 3V3 |
| VCC | **3V3 (OUT)** | |
| GND | **GND** | tierra común con el INA |
| /DRDY | GP7 | *opcional*, solo si `ADS_USE_DRDY_PIN=1` |

Por defecto (`ADS_USE_DRDY_PIN=0`) basta con SDA/SCL/3V3/GND: el firmware sondea el registro
STATUS del ADS1219. Es el mismo cableado mínimo que ya usabas con el Analog Discovery.

> **El gain debe coincidir** con el GUI: firmware `ADS_GAIN=4` ⇄ Python `ADS1219_GAIN=4`.

## Compilar

Toolchain ya instalado por la extensión Pico de VS Code (SDK 2.2.0, GCC ARM 14.2, ninja).

**VS Code:** abrir esta carpeta → extensión *Raspberry Pi Pico* → *Compile*.

**Línea de comandos (PowerShell):**
```powershell
$cmake = "$env:USERPROFILE\.pico-sdk\cmake\v3.31.5\bin\cmake.exe"
$ninja = "$env:USERPROFILE\.pico-sdk\ninja\v1.12.1\ninja.exe"
& $cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM="$ninja"
& $cmake --build build
# Para Pico 2 W:  agrega  -DPICO_BOARD=pico2_w  al primer comando
```
Salida: `build/fuerza_pico_ads1219.uf2`

## Flashear

1. Mantén **BOOTSEL** y conecta el Pico por USB → aparece la unidad `RPI-RP2`.
2. Copia `build/fuerza_pico_ads1219.uf2` a esa unidad. El Pico reinicia y arranca el firmware.
3. Aparece un **puerto COM** nuevo (VID 0x2E8A). El GUI lo autodetecta.

## Protocolo (Pico → PC, little-endian)

Cabecera de 16 bytes + payload:
```
uint8  magic0 = 0xA5
uint8  magic1 = 0x5A
uint16 n            # muestras en el bloque
uint32 seq          # índice de la 1ª muestra del bloque (contador entregado)
uint32 t_us         # micros() del MCU de la 1ª muestra del bloque
uint32 dropped      # muestras perdidas acumuladas (overflow de la cola del Pico)
int32  raw[n]       # lectura cruda del ADC (24-bit con signo extendido)
```
Conversión en el PC: `volts = raw / 2^23 * (VREF / GAIN)`, con `VREF = 2.048 V`.

## Comandos (PC → Pico, 1 byte ASCII)

| Byte | Acción |
|---|---|
| `S` | inicia streaming (limpia backlog y contador) |
| `X` | detiene streaming |
| `P` | ping → responde `PONG\n` (solo si no está enviando) |
| `I` | línea de estado de texto (solo si no está enviando) |

## Notas / pendientes

- **DRDY por pin** (`-DADS_USE_DRDY_PIN=1`, GP7) da el muestreo más limpio; el sondeo de
  STATUS es el default por requerir menos cableado.
- El binario va por `stdio_usb` (CDC) con traducción CRLF desactivada. Para entrega 100%
  garantizada bajo carga extrema, migrar a TinyUSB CDC crudo (no necesario a 1000 Hz).
- Sin CRC en el bloque; el parser re-sincroniza por el magic y valida el rango de `n`.
  Si en el banco aparecen huecos, añadir CRC16 al final del bloque.

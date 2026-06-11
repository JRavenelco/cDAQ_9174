# Firmware Pico — adquisición de fuerza ADS1219 @ 1000 SPS por USB CDC (binario)
# Bare-metal multicore (sin FreeRTOS): core1 muestrea el ADS1219, core0 envía bloques.

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================

# pico_w (RP2040) por defecto. Para Pico 2 W: -DPICO_BOARD=pico2_w
set(PICO_BOARD pico_w CACHE STRING "Board type")

include(pico_sdk_import.cmake)

project(fuerza_pico_ads1219 C CXX ASM)

pico_sdk_init()

add_executable(fuerza_pico_ads1219 main.c)

pico_set_program_name(fuerza_pico_ads1219 "fuerza_pico_ads1219")
pico_set_program_version(fuerza_pico_ads1219 "0.1")

# Datos por USB CDC (puerto COM virtual). UART de stdio apagado.
pico_enable_stdio_uart(fuerza_pico_ads1219 0)
pico_enable_stdio_usb(fuerza_pico_ads1219 1)

target_include_directories(fuerza_pico_ads1219 PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Pines y opciones (override con -D... al configurar)
target_compile_definitions(fuerza_pico_ads1219 PRIVATE
        ADS_I2C_SDA_PIN=8        # GP8 -> SDA del ADS1219
        ADS_I2C_SCL_PIN=9        # GP9 -> SCL del ADS1219
        ADS_DRDY_PIN=7           # GP7 -> /DRDY (solo si ADS_USE_DRDY_PIN=1)
        ADS_USE_DRDY_PIN=0       # 0 = sondea el registro STATUS (solo SDA/SCL); 1 = usa pin /DRDY
        ADS_GAIN=4               # 1 o 4 (debe coincidir con el GUI Python)
)

target_link_libraries(fuerza_pico_ads1219
        pico_stdlib
        pico_multicore
        hardware_i2c
)

pico_add_extra_outputs(fuerza_pico_ads1219)

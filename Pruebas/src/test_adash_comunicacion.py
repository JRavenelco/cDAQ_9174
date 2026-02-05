#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para comunicación con ADASH 4900 Vibrio
Detecta puertos COM disponibles e intenta comunicarse
"""

import serial
import serial.tools.list_ports
import time
import sys
import argparse

def listar_puertos_com():
    """Lista todos los puertos COM disponibles"""
    print("=" * 60)
    print("PUERTOS COM DISPONIBLES:")
    print("=" * 60)
    
    puertos = serial.tools.list_ports.comports()
    
    if not puertos:
        print("❌ No se encontraron puertos COM")
        return []
    
    for i, puerto in enumerate(puertos):
        print(f"\n{i+1}. Puerto: {puerto.device}")
        print(f"   Descripción: {puerto.description}")
        print(f"   Fabricante: {puerto.manufacturer}")
        print(f"   VID:PID: {puerto.vid}:{puerto.pid}")
        print(f"   Serial: {puerto.serial_number}")
    
    print("=" * 60)
    return puertos

def probar_comunicacion_basica(puerto, baudrate=9600, timeout=1, enviar_comandos=True):
    """Intenta abrir comunicación básica con el puerto"""
    print(f"\n🔍 Probando comunicación en {puerto} @ {baudrate} baud...")

    ser = None
    try:
        ser = serial.Serial(
            port=puerto,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout
        )

        print(f"✅ Puerto {puerto} abierto correctamente")
        print(f"   Baudrate: {ser.baudrate}")
        print(f"   Timeout: {ser.timeout}s")

        # Intentar leer datos si hay algo en buffer
        time.sleep(0.5)
        if ser.in_waiting > 0:
            datos = ser.read(ser.in_waiting)
            print(f"   📥 Datos en buffer ({len(datos)} bytes): {datos}")
            try:
                print(f"   📥 Datos (texto): {datos.decode('ascii', errors='ignore')}")
            except Exception:
                pass
        else:
            print("   ℹ️  No hay datos en buffer")

        if enviar_comandos:
            # Intentar enviar comandos comunes de vibrómetros
            comandos_prueba = [
                b'*IDN?\r\n',      # Identificación (SCPI estándar)
                b'ID?\r\n',        # Identificación alternativa
                b'?\r\n',          # Query genérico
                b'\x05',           # ENQ (Enquiry)
                b'READ?\r\n',      # Leer medición
                b'MEAS?\r\n',      # Medición
            ]

            for cmd in comandos_prueba:
                print(f"\n   📤 Enviando: {cmd}")
                ser.write(cmd)
                ser.flush()
                time.sleep(0.5)

                # Leer lo que haya llegado tras el comando
                leidos = b''
                start = time.time()
                while time.time() - start < 1.0:
                    if ser.in_waiting:
                        leidos += ser.read(ser.in_waiting)
                    time.sleep(0.05)
                if leidos:
                    print(f"   📥 Respuesta ({len(leidos)} bytes): {leidos}")
                    try:
                        print(f"   📥 Respuesta (texto): {leidos.decode('ascii', errors='ignore')}")
                    except Exception:
                        pass
                else:
                    print("   ⚠️  Sin respuesta")

        return True

    except serial.SerialException as e:
        print(f"❌ Error abriendo puerto {puerto}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False
    finally:
        if ser is not None:
            try:
                ser.close()
                print(f"✅ Puerto {puerto} cerrado")
            except Exception as e:
                print(f"⚠️ Error al cerrar puerto {puerto}: {e}")

def probar_multiples_baudrates(puerto):
    """Prueba diferentes velocidades de baudrate"""
    baudrates = [9600, 19200, 38400, 57600, 115200]
    
    print(f"\n🔄 Probando múltiples baudrates en {puerto}...")
    print("=" * 60)
    
    for baud in baudrates:
        probar_comunicacion_basica(puerto, baudrate=baud, timeout=0.5)
        print("-" * 60)
        time.sleep(0.5)

def main():
    print("\n" + "=" * 60)
    print("TEST DE COMUNICACIÓN ADASH 4900 VIBRIO")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Pruebas de comunicación serie con ADASH 4900")
    parser.add_argument("--port", help="Puerto COM, ej. COM3")
    parser.add_argument("--baud", type=int, default=9600, help="Baudrate inicial (por defecto 9600)")
    parser.add_argument("--solo_abrir_cerrar", action="store_true", help="Solo abrir y cerrar el puerto (smoke test)")
    parser.add_argument("--probar_todos_baudrates", action="store_true", help="Probar múltiples baudrates comunes")
    parser.add_argument("--no_comandos", action="store_true", help="No enviar comandos de prueba")
    args = parser.parse_args()

    # Listar puertos si no se especificó --port
    puertos = listar_puertos_com()
    if not puertos:
        print("\n❌ Conecta el ADASH 4900 via USB y vuelve a ejecutar")
        return

    if args.port:
        puerto_seleccionado = args.port
        print(f"\n✅ Usando puerto indicado: {puerto_seleccionado}")
    else:
        if len(puertos) == 1:
            puerto_seleccionado = puertos[0].device
            print(f"\n✅ Usando puerto único detectado: {puerto_seleccionado}")
        else:
            # Pedir al usuario que seleccione
            try:
                seleccion = int(input(f"\nSelecciona puerto (1-{len(puertos)}): ")) - 1
                if 0 <= seleccion < len(puertos):
                    puerto_seleccionado = puertos[seleccion].device
                else:
                    print("❌ Selección inválida")
                    return
            except ValueError:
                print("❌ Entrada inválida")
                return

    # Smoke test (solo abrir/cerrar)
    if args.solo_abrir_cerrar:
        print("\n" + "=" * 60)
        print("PRUEBA: Solo abrir y cerrar puerto")
        print("=" * 60)
        probar_comunicacion_basica(puerto_seleccionado, baudrate=args.baud, enviar_comandos=not args.no_comandos)
    else:
        # Prueba básica
        print("\n" + "=" * 60)
        print("PRUEBA 1: Comunicación básica")
        print("=" * 60)
        probar_comunicacion_basica(puerto_seleccionado, baudrate=args.baud, enviar_comandos=not args.no_comandos)

    # Probar múltiples baudrates si se solicita
    if args.probar_todos_baudrates:
        probar_multiples_baudrates(puerto_seleccionado)

    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA")
    print("=" * 60)
    print("\n📋 SIGUIENTE PASO:")
    print("   1. Revisa la documentación del ADASH 4900 para comandos específicos")
    print("   2. Si obtuviste respuestas, anota el baudrate y formato")
    print("   3. Verifica el manual para protocolo de comunicación")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

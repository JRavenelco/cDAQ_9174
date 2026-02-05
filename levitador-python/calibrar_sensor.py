"""
Calibración del Sensor de Posición - Levitador Magnético
=========================================================

Este script ayuda a calibrar el sensor de posición midiendo
el valor ADC en posiciones conocidas.

Procedimiento:
1. Ejecutar este script
2. Colocar la esfera en posiciones conocidas (usar regla/calibrador)
3. Presionar ENTER para capturar cada punto
4. El script calcula los parámetros de calibración

Autor: Jesús (Doctorado UAQ)
"""

import time
from serial_win32 import Win32Serial
import numpy as np

# Parámetros actuales (a calibrar)
ESC_ACTUAL = 0.05 / 1023.0  # m por unidad ADC

def leer_adc_promedio(ser, n_muestras=100):
    """Lee n muestras del ADC y devuelve el promedio."""
    valores = []
    flagcom = 0
    pv = 0
    i = 0
    
    while len(valores) < n_muestras:
        b = ser.read(1)
        if not b:
            continue
        
        recibido = b[0]
        
        if flagcom != 0:
            flagcom += 1
        
        if (recibido == 0xAA) and (flagcom == 0):
            pv = 0
            i = 0
            flagcom = 1
            continue
        
        if flagcom == 2:
            pv = recibido << 8
            continue
        
        if flagcom == 3:
            pv = pv + recibido
            continue
        
        if flagcom == 4:
            i = recibido << 8
            continue
        
        if flagcom == 5:
            i = i + recibido
            
            if pv <= 1023 and i <= 1023:
                valores.append(pv)
                # Enviar PWM de mantenimiento (para que no caiga)
                pwm = 128  # Valor medio
                ser.write(bytes([pwm]))
            
            flagcom = 0
    
    return np.mean(valores), np.std(valores)


def main():
    print("="*60)
    print("🔧 CALIBRACIÓN DEL SENSOR DE POSICIÓN")
    print("="*60)
    print("""
Procedimiento:
1. Asegúrate de que la esfera esté levitando o sostenida
2. Usa una regla o calibrador para medir la distancia
3. La distancia se mide desde el electroimán hasta la esfera

IMPORTANTE: Necesitas al menos 2 puntos de calibración.
""")
    
    # Conectar
    print("🔌 Conectando a COM1...")
    ser = Win32Serial('COM1', 115200)
    print("✅ Conectado")
    
    # Esperar sincronización
    print("\n⏳ Esperando switch del microcontrolador (byte 0xAA)...")
    while True:
        b = ser.read(1)
        if b == b'\xAA':
            print("✅ Switch activado!")
            break
    
    puntos = []
    
    print("\n" + "="*60)
    print("📏 CAPTURA DE PUNTOS DE CALIBRACIÓN")
    print("="*60)
    
    while True:
        print(f"\n📍 Punto {len(puntos) + 1}")
        distancia = input("   Ingresa la distancia física en mm (o 'fin' para terminar): ")
        
        if distancia.lower() == 'fin':
            if len(puntos) < 2:
                print("   ⚠️ Necesitas al menos 2 puntos. Continúa capturando.")
                continue
            break
        
        try:
            dist_mm = float(distancia)
        except ValueError:
            print("   ❌ Valor inválido. Ingresa un número.")
            continue
        
        print("   📊 Capturando 100 muestras del ADC...")
        adc_mean, adc_std = leer_adc_promedio(ser, 100)
        
        puntos.append({
            'distancia_mm': dist_mm,
            'distancia_m': dist_mm / 1000,
            'adc_mean': adc_mean,
            'adc_std': adc_std
        })
        
        print(f"   ✅ Capturado: {dist_mm:.1f}mm → ADC={adc_mean:.1f} (σ={adc_std:.1f})")
    
    ser.close()
    
    # Calcular calibración
    print("\n" + "="*60)
    print("📐 CÁLCULO DE CALIBRACIÓN")
    print("="*60)
    
    # Regresión lineal: y = esc * (adc - offset)
    # donde y es la distancia en metros
    
    adc_vals = np.array([p['adc_mean'] for p in puntos])
    dist_vals = np.array([p['distancia_m'] for p in puntos])
    
    # Ajuste lineal: dist = m * adc + b
    # Entonces: esc = m, offset = -b/m
    coef = np.polyfit(adc_vals, dist_vals, 1)
    m, b = coef
    
    # Calcular R²
    dist_pred = m * adc_vals + b
    ss_res = np.sum((dist_vals - dist_pred)**2)
    ss_tot = np.sum((dist_vals - np.mean(dist_vals))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    print(f"\n📊 Puntos capturados:")
    for p in puntos:
        print(f"   {p['distancia_mm']:.1f}mm → ADC={p['adc_mean']:.1f}")
    
    print(f"\n📈 Regresión lineal:")
    print(f"   y = {m:.8f} * ADC + {b:.6f}")
    print(f"   R² = {r2:.4f}")
    
    # Parámetros para el código
    esc_nuevo = m
    offset_nuevo = -b / m if m != 0 else 0
    
    print(f"\n🔧 PARÁMETROS DE CALIBRACIÓN:")
    print(f"   esc = {esc_nuevo:.10f}  # m por unidad ADC")
    print(f"   offset = {offset_nuevo:.1f}  # unidades ADC")
    
    print(f"\n📝 Código para control.py:")
    print(f"   esc = {esc_nuevo:.10e}")
    print(f"   pv_offset = {offset_nuevo:.0f}")
    print(f"   y = esc * (pv - pv_offset)")
    
    # Comparar con valor actual
    print(f"\n📊 Comparación con valor actual:")
    print(f"   Actual:  esc = {ESC_ACTUAL:.10e}")
    print(f"   Nuevo:   esc = {esc_nuevo:.10e}")
    print(f"   Factor:  {esc_nuevo / ESC_ACTUAL:.2f}x")
    
    # Guardar calibración
    with open('calibracion_sensor.txt', 'w') as f:
        f.write("# Calibración del sensor de posición\n")
        f.write(f"# Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# R² = {r2:.4f}\n\n")
        f.write("# Puntos de calibración:\n")
        for p in puntos:
            f.write(f"# {p['distancia_mm']:.1f}mm -> ADC={p['adc_mean']:.1f}\n")
        f.write(f"\nesc = {esc_nuevo:.10e}\n")
        f.write(f"pv_offset = {offset_nuevo:.0f}\n")
    
    print(f"\n💾 Calibración guardada en: calibracion_sensor.txt")
    
    print("\n" + "="*60)
    print("✅ CALIBRACIÓN COMPLETADA")
    print("="*60)
    print("""
Para aplicar la calibración, edita control.py:

1. Busca la línea:
   esc = 0.05 / 1023.0
   
2. Reemplázala por:
   esc = {:.10e}
   
3. Busca la línea:
   pv_offset = 0
   
4. Reemplázala por:
   pv_offset = {:.0f}
""".format(esc_nuevo, offset_nuevo))


if __name__ == '__main__':
    main()

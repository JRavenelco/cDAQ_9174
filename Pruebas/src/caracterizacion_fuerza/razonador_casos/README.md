# Razonador de casos para cortes e histeresis

Este modulo convierte los cortes experimentales en una base de casos reutilizable para el flujo doctoral:

1. Captura de fuerza/aceleracion.
2. Exportacion de cortes y ventanas.
3. Preparacion de datos normalizados para Bouc-Wen/LuGre/PINN.
4. Construccion de memoria experimental para razonamiento basado en casos.
5. Integracion posterior con Hailo-8L y Gemma4 VLM.

## Primer comando

Desde `Pruebas/src/caracterizacion_fuerza`:

```powershell
python razonador_casos/construir_casos.py
```

## Salidas

El script genera `razonador_casos/salidas/casos_dataset/` con:

- `casos_historicos.csv`: indice maestro de casos.
- `casos_historicos.jsonl`: version por registro para agentes/VLM.
- `boucwen_ready/*.txt`: cortes normalizados con columnas `tiempo_s`, `fuerza_norm`, `entrada_norm`.
- `features/*.csv`: series y features por caso.
- `ventanas/*.txt`: ventanas candidatas para ajuste y entrenamiento.
- `resumen_casos.md`: lectura rapida para la tesis.

## Relacion con la tesis

La idea se alinea con el titulo doctoral: modelo predictivo de histeresis usando razonamiento basado en casos. Cada caso conserva la senal, sus descriptores, su energia/lazo de histeresis y enlaces a futuras curvas Bouc-Wen.

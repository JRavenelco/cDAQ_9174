# Virtual Instrument UDP Test

Prueba de banco para emular en Windows un instrumento virtual de fuerza/aceleracion y enviar las senales por UDP a la Jetson.

## 1. Jetson: recibir y guardar

En la Jetson:

```bash
cd ~/cDAQ_9174/Pruebas/src/remote_monitoring
python3 jetson_virtual_udp_receiver.py --bind-ip 0.0.0.0 --port 12011
```

Si estas probando desde el enlace USB, la Jetson tiene `192.168.55.1`.

## 2. Windows: simular instrumento y enviar

En Windows, desde el repo:

```powershell
cd "C:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\remote_monitoring"
python virtual_instrument_udp_sender.py --jetson-ip 192.168.55.1 --port 12011 --mode normal --duration 20
```

Modos disponibles:

```powershell
python virtual_instrument_udp_sender.py --jetson-ip 192.168.55.1 --mode normal  --duration 20
python virtual_instrument_udp_sender.py --jetson-ip 192.168.55.1 --mode chatter --duration 20
python virtual_instrument_udp_sender.py --jetson-ip 192.168.55.1 --mode wear    --duration 20
python virtual_instrument_udp_sender.py --jetson-ip 192.168.55.1 --mode impact  --duration 20
```

## Salidas en Jetson

El receiver guarda:

- `logs/virtual_instrument_YYYYMMDD_HHMMSS.csv`: senal completa recibida.
- `logs/virtual_instrument_YYYYMMDD_HHMMSS_cases.jsonl`: features por ventana para alimentar el razonador basado en casos.

## 3. Convertir logs UDP al formato de `razonador_casos`

Desde `Pruebas/src/remote_monitoring`:

```powershell
python udp_logs_to_cases.py
```

Ese comando busca automaticamente el ultimo par disponible en `logs/` y genera salidas en:

- `../caracterizacion_fuerza/razonador_casos/salidas/casos_virtual_udp/casos_virtuales.jsonl`
- `../caracterizacion_fuerza/razonador_casos/salidas/casos_virtual_udp/casos_virtuales.csv`
- `../caracterizacion_fuerza/razonador_casos/salidas/casos_virtual_udp/casos_comparables_con_historicos.jsonl`

Si quieres convertir archivos concretos:

```powershell
python udp_logs_to_cases.py --csv .\logs\virtual_instrument_YYYYMMDD_HHMMSS.csv --cases .\logs\virtual_instrument_YYYYMMDD_HHMMSS_cases.jsonl
```

Opciones utiles:

```powershell
python udp_logs_to_cases.py --limit 5
python udp_logs_to_cases.py --out-dir ..\caracterizacion_fuerza\razonador_casos\salidas\casos_virtual_udp_prueba
python udp_logs_to_cases.py --historicos-jsonl ..\caracterizacion_fuerza\razonador_casos\salidas\casos_dataset\casos_historicos.jsonl
```

El adaptador intenta reconstruir cada ventana usando el CSV crudo para producir archivos auxiliares compatibles con el flujo Bouc-Wen:

- `features/*.csv`
- `boucwen_ready/*.txt`
- `ventanas/*.txt`
- comando sugerido para `comparar_bouc_wen_shaker_corte.py`

## Protocolo

UDP binario pequeno, sin dependencias externas:

- puerto: `12011`
- frecuencia por defecto: `2500 Hz`
- bloque por defecto: `100 muestras`
- canales: `force_v`, `accel_g`

El emulador incluye una memoria tipo Bouc-Wen ligera para que la fuerza tenga lazo histeretico respecto a la aceleracion.

## 4. Comparar casos virtuales contra historicos

Despues de convertir logs UDP a casos:

```powershell
python compare_udp_cases.py
```

Ese comando lee por defecto:

- `../caracterizacion_fuerza/razonador_casos/salidas/casos_virtual_udp/casos_virtuales.jsonl`
- `../caracterizacion_fuerza/razonador_casos/salidas/casos_dataset/casos_historicos.jsonl`

Y genera:

- `../caracterizacion_fuerza/razonador_casos/salidas/casos_virtual_udp/casos_virtuales_ranked.jsonl`
- `../caracterizacion_fuerza/razonador_casos/salidas/casos_virtual_udp/casos_virtuales_ranked.csv`

Cada registro incluye:

- `matched_case_id`
- `similarity_score`
- `top_matches`
- `run_boucwen_next`
- `recommended_boucwen_reference`
- `diagnostic_label_seed`
- `llm_summary_seed`

Ejemplos:

```powershell
python compare_udp_cases.py --top-k 5
python compare_udp_cases.py --weights loop_area_norm=2,input_rms=1.2
python compare_udp_cases.py --virtual-jsonl ..\caracterizacion_fuerza\razonador_casos\salidas\casos_virtual_udp\casos_virtuales.jsonl --historical-jsonl ..\caracterizacion_fuerza\razonador_casos\salidas\casos_dataset\casos_historicos.jsonl
```

import pyshark
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def analyze_usb_capture(capture_file, vendor_id=0x3923, product_id=0x74a5, output_dir=None):
    """
    Analiza una captura de Wireshark para identificar y clasificar los comandos USB 
    enviados al dispositivo NI-DAQ.
    
    Args:
        capture_file: Ruta al archivo de captura (.pcapng)
        vendor_id: ID de vendedor del dispositivo (default: 0x3923 para NI)
        product_id: ID de producto del dispositivo (default: 0x74a5 para NI-9174)
        output_dir: Directorio para guardar resultados (default: mismo directorio que el script)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis')
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Analizando captura: {capture_file}")
    print(f"Buscando dispositivo: VID={vendor_id:04x}, PID={product_id:04x}")
    
    # Filtrar paquetes USB relacionados con el dispositivo
    filter_string = f"usb.idVendor == {vendor_id} and usb.idProduct == {product_id}"
    capture = pyshark.FileCapture(capture_file, display_filter=filter_string)
    
    # Recopilar información de los paquetes
    packets_data = []
    control_requests = []
    bulk_transfers = []
    
    try:
        for i, packet in enumerate(capture):
            # Obtener información básica del paquete
            try:
                timestamp = float(packet.frame_info.time_epoch)
                packet_type = "Unknown"
                
                # Determinar tipo de transferencia USB
                if hasattr(packet, 'usb') and hasattr(packet.usb, 'transfer_type'):
                    if packet.usb.transfer_type == "0":  # Control
                        packet_type = "Control"
                        
                        # Extraer detalles de la transferencia de control
                        if hasattr(packet.usb, 'setup') and hasattr(packet.usb, 'bmRequestType'):
                            request_type = int(packet.usb.bmRequestType, 16)
                            request = int(packet.usb.bRequest) if hasattr(packet.usb, 'bRequest') else None
                            value = int(packet.usb.wValue, 16) if hasattr(packet.usb, 'wValue') else None
                            index = int(packet.usb.wIndex, 16) if hasattr(packet.usb, 'wIndex') else None
                            length = int(packet.usb.wLength) if hasattr(packet.usb, 'wLength') else None
                            
                            # Obtener datos de la transferencia
                            data = None
                            if hasattr(packet, 'data') and hasattr(packet.data, 'data'):
                                data = packet.data.data
                            
                            control_requests.append({
                                'timestamp': timestamp,
                                'request_type': f"0x{request_type:02x}",
                                'request': f"0x{request:02x}" if request is not None else None,
                                'value': f"0x{value:04x}" if value is not None else None,
                                'index': f"0x{index:04x}" if index is not None else None,
                                'length': length,
                                'data': data,
                                'packet_num': i+1
                            })
                            
                    elif packet.usb.transfer_type == "2":  # Bulk
                        packet_type = "Bulk"
                        
                        # Extraer información del endpoint
                        endpoint = int(packet.usb.endpoint_address, 16) if hasattr(packet.usb, 'endpoint_address') else None
                        direction = "IN" if endpoint and (endpoint & 0x80) else "OUT"
                        
                        # Obtener datos de la transferencia
                        data_length = int(packet.usb.data_len) if hasattr(packet.usb, 'data_len') else 0
                        data = None
                        if hasattr(packet, 'data') and hasattr(packet.data, 'data'):
                            data = packet.data.data
                        
                        bulk_transfers.append({
                            'timestamp': timestamp,
                            'endpoint': f"0x{endpoint:02x}" if endpoint is not None else None,
                            'direction': direction,
                            'length': data_length,
                            'data': data,
                            'packet_num': i+1
                        })
                        
                    elif packet.usb.transfer_type == "1":  # Isochronous
                        packet_type = "Isochronous"
                    elif packet.usb.transfer_type == "3":  # Interrupt
                        packet_type = "Interrupt"
                
                # Guardar información del paquete
                packets_data.append({
                    'num': i+1,
                    'timestamp': timestamp,
                    'type': packet_type,
                    'description': str(packet.usb) if hasattr(packet, 'usb') else "N/A"
                })
                
            except Exception as e:
                print(f"Error al procesar paquete {i+1}: {e}")
    
    except Exception as e:
        print(f"Error al leer captura: {e}")
    
    finally:
        capture.close()
    
    # Convertir a DataFrames
    df_packets = pd.DataFrame(packets_data)
    df_control = pd.DataFrame(control_requests)
    df_bulk = pd.DataFrame(bulk_transfers)
    
    # Guardar resultados como CSV
    if not df_packets.empty:
        df_packets.to_csv(os.path.join(output_dir, f"packets_{timestamp}.csv"), index=False)
        print(f"Guardado: packets_{timestamp}.csv")
    
    if not df_control.empty:
        df_control.to_csv(os.path.join(output_dir, f"control_requests_{timestamp}.csv"), index=False)
        print(f"Guardado: control_requests_{timestamp}.csv")
        
        # Análisis de comandos de control
        analyze_control_requests(df_control, output_dir, timestamp)
    
    if not df_bulk.empty:
        df_bulk.to_csv(os.path.join(output_dir, f"bulk_transfers_{timestamp}.csv"), index=False)
        print(f"Guardado: bulk_transfers_{timestamp}.csv")
        
        # Análisis de transferencias bulk
        analyze_bulk_transfers(df_bulk, output_dir, timestamp)
    
    print("Análisis completado.")
    return df_packets, df_control, df_bulk

def analyze_control_requests(df_control, output_dir, timestamp):
    """Analiza las solicitudes de control para identificar patrones"""
    if df_control.empty:
        return
    
    print("Analizando solicitudes de control...")
    
    # Agrupar por tipo de solicitud
    request_counts = df_control.groupby(['request_type', 'request']).size().reset_index(name='count')
    request_counts = request_counts.sort_values('count', ascending=False)
    
    # Guardar resumen
    request_counts.to_csv(os.path.join(output_dir, f"control_summary_{timestamp}.csv"), index=False)
    print(f"Guardado: control_summary_{timestamp}.csv")
    
    # Visualizar solicitudes más comunes
    if len(request_counts) > 0:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(min(len(request_counts), 10)), 
                      request_counts['count'].head(10), 
                      tick_label=[f"{row['request_type']},{row['request']}" 
                                  for _, row in request_counts.head(10).iterrows()])
        plt.title('Solicitudes de Control más Comunes')
        plt.xlabel('Tipo de Solicitud')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Guardar gráfico
        plt.savefig(os.path.join(output_dir, f"control_requests_{timestamp}.png"))
        print(f"Guardado: control_requests_{timestamp}.png")
        
        # Cerrar para liberar memoria
        plt.close()
    
    # Analizar secuencias de comandos
    if len(df_control) > 1:
        # Ordenar por timestamp
        df_sorted = df_control.sort_values('timestamp')
        
        # Calcular diferencias de tiempo
        df_sorted['time_diff'] = df_sorted['timestamp'].diff()
        
        # Identificar secuencias (comandos que ocurren dentro de un intervalo de tiempo pequeño)
        threshold = 0.1  # 100 ms
        df_sorted['sequence'] = (df_sorted['time_diff'] > threshold).cumsum()
        
        # Agrupar comandos por secuencia
        sequences = df_sorted.groupby('sequence')
        
        # Guardar las secuencias identificadas
        sequence_data = []
        for seq_id, group in sequences:
            commands = []
            for _, row in group.iterrows():
                cmd = f"REQ={row['request']} VAL={row['value']} IDX={row['index']}"
                commands.append(cmd)
            
            sequence_data.append({
                'sequence_id': seq_id,
                'start_time': group['timestamp'].min(),
                'end_time': group['timestamp'].max(),
                'duration': group['timestamp'].max() - group['timestamp'].min(),
                'command_count': len(group),
                'commands': ' → '.join(commands)
            })
        
        df_sequences = pd.DataFrame(sequence_data)
        df_sequences.to_csv(os.path.join(output_dir, f"command_sequences_{timestamp}.csv"), index=False)
        print(f"Guardado: command_sequences_{timestamp}.csv")

def analyze_bulk_transfers(df_bulk, output_dir, timestamp):
    """Analiza las transferencias bulk para identificar patrones en los datos"""
    if df_bulk.empty:
        return
    
    print("Analizando transferencias bulk...")
    
    # Estadísticas básicas
    df_bulk['timestamp_rel'] = df_bulk['timestamp'] - df_bulk['timestamp'].min()
    
    # Separar transferencias IN y OUT
    df_in = df_bulk[df_bulk['direction'] == 'IN']
    df_out = df_bulk[df_bulk['direction'] == 'OUT']
    
    # Análisis de transferencias IN (datos recibidos del dispositivo)
    if not df_in.empty:
        # Tamaño promedio de las transferencias
        avg_size = df_in['length'].mean()
        
        # Frecuencia de transferencias
        if len(df_in) > 1:
            time_diffs = df_in['timestamp'].diff().dropna()
            avg_interval = time_diffs.mean()
            transfer_rate = 1.0 / avg_interval if avg_interval > 0 else 0
            
            # Guardar estadísticas
            stats = {
                'total_in_transfers': len(df_in),
                'avg_transfer_size': avg_size,
                'avg_interval': avg_interval,
                'transfers_per_second': transfer_rate,
                'estimated_sample_rate': transfer_rate * avg_size / 2  # Asumiendo 16 bits por muestra
            }
            
            pd.DataFrame([stats]).to_csv(os.path.join(output_dir, f"bulk_in_stats_{timestamp}.csv"), index=False)
            print(f"Guardado: bulk_in_stats_{timestamp}.csv")
            
            # Visualizar intervalos entre transferencias
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(time_diffs)), time_diffs.values * 1000)  # Convertir a ms
            plt.title('Intervalos entre Transferencias IN')
            plt.xlabel('Índice de Transferencia')
            plt.ylabel('Intervalo (ms)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"bulk_in_intervals_{timestamp}.png"))
            print(f"Guardado: bulk_in_intervals_{timestamp}.png")
            plt.close()
            
            # Visualizar tamaños de transferencia
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(df_in)), df_in['length'])
            plt.title('Tamaños de Transferencias IN')
            plt.xlabel('Índice de Transferencia')
            plt.ylabel('Tamaño (bytes)')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"bulk_in_sizes_{timestamp}.png"))
            print(f"Guardado: bulk_in_sizes_{timestamp}.png")
            plt.close()

if __name__ == "__main__":
    # Para usar este script, necesitas capturar tráfico USB con Wireshark y guardarlo como .pcapng
    # Luego, proporciona la ruta al archivo de captura como argumento
    import sys
    
    if len(sys.argv) > 1:
        capture_file = sys.argv[1]
        analyze_usb_capture(capture_file)
    else:
        print("Uso: python wireshark_analyzer.py <ruta_a_archivo_captura.pcapng>")
        print("\nEste script analiza capturas de Wireshark para identificar comandos USB enviados al NI-9174.")
        print("Para usar este script:")
        print("1. Captura tráfico USB con Wireshark mientras ejecutas logger_usb.py")
        print("2. Guarda la captura como archivo .pcapng")
        print("3. Ejecuta este script con la ruta al archivo de captura como argumento")

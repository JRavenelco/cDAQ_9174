#!/usr/bin/env python3
"""
Test de loopback: sender simulado → receiver en localhost.
Verifica que los datos se transmiten correctamente por UDP.
"""
import sys
import os
import time
import socket
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdaq_udp_protocol import (
    DATA_PORT, CTRL_PORT, pack_data, unpack_data,
    pack_start, pack_stop, pack_sync_req, pack_sync_rsp,
    unpack_sync_req, unpack_sync_rsp, ctrl_msg_type,
    MSG_SYNC_REQ, MSG_START, MSG_STOP,
)


def test_data_loopback():
    """Envía y recibe 10 paquetes de datos por UDP localhost."""
    print("=== Test Data Loopback ===")
    
    # Receiver socket
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx_sock.bind(("127.0.0.1", DATA_PORT))
    rx_sock.settimeout(2.0)
    
    # Sender socket
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    n_packets = 10
    spr = 100
    fs = 2500.0
    errors = 0
    
    for seq in range(n_packets):
        force = np.random.randn(spr).astype(np.float32)
        accel = np.random.randn(spr).astype(np.float32)
        t = time.perf_counter()
        
        pkt = pack_data(seq, t, fs, force, accel)
        tx_sock.sendto(pkt, ("127.0.0.1", DATA_PORT))
        
        try:
            buf, _ = rx_sock.recvfrom(65535)
            result = unpack_data(buf)
            if result is None:
                print(f"  Pkt {seq}: FAIL (parse error)")
                errors += 1
                continue
            
            r_seq, r_t, r_fs, r_n, r_nch, r_force, r_accel = result
            
            assert r_seq == seq, f"seq mismatch: {r_seq} != {seq}"
            assert abs(r_t - t) < 1e-9, f"t mismatch"
            assert abs(r_fs - fs) < 0.1, f"fs mismatch"
            assert r_n == spr, f"n_samp mismatch"
            assert np.allclose(r_force, force, atol=1e-6), "force mismatch"
            assert np.allclose(r_accel, accel, atol=1e-6), "accel mismatch"
            
        except socket.timeout:
            print(f"  Pkt {seq}: FAIL (timeout)")
            errors += 1
    
    rx_sock.close()
    tx_sock.close()
    
    if errors == 0:
        print(f"  {n_packets}/{n_packets} paquetes OK")
    else:
        print(f"  ERRORES: {errors}/{n_packets}")
    return errors == 0


def test_sync_loopback():
    """Simula un intercambio SYNC_REQ/RSP por localhost."""
    print("=== Test Sync Loopback ===")
    
    # "Sender" (responde a SYNC)
    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind(("127.0.0.1", CTRL_PORT))
    srv_sock.settimeout(2.0)
    
    def sync_responder():
        try:
            data, addr = srv_sock.recvfrom(256)
            mtype = ctrl_msg_type(data)
            if mtype == MSG_SYNC_REQ:
                t1 = unpack_sync_req(data)
                t2 = time.perf_counter()
                time.sleep(0.001)  # Simular procesamiento
                t3 = time.perf_counter()
                rsp = pack_sync_rsp(t1, t2, t3)
                srv_sock.sendto(rsp, addr)
        except socket.timeout:
            pass
    
    responder = threading.Thread(target=sync_responder, daemon=True)
    responder.start()
    
    # "Receiver" (envía SYNC_REQ)
    cli_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cli_sock.settimeout(2.0)
    
    t1 = time.perf_counter()
    cli_sock.sendto(pack_sync_req(t1), ("127.0.0.1", CTRL_PORT))
    
    try:
        data, _ = cli_sock.recvfrom(256)
        t4 = time.perf_counter()
        result = unpack_sync_rsp(data)
        assert result is not None, "Failed to parse SYNC_RSP"
        
        t1_echo, t2, t3 = result
        assert abs(t1_echo - t1) < 1e-9, "t1 echo mismatch"
        
        rtt = (t4 - t1) - (t3 - t2)
        offset = ((t2 - t1) + (t3 - t4)) / 2.0
        
        print(f"  RTT: {rtt*1000:.3f} ms")
        print(f"  Offset: {offset*1000:.3f} ms")
        print(f"  OK")
        ok = True
    except Exception as e:
        print(f"  FAIL: {e}")
        ok = False
    
    cli_sock.close()
    srv_sock.close()
    responder.join(timeout=1)
    return ok


def test_control_messages():
    """Prueba START/STOP por localhost."""
    print("=== Test Control Messages ===")
    
    rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx_sock.bind(("127.0.0.1", CTRL_PORT))
    rx_sock.settimeout(2.0)
    
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # START
    tx_sock.sendto(pack_start(), ("127.0.0.1", CTRL_PORT))
    data, _ = rx_sock.recvfrom(256)
    assert ctrl_msg_type(data) == MSG_START, "START type mismatch"
    print("  START: OK")
    
    # STOP
    tx_sock.sendto(pack_stop(), ("127.0.0.1", CTRL_PORT))
    data, _ = rx_sock.recvfrom(256)
    assert ctrl_msg_type(data) == MSG_STOP, "STOP type mismatch"
    print("  STOP: OK")
    
    rx_sock.close()
    tx_sock.close()
    return True


if __name__ == "__main__":
    results = []
    results.append(("Data Loopback", test_data_loopback()))
    results.append(("Control Messages", test_control_messages()))
    results.append(("Sync Loopback", test_sync_loopback()))
    
    print("\n=== RESUMEN ===")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False
    
    print(f"\n{'TODOS LOS TESTS PASARON' if all_ok else 'HAY ERRORES'}")
    sys.exit(0 if all_ok else 1)

import socket
import re
import numpy as np
import cv2
import time
import logging
import threading
from datetime import datetime
import sys
import argparse
import os
from queue import Queue

# Konfiguracja parametrów
parser = argparse.ArgumentParser(description='Odbiornik danych z kamer OAK')
parser.add_argument('--ip1', type=str, default="192.168.70.64", help='IP kamery lewej')
parser.add_argument('--ip2', type=str, default="192.168.70.62", help='IP kamery środkowej')
parser.add_argument('--ip3', type=str, default="192.168.70.65", help='IP kamery prawej')
parser.add_argument('--port', type=int, default=65432, help='Port do komunikacji')
parser.add_argument('--debug', action='store_true', help='Włącz tryb debug z dodatkowymi informacjami')
parser.add_argument('--log-dir', type=str, default="logs", help='Katalog na pliki logów')
args = parser.parse_args()

# Utwórz katalog na logi jeśli nie istnieje
os.makedirs(args.log_dir, exist_ok=True)

# Konfiguracja loggingu głównego
logging.basicConfig(
    level=logging.INFO if not args.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{args.log_dir}/host_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('host')

# Konfiguracja loggerów do czasów dla każdej kamery
timing_loggers = {}
for camera_name, ip in [("left", args.ip1), ("center", args.ip2), ("right", args.ip3)]:
    time_logger = logging.getLogger(f'timing_{camera_name}')
    time_logger.setLevel(logging.INFO)
    
    # Format logu: timestamp, operation, duration
    handler = logging.FileHandler(f"{args.log_dir}/timing_{ip.replace('.', '_')}.csv")
    handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s'))
    time_logger.addHandler(handler)
    
    timing_loggers[ip] = time_logger

# Logger dla zdarzeń połączenia
connection_logger = logging.getLogger('connections')
connection_logger.setLevel(logging.INFO)
conn_handler = logging.FileHandler(f"{args.log_dir}/connections.log")
conn_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
connection_logger.addHandler(conn_handler)

# Logger dla czasów połączonych (dla wszystkich 3 kamer)
combined_logger = logging.getLogger('combined_timing')
combined_logger.setLevel(logging.INFO)
combined_handler = logging.FileHandler(f"{args.log_dir}/combined_timing.csv")
combined_handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s'))
combined_logger.addHandler(combined_handler)

# Ustawienia dla kamer
OAK_IP1 = args.ip1  # Lewa kamera
OAK_IP2 = args.ip2  # Środkowa kamera
OAK_IP3 = args.ip3  # Prawa kamera
OAK_PORT = args.port

# Parametry wizualizacji
nH = 22
nV = 14
SEGMENTS_H = 3  # Liczba segmentów w poziomie dla każdego IP

# Globalne zmienne dla danych z kamer
camera_data = {
    OAK_IP1: {'distances': [0.0] * (nH * nV), 'last_update': 0, 'active': False, 'frame_count': 0, 'avg_time': 0, 'last_frame_time': 0},
    OAK_IP2: {'distances': [0.0] * (nH * nV), 'last_update': 0, 'active': False, 'frame_count': 0, 'avg_time': 0, 'last_frame_time': 0},
    OAK_IP3: {'distances': [0.0] * (nH * nV), 'last_update': 0, 'active': False, 'frame_count': 0, 'avg_time': 0, 'last_frame_time': 0}
}

# Wartości progów filtrowania dla każdego źródła (można dostosować w czasie działania)
thresholds = {
    OAK_IP1: {'min': 1.7, 'max': 3.2},  # Lewa kamera
    OAK_IP2: {'min': 1.7, 'max': 2.8},  # Środkowa kamera
    OAK_IP3: {'min': 1.7, 'max': 2.8}   # Prawa kamera
}

# Zmienne blokujące dla wątków
data_lock = threading.Lock()
frame_completion_lock = threading.Lock()
active_ip_idx = 0  # 0=OAK_IP1, 1=OAK_IP2, 2=OAK_IP3
active_ip_list = [OAK_IP1, OAK_IP2, OAK_IP3]
running = True

# Kolejka do synchronizacji zbierania ramek
frame_times_queue = Queue()
frame_id = 0
frame_data = {
    "current_id": 0,
    "start_time": 0,
    "cameras_completed": set(),
    "slowest_camera": "",
    "slowest_time": 0,
    "total_time": 0
}

def log_timing(ip, operation, start_time, duration=None):
    """Log timing information for operations"""
    if ip in timing_loggers:
        if duration is None:
            duration = time.time() - start_time
        # Format: operation,duration_in_ms
        timing_loggers[ip].info(f"{operation},{duration*1000:.3f}")

def log_combined_timing(frame_id, operation, duration):
    """Log timing information for combined operations"""
    # Format: operation,frame_id,duration_in_ms
    combined_logger.info(f"{operation},{frame_id},{duration*1000:.3f}")

def camera_receiver(ip_address):
    """Wątek odbierający dane z kamery"""
    global camera_data, frame_id, frame_data
    
    logger.info(f"Rozpoczęto odbieranie danych z kamery {ip_address}")
    connection_logger.info(f"THREAD_START,{ip_address}")
    
    while running:
        try:
            # Sprawdź czy mamy rozpocząć nową ramkę
            with frame_completion_lock:
                current_frame = frame_data["current_id"]
                
                # Jeśli ta kamera odpytywana jest jako pierwsza, oznacz początek nowej ramki
                if len(frame_data["cameras_completed"]) == 0:
                    frame_data["start_time"] = time.time()
                    frame_data["cameras_completed"] = set()
                    frame_data["slowest_camera"] = ""
                    frame_data["slowest_time"] = 0
            
            # Pomiar czasu całkowitego dla pojedynczej kamery
            frame_start_time = time.time()
            
            # Pomiar czasu połączenia
            conn_start_time = time.time()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)  # Timeout dla połączenia
                sock.connect((ip_address, OAK_PORT))
                conn_time = time.time() - conn_start_time
                log_timing(ip_address, "CONNECT", conn_start_time, conn_time)
                connection_logger.info(f"CONNECTED,{ip_address}")
                
                # Odczyt nagłówka
                header_start_time = time.time()
                HLEN = 32
                chunks = []
                bytes_recd = 0
                while running and bytes_recd < HLEN:
                    chunk = sock.recv(min(HLEN - bytes_recd, 2048))
                    if chunk == b'':
                        raise RuntimeError(f"Przerwano połączenie podczas odczytu nagłówka z {ip_address}")
                    chunks.append(chunk)
                    bytes_recd = bytes_recd + len(chunk)
                
                header = str(b''.join(chunks), encoding="ascii")
                header_time = time.time() - header_start_time
                log_timing(ip_address, "HEADER_READ", header_start_time, header_time)
                
                # Parsowanie nagłówka
                header_s = re.split(' +', header)
                if header_s[0] == "HEAD":
                    MESLEN = int(header_s[1])
                    
                    # Odczyt danych
                    data_start_time = time.time()
                    chunks = []
                    bytes_recd = 0
                    while running and bytes_recd < MESLEN:
                        chunk = sock.recv(min(MESLEN - bytes_recd, 2048))
                        if chunk == b'':
                            raise RuntimeError(f"Przerwano połączenie podczas odczytu danych z {ip_address}")
                        chunks.append(chunk)
                        bytes_recd = bytes_recd + len(chunk)
                    
                    msg = str(b''.join(chunks), encoding="ascii")
                    data_time = time.time() - data_start_time
                    log_timing(ip_address, "DATA_READ", data_start_time, data_time)
                    
                    # Parsowanie danych
                    parse_start_time = time.time()
                    distances = []
                    dets = msg.split("|")
                    for det in dets:
                        distances.append(float(det))
                    parse_time = time.time() - parse_start_time
                    log_timing(ip_address, "PARSE", parse_start_time, parse_time)
                    
                    # Oblicz łączny czas klatki dla pojedynczej kamery
                    total_frame_time = time.time() - frame_start_time
                    log_timing(ip_address, "TOTAL_FRAME", frame_start_time, total_frame_time)
                    
                    # Aktualizacja danych globalnych
                    with data_lock:
                        camera_data[ip_address]['distances'] = distances
                        camera_data[ip_address]['last_update'] = time.time()
                        camera_data[ip_address]['active'] = True
                        camera_data[ip_address]['frame_count'] += 1
                        camera_data[ip_address]['last_frame_time'] = total_frame_time
                        
                        # Aktualizuj średni czas (średnia ruchoma)
                        if camera_data[ip_address]['frame_count'] == 1:
                            camera_data[ip_address]['avg_time'] = total_frame_time
                        else:
                            # Średnia ważona: 90% poprzedniej średniej + 10% nowego pomiaru
                            camera_data[ip_address]['avg_time'] = (0.9 * camera_data[ip_address]['avg_time'] + 
                                                                 0.1 * total_frame_time)
                    
                    # Aktualizuj informacje o ramce dla synchronizacji
                    with frame_completion_lock:
                        # Dodaj tę kamerę do listy zakończonych dla bieżącej ramki
                        frame_data["cameras_completed"].add(ip_address)
                        
                        # Sprawdź czy to najwolniejsza kamera
                        if total_frame_time > frame_data["slowest_time"]:
                            frame_data["slowest_camera"] = ip_address
                            frame_data["slowest_time"] = total_frame_time
                        
                        # Jeśli wszystkie kamery zakończyły odbieranie
                        if set([OAK_IP1, OAK_IP2, OAK_IP3]) <= frame_data["cameras_completed"]:
                            # Oblicz całkowity czas ramki (od startu do zakończenia ostatniej kamery)
                            frame_data["total_time"] = time.time() - frame_data["start_time"]
                            
                            # Zaloguj całkowity czas oraz informacje o najwolniejszej kamerze
                            log_combined_timing(
                                frame_data["current_id"], 
                                f"COMBINED_FRAME,{frame_data['slowest_camera']}", 
                                frame_data["total_time"]
                            )
                            
                            # Zaktualizuj ID ramki
                            frame_data["current_id"] += 1
                            frame_times_queue.put({
                                "id": frame_data["current_id"],
                                "time": frame_data["total_time"],
                                "slowest": frame_data["slowest_camera"]
                            })
                            
                            # Resetuj listę zakończonych kamer
                            frame_data["cameras_completed"] = set()
                            frame_data["slowest_time"] = 0
                
                else:
                    logger.warning(f"Otrzymano nieprawidłowy nagłówek z {ip_address}: {header}")
                    connection_logger.info(f"INVALID_HEADER,{ip_address}")
        
        except (socket.timeout, ConnectionRefusedError) as e:
            logger.warning(f"Błąd połączenia z {ip_address}: {str(e)}")
            connection_logger.info(f"CONNECTION_ERROR,{ip_address},{str(e)}")
            with data_lock:
                camera_data[ip_address]['active'] = False
            time.sleep(1)  # Poczekaj przed ponowną próbą
            
        except Exception as e:
            logger.error(f"Błąd podczas odbierania danych z {ip_address}: {str(e)}")
            connection_logger.info(f"ERROR,{ip_address},{str(e)}")
            with data_lock:
                camera_data[ip_address]['active'] = False
            time.sleep(1)  # Poczekaj przed ponowną próbą

def check_segments_presence(heatmap1, heatmap2, heatmap3, mask1, mask2, mask3):
    """Sprawdza obecność obiektów w segmentach"""
    result = [0] * 9
    
    segment_width = nH // SEGMENTS_H
    
    for ip_idx, mask in enumerate([mask1, mask2, mask3]):
        for seg_idx in range(SEGMENTS_H):
            
            start_col = seg_idx * segment_width
            end_col = start_col + segment_width if seg_idx < SEGMENTS_H-1 else nH
            
            segment_mask = mask[:, start_col:end_col]
            
            result_idx = ip_idx * SEGMENTS_H + seg_idx
            
            if np.any(segment_mask):
                result[result_idx] = 1
    
    return result

def create_heatmap(distances1, distances2, distances3):
    """Tworzy mapę ciepła z danych z trzech kamer"""
    try:
        # Przekształć dane do macierzy
        heatmap1 = np.array(distances1).reshape(nV, nH)
        heatmap2 = np.array(distances2).reshape(nV, nH)
        heatmap3 = np.array(distances3).reshape(nV, nH)
        
        # Zastosuj progi filtrowania
        mask1 = (heatmap1 >= thresholds[OAK_IP1]['min']) & (heatmap1 <= thresholds[OAK_IP1]['max'])
        mask2 = (heatmap2 >= thresholds[OAK_IP2]['min']) & (heatmap2 <= thresholds[OAK_IP2]['max'])
        mask3 = (heatmap3 >= thresholds[OAK_IP3]['min']) & (heatmap3 <= thresholds[OAK_IP3]['max'])
        
        # Sprawdź obecność w segmentach
        presence = check_segments_presence(heatmap1, heatmap2, heatmap3, mask1, mask2, mask3)
        
        # Połącz dane z trzech kamer
        heatmap = np.hstack((heatmap1, heatmap2, heatmap3))
        mask = np.hstack((mask1, mask2, mask3))
        
        # Normalizacja do wyświetlania
        min_dist = np.min(heatmap)
        max_dist = np.max(heatmap)
        normalized = ((heatmap - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8)
        
        # Zastosuj kolorową mapę
        heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Zastosuj maskę
        filtered_heatmap = heatmap_colored.copy()
        filtered_heatmap[~mask] = [0, 0, 0]
        
        # Powiększ do wyświetlania
        scale_factor = 30
        heatmap_scaled = cv2.resize(filtered_heatmap, 
                                  (3*nH * scale_factor, nV * scale_factor), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Dodaj linie segmentów
        segment_width = nH * scale_factor // SEGMENTS_H
        
        for oak_idx in range(3):  # 3 kolumny (IP1, IP2, IP3)
            oak_x_start = oak_idx * nH * scale_factor
            
            for h_idx in range(1, SEGMENTS_H):
                x = oak_x_start + h_idx * segment_width
                cv2.line(heatmap_scaled, (x, 0), (x, nV*scale_factor), (255, 255, 255), 1)
        
        # Dodaj linie oddzielające kamery
        for i in range(1, 3):
            x = i * nH * scale_factor
            cv2.line(heatmap_scaled, (x, 0), (x, nV*scale_factor), (255, 255, 255), 2)
        
        # Dodaj wartości odległości
        for y in range(nV):
            for x in range(3*nH):
                text_x = x * scale_factor + 5
                text_y = y * scale_factor + scale_factor//2
                distance = heatmap[y, x]
                
                text_color = (255, 255, 255)
                
                if x < nH:  # IP1
                    if not mask1[y, x % nH]:
                        text_color = (128, 128, 128)
                elif x < 2*nH:  # IP2
                    if not mask2[y, x % nH]:
                        text_color = (128, 128, 128)
                else:  # IP3
                    if not mask3[y, x % nH]:
                        text_color = (128, 128, 128)
                
                cv2.putText(heatmap_scaled, f"{distance:.1f}", (text_x, text_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
        
        # Dodaj informacje o obecności w segmentach
        for segment_idx in range(9):
            oak_idx = segment_idx // SEGMENTS_H
            segment_in_oak = segment_idx % SEGMENTS_H
            
            segment_x = oak_idx * nH * scale_factor + segment_in_oak * segment_width + segment_width // 2
            segment_y = nV * scale_factor - 10
            
            color = (0, 255, 0) if presence[segment_idx] else (0, 0, 255)  # Zielony dla 1, Czerwony dla 0
            
            cv2.putText(heatmap_scaled, f"S{segment_idx+1}:{presence[segment_idx]}", 
                       (segment_x - 25, segment_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return heatmap_scaled, presence
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia mapy ciepła: {str(e)}")
        # Zwróć pustą mapę w przypadku błędu
        empty_map = np.zeros((nV * 30, 3 * nH * 30, 3), dtype=np.uint8)
        return empty_map, [0] * 9

def main():
    """Główna funkcja programu"""
    global running, active_ip_idx
    
    # Uruchom wątki odbierające dane z kamer
    threads = []
    for ip in [OAK_IP1, OAK_IP2, OAK_IP3]:
        thread = threading.Thread(target=camera_receiver, args=(ip,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    logger.info("Uruchomiono odbieranie danych z kamer")
    print("Sterowanie:")
    print("  1 - wybierz IP1 do modyfikacji progów")
    print("  2 - wybierz IP2 do modyfikacji progów")
    print("  3 - wybierz IP3 do modyfikacji progów")
    print("  z - zmniejsz maksymalny próg dla aktywnego IP o 0.1")
    print("  x - zwiększ maksymalny próg dla aktywnego IP o 0.1")
    print("  a - zmniejsz minimalny próg dla aktywnego IP o 0.1")
    print("  s - zwiększ minimalny próg dla aktywnego IP o 0.1")
    print("  q - zakończenie programu")
    
    active_ip = active_ip_list[active_ip_idx]
    
    # Główna pętla wyświetlania
    last_stats_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Dane połączonych czasów ramki
    combined_times = []
    avg_combined_time = 0
    
    while True:
        try:
            start_time = time.time()
            
            # Pobierz dane z kolejki czasów ramek
            while not frame_times_queue.empty():
                frame_time_data = frame_times_queue.get()
                combined_times.append(frame_time_data["time"])
                # Zachowaj tylko ostatnie 30 pomiarów
                if len(combined_times) > 30:
                    combined_times.pop(0)
                
                # Oblicz średni czas
                if combined_times:
                    avg_combined_time = sum(combined_times) / len(combined_times)
            
            # Pobierz dane z kamer
            with data_lock:
                distances1 = camera_data[OAK_IP1]['distances']
                distances2 = camera_data[OAK_IP2]['distances']
                distances3 = camera_data[OAK_IP3]['distances']
                
                # Informacje o stanie kamer
                camera_status = []
                for i, ip in enumerate([OAK_IP1, OAK_IP2, OAK_IP3]):
                    status = "✓" if camera_data[ip]['active'] else "✗"
                    avg_time = camera_data[ip]['avg_time'] * 1000  # w ms
                    last_time = camera_data[ip]['last_frame_time'] * 1000  # w ms
                    camera_status.append(f"IP{i+1}: {status} (śr: {avg_time:.1f}ms, ost: {last_time:.1f}ms)")
            
            # Utwórz mapę ciepła
            heatmap, presence = create_heatmap(distances1, distances2, distances3)
            
            # Licznik FPS
            fps_counter += 1
            if time.time() - last_stats_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_stats_time = time.time()
            
            # Informacje o progach
            thresh_info = []
            for i, ip in enumerate([OAK_IP1, OAK_IP2, OAK_IP3]):
                active_mark = "→ " if ip == active_ip else ""
                thresh_info.append(f"{active_mark}IP{i+1}: min={thresholds[ip]['min']:.1f} max={thresholds[ip]['max']:.1f}")
            
            # Dodaj informacje na obrazie
            for i, info in enumerate(camera_status):
                cv2.putText(heatmap, info, (10, 20 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset = len(camera_status) * 20 + 30
            
            # Dodaj informacje o połączonym czasie wszystkich kamer
            cv2.putText(heatmap, f"Czas pełnej ramki: {avg_combined_time*1000:.1f}ms", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            y_offset += 25
            
            for i, info in enumerate(thresh_info):
                color = (0, 255, 255) if i == active_ip_idx else (255, 255, 255)
                cv2.putText(heatmap, info, (10, y_offset + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Dodaj informację o FPS
            cv2.putText(heatmap, f"FPS: {fps}", (10, y_offset + len(thresh_info)*20 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Pokaż mapę ciepła
            cv2.imshow("Depth Heatmap", heatmap)
            
            # Obsługa klawiszy
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('1'):
                active_ip_idx = 0
                active_ip = active_ip_list[active_ip_idx]
                logger.info(f"Wybrano IP1 do modyfikacji progów")
            elif key == ord('2'):
                active_ip_idx = 1
                active_ip = active_ip_list[active_ip_idx]
                logger.info(f"Wybrano IP2 do modyfikacji progów")
            elif key == ord('3'):
                active_ip_idx = 2
                active_ip = active_ip_list[active_ip_idx]
                logger.info(f"Wybrano IP3 do modyfikacji progów")
            elif key == ord('z'):
                # Zmniejsz maksymalny próg o 0.1 dla aktywnego IP
                thresholds[active_ip]['max'] = max(thresholds[active_ip]['min'], thresholds[active_ip]['max'] - 0.1)
                logger.info(f"Zmniejszono maksymalny próg dla {active_ip}: {thresholds[active_ip]['max']:.1f}")
            elif key == ord('x'):
                # Zwiększ maksymalny próg o 0.1 dla aktywnego IP
                thresholds[active_ip]['max'] = thresholds[active_ip]['max'] + 0.1
                logger.info(f"Zwiększono maksymalny próg dla {active_ip}: {thresholds[active_ip]['max']:.1f}")
            elif key == ord('a'):
                # Zmniejsz minimalny próg o 0.1 dla aktywnego IP
                thresholds[active_ip]['min'] = max(0.1, thresholds[active_ip]['min'] - 0.1)
                logger.info(f"Zmniejszono minimalny próg dla {active_ip}: {thresholds[active_ip]['min']:.1f}")
            elif key == ord('s'):
                # Zwiększ minimalny próg o 0.1 dla aktywnego IP
                thresholds[active_ip]['min'] = min(thresholds[active_ip]['max'] - 0.1, thresholds[active_ip]['min'] + 0.1)
                logger.info(f"Zwiększono minimalny próg dla {active_ip}: {thresholds[active_ip]['min']:.1f}")
            
            # Poczekaj do końca cyklu, aby utrzymać stałą częstotliwość odświeżania
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # Próba utrzymania ~30 FPS
                time.sleep(0.033 - elapsed)
                
        except Exception as e:
            logger.error(f"Błąd w głównej pętli: {e}")
            time.sleep(0.1)
    
    # Zakończenie programu
    running = False
    logger.info("Zakończenie programu")
    cv2.destroyAllWindows()
    
    # Poczekaj na zakończenie wątków
    for thread in threads:
        thread.join(1.0)

if __name__ == "__main__":
    main()

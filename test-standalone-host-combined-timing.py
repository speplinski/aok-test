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
import locale

# Configuration parameters
parser = argparse.ArgumentParser(description='OAK Camera Data Receiver')
parser.add_argument('--ip1', type=str, default="192.168.70.64", help='Left camera IP')
parser.add_argument('--ip2', type=str, default="192.168.70.62", help='Center camera IP')
parser.add_argument('--ip3', type=str, default="192.168.70.65", help='Right camera IP')
parser.add_argument('--port', type=int, default=65432, help='Communication port')
parser.add_argument('--timeout', type=float, default=1.0, help='Connection timeout in seconds')
parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional information')
parser.add_argument('--log-dir', type=str, default="logs", help='Directory for log files')
args = parser.parse_args()

# Set appropriate encoding for the system
if sys.platform == 'win32':
    # Set UTF-8 encoding for Windows
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Create log directory if it doesn't exist
os.makedirs(args.log_dir, exist_ok=True)

# Configure main logging
logging.basicConfig(
    level=logging.INFO if not args.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{args.log_dir}/host_log.txt", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('host')

# Configure timing loggers for each camera
timing_loggers = {}
for camera_name, ip in [("left", args.ip1), ("center", args.ip2), ("right", args.ip3)]:
    time_logger = logging.getLogger(f'timing_{camera_name}')
    time_logger.setLevel(logging.INFO)
    
    # Log format: timestamp, operation, duration
    handler = logging.FileHandler(f"{args.log_dir}/timing_{ip.replace('.', '_')}.csv", encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s'))
    time_logger.addHandler(handler)
    
    timing_loggers[ip] = time_logger

# Logger for connection events
connection_logger = logging.getLogger('connections')
connection_logger.setLevel(logging.INFO)
conn_handler = logging.FileHandler(f"{args.log_dir}/connections.log", encoding='utf-8')
conn_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
connection_logger.addHandler(conn_handler)

# Logger for combined times (for all 3 cameras)
combined_logger = logging.getLogger('combined_timing')
combined_logger.setLevel(logging.INFO)
combined_handler = logging.FileHandler(f"{args.log_dir}/combined_timing.csv", encoding='utf-8')
combined_handler.setFormatter(logging.Formatter('%(asctime)s,%(message)s'))
combined_logger.addHandler(combined_handler)

# Camera settings
OAK_IP1 = args.ip1  # Left camera
OAK_IP2 = args.ip2  # Center camera
OAK_IP3 = args.ip3  # Right camera
OAK_PORT = args.port
SOCKET_TIMEOUT = args.timeout  # Connection timeout

# Visualization parameters
nH = 22
nV = 14
SEGMENTS_H = 3  # Number of segments horizontally for each IP

# Global variables for camera data
camera_data = {
    OAK_IP1: {'distances': [0.0] * (nH * nV), 'last_update': 0, 'active': False, 'frame_count': 0, 'avg_time': 0, 'last_frame_time': 0},
    OAK_IP2: {'distances': [0.0] * (nH * nV), 'last_update': 0, 'active': False, 'frame_count': 0, 'avg_time': 0, 'last_frame_time': 0},
    OAK_IP3: {'distances': [0.0] * (nH * nV), 'last_update': 0, 'active': False, 'frame_count': 0, 'avg_time': 0, 'last_frame_time': 0}
}

# Filtering threshold values for each source (can be adjusted during runtime)
thresholds = {
    OAK_IP1: {'min': 1.7, 'max': 3.2},  # Left camera
    OAK_IP2: {'min': 1.7, 'max': 2.8},  # Center camera
    OAK_IP3: {'min': 1.7, 'max': 2.8}   # Right camera
}

# Thread locking variables
data_lock = threading.Lock()
frame_completion_lock = threading.Lock()
active_ip_idx = 0  # 0=OAK_IP1, 1=OAK_IP2, 2=OAK_IP3
active_ip_list = [OAK_IP1, OAK_IP2, OAK_IP3]
running = True

# Queue for frame collection synchronization
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

def receive_data_with_timeout(sock, size, timeout=1.0):
    """Receive data from socket with timeout for each chunk"""
    chunks = []
    bytes_recd = 0
    start_time = time.time()
    
    while bytes_recd < size:
        if time.time() - start_time > timeout:
            raise socket.timeout("Timeout during data reception")
        
        # Set a short timeout for a single recv
        sock.settimeout(0.2)
        try:
            chunk = sock.recv(min(size - bytes_recd, 2048))
            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd += len(chunk)
        except socket.timeout:
            # Short timeout for a single recv - continue trying
            continue
    
    return b''.join(chunks)

def camera_receiver(ip_address):
    """Thread receiving data from camera"""
    global camera_data, frame_id, frame_data
    
    logger.info(f"Started receiving data from camera {ip_address}")
    connection_logger.info(f"THREAD_START,{ip_address}")
    
    while running:
        try:
            # Check if we need to start a new frame
            with frame_completion_lock:
                current_frame = frame_data["current_id"]
                
                # If this camera is being polled first, mark the beginning of a new frame
                if len(frame_data["cameras_completed"]) == 0:
                    frame_data["start_time"] = time.time()
                    frame_data["cameras_completed"] = set()
                    frame_data["slowest_camera"] = ""
                    frame_data["slowest_time"] = 0
            
            # Measure total time for a single camera
            frame_start_time = time.time()
            
            # Measure connection time
            conn_start_time = time.time()
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    # Connection timeout
                    sock.settimeout(SOCKET_TIMEOUT)
                    sock.connect((ip_address, OAK_PORT))
                    conn_time = time.time() - conn_start_time
                    log_timing(ip_address, "CONNECT", conn_start_time, conn_time)
                    connection_logger.info(f"CONNECTED,{ip_address}")
                    
                    # Read header
                    header_start_time = time.time()
                    HLEN = 32
                    
                    try:
                        header_data = receive_data_with_timeout(sock, HLEN, SOCKET_TIMEOUT * 2)
                        header = str(header_data, encoding="ascii")
                        header_time = time.time() - header_start_time
                        log_timing(ip_address, "HEADER_READ", header_start_time, header_time)
                        
                        # Parse header
                        header_s = re.split(' +', header)
                        if header_s[0] == "HEAD":
                            MESLEN = int(header_s[1])
                            
                            # Read data
                            data_start_time = time.time()
                            
                            try:
                                data = receive_data_with_timeout(sock, MESLEN, SOCKET_TIMEOUT * 2)
                                msg = str(data, encoding="ascii")
                                data_time = time.time() - data_start_time
                                log_timing(ip_address, "DATA_READ", data_start_time, data_time)
                                
                                # Parse data
                                parse_start_time = time.time()
                                distances = []
                                dets = msg.split("|")
                                for det in dets:
                                    distances.append(float(det))
                                parse_time = time.time() - parse_start_time
                                log_timing(ip_address, "PARSE", parse_start_time, parse_time)
                                
                                # Calculate total frame time for a single camera
                                total_frame_time = time.time() - frame_start_time
                                log_timing(ip_address, "TOTAL_FRAME", frame_start_time, total_frame_time)
                                
                                # Update global data
                                with data_lock:
                                    camera_data[ip_address]['distances'] = distances
                                    camera_data[ip_address]['last_update'] = time.time()
                                    camera_data[ip_address]['active'] = True
                                    camera_data[ip_address]['frame_count'] += 1
                                    camera_data[ip_address]['last_frame_time'] = total_frame_time
                                    
                                    # Update average time (moving average)
                                    if camera_data[ip_address]['frame_count'] == 1:
                                        camera_data[ip_address]['avg_time'] = total_frame_time
                                    else:
                                        # Weighted average: 90% previous average + 10% new measurement
                                        camera_data[ip_address]['avg_time'] = (0.9 * camera_data[ip_address]['avg_time'] + 
                                                                             0.1 * total_frame_time)
                                
                                # Update frame information for synchronization
                                with frame_completion_lock:
                                    # Add this camera to the list of completed ones for the current frame
                                    frame_data["cameras_completed"].add(ip_address)
                                    
                                    # Check if this is the slowest camera
                                    if total_frame_time > frame_data["slowest_time"]:
                                        frame_data["slowest_camera"] = ip_address
                                        frame_data["slowest_time"] = total_frame_time
                                    
                                    # If all cameras have finished receiving
                                    if set([OAK_IP1, OAK_IP2, OAK_IP3]) <= frame_data["cameras_completed"]:
                                        # Calculate total frame time (from start to completion of the last camera)
                                        frame_data["total_time"] = time.time() - frame_data["start_time"]
                                        
                                        # Log total time and information about the slowest camera
                                        log_combined_timing(
                                            frame_data["current_id"], 
                                            f"COMBINED_FRAME,{frame_data['slowest_camera']}", 
                                            frame_data["total_time"]
                                        )
                                        
                                        # Update frame ID
                                        frame_data["current_id"] += 1
                                        frame_times_queue.put({
                                            "id": frame_data["current_id"],
                                            "time": frame_data["total_time"],
                                            "slowest": frame_data["slowest_camera"]
                                        })
                                        
                                        # Reset the list of completed cameras
                                        frame_data["cameras_completed"] = set()
                                        frame_data["slowest_time"] = 0
                            
                            except (socket.timeout, RuntimeError) as e:
                                error_msg = f"Error reading data from {ip_address}: {str(e)}"
                                logger.warning(error_msg)
                                connection_logger.info(f"DATA_ERROR,{ip_address},{str(e)}")
                                log_timing(ip_address, "DATA_READ", data_start_time, 0)
                                log_timing(ip_address, "PARSE", time.time(), 0)
                                log_timing(ip_address, "TOTAL_FRAME", frame_start_time, time.time() - frame_start_time)
                                with data_lock:
                                    camera_data[ip_address]['active'] = False
                        
                        else:
                            error_msg = f"Received invalid header from {ip_address}: {header}"
                            logger.warning(error_msg)
                            connection_logger.info(f"INVALID_HEADER,{ip_address}")
                            with data_lock:
                                camera_data[ip_address]['active'] = False
                    
                    except (socket.timeout, RuntimeError) as e:
                        error_msg = f"Error reading header from {ip_address}: {str(e)}"
                        logger.warning(error_msg)
                        connection_logger.info(f"HEADER_ERROR,{ip_address},{str(e)}")
                        log_timing(ip_address, "HEADER_READ", header_start_time, 0)
                        log_timing(ip_address, "DATA_READ", time.time(), 0)
                        log_timing(ip_address, "PARSE", time.time(), 0)
                        log_timing(ip_address, "TOTAL_FRAME", frame_start_time, time.time() - frame_start_time)
                        with data_lock:
                            camera_data[ip_address]['active'] = False
            
            except (socket.timeout, ConnectionRefusedError) as e:
                error_msg = f"Connection error to {ip_address}: {str(e)}"
                logger.warning(error_msg)
                connection_logger.info(f"CONNECTION_ERROR,{ip_address},{str(e)}")
                log_timing(ip_address, "CONNECT", conn_start_time, 0)
                log_timing(ip_address, "HEADER_READ", time.time(), 0)
                log_timing(ip_address, "DATA_READ", time.time(), 0)
                log_timing(ip_address, "PARSE", time.time(), 0)
                log_timing(ip_address, "TOTAL_FRAME", frame_start_time, time.time() - frame_start_time)
                with data_lock:
                    camera_data[ip_address]['active'] = False
            
            # After each attempt (regardless of success or failure), wait a moment
            time.sleep(0.05)
                
        except Exception as e:
            error_msg = f"Unexpected error in camera thread {ip_address}: {str(e)}"
            logger.error(error_msg)
            connection_logger.info(f"THREAD_ERROR,{ip_address},{str(e)}")
            with data_lock:
                camera_data[ip_address]['active'] = False
            time.sleep(0.5)  # Longer pause for unexpected errors

def check_segments_presence(heatmap1, heatmap2, heatmap3, mask1, mask2, mask3):
    """Check for object presence in segments"""
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
    """Create a heatmap from data from three cameras"""
    try:
        # Transform data to matrices
        heatmap1 = np.array(distances1).reshape(nV, nH)
        heatmap2 = np.array(distances2).reshape(nV, nH)
        heatmap3 = np.array(distances3).reshape(nV, nH)
        
        # Apply filtering thresholds
        mask1 = (heatmap1 >= thresholds[OAK_IP1]['min']) & (heatmap1 <= thresholds[OAK_IP1]['max'])
        mask2 = (heatmap2 >= thresholds[OAK_IP2]['min']) & (heatmap2 <= thresholds[OAK_IP2]['max'])
        mask3 = (heatmap3 >= thresholds[OAK_IP3]['min']) & (heatmap3 <= thresholds[OAK_IP3]['max'])
        
        # Check presence in segments
        presence = check_segments_presence(heatmap1, heatmap2, heatmap3, mask1, mask2, mask3)
        
        # Combine data from three cameras
        heatmap = np.hstack((heatmap1, heatmap2, heatmap3))
        mask = np.hstack((mask1, mask2, mask3))
        
        # Normalization for display
        min_dist = np.min(heatmap)
        max_dist = np.max(heatmap)
        normalized = ((heatmap - min_dist) / (max_dist - min_dist) * 255).astype(np.uint8)
        
        # Apply color map
        heatmap_colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Apply mask
        filtered_heatmap = heatmap_colored.copy()
        filtered_heatmap[~mask] = [0, 0, 0]
        
        # Scale for display
        scale_factor = 30
        heatmap_scaled = cv2.resize(filtered_heatmap, 
                                  (3*nH * scale_factor, nV * scale_factor), 
                                  interpolation=cv2.INTER_NEAREST)
        
        # Add segment lines
        segment_width = nH * scale_factor // SEGMENTS_H
        
        for oak_idx in range(3):  # 3 columns (IP1, IP2, IP3)
            oak_x_start = oak_idx * nH * scale_factor
            
            for h_idx in range(1, SEGMENTS_H):
                x = oak_x_start + h_idx * segment_width
                cv2.line(heatmap_scaled, (x, 0), (x, nV*scale_factor), (255, 255, 255), 1)
        
        # Add lines separating cameras
        for i in range(1, 3):
            x = i * nH * scale_factor
            cv2.line(heatmap_scaled, (x, 0), (x, nV*scale_factor), (255, 255, 255), 2)
        
        # Add distance values
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
        
        # Add information about presence in segments
        for segment_idx in range(9):
            oak_idx = segment_idx // SEGMENTS_H
            segment_in_oak = segment_idx % SEGMENTS_H
            
            segment_x = oak_idx * nH * scale_factor + segment_in_oak * segment_width + segment_width // 2
            segment_y = nV * scale_factor - 10
            
            color = (0, 255, 0) if presence[segment_idx] else (0, 0, 255)  # Green for 1, Red for 0
            
            cv2.putText(heatmap_scaled, f"S{segment_idx+1}:{presence[segment_idx]}", 
                       (segment_x - 25, segment_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return heatmap_scaled, presence
    except Exception as e:
        logger.error(f"Error creating heatmap: {str(e)}")
        # Return an empty map in case of error
        empty_map = np.zeros((nV * 30, 3 * nH * 30, 3), dtype=np.uint8)
        return empty_map, [0] * 9

def main():
    """Main program function"""
    global running, active_ip_idx
    
    # Start threads receiving data from cameras
    threads = []
    for ip in [OAK_IP1, OAK_IP2, OAK_IP3]:
        thread = threading.Thread(target=camera_receiver, args=(ip,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    logger.info("Started receiving data from cameras")
    print("Controls:")
    print("  1 - select IP1 to modify thresholds")
    print("  2 - select IP2 to modify thresholds")
    print("  3 - select IP3 to modify thresholds")
    print("  z - decrease maximum threshold for active IP by 0.1")
    print("  x - increase maximum threshold for active IP by 0.1")
    print("  a - decrease minimum threshold for active IP by 0.1")
    print("  s - increase minimum threshold for active IP by 0.1")
    print("  q - exit program")
    
    active_ip = active_ip_list[active_ip_idx]
    
    # Main display loop
    last_stats_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Combined frame time data
    combined_times = []
    avg_combined_time = 0
    
    while True:
        try:
            start_time = time.time()
            
            # Get data from frame times queue
            while not frame_times_queue.empty():
                frame_time_data = frame_times_queue.get()
                combined_times.append(frame_time_data["time"])
                # Keep only the last 30 measurements
                if len(combined_times) > 30:
                    combined_times.pop(0)
                
                # Calculate average time
                if combined_times:
                    avg_combined_time = sum(combined_times) / len(combined_times)
            
            # Get data from cameras
            with data_lock:
                distances1 = camera_data[OAK_IP1]['distances']
                distances2 = camera_data[OAK_IP2]['distances']
                distances3 = camera_data[OAK_IP3]['distances']
                
                # Camera status information
                camera_status = []
                for i, ip in enumerate([OAK_IP1, OAK_IP2, OAK_IP3]):
                    status = "✓" if camera_data[ip]['active'] else "✗"
                    avg_time = camera_data[ip]['avg_time'] * 1000  # in ms
                    last_time = camera_data[ip]['last_frame_time'] * 1000  # in ms
                    camera_status.append(f"IP{i+1}: {status} (avg: {avg_time:.1f}ms, last: {last_time:.1f}ms)")
            
            # Create heatmap
            heatmap, presence = create_heatmap(distances1, distances2, distances3)
            
            # FPS counter
            fps_counter += 1
            if time.time() - last_stats_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                last_stats_time = time.time()
            
            # Threshold information
            thresh_info = []
            for i, ip in enumerate([OAK_IP1, OAK_IP2, OAK_IP3]):
                active_mark = "→ " if ip == active_ip else ""
                thresh_info.append(f"{active_mark}IP{i+1}: min={thresholds[ip]['min']:.1f} max={thresholds[ip]['max']:.1f}")
            
            # Add information to the image
            for i, info in enumerate(camera_status):
                cv2.putText(heatmap, info, (10, 20 + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset = len(camera_status) * 20 + 30
            
            # Add information about combined time of all cameras
            cv2.putText(heatmap, f"Full frame time: {avg_combined_time*1000:.1f}ms", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            y_offset += 25
            
            for i, info in enumerate(thresh_info):
                color = (0, 255, 255) if i == active_ip_idx else (255, 255, 255)
                cv2.putText(heatmap, info, (10, y_offset + i*20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add FPS information
            cv2.putText(heatmap, f"FPS: {fps}", (10, y_offset + len(thresh_info)*20 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show heatmap
            cv2.imshow("Depth Heatmap", heatmap)
            
            # Key handling
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('1'):
                active_ip_idx = 0
                active_ip = active_ip_list[active_ip_idx]
                logger.info(f"Selected IP1 for threshold modification")
            elif key == ord('2'):
                active_ip_idx = 1
                active_ip = active_ip_list[active_ip_idx]
                logger.info(f"Selected IP2 for threshold modification")
            elif key == ord('3'):
                active_ip_idx = 2
                active_ip = active_ip_list[active_ip_idx]
                logger.info(f"Selected IP3 for threshold modification")
            elif key == ord('z'):
                # Decrease maximum threshold by 0.1 for active IP
                thresholds[active_ip]['max'] = max(thresholds[active_ip]['min'], thresholds[active_ip]['max'] - 0.1)
                logger.info(f"Decreased maximum threshold for {active_ip}: {thresholds[active_ip]['max']:.1f}")
            elif key == ord('x'):
                # Increase maximum threshold by 0.1 for active IP
                thresholds[active_ip]['max'] = thresholds[active_ip]['max'] + 0.1
                logger.info(f"Increased maximum threshold for {active_ip}: {thresholds[active_ip]['max']:.1f}")
            elif key == ord('a'):
                # Decrease minimum threshold by 0.1 for active IP
                thresholds[active_ip]['min'] = max(0.1, thresholds[active_ip]['min'] - 0.1)
                logger.info(f"Decreased minimum threshold for {active_ip}: {thresholds[active_ip]['min']:.1f}")
            elif key == ord('s'):
                # Increase minimum threshold by 0.1 for active IP
                thresholds[active_ip]['min'] = min(thresholds[active_ip]['max'] - 0.1, thresholds[active_ip]['min'] + 0.1)
                logger.info(f"Increased minimum threshold for {active_ip}: {thresholds[active_ip]['min']:.1f}")
            
            # Wait until the end of the cycle to maintain constant refresh rate
            elapsed = time.time() - start_time
            if elapsed < 0.033:  # Try to maintain ~30 FPS
                time.sleep(0.033 - elapsed)
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(0.1)
    
    # Program termination
    running = False
    logger.info("Program termination")
    cv2.destroyAllWindows()
    
    # Wait for threads to finish
    for thread in threads:
        thread.join(1.0)

if __name__ == "__main__":
    main()

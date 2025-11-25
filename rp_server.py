"""
Red Pitaya Lock-In Server - Python 3.7+
Communicates with Python 2.7 client via socket
"""

import socket
import json
import threading
import time
import numpy as np
from pyrpl import Pyrpl

class RedPitayaServer:
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.rp = None
        self.lockin = None
        self.scope = None
        self.running = False
        self.data_buffer = {
            'mag': [],
            'phase': [],
            'X': [],
            'Y': []
        }
        self.lock = threading.Lock()
        
    def initialize_rp(self):
        """Initialize Red Pitaya hardware"""
        self.rp = Pyrpl(config='lockin_config', hostname='rp-f073ce.local')
        self.lockin = self.rp.rp.iq2
        self.scope = self.rp.rp.scope
        
        # Configure scope to read lock-in outputs
        self.scope.input1 = 'iq2'    # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = 8192
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        
        print("Red Pitaya initialized")
        
    def setup_lockin(self, params):
        """Configure lock-in amplifier"""
        frequency = params.get('frequency', 500)  # Hz
        amplitude = params.get('amplitude', 0.2)  # V
        bandwidth = params.get('bandwidth', 10)   # Hz
        phase = params.get('phase', 0)            # degrees
        
        # Turn off ASG0 - IQ module generates reference
        self.rp.rp.asg0.output_direct = 'off'
        
        # Setup IQ module
        self.lockin.setup(
            frequency=frequency,
            bandwidth=bandwidth,
            gain=0.0,
            phase=phase,
            acbandwidth=0,
            amplitude=amplitude,
            input='in1',
            output_direct='out1',
            output_signal='quadrature',
            quadrature_factor=1
        )
        
        print(f"Lock-in configured: {frequency} Hz, {amplitude} V, BW={bandwidth} Hz")
        
    def acquire_data(self):
        """Continuously acquire data from lock-in"""
        while self.running:
            try:
                self.scope.single()
                X = np.array(self.scope._data_ch1_current)  # iq2
                Y = np.array(self.scope._data_ch2_current)  # iq2_2
                
                # Calculate magnitude and phase
                R = np.sqrt(X**2 + Y**2)
                Theta = np.arctan2(Y, X)
                
                with self.lock:
                    self.data_buffer['X'].extend(X.tolist())
                    self.data_buffer['Y'].extend(Y.tolist())
                    self.data_buffer['mag'].extend(R.tolist())
                    self.data_buffer['phase'].extend(Theta.tolist())
                    
                    # Keep buffer size manageable
                    max_buffer = 10000
                    for key in self.data_buffer:
                        if len(self.data_buffer[key]) > max_buffer:
                            self.data_buffer[key] = self.data_buffer[key][-max_buffer:]
                
                time.sleep(0.01)  # Adjust as needed
                
            except Exception as e:
                print(f"Acquisition error: {e}")
                time.sleep(0.1)
    
    def handle_client(self, conn):
        """Handle commands from Python 2.7 client"""
        print("Client connected")
        
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                    
                try:
                    message = json.loads(data.decode('utf-8'))
                    command = message.get('command')
                    
                    if command == 'initialize':
                        self.initialize_rp()
                        response = {'status': 'ok', 'message': 'Initialized'}
                        
                    elif command == 'setup':
                        params = message.get('params', {})
                        self.setup_lockin(params)
                        response = {'status': 'ok', 'message': 'Setup complete'}
                        
                    elif command == 'start':
                        self.running = True
                        self.acq_thread = threading.Thread(target=self.acquire_data)
                        self.acq_thread.start()
                        response = {'status': 'ok', 'message': 'Acquisition started'}
                        
                    elif command == 'stop':
                        self.running = False
                        if hasattr(self, 'acq_thread'):
                            self.acq_thread.join()
                        response = {'status': 'ok', 'message': 'Acquisition stopped'}
                        
                    elif command == 'get_data':
                        with self.lock:
                            # Return recent data and clear buffer
                            response = {
                                'status': 'ok',
                                'data': self.data_buffer.copy()
                            }
                            # Clear buffer after sending
                            for key in self.data_buffer:
                                self.data_buffer[key] = []
                    
                    elif command == 'get_latest':
                        # Get just the most recent values
                        with self.lock:
                            n = min(100, len(self.data_buffer['mag']))
                            response = {
                                'status': 'ok',
                                'mag': self.data_buffer['mag'][-n:] if n > 0 else [],
                                'phase': self.data_buffer['phase'][-n:] if n > 0 else [],
                                'X': self.data_buffer['X'][-n:] if n > 0 else [],
                                'Y': self.data_buffer['Y'][-n:] if n > 0 else []
                            }
                    
                    elif command == 'shutdown':
                        response = {'status': 'ok', 'message': 'Shutting down'}
                        conn.sendall(json.dumps(response).encode('utf-8'))
                        break
                        
                    else:
                        response = {'status': 'error', 'message': f'Unknown command: {command}'}
                    
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    
                except json.JSONDecodeError as e:
                    error_response = {'status': 'error', 'message': f'JSON error: {str(e)}'}
                    conn.sendall(json.dumps(error_response).encode('utf-8'))
                    
        except Exception as e:
            print(f"Client handler error: {e}")
        finally:
            conn.close()
            print("Client disconnected")
    
    def start_server(self):
        """Start the socket server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(1)
        
        print(f"Red Pitaya server listening on {self.host}:{self.port}")
        
        try:
            while True:
                conn, addr = server_socket.accept()
                print(f"Connection from {addr}")
                client_thread = threading.Thread(target=self.handle_client, args=(conn,))
                client_thread.start()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            server_socket.close()

if __name__ == '__main__':
    server = RedPitayaServer()
    server.start_server()

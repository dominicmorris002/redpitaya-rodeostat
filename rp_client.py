# -*- coding: utf-8 -*-
"""
Red Pitaya Client - Python 2.7 compatible
Communicates with Python 3 server running Red Pitaya
"""

import socket
import json
import numpy as np
import threading
import time

class RedPitayaClient(object):
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
        # Data buffers matching SEED interface
        self.mag = np.array([])
        self.phase = np.array([])
        self.X = np.array([])
        self.Y = np.array([])
        self.sensitivity = 1.0  # Compatibility with SEED code
        
    def connect(self):
        """Connect to Red Pitaya server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print("Connected to Red Pitaya server")
            return True
        except Exception as e:
            print("Connection failed: {}".format(e))
            return False
    
    def send_command(self, command, params=None):
        """Send command to server and get response"""
        if not self.connected:
            return {'status': 'error', 'message': 'Not connected'}
        
        try:
            message = {'command': command}
            if params:
                message['params'] = params
            
            self.socket.sendall(json.dumps(message).encode('utf-8'))
            
            response_data = self.socket.recv(65536)
            response = json.loads(response_data.decode('utf-8'))
            return response
            
        except Exception as e:
            print("Command error: {}".format(e))
            return {'status': 'error', 'message': str(e)}
    
    def initialize(self):
        """Initialize Red Pitaya hardware"""
        return self.send_command('initialize')
    
    def setup(self, frequency=500, amplitude=0.2, bandwidth=10, phase=0):
        """Setup lock-in parameters"""
        params = {
            'frequency': frequency,
            'amplitude': amplitude,
            'bandwidth': bandwidth,
            'phase': phase
        }
        return self.send_command('setup', params)
    
    def start_acquisition(self):
        """Start data acquisition"""
        return self.send_command('start')
    
    def stop_acquisition(self):
        """Stop data acquisition"""
        return self.send_command('stop')
    
    def get_data(self):
        """Get all buffered data from server"""
        response = self.send_command('get_data')
        if response['status'] == 'ok':
            data = response['data']
            
            # Append to existing arrays
            if len(data['mag']) > 0:
                self.mag = np.append(self.mag, np.array(data['mag']))
                self.phase = np.append(self.phase, np.array(data['phase']))
                self.X = np.append(self.X, np.array(data['X']))
                self.Y = np.append(self.Y, np.array(data['Y']))
            
            return len(data['mag'])
        return 0
    
    def get_latest(self):
        """Get just the most recent data points"""
        response = self.send_command('get_latest')
        if response['status'] == 'ok':
            return {
                'mag': np.array(response['mag']),
                'phase': np.array(response['phase']),
                'X': np.array(response['X']),
                'Y': np.array(response['Y'])
            }
        return None
    
    def disconnect(self):
        """Disconnect from server"""
        if self.connected:
            self.send_command('shutdown')
            self.socket.close()
            self.connected = False
            print("Disconnected from Red Pitaya server")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()


# Test code
if __name__ == '__main__':
    client = RedPitayaClient()
    
    if client.connect():
        print("Initializing...")
        print(client.initialize())
        
        print("Setting up lock-in at 500 Hz, 0.2 V...")
        print(client.setup(frequency=500, amplitude=0.2))
        
        print("Starting acquisition...")
        print(client.start_acquisition())
        
        # Collect data for a few seconds
        for i in range(10):
            time.sleep(0.5)
            n = client.get_data()
            print("Collected {} samples, total: {}".format(n, len(client.mag)))
        
        print("Stopping...")
        print(client.stop_acquisition())
        
        print("\nFinal data shapes:")
        print("  mag: {}".format(client.mag.shape))
        print("  phase: {}".format(client.phase.shape))
        print("  X: {}".format(client.X.shape))
        print("  Y: {}".format(client.Y.shape))
        
        if len(client.mag) > 0:
            print("\nMean magnitude: {:.6f} V".format(np.mean(client.mag)))
        
        client.disconnect()

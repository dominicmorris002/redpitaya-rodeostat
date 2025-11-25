"""
redpitaya_bridge.py - Python 3.7+ Bridge for Red Pitaya
Run this with Python 3, it will communicate with your Python 2.7 script via socket

Usage:
    python redpitaya_bridge.py

Requirements (Python 3 only):
    pip install pyrpl numpy
"""

import socket
import json
import time
import numpy as np
from pyrpl import Pyrpl
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
BRIDGE_HOST = 'localhost'
BRIDGE_PORT = 9999
RED_PITAYA_HOSTNAME = 'rp-f073ce.local'  # UPDATE THIS


# ============================================================================


class RedPitayaBridge:
    """Bridge server that runs in Python 3 and provides Red Pitaya access"""

    def __init__(self, rp_hostname):
        print("=" * 60)
        print("RED PITAYA BRIDGE SERVER (Python 3)")
        print("=" * 60)
        print(f"Python version: {sys.version}")
        print(f"Connecting to Red Pitaya: {rp_hostname}")

        # Initialize Red Pitaya with PyRPL
        self.rp = Pyrpl(config='lockin_bridge_config', hostname=rp_hostname)
        self.rp_modules = self.rp.rp
        self.lock_in = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope = self.rp_modules.scope

        print("Red Pitaya initialized successfully!")
        print("=" * 60)

        # Configuration storage
        self.frequency = 500
        self.amplitude = 0.2
        self.is_configured = False

    def setup_lockin(self, frequency, amplitude, filter_bw=10):
        """Setup lock-in amplifier"""
        self.frequency = frequency
        self.amplitude = amplitude

        # Turn off ASG0
        self.ref_sig.output_direct = 'off'

        # Setup IQ module
        self.lock_in.setup(
            frequency=frequency,
            bandwidth=filter_bw,
            gain=0.0,
            phase=0,
            acbandwidth=0,
            amplitude=amplitude,
            input='in1',
            output_direct='out1',
            output_signal='quadrature',
            quadrature_factor=1
        )

        # Configure scope
        self.scope.input1 = 'iq2'  # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = 64
        self.scope._start_acquisition_rolling_mode()

        self.is_configured = True
        print(f"Lock-in configured: {frequency} Hz @ {amplitude} V")

        return {"status": "ok", "frequency": frequency, "amplitude": amplitude}

    def get_XY(self):
        """Get current X and Y values"""
        try:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)  # X
            ch2 = np.array(self.scope._data_ch2_current)  # Y
            X = float(np.mean(ch1))
            Y = float(np.mean(ch2))
            return {"status": "ok", "X": X, "Y": Y}
        except Exception as e:
            return {"status": "error", "message": str(e), "X": 0.0, "Y": 0.0}

    def handle_command(self, command):
        """Handle incoming commands from Python 2.7"""
        try:
            cmd = command.get('cmd', '')

            if cmd == 'setup':
                freq = command.get('frequency', 500)
                amp = command.get('amplitude', 0.2)
                bw = command.get('bandwidth', 10)
                return self.setup_lockin(freq, amp, bw)

            elif cmd == 'get_xy':
                return self.get_XY()

            elif cmd == 'ping':
                return {"status": "ok", "message": "pong"}

            elif cmd == 'shutdown':
                return {"status": "ok", "message": "shutting down"}

            else:
                return {"status": "error", "message": f"Unknown command: {cmd}"}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def run_server(self, host='localhost', port=9999):
        """Run the bridge server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(1)

        print(f"Bridge server listening on {host}:{port}")
        print("Waiting for Python 2.7 client connection...")
        print("=" * 60)

        while True:
            client_socket, address = server_socket.accept()
            print(f"Client connected from {address}")

            try:
                while True:
                    # Receive command
                    data = client_socket.recv(4096)
                    if not data:
                        break

                    # Parse JSON command
                    command = json.loads(data.decode('utf-8'))

                    # Handle command
                    response = self.handle_command(command)

                    # Send response
                    client_socket.sendall(json.dumps(response).encode('utf-8') + b'\n')

                    # Check for shutdown
                    if command.get('cmd') == 'shutdown':
                        print("Shutdown command received")
                        client_socket.close()
                        server_socket.close()
                        return

            except Exception as e:
                print(f"Error handling client: {e}")
            finally:
                client_socket.close()
                print("Client disconnected")


if __name__ == '__main__':
    try:
        bridge = RedPitayaBridge(RED_PITAYA_HOSTNAME)
        bridge.run_server(BRIDGE_HOST, BRIDGE_PORT)
    except KeyboardInterrupt:
        print("\nShutting down bridge server...")
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback

        traceback.print_exc()

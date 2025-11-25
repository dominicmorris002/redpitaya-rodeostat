"""
LIC_Redpitaya.py - Python 2.7 Compatible with Python 3 Bridge
Uses a Python 3 bridge process to communicate with Red Pitaya via PyRPL

SETUP:
1. Start the Python 3 bridge first:
   python3 redpitaya_bridge.py

2. Then run this Python 2.7 script:
   python2 LIC_Redpitaya.py

Requirements:
    Python 2.7: PyDAQmx numpy matplotlib
    Python 3.7+: pyrpl numpy (separate environment)
"""

import threading
import time
from PyDAQmx import *
import ctypes
import numpy as np
import socket
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import csv

# ============================================================================
# USER SETTINGS
# ============================================================================
BRIDGE_HOST = 'rp-f073ce.local'  # Bridge server address
BRIDGE_PORT = 9999  # Bridge server port
MEASUREMENT_DURATION = 12  # seconds
LOCK_IN_FREQUENCY = 500  # Hz
LOCK_IN_AMPLITUDE = 0.2  # V
FILTER_BANDWIDTH = 10  # Hz
SAMPLE_RATE = 500  # samples per second
DATA_BUFFER_LENGTH = 250  # samples per buffer

SAVE_DIRECTORY = 'C:\\SEED 3.2 Data\\Joydip\\e2025\\Novenber\\Batch 3 Chip testing'
AUTO_SAVE = True
AUTO_SAVE_FILENAME = 'measurement'


# ============================================================================


class RedPitayaClient(object):
    """Client to communicate with Python 3 bridge"""

    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        """Connect to bridge server"""
        print("Connecting to Red Pitaya bridge at {}:{}".format(self.host, self.port))
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))

            # Test connection
            response = self.send_command({'cmd': 'ping'})
            if response.get('status') == 'ok':
                print("Successfully connected to Red Pitaya bridge!")
            else:
                raise Exception("Bridge not responding correctly")
        except socket.error as e:
            print("=" * 60)
            print("ERROR: Cannot connect to Python 3 bridge!")
            print("=" * 60)
            print("Make sure you start the bridge first:")
            print("  1. Open a NEW terminal/command prompt")
            print("  2. Activate Python 3 environment")
            print("  3. Run: python redpitaya_bridge.py")
            print("=" * 60)
            raise

    def send_command(self, command):
        """Send command and get response"""
        try:
            # Send command
            self.socket.sendall(json.dumps(command).encode('utf-8'))

            # Receive response
            data = self.socket.recv(4096)
            response = json.loads(data.decode('utf-8'))
            return response
        except Exception as e:
            print("Bridge communication error: {}".format(e))
            return {'status': 'error', 'message': str(e)}

    def setup(self, frequency, amplitude, bandwidth=10):
        """Setup lock-in amplifier"""
        cmd = {
            'cmd': 'setup',
            'frequency': frequency,
            'amplitude': amplitude,
            'bandwidth': bandwidth
        }
        return self.send_command(cmd)

    def get_XY(self):
        """Get X and Y values"""
        response = self.send_command({'cmd': 'get_xy'})
        if response.get('status') == 'ok':
            return response.get('X', 0.0), response.get('Y', 0.0)
        else:
            return 0.0, 0.0

    def close(self):
        """Close connection"""
        try:
            self.send_command({'cmd': 'shutdown'})
            self.socket.close()
        except:
            pass


class SEEDTask(Task):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        Task.__init__(self)

        self.dev_name = "Dev1"

        # Red Pitaya via Python 3 Bridge
        print("=" * 60)
        print("Initializing Red Pitaya via Python 3 Bridge...")
        print("=" * 60)

        try:
            self.rp = RedPitayaClient(BRIDGE_HOST, BRIDGE_PORT)
        except Exception as e:
            print("FATAL ERROR: Cannot connect to bridge")
            raise

        # Frequency and amplitude
        self.frequency = frequency
        self.amplitude = amplitude

        # Setup lock-in via bridge
        result = self.rp.setup(self.frequency, self.amplitude, FILTER_BANDWIDTH)
        if result.get('status') != 'ok':
            raise Exception("Failed to setup Red Pitaya: {}".format(result.get('message')))

        self.sensitivity = 1.0

        print("Red Pitaya Lock-in: {} Hz @ {} V".format(self.frequency, self.amplitude))
        print("Filter Bandwidth: {} Hz".format(FILTER_BANDWIDTH))
        print("=" * 60)

        time.sleep(0.5)

        self.rate = float(rate)
        self.duration = float(duration)
        self.time = np.arange(0, self.duration, 1 / self.rate)

        self.dataLen = data_len
        self._data = np.zeros((self.dataLen, 7))

        # Preallocate data arrays
        self.mag = np.zeros((int(self.duration) * int(self.rate),))
        self.dcRamp = np.zeros((int(self.duration) * int(self.rate),))
        self.phase = np.zeros((int(self.duration) * int(self.rate),))
        self.signal = np.zeros((int(self.duration) * int(self.rate),))
        self.X_rp = np.zeros((int(self.duration) * int(self.rate),))
        self.Y_rp = np.zeros((int(self.duration) * int(self.rate),))

        self.read = int32()
        self.compError = 0
        self.numRecord = 0

        # NI DAQ channels
        self.CreateAIVoltageChan(self.dev_name + "/ai0", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai1", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai2", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai3", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai4", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai5", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai6", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)

        self.CfgSampClkTiming("", self.rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.dataLen)

        self.Kp = 0
        self.Kd = 0
        self.Ki = 0
        self.units = 'pm'

    def get_red_pitaya_XY(self):
        """Get X and Y from Red Pitaya via bridge"""
        return self.rp.get_XY()

    def compensate(self):
        return 0

    def set_comp_voltage(self, v):
        return 0.0

    def continuousRecord(self):
        self._data_lock = threading.Lock()
        self._iP_lock = threading.Lock()
        self._newdata_event = threading.Event()
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.dataLen, 0)
        self.AutoRegisterDoneEvent(0)

    def EveryNCallback(self):
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                # Read NI DAQ
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, self._data, self.dataLen * 7,
                                   ctypes.byref(self.read), None)

                # Get Red Pitaya data
                for i in range(self.dataLen):
                    X, Y = self.get_red_pitaya_XY()

                    self.X_rp[self.numRecord + i] = X
                    self.Y_rp[self.numRecord + i] = Y

                    R = np.sqrt(X ** 2 + Y ** 2)
                    Theta = np.arctan2(Y, X)

                    self.mag[self.numRecord + i] = R
                    self.phase[self.numRecord + i] = Theta * 20

                self._newdata_event.set()

                # dcRamp from NI DAQ ai1
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]

                # Calculate signal
                self.signal[self.numRecord:self.numRecord + self.dataLen] = self.mag[
                                                                            self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.707

                self.compensate()
                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0

    def SetCompensatorStatus(self, status):
        self.CompensatorStatus = 0 if status == "off" else 1

    def DoneCallback(self, status):
        print("Status", status.value)
        return 0

    def get_data(self, blocking=True, timeout=None):
        if blocking:
            if not self._newdata_event.wait(timeout):
                raise ValueError("timeout waiting for data from device")
        with self._data_lock:
            self._newdata_event.clear()
            return self._data.copy()

    def getIntialVoltageList(self):
        return 0.0

    def initialNull(self):
        return 0

    def plotCompResponse(self):
        return 0

    def reset(self):
        self.mag = np.zeros((int(self.duration) * int(self.rate),))
        self.dcRamp = np.zeros((int(self.duration) * int(self.rate),))
        self.phase = np.zeros((int(self.duration) * int(self.rate),))
        self.signal = np.zeros((int(self.duration) * int(self.rate),))
        self.X_rp = np.zeros((int(self.duration) * int(self.rate),))
        self.Y_rp = np.zeros((int(self.duration) * int(self.rate),))
        self.read = int32()
        self.compError = 0
        self.numRecord = 0

    def cleanup(self):
        """Cleanup"""
        try:
            self.rp.close()
        except:
            pass


class LockInCurrent(SEEDTask):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        SEEDTask.__init__(self, duration, rate, data_len, frequency, amplitude)
        self.units = 'mA'

    def EveryNCallback(self):
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, self._data, self.dataLen * 7,
                                   ctypes.byref(self.read), None)

                for i in range(self.dataLen):
                    X, Y = self.get_red_pitaya_XY()

                    self.X_rp[self.numRecord + i] = X
                    self.Y_rp[self.numRecord + i] = Y

                    R = np.sqrt(X ** 2 + Y ** 2)
                    Theta = np.arctan2(Y, X)

                    self.mag[self.numRecord + i] = np.absolute(R)
                    self.phase[self.numRecord + i] = Theta * 20

                self._newdata_event.set()

                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                self.signal[self.numRecord:self.numRecord + self.dataLen] = self.mag[
                                                                            self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.707

                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0

    def compensate(self):
        return 0


def save_LIC(taskName, fileName):
    tempDir = os.getcwd()
    save_directory = SAVE_DIRECTORY

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print("Created directory: {}".format(save_directory))

    os.chdir(save_directory)
    mag = taskName.mag
    dcRamp = taskName.dcRamp
    time = taskName.time
    phase = taskName.phase
    signal = taskName.signal

    np.savez(fileName + '.npz', mag=mag, dcRamp=dcRamp, phase=phase, time=time, signal=signal)
    dat = loadSEEDDatanpz(fileName + '.npz')
    os.chdir(save_directory)

    if not os.path.exists(fileName):
        os.makedirs(fileName)

    os.chdir(save_directory + '\\' + fileName)
    csvGenerate(dat)
    os.chdir(tempDir)


def loadSEEDDatanpz(fileName):
    s = np.load(fileName)
    dcRamp = s['dcRamp']
    mag = s['mag']
    phase = s['phase']
    time = s['time']
    signal = s['signal']
    return {'dcRamp': dcRamp, 'magnitude': mag, 'phase': phase, 'time': time, 'signal': signal}


def csvGenerate(dat):
    for i in dat.keys():
        with open(i + '.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(dat[i])


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("RED PITAYA LOCK-IN CURRENT MEASUREMENT")
    print("Python 2.7 with Python 3 Bridge")
    print("=" * 60)
    print("Duration: {} seconds".format(MEASUREMENT_DURATION))
    print("Frequency: {} Hz".format(LOCK_IN_FREQUENCY))
    print("Amplitude: {} V".format(LOCK_IN_AMPLITUDE))
    print("Sample Rate: {} Hz".format(SAMPLE_RATE))
    print("=" * 60)

    task = LockInCurrent(
        duration=MEASUREMENT_DURATION,
        rate=SAMPLE_RATE,
        data_len=DATA_BUFFER_LENGTH,
        frequency=LOCK_IN_FREQUENCY,
        amplitude=LOCK_IN_AMPLITUDE
    )

    # Setup plotting
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(task.units, color=(31 / 255., 119 / 255., 180 / 255.))
    ax1.set_ylabel('DC Ramp', color=(255 / 255., 127 / 255., 14 / 255.))

    task.continuousRecord()
    task.StartTask()

    plt.show()


    def animate(i):
        ax2.plot(task.time[0:task.numRecord], task.signal[0:task.numRecord], color=(31 / 255., 119 / 255., 180 / 255.))
        ax1.plot(task.time[0:task.numRecord], task.dcRamp[0:task.numRecord], color=(255 / 255., 127 / 255., 14 / 255.))


    ani = animation.FuncAnimation(fig, animate, interval=50)
    plt.show()

    task.cleanup()

    if AUTO_SAVE:
        print("=" * 60)
        print("SAVING DATA...")
        print("=" * 60)
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "{}_{}".format(AUTO_SAVE_FILENAME, timestamp)
            save_LIC(task, filename)
            print("Data saved successfully as: {}".format(filename))
            print("Location: {}".format(SAVE_DIRECTORY))
            print("=" * 60)
        except Exception as e:
            print("Error saving data: {}".format(e))
            print("=" * 60)

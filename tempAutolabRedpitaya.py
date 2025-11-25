"""
LIC_Redpitaya.py - Python 2.7 Compatible Version
Complete Lock-In Current Measurement System with Red Pitaya

SETUP:
- Red Pitaya generates AC on OUT1 and measures on IN1
- NI DAQ reads dcRamp from Autolab on ai1
- Real-time plotting of signal vs dcRamp

USAGE:
Run in IPython: run LIC_Redpitaya

pip install PyDAQmx numpy matplotlib pyrpl pyvisa scipy pyqtgraph PyQt5
"""

import threading
import time
from PyDAQmx import *
import ctypes
import numpy as np
from pyrpl import Pyrpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import csv


# ============================================================================
# USER SETTINGS - CHANGE THESE
# ============================================================================
RED_PITAYA_HOSTNAME = 'rp-f073ce.local'  # UPDATE THIS to your Red Pitaya IP/hostname
MEASUREMENT_DURATION = 12  # seconds
LOCK_IN_FREQUENCY = 500    # Hz
LOCK_IN_AMPLITUDE = 0.2    # V
FILTER_BANDWIDTH = 10      # Hz - lock-in filter bandwidth
SAMPLE_RATE = 500          # samples per second
DATA_BUFFER_LENGTH = 250   # samples per buffer

SAVE_DIRECTORY = 'C:\\SEED 3.2 Data\\Joydip\\e2025\\Novenber\\Batch 3 Chip testing'
AUTO_SAVE = True           # Set to True to automatically save data at end
AUTO_SAVE_FILENAME = 'measurement'  # Default filename for auto-save
# ============================================================================


class SEEDTask(Task):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        Task.__init__(self)

        self.dev_name = "Dev1"

        # Red Pitaya Lock-In Setup (replaces GPIB lock-in)
        print("=" * 60)
        print("Initializing Red Pitaya Lock-In...")
        print("=" * 60)
        self.rp = Pyrpl(config='lockin_config', hostname=RED_PITAYA_HOSTNAME)
        self.rp_modules = self.rp.rp
        self.lock_in = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope = self.rp_modules.scope

        # Frequency and amplitude (same parameters as GPIB version)
        self.frequency = frequency  # Hz
        self.amplitude = amplitude  # V
        
        # Lock-in amplifier settings
        filter_bw = FILTER_BANDWIDTH
        
        # Turn off ASG0 - IQ module handles signal generation
        self.ref_sig.output_direct = 'off'
        
        # Setup IQ module (equivalent to GPIB lock-in settings)
        self.lock_in.setup(
            frequency=self.frequency,
            bandwidth=filter_bw,
            gain=0.0,
            phase=0,
            acbandwidth=0,
            amplitude=self.amplitude,
            input='in1',
            output_direct='out1',  # Output AC signal on OUT1
            output_signal='quadrature',
            quadrature_factor=1
        )
        
        # Configure scope to read lock-in X and Y outputs
        self.scope.input1 = 'iq2'    # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = 64
        self.scope._start_acquisition_rolling_mode()
        self.sample_rate_rp = 125e6 / self.scope.decimation
        
        # Sensitivity (equivalent to GPIB SEN query)
        self.sensitivity = 1.0  # Red Pitaya outputs directly in Volts
        
        print("Red Pitaya Lock-in: {} Hz @ {} V".format(self.frequency, self.amplitude))
        print("Filter Bandwidth: {} Hz".format(filter_bw))
        print("Output: OUT1 | Input: IN1")
        print("=" * 60)
        
        # Wait for lock-in to settle
        time.sleep(0.5)

        self.rate = float(rate)
        self.duration = float(duration)
        self.time = np.arange(0, self.duration, 1 / self.rate)

        self.dataLen = data_len
        self._data = np.zeros((self.dataLen, 7))

        # Preallocate data logging arrays (same as original)
        self.mag = np.zeros((int(self.duration) * int(self.rate),))
        self.dcRamp = np.zeros((int(self.duration) * int(self.rate),))
        self.phase = np.zeros((int(self.duration) * int(self.rate),))
        self.signal = np.zeros((int(self.duration) * int(self.rate),))
        
        # Store X and Y from Red Pitaya
        self.X_rp = np.zeros((int(self.duration) * int(self.rate),))
        self.Y_rp = np.zeros((int(self.duration) * int(self.rate),))
        
        self.read = int32()
        self.compError = 0
        self.numRecord = 0

        # Create Voltage Channels (same as original)
        # ai1 - dcRamp from Autolab (main channel we need)
        self.CreateAIVoltageChan(self.dev_name + "/ai0", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai1", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai2", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai3", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai4", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai5", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai6", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)

        self.CfgSampClkTiming("", self.rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.dataLen)

        # PID Gains
        self.Kp = 0
        self.Kd = 0
        self.Ki = 0

        self.units = 'pm'

    def get_red_pitaya_XY(self):
        """Get current X and Y values from Red Pitaya scope"""
        try:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)  # X (in-phase)
            ch2 = np.array(self.scope._data_ch2_current)  # Y (quadrature)
            # Return mean values
            return np.mean(ch1), np.mean(ch2)
        except:
            return 0.0, 0.0

    def compensate(self):
        # Placeholder - same as original
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
                # Read NI DAQ (gets dcRamp and other channels)
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                
                # Get Red Pitaya X/Y data for each sample in the buffer
                for i in range(self.dataLen):
                    X, Y = self.get_red_pitaya_XY()
                    
                    # Store X and Y
                    self.X_rp[self.numRecord + i] = X
                    self.Y_rp[self.numRecord + i] = Y
                    
                    # Calculate magnitude (R) and phase (Theta) from X and Y
                    R = np.sqrt(X**2 + Y**2)
                    Theta = np.arctan2(Y, X)
                    
                    # Store in mag and phase arrays (replaces ai0 and ai4)
                    self.mag[self.numRecord + i] = R
                    self.phase[self.numRecord + i] = Theta * 20  # Scale factor same as original
                
                self._newdata_event.set()
                
                # Update dcRamp from NI DAQ ai1 (same as original)
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                
                # Calculate signal (same formula as original)
                self.signal[self.numRecord:self.numRecord + self.dataLen] = self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.707
                
                self.compensate()
                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0

    def SetCompensatorStatus(self, status):
        if status == "off":
            self.CompensatorStatus = 0
        else:
            self.CompensatorStatus = 1

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


class LockInCurrent(SEEDTask):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        SEEDTask.__init__(self, duration, rate, data_len, frequency, amplitude)
        self.units = 'mA'

    def EveryNCallback(self):
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                # Read NI DAQ (gets dcRamp and other channels)
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                
                # Get Red Pitaya X/Y data for each sample in the buffer
                for i in range(self.dataLen):
                    X, Y = self.get_red_pitaya_XY()
                    
                    # Store X and Y
                    self.X_rp[self.numRecord + i] = X
                    self.Y_rp[self.numRecord + i] = Y
                    
                    # Calculate magnitude (R) and phase (Theta) from X and Y
                    R = np.sqrt(X**2 + Y**2)
                    Theta = np.arctan2(Y, X)
                    
                    # Store in mag and phase arrays (replaces _data[:, 0] and _data[:, 4])
                    self.mag[self.numRecord + i] = np.absolute(R)
                    self.phase[self.numRecord + i] = Theta * 20  # Same scale factor as original
                
                self._newdata_event.set()
                
                # Update dcRamp from NI DAQ ai1 (same as original)
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                
                # Calculate signal (same formula as original)
                self.signal[self.numRecord:self.numRecord + self.dataLen] = self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.707
                
                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0

    def compensate(self):
        return 0


def save_LIC(taskName, fileName):
    tempDir = os.getcwd()
    save_directory = SAVE_DIRECTORY
    
    # Create the save directory if it doesn't exist
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
    print("=" * 60)
    print("Duration: {} seconds".format(MEASUREMENT_DURATION))
    print("Frequency: {} Hz".format(LOCK_IN_FREQUENCY))
    print("Amplitude: {} V".format(LOCK_IN_AMPLITUDE))
    print("Sample Rate: {} Hz".format(SAMPLE_RATE))
    print("=" * 60)
    
    # Create task with Red Pitaya lock-in
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

    # Start continuous recording
    task.continuousRecord()
    task.StartTask()

    plt.show()

    def animate(i):
        ax2.plot(task.time[0:task.numRecord], task.signal[0:task.numRecord], color=(31 / 255., 119 / 255., 180 / 255.))
        ax1.plot(task.time[0:task.numRecord], task.dcRamp[0:task.numRecord], color=(255 / 255., 127 / 255., 14 / 255.))

    ani = animation.FuncAnimation(fig, animate, interval=50)
    plt.show()
    
    # Auto-save data if enabled
    if AUTO_SAVE:
        print("=" * 60)
        print("SAVING DATA...")
        print("=" * 60)
        try:
            # Add timestamp to filename to avoid overwriting
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = "{}_{}".format(AUTO_SAVE_FILENAME, timestamp)
            save_LIC(task, filename)
            print("Data saved successfully as: {}".format(filename))
            print("Location: {}".format(SAVE_DIRECTORY))
            print("=" * 60)
        except Exception as e:
            print("Error saving data: {}".format(e))
            print("You can manually save with: save_LIC(task, 'your_filename')")
            print("=" * 60)

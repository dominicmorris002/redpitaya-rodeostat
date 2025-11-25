# -*- coding: utf-8 -*-
"""
Red Pitaya version of LIC_Measurement_Types
Same interface as original, just uses Red Pitaya instead of SR7280
"""

from PyDAQmx import *
import threading
import time
import ctypes
import numpy as np
import os
import csv
from rp_client import RedPitayaClient


class SEEDTask_RP(Task):
    """Red Pitaya version of SEEDTask - replaces SR7280 lock-in"""
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        Task.__init__(self)

        self.dev_name = "Dev1"

        # RED PITAYA LOCK-IN (replaces GPIB lock-in)
        print("Connecting to Red Pitaya...")
        self.lock_in = RedPitayaClient(host='localhost', port=5555, auto_start_server=True)
        
        # Wait for connection
        max_retries = 10
        for i in range(max_retries):
            if self.lock_in.connect():
                break
            print("Waiting for server... {}/{}".format(i+1, max_retries))
            time.sleep(1)
        
        if not self.lock_in.connected:
            raise RuntimeError("Could not connect to Red Pitaya server!")
        
        # Initialize Red Pitaya
        print("Initializing Red Pitaya...")
        result = self.lock_in.initialize()
        if result['status'] != 'ok':
            raise RuntimeError("Red Pitaya initialization failed: {}".format(result.get('message', 'Unknown error')))
        
        # Setup lock-in
        self.frequency = frequency  # Hz
        self.amplitude = amplitude  # V peak
        
        print("Configuring lock-in: {} Hz, {} V".format(frequency, amplitude))
        result = self.lock_in.setup(
            frequency=frequency,
            amplitude=amplitude,
            bandwidth=10,  # 10 Hz filter
            phase=0
        )
        if result['status'] != 'ok':
            raise RuntimeError("Lock-in setup failed: {}".format(result.get('message', 'Unknown error')))
        
        self.sensitivity = 1.0  # Red Pitaya already scaled
        self.rate = float(rate)
        self.duration = float(duration)
        self.time = np.arange(0, self.duration, 1 / self.rate)

        self.dataLen = data_len
        self._data = np.zeros((self.dataLen, 7))

        # Preallocate data arrays (same as original)
        self.mag = np.zeros((int(self.duration) * int(self.rate),))
        self.dcRamp = np.zeros((int(self.duration) * int(self.rate),))
        self.phase = np.zeros((int(self.duration) * int(self.rate),))
        self.signal = np.zeros((int(self.duration) * int(self.rate),))
        self.read = int32()
        self.compError = 0
        self.numRecord = 0

        # Create DAQ Voltage Channels (same as original)
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
        
        # Red Pitaya polling
        self.rp_running = False
        self.rp_thread = None

    def start_rp_acquisition(self):
        """Start Red Pitaya acquisition"""
        result = self.lock_in.start_acquisition()
        if result['status'] == 'ok':
            self.rp_running = True
            self.rp_thread = threading.Thread(target=self._poll_rp_data)
            self.rp_thread.daemon = True
            self.rp_thread.start()
            print("Red Pitaya acquisition started")

    def stop_rp_acquisition(self):
        """Stop Red Pitaya acquisition"""
        self.rp_running = False
        if self.rp_thread:
            self.rp_thread.join(timeout=2.0)
        self.lock_in.stop_acquisition()

    def _poll_rp_data(self):
        """Poll Red Pitaya data in background"""
        while self.rp_running:
            try:
                self.lock_in.get_data()
                time.sleep(0.02)
            except Exception as e:
                print("RP poll error: {}".format(e))
                time.sleep(0.1)

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
        self.start_rp_acquisition()

    def EveryNCallback(self):
        """Same as original but gets lock-in data from Red Pitaya"""
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                # Read DAQ (DC channels) - SAME AS ORIGINAL
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                                 self._data, self.dataLen * 7, 
                                 ctypes.byref(self.read), None)
                
                # Get Red Pitaya lock-in data (THIS REPLACES SR7280)
                rp_data = self.lock_in.get_latest()
                
                if rp_data is not None and len(rp_data['mag']) > 0:
                    # Match data length
                    n_rp = len(rp_data['mag'])
                    if n_rp >= self.dataLen:
                        # Use most recent data
                        mag_data = rp_data['mag'][-self.dataLen:]
                        phase_data = rp_data['phase'][-self.dataLen:]
                    else:
                        # Interpolate if needed
                        mag_data = np.interp(
                            np.linspace(0, n_rp - 1, self.dataLen),
                            np.arange(n_rp),
                            rp_data['mag']
                        )
                        phase_data = np.interp(
                            np.linspace(0, n_rp - 1, self.dataLen),
                            np.arange(n_rp),
                            rp_data['phase']
                        )
                    
                    # Store mag and phase
                    self.mag[self.numRecord:self.numRecord + self.dataLen] = mag_data
                    self.phase[self.numRecord:self.numRecord + self.dataLen] = phase_data
                    
                    # Calculate signal (same formula as original)
                    self.signal[self.numRecord:self.numRecord + self.dataLen] = mag_data * self.sensitivity * 0.707
                
                # Store DC ramp (same as original)
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                
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
        print "Status", status.value
        self.stop_rp_acquisition()
        return 0

    def get_data(self, blocking=True, timeout=None):
        if blocking:
            if not self._newdata_event.wait(timeout):
                raise ValueError("timeout waiting for data from device")
        with self._data_lock:
            self._newdata_event.clear()
            return self._data.copy()

    def reset(self):
        self.mag = np.zeros((self.duration * self.rate,))
        self.dcRamp = np.zeros((self.duration * self.rate,))
        self.phase = np.zeros((self.duration * self.rate,))
        self.signal = np.zeros((self.duration * self.rate,))
        self.read = int32()
        self.compError = 0
        self.numRecord = 0
        
    def __del__(self):
        self.stop_rp_acquisition()
        if hasattr(self, 'lock_in'):
            self.lock_in.disconnect()


class LockInCurrent(SEEDTask_RP):
    """EXACT same as their LockInCurrent but uses Red Pitaya"""
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        SEEDTask_RP.__init__(self, duration, rate, data_len, frequency, amplitude)
        self.units = 'mA'

    def EveryNCallback(self):
        """Same as original LockInCurrent callback"""
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                # Read DAQ
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                                 self._data, self.dataLen * 7, 
                                 ctypes.byref(self.read), None)
                
                # Get Red Pitaya data
                rp_data = self.lock_in.get_latest()
                
                if rp_data is not None and len(rp_data['mag']) > 0:
                    n_rp = len(rp_data['mag'])
                    if n_rp >= self.dataLen:
                        mag_data = np.absolute(rp_data['mag'][-self.dataLen:])
                        phase_data = rp_data['phase'][-self.dataLen:]
                    else:
                        mag_data = np.absolute(np.interp(
                            np.linspace(0, n_rp - 1, self.dataLen),
                            np.arange(n_rp),
                            rp_data['mag']
                        ))
                        phase_data = np.interp(
                            np.linspace(0, n_rp - 1, self.dataLen),
                            np.arange(n_rp),
                            rp_data['phase']
                        )
                    
                    # Same as their original code
                    self.mag[self.numRecord:self.numRecord + self.dataLen] = mag_data
                    self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                    self.phase[self.numRecord:self.numRecord + self.dataLen] = phase_data * 20
                    self.signal[self.numRecord:self.numRecord + self.dataLen] = mag_data * self.sensitivity * 0.707
                
                self.numRecord += self.dataLen
                self._newdata_event.set()
        
        return 0
    
    def compensate(self):
        return 0


# SAME save functions as their original
def save_LIC(taskName, fileName):
    tempDir = os.getcwd()
    save_directory = 'C:\SEED 3.2 Data\Joydip\e2025\Novenber\Batch 3 Chip testing'
    
    os.chdir(save_directory)
    mag = taskName.mag
    dcRamp = taskName.dcRamp
    time = taskName.time
    phase = taskName.phase
    signal = taskName.signal
    
    np.savez(fileName + '.npz', mag=mag, dcRamp=dcRamp, phase=phase, time=time, signal=signal)
    dat = loadSEEDDatanpz(fileName + '.npz')
    os.chdir(save_directory)
    os.mkdir(fileName)
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

# -*- coding: utf-8 -*-
"""
Modified SEED Task using Red Pitaya instead of Signal Recovery 7280
Python 2.7 compatible
"""

import threading
import time
from PyDAQmx import *
import ctypes
import numpy as np
from rp_client import RedPitayaClient


class SEEDTask_RP(Task):
    """
    Modified SEEDTask that uses Red Pitaya lock-in instead of SR7280
    Maintains same interface as original for compatibility
    """
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        Task.__init__(self)

        self.dev_name = "Dev1"

        # Red Pitaya lock-in (replaces GPIB lock-in)
        self.lock_in = RedPitayaClient(host='localhost', port=5555)
        if not self.lock_in.connect():
            raise RuntimeError("Could not connect to Red Pitaya server")
        
        # Initialize Red Pitaya
        print("Initializing Red Pitaya...")
        result = self.lock_in.initialize()
        if result['status'] != 'ok':
            raise RuntimeError("Red Pitaya initialization failed")
        
        # Setup lock-in
        self.frequency = frequency  # Hz (not mHz like SR7280)
        self.amplitude = amplitude  # V peak (Red Pitaya uses peak, not RMS)
        
        print("Configuring Red Pitaya lock-in...")
        result = self.lock_in.setup(
            frequency=self.frequency,
            amplitude=self.amplitude,
            bandwidth=10,  # 10 Hz filter bandwidth
            phase=0
        )
        if result['status'] != 'ok':
            raise RuntimeError("Red Pitaya setup failed")
        
        self.sensitivity = 1.0  # Red Pitaya outputs already scaled
        self.rate = float(rate)
        self.duration = float(duration)
        self.time = np.arange(0, self.duration, 1 / self.rate)

        self.dataLen = data_len
        self._data = np.zeros((self.dataLen, 7))

        # Preallocate data logging arrays
        self.mag = np.zeros((int(self.duration) * int(self.rate),))
        self.dcRamp = np.zeros((int(self.duration) * int(self.rate),))
        self.phase = np.zeros((int(self.duration) * int(self.rate),))
        self.signal = np.zeros((int(self.duration) * int(self.rate),))
        self.read = int32()
        self.compError = 0
        self.numRecord = 0

        # Create Voltage Channels (DAQ card for DC measurements)
        self.CreateAIVoltageChan(self.dev_name + "/ai0", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai1", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai2", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai3", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai4", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai5", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai6", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)

        self.CfgSampClkTiming("", self.rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.dataLen)

        # PID Gains (for nulling)
        self.Kp = 0
        self.Kd = 0
        self.Ki = 0

        self.units = 'pm'
        
        # Red Pitaya data polling thread
        self.rp_running = False
        self.rp_thread = None

    def start_rp_acquisition(self):
        """Start Red Pitaya data acquisition"""
        result = self.lock_in.start_acquisition()
        if result['status'] == 'ok':
            self.rp_running = True
            self.rp_thread = threading.Thread(target=self._poll_rp_data)
            self.rp_thread.daemon = True
            self.rp_thread.start()
            print("Red Pitaya acquisition started")
        else:
            print("Failed to start Red Pitaya acquisition")

    def stop_rp_acquisition(self):
        """Stop Red Pitaya data acquisition"""
        self.rp_running = False
        if self.rp_thread:
            self.rp_thread.join(timeout=2.0)
        self.lock_in.stop_acquisition()
        print("Red Pitaya acquisition stopped")

    def _poll_rp_data(self):
        """Background thread to poll Red Pitaya data"""
        while self.rp_running:
            try:
                self.lock_in.get_data()
                time.sleep(0.05)  # Poll every 50ms
            except Exception as e:
                print("Error polling Red Pitaya: {}".format(e))
                time.sleep(0.1)

    def compensate(self):
        """Compensator nulling algorithm (placeholder)"""
        return 0.0

    def set_comp_voltage(self, v):
        """Set compensation voltage (placeholder)"""
        return 0.0

    def continuousRecord(self):
        """Setup continuous recording"""
        self._data_lock = threading.Lock()
        self._newdata_event = threading.Event()
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.dataLen, 0)
        self.AutoRegisterDoneEvent(0)
        
        # Start Red Pitaya acquisition
        self.start_rp_acquisition()

    def EveryNCallback(self):
        """Called every N samples by DAQ"""
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                # Read DAQ card data (DC measurements)
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                                 self._data, self.dataLen * 7, 
                                 ctypes.byref(self.read), None)
                
                # Get Red Pitaya lock-in data
                rp_data = self.lock_in.get_latest()
                
                if rp_data is not None and len(rp_data['mag']) > 0:
                    # Interpolate Red Pitaya data to match DAQ sample count
                    n_daq = self.dataLen
                    n_rp = len(rp_data['mag'])
                    
                    if n_rp >= n_daq:
                        # Downsample if we have more RP data
                        idx = np.linspace(0, n_rp - 1, n_daq).astype(int)
                        mag_interp = rp_data['mag'][idx]
                        phase_interp = rp_data['phase'][idx]
                    else:
                        # Upsample if we have less RP data
                        mag_interp = np.interp(
                            np.linspace(0, n_rp - 1, n_daq),
                            np.arange(n_rp),
                            rp_data['mag']
                        )
                        phase_interp = np.interp(
                            np.linspace(0, n_rp - 1, n_daq),
                            np.arange(n_rp),
                            rp_data['phase']
                        )
                    
                    # Store lock-in data
                    self.mag[self.numRecord:self.numRecord + self.dataLen] = mag_interp
                    self.phase[self.numRecord:self.numRecord + self.dataLen] = phase_interp
                    
                    # Calculate signal (matching original scaling)
                    # Original: mag * sensitivity * 0.707
                    self.signal[self.numRecord:self.numRecord + self.dataLen] = mag_interp * self.sensitivity * 0.707
                
                # Store DC ramp from DAQ
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                
                self.compensate()
                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0

    def DoneCallback(self, status):
        """Called when acquisition is done"""
        print "Status", status.value
        self.stop_rp_acquisition()
        return 0

    def get_data(self, blocking=True, timeout=None):
        """Get latest data"""
        if blocking:
            if not self._newdata_event.wait(timeout):
                raise ValueError("timeout waiting for data from device")
        with self._data_lock:
            self._newdata_event.clear()
            return self._data.copy()

    def reset(self):
        """Reset all data arrays"""
        self.mag = np.zeros((self.duration * self.rate,))
        self.dcRamp = np.zeros((self.duration * self.rate,))
        self.phase = np.zeros((self.duration * self.rate,))
        self.signal = np.zeros((self.duration * self.rate,))
        self.read = int32()
        self.compError = 0
        self.numRecord = 0
        
    def __del__(self):
        """Cleanup"""
        self.stop_rp_acquisition()
        if hasattr(self, 'lock_in'):
            self.lock_in.disconnect()

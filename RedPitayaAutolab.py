"""
FILE 1: LIC_Object.py (Red Pitaya version)
EXACT REPLACEMENT - Signal Recovery GPIB → Red Pitaya (using your working lock-in code)
"""

import threading
import time
from PyDAQmx import *
import ctypes
import numpy as np
from pyrpl import Pyrpl

# ============================================================
# RED PITAYA CONNECTION SETTINGS
# ============================================================
RP_HOSTNAME = 'rp-f073ce.local'
RP_CONFIG = 'lockin_config'
# ============================================================


class SEEDTask(Task):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        Task.__init__(self)

        self.dev_name = "Dev1"

        # REPLACED: Signal Recovery GPIB lock-in → Red Pitaya
        print("Connecting to Red Pitaya lock-in...")
        self.rp = Pyrpl(config=RP_CONFIG, hostname=RP_HOSTNAME)
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.scope = self.rp_modules.scope

        # Convert frequency and amplitude (same as Signal Recovery did)
        self.frequency = frequency  # Hz
        self.amplitude = round((amplitude/1.414), 3)  # Convert V to Vrms

        print(f"Setting up Red Pitaya lock-in: {frequency} Hz, {self.amplitude} Vrms")
        
        # Turn off ASG0 (IQ module generates the signal)
        self.rp_modules.asg0.output_direct = 'off'
        
        # Setup IQ2 module as lock-in (from your working code)
        self.lockin.setup(
            frequency=frequency,
            bandwidth=10,              # Filter bandwidth (equivalent to TC 13)
            gain=0.0,                  # No feedback
            phase=0,                   # Phase offset
            acbandwidth=0,             # DC-coupled input
            amplitude=self.amplitude,  # Output amplitude (Vrms)
            input='in1',               # Measure from IN1
            output_direct='out1',      # Output excitation to OUT1
            output_signal='quadrature',
            quadrature_factor=1)
        
        # Setup scope to read lock-in X and Y outputs (from your working code)
        self.scope.input1 = 'iq2'    # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = 64
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True
        self.rp_sample_rate = 125e6 / self.scope.decimation
        
        # Sensitivity (Red Pitaya outputs in Volts directly)
        self.sensitivity = 1.0
        
        print(f"Red Pitaya lock-in ready. Sensitivity: {self.sensitivity}")
        
        # All original settings
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

        # Create Voltage Channels
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
        
        time.sleep(0.5)  # Let lock-in settle

    def read_lockin_magnitude(self):
        """Read magnitude from Red Pitaya lock-in (from your working code)"""
        try:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)  # iq2 = X
            ch2 = np.array(self.scope._data_ch2_current)  # iq2_2 = Y
            X = np.mean(ch1)
            Y = np.mean(ch2)
            R = np.sqrt(X**2 + Y**2)  # Magnitude
            return R
        except:
            return 0.0

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
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                                  self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                self._newdata_event.set()
                
                # KEY CHANGE: Read magnitude from Red Pitaya instead of DAQ ai0
                lock_in_mag = self.read_lockin_magnitude()
                
                # Update arrays (mag now comes from Red Pitaya, everything else from DAQ)
                self.mag[self.numRecord:self.numRecord + self.dataLen] = lock_in_mag
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                self.phase[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 4] * 20
                self.signal[self.numRecord:self.numRecord + self.dataLen] = \
                    self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.707
                
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
        compensatorVoltages = self.getIntialVoltageList()
        self.voltageList = []

        def updateCompList():
            self.StartTask()
            self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                             self._data, self.dataLen * 5, ctypes.byref(self.read), None)
            d = np.mean(self._data[:, 2] - self._data[:, 3])
            self.voltageList.append(d)
            self.StopTask()

        for x in compensatorVoltages:
            self.setCompVoltage(x)
            time.sleep(0.1)
            updateCompList()

        self.voltageList = np.absolute(self.voltageList)
        self.compVoltage = compensatorVoltages[np.argmin(self.voltageList)]
        self.setCompVoltage(self.compVoltage)

    def plotCompResponse(self):
        self.compensatorVoltages = np.arange(-4.0, 4.0, .05)
        self.voltageList = []

        def updateCompList():
            self.StartTask()
            self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                             self._data, self.dataLen * 5, ctypes.byref(self.read), None)
            time.sleep(0.1)
            d = np.mean(self._data[:, 2] - self._data[:, 3])
            self.voltageList.append(d)
            self.StopTask()

        for x in self.compensatorVoltages:
            self.setCompVoltage(x)
            time.sleep(0.01)
            updateCompList()

        import matplotlib.pyplot as plt
        plt.plot(self.compensatorVoltages, self.voltageList)

    def reset(self):
        self.mag = np.zeros((int(self.duration) * int(self.rate),))
        self.dcRamp = np.zeros((int(self.duration) * int(self.rate),))
        self.phase = np.zeros((int(self.duration) * int(self.rate),))
        self.signal = np.zeros((int(self.duration) * int(self.rate),))
        self.read = int32()
        self.compError = 0
        self.numRecord = 0


# ============================================================
# FILE 2: LIC_Measurement_Types.py
# ============================================================
import os
import csv

class LockInCurrent(SEEDTask):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        SEEDTask.__init__(self, duration, rate, data_len, frequency, amplitude)
        self.units = 'mA'

    def EveryNCallback(self):
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                                  self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                self._newdata_event.set()
                
                # KEY CHANGE: Read magnitude from Red Pitaya
                lock_in_mag = self.read_lockin_magnitude()
                
                # Update arrays
                self.mag[self.numRecord:self.numRecord + self.dataLen] = np.absolute(lock_in_mag)
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                self.phase[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 4] * 20
                self.signal[self.numRecord:self.numRecord + self.dataLen] = \
                    self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.707
                
                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0

    def compensate(self):
        return 0


def save_LIC(taskName, fileName):
    tempDir = os.getcwd()
    
    # Auto-create save directory with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(os.getcwd(), f'RedPitaya_Data_{timestamp}')
    
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        print(f"Created save directory: {save_directory}")

    os.chdir(save_directory)
    mag = taskName.mag
    dcRamp = taskName.dcRamp
    time_array = taskName.time
    phase = taskName.phase
    signal = taskName.signal
    
    np.savez(fileName + '.npz', mag=mag, dcRamp=dcRamp, phase=phase, time=time_array, signal=signal)
    dat = loadSEEDDatanpz(fileName + '.npz')
    os.chdir(save_directory)
    os.mkdir(fileName)
    os.chdir(save_directory + '\\' + fileName)
    csvGenerate(dat)
    os.chdir(tempDir)
    
    print(f"Data saved to: {save_directory}\\{fileName}")


def loadSEEDDatanpz(fileName):
    s = np.load(fileName)
    dcRamp = s['dcRamp']
    mag = s['mag']
    phase = s['phase']
    time_array = s['time']
    signal = s['signal']
    return {'dcRamp': dcRamp, 'magnitude': mag, 'phase': phase, 'time': time_array, 'signal': signal}


def csvGenerate(dat):
    for i in dat.keys():
        with open(i + '.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(dat[i])


# ============================================================
# FILE 3: Main execution script
# ============================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    print("=" * 60)
    print("RED PITAYA LOCK-IN CURRENT MEASUREMENT")
    print("=" * 60)
    print("Replacing Signal Recovery GPIB lock-in with Red Pitaya")
    print("=" * 60)

    task = LockInCurrent(duration=12)  # frequency in Hz, amplitude in V

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(task.units, color=(31 / 255., 119 / 255., 180 / 255.))
    ax1.set_ylabel('DC Ramp', color=(255 / 255., 127 / 255., 14 / 255.))
    
    task.continuousRecord()
    task.StartTask()

    def animate(i):
        ax2.plot(task.time[0:task.numRecord], task.signal[0:task.numRecord], 
                color=(31 / 255., 119 / 255., 180 / 255.))
        ax1.plot(task.time[0:task.numRecord], task.dcRamp[0:task.numRecord], 
                color=(255 / 255., 127 / 255., 14 / 255.))

    ani = animation.FuncAnimation(fig, animate, interval=50)
    plt.show()
    
    # After plot closes, save data
    task.StopTask()
    task.ClearTask()
    
    save_LIC(task, 'experiment_data')
    
    print("=" * 60)
    print("Experiment complete!")
    print("=" * 60)

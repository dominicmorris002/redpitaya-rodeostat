"""
Unified Red Pitaya Lock-In Amplifier Experiment
Combines NI DAQ data acquisition with Red Pitaya lock-in detection
"""

import threading
import time
import os
import csv
from PyDAQmx import *
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyrpl import Pyrpl

# ============================================================
# EXPERIMENT PARAMETERS - CHANGE THESE
# ============================================================
# Lock-in settings
REF_FREQUENCY = 500         # Hz - AC excitation frequency
REF_AMPLITUDE = 0.2         # V - AC signal amplitude
OUTPUT_CHANNEL = 'out1'     # 'out1' or 'out2'
PHASE_OFFSET = 0            # degrees
FILTER_BANDWIDTH = 10       # Hz - lock-in filter bandwidth

# Data acquisition settings
DURATION = 12               # seconds - measurement duration
RATE = 500                  # Hz - DAQ sampling rate
DATA_LEN = 250              # samples per buffer

# Data saving
SAVE_DATA = True            # True = save data files
SAVE_DIRECTORY = 'C:\\SEED 3.2 Data\\Joydip\\e2025\\November\\Batch 3 Chip testing'
FILE_NAME = 'rp_lockin_test'  # Base filename for saved data

# Red Pitaya settings
RP_HOSTNAME = 'rp-f073ce.local'
RP_CONFIG = 'lockin_config'
# ============================================================


class RedPitayaLockInTask(Task):
    """Unified task combining Red Pitaya lock-in with NI DAQ acquisition"""
    
    def __init__(self, duration=3600, rate=500, data_len=250, 
                 frequency=500, amplitude=0.2, filter_bw=10,
                 rp_hostname='rp-f073ce.local', rp_config='lockin_config'):
        Task.__init__(self)
        
        self.dev_name = "Dev1"
        
        # Experiment parameters
        self.frequency = frequency
        self.amplitude = amplitude
        self.rate = float(rate)
        self.duration = float(duration)
        self.dataLen = data_len
        self.time = np.arange(0, self.duration, 1 / self.rate)
        self.units = 'mA'
        
        # Initialize Red Pitaya
        print("Initializing Red Pitaya lock-in amplifier...")
        self.rp = Pyrpl(config=rp_config, hostname=rp_hostname)
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.scope = self.rp_modules.scope
        
        # Setup Red Pitaya lock-in
        self.setup_red_pitaya_lockin(frequency, amplitude, filter_bw)
        
        # Preallocate data arrays
        total_samples = int(self.duration * self.rate)
        self.mag = np.zeros((total_samples,))
        self.dcRamp = np.zeros((total_samples,))
        self.phase = np.zeros((total_samples,))
        self.signal = np.zeros((total_samples,))
        self.rp_X = np.zeros((total_samples,))
        self.rp_Y = np.zeros((total_samples,))
        self.rp_R = np.zeros((total_samples,))
        self.rp_theta = np.zeros((total_samples,))
        
        self._data = np.zeros((self.dataLen, 7))
        self.read = int32()
        self.numRecord = 0
        
        # Get Red Pitaya sensitivity (equivalent to lock-in sensitivity)
        self.sensitivity = 1.0  # Red Pitaya outputs directly in Volts
        
        # Create NI DAQ Voltage Channels
        print("Setting up NI DAQ channels...")
        self.CreateAIVoltageChan(self.dev_name + "/ai0", '', DAQmx_Val_Cfg_Default, 
                                -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai1", '', DAQmx_Val_Cfg_Default, 
                                -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai2", '', DAQmx_Val_Cfg_Default, 
                                -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai3", '', DAQmx_Val_Cfg_Default, 
                                -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai4", '', DAQmx_Val_Cfg_Default, 
                                -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai5", '', DAQmx_Val_Cfg_Default, 
                                -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai6", '', DAQmx_Val_Cfg_Default, 
                                -10.0, 10.0, DAQmx_Val_Volts, None)
        
        self.CfgSampClkTiming("", self.rate, DAQmx_Val_Rising, 
                             DAQmx_Val_ContSamps, self.dataLen)
        
        print("Initialization complete!")
    
    def setup_red_pitaya_lockin(self, frequency, amplitude, filter_bw):
        """Setup Red Pitaya IQ module as lock-in amplifier"""
        # Turn off ASG0 - IQ module will generate reference
        self.rp_modules.asg0.output_direct = 'off'
        
        # Setup IQ module to generate reference and demodulate
        self.lockin.setup(
            frequency=frequency,
            bandwidth=filter_bw,
            gain=0.0,
            phase=PHASE_OFFSET,
            acbandwidth=0,
            amplitude=amplitude,
            input='in1',
            output_direct=OUTPUT_CHANNEL,
            output_signal='quadrature',
            quadrature_factor=1)
        
        # Setup scope to read lock-in outputs
        self.scope.input1 = 'iq2'    # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        self.scope.decimation = 64
        self.rp_sample_rate = 125e6 / self.scope.decimation
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True
        
        print(f"Red Pitaya Lock-in configured:")
        print(f"  Frequency: {frequency} Hz")
        print(f"  Amplitude: {amplitude} V")
        print(f"  Filter BW: {filter_bw} Hz")
        print(f"  Output: {OUTPUT_CHANNEL}")
        
        # Let lock-in settle
        time.sleep(0.5)
    
    def read_red_pitaya_lockin(self):
        """Read current X, Y values from Red Pitaya lock-in"""
        try:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)  # X
            ch2 = np.array(self.scope._data_ch2_current)  # Y
            
            # Get mean values (lock-in output should be DC)
            X = np.mean(ch1)
            Y = np.mean(ch2)
            R = np.sqrt(X**2 + Y**2)
            theta = np.arctan2(Y, X)
            
            return X, Y, R, theta
        except:
            return 0.0, 0.0, 0.0, 0.0
    
    def continuousRecord(self):
        """Setup continuous recording with callbacks"""
        self._data_lock = threading.Lock()
        self._newdata_event = threading.Event()
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, 
                                           self.dataLen, 0)
        self.AutoRegisterDoneEvent(0)
    
    def EveryNCallback(self):
        """Callback for each data buffer - combines NI DAQ and Red Pitaya data"""
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                # Read NI DAQ data
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, 
                                  DAQmx_Val_GroupByScanNumber, 
                                  self._data, self.dataLen * 7, 
                                  ctypes.byref(self.read), None)
                
                # Read Red Pitaya lock-in outputs
                X, Y, R, theta = self.read_red_pitaya_lockin()
                
                # Store NI DAQ data
                self.mag[self.numRecord:self.numRecord + self.dataLen] = \
                    np.absolute(self._data[:, 0])
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = \
                    -self._data[:, 1]
                self.phase[self.numRecord:self.numRecord + self.dataLen] = \
                    self._data[:, 4] * 20
                
                # Store Red Pitaya lock-in data (replicate for buffer length)
                self.rp_X[self.numRecord:self.numRecord + self.dataLen] = X
                self.rp_Y[self.numRecord:self.numRecord + self.dataLen] = Y
                self.rp_R[self.numRecord:self.numRecord + self.dataLen] = R
                self.rp_theta[self.numRecord:self.numRecord + self.dataLen] = theta
                
                # Calculate signal using Red Pitaya magnitude
                # Convert to current: R * sensitivity * 0.707 (RMS conversion)
                self.signal[self.numRecord:self.numRecord + self.dataLen] = \
                    R * self.sensitivity * 0.707
                
                self.numRecord += self.dataLen
                self._newdata_event.set()
        
        return 0
    
    def DoneCallback(self, status):
        """Callback when acquisition is complete"""
        print("Acquisition complete, status:", status.value)
        return 0
    
    def get_data(self, blocking=True, timeout=None):
        """Get current data snapshot"""
        if blocking:
            if not self._newdata_event.wait(timeout):
                raise ValueError("Timeout waiting for data from device")
        with self._data_lock:
            self._newdata_event.clear()
            return self._data.copy()
    
    def reset(self):
        """Reset all data arrays"""
        total_samples = int(self.duration * self.rate)
        self.mag = np.zeros((total_samples,))
        self.dcRamp = np.zeros((total_samples,))
        self.phase = np.zeros((total_samples,))
        self.signal = np.zeros((total_samples,))
        self.rp_X = np.zeros((total_samples,))
        self.rp_Y = np.zeros((total_samples,))
        self.rp_R = np.zeros((total_samples,))
        self.rp_theta = np.zeros((total_samples,))
        self.read = int32()
        self.numRecord = 0


def save_experiment_data(task, fileName, save_directory):
    """Save all experimental data to files"""
    tempDir = os.getcwd()
    
    try:
        os.chdir(save_directory)
        
        # Save main data as NPZ
        np.savez(fileName + '.npz', 
                 mag=task.mag,
                 dcRamp=task.dcRamp,
                 phase=task.phase,
                 time=task.time,
                 signal=task.signal,
                 rp_X=task.rp_X,
                 rp_Y=task.rp_Y,
                 rp_R=task.rp_R,
                 rp_theta=task.rp_theta)
        
        # Load and create CSV files
        dat = load_experiment_data(fileName + '.npz')
        
        # Create subdirectory for CSV files
        if not os.path.exists(fileName):
            os.mkdir(fileName)
        os.chdir(os.path.join(save_directory, fileName))
        
        # Generate CSV files
        generate_csv_files(dat)
        
        print(f"Data saved successfully to {save_directory}\\{fileName}")
        
    finally:
        os.chdir(tempDir)


def load_experiment_data(fileName):
    """Load experimental data from NPZ file"""
    s = np.load(fileName)
    return {
        'dcRamp': s['dcRamp'],
        'magnitude': s['mag'],
        'phase': s['phase'],
        'time': s['time'],
        'signal': s['signal'],
        'rp_X': s['rp_X'],
        'rp_Y': s['rp_Y'],
        'rp_R': s['rp_R'],
        'rp_theta': s['rp_theta']
    }


def generate_csv_files(dat):
    """Create CSV files for each data channel"""
    for key in dat.keys():
        with open(key + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', 
                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(dat[key])


def run_experiment():
    """Main experiment execution"""
    print("=" * 60)
    print("RED PITAYA LOCK-IN AMPLIFIER EXPERIMENT")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE} V")
    print(f"Duration: {DURATION} seconds")
    print(f"Sample Rate: {RATE} Hz")
    print(f"Filter Bandwidth: {FILTER_BANDWIDTH} Hz")
    print("=" * 60)
    
    # Create task
    task = RedPitayaLockInTask(
        duration=DURATION,
        rate=RATE,
        data_len=DATA_LEN,
        frequency=REF_FREQUENCY,
        amplitude=REF_AMPLITUDE,
        filter_bw=FILTER_BANDWIDTH,
        rp_hostname=RP_HOSTNAME,
        rp_config=RP_CONFIG
    )
    
    # Setup plotting
    fig = plt.figure(figsize=(12, 8))
    
    # Subplot 1: Signal and DC Ramp
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(task.units, color=(31/255., 119/255., 180/255.))
    ax1.set_ylabel('DC Ramp (V)', color=(255/255., 127/255., 14/255.))
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Signal and DC Ramp vs Time')
    
    # Subplot 2: Red Pitaya X and Y
    ax3 = fig.add_subplot(2, 2, 2)
    ax3.set_ylabel('Voltage (V)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Red Pitaya Lock-in: X and Y')
    
    # Subplot 3: Red Pitaya R
    ax4 = fig.add_subplot(2, 2, 3)
    ax4.set_ylabel('R (V)')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Red Pitaya Lock-in: Magnitude')
    
    # Subplot 4: Red Pitaya Phase
    ax5 = fig.add_subplot(2, 2, 4)
    ax5.set_ylabel('Phase (rad)')
    ax5.set_xlabel('Time (s)')
    ax5.set_title('Red Pitaya Lock-in: Phase')
    
    plt.tight_layout()
    
    # Animation function
    def animate(i):
        n = task.numRecord
        if n > 0:
            t = task.time[0:n]
            
            # Clear and replot
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax5.clear()
            
            # Plot 1: Signal and DC Ramp
            ax2.plot(t, task.signal[0:n], color=(31/255., 119/255., 180/255.), 
                    label='Signal')
            ax1.plot(t, task.dcRamp[0:n], color=(255/255., 127/255., 14/255.), 
                    label='DC Ramp')
            ax2.set_ylabel(task.units, color=(31/255., 119/255., 180/255.))
            ax1.set_ylabel('DC Ramp (V)', color=(255/255., 127/255., 14/255.))
            ax1.set_xlabel('Time (s)')
            ax1.set_title('Signal and DC Ramp vs Time')
            
            # Plot 2: X and Y
            ax3.plot(t, task.rp_X[0:n], 'b-', label='X (in-phase)', linewidth=1)
            ax3.plot(t, task.rp_Y[0:n], 'r-', label='Y (quadrature)', linewidth=1)
            ax3.set_ylabel('Voltage (V)')
            ax3.set_xlabel('Time (s)')
            ax3.set_title('Red Pitaya Lock-in: X and Y')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 3: R
            ax4.plot(t, task.rp_R[0:n], 'm-', linewidth=1)
            ax4.set_ylabel('R (V)')
            ax4.set_xlabel('Time (s)')
            ax4.set_title('Red Pitaya Lock-in: Magnitude')
            ax4.grid(True)
            
            # Plot 4: Phase
            ax5.plot(t, task.rp_theta[0:n], 'c-', linewidth=1)
            ax5.set_ylabel('Phase (rad)')
            ax5.set_xlabel('Time (s)')
            ax5.set_title('Red Pitaya Lock-in: Phase')
            ax5.grid(True)
    
    # Start acquisition
    task.continuousRecord()
    task.StartTask()
    
    # Start animation
    ani = animation.FuncAnimation(fig, animate, interval=50)
    plt.show()
    
    # Stop acquisition
    task.StopTask()
    task.ClearTask()
    
    # Save data if requested
    if SAVE_DATA:
        save_experiment_data(task, FILE_NAME, SAVE_DIRECTORY)
    
    print("Experiment complete!")
    
    return task


if __name__ == '__main__':
    task = run_experiment()

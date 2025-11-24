"""
CORRECTED Red Pitaya Lock-In Amplifier for CV Experiments
Fixes:
1. Proper amplitude handling (peak voltage, not RMS)
2. Continuous lock-in reading synchronized with DAQ
3. Correct sensitivity scaling
4. Better scope configuration
5. Automatic data saving with comparison plots
"""

import threading
import time
from PyDAQmx import *
import ctypes
import numpy as np
from pyrpl import Pyrpl
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
# RED PITAYA CONNECTION SETTINGS
# ============================================================
RP_HOSTNAME = 'rp-f073ce.local'
RP_CONFIG = 'lockin_config'

# EXPERIMENT PARAMETERS
MEASUREMENT_DURATION = 12  # seconds
DAQ_SAMPLE_RATE = 500      # Hz
DAQ_BUFFER_SIZE = 250      # samples per callback
LOCKIN_FREQUENCY = 500     # Hz - AC excitation frequency
LOCKIN_AMPLITUDE = 0.2     # V - PEAK amplitude (NOT RMS)
FILTER_BANDWIDTH = 10      # Hz - lock-in filter bandwidth

# DATA SAVING
SAVE_DATA = True           # Always save data for comparison
EXPERIMENT_NAME = 'cv_experiment'  # Base filename
# ============================================================


class SEEDTask(Task):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        Task.__init__(self)

        self.dev_name = "Dev1"

        # Connect to Red Pitaya
        print("Connecting to Red Pitaya lock-in...")
        self.rp = Pyrpl(config=RP_CONFIG, hostname=RP_HOSTNAME)
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.scope = self.rp_modules.scope

        # Store parameters
        self.frequency = frequency  # Hz
        self.amplitude = amplitude  # Peak voltage (NOT RMS!)

        print(f"Setting up Red Pitaya lock-in: {frequency} Hz, {amplitude}V peak")
        
        # Turn off ASG0 (IQ module generates the signal)
        self.rp_modules.asg0.output_direct = 'off'
        
        # Setup IQ2 module as lock-in
        # CRITICAL FIX: amplitude parameter expects PEAK voltage
        self.lockin.setup(
            frequency=frequency,
            bandwidth=FILTER_BANDWIDTH,  # Filter bandwidth
            gain=0.0,                    # No feedback
            phase=0,                     # Phase offset
            acbandwidth=0,               # DC-coupled input
            amplitude=amplitude,         # PEAK amplitude (0.2V peak, not RMS!)
            input='in1',                 # Measure from IN1
            output_direct='out1',        # Output excitation to OUT1
            output_signal='quadrature',
            quadrature_factor=1)
        
        # Setup scope to read lock-in X and Y outputs
        self.scope.input1 = 'iq2'    # X (in-phase)
        self.scope.input2 = 'iq2_2'  # Y (quadrature)
        
        # CRITICAL FIX: Use higher decimation for smoother lock-in output
        # We'll average the scope data to match DAQ rate
        self.scope.decimation = 8192  # ~15 kHz sample rate
        self.scope._start_acquisition_rolling_mode()
        self.scope.average = True
        self.rp_sample_rate = 125e6 / self.scope.decimation
        
        # Calculate how many scope samples per DAQ sample
        self.samples_per_daq_point = int(self.rp_sample_rate / rate)
        
        print(f"Red Pitaya sample rate: {self.rp_sample_rate:.1f} Hz")
        print(f"Averaging {self.samples_per_daq_point} scope samples per DAQ point")
        
        # Sensitivity: Red Pitaya outputs in Volts directly
        # For current measurement, you may need to adjust this based on your transimpedance gain
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
        
        # Buffer for continuous lock-in readings
        self.lockin_buffer_X = []
        self.lockin_buffer_Y = []

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

    def read_lockin_magnitude_array(self, n_samples):
        """
        Read multiple lock-in magnitude values to match DAQ sampling
        
        CRITICAL FIX: Instead of reading once per callback, we read continuously
        and downsample to match DAQ rate
        """
        try:
            # Capture scope data
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)  # iq2 = X
            ch2 = np.array(self.scope._data_ch2_current)  # iq2_2 = Y
            
            # We have high-rate data from scope, need to downsample to DAQ rate
            # Split into n_samples chunks and average each chunk
            chunk_size = len(ch1) // n_samples
            
            if chunk_size < 1:
                # If not enough samples, just repeat the mean
                X_mean = np.mean(ch1)
                Y_mean = np.mean(ch2)
                R = np.sqrt(X_mean**2 + Y_mean**2)
                return np.full(n_samples, R)
            
            # Average chunks to get n_samples points
            R_array = []
            for i in range(n_samples):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size
                X_chunk = np.mean(ch1[start_idx:end_idx])
                Y_chunk = np.mean(ch2[start_idx:end_idx])
                R = np.sqrt(X_chunk**2 + Y_chunk**2)
                R_array.append(R)
            
            return np.array(R_array)
        except Exception as e:
            print(f"Lock-in read error: {e}")
            return np.zeros(n_samples)

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
                # Read DAQ channels
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                                  self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                self._newdata_event.set()
                
                # CRITICAL FIX: Read lock-in magnitude array (not just one value!)
                lock_in_mag_array = self.read_lockin_magnitude_array(self.dataLen)
                
                # Update arrays with proper lock-in data
                self.mag[self.numRecord:self.numRecord + self.dataLen] = lock_in_mag_array
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                self.phase[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 4] * 20
                
                # CRITICAL FIX: No 0.707 scaling - Red Pitaya already outputs peak magnitude
                # If you need RMS, multiply by 0.707 here, but for peak-to-peak use as-is
                self.signal[self.numRecord:self.numRecord + self.dataLen] = \
                    self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity
                
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
# LockInCurrent Class
# ============================================================

class LockInCurrent(SEEDTask):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        SEEDTask.__init__(self, duration, rate, data_len, frequency, amplitude)
        self.units = 'mA'

    def EveryNCallback(self):
        """Same as parent class but with current units"""
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, 
                                  self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                self._newdata_event.set()
                
                # CRITICAL FIX: Read lock-in magnitude array
                lock_in_mag_array = self.read_lockin_magnitude_array(self.dataLen)
                
                # Update arrays
                self.mag[self.numRecord:self.numRecord + self.dataLen] = np.absolute(lock_in_mag_array)
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                self.phase[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 4] * 20
                
                # CRITICAL FIX: No 0.707 scaling
                self.signal[self.numRecord:self.numRecord + self.dataLen] = \
                    self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity
                
                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0

    def compensate(self):
        return 0


# ============================================================
# Data Saving Functions
# ============================================================

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
    
    # Save .npz file
    np.savez(fileName + '.npz', mag=mag, dcRamp=dcRamp, phase=phase, time=time_array, signal=signal)
    
    # Load and save as CSV
    dat = loadSEEDDatanpz(fileName + '.npz')
    csv_dir = os.path.join(save_directory, fileName)
    os.mkdir(csv_dir)
    os.chdir(csv_dir)
    csvGenerate(dat)
    
    os.chdir(tempDir)
    
    print(f"Data saved to: {save_directory}/{fileName}")
    return save_directory


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


def create_summary_plots(task, save_directory):
    """Create comprehensive summary plots for data comparison"""
    
    summary_fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: Signal vs Time
    ax1 = summary_fig.add_subplot(3, 2, 1)
    ax1.plot(task.time[0:task.numRecord], task.signal[0:task.numRecord], 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Signal ({task.units})')
    ax1.set_title('Lock-in Signal vs Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: DC Ramp vs Time
    ax2 = summary_fig.add_subplot(3, 2, 2)
    ax2.plot(task.time[0:task.numRecord], task.dcRamp[0:task.numRecord], 'r-', linewidth=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('DC Ramp (V)')
    ax2.set_title('DC Ramp vs Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Signal vs DC Ramp (CV curve)
    ax3 = summary_fig.add_subplot(3, 2, 3)
    ax3.plot(task.dcRamp[0:task.numRecord], task.signal[0:task.numRecord], 'g-', linewidth=1)
    ax3.set_xlabel('DC Ramp (V)')
    ax3.set_ylabel(f'Signal ({task.units})')
    ax3.set_title('CV Curve: Signal vs DC Ramp')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Magnitude vs Time
    ax4 = summary_fig.add_subplot(3, 2, 4)
    ax4.plot(task.time[0:task.numRecord], task.mag[0:task.numRecord], 'm-', linewidth=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Magnitude (V)')
    ax4.set_title('Lock-in Magnitude vs Time')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Phase vs Time
    ax5 = summary_fig.add_subplot(3, 2, 5)
    ax5.plot(task.time[0:task.numRecord], task.phase[0:task.numRecord], 'c-', linewidth=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Phase (deg)')
    ax5.set_title('Phase vs Time')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Statistics Summary
    ax6 = summary_fig.add_subplot(3, 2, 6)
    ax6.axis('off')
    
    stats_text = f"""
EXPERIMENT STATISTICS

Lock-in Settings:
  Frequency: {task.frequency} Hz
  Amplitude: {task.amplitude} V (peak)
  Filter BW: {FILTER_BANDWIDTH} Hz

Data Acquisition:
  Duration: {task.numRecord / task.rate:.2f} s
  Sample Rate: {task.rate} Hz
  Total Samples: {task.numRecord}

Signal Statistics:
  Mean Signal: {np.mean(task.signal[0:task.numRecord]):.6f} {task.units}
  Std Dev: {np.std(task.signal[0:task.numRecord]):.6f} {task.units}
  SNR: {np.mean(task.signal[0:task.numRecord]) / (np.std(task.signal[0:task.numRecord]) + 1e-9):.2f}
  
Magnitude Statistics:
  Mean Mag: {np.mean(task.mag[0:task.numRecord]):.6f} V
  Std Dev: {np.std(task.mag[0:task.numRecord]):.6f} V
  Range: [{np.min(task.mag[0:task.numRecord]):.6f}, {np.max(task.mag[0:task.numRecord]):.6f}] V

DC Ramp Statistics:
  Range: [{np.min(task.dcRamp[0:task.numRecord]):.3f}, {np.max(task.dcRamp[0:task.numRecord]):.3f}] V
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = os.path.join(save_directory, f'{EXPERIMENT_NAME}_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Summary plot saved to: {summary_path}")
    
    plt.show()


# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("RED PITAYA LOCK-IN CURRENT MEASUREMENT")
    print("=" * 60)
    print("CORRECTED VERSION - Proper amplitude and sampling")
    print("=" * 60)
    print(f"Frequency: {LOCKIN_FREQUENCY} Hz")
    print(f"Amplitude: {LOCKIN_AMPLITUDE}V PEAK")
    print(f"Duration: {MEASUREMENT_DURATION}s")
    print(f"DAQ Rate: {DAQ_SAMPLE_RATE} Hz")
    print(f"Filter BW: {FILTER_BANDWIDTH} Hz")
    print(f"Save Data: {SAVE_DATA}")
    print("=" * 60)

    task = LockInCurrent(
        duration=MEASUREMENT_DURATION,
        rate=DAQ_SAMPLE_RATE,
        data_len=DAQ_BUFFER_SIZE,
        frequency=LOCKIN_FREQUENCY,
        amplitude=LOCKIN_AMPLITUDE
    )

    # Create live plotting figure
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    ax2.set_ylabel(task.units, color=(31 / 255., 119 / 255., 180 / 255.))
    ax1.set_ylabel('DC Ramp', color=(255 / 255., 127 / 255., 14 / 255.))
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Live Data Acquisition')
    
    task.continuousRecord()
    task.StartTask()

    def animate(i):
        if task.numRecord > 0:
            ax2.clear()
            ax1.clear()
            ax2.plot(task.time[0:task.numRecord], task.signal[0:task.numRecord], 
                    color=(31 / 255., 119 / 255., 180 / 255.))
            ax1.plot(task.time[0:task.numRecord], task.dcRamp[0:task.numRecord], 
                    color=(255 / 255., 127 / 255., 14 / 255.))
            ax2.set_ylabel(task.units, color=(31 / 255., 119 / 255., 180 / 255.))
            ax1.set_ylabel('DC Ramp', color=(255 / 255., 127 / 255., 14 / 255.))
            ax1.set_xlabel('Time (s)')
            ax1.set_title(f'Live Data Acquisition - {task.numRecord}/{int(task.duration * task.rate)} samples')

    ani = animation.FuncAnimation(fig, animate, interval=50)
    plt.show()
    
    # After plot closes, stop task
    print("\nStopping acquisition...")
    task.StopTask()
    task.ClearTask()
    
    if SAVE_DATA:
        print("\nSaving data...")
        save_dir = save_LIC(task, EXPERIMENT_NAME)
        
        print("\nGenerating summary plots...")
        create_summary_plots(task, save_dir)
    
    print("=" * 60)
    print("Experiment complete!")
    print(f"Total samples recorded: {task.numRecord}")
    print(f"Actual duration: {task.numRecord / task.rate:.2f} seconds")
    print("=" * 60)

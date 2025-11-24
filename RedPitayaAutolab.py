"""
SIMPLE Red Pitaya Lock-In Experiment for Electrochemistry

PHYSICAL CONNECTIONS NEEDED:
============================
1. Red Pitaya OUT1 → Apply to your electrochemical cell (excitation voltage)
2. Cell response signal → Red Pitaya IN1 (measures current response)
3. NI DAQ ai0 - ai6 → Other signals from your experiment
4. Make sure Red Pitaya and PC are on same network

WHAT THIS DOES:
===============
- Red Pitaya generates AC voltage at your frequency (e.g., 500 Hz)
- This excites your electrochemical cell
- Red Pitaya measures the AC current response using lock-in detection
- NI DAQ records additional signals (DC ramp, phase, etc.)
- Everything is plotted in real-time
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
# STEP 1: SET YOUR PARAMETERS HERE
# ============================================================
FREQUENCY = 500              # Hz - How fast the AC signal oscillates
AMPLITUDE = 0.2              # Volts - Size of AC excitation
MEASUREMENT_TIME = 12        # Seconds - How long to measure

# Where to save data (change this to your path!)
SAVE_DIRECTORY = 'C:\\test_data'
FILE_NAME = 'my_experiment'

# Red Pitaya network address (check your Red Pitaya's IP/hostname)
RED_PITAYA_ADDRESS = 'rp-f073ce.local'
# ============================================================


class SimpleRedPitayaExperiment(Task):
    """Simple lock-in amplifier experiment combining Red Pitaya + NI DAQ"""
    
    def __init__(self):
        Task.__init__(self)
        
        print("\n" + "="*60)
        print("STARTING RED PITAYA LOCK-IN EXPERIMENT")
        print("="*60)
        
        # Basic settings
        self.frequency = FREQUENCY
        self.amplitude = AMPLITUDE
        self.duration = MEASUREMENT_TIME
        self.rate = 500.0  # DAQ sampling rate (Hz)
        self.dataLen = 250  # Samples per read
        
        # Initialize Red Pitaya
        print("\nConnecting to Red Pitaya at:", RED_PITAYA_ADDRESS)
        print("(This might take 10-20 seconds...)")
        try:
            self.rp = Pyrpl(config='my_config', hostname=RED_PITAYA_ADDRESS)
            self.lockin = self.rp.rp.iq2  # Lock-in module
            self.scope = self.rp.rp.scope  # Scope for reading data
            print("✓ Red Pitaya connected!")
        except Exception as e:
            print("✗ ERROR: Could not connect to Red Pitaya!")
            print("  Check that it's on the network and the hostname is correct")
            raise e
        
        # Setup Red Pitaya as lock-in amplifier
        print("\nConfiguring lock-in amplifier...")
        print(f"  Frequency: {self.frequency} Hz")
        print(f"  Amplitude: {self.amplitude} V")
        
        self.lockin.setup(
            frequency=self.frequency,      # AC frequency
            bandwidth=10,                  # Filter bandwidth (Hz)
            gain=0.0,                      # No feedback
            phase=0,                       # Phase offset
            acbandwidth=0,                 # DC-coupled
            amplitude=self.amplitude,      # Output amplitude
            input='in1',                   # Read from IN1
            output_direct='out1',          # Output to OUT1
            output_signal='quadrature',
            quadrature_factor=1
        )
        
        # Setup scope to read lock-in outputs
        self.scope.input1 = 'iq2'     # X (in-phase component)
        self.scope.input2 = 'iq2_2'   # Y (quadrature component)
        self.scope.decimation = 64
        self.scope._start_acquisition_rolling_mode()
        
        print("✓ Lock-in configured!")
        print("\n  OUT1 is now outputting", self.amplitude, "V sine wave at", self.frequency, "Hz")
        print("  IN1 is being measured by the lock-in amplifier")
        
        # Setup NI DAQ
        print("\nSetting up NI DAQ...")
        self.CreateAIVoltageChan("Dev1/ai0", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan("Dev1/ai1", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan("Dev1/ai2", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan("Dev1/ai3", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan("Dev1/ai4", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan("Dev1/ai5", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan("Dev1/ai6", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CfgSampClkTiming("", self.rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.dataLen)
        print("✓ NI DAQ ready!")
        
        # Create data arrays
        total_samples = int(self.duration * self.rate)
        self.time = np.arange(0, self.duration, 1/self.rate)
        self.mag = np.zeros(total_samples)          # Lock-in magnitude
        self.dcRamp = np.zeros(total_samples)       # DC ramp from DAQ
        self.phase = np.zeros(total_samples)        # Phase from DAQ
        self.signal = np.zeros(total_samples)       # Calculated signal (mA)
        self.lock_X = np.zeros(total_samples)       # Lock-in X output
        self.lock_Y = np.zeros(total_samples)       # Lock-in Y output
        
        self._data = np.zeros((self.dataLen, 7))
        self.read = int32()
        self.numRecord = 0
        
        print("\n" + "="*60)
        print("READY TO START!")
        print("="*60)
        time.sleep(0.5)  # Let everything settle
    
    def read_lockin(self):
        """Read X and Y from Red Pitaya lock-in"""
        self.scope.single()
        X_array = np.array(self.scope._data_ch1_current)
        Y_array = np.array(self.scope._data_ch2_current)
        X = np.mean(X_array)  # Average value
        Y = np.mean(Y_array)
        return X, Y
    
    def continuousRecord(self):
        """Setup continuous data acquisition"""
        self._data_lock = threading.Lock()
        self._newdata_event = threading.Event()
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.dataLen, 0)
        self.AutoRegisterDoneEvent(0)
    
    def EveryNCallback(self):
        """Called every time new data is available"""
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                # Read NI DAQ
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber,
                                  self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                
                # Read Red Pitaya lock-in
                X, Y = self.read_lockin()
                R = np.sqrt(X**2 + Y**2)  # Magnitude
                
                # Store data
                idx = slice(self.numRecord, self.numRecord + self.dataLen)
                self.mag[idx] = np.abs(self._data[:, 0])
                self.dcRamp[idx] = -self._data[:, 1]
                self.phase[idx] = self._data[:, 4] * 20
                self.lock_X[idx] = X
                self.lock_Y[idx] = Y
                self.signal[idx] = R * 0.707  # Convert to RMS current (mA)
                
                self.numRecord += self.dataLen
                self._newdata_event.set()
        return 0
    
    def DoneCallback(self, status):
        print("\nAcquisition complete!")
        return 0


def save_data(task, filename, directory):
    """Save all data to files"""
    print("\nSaving data...")
    
    # Create directory if needed
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save as NPZ (numpy format)
    filepath = os.path.join(directory, filename + '.npz')
    np.savez(filepath,
             time=task.time,
             signal=task.signal,
             dcRamp=task.dcRamp,
             phase=task.phase,
             mag=task.mag,
             lock_X=task.lock_X,
             lock_Y=task.lock_Y)
    
    print(f"✓ Saved to: {filepath}")
    
    # Also save as CSV
    csv_dir = os.path.join(directory, filename + '_csv')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    data_dict = {
        'time': task.time,
        'signal_mA': task.signal,
        'dcRamp': task.dcRamp,
        'phase': task.phase,
        'lock_X': task.lock_X,
        'lock_Y': task.lock_Y
    }
    
    for name, data in data_dict.items():
        csv_path = os.path.join(csv_dir, name + '.csv')
        np.savetxt(csv_path, data, delimiter=',', header=name, comments='')
    
    print(f"✓ CSV files saved to: {csv_dir}")


def run_experiment():
    """Main function to run the experiment"""
    
    # Create experiment
    experiment = SimpleRedPitayaExperiment()
    
    # Setup plot
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f'Red Pitaya Lock-In Experiment: {FREQUENCY} Hz, {AMPLITUDE} V', fontsize=14)
    
    # Create subplots
    ax1 = fig.add_subplot(2, 2, 1)  # Signal vs time
    ax1_twin = ax1.twinx()
    ax2 = fig.add_subplot(2, 2, 2)  # Lock-in X and Y
    ax3 = fig.add_subplot(2, 2, 3)  # DC Ramp
    ax4 = fig.add_subplot(2, 2, 4)  # Phase
    
    def animate(i):
        """Update plots"""
        n = experiment.numRecord
        if n > 10:
            t = experiment.time[:n]
            
            # Clear all axes
            ax1.clear()
            ax1_twin.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            
            # Plot 1: Signal (mA)
            ax1.plot(t, experiment.signal[:n], 'b-', linewidth=1, label='Signal')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Signal (mA)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.grid(True)
            ax1.set_title('Lock-In Signal')
            
            # Plot 2: Lock-in X and Y
            ax2.plot(t, experiment.lock_X[:n], 'r-', linewidth=1, label='X (in-phase)')
            ax2.plot(t, experiment.lock_Y[:n], 'b-', linewidth=1, label='Y (quadrature)')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Voltage (V)')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('Lock-In Outputs')
            
            # Plot 3: DC Ramp
            ax3.plot(t, experiment.dcRamp[:n], 'g-', linewidth=1)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('DC Ramp (V)')
            ax3.grid(True)
            ax3.set_title('DC Ramp')
            
            # Plot 4: Phase
            ax4.plot(t, experiment.phase[:n], 'm-', linewidth=1)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Phase')
            ax4.grid(True)
            ax4.set_title('Phase')
            
            # Show progress
            progress = (n / (experiment.duration * experiment.rate)) * 100
            fig.suptitle(f'Red Pitaya Lock-In: {FREQUENCY} Hz, {AMPLITUDE} V  |  Progress: {progress:.1f}%', 
                        fontsize=14)
    
    # Start acquisition
    print("\nStarting measurement...")
    experiment.continuousRecord()
    experiment.StartTask()
    
    # Start animation
    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.tight_layout()
    plt.show()
    
    # Stop and cleanup
    experiment.StopTask()
    experiment.ClearTask()
    
    # Save data
    save_data(experiment, FILE_NAME, SAVE_DIRECTORY)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"Total samples collected: {experiment.numRecord}")
    print(f"Mean signal: {np.mean(experiment.signal):.4f} mA")
    print("="*60)
    
    return experiment


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RED PITAYA LOCK-IN AMPLIFIER - SIMPLE VERSION")
    print("="*60)
    print("\nMAKE SURE:")
    print("  1. Red Pitaya is connected to network")
    print("  2. OUT1 connected to your electrochemical cell")
    print("  3. Cell response connected to IN1")
    print("  4. NI DAQ connected to other signals")
    print("="*60)
    
    input("\nPress ENTER to start experiment...")
    
    try:
        experiment = run_experiment()
    except KeyboardInterrupt:
        print("\n\nExperiment stopped by user")
    except Exception as e:
        print("\n\nERROR:", str(e))
        print("\nTroubleshooting:")
        print("  - Check Red Pitaya network connection")
        print("  - Verify NI DAQ is connected")
        print("  - Check all cables are properly connected")

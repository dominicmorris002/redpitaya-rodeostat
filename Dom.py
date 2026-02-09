"""
Red Pitaya AC Cyclic Voltammetry - Single Device
Synchronized lock-in (X,Y) on IN1 and DC voltage on IN2
No scope buffering - direct module readout for perfect sync
"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
from pyrpl import Pyrpl

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
REF_FREQUENCY = 500  # Hz
REF_AMPLITUDE = 0.2  # V
OUTPUT_CHANNEL = 'out1'
PHASE_OFFSET = 0  # degrees
MEASUREMENT_TIME = 12  # seconds
SAMPLE_INTERVAL = 0.01  # seconds between readings (100 Hz effective rate)

# INPUT MODE FOR IN1 (AC signal): 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE_IN1 = 'MANUAL'
AUTOLAB_GAIN = 1  # Based on Autolab "Current Scale" if Scale = 1mA : Set to 1e-3
MANUAL_GAIN_FACTOR_IN1 = 1 * AUTOLAB_GAIN
MANUAL_DC_OFFSET_IN1 = 0

# INPUT MODE FOR IN2 (DC voltage): 'AUTO', 'LV', 'HV', or 'MANUAL'
INPUT_MODE_IN2 = 'MANUAL'
MANUAL_GAIN_FACTOR_IN2 = 28.93002007 * -1
MANUAL_DC_OFFSET_IN2 = -0.016903

FILTER_BANDWIDTH = 10  # Hz

SAVE_DATA = True
OUTPUT_DIRECTORY = 'test_data'

AUTO_CALIBRATE_IN1 = True
AUTO_CALIBRATE_IN2 = True
CALIBRATION_TIME = 2.0  # seconds
# ============================================================


class RedPitayaACCV:
    """AC Cyclic Voltammetry with synchronized IN1 (lock-in) and IN2 (DC voltage)"""

    def __init__(self, output_dir='test_data', 
                 input_mode_in1='AUTO', manual_gain_in1=1.0, manual_offset_in1=0.0,
                 input_mode_in2='AUTO', manual_gain_in2=1.0, manual_offset_in2=0.0):
        
        self.rp = Pyrpl(config='accv_config', hostname='rp-f073ce.local')
        self.output_dir = output_dir
        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.scope = self.rp_modules.scope

        self.lockin_X = []
        self.lockin_Y = []
        self.dc_voltage = []
        self.capture_times = []

        # Setup IN1 (AC signal) gain and offset
        self.input_gain_factor_in1 = manual_gain_in1
        self.input_dc_offset_in1 = manual_offset_in1
        self.input_mode_setting_in1 = input_mode_in1.upper()
        self.input_mode_in1 = "Unknown"

        if self.input_mode_setting_in1 == 'MANUAL':
            self.input_mode_in1 = f"MANUAL ({manual_gain_in1}x gain, {manual_offset_in1}V offset)"
        elif self.input_mode_setting_in1 == 'LV':
            self.input_gain_factor_in1 = 1.0
            self.input_dc_offset_in1 = 0.0
            self.input_mode_in1 = "LV (±1V)"
        elif self.input_mode_setting_in1 == 'HV':
            self.input_gain_factor_in1 = 20.0
            self.input_dc_offset_in1 = 0.0
            self.input_mode_in1 = "HV (±20V, 20:1 divider)"
        elif self.input_mode_setting_in1 == 'AUTO':
            self.input_mode_in1 = "AUTO (will calibrate)"

        # Setup IN2 (DC voltage) gain and offset
        self.input_gain_factor_in2 = manual_gain_in2
        self.input_dc_offset_in2 = manual_offset_in2
        self.input_mode_setting_in2 = input_mode_in2.upper()
        self.input_mode_in2 = "Unknown"

        if self.input_mode_setting_in2 == 'MANUAL':
            self.input_mode_in2 = f"MANUAL ({manual_gain_in2}x gain, {manual_offset_in2}V offset)"
        elif self.input_mode_setting_in2 == 'LV':
            self.input_gain_factor_in2 = 1.0
            self.input_dc_offset_in2 = 0.0
            self.input_mode_in2 = "LV (±1V)"
        elif self.input_mode_setting_in2 == 'HV':
            self.input_gain_factor_in2 = 20.0
            self.input_dc_offset_in2 = 0.0
            self.input_mode_in2 = "HV (±20V, 20:1 divider)"
        elif self.input_mode_setting_in2 == 'AUTO':
            self.input_mode_in2 = "AUTO (will calibrate)"

        print(f"IN1 (AC) mode: {self.input_mode_in1}")
        print(f"IN2 (DC) mode: {self.input_mode_in2}")

    def calibrate_input_gain(self, input_channel='in1', cal_freq=100, cal_amp=1.0, cal_time=2.0, force=False):
        """
        Calibrate gain and offset for specified input channel
        """
        if input_channel == 'in1':
            if not force and self.input_mode_setting_in1 != 'AUTO':
                print(f"Skipping IN1 calibration - using {self.input_mode_in1}")
                return self.input_gain_factor_in1
        elif input_channel == 'in2':
            if not force and self.input_mode_setting_in2 != 'AUTO':
                print(f"Skipping IN2 calibration - using {self.input_mode_in2}")
                return self.input_gain_factor_in2

        print("\n" + "=" * 60)
        print(f"CALIBRATING {input_channel.upper()} SCALING AND DC OFFSET...")
        print("=" * 60)

        # Step 1: Measure DC offset with no signal
        print(f"Step 1: Measuring DC offset on {input_channel.upper()} (no signal)...")
        self.ref_sig.output_direct = 'off'
        self.lockin.output_direct = 'off'

        # Point scope at the input we're calibrating
        self.scope.input1 = input_channel
        self.scope.input2 = input_channel
        self.scope.decimation = 1024
        time.sleep(0.3)

        offset_samples = []
        for _ in range(10):
            self.scope.single()
            offset_samples.append(np.mean(self.scope._data_ch1_current))

        measured_offset = np.mean(offset_samples)
        print(f"  Measured DC offset: {measured_offset:.6f}V")

        # Step 2: Measure gain with calibration signal
        print(f"\nStep 2: Measuring gain with {cal_amp}V @ {cal_freq} Hz...")

        self.scope.input1 = 'out1'
        self.scope.input2 = input_channel

        self.ref_sig.setup(
            frequency=cal_freq,
            amplitude=cal_amp,
            offset=0,
            waveform='sin',
            trigger_source='immediately'
        )
        self.ref_sig.output_direct = 'out1'

        time.sleep(0.5)

        cal_out1 = []
        cal_in = []
        start_time = time.time()

        while (time.time() - start_time) < cal_time:
            self.scope.single()
            ch1 = np.array(self.scope._data_ch1_current)
            ch2 = np.array(self.scope._data_ch2_current)
            cal_out1.append(ch1)
            cal_in.append(ch2)

        all_out1 = np.concatenate(cal_out1)
        all_in = np.concatenate(cal_in)

        all_in_corrected = all_in - measured_offset

        out1_peak = (np.max(all_out1) - np.min(all_out1)) / 2
        in_peak = (np.max(all_in_corrected) - np.min(all_in_corrected)) / 2

        out1_rms = np.sqrt(np.mean(all_out1 ** 2))
        in_rms = np.sqrt(np.mean(all_in_corrected ** 2))

        measured_gain = out1_peak / in_peak
        gain_rms = out1_rms / in_rms

        # Classify the mode
        if measured_gain < 1.05:
            detected_mode = "LV (±1V)"
        elif measured_gain < 2.0:
            detected_mode = f"LV with loading ({measured_gain:.2f}x)"
        elif measured_gain < 15:
            detected_mode = f"Custom/Unknown mode ({measured_gain:.2f}x)"
        else:
            detected_mode = f"HV (±20V, {measured_gain:.1f}:1 divider)"

        print(f"\n  OUT1 peak: {out1_peak:.4f}V, RMS: {out1_rms:.4f}V")
        print(f"  {input_channel.upper()} peak (after offset correction): {in_peak:.4f}V, RMS: {in_rms:.4f}V")
        print(f"  Gain (peak-based): {measured_gain:.4f}x")
        print(f"  Gain (RMS-based): {gain_rms:.4f}x")
        print(f"  DC offset: {measured_offset:.6f}V")
        print(f"  Detected mode: {detected_mode}")
        print("=" * 60 + "\n")

        # Turn off calibration signal
        self.ref_sig.output_direct = 'off'

        # Store results in appropriate variables
        if input_channel == 'in1':
            self.input_gain_factor_in1 = measured_gain
            self.input_dc_offset_in1 = measured_offset
            self.input_mode_in1 = detected_mode
            return self.input_gain_factor_in1
        else:
            self.input_gain_factor_in2 = measured_gain
            self.input_dc_offset_in2 = measured_offset
            self.input_mode_in2 = detected_mode
            return self.input_gain_factor_in2

    def setup_lockin(self, params):
        """Configure the lock-in on IN1"""
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 10)
        phase_setting = params.get('phase', 0)

        self.ref_sig.output_direct = 'off'

        self.lockin.setup(
            frequency=self.ref_freq,
            bandwidth=filter_bw,
            gain=0.0,
            phase=phase_setting,
            acbandwidth=0,
            amplitude=ref_amp,
            input='in1',
            output_direct=params['output_ref'],
            output_signal='quadrature',
            quadrature_factor=1.0)

        actual_freq = self.lockin.frequency
        actual_amp = self.lockin.amplitude

        print(f"Lock-in frequency: {self.ref_freq} Hz (actual: {actual_freq:.2f} Hz)")
        print(f"Lock-in bandwidth: {filter_bw} Hz")
        print(f"Reference amplitude: {ref_amp}V on {params['output_ref']} (actual: {actual_amp:.3f}V)")

        if abs(actual_freq - self.ref_freq) > 0.1:
            print(f"⚠ WARNING: Requested {self.ref_freq} Hz but PyRPL set {actual_freq:.2f} Hz!")
        
        print(f"IN1 gain correction: {self.input_gain_factor_in1:.4f}x")
        print(f"IN1 DC offset correction: {self.input_dc_offset_in1:.6f}V")
        print(f"IN2 gain correction: {self.input_gain_factor_in2:.4f}x")
        print(f"IN2 DC offset correction: {self.input_dc_offset_in2:.6f}V")

    def capture_synchronized_point(self):
        """
        Capture one synchronized measurement of X, Y (from lock-in on IN1) and DC voltage (from IN2)
        All values are from the same FPGA clock cycle - perfectly synchronized
        """
        capture_time = time.time()
        
        # Read lock-in outputs directly from the IQ module (no scope needed!)
        # These are the demodulated values
        X_raw = self.lockin.output_signal
        Y_raw = self.lockin.quadrature_signal
        
        # Read raw DC voltage from IN2 using scope's voltage reading
        # We'll use the scope as a voltmeter for IN2
        DC_raw = self.scope.voltage_in2
        
        # Apply calibration corrections
        X = X_raw * self.input_gain_factor_in1
        Y = Y_raw * self.input_gain_factor_in1
        DC = (DC_raw - self.input_dc_offset_in2) * self.input_gain_factor_in2
        
        self.lockin_X.append(X)
        self.lockin_Y.append(Y)
        self.dc_voltage.append(DC)
        self.capture_times.append(capture_time)
        
        return X, Y, DC

    def run(self, params):
        """Main acquisition loop for ACCV"""

        # Run calibration if requested
        if params.get('auto_calibrate_in1', False):
            self.calibrate_input_gain(
                input_channel='in1',
                cal_freq=params['ref_freq'],
                cal_amp=params['ref_amp'],
                cal_time=params.get('calibration_time', 2.0)
            )
        
        if params.get('auto_calibrate_in2', False):
            self.calibrate_input_gain(
                input_channel='in2',
                cal_freq=100,  # Use 100 Hz for IN2 calibration
                cal_amp=1.0,
                cal_time=params.get('calibration_time', 2.0)
            )

        # Setup lock-in
        self.setup_lockin(params)
        print("Allowing lock-in to settle...")
        time.sleep(0.5)

        acquisition_start_time = time.time()
        sample_count = 0
        print(f"Started: {datetime.fromtimestamp(acquisition_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        # Collect synchronized data points
        loop_start = time.time()
        sample_interval = params.get('sample_interval', 0.01)
        
        while (time.time() - loop_start) < params['timeout']:
            self.capture_synchronized_point()
            sample_count += 1
            time.sleep(sample_interval)

        acquisition_end_time = time.time()
        actual_duration = acquisition_end_time - acquisition_start_time

        print(f"✓ Captured {sample_count} synchronized measurements")

        # Convert to numpy arrays
        all_X = np.array(self.lockin_X)
        all_Y = np.array(self.lockin_Y)
        all_DC = np.array(self.dc_voltage)
        all_times = np.array(self.capture_times) - acquisition_start_time

        # Calculate polar coordinates
        R = np.sqrt(all_X ** 2 + all_Y ** 2)
        Theta = np.degrees(np.arctan2(all_Y, all_X))

        # Statistics
        print("\n" + "=" * 60)
        print("AC CYCLIC VOLTAMMETRY RESULTS")
        print("=" * 60)
        print(f"IN1 mode: {self.input_mode_in1}")
        print(f"IN2 mode: {self.input_mode_in2}")
        print(f"Duration: {actual_duration:.3f}s")
        print(f"Samples collected: {sample_count}")
        print(f"Effective sample rate: {sample_count / actual_duration:.2f} Hz")
        print(f"\nMean R: {np.mean(R):.6f} ± {np.std(R):.6f} V")
        print(f"Mean X: {np.mean(all_X):.6f} ± {np.std(all_X):.6f} V")
        print(f"Mean Y: {np.mean(all_Y):.6f} ± {np.std(all_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.3f} ± {np.std(Theta):.3f}°")
        print(f"Mean DC Voltage: {np.mean(all_DC):.6f} ± {np.std(all_DC):.6f} V")
        print(f"\nCorrelations:")
        print(f"  R vs DC:     {np.corrcoef(R, all_DC)[0,1]:.4f}")
        print(f"  Theta vs DC: {np.corrcoef(Theta, all_DC)[0,1]:.4f}")
        print("=" * 60)

        # Create plots
        fig = plt.figure(figsize=(18, 12))

        # Row 1: Time series
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(all_times, all_X, 'b-', linewidth=1)
        ax1.axhline(np.mean(all_X), color='r', linestyle='--', label=f'Mean: {np.mean(all_X):.6f}V')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('X (V)')
        ax1.set_title('In-phase (X) vs Time')
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(all_times, all_Y, 'r-', linewidth=1)
        ax2.axhline(np.mean(all_Y), color='b', linestyle='--', label=f'Mean: {np.mean(all_Y):.6f}V')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Y (V)')
        ax2.set_title('Quadrature (Y) vs Time')
        ax2.legend()
        ax2.grid(True)

        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(all_times, all_DC, 'g-', linewidth=1)
        ax3.axhline(np.mean(all_DC), color='r', linestyle='--', label=f'Mean: {np.mean(all_DC):.6f}V')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('DC Voltage (V)')
        ax3.set_title('DC Voltage (IN2) vs Time')
        ax3.legend()
        ax3.grid(True)

        # Row 2: Polar representation
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(all_times, R * 1e6, 'm-', linewidth=1)
        ax4.axhline(np.mean(R) * 1e6, color='b', linestyle='--', label=f'Mean: {np.mean(R)*1e6:.6f}µV')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('R (µV)')
        ax4.set_title('Magnitude vs Time')
        ax4.legend()
        ax4.grid(True)

        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(all_times, Theta, 'c-', linewidth=1)
        ax5.axhline(np.mean(Theta), color='r', linestyle='--', label=f'Mean: {np.mean(Theta):.3f}°')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Theta (°)')
        ax5.set_title('Phase vs Time')
        ax5.legend()
        ax5.grid(True)

        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(all_X, all_Y, 'g.', markersize=2, alpha=0.5)
        ax6.plot(np.mean(all_X), np.mean(all_Y), 'r+', markersize=15, markeredgewidth=2, label='Mean')
        ax6.set_xlabel('X (V)')
        ax6.set_ylabel('Y (V)')
        ax6.set_title('IQ Plot')
        ax6.legend()
        ax6.grid(True)
        ax6.axis('equal')

        # Row 3: ACCV plots (the money shots!)
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(all_DC, R, 'b-', linewidth=2, alpha=0.8)
        ax7.set_xlabel('DC Potential (V)', fontweight='bold', fontsize=12)
        ax7.set_ylabel('AC Magnitude R (V)', fontweight='bold', fontsize=12, color='b')
        ax7.tick_params(axis='y', labelcolor='b')
        ax7.set_title('AC Magnitude vs DC Potential', fontweight='bold', fontsize=13)
        ax7.grid(True, alpha=0.3)

        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(all_DC, Theta, 'r-', linewidth=2, alpha=0.8)
        ax8.set_xlabel('DC Potential (V)', fontweight='bold', fontsize=12)
        ax8.set_ylabel('Phase Angle (°)', fontweight='bold', fontsize=12, color='r')
        ax8.tick_params(axis='y', labelcolor='r')
        ax8.set_title('Phase vs DC Potential', fontweight='bold', fontsize=13)
        ax8.grid(True, alpha=0.3)

        ax9 = plt.subplot(3, 3, 9)
        scatter = ax9.scatter(all_DC, R, c=Theta, s=20, alpha=0.6, cmap='viridis')
        ax9.set_xlabel('DC Potential (V)', fontweight='bold', fontsize=12)
        ax9.set_ylabel('AC Magnitude R (V)', fontweight='bold', fontsize=12)
        ax9.set_title('AC Response Map (colored by phase)', fontweight='bold', fontsize=13)
        ax9.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax9)
        cbar.set_label('Phase (°)', fontweight='bold', fontsize=10)

        fig.suptitle(f'AC Cyclic Voltammetry - Single Red Pitaya\n{sample_count} points @ {sample_count/actual_duration:.1f} Hz',
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save everything
        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save plot
            img_path = os.path.join(self.output_dir, f'accv_single_rp_{timestamp_str}.png')
            plt.savefig(img_path, dpi=150)
            print(f"\n✓ Saved plot: {img_path}")

            # Save CSV
            data = np.column_stack((np.arange(sample_count), all_times, R, Theta, all_X, all_Y, all_DC))
            csv_path = os.path.join(self.output_dir, f'accv_single_rp_{timestamp_str}.csv')

            with open(csv_path, 'w', newline='') as f:
                f.write(f"# Red Pitaya AC Cyclic Voltammetry - Single Device\n")
                f.write(f"# IN1 mode: {self.input_mode_in1}\n")
                f.write(f"# IN1 gain correction: {self.input_gain_factor_in1:.6f}\n")
                f.write(f"# IN1 DC offset correction: {self.input_dc_offset_in1:.6f} V\n")
                f.write(f"# IN2 mode: {self.input_mode_in2}\n")
                f.write(f"# IN2 gain correction: {self.input_gain_factor_in2:.6f}\n")
                f.write(f"# IN2 DC offset correction: {self.input_dc_offset_in2:.6f} V\n")
                f.write(f"# Reference frequency: {self.ref_freq} Hz\n")
                f.write(f"# Reference amplitude: {params['ref_amp']} V\n")
                f.write(f"# Filter bandwidth: {params.get('filter_bandwidth', 10)} Hz\n")
                f.write(f"# Duration: {actual_duration:.3f} s\n")
                f.write(f"# Samples: {sample_count}\n")
                f.write(f"# Sample rate: {sample_count / actual_duration:.2f} Hz\n")
                f.write("Index,Time(s),R(V),Theta(°),X(V),Y(V),DC_Voltage(V)\n")
                np.savetxt(f, data, delimiter=",", fmt='%.10f')

            print(f"✓ Saved data: {csv_path}")
        else:
            plt.show()


if __name__ == '__main__':
    rp = RedPitayaACCV(
        output_dir=OUTPUT_DIRECTORY,
        input_mode_in1=INPUT_MODE_IN1,
        manual_gain_in1=MANUAL_GAIN_FACTOR_IN1,
        manual_offset_in1=MANUAL_DC_OFFSET_IN1,
        input_mode_in2=INPUT_MODE_IN2,
        manual_gain_in2=MANUAL_GAIN_FACTOR_IN2,
        manual_offset_in2=MANUAL_DC_OFFSET_IN2
    )

    run_params = {
        'ref_freq': REF_FREQUENCY,
        'ref_amp': REF_AMPLITUDE,
        'output_ref': OUTPUT_CHANNEL,
        'phase': PHASE_OFFSET,
        'timeout': MEASUREMENT_TIME,
        'sample_interval': SAMPLE_INTERVAL,
        'filter_bandwidth': FILTER_BANDWIDTH,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
        'auto_calibrate_in1': AUTO_CALIBRATE_IN1,
        'auto_calibrate_in2': AUTO_CALIBRATE_IN2,
        'calibration_time': CALIBRATION_TIME,
    }

    print("=" * 60)
    print("RED PITAYA AC CYCLIC VOLTAMMETRY - SINGLE DEVICE")
    print("=" * 60)
    print(f"Reference: {REF_FREQUENCY} Hz @ {REF_AMPLITUDE}V")
    print(f"Filter bandwidth: {FILTER_BANDWIDTH} Hz")
    print(f"Measurement time: {MEASUREMENT_TIME}s")
    print(f"Sample interval: {SAMPLE_INTERVAL}s ({1/SAMPLE_INTERVAL:.1f} Hz)")
    print(f"IN1 (AC) mode: {INPUT_MODE_IN1}")
    print(f"IN2 (DC) mode: {INPUT_MODE_IN2}")
    print("=" * 60)

    rp.run(run_params)

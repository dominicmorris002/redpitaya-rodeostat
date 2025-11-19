"""
Red Pitaya Lock-In Amplifier
Replaces SR7280 Lock-In Amplifier for ACV Measurements

EASY ACCESS PARAMETERS - MODIFY THESE FOR YOUR EXPERIMENT
"""

# ============================================================
# MEASUREMENT PARAMETERS - CHANGE THESE
# ============================================================
REF_FREQUENCY = 100  # Hz - AC excitation frequency (typical: 50-1000 Hz)
REF_AMPLITUDE = 0.4  # V - AC signal amplitude (typical: 0.1-0.5 V)
OUTPUT_CHANNEL = 'out1'  # 'out1' or 'out2' - where to send AC signal
PHASE_OFFSET = 0  # degrees - phase adjustment (0, 90, 180, 270)

MEASUREMENT_TIME = 10.0  # seconds - how long to measure (longer = more accurate)
AVERAGING_WINDOW = 50  # samples - moving average for noise reduction
# HIGHER = MORE ACCURATE but slower response
# For science: 50-100, For testing: 10, For fast response: 1

# LOW-PASS FILTER BANDWIDTH (CRITICAL FOR CLEAN SIGNALS!)
FILTER_BANDWIDTH = 2  # Hz - LOWER = cleaner but slower settling
# For precise measurements: 1-5 Hz (recommended)
# For fast response: 10-20 Hz
# Your old setting was ±200 Hz which let through 200 Hz ripple!

# Data saving
SAVE_DATA = False  # True = save to files, False = just show plots
OUTPUT_DIRECTORY = 'test_data'  # where to save results

# Advanced settings (usually don't need to change)
DECIMATION = 64  # sample rate = 125MHz/decimation (64 = good default)
SHOW_FFT = True  # whether to compute and show FFT

# ============================================================

import math
from pyrpl import Pyrpl
import numpy as np
from matplotlib import pyplot as plt
import time
import csv
import os

N_FFT_SHOW = 10


class RedPitaya:
    electrode_map = {'A': (False, False), 'B': (True, False), 'C': (False, True), 'D': (True, True)}
    current_range_map = {'10uA': (False, True, True, True), '100uA': (True, False, True, True),
                         '1mA': (True, True, False, True), '10mA': (True, True, True, False)}
    dac_gain_map = {'1X': (False, False), '5X': (False, True), '2X': (True, False), '10X': (True, True)}
    current_scaling_map = {'10mA': 65, '1mA': 600, '100uA': 6000, '10uA': 60000}
    allowed_decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, output_dir='test_data'):
        self.rp = Pyrpl(config='lockin_config', hostname='rp-f073ce.local')
        self.output_dir = output_dir

        self.rp_modules = self.rp.rp
        self.lockin = self.rp_modules.iq2
        self.ref_sig = self.rp_modules.asg0
        self.ref_start_t = 0.0
        self.lockin_X = []
        self.all_X = []
        self.lockin_Y = []
        self.all_Y = []

        self.pid = self.rp_modules.pid0
        self.kp = self.pid.p
        self.ki = self.pid.i
        self.ival = self.pid.ival

        self.scope = self.rp_modules.scope
        self.scope.input1 = 'iq2'
        self.scope.input2 = 'iq2_2'
        self.scope.decimation = 64

        if self.scope.decimation not in self.allowed_decimations:
            print('Invalid decimation')
            exit()

        self.scope._start_acquisition_rolling_mode()
        self.scope.average = 'true'
        self.sample_rate = 125e6 / self.scope.decimation

    def setup_lockin(self, params):
        self.ref_freq = params['ref_freq']
        self.ref_period = 1 / self.ref_freq
        ref_amp = params['ref_amp']
        filter_bw = params.get('filter_bandwidth', 2)  # Get filter bandwidth from params

        # Get phase setting - can be 'auto' or a specific value in degrees
        phase_setting = params.get('phase', 0)

        self.ref_sig.setup(waveform='sin',
                           amplitude=ref_amp,
                           frequency=self.ref_freq)

        self.ref_start_t = time.time()

        if params['output_ref'] == 'out1' or params['output_ref'] == 'out2':
            self.ref_sig.output_direct = params['output_ref']
        else:
            self.ref_sig.output_direct = 'off'

        # FIXED BANDWIDTH - now using narrow filter for clean DC output
        self.lockin.setup(frequency=self.ref_freq,
                          bandwidth=[filter_bw],  # Single value = symmetric bandwidth ±filter_bw Hz
                          gain=1.0,
                          phase=phase_setting,
                          acbandwidth=0,
                          amplitude=ref_amp,
                          input='in1',
                          output_direct='out2',
                          output_signal='output_direct',
                          quadrature_factor=10)
        
        print(f"Lock-in configured: {self.ref_freq} Hz, ±{filter_bw} Hz bandwidth")

    def capture_lockin(self):
        """
        captures a self.scope.decimation length capture and appends them to the X and Y arrays
        :return:
        """
        self.scope.single()
        ch1 = np.array(self.scope._data_ch1_current)
        ch2 = np.array(self.scope._data_ch2_current)

        if self.scope.input1 == 'iq2' and self.scope.input2 == 'iq2_2':
            self.lockin_X.append(ch1)
            self.lockin_Y.append(ch2)

        return ch1, ch2
    
    def capture_raw_signals(self):
        """
        Capture raw input signal (in1) and reference output (asg0) for diagnostics
        """
        # Temporarily switch scope to raw signals
        original_input1 = self.scope.input1
        original_input2 = self.scope.input2
        
        self.scope.input1 = 'in1'  # Raw input signal
        self.scope.input2 = 'asg0'  # Reference signal
        
        self.scope.single()
        raw_input = np.array(self.scope._data_ch1_current)
        raw_ref = np.array(self.scope._data_ch2_current)
        
        # Restore original scope settings
        self.scope.input1 = original_input1
        self.scope.input2 = original_input2
        
        return raw_input, raw_ref

    def run(self, params):
        timeout = params['timeout']

        self.setup_lockin(params)
        time.sleep(0.1)  # Increased settling time for narrow filter

        loop_start = time.time()

        while (time.time() - loop_start) < timeout:
            self.capture_lockin()

        self.all_X = np.array(np.concatenate(self.lockin_X))
        self.all_Y = np.array(np.concatenate(self.lockin_Y))

        # Apply moving average filter for more accurate measurements
        averaging_window = params.get('averaging_window', 1)
        if averaging_window > 1:
            # Smooth X and Y with moving average
            self.all_X = np.convolve(self.all_X, np.ones(averaging_window) / averaging_window, mode='valid')
            self.all_Y = np.convolve(self.all_Y, np.ones(averaging_window) / averaging_window, mode='valid')

        R = np.sqrt(self.all_X ** 2 + self.all_Y ** 2)
        Theta = np.arctan2(self.all_Y, self.all_X)

        # Time array
        t = np.arange(start=0, stop=len(self.all_X) / self.sample_rate, step=1 / self.sample_rate)

        # FFT calculations
        iq = self.all_X + 1j * self.all_Y
        n_pts = len(iq)
        win = np.hanning(n_pts)
        IQwin = iq * win
        IQfft = np.fft.fftshift(np.fft.fft(IQwin))
        freqs_lock = np.fft.fftshift(np.fft.fftfreq(n_pts, 1.0 / self.sample_rate))
        psd_lock = (np.abs(IQfft) ** 2) / (self.sample_rate * np.sum(win ** 2))

        idx = np.argmax(psd_lock)
        
        # Calculate ripple frequency and amplitude (diagnostic for filter quality)
        # Look for second-highest peak (should be 2x ref freq ripple)
        psd_copy = psd_lock.copy()
        psd_copy[idx] = 0  # Zero out the main peak
        idx2 = np.argmax(psd_copy)
        ripple_freq = freqs_lock[idx2]
        ripple_power_ratio = psd_copy[idx2] / psd_lock[idx]
        
        # Calculate signal quality metrics
        x_ripple_pp = np.max(self.all_X) - np.min(self.all_X)  # peak-to-peak ripple
        y_ripple_pp = np.max(self.all_Y) - np.min(self.all_Y)
        r_ripple_pp = np.max(R) - np.min(R)
        
        # Calculate Allan deviation (stability metric for precision measurements)
        # This tells you how stable your measurement is over time
        def allan_deviation(data, max_tau=None):
            """Calculate Allan deviation for different averaging times"""
            if max_tau is None:
                max_tau = len(data) // 10
            taus = []
            adevs = []
            for tau in range(1, max_tau, max(1, max_tau // 20)):
                # Compute successive differences with averaging window tau
                m = len(data) // tau
                if m < 2:
                    break
                bins = data[:m*tau].reshape(m, tau).mean(axis=1)
                adev = np.sqrt(0.5 * np.mean(np.diff(bins)**2))
                taus.append(tau / self.sample_rate)  # Convert to seconds
                adevs.append(adev)
            return np.array(taus), np.array(adevs)
        
        allan_taus, allan_devs = allan_deviation(R)
        
        print("=" * 70)
        print("LOCK-IN AMPLIFIER DIAGNOSTICS")
        print("=" * 70)
        print("\n[CONFIGURATION]")
        print(f"  Reference Frequency:     {self.ref_freq} Hz")
        print(f"  Reference Amplitude:     {params['ref_amp']} V")
        print(f"  Filter Bandwidth:        ±{params.get('filter_bandwidth', 'N/A')} Hz")
        print(f"  Phase Offset:            {params.get('phase', 0)}°")
        print(f"  Sample Rate:             {self.sample_rate:.2f} Hz")
        print(f"  Measurement Duration:    {len(self.all_X) / self.sample_rate:.3f} s")
        print(f"  Total Samples:           {len(self.all_X)}")
        print(f"  Averaging Window:        {params.get('averaging_window', 1)} samples")
        
        print("\n[FREQUENCY ANALYSIS]")
        print(f"  FFT Peak Frequency:      {freqs_lock[idx]:.4f} Hz")
        print(f"  Peak Offset from DC:     {abs(freqs_lock[idx]):.4f} Hz")
        if abs(freqs_lock[idx]) > 0.5:
            print(f"  ⚠ WARNING: Peak not at DC! Lock may be unstable.")
        else:
            print(f"  ✓ Peak at DC - good lock")
        
        print(f"\n  Secondary Peak at:       {ripple_freq:.2f} Hz")
        print(f"  Ripple Power Ratio:      {ripple_power_ratio:.2e}")
        if abs(ripple_freq) > self.ref_freq * 1.5 and abs(ripple_freq) < self.ref_freq * 2.5:
            print(f"  → This is 2×f₀ mixing product (expected)")
            if ripple_power_ratio > 0.01:
                print(f"  ⚠ Ripple strong - consider narrower filter")
            else:
                print(f"  ✓ Ripple well-suppressed")
        
        print("\n[MAGNITUDE (R) - YOUR MAIN MEASUREMENT]")
        print(f"  Mean R:                  {np.mean(R):.6f} V")
        print(f"  Std Dev R:               {np.std(R):.6f} V")
        print(f"  Relative Uncertainty:    {(np.std(R)/np.mean(R))*100:.3f} %")
        print(f"  SNR (R):                 {np.mean(R) / np.std(R):.2f}")
        print(f"  R Range:                 [{np.min(R):.6f}, {np.max(R):.6f}] V")
        print(f"  Peak-to-Peak Ripple:     {r_ripple_pp:.6f} V ({(r_ripple_pp/np.mean(R))*100:.2f}%)")
        if (np.std(R)/np.mean(R)) < 0.01:
            print(f"  ✓ Excellent precision (<1% variation)")
        elif (np.std(R)/np.mean(R)) < 0.05:
            print(f"  ✓ Good precision (<5% variation)")
        else:
            print(f"  ⚠ High noise - increase averaging or check connections")
        
        print("\n[PHASE (Theta)]")
        print(f"  Mean Theta:              {np.mean(Theta):.6f} rad ({np.degrees(np.mean(Theta)):.2f}°)")
        print(f"  Std Dev Theta:           {np.std(Theta):.6f} rad ({np.degrees(np.std(Theta)):.2f}°)")
        print(f"  Theta Range:             [{np.min(Theta):.6f}, {np.max(Theta):.6f}] rad")
        if np.std(Theta) < 0.1:
            print(f"  ✓ Excellent phase stability")
        elif np.std(Theta) < 0.5:
            print(f"  ✓ Good phase stability")
        else:
            print(f"  ⚠ Phase unstable - check reference connection")
        
        print("\n[IN-PHASE (X) & QUADRATURE (Y)]")
        print(f"  Mean X:                  {np.mean(self.all_X):.6f} V ± {np.std(self.all_X):.6f} V")
        print(f"  X Peak-to-Peak Ripple:   {x_ripple_pp:.6f} V")
        print(f"  Mean Y:                  {np.mean(self.all_Y):.6f} V ± {np.std(self.all_Y):.6f} V")
        print(f"  Y Peak-to-Peak Ripple:   {y_ripple_pp:.6f} V")
        
        # Check if signal is mostly in X (good) or split between X and Y (phase offset)
        x_magnitude = abs(np.mean(self.all_X))
        y_magnitude = abs(np.mean(self.all_Y))
        if x_magnitude > 5 * y_magnitude:
            print(f"  ✓ Signal mostly in X - phase aligned")
        elif y_magnitude > 5 * x_magnitude:
            print(f"  ⚠ Signal mostly in Y - try phase adjustment")
        else:
            print(f"  → Signal in both X&Y - may need phase optimization")
        
        print("\n[STABILITY ANALYSIS]")
        if len(allan_devs) > 0:
            best_stability_idx = np.argmin(allan_devs)
            print(f"  Best stability at τ={allan_taus[best_stability_idx]:.3f}s: σ={allan_devs[best_stability_idx]:.2e} V")
            print(f"  1-second Allan dev:      {allan_devs[0] if len(allan_devs) > 0 else 'N/A':.2e} V")
        
        # Drift analysis
        first_half_mean = np.mean(R[:len(R)//2])
        second_half_mean = np.mean(R[len(R)//2:])
        drift = abs(second_half_mean - first_half_mean)
        drift_percent = (drift / np.mean(R)) * 100
        print(f"  Signal Drift (1st→2nd half): {drift:.6f} V ({drift_percent:.3f}%)")
        if drift_percent < 1:
            print(f"  ✓ Minimal drift - very stable")
        elif drift_percent < 5:
            print(f"  ✓ Low drift")
        else:
            print(f"  ⚠ Significant drift - check temperature/connections")
        
        print("\n[MEASUREMENT QUALITY SUMMARY]")
        quality_score = 0
        checks = []
        
        if abs(freqs_lock[idx]) < 0.5:
            quality_score += 1
            checks.append("✓ Locked to DC")
        else:
            checks.append("✗ Not locked to DC")
            
        if (np.std(R)/np.mean(R)) < 0.05:
            quality_score += 1
            checks.append("✓ Low noise")
        else:
            checks.append("✗ High noise")
            
        if np.std(Theta) < 0.5:
            quality_score += 1
            checks.append("✓ Stable phase")
        else:
            checks.append("✗ Unstable phase")
            
        if drift_percent < 5:
            quality_score += 1
            checks.append("✓ Minimal drift")
        else:
            checks.append("✗ Significant drift")
            
        for check in checks:
            print(f"  {check}")
        
        print(f"\n  Overall Quality: {quality_score}/4", end="")
        if quality_score == 4:
            print(" - EXCELLENT ✓")
        elif quality_score == 3:
            print(" - GOOD")
        elif quality_score == 2:
            print(" - ACCEPTABLE")
        else:
            print(" - POOR ⚠")
        
        print("\n[RECOMMENDED RESULT]")
        print(f"  ╔═══════════════════════════════════════════════╗")
        print(f"  ║  Measured Amplitude: {np.mean(R):.6f} ± {np.std(R):.6f} V  ║")
        print(f"  ║  Phase Angle:        {np.degrees(np.mean(Theta)):7.2f} ± {np.degrees(np.std(Theta)):5.2f}°    ║")
        print(f"  ╚═══════════════════════════════════════════════╝")
        
        print("=" * 70)

        # Create comprehensive plot with all lock-in outputs
        fig = plt.figure(figsize=(16, 10))

        # 1. FFT Spectrum
        ax1 = plt.subplot(3, 3, 1)
        ax1.semilogy(freqs_lock, psd_lock, label='Lock-in PSD')
        ax1.axvline(0, color='r', linestyle='--', alpha=0.5, label='DC (0 Hz)')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (a.u.)')
        ax1.set_title('FFT Spectrum (baseband)')
        ax1.legend()
        ax1.grid(True)

        # 2. X vs Time
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(t, self.all_X, 'b-', linewidth=0.5, alpha=0.7)
        ax2.axhline(np.mean(self.all_X), color='r', linestyle='--', label=f'Mean: {np.mean(self.all_X):.4f}V')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('X (V)')
        ax2.set_title('In-phase (X) vs Time')
        ax2.legend()
        ax2.grid(True)

        # 3. Y vs Time
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(t, self.all_Y, 'r-', linewidth=0.5, alpha=0.7)
        ax3.axhline(np.mean(self.all_Y), color='b', linestyle='--', label=f'Mean: {np.mean(self.all_Y):.4f}V')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Y (V)')
        ax3.set_title('Quadrature (Y) vs Time')
        ax3.legend()
        ax3.grid(True)

        # 4. X vs Y (IQ plot) - Full data
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(self.all_X, self.all_Y, 'g.', markersize=1, alpha=0.5)
        ax4.plot(np.mean(self.all_X), np.mean(self.all_Y), 'r*', markersize=15, label='Mean')
        ax4.set_xlabel('X (V)')
        ax4.set_ylabel('Y (V)')
        ax4.set_title('IQ Plot (X vs Y) - Full Trace')
        ax4.legend()
        ax4.grid(True)
        ax4.axis('equal')

        # 5. R vs Time
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(t, R, 'm-', linewidth=0.5, alpha=0.7)
        ax5.axhline(np.mean(R), color='k', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(R):.6f}V')
        ax5.fill_between(t, np.mean(R)-np.std(R), np.mean(R)+np.std(R), 
                         alpha=0.2, color='gray', label=f'±1σ: {np.std(R):.6f}V')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('R (V)')
        ax5.set_title('Magnitude (R) vs Time - YOUR MEASUREMENT')
        ax5.legend()
        ax5.grid(True)

        # 6. Theta vs Time
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(t, Theta, 'c-', linewidth=0.5, alpha=0.7)
        ax6.axhline(np.mean(Theta), color='k', linestyle='--', 
                    label=f'Mean: {np.degrees(np.mean(Theta)):.2f}°')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Theta (rad)')
        ax6.set_title('Phase (Theta) vs Time')
        ax6.legend()
        ax6.grid(True)

        # 7. Allan Deviation Plot (stability)
        ax7 = plt.subplot(3, 3, 7)
        if len(allan_taus) > 0:
            ax7.loglog(allan_taus, allan_devs, 'b-o', markersize=4)
            ax7.set_xlabel('Averaging Time τ (s)')
            ax7.set_ylabel('Allan Deviation (V)')
            ax7.set_title('Stability vs Averaging Time')
            ax7.grid(True, which='both', alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Not enough data\nfor Allan deviation', 
                    ha='center', va='center', transform=ax7.transAxes)

        # 8. Phasor plot - Short window
        ax8 = plt.subplot(3, 3, 8)
        window_samples = int(0.1 * self.sample_rate)
        if len(self.all_X) > window_samples:
            X_window = self.all_X[:window_samples]
            Y_window = self.all_Y[:window_samples]
        else:
            X_window = self.all_X
            Y_window = self.all_Y

        ax8.plot(X_window, Y_window, 'b-', linewidth=1.5, alpha=0.8)
        ax8.plot(X_window[0], Y_window[0], 'go', markersize=8, label='Start')
        ax8.plot(X_window[-1], Y_window[-1], 'ro', markersize=8, label='End')
        ax8.set_xlabel('X (V)')
        ax8.set_ylabel('Y (V)')
        ax8.set_title('Phasor Plot (0.1s window)')
        ax8.legend()
        ax8.grid(True)
        ax8.axis('equal')

        max_r = np.max(np.sqrt(X_window ** 2 + Y_window ** 2))
        circle = plt.Circle((0, 0), max_r, color='gray', fill=False, linestyle='--', alpha=0.3)
        ax8.add_patch(circle)

        # 9. Histogram of R values
        ax9 = plt.subplot(3, 3, 9)
        ax9.hist(R, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax9.axvline(np.mean(R), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(R):.6f}V')
        ax9.set_xlabel('R (V)')
        ax9.set_ylabel('Count')
        ax9.set_title('Magnitude Distribution')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.suptitle(f'Lock-In Results | {self.ref_freq}Hz, {params["ref_amp"]}V, BW=±{params.get("filter_bandwidth", "?")}Hz | SNR: {np.mean(R)/np.std(R):.1f}',
                    fontsize=12, fontweight='bold', y=0.998)
        plt.tight_layout(rect=[0, 0, 1, 0.995])

        if params['save_file']:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            img_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.png')
            data = np.column_stack((t, R, Theta, self.all_X, self.all_Y))
            csv_path = os.path.join(self.output_dir, f'lockin_results_rf_{self.ref_freq}.csv')
            np.savetxt(csv_path, data, delimiter=",", header="Time(s),R(V),Theta(rad),X(V),Y(V)", comments='', fmt='%.9f')
            plt.savefig(img_path, dpi=150)
            print(f"\n[FILES SAVED]")
            print(f"  Plot: {img_path}")
            print(f"  Data: {csv_path}")
        else:
            plt.show()


if __name__ == '__main__':
    rp = RedPitaya()

    # NOW USING THE TOP PARAMETERS!
    run_params = {
        'ref_freq': REF_FREQUENCY,
        'ref_amp': REF_AMPLITUDE,
        'output_ref': OUTPUT_CHANNEL,
        'phase': PHASE_OFFSET,
        'filter_bandwidth': FILTER_BANDWIDTH,  # NEW: narrow filter for clean signals
        'timeout': MEASUREMENT_TIME,
        'averaging_window': AVERAGING_WINDOW,
        'output_dir': OUTPUT_DIRECTORY,
        'save_file': SAVE_DATA,
        'fft': SHOW_FFT,
    }

    # For finding optimal phase (gives cleanest circle in phasor plots):
    # Try: 0, 45, 90, 135, 180, 225, 270, 315 degrees
    # Pick the one where X is maximum and Y is minimum

    rp.run(run_params)

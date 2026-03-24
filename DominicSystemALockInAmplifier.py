"""
Red Pitaya Lock-In Amplifier

Uses PyRPL's built-in lock-in (iq2 and iq2_2 module) to log data of X(In-Phase) and Y(Out-Phase).
For testing: connect OUT1 to IN1 with a cable.
Supports LV, HV, or MANUAL gain modes.
Phase is in degrees!

I recommend CONTINUOUS Mode for good Data.

Have a great day :)

Dominic Morris
"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
from pyrpl import Pyrpl
from scipy.signal import find_peaks

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
Ref_Frequency = 500       # Hz
Ref_Amplitude = 0.2       # V
Output_Channel = 'out1'
Phase_Offset = 0           # degrees
Measurement_Time = 15      # seconds

# INPUT MODE: 'LV', 'HV', or 'MANUAL'
Input_Mode = 'MANUAL'
Autolab_Gain = 2.569
Manual_Gain_Factor = 1 * Autolab_Gain
Manual_Dc_Offset = 0

Filter_Bandwidth = 6       # Hz

# Controls scope sample rate: sample_rate = 125e6 / Decimation
Decimation = 8

Averaging_Window = 1       # samples (moving average on logged data)

Save_Data = True
Output_Directory = 'test_data'

Calibration_Time = 2.0     # seconds

# ACQUISITION MODE: 'SINGLE_SHOT' or 'CONTINUOUS'
Acquisition_Mode = 'CONTINUOUS'

# ============================================================
# PLOT DOWNSAMPLING
# CSV always saves full resolution data.
# ============================================================
Plot_Downsample_Enabled = True
Plot_Max_Points = 50_000

Start_Time_File = "start_time.txt"
try:
    with open(Start_Time_File, "r") as f:
        Start_Time = datetime.fromisoformat(f.read().strip())
    while datetime.now() < Start_Time:
        time.sleep(0.001)
except FileNotFoundError:
    pass


# ============================================================
# Data Helpers
# ============================================================

def Downsample(Arrays, N_Samples, Max_Points, Enabled=True):
    if not Enabled or N_Samples <= Max_Points:
        return Arrays, 1, N_Samples
    Step = max(1, N_Samples // Max_Points)
    Ds = [Arr[::Step] for Arr in Arrays]
    return Ds, Step, len(Ds[0])


def Detect_Peaks_And_Annotate(Ax, T, Signal, Unit_Str, Color_Peak='red',
                               Color_Trough='blue', Min_Prominence_Frac=0.1,
                               Scale=1.0, Label_Prefix=''):
    Sig = Signal * Scale
    Sig_Range = np.max(Sig) - np.min(Sig)
    if Sig_Range == 0:
        return False

    Prominence = Min_Prominence_Frac * Sig_Range
    Peak_Idx, _   = find_peaks( Sig, prominence=Prominence, distance=max(1, len(Sig) // 30))
    Trough_Idx, _ = find_peaks(-Sig, prominence=Prominence, distance=max(1, len(Sig) // 30))

    if len(Peak_Idx) < 2 and len(Trough_Idx) < 1:
        return False

    Peak_Times, Peak_Vals = [], []
    for I, Idx in enumerate(Peak_Idx):
        Pt, Pv = T[Idx], Sig[Idx]
        Peak_Times.append(Pt); Peak_Vals.append(Pv)
        Ax.plot(Pt, Pv, 'v', color=Color_Peak, markersize=8, zorder=5)
        Ax.annotate(f'P{I+1}\n{Pv:.0f}{Unit_Str}',
                    xy=(Pt, Pv), xytext=(0, 12), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6.5, color=Color_Peak,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                              ec=Color_Peak, lw=0.8))

    Trough_Vals = []
    for I, Idx in enumerate(Trough_Idx):
        Tt, Tv = T[Idx], Sig[Idx]
        Trough_Vals.append(Tv)
        Ax.plot(Tt, Tv, '^', color=Color_Trough, markersize=8, zorder=5)
        Ax.annotate(f'T{I+1}\n{Tv:.0f}{Unit_Str}',
                    xy=(Tt, Tv), xytext=(0, -14), textcoords='offset points',
                    ha='center', va='top', fontsize=6.5, color=Color_Trough,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7,
                              ec=Color_Trough, lw=0.8))

    if len(Peak_Idx) >= 2:
        Arrow_Y = np.max(Sig) + 0.08 * Sig_Range
        Pp_Times = []
        for I in range(len(Peak_Idx) - 1):
            T0, T1 = Peak_Times[I], Peak_Times[I + 1]
            Dt = T1 - T0
            Pp_Times.append(Dt)
            Ax.annotate('', xy=(T1, Arrow_Y), xytext=(T0, Arrow_Y),
                        arrowprops=dict(arrowstyle='<->', color='purple',
                                        lw=1.3, shrinkA=0, shrinkB=0))
            Ax.text((T0 + T1) / 2, Arrow_Y + 0.01 * Sig_Range,
                    f'dt={Dt*1000:.1f}ms', ha='center', va='bottom',
                    fontsize=6, color='purple',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8,
                              ec='purple', lw=0.6))

        Mean_Dt = np.mean(Pp_Times)
        Std_Dt  = np.std(Pp_Times)
        Mean_F  = 1.0 / Mean_Dt if Mean_Dt > 0 else 0.0
        Lines = [f'{Label_Prefix}Peak-to-peak:',
                 f'  dt = {Mean_Dt*1000:.1f} +/- {Std_Dt*1000:.1f} ms',
                 f'  f  = {Mean_F:.3f} Hz']
        if Trough_Vals:
            Lines.append(f'  dR = {np.mean(Peak_Vals) - np.mean(Trough_Vals):.0f} {Unit_Str}')
        Ax.plot([], [], color='purple', lw=0, marker='', label='\n'.join(Lines))

    return True


# ============================================================
# MAIN CLASS
# ============================================================

class Red_Pitaya_Lock_In_Logger:
    Allowed_Decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, Output_Dir='test_data', Input_Mode_Param='MANUAL',
                 Manual_Gain=1.0, Manual_Offset=0.0):
        self.Rp = Pyrpl(config='lockin_config10', hostname='rp-f073ce.local')
        self.Output_Dir = Output_Dir
        self.Rp_Modules = self.Rp.rp
        self.Lockin  = self.Rp_Modules.iq2
        self.Ref_Sig = self.Rp_Modules.asg0
        self.Scope   = self.Rp_Modules.scope

        self.Lockin_X = []
        self.Lockin_Y = []
        self.Capture_Times = []

        self.Input_Gain_Factor  = Manual_Gain
        self.Input_Dc_Offset    = Manual_Offset
        self.Input_Mode_Setting = Input_Mode_Param.upper()
        self.Input_Mode         = "Unknown"

        if self.Input_Mode_Setting == 'MANUAL':
            self.Input_Mode = f"MANUAL ({Manual_Gain}x gain, {Manual_Offset}V offset)"
        elif self.Input_Mode_Setting == 'LV':
            self.Input_Gain_Factor = 1.0
            self.Input_Dc_Offset   = 0.0
            self.Input_Mode = "LV (+-1V)"
        elif self.Input_Mode_Setting == 'HV':
            self.Input_Gain_Factor = 20.0
            self.Input_Dc_Offset   = 0.0
            self.Input_Mode = "HV (+-20V, 20:1 divider)"
        print(f"Input mode: {self.Input_Mode}")

        self.Scope.input1 = 'iq2'
        self.Scope.input2 = 'iq2_2'
        self.Scope.decimation = Decimation

        if self.Scope.decimation not in self.Allowed_Decimations:
            raise ValueError(f'Invalid decimation. Must be one of {self.Allowed_Decimations}')

        self.Scope._start_acquisition_rolling_mode()
        self.Scope.average = True
        self.Nominal_Sample_Rate = 125e6 / self.Scope.decimation

    def Setup_Lockin(self, Params):
        self.Ref_Freq   = Params['ref_freq']
        self.Ref_Period = 1 / self.Ref_Freq
        Ref_Amp   = Params['ref_amp']
        Filter_Bw = Params.get('filter_bandwidth', 10)

        self.Ref_Sig.output_direct = 'off'
        self.Lockin.setup(
            frequency=self.Ref_Freq,
            bandwidth=Filter_Bw,
            gain=0.0,
            phase=Params.get('phase', 0),
            acbandwidth=0,
            amplitude=Ref_Amp,
            input='in1',
            output_direct=Params['output_ref'],
            output_signal='quadrature',
            quadrature_factor=1.0)

        Actual_Freq = self.Lockin.frequency
        Actual_Amp  = self.Lockin.amplitude
        Tc_Ms = 1e3 / (2 * np.pi * Filter_Bw)
        print(f"Lock-in frequency: {self.Ref_Freq} Hz (actual: {Actual_Freq:.2f} Hz)")
        print(f"Lock-in bandwidth: {Filter_Bw} Hz  (time constant: {Tc_Ms:.2f} ms)")
        print(f"Reference: {Ref_Amp}V on {Params['output_ref']} (actual: {Actual_Amp:.3f}V)")
        if abs(Actual_Freq - self.Ref_Freq) > 0.1:
            print(f"WARNING: Requested {self.Ref_Freq} Hz but got {Actual_Freq:.2f} Hz!")
        if Filter_Bw < 50:
            print(f"NOTE: Filter BW is {Filter_Bw} Hz (tau = {Tc_Ms:.1f} ms) -- peaks will be "
                  f"rounded. Consider raising Filter_Bandwidth to 100+ Hz for sharper peaks.")

    def Capture_Lockin(self):
        self.Scope.single()
        Ch1 = np.array(self.Scope._data_ch1_current)
        Ch2 = np.array(self.Scope._data_ch2_current)
        self.Lockin_X.append(Ch1)
        self.Lockin_Y.append(Ch2)
        self.Capture_Times.append(time.time())
        return Ch1, Ch2

    def Capture_Lockin_Continuous(self):
        Ch1 = np.array(self.Scope._data_ch1_current)
        Ch2 = np.array(self.Scope._data_ch2_current)
        self.Lockin_X.append(Ch1)
        self.Lockin_Y.append(Ch2)
        self.Capture_Times.append(time.time())
        return Ch1, Ch2

    def Run(self, Params):
        self.Setup_Lockin(Params)

        print()
        print("=" * 60)
        print("AC signal is running. Press ENTER to start measurement.")
        print("=" * 60)
        input()

        print("Allowing lock-in to settle...")
        time.sleep(0.5)

        Acq_Start = time.time()
        Capture_Count = 0
        print(f"Started: {datetime.fromtimestamp(Acq_Start).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        Acq_Mode = Params.get('acquisition_mode', 'SINGLE_SHOT')
        if Acq_Mode == 'CONTINUOUS':
            self.Scope.continuous()
            time.sleep(0.1)
            Loop_Start = time.time()
            while (time.time() - Loop_Start) < Params['timeout']:
                self.Capture_Lockin_Continuous()
                Capture_Count += 1
                time.sleep(0.001)
        else:
            Loop_Start = time.time()
            while (time.time() - Loop_Start) < Params['timeout']:
                self.Capture_Lockin()
                Capture_Count += 1

        Actual_Duration = time.time() - Acq_Start
        print(f"Captured {Capture_Count} buffers")

        All_X = np.concatenate(self.Lockin_X) * self.Input_Gain_Factor
        All_Y = np.concatenate(self.Lockin_Y) * self.Input_Gain_Factor

        Avg_Win = Params.get('averaging_window', 1)
        if Avg_Win > 1:
            K = np.ones(Avg_Win) / Avg_Win
            All_X = np.convolve(All_X, K, mode='valid')
            All_Y = np.convolve(All_Y, K, mode='valid')
            print(f"Applied {Avg_Win}-sample moving average")

        R     = np.sqrt(All_X ** 2 + All_Y ** 2)
        Theta = np.degrees(np.arctan2(All_Y, All_X))

        N_Samples    = len(All_X)
        Sample_Index = np.arange(N_Samples)
        T = Sample_Index / (N_Samples / Actual_Duration)

        Filter_Bw = Params.get('filter_bandwidth', 10)
        Tc_Ms = 1e3 / (2 * np.pi * Filter_Bw)

        Ds_On  = Params.get('plot_downsample_enabled', True)
        Ds_Max = Params.get('plot_max_points', 50_000)

        Ds_Arrays, Ds_Step, N_Plot = Downsample(
            [T, R, Theta, All_X, All_Y],
            N_Samples, Ds_Max, Enabled=Ds_On)
        T_P, R_P, Theta_P, X_P, Y_P = Ds_Arrays

        if Ds_Step > 1:
            print(f"\nDownsampled: {N_Samples:,} -> {N_Plot:,} pts "
                  f"(step={Ds_Step}) for plotting. CSV = full res.")
        else:
            print(f"\nNo downsampling needed ({N_Samples:,} pts)")

        Ds_Sample_Rate = N_Plot / Actual_Duration
        print(f"Downsampled sample rate: {Ds_Sample_Rate:.1f} Hz")

        print("\nCapturing scope snapshot...")
        self.Scope.input1 = 'out1'
        self.Scope.input2 = 'in1'
        time.sleep(0.05)
        self.Scope.single()
        Out1_Raw = np.array(self.Scope._data_ch1_current)
        In1_Raw  = (np.array(self.Scope._data_ch2_current) - self.Input_Dc_Offset) * self.Input_Gain_Factor
        T_Raw    = np.linspace(0, len(Out1_Raw) / self.Nominal_Sample_Rate, len(Out1_Raw))
        self.Scope.input1 = 'iq2'
        self.Scope.input2 = 'iq2_2'

        print("Computing FFT...")
        N_Pts = len(X_P)
        Win   = np.hanning(N_Pts)
        Iq_Fft = np.fft.fftshift(np.fft.fft((X_P + 1j * Y_P) * Win))
        Freqs = np.fft.fftshift(np.fft.fftfreq(N_Pts, T_P[1] - T_P[0]))
        Psd   = (np.abs(Iq_Fft) ** 2) / (N_Pts * np.sum(Win ** 2))

        Dc_Power = Psd[np.argmin(np.abs(Freqs))]
        Psd_No_Dc = Psd.copy()
        Psd_No_Dc[np.abs(Freqs) <= 50] = 0
        Idx_Spur   = np.argmax(Psd_No_Dc)
        Freq_Spur  = Freqs[Idx_Spur]
        Power_Spur = Psd_No_Dc[Idx_Spur]

        Sig_Idx    = np.where(Psd > 0.01 * Dc_Power)[0]
        Sort_Order = np.argsort(Psd[Sig_Idx])[::-1]
        Top_Freqs  = Freqs[Sig_Idx][Sort_Order[:10]]
        Top_Powers = Psd[Sig_Idx][Sort_Order[:10]]

        print(f"\nFFT:  DC={Dc_Power:.2e}   Spurious={Freq_Spur:+.1f} Hz "
              f"({Power_Spur/Dc_Power*100:.1f}% of DC)")
        for I, (F, P) in enumerate(zip(Top_Freqs, Top_Powers)):
            Flag = " <- DC" if abs(F) < 5 else (" SPURIOUS!" if abs(F) > 50 and P > 0.01*Dc_Power else "")
            print(f"  {I+1:2d}. {F:+8.2f} Hz  {P:.2e}{Flag}")
        if Power_Spur > 0.1 * Dc_Power:
            print(f"WARNING: Spurious at {Freq_Spur:+.2f} Hz is {Power_Spur/Dc_Power*100:.1f}% of DC!")
        else:
            print("Spectrum dominated by DC (properly locked)")

        Spb = N_Samples / Capture_Count
        Gap = (Actual_Duration / Capture_Count) - (Spb / self.Nominal_Sample_Rate)

        print("\n" + "=" * 60)
        print("LOCK-IN RESULTS")
        print("=" * 60)
        print(f"Mode:         {self.Input_Mode}")
        print(f"Duration:     {Actual_Duration:.3f}s   Samples: {N_Samples:,}")
        print(f"Logging rate: {N_Samples/Actual_Duration:.2f} Hz")
        print(f"Filter BW:    {Filter_Bw} Hz  (time constant: {Tc_Ms:.2f} ms)")
        print(f"Decimation:   {Decimation}  (scope sample rate: {self.Nominal_Sample_Rate/1e3:.1f} kHz)")
        print(f"Buffers:      {Capture_Count}   Samples/buf: {Spb:.0f}   Gap/buf: {Gap*1000:.1f} ms")
        print(f"\nMean R:     {np.mean(R):.6f} +/- {np.std(R):.6f} V")
        print(f"Mean X:     {np.mean(All_X):.6f} +/- {np.std(All_X):.6f} V")
        print(f"Mean Y:     {np.mean(All_Y):.6f} +/- {np.std(All_Y):.6f} V")
        print(f"Mean Theta: {np.mean(Theta):.3f} +/- {np.std(Theta):.3f} deg")
        print("=" * 60)

        Zoom_Frac = 0.2
        N_Z = max(2, int(N_Plot * Zoom_Frac))
        Zs  = N_Plot // 2 - N_Z // 2
        Ze  = Zs + N_Z

        T_Z   = T_P[Zs:Ze];   R_Z   = R_P[Zs:Ze];   Th_Z  = Theta_P[Zs:Ze]
        X_Z   = X_P[Zs:Ze];   Y_Z   = Y_P[Zs:Ze]

        Ds_Note   = f' [1:{Ds_Step} for plot]' if Ds_Step > 1 else ''
        Z_Raw_Lbl = f'Zoomed {T_Z[0]:.1f}-{T_Z[-1]:.1f}s ({Zoom_Frac*100:.0f}%) RAW{Ds_Note}'

        # Layout: 4 rows x 3 cols
        #   Row 1: Reference signal | Input signal | FFT
        #   Row 2: X (full)        | Y (full)      | IQ plot (full)
        #   Row 3: R (full)        | Theta (full)  | Phase vs Magnitude
        #   Row 4: R zoomed        | Theta zoomed  | IQ zoomed
        print("\nRendering plots...")
        Fig = plt.figure(figsize=(18, 18))

        def _Margin(Arr):
            return 5 * max(np.max(Arr) - np.min(Arr), 1e-12)

        # Row 1
        Ax1 = plt.subplot(4, 3, 1)
        N_Raw = min(int(5 * self.Nominal_Sample_Rate / self.Ref_Freq), len(Out1_Raw))
        Ax1.plot(T_Raw[:N_Raw] * 1000, Out1_Raw[:N_Raw], 'b-', lw=1)
        Ax1.set(xlabel='Time (ms)', ylabel='OUT1 (V)', title=f'Reference @ {self.Ref_Freq} Hz')
        Ax1.grid(True)

        Ax2 = plt.subplot(4, 3, 2)
        Ax2.plot(T_Raw[:N_Raw] * 1000, In1_Raw[:N_Raw], 'r-', lw=1)
        Ax2.set(xlabel='Time (ms)', ylabel='IN1 (V, corrected)', title=f'Input -- {self.Input_Mode}')
        Ax2.grid(True)

        Ax3 = plt.subplot(4, 3, 3)
        Ax3.semilogy(Freqs, Psd, label='Lock-in PSD')
        if Power_Spur > 0.01 * Dc_Power:
            Ax3.axvline(Freq_Spur, color='orange', ls='--', alpha=0.7,
                        label=f'Spurious ({Freq_Spur:.0f} Hz)')
        Ax3.set(xlabel='Frequency (Hz)', ylabel='Power', title='FFT of Demodulated Signal')
        Ax3.legend(); Ax3.grid(True)

        # Row 2
        Ax4 = plt.subplot(4, 3, 4)
        Ax4.plot(T_P, X_P, 'b-', lw=0.5)
        Ax4.axhline(np.mean(All_X), color='r', ls='--', label=f'Mean: {np.mean(All_X):.6f}V')
        Ax4.set(xlabel='Time (s)', ylabel='X (V)', title=f'In-phase (X){Ds_Note}')
        Ax4.legend(); Ax4.grid(True); Ax4.set_xlim(T_P[0], T_P[-1])
        M = _Margin(X_P); Ax4.set_ylim(np.min(X_P)-M, np.max(X_P)+M)

        Ax5 = plt.subplot(4, 3, 5)
        Ax5.plot(T_P, Y_P, 'r-', lw=0.5)
        Ax5.axhline(np.mean(All_Y), color='b', ls='--', label=f'Mean: {np.mean(All_Y):.6f}V')
        Ax5.set(xlabel='Time (s)', ylabel='Y (V)', title=f'Quadrature (Y){Ds_Note}')
        Ax5.legend(); Ax5.grid(True); Ax5.set_xlim(T_P[0], T_P[-1])
        M = _Margin(Y_P); Ax5.set_ylim(np.min(Y_P)-M, np.max(Y_P)+M)

        Ax6 = plt.subplot(4, 3, 6)
        Ax6.plot(X_P, Y_P, 'g.', ms=1, alpha=0.5)
        Ax6.plot(np.mean(All_X), np.mean(All_Y), 'r+', ms=15, mew=2, label='Mean')
        Ax6.set(xlabel='X (V)', ylabel='Y (V)', title=f'IQ Plot{Ds_Note}')
        Ax6.legend(); Ax6.grid(True); Ax6.axis('equal')

        # Row 3
        Ax7 = plt.subplot(4, 3, 7)
        R_P_Ua = R_P * 1e6
        Ax7.plot(T_P, R_P_Ua, 'm-', lw=0.5)
        Ax7.axhline(np.mean(R)*1e6, color='b', ls='--', label=f'Mean: {np.mean(R)*1e6:.4f} uA')
        Ax7.set(xlabel='Time (s)', ylabel='R (uA)', title=f'Magnitude (R) -- Full{Ds_Note}')
        Ax7.grid(True); Ax7.set_xlim(T_P[0], T_P[-1])
        M = _Margin(R_P_Ua); Ax7.set_ylim(np.min(R_P_Ua)-M, np.max(R_P_Ua)+M)
        Ax7.axvspan(T_Z[0], T_Z[-1], color='yellow', alpha=0.25, label=f'Zoom ({Zoom_Frac*100:.0f}%)')
        Ax7.legend(fontsize=7)

        Ax8 = plt.subplot(4, 3, 8)
        Ax8.plot(T_P, Theta_P, 'c-', lw=0.5)
        Ax8.axhline(np.mean(Theta), color='r', ls='--', label=f'Mean: {np.mean(Theta):.3f} deg')
        Ax8.set(xlabel='Time (s)', ylabel='Theta (deg)', title=f'Phase (Theta) -- Full{Ds_Note}')
        Ax8.grid(True); Ax8.set_xlim(T_P[0], T_P[-1])
        M = _Margin(Theta_P); Ax8.set_ylim(np.min(Theta_P)-M, np.max(Theta_P)+M)
        Ax8.axvspan(T_Z[0], T_Z[-1], color='yellow', alpha=0.25, label=f'Zoom ({Zoom_Frac*100:.0f}%)')
        Ax8.legend(fontsize=7)

        Ax9 = plt.subplot(4, 3, 9)
        Ax9.plot(Theta_P, R_P, 'g.', ms=1, alpha=0.5)
        Ax9.plot(np.mean(Theta), np.mean(R), 'r+', ms=15, mew=2, label='Mean')
        Ax9.set(xlabel='Theta (deg)', ylabel='R (V)', title='Phase vs Magnitude')
        Ax9.legend(); Ax9.grid(True)

        # Row 4: Zoomed
        def _Zoom_Plot(Ax, T_Z, Sig, Scale, Ylabel, Title, Color, Unit_Str, Lp):
            S = Sig * Scale
            Mn, Sd = np.mean(S), np.std(S)
            Ax.plot(T_Z, S, color=Color, lw=1.0)
            Ax.axhline(Mn, color='b', ls='--', label=f'Mean: {Mn:.2f}\nStd: {Sd:.2f}')
            Ax.fill_between(T_Z, Mn-Sd, Mn+Sd, alpha=0.15, color='blue', label='+/-1 sigma')
            Ax.set(xlabel='Time (s)', ylabel=Ylabel, title=Title)
            Ax.grid(True); Ax.set_xlim(T_Z[0], T_Z[-1])
            Detect_Peaks_And_Annotate(Ax, T_Z, Sig, Unit_Str=Unit_Str,
                                      Color_Peak='darkred', Color_Trough='navy',
                                      Min_Prominence_Frac=0.15, Scale=Scale, Label_Prefix=Lp)
            Ax.legend(fontsize=7, loc='upper right')

        _Zoom_Plot(plt.subplot(4,3,10), T_Z, R_Z,  1e6, 'R (uA)',
                   f'R -- Zoomed\n{Z_Raw_Lbl}',      'm', ' uA',  'R: ')
        _Zoom_Plot(plt.subplot(4,3,11), T_Z, Th_Z, 1.0, 'Theta (deg)',
                   f'Theta -- Zoomed\n{Z_Raw_Lbl}',  'c', ' deg', 'Th: ')

        Ax12 = plt.subplot(4, 3, 12)
        Ax12.plot(X_Z, Y_Z, 'g.', ms=2, alpha=0.6)
        Ax12.plot(np.mean(X_Z), np.mean(Y_Z), 'r+', ms=15, mew=2, label='Mean')
        Ax12.set(xlabel='X (V)', ylabel='Y (V)', title=f'IQ -- Zoomed\n{Z_Raw_Lbl}')
        Ax12.legend(fontsize=7); Ax12.grid(True); Ax12.axis('equal')

        plt.tight_layout()

        if Params['save_file']:
            os.makedirs(self.Output_Dir, exist_ok=True)
            Ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            Img_Path = os.path.join(self.Output_Dir, f'lockin_results_{Ts}.png')
            plt.savefig(Img_Path, dpi=150)
            print(f"\nPlot:  {Img_Path}")

            Csv_Path = os.path.join(self.Output_Dir, f'lockin_results_{Ts}.csv')
            Data = np.column_stack((Sample_Index, T, R, Theta, All_X, All_Y))
            with open(Csv_Path, 'w', newline='', encoding='ascii') as F:
                F.write("# Red Pitaya Lock-In Amplifier\n")
                F.write(f"# Mode: {self.Input_Mode}\n")
                F.write(f"# Gain: {self.Input_Gain_Factor:.6f}  DC offset: {self.Input_Dc_Offset:.6f} V\n")
                F.write(f"# Ref: {self.Ref_Freq} Hz @ {Params['ref_amp']} V\n")
                F.write(f"# Filter BW: {Filter_Bw} Hz  (time constant: {Tc_Ms:.2f} ms)\n")
                F.write(f"# Decimation: {Decimation}  (scope rate: {self.Nominal_Sample_Rate/1e3:.1f} kHz)\n")
                F.write(f"# Duration: {Actual_Duration:.3f} s  Samples: {N_Samples}\n")
                F.write(f"# Sample rate: {N_Samples/Actual_Duration:.2f} Hz\n")
                F.write(f"# Acquisition: {Acq_Mode}\n")
                F.write(f"# Plot downsample step: {Ds_Step}x  (CSV is full resolution)\n")
                F.write("Index,Time(s),R(V),Theta(deg),X(V),Y(V)\n")
                np.savetxt(F, Data, delimiter=",", fmt='%.10f')
            print(f"Data:  {Csv_Path}")
        else:
            plt.show()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    Rp = Red_Pitaya_Lock_In_Logger(
        Output_Dir=Output_Directory,
        Input_Mode_Param=Input_Mode,
        Manual_Gain=Manual_Gain_Factor,
        Manual_Offset=Manual_Dc_Offset
    )

    Run_Params = {
        'ref_freq':                Ref_Frequency,
        'ref_amp':                 Ref_Amplitude,
        'output_ref':              Output_Channel,
        'phase':                   Phase_Offset,
        'timeout':                 Measurement_Time,
        'filter_bandwidth':        Filter_Bandwidth,
        'averaging_window':        Averaging_Window,
        'output_dir':              Output_Directory,
        'save_file':               Save_Data,
        'calibration_time':        Calibration_Time,
        'acquisition_mode':        Acquisition_Mode,
        'plot_downsample_enabled': Plot_Downsample_Enabled,
        'plot_max_points':         Plot_Max_Points,
    }

    print("=" * 60)
    print("RED PITAYA LOCK-IN DATA LOGGER")
    print("=" * 60)
    print(f"Reference:    {Ref_Frequency} Hz @ {Ref_Amplitude}V")
    print(f"Filter BW:    {Filter_Bandwidth} Hz")
    print(f"Decimation:   {Decimation}  (scope rate: {125e6/Decimation/1e3:.1f} kHz, "
          f"buf covers {16384/(125e6/Decimation)*1000:.1f} ms)")
    print(f"Meas. time:   {Measurement_Time}s")
    print(f"Input mode:   {Input_Mode}")
    if Input_Mode.upper() == 'MANUAL':
        print(f"  Gain:   {Manual_Gain_Factor}x")
        print(f"  Offset: {Manual_Dc_Offset}V")
    print(f"Acq. mode:    {Acquisition_Mode}")
    print(f"Plot DS:      {'ON -- max ' + str(Plot_Max_Points) + ' pts/signal' if Plot_Downsample_Enabled else 'OFF'}")
    print("=" * 60)

    Rp.Run(Run_Params)

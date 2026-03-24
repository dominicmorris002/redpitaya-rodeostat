"""
Red Pitaya DC Voltage Data Logger
Single plot: Voltage vs Time
Connect your DC signal to IN1.

To do AC CV use the startboth file

Have a Great Day ;)

Dominic Morris
"""

from datetime import datetime
import time
import numpy as np
from matplotlib import pyplot as plt
import os
from pyrpl import Pyrpl
from scipy.signal import butter, filtfilt

# ============================================================
# MEASUREMENT PARAMETERS
# ============================================================
Measurement_Time = 12.0       # seconds
Decimation = 1024
Averaging_Window = 1

# INPUT MODE: 'AUTO', 'LV', 'HV', or 'MANUAL'
Input_Mode = 'MANUAL'
Manual_Gain_Factor = -1
Manual_Dc_Offset = 0

Auto_Calibrate = True         # Only used if Input_Mode = 'AUTO'
Calibration_Time = 2.0        # seconds

Save_Data = True
Output_Directory = 'test_data'

# ACQUISITION MODE: 'SINGLE_SHOT' or 'CONTINUOUS'
Acquisition_Mode = 'CONTINUOUS'

# ============================================================
# AC BIAS REMOVAL
# Low-pass filter to recover DC ramp from signal containing AC.
# Cutoff must be well below AC frequency and well above DC bandwidth.
# ============================================================
Ac_Removal_Enabled = True
Ac_Removal_Cutoff  = 6        # Hz
Ac_Removal_Order   = 4        # Butterworth order

# ============================================================
# PLOT DOWNSAMPLING
# CSV always saves full resolution data.
# ============================================================
Plot_Downsample_Enabled = True
Plot_Max_Points = 50_000

Start_Time_File = "start_time.txt"

# Synchronization with other processes
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


def Remove_Ac_Bias(Signal, Fs, Cutoff, Order=4):
    """
    Remove AC bias using a zero-phase Butterworth low-pass filter.
    Cutoff must be below AC frequency and above DC ramp bandwidth.
    """
    Nyq = Fs / 2.0
    if Cutoff >= Nyq:
        raise ValueError(
            f"Ac_Removal_Cutoff ({Cutoff} Hz) must be less than Nyquist "
            f"({Nyq:.1f} Hz). Lower the cutoff or raise Decimation."
        )
    if Cutoff <= 0:
        raise ValueError("Ac_Removal_Cutoff must be greater than 0 Hz.")

    B, A = butter(Order, Cutoff / Nyq, btype='low')
    return filtfilt(B, A, Signal)


class Red_Pitaya_Dc_Logger:
    """DC voltage logger using lock-in style acquisition"""

    Allowed_Decimations = [1, 8, 64, 1024, 8192, 65536]

    def __init__(self, Output_Dir='test_data', Input_Mode_Param='AUTO',
                 Manual_Gain=1.0, Manual_Offset=0.0):

        self.Rp = Pyrpl(config='dc_config5', hostname='rp-f0909c.local')
        self.Rp_Modules = self.Rp.rp
        self.Scope = self.Rp_Modules.scope
        self.Asg = self.Rp_Modules.asg0
        self.Output_Dir = Output_Dir

        self.Buffers = []
        self.Capture_Times = []

        self.Input_Gain_Factor = Manual_Gain
        self.Input_Dc_Offset = Manual_Offset
        self.Input_Mode_Setting = Input_Mode_Param.upper()
        self.Input_Mode = "Unknown"

        if self.Input_Mode_Setting == 'MANUAL':
            self.Input_Gain_Factor = Manual_Gain
            self.Input_Dc_Offset = Manual_Offset
            self.Input_Mode = f"MANUAL ({Manual_Gain}x gain, {Manual_Offset}V offset)"
            print(f"Input mode: {self.Input_Mode}")
        elif self.Input_Mode_Setting == 'LV':
            self.Input_Gain_Factor = 1.0
            self.Input_Dc_Offset = 0.0
            self.Input_Mode = "LV (±1V)"
            print(f"Input mode: {self.Input_Mode}")
        elif self.Input_Mode_Setting == 'HV':
            self.Input_Gain_Factor = 20.0
            self.Input_Dc_Offset = 0.0
            self.Input_Mode = "HV (±20V, 20:1 divider)"
            print(f"Input mode: {self.Input_Mode}")
        elif self.Input_Mode_Setting == 'AUTO':
            self.Input_Mode = "AUTO (will calibrate)"
            print("Input mode: AUTO - will auto-detect")

        self.Scope.input1 = 'in1'
        self.Scope.input2 = 'in1'
        self.Scope.decimation = Decimation

        if self.Scope.decimation not in self.Allowed_Decimations:
            raise ValueError(f"Invalid decimation {Decimation}")

        self.Scope._start_acquisition_rolling_mode()
        self.Scope.average = True
        self.Nominal_Sample_Rate = 125e6 / self.Scope.decimation

        print(f"DC Logger initialized")
        print(f"Nominal sample rate: {self.Nominal_Sample_Rate:.2f} Hz")

    def Calibrate_Input_Gain(self, Cal_Freq=100, Cal_Amp=1.0, Cal_Time=2.0, Force=False):
        if not Force and self.Input_Mode_Setting != 'AUTO':
            print(f"Skipping calibration - using {self.Input_Mode}")
            return self.Input_Gain_Factor

        print("\n" + "=" * 60)
        print("CALIBRATING INPUT SCALING AND DC OFFSET...")
        print("=" * 60)

        print("Step 1: Measuring DC offset on IN1 (no signal)...")
        self.Asg.output_direct = 'off'
        self.Scope.input1 = 'in1'
        self.Scope.input2 = 'in1'
        time.sleep(0.3)

        Offset_Samples = []
        for _ in range(10):
            self.Scope.single()
            Offset_Samples.append(np.mean(self.Scope._data_ch1_current))

        self.Input_Dc_Offset = np.mean(Offset_Samples)
        print(f"  Measured DC offset: {self.Input_Dc_Offset:.6f}V")

        print(f"\nStep 2: Measuring gain with {Cal_Amp}V @ {Cal_Freq} Hz...")
        self.Scope.input1 = 'out1'
        self.Scope.input2 = 'in1'

        self.Asg.setup(
            frequency=Cal_Freq,
            amplitude=Cal_Amp,
            offset=0,
            waveform='sin',
            trigger_source='immediately'
        )
        self.Asg.output_direct = 'out1'
        time.sleep(0.5)

        Cal_Out1 = []
        Cal_In1 = []
        Cal_Start = time.time()

        while (time.time() - Cal_Start) < Cal_Time:
            self.Scope.single()
            Cal_Out1.append(np.array(self.Scope._data_ch1_current))
            Cal_In1.append(np.array(self.Scope._data_ch2_current))

        All_Out1 = np.concatenate(Cal_Out1)
        All_In1 = np.concatenate(Cal_In1)
        All_In1_Corrected = All_In1 - self.Input_Dc_Offset

        Out1_Peak = (np.max(All_Out1) - np.min(All_Out1)) / 2
        In1_Peak = (np.max(All_In1_Corrected) - np.min(All_In1_Corrected)) / 2
        Out1_Rms = np.sqrt(np.mean(All_Out1 ** 2))
        In1_Rms = np.sqrt(np.mean(All_In1_Corrected ** 2))

        self.Input_Gain_Factor = Out1_Peak / In1_Peak
        Gain_Rms = Out1_Rms / In1_Rms

        if self.Input_Gain_Factor < 1.05:
            self.Input_Mode = "LV (±1V)"
        elif self.Input_Gain_Factor < 2.0:
            self.Input_Mode = f"LV with loading ({self.Input_Gain_Factor:.2f}x)"
        elif self.Input_Gain_Factor < 15:
            self.Input_Mode = f"Custom/Unknown mode ({self.Input_Gain_Factor:.2f}x)"
        else:
            self.Input_Mode = f"HV (±20V, {self.Input_Gain_Factor:.1f}:1 divider)"

        print(f"\n  OUT1 peak: {Out1_Peak:.4f}V, RMS: {Out1_Rms:.4f}V")
        print(f"  IN1 peak (after offset correction): {In1_Peak:.4f}V, RMS: {In1_Rms:.4f}V")
        print(f"  Gain (peak-based): {self.Input_Gain_Factor:.4f}x")
        print(f"  Gain (RMS-based): {Gain_Rms:.4f}x")
        print(f"  DC offset: {self.Input_Dc_Offset:.6f}V")
        print(f"  Detected mode: {self.Input_Mode}")
        print("=" * 60 + "\n")

        self.Asg.output_direct = 'off'
        self.Scope.input1 = 'in1'
        self.Scope.input2 = 'in1'

        return self.Input_Gain_Factor

    def Capture_Buffer(self):
        """Single-shot: trigger a capture and wait for it"""
        self.Scope.single()
        Ch1 = np.array(self.Scope._data_ch1_current)
        self.Buffers.append(Ch1)
        self.Capture_Times.append(time.time())

    def Capture_Buffer_Continuous(self):
        """Continuous: read whatever is currently in the buffer"""
        Ch1 = np.array(self.Scope._data_ch1_current)
        self.Buffers.append(Ch1)
        self.Capture_Times.append(time.time())

    def Run(self, Params):
        if Params.get('auto_calibrate', False):
            self.Calibrate_Input_Gain(
                Cal_Freq=100,
                Cal_Amp=1.0,
                Cal_Time=Params.get('calibration_time', 2.0)
            )

        input("\nPress Enter to start measurement...")
        print("")

        print("\nAllowing scope to settle...")
        time.sleep(0.3)

        Acquisition_Start = time.time()
        print(f"Started: {datetime.fromtimestamp(Acquisition_Start).strftime('%Y-%m-%d %H:%M:%S.%f')}")

        Acq_Mode = Params.get('acquisition_mode', 'SINGLE_SHOT')

        if Acq_Mode == 'CONTINUOUS':
            self.Scope.continuous()
            time.sleep(0.1)
            Loop_Start = time.time()
            while (time.time() - Loop_Start) < Params['timeout']:
                self.Capture_Buffer_Continuous()
                time.sleep(0.001)
        else:
            Loop_Start = time.time()
            while (time.time() - Loop_Start) < Params['timeout']:
                self.Capture_Buffer()

        Actual_Duration = time.time() - Acquisition_Start
        Capture_Count = len(self.Buffers)
        print(f"Captured {Capture_Count} buffers")

        Raw = np.concatenate(self.Buffers)
        Corrected = (Raw - self.Input_Dc_Offset) * self.Input_Gain_Factor

        W = Params.get('averaging_window', 1)
        if W > 1:
            Corrected = np.convolve(Corrected, np.ones(W) / W, mode='valid')
            print(f"Applied {W}-sample moving average")

        Ac_Enabled = Params.get('ac_removal_enabled', False)
        Ac_Cutoff  = Params.get('ac_removal_cutoff',  50)
        Ac_Order   = Params.get('ac_removal_order',   4)
        Ac_Label   = ''

        if Ac_Enabled:
            Fs = self.Nominal_Sample_Rate
            print(f"\nApplying AC bias removal...")
            print(f"  Sample rate:     {Fs:.1f} Hz")
            print(f"  Low-pass cutoff: {Ac_Cutoff} Hz  (order {Ac_Order} Butterworth)")
            print(f"  Attenuation @ 500 Hz: ~{20*Ac_Order*np.log10(500/Ac_Cutoff):.0f} dB")
            try:
                Corrected_Filtered = Remove_Ac_Bias(Corrected, Fs, Ac_Cutoff, Ac_Order)
                Ac_Removed = Corrected - Corrected_Filtered
                Ac_Rms  = np.sqrt(np.mean(Ac_Removed ** 2))
                Ac_Peak = (np.max(Ac_Removed) - np.min(Ac_Removed)) / 2
                print(f"  AC component RMS:  {Ac_Rms*1000:.4f} mV  peak: {Ac_Peak*1000:.4f} mV")
                Corrected = Corrected_Filtered
                Ac_Label = f' [AC<{Ac_Cutoff}Hz LP filtered]'
                print(f"  AC bias removal applied successfully.")
            except ValueError as E:
                print(f"  WARNING: AC bias removal skipped -- {E}")
                Ac_Enabled = False

        N_Samples = len(Corrected)
        Sample_Index = np.arange(N_Samples)
        T = Sample_Index / (N_Samples / Actual_Duration)
        Effective_Sample_Rate = N_Samples / Actual_Duration

        Samples_Per_Buffer = N_Samples / Capture_Count
        Data_Time_Per_Buffer = Samples_Per_Buffer / self.Nominal_Sample_Rate
        Buffer_Spacing = Actual_Duration / Capture_Count
        Dead_Time = Buffer_Spacing - Data_Time_Per_Buffer

        print("\n" + "=" * 60)
        print("DC MEASUREMENT RESULTS")
        print("=" * 60)
        print(f"Mode:          {self.Input_Mode}")
        print(f"Acq. mode:     {Acq_Mode}")
        print(f"Gain:          {self.Input_Gain_Factor:.4f}x")
        print(f"DC offset:     {self.Input_Dc_Offset:.6f}V")
        print(f"AC removal:    {'ON -- cutoff ' + str(Ac_Cutoff) + ' Hz' if Ac_Enabled else 'OFF'}")
        print(f"Duration:      {Actual_Duration:.3f}s")
        print(f"Samples:       {N_Samples:,}")
        print(f"Sample rate:   {Effective_Sample_Rate:.2f} Hz")
        print(f"Mean voltage:  {np.mean(Corrected):.6f} V")
        print(f"Std dev:       {np.std(Corrected):.6f} V")
        print(f"\nBuffer stats:")
        print(f"  Buffers:     {Capture_Count}")
        print(f"  Samples/buf: {Samples_Per_Buffer:.0f}")
        print(f"  Gap/buf:     {Dead_Time * 1000:.1f} ms")
        print(f"  Dead time:   {Dead_Time * Capture_Count:.2f}s "
              f"({Dead_Time * Capture_Count / Actual_Duration * 100:.1f}%)")
        print("=" * 60)

        Ds_On  = Params.get('plot_downsample_enabled', True)
        Ds_Max = Params.get('plot_max_points', 50_000)

        Ds_Arrays, Ds_Step, N_Plot = Downsample(
            [T, Corrected], N_Samples, Ds_Max, Enabled=Ds_On)
        T_P, V_P = Ds_Arrays

        if Ds_Step > 1:
            print(f"\nDownsampled: {N_Samples:,} -> {N_Plot:,} pts "
                  f"(step={Ds_Step}) for plotting. CSV = full res.")
        else:
            print(f"\nNo downsampling needed ({N_Samples:,} pts)")

        Ds_Note = f' [1:{Ds_Step} for plot]' if Ds_Step > 1 else ''

        plt.figure(figsize=(14, 6))
        plt.plot(T_P, V_P, linewidth=0.8, label=f'DC Ramp{Ds_Note}{Ac_Label}')
        plt.axhline(np.mean(Corrected), color='r', linestyle='--',
                    label=f"Mean {np.mean(Corrected):.6f} V")
        plt.fill_between(T_P,
                         np.mean(Corrected) - np.std(Corrected),
                         np.mean(Corrected) + np.std(Corrected),
                         alpha=0.15, color='blue', label=f'+/-1σ ({np.std(Corrected):.6f} V)')
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(f"DC Voltage vs Time - {self.Input_Mode}\n"
                  f"Acq: {Acq_Mode}  |  {N_Samples:,} samples @ {Effective_Sample_Rate:.1f} Hz"
                  f"{Ac_Label}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if Params['save_file']:
            os.makedirs(self.Output_Dir, exist_ok=True)
            Ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            Png_Path = os.path.join(self.Output_Dir, f"dc_voltage_{Ts}.png")
            plt.savefig(Png_Path, dpi=150)
            print(f"\nPlot: {Png_Path}")

            Csv_Path = os.path.join(self.Output_Dir, f"dc_voltage_{Ts}.csv")
            with open(Csv_Path, 'w', newline='', encoding='ascii') as F:
                F.write("# Red Pitaya DC Logger\n")
                F.write(f"# Mode: {self.Input_Mode}\n")
                F.write(f"# Acquisition: {Acq_Mode}\n")
                F.write(f"# Gain: {self.Input_Gain_Factor:.6f}\n")
                F.write(f"# Offset: {self.Input_Dc_Offset:.6f}\n")
                F.write(f"# AC removal: {'cutoff=' + str(Ac_Cutoff) + ' Hz order=' + str(Ac_Order) if Ac_Enabled else 'disabled'}\n")
                F.write(f"# Duration: {Actual_Duration:.6f}\n")
                F.write(f"# Sample rate: {Effective_Sample_Rate:.3f} Hz\n")
                F.write(f"# Samples: {N_Samples}\n")
                F.write(f"# Plot downsample step: {Ds_Step}x  (CSV is full resolution)\n")
                F.write("Index,Time(s),Voltage(V)\n")
                np.savetxt(F, np.column_stack((Sample_Index, T, Corrected)),
                           delimiter=",", fmt="%.10f")
            print(f"Data: {Csv_Path}")
        else:
            plt.show()


if __name__ == "__main__":
    Rp = Red_Pitaya_Dc_Logger(
        Output_Dir=Output_Directory,
        Input_Mode_Param=Input_Mode,
        Manual_Gain=Manual_Gain_Factor,
        Manual_Offset=Manual_Dc_Offset
    )

    Run_Params = {
        'timeout':                 Measurement_Time,
        'averaging_window':        Averaging_Window,
        'save_file':               Save_Data,
        'auto_calibrate':          Auto_Calibrate,
        'calibration_time':        Calibration_Time,
        'acquisition_mode':        Acquisition_Mode,
        'plot_downsample_enabled': Plot_Downsample_Enabled,
        'plot_max_points':         Plot_Max_Points,
        'ac_removal_enabled':      Ac_Removal_Enabled,
        'ac_removal_cutoff':       Ac_Removal_Cutoff,
        'ac_removal_order':        Ac_Removal_Order,
    }

    print("=" * 60)
    print("RED PITAYA DC VOLTAGE DATA LOGGER")
    print("=" * 60)
    print(f"Measurement time: {Measurement_Time}s")
    print(f"Input mode:       {Input_Mode}")
    if Input_Mode.upper() == 'MANUAL':
        print(f"  Gain:   {Manual_Gain_Factor}x")
        print(f"  Offset: {Manual_Dc_Offset}V")
    print(f"Acq. mode:        {Acquisition_Mode}")
    print(f"AC bias removal:  {'ON -- cutoff ' + str(Ac_Removal_Cutoff) + ' Hz, order ' + str(Ac_Removal_Order) if Ac_Removal_Enabled else 'OFF'}")
    print(f"Plot DS:          {'ON -- max ' + str(Plot_Max_Points) + ' pts' if Plot_Downsample_Enabled else 'OFF'}")
    print("=" * 60)

    Rp.Run(Run_Params)

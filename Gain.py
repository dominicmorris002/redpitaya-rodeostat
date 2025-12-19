"""
Red Pitaya DC Gain Test: OUT1 → IN1

Purpose:
- Output a DC voltage on OUT1
- Measure it back on IN1 (HV mode)
- Apply TWO gain factors:
    1) Red Pitaya input gain (HV divider, user-corrected)
    2) Autolab monitor output gain (manual, per range)

This file is intentionally SIMPLE and does NOT use the lock-in.
It is meant for DC sanity checks and gain verification.

SETUP:
- Connect OUT1 directly to IN1  (Red Pitaya only)
- OR connect Autolab Eout / Iout → IN1 (through monitor cable)

Author: ChatGPT (for Dominic)
"""

import time
import numpy as np
from pyrpl import Pyrpl
from datetime import datetime

# ================= USER SETTINGS =================

# DC voltage to output on OUT1 (Volts)
DC_OUTPUT_VOLTAGE = 1.0

# Measurement duration (seconds)
MEASUREMENT_TIME = 5.0

# Red Pitaya input mode
# HV mode = ±20 V range, nominal 20:1 divider
REDPITAYA_INPUT_MODE = 'HV'   # 'LV' or 'HV'

# Optional manual correction if your HV divider is slightly off
REDPITAYA_GAIN_CORRECTION = 1.0

# Which Autolab monitor output are you using?
AUTOLAB_OUTPUT = 'Eout'  # 'Eout' or 'Iout'

# ================= MANUAL AUTOLAB GAINS =================
# Fill in these once you know or measure them

# Example: for Iout, gain = V_out / I_applied (V/A)
# For Eout, gain = V_out / E_applied (V/V)
AUTOLAB_MANUAL_GAINS = {
    'Eout': 1.0,   # Replace 1.0 with your measured V/V
    'Iout': 1e6    # Replace 1e6 with your measured V/A
}

# ==================================================

# ---------- Red Pitaya nominal gain ----------
if REDPITAYA_INPUT_MODE.upper() == 'LV':
    REDPITAYA_NOMINAL_GAIN = 1.0
elif REDPITAYA_INPUT_MODE.upper() == 'HV':
    REDPITAYA_NOMINAL_GAIN = 20.0
else:
    raise ValueError("REDPITAYA_INPUT_MODE must be 'LV' or 'HV'")

# Total gain correction
AUTOLAB_GAIN = AUTOLAB_MANUAL_GAINS.get(AUTOLAB_OUTPUT, 1.0)
TOTAL_GAIN = REDPITAYA_NOMINAL_GAIN * REDPITAYA_GAIN_CORRECTION * AUTOLAB_GAIN

print("=" * 60)
print("RED PITAYA DC GAIN TEST")
print("=" * 60)
print(f"DC Output Voltage (OUT1): {DC_OUTPUT_VOLTAGE:.3f} V")
print(f"Red Pitaya Mode: {REDPITAYA_INPUT_MODE}")
print(f"Red Pitaya Nominal Gain: {REDPITAYA_NOMINAL_GAIN:.2f}x")
print(f"Red Pitaya Manual Correction: {REDPITAYA_GAIN_CORRECTION:.6f}x")
print(f"Autolab Output: {AUTOLAB_OUTPUT}")
print(f"Autolab Manual Gain: {AUTOLAB_GAIN:.6e}x")
print(f"TOTAL GAIN CORRECTION: {TOTAL_GAIN:.6f}x")
print("=" * 60)

# ---------- Connect to Red Pitaya ----------
rp = Pyrpl(config='dc_gain_test', hostname='rp-f073ce.local')
rp_modules = rp.rp

asg = rp_modules.asg0
scope = rp_modules.scope

# ---------- Configure DC output ----------
asg.setup(
    waveform='dc',
    offset=DC_OUTPUT_VOLTAGE,
    output_direct='out1'
)

print("OUT1 set to DC mode")

# ---------- Configure scope ----------
scope.input1 = 'out1'
scope.input2 = 'in1'
scope.decimation = 1024
scope.average = 'true'

scope._start_acquisition_rolling_mode()

# ---------- Acquire data ----------
print("Measuring DC levels...")
start = time.time()
out1_vals = []
in1_vals = []

while time.time() - start < MEASUREMENT_TIME:
    scope.single()
    out1_vals.append(np.mean(scope._data_ch1_current))
    in1_vals.append(np.mean(scope._data_ch2_current))

out1_vals = np.array(out1_vals)
in1_vals = np.array(in1_vals)

# ---------- Apply gain correction ----------
in1_corrected = in1_vals * TOTAL_GAIN

# ---------- Results ----------
out1_mean = np.mean(out1_vals)
in1_raw_mean = np.mean(in1_vals)
in1_corr_mean = np.mean(in1_corrected)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"OUT1 mean (measured): {out1_mean:.6f} V")
print(f"IN1 mean (raw):       {in1_raw_mean:.6f} V")
print(f"IN1 mean (corrected): {in1_corr_mean:.6f} V")
print("-")
print(f"Expected IN1 (ideal): {DC_OUTPUT_VOLTAGE:.6f} V")
print(f"Error after correction: {in1_corr_mean - DC_OUTPUT_VOLTAGE:.6e} V")
print("=" * 60)

print("\nTimestamp:", datetime.now().isoformat())
print("Test complete.")


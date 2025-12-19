"""
Red Pitaya DC Gain Test: OUT1 → IN1
Purpose:
- Output a DC voltage on OUT1
- Measure it back on IN1 (HV mode)
- Apply total gain correction:
    Total Gain = HV Gain + PyRPL Scope Gain + Other Gain

This file is intentionally SIMPLE and does NOT use the lock-in.
It is meant for DC sanity checks and gain verification.

SETUP:
- Connect OUT1 directly to IN1  (Red Pitaya only)
- OR connect Autolab Eout / Iout → IN1 (through monitor cable)

Author: Fixed by Claude (for Dominic)
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
REDPITAYA_INPUT_MODE = 'LV'  # 'LV' or 'HV'

# ================= GAIN SETTINGS (ADDITIVE) =================

# Measured gains for LV and HV modes
# LV mode: Total gain needed = 1.082336592
# HV mode: Total gain needed = 40.42854255 (with standard 20:1 divider)

if REDPITAYA_INPUT_MODE.upper() == 'LV':
    # LV Mode gains
    HV_GAIN = 1.0  # No physical divider in LV mode
    PYRPL_SCOPE_GAIN = 0.092611976  # PyRPL correction for LV
    OTHER_GAIN = 0.0  # No other corrections needed in LV

elif REDPITAYA_INPUT_MODE.upper() == 'HV':
    # HV Mode gains
    HV_GAIN = 20.0  # Standard 20:1 physical divider
    PYRPL_SCOPE_GAIN = 30.08263635  # PyRPL correction for HV
    OTHER_GAIN = 0  # Hardware calibration + other corrections
    # Total = 20 + 11.7 + 8.72854255 = 40.42854255

else:
    raise ValueError("REDPITAYA_INPUT_MODE must be 'LV' or 'HV'")

# Total Gain (additive)
TOTAL_GAIN = HV_GAIN + PYRPL_SCOPE_GAIN + OTHER_GAIN

# ==================================================

print("=" * 70)
print("RED PITAYA DC GAIN TEST")
print("=" * 70)
print(f"DC Output Voltage (OUT1): {DC_OUTPUT_VOLTAGE:.3f} V")
print(f"Red Pitaya Mode: {REDPITAYA_INPUT_MODE}")
print()
print("GAIN BREAKDOWN (ADDITIVE):")
print(f"  HV Gain:                  {HV_GAIN:.1f}")
print(f"  + PyRPL Scope Gain:       {PYRPL_SCOPE_GAIN:.1f}")
print(f"  + Other Gain:             {OTHER_GAIN:.1f}")
print(f"  ─────────────────────────────────────────")
print(f"  TOTAL GAIN (sum):         {TOTAL_GAIN:.1f}")
print("=" * 70)

# ---------- Connect to Red Pitaya ----------
print("Connecting to Red Pitaya...")
rp = Pyrpl(config='dc_gain_test', hostname='rp-f0909c.local')
rp_modules = rp.rp
asg = rp_modules.asg0
scope = rp_modules.scope

# ---------- Configure DC output ----------
print("Configuring OUT1 for DC output...")
asg.setup(
    waveform='dc',
    offset=DC_OUTPUT_VOLTAGE,
    output_direct='out1'
)
print(f"OUT1 set to DC mode: {DC_OUTPUT_VOLTAGE:.3f} V")

# ---------- Configure scope ----------
print("Configuring scope...")
scope.input1 = 'in1'  # Monitor the physical IN1 input
scope.decimation = 1024
scope.average = True

print(f"Scope input: {scope.input1}")
print(f"Scope decimation: {scope.decimation}")

# Start acquisition
scope._start_acquisition_rolling_mode()
time.sleep(0.5)  # Give it a moment to stabilize

# ---------- Acquire data ----------
print(f"\nMeasuring DC levels for {MEASUREMENT_TIME:.1f} seconds...")
print("TIP: Check the Red Pitaya web UI scope to compare readings!")
start = time.time()
in1_vals = []

sample_count = 0
while time.time() - start < MEASUREMENT_TIME:
    scope.single()
    in1_vals.append(np.mean(scope._data_ch1_current))
    sample_count += 1

    # Progress indicator
    if sample_count % 10 == 0:
        print(f"  Samples: {sample_count} | Current raw: {in1_vals[-1] * 1000:.2f} mV", end='\r')

print(f"\n  Total samples collected: {sample_count}")

in1_vals = np.array(in1_vals)

# ---------- Apply total gain correction ----------
in1_raw_mean = np.mean(in1_vals)
in1_raw_std = np.std(in1_vals)

# Apply total gain (single multiplication)
in1_corrected = in1_vals * TOTAL_GAIN
in1_corr_mean = np.mean(in1_corrected)
in1_corr_std = np.std(in1_corrected)

# ---------- Results ----------
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"OUT1 set to:              {DC_OUTPUT_VOLTAGE:.6f} V")
print()
print(f"IN1 (PyRPL raw reading):  {in1_raw_mean:.6f} ± {in1_raw_std:.6f} V")
print(f"                          ({in1_raw_mean * 1000:.3f} ± {in1_raw_std * 1000:.3f} mV)")
print()
print(f"Total Gain Applied:       {TOTAL_GAIN:.1f}")
print(f"  = {HV_GAIN:.1f} (HV) + {PYRPL_SCOPE_GAIN:.1f} (PyRPL) + {OTHER_GAIN:.1f} (Other)")
print()
print(f"IN1 (corrected):          {in1_corr_mean:.6f} ± {in1_corr_std:.6f} V")
print()
print(f"Expected IN1:             {DC_OUTPUT_VOLTAGE:.6f} V")
print(f"Error (absolute):         {in1_corr_mean - DC_OUTPUT_VOLTAGE:.6e} V")
print(f"Error (relative):         {100 * (in1_corr_mean - DC_OUTPUT_VOLTAGE) / DC_OUTPUT_VOLTAGE:.2f}%")
print("=" * 70)

# ---------- Diagnostic information ----------
print("\nDIAGNOSTIC INFO:")
print(f"Measurement stability (noise): {(in1_raw_std / in1_raw_mean) * 100:.3f}% RMS")

if abs(in1_corr_mean / DC_OUTPUT_VOLTAGE - 1.0) > 0.1:
    print("\n⚠️  WARNING: Error > 10%")
    print("Suggested total gain adjustment:")
    suggested_gain = DC_OUTPUT_VOLTAGE / in1_raw_mean
    print(f"  TOTAL_GAIN = {suggested_gain:.2f}")
    print(f"  You can adjust individual components:")
    print(f"    HV_GAIN = {HV_GAIN:.1f} (fixed)")
    print(f"    PYRPL_SCOPE_GAIN = {PYRPL_SCOPE_GAIN:.1f} (adjust if needed)")
    print(f"    OTHER_GAIN = {suggested_gain - HV_GAIN - PYRPL_SCOPE_GAIN:.1f} (adjust if needed)")
else:
    print("\n✓ Measurement accurate within 10%")
    if abs(in1_corr_mean / DC_OUTPUT_VOLTAGE - 1.0) < 0.01:
        print("✓✓ Excellent accuracy (<1% error)!")

print("\n" + "=" * 70)
print("Timestamp:", datetime.now().isoformat())
print("Test complete.")
print("=" * 70)

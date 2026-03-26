"""
Rodeostat DC / Ramp / CV Test
Dominic Morris
Collects, saves, and plots voltage/current using the Rodeostat waveform engine.

DC: Emulated with cyclic test where volt_min = volt_max
Ramp: Emulated with cyclic test where volt_min != volt_max and num_cycles=1
CV: Standard cyclic voltammetry

When launched by Startsystem.py, port selection and setup happens on launch,
then a single input() waits for the synchronized trigger Enter.

Have A Great Day!! :)
-Dominic
"""

from potentiostat import Potentiostat
import matplotlib
matplotlib.use('Agg')  # saves plots silently, no popups
import matplotlib.pyplot as plt
import serial.tools.list_ports
import traceback
import numpy as np
import os
from datetime import datetime

# Change Com_Port to match your Rodeostat (check Device Manager)
Com_Port = 'COM6'

port = Com_Port
print(f"Rodeostat port: {port}")

# Output directory setup
Output_Dir = 'rodeostat_data'
os.makedirs(Output_Dir, exist_ok=True)

Timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Test parameters
Mode = 'CV'           # Options: 'DC', 'RAMP', 'CV'
Curr_Range = '1000uA'
Sample_Rate = 1000.0  # Hz
Quiet_Time = 0
Quiet_Value = 0.0

# DC / Ramp settings
V_Start = 0.8         # Start voltage for RAMP, constant for DC
V_End = 1.0           # End voltage for RAMP
Dc_Runtime = 30       # seconds, DC mode only

# CV settings
Volt_Min = 0.5
Volt_Max = 1.0
Volt_Per_Sec = 0.2
Num_Cycles = 1

# Convert mode to cyclic parameters
if Mode.upper() == 'CV':
    volt_min = Volt_Min
    volt_max = Volt_Max
    num_cycles = Num_Cycles
    ramp_direction = 'normal'

elif Mode.upper() == 'DC':
    volt_min = V_Start
    volt_max = V_Start
    amplitude = 0
    offset = V_Start
    period_ms = 1000
    num_cycles = max(1, int(Dc_Runtime * 1000 / period_ms))
    ramp_direction = 'normal'

elif Mode.upper() == 'RAMP':
    if V_Start < V_End:
        volt_min = V_Start
        volt_max = V_End
        ramp_direction = 'ascending'
    else:
        volt_min = V_End
        volt_max = V_Start
        ramp_direction = 'descending'
    num_cycles = 1

else:
    raise ValueError("Invalid Mode, choose 'DC', 'RAMP', or 'CV'")

# Derived waveform settings
if Mode.upper() != 'DC':
    amplitude = (volt_max - volt_min) / 2
    offset    = (volt_max + volt_min) / 2
    period_ms = int(1000 * 4 * amplitude / Volt_Per_Sec) if amplitude != 0 else 1000

shift = 0.5 if (Mode.upper() == 'RAMP' and ramp_direction == 'descending') else 0.0

test_param = {
    'quietValue': Quiet_Value,
    'quietTime':  Quiet_Time,
    'amplitude':  amplitude,
    'offset':     offset,
    'period':     period_ms,
    'numCycles':  num_cycles,
    'shift':      shift,
}

# Configure device
dev = Potentiostat(port)

try:
    _ = dev.get_all_curr_range()
except KeyError:
    print("Unknown firmware. Adding current range list manually.")
    dev.hw_variant = 'manual_patch'
    dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

try:
    dev.set_curr_range(Curr_Range)
    dev.set_sample_rate(Sample_Rate)
    dev.set_param('cyclic', test_param)
except Exception as e:
    print("Error configuring device:", e)
    traceback.print_exc()
    raise SystemExit

print()
print("=" * 60)
print("Rodeostat configured and ready.")
print(f"  Mode:        {Mode.upper()}")
print(f"  Voltage:     {volt_min:.3f} V  ->  {volt_max:.3f} V")
print(f"  Rate:        {Volt_Per_Sec} V/s    Cycles: {num_cycles}")
print(f"  Curr range:  {Curr_Range}    Sample rate: {Sample_Rate} Hz")
print("Waiting for synchronized start trigger...")
print("=" * 60)

input()  # Startsystem.py sends \n here to fire all three at once

# Run test
print(f"Running {Mode.upper()} test (using cyclic waveform)")
if Mode.upper() == 'RAMP':
    print(f"Ramp direction: {V_Start}V -> {V_End}V ({ramp_direction})")

try:
    t, volt, curr = dev.run_test('cyclic', display='data', filename=None)
except Exception as e:
    print("Error running test:", e)
    traceback.print_exc()
    raise SystemExit

print("Test complete.")

# Save to CSV
Datafile_Csv = os.path.join(Output_Dir, f'{Mode.lower()}_data_{Timestamp}.csv')
csv_data = np.column_stack((t, volt, curr))
header = 'Time(s),Voltage(V),Current(uA)'
np.savetxt(Datafile_Csv, csv_data, delimiter=',', header=header, comments='')
print(f"CSV data saved to {Datafile_Csv}")

# Plot results
plt.figure(1, figsize=(10, 8))
plt.subplot(211)
plt.plot(t, volt, linewidth=1.5)
plt.ylabel('Voltage (V)')
plt.title(f'{Mode.upper()} Test Results')
plt.grid(True)

plt.subplot(212)
plt.plot(t, curr, linewidth=1.5)
plt.ylabel('Current (uA)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.figure(2, figsize=(8, 6))
plt.plot(volt, curr, linewidth=1.5)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (uA)')
plt.title(f'{Mode.upper()} Test - I-V Curve')
plt.grid(True)

plt.tight_layout()
Rodeostat_Png1 = os.path.join(Output_Dir, f'{Mode.lower()}_timeseries_{Timestamp}.png')
plt.figure(1)
plt.savefig(Rodeostat_Png1, dpi=150, bbox_inches='tight')
print(f"Plot saved to {Rodeostat_Png1}")

Rodeostat_Png2 = os.path.join(Output_Dir, f'{Mode.lower()}_ivcurve_{Timestamp}.png')
plt.figure(2)
plt.savefig(Rodeostat_Png2, dpi=150, bbox_inches='tight')
print(f"Plot saved to {Rodeostat_Png2}")

plt.close('all')

"""
Rodeostat DC / Ramp / CV Test
Dominic Morris
Collects, saves, and plots voltage/current using built-in Rodeostat waveform engine.
- DC: Emulated with cyclic test where volt_min = volt_max
- Ramp: Emulated with cyclic test where volt_min != volt_max and num_cycles=1
- CV: Standard cyclic voltammetry

Have A Great Day!! :)
-Dominic
"""

from potentiostat import Potentiostat
import matplotlib.pyplot as plt
import serial.tools.list_ports
import traceback
import numpy as np
import os
from datetime import datetime

# ---------------------------------------------------------------------
# Port setup
# ---------------------------------------------------------------------
ports = serial.tools.list_ports.comports()
if not ports:
    raise SystemExit("No serial ports found. Connect your Rodeostat and try again.")

print("Available COM ports:")
for i, p in enumerate(ports):
    print(f"{i}: {p.device} - {p.description}")
choice = int(input("Select port number: "))
port = ports[choice].device
print("Using port:", port)

# ---------------------------------------------------------------------
# File setup
# ---------------------------------------------------------------------
output_dir = 'rodeostat_data'
os.makedirs(output_dir, exist_ok=True)

# Generate timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------
# Test Parameters (all editable)
# ---------------------------------------------------------------------
MODE = 'CV'  # Options: 'DC', 'RAMP', 'CV'
CURR_RANGE = '1000uA'
SAMPLE_RATE = 1000.0  # Hz
QUIET_TIME = 0
QUIET_VALUE = 0.0

# DC / Ramp settings
V_START = 0.8  # Start voltage for RAMP, constant value for DC mode
V_END = 1  # End voltage for RAMP (can be higher or lower than V_START)
DC_RUNTIME = 30  # seconds, default for DC mode

# CV settings
VOLT_MIN = 0.5
VOLT_MAX = 1.0
VOLT_PER_SEC = 0.2
NUM_CYCLES = 1

# ---------------------------------------------------------------------
# Convert mode to cyclic parameters
# ---------------------------------------------------------------------
if MODE.upper() == 'CV':
    volt_min = VOLT_MIN
    volt_max = VOLT_MAX
    num_cycles = NUM_CYCLES
    ramp_direction = 'normal'

elif MODE.upper() == 'DC':
    volt_min = V_START
    volt_max = V_START
    try:
        DC_RUNTIME = float(input("Enter DC test runtime in seconds: "))
    except:
        DC_RUNTIME = 10
    amplitude = 0
    offset = V_START
    period_ms = 1000  # 1 second per cycle (arbitrary)
    num_cycles = max(1, int(DC_RUNTIME * 1000 / period_ms))
    ramp_direction = 'normal'

elif MODE.upper() == 'RAMP':
    # Handle both ascending and descending ramps
    if V_START < V_END:
        volt_min = V_START
        volt_max = V_END
        ramp_direction = 'ascending'
    else:
        volt_min = V_END
        volt_max = V_START
        ramp_direction = 'descending'
    num_cycles = 1

else:
    raise ValueError("Invalid MODE, choose 'DC', 'RAMP', or 'CV'")

# Derived waveform settings (for CV or RAMP)
if MODE.upper() != 'DC':
    amplitude = (volt_max - volt_min) / 2
    offset = (volt_max + volt_min) / 2
    period_ms = int(1000 * 4 * amplitude / VOLT_PER_SEC) if amplitude != 0 else 1000

# Adjust shift for descending ramps
if MODE.upper() == 'RAMP' and ramp_direction == 'descending':
    shift = 0.5  # Start at the peak (descending)
else:
    shift = 0.0  # Start at the valley (ascending)

test_param = {
    'quietValue': QUIET_VALUE,
    'quietTime': QUIET_TIME,
    'amplitude': amplitude,
    'offset': offset,
    'period': period_ms,
    'numCycles': num_cycles,
    'shift': shift
}

# ---------------------------------------------------------------------
# Create and configure device
# ---------------------------------------------------------------------
dev = Potentiostat(port)

# Patch for unknown firmware
try:
    _ = dev.get_all_curr_range()
except KeyError:
    print("Unknown firmware. Adding current range list manually.")
    dev.hw_variant = 'manual_patch'
    dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

try:
    dev.set_curr_range(CURR_RANGE)
    dev.set_sample_rate(SAMPLE_RATE)
    dev.set_param('cyclic', test_param)  # Always use 'cyclic'
except Exception as e:
    print("Error configuring device:", e)
    traceback.print_exc()
    raise SystemExit

# ---------------------------------------------------------------------
# Run test
# ---------------------------------------------------------------------
print(f"Running {MODE.upper()} test (using cyclic waveform)")
if MODE.upper() == 'RAMP':
    print(f"Ramp direction: {V_START}V → {V_END}V ({ramp_direction})")

try:
    t, volt, curr = dev.run_test('cyclic', display='data', filename=None)
except Exception as e:
    print("Error running test:", e)
    traceback.print_exc()
    raise SystemExit

print("Test complete.")

# ---------------------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------------------
datafile_csv = os.path.join(output_dir, f'{MODE.lower()}_data_{timestamp}.csv')

# Prepare data for CSV
csv_data = np.column_stack((t, volt, curr))
header = 'Time(s),Voltage(V),Current(uA)'

# Save to CSV
np.savetxt(datafile_csv, csv_data, delimiter=',', header=header, comments='')
print(f"💾 CSV data saved to {datafile_csv}")

# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------
plt.figure(1, figsize=(10, 8))
plt.subplot(211)
plt.plot(t, volt, linewidth=1.5)
plt.ylabel('Voltage (V)')
plt.title(f'{MODE.upper()} Test Results')
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
plt.title(f'{MODE.upper()} Test - I-V Curve')
plt.grid(True)

plt.tight_layout()
plt.show()

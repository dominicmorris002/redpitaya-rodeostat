"""
Rodeostat DC / Ramp / CV Test
Dominic Morris
Collects, saves, and plots voltage/current using built-in Rodeostat waveform engine.
- DC: Emulated with cyclic test where volt_min = volt_max
- Ramp: Emulated with cyclic test where volt_min != volt_max and num_cycles=1
- CV: Standard cyclic voltammetry
"""

from potentiostat import Potentiostat
import matplotlib.pyplot as plt
import serial.tools.list_ports
import traceback

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
datafile = 'data.txt'

# ---------------------------------------------------------------------
# Test Parameters (all editable)
# ---------------------------------------------------------------------
MODE = 'DC'  # Options: 'DC', 'RAMP', 'CV'
CURR_RANGE = '100uA'
SAMPLE_RATE = 100.0  # Hz
QUIET_TIME = 0
QUIET_VALUE = 0.0

# DC / Ramp settings
V_START = 1.0
V_END = 0.5

# CV settings
VOLT_MIN = -0.1
VOLT_MAX = 1.0
VOLT_PER_SEC = 0.05
NUM_CYCLES = 1

# ---------------------------------------------------------------------
# Convert mode to cyclic parameters
# ---------------------------------------------------------------------
if MODE.upper() == 'CV':
    volt_min = VOLT_MIN
    volt_max = VOLT_MAX
    num_cycles = NUM_CYCLES
elif MODE.upper() == 'DC':
    volt_min = V_START
    volt_max = V_START
    num_cycles = 1
elif MODE.upper() == 'RAMP':
    volt_min = V_START
    volt_max = V_END
    num_cycles = 1
else:
    raise ValueError("Invalid MODE, choose 'DC', 'RAMP', or 'CV'")

# Derived waveform settings
amplitude = (volt_max - volt_min) / 2
offset = (volt_max + volt_min) / 2
period_ms = int(1000 * 4 * amplitude / VOLT_PER_SEC) if amplitude != 0 else 1000
shift = 0.0

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
try:
    t, volt, curr = dev.run_test('cyclic', display='data', filename=datafile)
except Exception as e:
    print("Error running test:", e)
    traceback.print_exc()
    raise SystemExit

print("Test complete. Data saved to", datafile)

# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------
plt.figure(1)
plt.subplot(211)
plt.plot(t, volt)
plt.ylabel('Voltage (V)')
plt.grid(True)

plt.subplot(212)
plt.plot(t, curr)
plt.ylabel('Current (uA)')
plt.xlabel('Time (s)')
plt.grid(True)

plt.figure(2)
plt.plot(volt, curr)
plt.xlabel('Voltage (V)')
plt.ylabel('Current (uA)')
plt.title(f'{MODE.upper()} Test')
plt.grid(True)

plt.tight_layout()
plt.show()

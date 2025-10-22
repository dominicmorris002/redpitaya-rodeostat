"""
Cyclic Voltammetry for IO Rodeostat
Collects, saves, and plots CV data in µA with 3 graphs, Voltage/Time , Current/Time, Current/Voltage.
Code Reference: https://iorodeo.github.io/iorodeo-potentiostat-docs-build/examples.html#cyclic-voltammetry
Install These in Terminal: pip install potentiostat pyserial matplotlib
Dominic Morris
Have a Great Day! :)
"""


from potentiostat import Potentiostat
import matplotlib.pyplot as plt
import serial.tools.list_ports
import traceback

# ---------------------------------------------------------------------
# Port and File Setup
# ---------------------------------------------------------------------
ports = serial.tools.list_ports.comports()
if not ports:
    raise SystemExit("No serial ports found. Connect your Rodeostat and try again.")
port = ports[0].device
print("Connecting to Rodeostat on", port)

datafile = 'data.txt'  # File to save time, voltage, and current data

# ---------------------------------------------------------------------
# CV Parameters (Easy to change)
# ---------------------------------------------------------------------
test_name = 'cyclic'      # Test type
curr_range = '100uA'      # Current range: choose '1uA', '10uA', '100uA', or '1000uA'
sample_rate = 100.0        # Samples per second (max 200 Hz)

quiet_time = 0            # Time to stabilize before starting (seconds)
quiet_value = 0.0         # Voltage during quiet time (V)

volt_min = -0.1           # Minimum voltage (V)
volt_max = 1.0            # Maximum voltage (V)
volt_per_sec = 0.05       # Scan rate (V/s)
num_cycles = 1            # Number of full cycles (min→max→min)

# Derived waveform settings
amplitude = (volt_max - volt_min) / 2
offset = (volt_max + volt_min) / 2
period_ms = int(1000 * 4 * amplitude / volt_per_sec)
shift = 0.0

test_param = {
    'quietValue': quiet_value,
    'quietTime': quiet_time,
    'amplitude': amplitude,
    'offset': offset,
    'period': period_ms,
    'numCycles': num_cycles,
    'shift': shift
}

# ---------------------------------------------------------------------
# Create and Configure Potentiostat
# ---------------------------------------------------------------------
dev = Potentiostat(port)

# Patch for unknown firmware
try:
    _ = dev.get_all_curr_range()
except KeyError:
    print("Unknown firmware. Adding current range list manually.")
    dev.hw_variant = 'manual_patch'
    dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

# Apply settings
try:
    dev.set_curr_range(curr_range)
    dev.set_sample_rate(sample_rate)
    dev.set_param(test_name, test_param)
except Exception as e:
    print("Error configuring device:", e)
    traceback.print_exc()
    raise SystemExit

# ---------------------------------------------------------------------
# Run Cyclic Voltammetry Test
# ---------------------------------------------------------------------
print("Running cyclic voltammetry test")
try:
    t, volt, curr = dev.run_test(test_name, display='data', filename=datafile)
except Exception as e:
    print("Error running test:", e)
    traceback.print_exc()
    raise SystemExit

print("Test complete. Data saved to", datafile)

# ---------------------------------------------------------------------
# Plot Results
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
plt.title('Cyclic Voltammetry')
plt.grid(True)

plt.tight_layout()
plt.show()

import sys
import time
import matplotlib.pyplot as plt
import serial.tools.list_ports
import traceback

# -----------------------------
# Red Pitaya SCPI Setup
# -----------------------------
try:
    import redpitaya_scpi as scpi
    rp_ip = 'rp-f073ce.local'        # Red Pitaya address
    rp = scpi.scpi(rp_ip)

    # Reset Red Pitaya
    rp.tx_txt('GEN:RST')
    rp.tx_txt('ACQ:RST')

    # AC waveform settings
    wave_form = 'sine'
    freq = 1000000      # 1 MHz
    ampl = 1            # 1 V

    rp.tx_txt(f'SOUR1:FUNC {wave_form.upper()}')
    rp.tx_txt(f'SOUR1:FREQ:FIX {freq}')
    rp.tx_txt(f'SOUR1:VOLT {ampl}')
    rp.tx_txt('SOUR1:BURS:STAT BURST')
    rp.tx_txt('SOUR1:BURS:NCYC 3')
    rp.tx_txt('ACQ:DEC 1')
    rp.tx_txt('ACQ:TRig:LEV 0')
    rp.tx_txt('ACQ:TRig:DLY 0')
    rp.tx_txt('ACQ:START')
    time.sleep(1)
    rp.tx_txt('ACQ:TRig AWG_PE')
    rp.tx_txt('OUTPUT1:STATE ON')
    time.sleep(1)
    rp.tx_txt('SOUR1:TRig:INT')

    # Wait for trigger
    while True:
        rp.tx_txt('ACQ:TRig:STAT?')
        if rp.rx_txt() == 'TD':
            break

    while True:
        rp.tx_txt('ACQ:TRig:FILL?')
        if rp.rx_txt() == '1':
            break

    # Read Red Pitaya data
    rp.tx_txt('ACQ:SOUR1:DATA?')
    data_string = rp.rx_txt()
    data_string = data_string.strip('{}\n\r').replace("  ", "").split(',')
    rp_data = list(map(float, data_string))

except Exception as e:
    print("Error with Red Pitaya:", e)
    rp_data = []

# -----------------------------
# Rodeostat CV Setup
# -----------------------------
try:
    from potentiostat import Potentiostat

    # Find Rodeostat
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise SystemExit("No serial ports found. Connect your Rodeostat.")
    rodeo_port = ports[1].device
    print("Connecting to Rodeostat on", rodeo_port)

    dev = Potentiostat(rodeo_port)

    # Patch unknown firmware
    try:
        _ = dev.get_all_curr_range()
    except KeyError:
        dev.hw_variant = 'manual_patch'
        dev.get_all_curr_range = lambda: ['1uA', '10uA', '100uA', '1000uA']

    # CV parameters
    test_name = 'cyclic'
    curr_range = '100uA'
    sample_rate = 100.0
    volt_min = -0.1
    volt_max = 1.0
    volt_per_sec = 0.05
    num_cycles = 1

    amplitude = (volt_max - volt_min) / 2
    offset = (volt_max + volt_min) / 2
    period_ms = int(1000 * 4 * amplitude / volt_per_sec)
    shift = 0.0

    test_param = {
        'quietValue': 0.0,
        'quietTime': 0,
        'amplitude': amplitude,
        'offset': offset,
        'period': period_ms,
        'numCycles': num_cycles,
        'shift': shift
    }

    dev.set_curr_range(curr_range)
    dev.set_sample_rate(sample_rate)
    dev.set_param(test_name, test_param)

    # Run CV test
    print("Running cyclic voltammetry test")
    t, volt, curr = dev.run_test(test_name, display='data', filename='data.txt')
except Exception as e:
    print("Error with Rodeostat:", e)
    t, volt, curr = [], [], []

# -----------------------------
# Plot everything
# -----------------------------
plt.figure(1)
plt.subplot(311)
if rp_data:
    plt.plot(rp_data)
    plt.title("Red Pitaya AC Output")
plt.grid(True)

plt.subplot(312)
if t and volt:
    plt.plot(t, volt)
    plt.ylabel('Voltage (V)')
    plt.grid(True)

plt.subplot(313)
if t and curr:
    plt.plot(t, curr)
    plt.ylabel('Current (uA)')
    plt.xlabel('Time (s)')
    plt.grid(True)

plt.tight_layout()
plt.show()

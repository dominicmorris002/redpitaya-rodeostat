"""
AC + DC Potentiostat experiment
Rodeostat provides DC bias
Red Pitaya provides AC perturbation and lock-in measurement
Dominic Morris, 2025
"""

import time
import traceback
import serial.tools.list_ports
import matplotlib.pyplot as plt
import numpy as np

# import the two working modules exactly as they are
from potentiostat import Potentiostat           # from iorodeo library
from redpitaya_module import RedPitaya           # your file above


def run_ac_dc_experiment(
        dc_bias=0.5,                # volts vs RE
        ac_freq=100,                # Hz
        ac_amp=0.01,                # volts
        curr_range='100uA',
        sample_rate=100.0):

    # -----------------------------------------------------------------
    # 1.  Connect to Rodeostat and hold DC bias
    # -----------------------------------------------------------------
    ports = serial.tools.list_ports.comports()
    if not ports:
        raise SystemExit("No serial ports found. Connect Rodeostat and try again.")
    port = ports[0].device
    print(f"Connecting to Rodeostat on {port}")
    dev = Potentiostat(port)

    try:
        dev.set_curr_range(curr_range)
        dev.set_sample_rate(sample_rate)
        dev.set_param('constant', {'voltage': dc_bias})
        dev.run_test('constant', display=None)
        print(f"Holding {dc_bias:.3f} V DC bias...")
    except Exception as e:
        print("Error configuring Rodeostat:", e)
        traceback.print_exc()
        return

    # let bias settle
    time.sleep(1.0)

    # -----------------------------------------------------------------
    # 2.  Start Red Pitaya lock-in
    # -----------------------------------------------------------------
    rp = RedPitaya(output_dir='ac_test_data')

    run_params = {
        'test_freq': ac_freq,
        'test_amp': ac_amp,
        'noise_freq': 0,
        'noise_amp': 0.0,
        'ref_freq': ac_freq,
        'ref_amp': ac_amp
    }

    print(f"Running lock-in at {ac_freq} Hz, amplitude = {ac_amp} V")
    rp.run(run_params, save_file=False, test=True, fft=False)

    # -----------------------------------------------------------------
    # 3.  Retrieve lock-in results
    # -----------------------------------------------------------------
    X, Y = rp.capture()
    amp = np.sqrt(np.mean(X**2 + Y**2))
    phase = np.degrees(np.arctan2(np.mean(Y), np.mean(X)))

    print(f"Lock-in amplitude ≈ {amp:.4f}, phase = {phase:.1f} °")

    # optional plot
    plt.figure()
    plt.plot(X, label='X')
    plt.plot(Y, label='Y')
    plt.title(f"Lock-in outputs at {ac_freq} Hz")
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (a.u.)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Experiment complete.")


if __name__ == "__main__":
    run_ac_dc_experiment(dc_bias=0.5, ac_freq=100, ac_amp=0.01)

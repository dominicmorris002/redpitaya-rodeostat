import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyrpl import Pyrpl
import sys
import time
import csv
from datetime import datetime
import os

# ===========================
# USER PARAMETERS (change here)
IP_ADDRESS      = "rp-f073ce.local"
FREQUENCY       = 1000      # Hz
AMPLITUDE       = 0.1       # V
DURATION        = 10        # seconds
SAMPLE_POINTS   = 16384     # points to capture each frame
UPDATE_INTERVAL = 50        # ms
OUTPUT_CHANNEL  = 'out1'
INPUT1_CHANNEL  = 'in1'
INPUT2_CHANNEL  = 'in2'
ZOOM_POINTS     = 500       # number of points to display (zoomed)
OUTPUT_DIR      = "redpitaya_data"
# ===========================

class LiveRedPitaya:
    def __init__(self):
        # Connect to Red Pitaya
        self.rp = Pyrpl(config="", hostname=IP_ADDRESS)
        self.scope = self.rp.rp.scope
        self.asg = self.rp.rp.asg0

        # Setup scope
        self.scope.input1 = INPUT1_CHANNEL
        self.scope.input2 = INPUT2_CHANNEL
        self.scope.decimation = 16384
        self.scope.average = True
        self.scope._start_acquisition_rolling_mode()

        # Setup waveform
        self.asg.setup(
            waveform='sin',
            frequency=FREQUENCY,
            amplitude=AMPLITUDE,
            output_signal='sin',
            output_direct=OUTPUT_CHANNEL
        )

        # PyQtGraph setup
        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Red Pitaya Live AC Measurement")
        self.win.resize(1000,600)

        self.plot_voltage = self.win.addPlot(title="Voltage Input/Output")
        self.plot_voltage.addLegend()
        self.plot_voltage.setLabel('left', 'Voltage', units='V')
        self.plot_voltage.setLabel('bottom', 'Sample Index')

        self.curve_ch1 = self.plot_voltage.plot(pen='b', name='Input (CH1)')
        self.curve_ch2 = self.plot_voltage.plot(pen='r', name='Output (CH2)')

        self.win.nextRow()
        self.plot_phase = self.win.addPlot(title="Phase Difference")
        self.plot_phase.setLabel('left', 'Phase', units='deg')
        self.plot_phase.setLabel('bottom', 'Sample Index')
        self.curve_phase = self.plot_phase.plot(pen='g', name='Phase')

        # Timer for live updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(UPDATE_INTERVAL)

        self.start_time = time.time()
        self.data_log = []  # store all measurements

    def compute_phase(self, ch1, ch2):
        if len(ch1) == 0 or len(ch2) == 0:
            return 0
        ch1_norm = ch1 / np.max(np.abs(ch1)) if np.max(np.abs(ch1)) else ch1
        ch2_norm = ch2 / np.max(np.abs(ch2)) if np.max(np.abs(ch2)) else ch2
        corr = np.correlate(ch1_norm, ch2_norm, mode='full')
        max_idx = np.argmax(corr)
        phase_samples = max_idx - len(ch1) + 1
        phase_rad = 2 * np.pi * phase_samples / len(ch1)
        phase_deg = np.degrees(phase_rad)
        return phase_deg

    def update(self):
        # Capture latest data
        ch1_full = np.array(self.scope._data_ch1_current[:SAMPLE_POINTS])
        ch2_full = np.array(self.scope._data_ch2_current[:SAMPLE_POINTS])

        # Zoomed-in display
        ch1 = ch1_full[:ZOOM_POINTS]
        ch2 = ch2_full[:ZOOM_POINTS]

        # Compute phase
        phase_val = self.compute_phase(ch1, ch2)

        # Update plots
        self.curve_ch1.setData(ch1)
        self.curve_ch2.setData(ch2)
        self.curve_phase.setData(np.full_like(ch1, phase_val))

        # Auto-range
        self.plot_voltage.enableAutoRange('xy', True)
        self.plot_phase.enableAutoRange('xy', True)

        # Store measurement
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        for i in range(len(ch1)):
            self.data_log.append([timestamp, ch1[i], ch2[i], phase_val])

        # Stop waveform after duration
        if time.time() - self.start_time > DURATION:
            self.timer.stop()
            self.asg.output_direct = 'off'
            print("✓ Measurement complete")
            self.save_csv()
            print("✓ Data saved")
            # window remains open

    def save_csv(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, f"redpitaya_data_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'CH1', 'CH2', 'Phase_deg'])
            writer.writerows(self.data_log)

    def run(self):
        QtGui.QApplication.instance().exec_()


if __name__ == "__main__":
    rp_live = LiveRedPitaya()
    rp_live.run()

import threading
import time
from PyDAQmx import *
import ctypes
import numpy as np
import pyvisa


class SEEDTask(Task):
    def __init__(self, duration=3600, rate=500, data_len=250, frequency=500, amplitude=0.2):
        Task.__init__(self)

        self.dev_name = "Dev1"

        # User Altered Settings
        ##
        rm = pyvisa.ResourceManager()
        self.lock_in = rm.open_resource('GPIB0::12::INSTR')

        #convert frequency to appropriate units
        self.frequency = str(frequency*1000)    #convert Hz to mHz
        self.amplitude = str(round((amplitude/1.414),3))   #convert V to Vrms

     ###### Lock-in amplifier setting #####
        self.lock_in.write('REFMODE 0')  # check
        self.lock_in.write('IE 0')
        self.lock_in.write('CH 1 2')    # mag
        self.lock_in.write('CH 2 3')    # phase
        self.lock_in.write('TC 13')  # set time constant
        self.lock_in.write('SYNC 1')  # set time constant Sync
        #####################################################################
        self.lock_in.write('SEN 26')  # set sensitivity 23 = 50mV, 26 = 500mV
        self.lock_in.write('ACGAIN 0')  # set Gain, for SEED is 0
        #####################################################################
        self.lock_in.write('SLOPE 3')  # set Filter Slope
        self.lock_in.write('OF '+self.frequency)     #set frequency (mHZ)
        self.lock_in.write('OA '+self.amplitude)   #set amplitude in V
        self.sensitivity = float(self.lock_in.query('SEN.'))  # Lock-in amplifier Sensitivity
        self.rate = float(rate)  # Rate per second of Data Acquisition by NI Hardware
        self.duration = float(duration)
        self.time = np.arange(0, self.duration, 1 / self.rate)

        self.dataLen = data_len
        self._data = np.zeros((self.dataLen, 7))

        # preallocate data logging arrays
        self.mag = np.zeros((int(self.duration) * int(self.rate),))
        self.dcRamp = np.zeros((int(self.duration) * int(self.rate),))
        #self.aDC = np.zeros((int(self.duration) * int(self.rate),))
        #self.bDC = np.zeros((int(self.duration) * int(self.rate),))
        self.phase = np.zeros((int(self.duration) * int(self.rate),))
        #self.signal_numerator = np.zeros((self.dataLen,))
        #self.signal_denomenator = np.zeros((self.dataLen,))
        self.signal = np.zeros((int(self.duration) * int(self.rate),))  # Calculated 1f signal (in nm)
        #self.comp_voltage = np.zeros((int(self.duration) * int(self.rate),))
        #self.current = np.zeros((int(self.duration) * int(self.rate),))
        self.read = int32()
        self.compError = 0
        self.numRecord = 0

        # create Voltage Channels
        self.CreateAIVoltageChan(self.dev_name + "/ai0", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai1", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai2", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai3", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai4", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai5", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
        self.CreateAIVoltageChan(self.dev_name + "/ai6", '', DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)

        self.CfgSampClkTiming("", self.rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, self.dataLen)

        # PID Gains
        self.Kp = 0
        self.Kd = 0
        self.Ki = 0

        self.units = 'pm'

    def compensate(self):
        error = self.aDC[self.numRecord] - self.bDC[self.numRecord]
        new_comp_voltage = self.comp_voltage[self.numRecord] + self.Kp * error
        self.set_comp_voltage(new_comp_voltage)
        self.comp_voltage[self.numRecord:self.numRecord + self.dataLen] = new_comp_voltage

    def set_comp_voltage(self, v):
        return 0.0

    def continuousRecord(self):
        self._data_lock = threading.Lock()
        self._iP_lock = threading.Lock()
        self._newdata_event = threading.Event()
        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.dataLen, 0)
        self.AutoRegisterDoneEvent(0)

    def EveryNCallback(self):
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                self._newdata_event.set()
                # update arrays that encode SEED data
                self.mag[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 0]
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                #self.aDC[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 2]
                #self.bDC[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 3]
                self.phase[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 4] * 20
                #self.comp_voltage[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 5]
                #self.current[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 6]
                #self.signal_numerator = (self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.1) * 633 / (2 * np.pi)
                #self.signal_denomenator = self.aDC[self.numRecord:self.numRecord + self.dataLen] + self.bDC[self.numRecord:self.numRecord + self.dataLen]
                self.signal[self.numRecord:self.numRecord + self.dataLen] = self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.707 #in mA? (check)
                self.compensate()  # run compensator nulling algorithm
                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0  # The function should return an integer

    def SetCompensatorStatus(self, status):
        if status == "off":
            self.CompensatorStatus = 0
        else:
            self.CompensatorStatus = 1

    def DoneCallback(self, status):
        print "Status", status.value
        return 0  # The function should return an integer

    def get_data(self, blocking=True, timeout=None):
        if blocking:
            if not self._newdata_event.wait(timeout):
                raise ValueError("timeout waiting for data from device")
        with self._data_lock:
            self._newdata_event.clear()
            return self._data.copy()

    def getIntialVoltageList(self):
        return 0.0

    def initialNull(self):
        compensatorVoltages = self.getIntialVoltageList()
        self.voltageList = []

        def updateCompList():
            self.StartTask()
            self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, self._data, self.dataLen * 5,
                               byref(self.read), None)
            d = np.mean(self._data[:, 2] - self._data[:, 3])
            self.voltageList.append(d)
            self.StopTask()

        for x in compensatorVoltages:
            self.setCompVoltage(x)
            time.sleep(0.1)
            updateCompList()

        self.voltageList = np.absolute(self.voltageList)
        self.compVoltage = compensatorVoltages[np.argmin(self.voltageList)]
        self.setCompVoltage(self.compVoltage)

    def plotCompResponse(self):
        self.compensatorVoltages = np.arange(-4.0, 4.0, .05)
        self.voltageList = []

        def updateCompList():
            self.StartTask()
            self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, self._data, self.dataLen * 5,
                               byref(self.read), None)
            time.sleep(0.1)
            d = np.mean(self._data[:, 2] - self._data[:, 3])
            self.voltageList.append(d)
            self.StopTask()

        for x in self.compensatorVoltages:
            self.setCompVoltage(x)
            time.sleep(0.01)
            updateCompList()

        plt.plot(self.compensatorVoltages, self.voltageList)

    def reset(self):
        self.mag = np.zeros((self.duration * self.rate,))
        self.dcRamp = np.zeros((self.duration * self.rate,))
        #self.aDC = np.zeros((self.duration * self.rate,))
        #self.bDC = np.zeros((self.duration * self.rate,))
        #self.aPlusB = np.zeros((self.duration * self.rate,))
        self.phase = np.zeros((self.duration * self.rate,))
        self.signal = np.zeros((self.duration * self.rate,))  # Calculated 1f signal (in nm)
        self.compV = np.zeros((self.duration * self.rate,))
        self.read = int32()
        self.compError = 0
        self.numRecord = 0

from LIC_Object import *
from PyDAQmx import *
import os
import csv


class LockInCurrent (SEEDTask):
    def __init__(self, duration=3600,rate=500, data_len=250, frequency=500, amplitude=0.2):
        SEEDTask.__init__(self, duration, rate, data_len, frequency, amplitude)
        self.units = 'mA'

    def EveryNCallback(self):
        if self.numRecord < self.duration * self.rate:
            with self._data_lock:
                self.ReadAnalogF64(DAQmx_Val_Auto, 10.0, DAQmx_Val_GroupByScanNumber, self._data, self.dataLen * 7, ctypes.byref(self.read), None)
                self._newdata_event.set()
                # update arrays that encode SEED data
                self.mag[self.numRecord:self.numRecord + self.dataLen] = np.absolute(self._data[:, 0])
                self.dcRamp[self.numRecord:self.numRecord + self.dataLen] = -self._data[:, 1]
                self.phase[self.numRecord:self.numRecord + self.dataLen] = self._data[:, 4] * 20 #(check)
                #self.aDC[self.numRecord:self.numRecord + self.dataLen] = np.absolute(self._data[:, 2])
                #self.bDC[self.numRecord:self.numRecord + self.dataLen] = np.absolute(self._data[:, 3])
                #self.signal_numerator = (self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 10000 * 0.4)
                #self.signal_denomenator = self.aDC[self.numRecord:self.numRecord + self.dataLen]
                self.signal[self.numRecord:self.numRecord + self.dataLen] = self.mag[self.numRecord:self.numRecord + self.dataLen] * self.sensitivity * 0.707 #multiplying by 0.001 is A, without any scaling is mA, 0.707 to get V_abs (Vrms = Vabs/sqrt(2))
                self.numRecord += self.dataLen
                self._newdata_event.set()

        return 0  # The function should return an integer

    def compensate(self):
        return 0


def save_LIC(taskName, fileName):
    tempDir=os.getcwd()  # tempDir keeps track of old working directory to change directory back once program has run
    #save_directory = 'C:\SEED 3.2 Data\Joydip\e2025\February\Capacitance Test'
    #save_directory = 'C:\SEED 3.2 Data\Thomas\Three hole chip'
    #save_directory = 'C:\SEED 3.2 Data\Tawfiq\A_2025\March\A_18_01\T_10 micro K3PO4'
    save_directory = 'C:\SEED 3.2 Data\Joydip\e2025\Novenber\Batch 3 Chip testing'
    #save_directory = 'C:\SEED 3.2 Data\Thomas\holes on all electrodeNewchip'

    os.chdir(save_directory)
    mag = taskName.mag
    dcRamp = taskName.dcRamp
    #aDC = taskName.aDC
    #bDC = taskName.bDC
    time = taskName.time
    phase = taskName.phase
    signal = taskName.signal
    #current = taskName.current
    #comp_voltage = taskName.comp_voltage
    #np.savez(fileName+'.npz', mag=mag, dcRamp=dcRamp, aDC=aDC, bDC=bDC, phase=phase, time=time, signal=signal, comp_voltage=comp_voltage, current=current)
    np.savez(fileName + '.npz', mag=mag, dcRamp=dcRamp, phase=phase, time=time, signal=signal)
    dat = loadSEEDDatanpz(fileName+'.npz')
    os.chdir(save_directory)
    os.mkdir(fileName)
    os.chdir(save_directory + '\\' + fileName)
    csvGenerate(dat)
    os.chdir(tempDir)

def loadSEEDDatanpz(fileName):
    s = np.load(fileName)
    #aDC = s['aDC']
    #bDC = s['bDC']
    dcRamp = s['dcRamp']
    mag = s['mag']
    phase = s['phase']
    time = s['time']
    signal = s['signal']
    #comp_voltage = s['comp_voltage']
    #current = s['current']
    #return {'aDC': aDC, 'bDC': bDC, 'dcRamp': dcRamp, 'magnitude': mag, 'phase': phase, 'time': time, 'signal': signal, 'comp_voltage': comp_voltage, 'current': current}
    return {'dcRamp': dcRamp, 'magnitude': mag, 'phase': phase, 'time': time, 'signal': signal}

def csvGenerate(dat):
    #create CSV files relating to SEED outputs
    for i in dat.keys():
        with open(i+'.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerow(dat[i])



import sys
import time
import matplotlib.pyplot as plt
import redpitaya_scpi as scpi

IP = 'rp-f073ce.local'        # My Red Pitaya Address
rp = scpi.scpi(IP)

wave_form = 'sine'
freq = 1000000
ampl = 1

# Reset Generation and Acquisition
rp.tx_txt('GEN:RST')
rp.tx_txt('ACQ:RST')

##### Generation #####
rp.tx_txt('SOUR1:FUNC ' + str(wave_form).upper())
rp.tx_txt('SOUR1:FREQ:FIX ' + str(freq))
rp.tx_txt('SOUR1:VOLT ' + str(ampl))

rp.tx_txt('SOUR1:BURS:STAT BURST')        # Mode set to BURST
rp.tx_txt('SOUR1:BURS:NCYC 3')            # 3 periods in each burst

##### Acqusition #####
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
while 1:
    rp.tx_txt('ACQ:TRig:STAT?')           # Get Trigger Status
    if rp.rx_txt() == 'TD':               # Triggerd?
        break

## ! OS 2.00 or higher only ! ##
while 1:
    rp.tx_txt('ACQ:TRig:FILL?')
    if rp.rx_txt() == '1':
        break

# Read data and plot
rp.tx_txt('ACQ:SOUR1:DATA?')              # Read full buffer (source 1)
data_string = rp.rx_txt()                 # data into a string

# Remove brackets and empty spaces + string => float
data_string = data_string.strip('{}\n\r').replace("  ", "").split(',')
data = list(map(float, data_string))        # transform data into float

plt.plot(data)
plt.show()

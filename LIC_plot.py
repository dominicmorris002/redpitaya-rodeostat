import matplotlib.pyplot as plt
import matplotlib.animation as animation
from LIC_Measurement_Types import *
#from save_SEED import*

task = LockInCurrent(duration=12)     #frequency in Hz, amplitude in V





fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = ax1.twinx()
ax2.set_ylabel(task.units, color=(31 / 255., 119 / 255., 180 / 255.))
#ax2.set_ylim([0, 100]) #use for diff test
ax1.set_ylabel('DC Ramp', color=(255 / 255., 127 / 255., 14 / 255.))
plt.show()

task.continuousRecord()
task.StartTask()
plt.show()
def animate(i):
    ax2.plot(task.time[0:task.numRecord], task.signal[0:task.numRecord], color=(31 / 255., 119 / 255., 180 / 255.))
    ax1.plot(task.time[0:task.numRecord], task.dcRamp[0:task.numRecord], color=(255 / 255., 127 / 255., 14 / 255.))

ani = animation.FuncAnimation(fig, animate, interval=50)


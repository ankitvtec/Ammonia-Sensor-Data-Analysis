import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as sci
import statistics as stats
from numpy import polyfit, polyval, interp, abs, gradient
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sympy import symbols, diff
import math
from scipy.fft import irfft



def derivative(x,y,order=1):

    if order==1:
        return gradient(y)
    elif order==2:
        second=gradient(gradient(y))
        return second

def second_derivative(x,y):
    # dydx = gradient(y, x)  # First derivative.
    # return gradient(dydx, x)  # Second derivative.
    first_derivative(x,y)

def closest(myList, myNumber):
    return min(myList, key=lambda x: abs(x - myNumber))


def Average(lst):
    return sum(lst) / len(lst)


def volts2Temp(v):
    c1 = 0.001129148
    c2 = 0.000234125
    c3 = 0.0000000876741
    R1 = 7426
    R2 = R1 * (5 / v - 1)
    logR2 = math.log(R2)
    T = (1.0 / (c1 + c2 * logR2 + c3 * logR2 * logR2 * logR2))
    T = T - 273.1
    return T


def temp_to_wl(temp):
    # return 1529.1+(temp)*0.0891
    return 1530.916 + (temp - 16.4) * 0.092


def difference(x, points=1):
    xdiff = [abs(x[n] - x[n - points]) for n in range(points, len(x))]
    for i in range(points):
        xdiff.insert(i, int(0))
    return xdiff


def phase(list1):
    return list(math.degrees(math.acos((i - 160) / 243)) for i in list1)


def phase1(list1):
    return list(math.degrees(math.acos(((i - 60) ** 2 - 2 * (30) ** 2) / 2 / 30 / 30)) for i in list1)
    # return list(math.degrees(math.acos(((i-90)**2-2*(75/2)**2)*2/75*75)) for i in list1)


folder = 'D:\Ankit\Projects\Ammonia Sensor\MZi Setup\Measurements'  # Folder Directory
sweep_filename = '\PumpModulation2024-06-19_16-15-46.csv'  # File name##
data_filename = '\Aurduino_2024_06_19_161553.csv'  #####

data = open(folder + sweep_filename, 'r').readlines()
arduino_data = open(folder + data_filename, 'r').readlines()
start_point = 1  # Choose starting point of plot
start_point_arduino = 5
n_points = -1  # Choose End point of plot ,-1 means whole data
n_points_arduino = -1
#
sample_points = 300

Room_temperature_plot = 0

time_stamp_tempsweep = []  # To store Time stamps
time_stamp_arduino = []
Power = []  # Output mV
V = []  # Room Temperature voltage V
Pavg = []
T_avg = []
Time_arduino = []

# Arduino Data Extraction

try:
    start_time_arduino = arduino_data[0].split()[3] + " " + arduino_data[0].split()[4]
except:
    start_time_arduino = "2024.05.31 16:49:38"
start_timestamp = datetime.timestamp(datetime.strptime(start_time_arduino, "%Y.%m.%d %H:%M:%S"))
print(start_time_arduino)
for j in arduino_data[start_point_arduino:n_points_arduino - 2]:
    # print(len(j))
    try:
        p, dt = j.split()
        v = p
        if abs(float(p)) < 0.55:
            Power.append(abs(float(p)) * 1000)
            if len(Power) == sample_points:
                Pavg.append(Average(Power))
                time_stamp_arduino.append(float(start_timestamp) + float(dt) * 1e-3)
                Power = []
                Time_arduino.append(datetime.fromtimestamp(time_stamp_arduino[-1]))
    except:
        continue
realtime = []  # To store real datetime
Time = []  # To store time in seconds
T_laser = []
temp1 = []
temp2 = []
Time_sweep = []

for i in data[start_point:n_points]:
    t_sec = float(i.split()[1])
    t_las = float(i.split()[-1])
    time_stamp_tempsweep.append(float(i.split()[0]))
    Time.append(t_sec)
    T_laser.append(t_las)
    Time_sweep.append(datetime.fromtimestamp(time_stamp_tempsweep[-1]))

# Start point time for the plot with date
start_time = datetime.fromtimestamp(time_stamp_arduino[0])
# End point time of the plot with date
end_time = datetime.fromtimestamp(time_stamp_arduino[-1])

t_step = (time_stamp_arduino[10] - time_stamp_arduino[1]) / 9

t_new = []
P = []
peaks=[]
Period_cycle=20/5 # Seconds, need to be very accurate
n_cycle = int(Period_cycle * 750 // sample_points) // 2
min_temp = time_stamp_tempsweep[T_laser.index(min(T_laser[0:100]))]
start_cycle = time_stamp_arduino.index(closest(time_stamp_arduino, min_temp))
print("Start Point of Cycle:", start_cycle)
for i in range(len(time_stamp_arduino) // n_cycle - 100):
    p1 = Pavg[start_cycle + i * n_cycle:start_cycle + i * n_cycle + n_cycle]
    t1 = time_stamp_arduino[start_cycle + i * n_cycle:start_cycle + i * n_cycle + n_cycle]
    print(p1)
    P.append(max(p1)-min(p1))
    t_new.append((max(t1)+min(t1))/2)
    peaks.append(max(p1))


# plt.figure(10)
# plt.plot(t_new,P,'-o')
fig, ax = plt.subplots(2, 1, figsize=(16, 11))
ax[0].plot(t_new, peaks, '-o')
ax[0].set_title('Sensor Output over time')
ax[0].set_xlabel('Timestamp)')
ax[0].set_ylabel('Difference voltage between Min & Max (mV)')
ax[0].grid()
ax[1].plot(time_stamp_arduino, Pavg,t_new,peaks,'o')
ax[1].set_title('ADC out over time')
ax[1].set_xlabel('Timestamp')
ax[1].set_ylabel('Voltage(mV)')
ax[1].grid()

Pdiff = difference(Pavg, 1)


# Phase Extraction

# Phase=phase1(Pavg)
# Phase_diff=difference(Phase,1)
# fig,ax=plt.subplots(2,1,figsize=(16,11))
# ax[0].plot(Time_arduino,Pavg)
# ax[0].set_title('Sensor Output over time')
# ax[0].set_xlabel('Time (Day/Hour)')
# ax[0].set_ylabel('Difference voltage (mV)')
#
# ax[1].plot(Time_arduino,Phase_diff)
# ax[1].set_title('Phase difference over time')
# ax[1].set_xlabel('Time (Day/Hour)')
# ax[1].set_ylabel('Phase shift (deg)')
# print("Air Temperature",T_avg)

#
# # Extracting Temperature Sweep Data into arrays


#
t_step_temp = (time_stamp_tempsweep[10] - time_stamp_tempsweep[1]) / 9

N = len(Pavg)
### FFT Analysis

# target_frequency = 0.005
# start_freq = 10
# j = 0
# noise_region = 25000
# yf = rfft(Pavg)[start_freq:]
# xf = rfftfreq(N, t_step)[start_freq:]
# plt.figure(6)
# plt.plot(xf, abs(yf))
# # The maximum frequency is half the sample rate
# points_per_freq = len(xf) / (1/ t_step / 2)
#
# # Our target frequency is 4000 Hz
# target_idx1 = int(points_per_freq * target_frequency)
# yf[target_idx1 - 1 : target_idx1 + noise_region] = 0
# # plt.plot(xf[start_freq:], abs(yf[start_freq:]))
# plt.title('FFT of Original Signal')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('FFT (unit)')
# plt.legend(['Original Signal','Modified Signal'])
# new_sig = irfft(yf)


print('Sweep time step(s):', t_step_temp)
print('Arduino time step(s):', t_step)

# # # Plotting change in power and Air temperature with time
color = 'tab:red'
_, ax3 = plt.subplots(figsize=(15, 8))

ax3.set_xlabel('Time')
ax3.set_ylabel('Sensor Out (mV)', color=color)
# ax3.set_ylim(-1.3,1)
ax3.plot(Time_arduino, Pavg, color=color)
ax3.tick_params(axis='y', labelcolor=color)
# ax3.text(min(time_stamp_arduino) + 10, min(Pavg)-1, f"Start time: {start_time}\nEnd time: {end_time}, Sampling Time: {t_step} s")
#
# Adding Twin Axis for Temperature values
ax4 = ax3.twinx()
color = 'tab:cyan'
ax4.set_ylabel('Laser Temperature (Degrees)', color=color)
ax4.plot(Time_sweep, T_laser, color=color)
# ax4.plot(time_stamp_arduino, T_avg, color=color)
# ax4.hlines(18.24,min(time_stamp_tempsweep),max(time_stamp_tempsweep))
ax4.tick_params(axis='y', labelcolor=color)
ax3.set_title('Michelson Interferometer behaviour over time')


# # corr, P = sci.pearsonr(T_avg, Pavg)
# # print(f"Correlation Coefficient between Air Temperature and Sensor Output: {corr}")



plt.show()

#   EERI 414
#   FIR TEST
#   ADJUSTABLE LOW PASS

# IMPORTS
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pylab as pl
import math  as m
import sympy as sim
from scipy import signal

# READ DATA
sampling_period = 4
time = []; volt = []
counter = 0
with open('ECGdata.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        time.append(counter*sampling_period/1000)
        volt.append(float(row['ECG [V]']))
        counter += 1
# FORMAT DATA
time = np.array(time,dtype=float)  
voltplot = np.array(volt ,dtype=float)  

# PLOT SIGNAL
plt.figure(1)
plt.plot(time[4500:5000],voltplot[4500:5000])
plt.grid('true')
plt.ylabel('Volt [V]')
plt.xlabel('Time [s]')
plt.title('INPUT SIGNAL')

# FILTER SPECIFICATIONS
Fs = 12e03              #Hz
i = 0
INPUT_1 = "calc"
Wp  = 100               #Hz
Ws  = 150               #Hz
Wp2 = 0
Ws2 = 0
wp  = 0.3*np.pi                  #rad/s
ws  = 0.5*np.pi                  #rad/s
wp2 = 0
ws2 = 0
INPUT_2 = "calc"
Ap = 1                   #dB
As = 40                  #dB
Dp = 0.0114469
Ds = 0.01778279

# NORMALIZE EDGE FREQUENCIES
if INPUT_1 == "calc":
    wp = 2 * np.pi * Wp / Fs
    ws = 2 * np.pi * Ws / Fs
    wp2 = 2 * np.pi * Wp2 / Fs
    ws2 = 2 * np.pi * Ws2 / Fs

# CALCULATE delta VALUES
if INPUT_2 == "calc":
    Dp = 1 - 10**(Ap / (-20))
    Ds = 10**(As / (-20))

# BANDWIDTH & CUT-OFF
wc = (ws + wp)/2
wc2 = (ws2 + wp2)/2
Dw = abs(ws - wp)

# CALCULATE APPROXIMATE FILTER ORDER
# KAISER 
N = abs(np.ceil((As - 8)/(2.285*Dw)))
if (N % 2) != 0:
    N += 1

# KAISER CALCULATIONS 
M = np.int(N/2)
if As > 50:
    beta = 0.1102*(As - 8.7)
elif (As >= 21) & (As <= 50):
    beta = 0.5842*(As - 21)**0.4 + 0.07886*(As -21)
elif As < 21:
    beta = 0

def WINDOW_SUM(coeff):
    ANSWER = 0
    for counter2 in range(20):
        ANSWER += ((coeff/2)**(counter2+1)/(m.factorial(counter2+1)))**2
    ANSWER += 1
    return(ANSWER)

# SETUP WINDOW SIZE
window = np.zeros(2 * M + 1)
ideal  = np.zeros(2 * M + 1)
final  = np.zeros(2 * M + 1)
index  = np.zeros(2 * M + 1) 

# GENERATE FILTER TRANSFER FUNCTION
k = 0
sum = 0
n = -1 * M

while n <= M:
    # KAISER WINDOW
    window_num_coeff = beta*np.sqrt(1-(n/M)**2)
    window_den_coeff = beta
    window_num_sum = WINDOW_SUM(window_num_coeff)
    window_den_sum = WINDOW_SUM(window_den_coeff)
    window[k] = (window_num_sum)/(window_den_sum)
    if n==0:
        # LOW PASS
        ideal[k] = (wc / np.pi)
        # HIGH PASS
        #ideal[k] = 1 - (wc / np.pi)
        # BANDPASS
        #ideal[k] = (wc / np.pi) - (wc / np.pi)
        # BAND STOP
        #1 - ((wc2 - wc) / np.pi)
    else:
        # LOW PASS
        ideal[k] = np.sin(wc * n) / (np.pi * n)
        # HIGH PASS
        #ideal[k] = -np.sin(wc * n) / (np.pi * n)
        # BANDPASS
        #ideal[k] = (np.sin(wc2 * n) / (np.pi * n)) - (np.sin(wc * n) / (np.pi * n))
        # BAND STOP
        #ideal[k] = (np.sin(wc * n) / (np.pi * n)) - (np.sin(wc2 * n) / (np.pi * n))
    n += 1
    k += 1
final = ideal * window

# TRANSFER FUNCTION
numerator = final
denominator = 1

# STEP & IMPULSE RESPONSE OF FILTER
def impz(b,a=1):
 l = len(b)
 impulse = np.repeat(0.,l); impulse[0] =1.
 x = np.arange(0,l)
 counter = 1
 response  = np.zeros(l)
 while counter < (len(b)-1):
     response[counter] = b[l-counter] * 1
     counter += 1
 plt.figure(2)
 plt.subplot(2,1,1)
 plt.stem(x, response, use_line_collection=True)
 plt.ylabel('Amplitude')
 plt.xlabel('Samples')
 plt.title('Impulse Response')
 plt.subplot(2,1,2)
 step =  np.cumsum(response)
 plt.stem(x,step, use_line_collection=True)
 plt.ylabel('Amplitude')
 plt.xlabel('Samples')
 plt.title('Step Response')
 plt.subplots_adjust(hspace=0.5)
impz(final)

# FREQUENCY RESPONSE OF FILTER
w, h = sp.signal.freqz(numerator,denominator)
gain = abs(h)
gaindB = 20*np.log10(gain)
phase_rad = np.unwrap(np.angle(h))

# FREQUENCY MARKERS
index = np.where(w >= wc)
cutoff_index = index[0][0]
index = np.where(w >= wp)
pass_index = index[0][0]
index = np.where(w >= ws)
stop_index = index[0][0]

# DENORMALIZE
w = (w * Fs) / (2 * np.pi)

# MAGNITUDE RESPONSE
plt.figure(3)
plt.subplot(2,1,1)
plt.plot(w,gaindB)
plt.grid('true')
plt.ylabel('Gain [dB]')
plt.title('Bode Plot')
# MARKERS
plt.plot(w[pass_index], gaindB[pass_index], 'ro')
plt.plot(w[stop_index], gaindB[stop_index], 'ro')
plt.plot(w[cutoff_index], gaindB[cutoff_index], 'r*')
# PHASE RESPONSE
plt.subplot(2,1,2)
plt.plot(w,phase_rad)
plt.grid('true')
plt.ylabel('Phase [rad]')
plt.xlabel('Normalised Freq [Hz]')

# APPLY FILTER
OutputSignal = []
OutputSignal = np.array(OutputSignal,dtype=float)
InputSignal = np.array(volt,dtype=float)
InputSignal = np.pad(InputSignal, (0,len(final)),'constant')
final = np.pad(final, (0,len(InputSignal)-len(final)),'constant')
OutputSignal = np.convolve(final,InputSignal,mode='full')
l = len(OutputSignal)
impulse = np.repeat(0.,l); impulse[0] =1.
x = np.arange(0,l)
plt.figure(4)
plt.plot(time[4500:5000], OutputSignal[4500:5000])
plt.ylabel('Volt [V]')
plt.xlabel('Time [s]')
plt.grid('true')
plt.title('Filtered Signal')

# PROGRAM OUTPUT
print(f'\n\n------------------------- PROGRAM OUTPUT -----------------------------')
print(f'------------------------- FREQUENCY INFORMATION ----------------------')
if INPUT_1 == "calc":
    print(f'  low pass passband edge : {Wp} Hz --> {wp} rad/s')
    print(f'  low pass stopband edge : {Ws} Hz --> {ws} rad/s')
else:
    print(f'           passband edge : {wp} rad/s')
    print(f'           stopband edge : {ws} rad/s')
print(f'------------------------- ATTENUATION INFORMATION --------------------')
if INPUT_2 == "calc":
    print(f'         passband ripple : {Ap} dB --> {Dp}')
    print(f'         stopband ripple : {As} dB --> {Ds}')
else:
    print(f'         passband ripple : {Dp}')
    print(f'         stopband ripple : {Ds}')

print(f'------------------------- FILTER INFORMATION -------------------------')
print(f'         Kaiser : {M} & {N}\n\n')

# Show Plots
plt.show()
#   EERI 414
#   FIR TEST
#   STATIC BAND STOP

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
Fs = 2e03              #Hz
i = 0
INPUT_1 = "calc"
Wp  = 10               #Hz
Ws  = 50               #Hz
Wp2 = 140
Ws2 = 100
wp  = 0                  #rad/s
ws  = 0                  #rad/s
wp2 = 0
ws2 = 0
INPUT_2 = "calc"
Ap = 0.1                 #dB
As = 35                  #dB
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
Nk = np.ceil((-20 * np.log10((Dp * Ds)**(0.5)) - 13) / (14.6 * (ws - wp)/(2 * np.pi)))
if (Nk % 2) != 0:
    Nk += 1
# BELLANGER
Nb = np.ceil((-2 * np.log10(10 * Dp * Ds) / (3 * (ws - wp) / (2 * np.pi))) - 1)
if (Nb % 2) != 0:
    Nb += 1
# HERMAN
if Dp < Ds:
    i = Dp
    Dp = Ds
    Ds = i
a1 = 0.005309 
a2 = 0.07114
a3 = -0.4761
a4 = 0.00266
a5 = 0.5941
a6 = 0.4278
b1 = 11.01217
b2 = 0.51244
F = b1 + b2*(np.log10(Dp) - np.log10(Ds))
Dinf = np.log10(Ds)*(a1*(np.log10(Dp))**2 + a2*(np.log10(Dp)) + a3) - (a4*(np.log10(Dp))**2 + a5*(np.log10(Dp)) + a6)
Nh = np.ceil((Dinf - F*((ws - wp)/(2*np.pi))**2)/((ws - wp)/(2*np.pi)))
if (Nh % 2) != 0:
    Nh += 1
if i == Ds:
    i = Dp
    Dp = Ds
    Ds = i

# CALCULATE WINDOW SIZE
Mhan = np.ceil(3.11 * np.pi / Dw)
Mham = np.ceil(3.32 * np.pi / Dw)
Mbl  = np.ceil(5.56 * np.pi /Dw)

# SET DESIRED WINDOW
M = np.int(Mbl)

# SETUP WINDOW SIZE
window = np.zeros(2 * M + 1)
ideal  = np.zeros(2 * M + 1)
final  = np.zeros(2 * M + 1)
index  = np.zeros(2 * M + 1)

# GENERATE FILTER TRANSFER FUNCTION
k =0
n = -1 * M
while n <= M:
    # RECTANGUAR WINDOW
    #window[k] = 1
    # HANN WINDOW
    #window[k] = (1/2)*(1 + np.cos(np.pi * n / M))
    # HAMMING WINDOW
    #window[k] = 0.54 + 0.46*np.cos(np.pi * n / M)
    # BARLETT WINDOW
    #window[k] = 1 - abs(n)/(M + 1)
    # BLACKMAN
    window[k] = 0.42 + 0.5 * np.cos(np.pi * n / M) + 0.08 * np.cos(2 * np.pi * n / M)
    if n==0:
        # LOW PASS
        #ideal[k] = (wc / np.pi)
        # HIGH PASS
        #ideal[k] = 1 - (wc / np.pi)
        # BANDPASS
        #ideal[k] = (wc / np.pi) - (wc / np.pi)
        # BAND STOP
        ideal[k] =1 - ((wc2 - wc) / np.pi)
    else:
        # LOW PASS
        #ideal[k] = np.sin(wc * n) / (np.pi * n)
        # HIGH PASS
        #ideal[k] = -np.sin(wc * n) / (np.pi * n)
        # BANDPASS
        #ideal[k] = (np.sin(wc2 * n) / (np.pi * n)) - (np.sin(wc * n) / (np.pi * n))
        # BAND STOP
        ideal[k] = (np.sin(wc * n) / (np.pi * n)) - (np.sin(wc2 * n) / (np.pi * n))
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
Wc = (wc * Fs) / (2 * np.pi)
Wc2 = (wc2 * Fs) / (2 * np.pi)

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
plt.rcParams['lines.linestyle'] = '--'
plt.axvline(x=Wp,color='r')
plt.axvline(x=Wc,color='r')
plt.axvline(x=Ws,color='r')
plt.axvline(x=Wp2,color='b')
plt.axvline(x=Wc2,color='b')
plt.axvline(x=Ws2,color='b')
plt.rcParams['lines.linestyle'] = '-'
# PHASE RESPONSE
plt.subplot(2,1,2)
plt.plot(w,phase_rad)
plt.grid('true')
plt.ylabel('Phase [rad]')
plt.xlabel('Normalised Freq [Hz]')
# MARKERS
plt.rcParams['lines.linestyle'] = '--'
plt.axvline(x=Ws,color='r')
plt.axvline(x=Ws2,color='b')
plt.rcParams['lines.linestyle'] = '-'

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
    print(f' high pass passband edge : {Wp2} Hz --> {wp2} rad/s')
    print(f' high pass stopband edge : {Ws2} Hz --> {ws2} rad/s')
else:
    print(f'           passband edge : {wp} rad/s')
    print(f'           stopband edge : {ws} rad/s')
    print(f'           passband edge : {wp2} rad/s')
    print(f'           stopband edge : {ws2} rad/s')
print(f'------------------------- ATTENUATION INFORMATION --------------------')
if INPUT_2 == "calc":
    print(f'         passband ripple : {Ap} dB --> {Dp}')
    print(f'         stopband ripple : {As} dB --> {Ds}')
else:
    print(f'         passband ripple : {Dp}')
    print(f'         stopband ripple : {Ds}')
print(f'------------------------- APPROXIMATE FILTER ORDER -------------------')
print(f'               Kaiser    : {Nk}')        
print(f'               Bellanger : {Nb}')
print(f'               Herman    : {Nh}')
print(f'------------------------- FILTER INFORMATION -------------------------')
print(f'                    Hann : {Mhan}')
print(f'                 Hamming : {Mham}')
print(f'                Blackman : {Mbl}\n\n')

# Show Plots
plt.show()
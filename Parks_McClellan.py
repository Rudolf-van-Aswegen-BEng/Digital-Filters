#   EERI 414
#   FIR TEST
#   PARKS McCLELLAN

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
# plt.figure(1)
# plt.plot(time[4500:5000],voltplot[4500:5000])
# plt.grid('true')
# plt.ylabel('Volt [V]')
# plt.xlabel('Time [s]')
# plt.title('INPUT SIGNAL')

# FILTER SPECIFICATIONS
Fs = 10e03               #Hz
i = 0
INPUT_1 = "calc"
Wp  = 2000               #Hz
Ws  = 2500               #Hz
Wp2 = 0
Ws2 = 0
wp  = 0                  #rad/s
ws  = 0                  #rad/s
wp2 = 0
ws2 = 0
INPUT_2 = "calc"
Ap = 0.1                #dB
As = 30                  #dB
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
Nk = abs(np.ceil((-20 * np.log10((Dp * Ds)**(0.5)) - 13) / (14.6 * (ws - wp)/(2 * np.pi))))
if (Nk % 2) != 0:
    Nk += 1
# BELLANGER
Nb = abs(np.ceil((-2 * np.log10(10 * Dp * Ds) / (3 * (ws - wp) / (2 * np.pi))) - 1))
if (Nb % 2) != 0:
    Nb += 1
# HERMAN
if Dp < Ds:
    i  = Dp
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
F  = b1 + b2*(np.log10(Dp) - np.log10(Ds))
Dinf = np.log10(Ds)*(a1*(np.log10(Dp))**2 + a2*(np.log10(Dp)) + a3) - (a4*(np.log10(Dp))**2 + a5*(np.log10(Dp)) + a6)
Nh = abs(np.ceil((Dinf - F*((ws - wp)/(2*np.pi))**2)/((ws - wp)/(2*np.pi))))
if (Nh % 2) != 0:
    Nh += 1
if i == Ds:
    i = Dp
    Dp = Ds
    Ds = i

print(f'{Nk}')
print(f'{Nb}')
print(f'{Nh}')

# FINALIZE FILTER ORDER
if (Nh < Nb) & (Nh > Nk):
    N = Nh
else:
    N = (Nk + Nb)/2
    if (N % 2) != 0:
        N += 1

N = Nk

# WINDOW CALCULATION
M = np.int(N/2)

# CHOOSE K VALUES
Kp = 1
Ks = 1

# WEIGHT FUNCTION 
# Set up axis
axis_scale = 1000
w          = np.arange(0,np.pi,np.pi/axis_scale)
weights    = np.zeros(axis_scale)
desired    = np.zeros(axis_scale)

# GENERATE WEIGHT FUNCTION 
# LOW PASS
weights[np.where(w <= wp)] = Kp
weights[np.where(w >= ws)] = Ks
desired[np.where(w <= wc)] = 1
# HIGH PASS
# weights[np.where(w >= wp)] = Kp
# weights[np.where(w <= ws)] = Ks
# desired[np.where(w >= wc)] = 1
# BAND PASS
# BAND STOP
index     = np.nonzero( weights > 0)
Nindex    = np.count_nonzero(index)
increment = np.intc(np.floor(Nindex/(M + 2)))
counter1  = 0
loops     = 0
indices = np.zeros(M + 2)
while (counter1 < Nindex) & (loops < (M + 2)):
    indices[loops] = index[0][counter1]
    counter1 += increment
    loops += 1
indices = np.intc(indices)

# PLOT WEIGHT FUNCTION
plt.figure(2)
plt.rcParams['lines.linestyle'] = '--'
plt.plot(w,desired,'b',label='Desired')
plt.rcParams['lines.linestyle'] = '-'
plt.plot(w,weights,label='Weighted')
plt.grid('true')
plt.ylabel('Weight [W] & Tanget [D]')
plt.xlabel('Normalized Frequency [rad/s]')
plt.legend()

# ITERATIVE ALGORITHM
m = np.arange(0,M + 1)
r = np.arange(0,M + 2)
stop = 1
mainloop = 0
while (stop == 1):
    # SET UP LINEAR EQUEATIONS
    mat = np.zeros((M + 2 , M + 2))
    D = np.zeros((M + 2,))
    for row in range(0,M + 2):
        mat[row,0] = 1
        for col in range(1,M + 2):
            mat[row,col] = np.cos(w[indices[row]]*col)
        mat[row,M + 1] = ((-1)**row)/weights[indices[row]]
        D[row] = desired[indices[row]]
    
    # SOLVE LINEAR EQUATIONS
    coeffiients = np.linalg.solve(mat,D)
    delta = coeffiients[M + 1]

    # ERROR FUNCTION
    H = np.zeros(axis_scale)
    E = np.zeros(axis_scale)
    for n in range(0,axis_scale-1):
        sum = 0
        for k in range(0,M+1):
            sum += coeffiients[k]*np.cos(k*w[n])
        H[n] = sum
        E[n] = (H[n] - desired[n])*weights[n]
    if mainloop == 0:
        plt.figure(3)
        plt.plot(w,E,label=mainloop)
        plt.figure(4)  
        plt.plot(w,H,label=mainloop)    
    
    # UPDATE CANDATE EXTREMAL POINTS
    peaks, troughs = signal.find_peaks(abs(E))
    if (peaks[0] != 0):
        peaks = np.hstack(([0],peaks))
    if (len(peaks) > (M+2)):
        peaks = peaks[0:(M+2)]
    signs = np.sign(E[peaks])
    indices = peaks

    mainloop += 1

    if mainloop == 5:
         stop = 2

# PLOT ERROR CURVE
plt.figure(3)
BOTTOM = np.ones(axis_scale)*E[peaks[1]]
TOP = np.ones(axis_scale)*E[peaks[0]]
plt.rcParams['lines.linestyle'] = '--'
plt.plot(w,BOTTOM,'g')
plt.plot(w,TOP,'g')
plt.rcParams['lines.linestyle'] = '-'
plt.plot(w,E,label=mainloop)
plt.grid('true')
plt.ylabel('Error Function')
plt.xlabel('Normalized Frequency [rad/s]')
plt.title('E')
plt.legend()

plt.figure(4)
plt.plot(w,H,label=mainloop) 
plt.rcParams['lines.linestyle'] = '--'
plt.plot(w,desired,'b',label='Desired')
plt.grid('true')
plt.ylabel('Error Function')
plt.xlabel('Normalized Frequency [rad/s]')
plt.title('H')
plt.legend()

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
print(f'------------------------- APPROXIMATE FILTER ORDER -------------------')
print(f'               Kaiser    : {Nk}')        
print(f'               Bellanger : {Nb}')
print(f'               Herman    : {Nh}')
print(f'------------------------- FILTER INFORMATION -------------------------')
print(f'         Parks McClellan : {mainloop}')

# Show Plots
plt.show()
 
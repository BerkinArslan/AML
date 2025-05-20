import matplotlib
from matplotlib import pyplot as plt
import numpy as np
#matplotlib.use('TkAgg')

def plotting(X,Y):
    plt.plot(X, Y, color = 'r', linewidth = 2)
    plt.show()

def signal(t):
    return np.pi *  np.cos(10 * t)

time_intervall = list(np.arange(0, 8.01, 0.01))
time_intervall = list(np.linspace(0, 8, 801))

Signal = []
l = len(time_intervall)
for i in range(l):
    Signal.append(signal(time_intervall[i]))

#plotting(time_intervall, Signal)

def signal2(f, Amplitude):
    t =np.arange(0, 8.01, 0.01)
    values = Amplitude * np.cos(2 * np.pi * f * t)
    return values

#plotting(time_intervall, signal2(3, 2))

####Implementing the newtons method####

def function1(t, f = 2, A = 3):
    return A *  np.cos(f* np.pi * t)


def numarical_derivation_easy(function, x, step = 0.00001):
    df = (function(x + step) - function(x))/ step
    return df

def newtons_method(function, start):
    next = start - (function(start)/numarical_derivation_easy(function, start))
    if abs(next - start) >= 1e-4 or abs(function(start)) >= 1e-7:
        return newtons_method(function, next)
    return start

#print(newtons_method(lambda x: x**2 - 2, 2))




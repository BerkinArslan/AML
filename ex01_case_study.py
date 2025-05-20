import sklearn
from matplotlib import pyplot as plt
import numpy as np
from exercise01 import lin_regression as linreg
from my_r2_score import r2_score as r2

#Rolling resistance estimation


#Glolbar VARs
CW = 0.4
A = 1.5 #m^2
RHO_AIR = 1.2 #kg/m^3
G = 9.81 #m/s^2
M_VECHICLE = 2400 #kg

data = np.genfromtxt('driving_data.csv', delimiter = ',')

def calculate_wind_res(v: np.ndarray) -> np.ndarray:
    Res = (CW * A * RHO_AIR * v ** 2) / 2
    return Res

def calculate_resistance_coeff(Fw: np.ndarray, P: np.ndarray,
                               v: np.ndarray) -> np.ndarray:
    cw = (P / v) - Fw
    return cw

def estimate_rolling_resistance_coeff(v: np.ndarray, P: np.ndarray) -> np.ndarray:
    Fw = calculate_wind_res(v)
    cw = calculate_resistance_coeff(Fw, P, v)
    theta_0 , theta_1 = linreg(v, cw)

    plt.scatter(v, cw)
    plt.plot(v, theta_0 + theta_1 * v, color = 'red')
    plt.show()

v = data[:, 0]
P = data[:, 1]
estimate_rolling_resistance_coeff(v, P)

def estimate_rolling_resistance(v: np.ndarray, P: np.ndarray) -> np.ndarray:
    Fw = calculate_wind_res(v)
    theta_0, theta_1 = linreg(P, Fw)

    plt.scatter(P, Fw)
    plt.plot(P, theta_0 + theta_1 * P, color = 'red')
    plt.show()

estimate_rolling_resistance(v, P)

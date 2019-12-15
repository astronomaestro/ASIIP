#if you want to calculate an error for your telescope instrument, you can use this script to assist you in doing so
#it returns the value you need for the parameter 'sigmaTel'

import numpy as np

#The magnitude of the target you are measuring
m = 1.9
#Your integration time in seconds
T = 1800
#the electronic bandwidth of your detector in Hz
electronic_bandwidth = 100 * 10**6
#The total area of your telescope in meters
v_area = 6**2 * np.pi
#The quantum efficiency associated with the detector
quant_eff = .1
#A calibration constant determined empirically
C = 2.05



def tel_err(area, quant_eff, bandwidth, m, T, C):
    """
    This is a simple function to assist with the calcuation of sigmaTel used in ASIIP's input parameter file
    :param area: The area of the telescope
    :param quant_eff: The quantum efficiency of the detector
    :param bandwidth: The bandwidth of your detector
    :param m: The magnitude of your target
    :param T: The integration time of the target
    :param N: The number of baselines in your array
    :param C: The calibration constant
    :return: The calculated error
    """
    #a constant associated with calculation the spectral densiy
    n0 = 5e-5
    #the spectral density in uniits of m^-2*s^-1*Hz^-1
    spec_dens = n0 * 2.5**(-m)

    calculated_error = 1/(area*quant_eff*spec_dens) * np.sqrt(2/(bandwidth*T)) * C

    print("For the given input parameters\n"
          "Telescope area = %s\n"
          "Quantum Efficiency = %s\n"
          "Star Magnitude = %s\n"
          "Integration Time = %s\n"
          "The error for your telescope is %.3f" % (area, quant_eff, m, T, calculated_error))

    return calculated_error

tel_err(v_area, quant_eff, electronic_bandwidth, m, T, C)



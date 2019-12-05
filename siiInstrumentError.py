import numpy as np

#The magnitude of the target you are measuring
m = 1.7
#Your integration time
T = 800
#the electronic bandwidth of your detector
electronic_bandwidth = 600
#The area of your telescope in meters
v_area = 6**2 * np.pi
#The quantum efficiency associated with the detector
quant_eff = 1



def tel_err(area, quant_eff, bandwidth, m, T):
    """
    This is a simple function to assist with the calcuation of sigmaTel used in ASIIP's input parameter file
    :param area: The area of the telescope
    :param quant_eff: The quantum efficiency of the detector
    :param bandwidth: The bandwidth of your detector
    :param m: The magnitude of your target
    :param T: The integration time of the target
    :return: The calculated error
    """
    #a constant associated with calculation the spectral densiy
    n0 = 5e-5
    #the spectral density
    spec_dens = n0 * 2.5**(-m)

    calculated_error = 1/(area*quant_eff*spec_dens) * np.sqrt(2/(bandwidth*T))

    print("For the given input parameters\n"
          "Telescope area = %s\n"
          "Quantum Efficiency = %s\n"
          "Star Magnitude = %s\n"
          "Integration Time = %s\n"
          "The error for your telescope is %.2f" % (area, quant_eff, m, T, calculated_error))

    return calculated_error

tel_err(v_area, quant_eff, electronic_bandwidth, m, T)



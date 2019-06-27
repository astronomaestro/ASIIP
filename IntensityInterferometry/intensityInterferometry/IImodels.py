import numpy as np
from astropy.modeling.models import custom_model
import astropy.visualization as viz

norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())




def sine_model(x, amplitude=1., frequency=1.):
    return amplitude * np.sin(2 * np.pi * frequency * x)
def sine_deriv(x, amplitude=1., frequency=1.):
    return 2 * np.pi * amplitude * np.cos(2 * np.pi * frequency * x)
SineModel = custom_model(sine_model, fit_deriv=sine_deriv)

def cust_airy_disk2D(x, y, x_0, y_0, theta, wavelength):
    from scipy.special import j1, jn_zeros
    rz = jn_zeros(1, 1)[0] / np.pi

    radius = 1.22 * wavelength / theta

    r = np.sqrt((x - x_0) ** 2 + (y - y_0) ** 2) / (radius / rz)

    z = np.ones(r.shape)
    rt = np.pi * r[r > 0]
    z[r > 0] = (2.0 * j1(rt) / rt) ** 2

def airy_disk2D(shape, xpos, ypos, arcsec, wavelength):
    from astropy.modeling.functional_models import AiryDisk2D
    from astropy.modeling import fitting

    r = 1.22 * wavelength.to('m').value / arcsec.to('rad').value

    y, x = np.mgrid[:shape[0], :shape[1]]
    # Fit the data using astropy.modeling
    airy_init = AiryDisk2D(x_0=xpos, y_0=ypos, radius=r)
    fit_p = fitting.LevMarLSQFitter()
    return airy_init(x,y), airy_init

def binary_visibility2D(shape, flux_ratio, separation, wavelength, arcsec1, arcsec2):

    y, x = np.mgrid[:shape[0], :shape[1]]
    xpos = shape[0]/2
    ypos = shape[1]/2
    met_wav = wavelength.to('m').value
    sep_rad = separation.to('rad').value

    cen_offset = 1.22*met_wav/sep_rad/2

    V_1, v1func = airy_disk2D(shape, xpos, ypos, arcsec1, wavelength)
    V_2, v2func = airy_disk2D(shape, xpos, ypos, arcsec2, wavelength)





    v1_v2_term = V_1**2 + (flux_ratio*V_2)**2
    abs_term = 2*flux_ratio*np.abs(V_1)*np.abs(V_2)
    cos_term = np.cos(2*np.pi/met_wav * 0*np.cos(0*sep_rad))
    result = v1_v2_term * abs_term * cos_term / (1 + flux_ratio) ** 2
    return result

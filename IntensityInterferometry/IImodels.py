import numpy as np
from astropy.modeling.models import custom_model
import astropy.visualization as viz
from IntensityInterferometry import IImodels, IIdisplay, IItools, IIdata
from matplotlib import pyplot as plt

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

    r = wavelength.to('m').value / arcsec.to('rad').value

    y, x = np.mgrid[:shape[0], :shape[1]]
    # Fit the data using astropy.modeling
    airy_init = AiryDisk2D(x_0=xpos, y_0=ypos, radius=r)
    fit_p = fitting.LevMarLSQFitter()
    return airy_init(x,y), airy_init

def airy1D(xr, r):
    from scipy.special import j1
    con = 1.2196698912665045
    airy_mod = (2*j1(con*np.pi*xr/r) / (np.pi * xr * con/r))**2
    return airy_mod


def fit_airy1D(rx,airy_amp, guess_r, errs, bounds=(-np.inf, np.inf)):
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(f=airy1D,
              xdata=rx,
              ydata=airy_amp,
              p0=[guess_r],
              sigma=errs,
              bounds=bounds,
              maxfev=1000,
              absolute_sigma=True)

    return popt, pcov

def airy_disk1D_Integration(tel_tracks, airy_func, err):
    amps = []
    rads = []
    Ints = []
    Irads = []
    aerrs = []
    xerrs = []
    x_0 = airy_func.x_0.value
    y_0 = airy_func.y_0.value
    for i, track in enumerate(tel_tracks):
        utrack = track[0][:, 0] + x_0
        vtrack = track[0][:, 1] + y_0
        amps.append(airy_func(utrack, vtrack))
        rads.append(np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2))
        airy_I, trap_err, Irad = IItools.trap_w_err(amps[i], rads[i], err, err)
        Ints.append(airy_I/Irad)
        Irads.append(.5*Irad + rads[i][:-1])
        aerrs.append(np.ones(np.alen(Ints[i])) * np.random.normal(0,err, np.alen(airy_I)))
        xerrs.append(Irad)
    return np.ravel(amps), np.ravel(rads), np.ravel(Ints), np.ravel(Irads), np.ravel(aerrs), np.ravel(xerrs)

def avg_air1D(tel_tracks, airy_func, err):
    amps = []
    rads = []
    avg_amps = []
    avg_rads = []
    x_0 = airy_func.x_0.value
    y_0 = airy_func.y_0.value
    for i, track in enumerate(tel_tracks):
        utrack = track[0][:, 0] + x_0
        vtrack = track[0][:, 1] + y_0
        amps.append(airy_func(utrack, vtrack))
        rads.append(np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2))
        airy_avg = IItools.trapezoidal_average(num_f=amps[i])
        avg_rad = IItools.trapezoidal_average(num_f=rads[i])
        avg_amps.append(airy_avg)
        avg_rads.append(avg_rad)
    return np.array(rads), np.array(amps), np.array(avg_rads), np.array(avg_amps)


def fit_airy_avg(rads, amps, avg_rads, avg_amps, func, err, guess_r, real_r):

    from scipy.optimize import curve_fit
    from scipy.optimize import least_squares
    def airy_avg(xr,r):
        mod_Int = np.array([IItools.trapezoidal_average(airy1D(rad, r)) for rad in rads])
        return mod_Int.ravel()


    # x = np.linspace(0, 100, 100000)
    # n=5
    # plt.figure(figsize=(24, 18))
    # plt.plot(x, airy1D(x, guess_r))
    # plt.plot(avg_rads.ravel(), airy_avg(rads,guess_r).ravel(), 'o')
    # plt.plot(rads.ravel(), amps.ravel(), 'o')

    sigmas = np.full(np.alen(avg_amps.ravel()), err)
    try:
        smartFit, serr = curve_fit(f=airy_avg,
                                   xdata=avg_rads.ravel(),
                                   ydata=avg_amps.ravel(),
                                   p0=[guess_r],
                                   sigma=sigmas,
                                   absolute_sigma=True,
                                   maxfev=1000)
    except:
        smartFit, serr = np.nan
    try:
        avgp, avgerr = curve_fit(f=airy1D,
                                 xdata=avg_rads.ravel(),
                                 ydata=avg_amps.ravel(),
                                 p0=[guess_r],
                                 sigma=sigmas,
                                 absolute_sigma=True)
    except:
        avgp, avgerr = np.nan


    # print("integralFit%s, airyFit%s, actual_r%s"%(smartFit[0]/real_r, avgp[0]/real_r, real_r))

    return smartFit, serr, avgp, avgerr, sigmas


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

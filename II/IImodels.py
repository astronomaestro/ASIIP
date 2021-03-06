import numpy as np
from astropy.modeling.models import custom_model
import astropy.visualization as viz
from II import IImodels, IIdisplay, IItools, IIdata
from scipy.optimize import curve_fit
from scipy.special import j1, jn_zeros

from matplotlib import pyplot as plt

#norm = viz.ImageNormalize(1, stretch=viz.SqrtStretch())





def airy_disk2D(shape, xpos, ypos, angdiam, wavelength):
    """
    Function used to created an appropriate 2D airy function according to the parameters passed in. Astropy defines
    r to be 1.22*lambda/angdiam which is why it is defined as so in this function to be consistent with the rest of
    the models.
    :param shape: The size of the data structure used to store the 2D airy disk
    :param xpos: The x position of the center of the airy disk
    :param ypos: The y position of the center of the airy disk
    :param angdiam: The angular diameter of the target you are modeling
    :param wavelength: The wavelength of the filter you are using
    :return:
    """
    from astropy.modeling.functional_models import AiryDisk2D
    from astropy.modeling import fitting

    con = jn_zeros(1, 1)[0] / np.pi
    r = con * wavelength / angdiam

    y, x = np.mgrid[:shape[0], :shape[1]]
    # Fit the data using astropy.modeling
    airy_init = AiryDisk2D(x_0=xpos, y_0=ypos, radius=r)
    fit_p = fitting.LevMarLSQFitter()
    return airy_init(x,y), airy_init

def airy1D(xr, r):
    """
    A function used to model a 1D Visibility curve of an airy disk
    :param xr: The radius of your airy disk
    :param r: the radius of the first zero of the airy disk
    :return:
    """
    con = jn_zeros(1,1)[0]/np.pi
    # con = 1
    bessel_1 = 2*j1(con * np.pi * xr / r)
    bessle_norm = (np.pi * xr * con / r)

    if np.any(xr == 0):
        both_zero = np.where((bessel_1 == 0) & (bessle_norm == 0))
        bessel_1[both_zero] = 1
        bessle_norm[both_zero] = 1
    airy_mod = (bessel_1 / bessle_norm) ** 2

    return airy_mod

def visibility2dTo1d(tel_tracks, visibility_func, x_0, y_0):
    """
    Take the tracks generated from a 2D airy disk curve and convert them into a 1D airy disk curve
    :param x_0: The x cfoordinate 0 point of the visibility model
    :param y_0: The y coordinate 0 point of the visibility model
    :param tel_tracks: The tracks from the 2D airy disk
    :param visibility_func: The 2D visibility function used to generate the tracks
    :return: The radial x values, The amplitude at those radii, average integrated radial x values, the average
    integrated amplitude at those raddi
    """
    amps = []
    rads = []
    avg_amps = []
    avg_rads = []

    for i, track in enumerate(tel_tracks):
        try:
            utrack = track[0][:, 0] + x_0
        except:
            adsf=34
        vtrack = track[0][:, 1] + y_0
        amps.append(visibility_func(utrack, vtrack))
        rads.append(np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2))
        airy_avg = IItools.trapezoidal_average(num_f=amps[i])
        avg_rad = IItools.trapezoidal_average(num_f=rads[i])
        avg_amps.append(airy_avg)
        avg_rads.append(avg_rad)
    return np.array(rads), np.array(amps), np.array(avg_rads), np.array(avg_amps)


def fit_airy_avg(rads, avg_rads, avg_amps, err, guess_r):
    """
    Takes a 1D airy function, integrates it, then fits that integration to avg_amps. Since we are not actually fitting
    an actual airy function, put a piecewise integrated average of an airy function, but need to know what the true
    airy function is, I needed curve_fit to fit an integrated airy function which came from an actual airy function. I
    accomplished this by defining the x averages outside of the fitted function and then constructing the y values
    within the fitted function so as to keep the shape of the fitting arrays the same. A bit of a hack, but it works.
    :param rads: The rads used to constructed the airy function which will then be integrated
    :param avg_rads: The x values used in fitting the integrated airy function
    :param avg_amps: The y values used in fitting the integrated airy function
    :param err: The err passed into curve_fit
    :param guess_r: the initial guess diameter
    :return: The fitted parameters, the error of that fit, the constructed error array
    """

    #This slightly hacky piece of code allows for the construction of a piecewise integration function. It's one of the
    #more complex parts, but is extremely important if accurate fits are to be made, especially with more complicated
    #visibility models.
    def airy_avg(xr,r):
        mod_Int = np.array([IItools.trapezoidal_average(airy1D(rad, r)) for rad in rads])
        return mod_Int.ravel()

    sigmas = np.full(len(avg_amps.ravel()), err)
    smartFit, serr = curve_fit(f=airy_avg,
                               xdata=avg_rads.ravel(),
                               ydata=avg_amps.ravel(),
                               p0=[guess_r],
                               sigma=sigmas,
                               absolute_sigma=True,
                               maxfev=1500)
    return smartFit, serr, sigmas

def fit_airy(avg_rads, avg_amps, err, guess_r):
    """
    This function uses curve_fit to fit normally to fit a set of integrated averages directly to an actualy airy
    function. Not as cool or accurate as fitting the integration, but uses curve_fit more normally
    :param avg_rads:
    :param avg_amps:
    :param err:
    :param guess_r:
    :return:
    """
    sigmas = np.full(len(avg_amps.ravel()), err)

    avgp, avgerr = curve_fit(f=airy1D,
                             xdata=avg_rads.ravel(),
                             ydata=avg_amps.ravel(),
                             p0=[guess_r],
                             sigma=sigmas,
                             absolute_sigma=True,
                             maxfev=1500)
    return avgp, avgerr

def binary_visibility2D(shape, flux_ratio, separation, wavelength, arcsec1, arcsec2):

    y, x = np.mgrid[:shape[0], :shape[1]]
    xpos = shape[0]/2
    ypos = shape[1]/2
    met_wav = wavelength.to('m').value
    sep_rad = separation.to('rad').value

    cen_offset = 1.22*met_wav/sep_rad/2

    # V_1, v1func = airy_disk2D(shape, xpos, ypos, arcsec1, met_wav)
    # V_2, v2func = airy_disk2D(shape, xpos, ypos, arcsec2, met_wav)
    #
    # norm = viz.ImageNormalize(1, stretch=viz.LogStretch())
    #
    # v1_v2_term = V_1 ** 2 + (flux_ratio * V_2) ** 2
    # abs_term = 2 * flux_ratio * np.abs(V_1) * np.abs(V_2)


    baselines_vec = met_wav*np.sqrt((x - xpos) ** 2 + (y - ypos) ** 2)
    cos_term = np.cos(2 * np.pi / met_wav * np.dot(baselines_vec, sep_rad))
    result = (1 + flux_ratio ** 2 + 2 * flux_ratio * cos_term) / (1 + flux_ratio) ** 2
    return result

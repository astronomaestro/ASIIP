import astropy
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, GeocentricTrueEcliptic

def radial_profile(data, center=None):
    if center == None:
        center = (data.shape[0] / 2, data.shape[1] / 2)

    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def proj_baseline(lat, dec, hour):

    ar = np.array
    Bp1 = ar([ar([-np.sin(lat) * np.sin(hour), np.cos(hour), np.cos(lat) * np.sin(hour)]),

              ar([np.sin(lat) * np.cos(hour) * np.sin(dec) + np.cos(lat) * np.cos(dec), np.sin(hour) * np.sin(dec),
                  -np.cos(lat) * np.cos(hour) * np.sin(dec) + np.sin(lat) * np.cos(dec)]),

              ar([-np.sin(lat) * np.cos(hour) * np.cos(dec) + np.cos(lat) * np.sin(dec),
                  -np.sin(hour) * np.cos(dec),
                  np.cos(lat) * np.cos(hour) * np.cos(dec) + np.sin(lat) * np.sin(dec)])])

    return Bp1


def uv_tracks(lat, dec, hours, Bn, Be, Bu):

    baselines = np.transpose(np.array([Bn, Be, Bu]))
    track = np.array([np.dot(proj_baseline(lat,dec,hour), baselines) for hour in hours])

    ref_track = np.array([np.dot(proj_baseline(lat, dec, hour), -baselines) for hour in hours])
    return track, ref_track

def array_baselines(tel_locs):
    n = np.alen(tel_locs)
    N = n*(n-1)/2
    baselines = []

    for i in range(n):
        for j in range(1,n-i):
            baselines.append(tel_locs[i] - tel_locs[i+j])

    return baselines


def airy_disk2D(shape, xpos, ypos, r):
    from astropy.modeling.functional_models import AiryDisk2D
    from astropy.modeling import models, fitting

    y, x = np.mgrid[:shape[0], :shape[1]]
    # Fit the data using astropy.modeling
    p_init = AiryDisk2D(x_0=xpos, y_0=ypos, radius=r)
    fit_p = fitting.LevMarLSQFitter()
    return p_init(x,y), p_init


def gaussian_disk2D():
    from astropy.modeling import models
    import warnings
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.modeling import models, fitting

    np.random.seed(0)
    x = np.linspace(-5., 5., 200)
    y = 3 * np.exp(-0.5 * (x - 1.3) ** 2 / 0.8 ** 2)
    y += np.random.normal(0., 0.2, x.shape)

    # Generate fake data
    np.random.seed(0)
    y, x = np.mgrid[:128, :128]
    z = 3 * np.exp(-.0001 * (y-50)**2 * (x - 50) ** 2 / 0.8 ** 2)
    z += np.random.normal(0., 0.1, z.shape) * 3

    # Fit the data using astropy.modeling
    p_init = models.Gaussian2D(amplitude=1, x_mean=1, y_mean=1)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)

    # Plot the data with the best-fit model
    plt.figure(figsize=(8, 2.5))
    plt.subplot(1, 3, 1)
    plt.imshow(z, origin='lower', interpolation='nearest')
    plt.title("Data")
    plt.subplot(1, 3, 2)
    plt.imshow(p(x, y), origin='lower', interpolation='nearest')
    plt.title("Model")
    plt.subplot(1, 3, 3)
    plt.imshow(z - p(x, y), origin='lower', interpolation='nearest')
    plt.title("Residual")


def track_coverage(tel_tracks, airy_func):
    x_0 = airy_func.x_0.value
    y_0 = airy_func.y_0.value
    ranges = []
    r_0 = airy_func.radius.value
    for i, track in enumerate(tel_tracks):
        utrack = track[0][:, 0] + x_0
        vtrack = track[0][:, 1] + y_0
        # airy_amp = airy_func(utrack, vtrack)
        airy_radius = np.sqrt((utrack - x_0) ** 2 + (vtrack - y_0) ** 2)
        ranges.append([np.min(airy_radius), np.max(airy_radius)])

    merged_ranges = interval_merger(ranges)
    filt_ranges = [ran for ran in merged_ranges if ran[0] < r_0]

    if len(filt_ranges) == 0:
        return 0
    if filt_ranges[-1][-1] > r_0:
        r0_cor = filt_ranges[-1][-1] - r_0
    else:
        r0_cor = 0
    r0_coverage = np.ptp(filt_ranges) - r0_cor
    percent_cov = 1 - r0_coverage/r_0
    return percent_cov

def interval_merger(intervals):
    sint = sorted(intervals, key=lambda i: i[0])
    out = [sint[0]]
    for current in sorted(intervals, key=lambda i: i[0]):
        previous = out[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            out.append(current)
    return out







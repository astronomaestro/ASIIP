import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, GeocentricTrueEcliptic, Angle
import astroquery
ar = np.array
import json
from astropy.io import fits, ascii
import os
from astroquery.vizier import Vizier
from IntensityInterferometry import IItools
Vizier.ROW_LIMIT = -1
from astropy.coordinates import get_sun
from scipy.stats import chisquare
from astropy.table import Table
from astropy.table import Column as col

class IItelescope():
    def __init__(self, telLat, telLon, telElv, time, steps, sig1=.11, m1=1.7, t1=800, xlen=500, ylen=500,
                 mag_range=(-3, 3), dec_range=(-20, 90), ra_range=(0,0)):

        self.Bews = []
        self.Bnss = []
        self.Buds = []

        self.xlen = xlen
        self.ylen = ylen
        self.mag_range = mag_range
        self.dec_range = dec_range

        self.telLat = telLat * u.deg
        self.telLon = telLon * u.deg
        self.telElv = telElv * u.m
        self.tel_loc = EarthLocation(lat=telLat * u.deg, lon=telLon * u.deg, height=telElv * u.m)

        self.err_sig = sig1
        self.err_mag = m1
        self.err_t1 = t1

        self.star_degs = None
        self.time_info = None

        self.time_info = Time(time, location=self.tel_loc)
        self.delta_time = np.linspace(-12, 12, steps) * u.hour

        self.telFrame = AltAz(obstime=self.time_info + self.delta_time, location=self.tel_loc)

        #get indicies for when sky is dark
        from astropy.coordinates import get_sun
        self.sunaltazs = get_sun(self.delta_time+self.time_info).transform_to(self.telFrame)
        dark_times = np.where((self.sunaltazs.alt < -15*u.deg))
        self.dark_times = self.telFrame.obstime.sidereal_time('apparent')[dark_times]

        #the hour_correction is to shift the sky to include stars that have just barely risen
        hour_correction = 4 * u.hourangle
        #calculate the possible ra range of the telescope for the given night
        if ra_range:
            self.ra_range = ra_range
        else:
            self.ra_range = ((self.dark_times[0] - hour_correction).to('deg').value, (self.dark_times[-1] + hour_correction).to('deg').value)
        if self.ra_range[0] > self.ra_range[1]:
            self.ra_range = self.ra_range[::-1]

        self.catalogs = []
        self.cat_names = []

        self.star_dict = {}


    def modify_obs_times(self, start,end, int_time):
        wanted_times = (np.arange(start.to('rad').value, end.to('rad').value, Angle(int_time).to('rad').value)*u.rad).to('hourangle')
        self.star_dict[star_id]["IntTimes"] = wanted_times


    def add_baseline(self, Bew, Bns, Bud):
        self.Bews.append(Bew)
        self.Bnss.append(Bns)
        self.Buds.append(Bud)

    def star_track(self, ra=None, dec=None, sunangle=-15, veritas_ang=20, obs_start=None, obs_end=None, Itime = 1800*u.s):

        ra_dec = str(ra) + str(dec)
        starToTrack = SkyCoord(ra=ra, dec=dec)


        starLoc = starToTrack.transform_to(self.telFrame)


        sky_ind = np.where((self.sunaltazs.alt < sunangle*u.deg) & (starLoc.alt > veritas_ang*u.deg))[0]
        observable_times = self.delta_time[sky_ind]
        if np.alen(observable_times)==0:
            if ra_dec not in self.star_dict:self.star_dict[ra_dec] = {}
            self.star_dict[ra_dec]["ObsTimes"] = np.nan
            return 0
        mintime = np.min(observable_times)
        maxtime = np.max(observable_times)
        if not obs_start: obs_start = mintime
        if not obs_end: obs_end = maxtime

        if ra_dec not in self.star_dict:
            self.star_dict[ra_dec] = {}
            self.star_dict[ra_dec]["RA"] = ra
            self.star_dict[ra_dec]["DEC"] = dec
            self.star_dict[ra_dec]["ObsTimes"] = observable_times
            self.star_dict[ra_dec]["SideTimes"] = self.telFrame.obstime.sidereal_time('apparent')[sky_ind] - starToTrack.ra
            self.observable_times = observable_times


        time_overlap = IItools.getIntersection([obs_start.to('h').value, obs_end.to('h').value],
                                               [mintime.to('h').value,maxtime.to('h').value])
        if time_overlap:
            self.star_dict[ra_dec]["IntTimes"] = np.arange(time_overlap[0], time_overlap[1], Itime.to('h').value)*u.hour
            self.star_dict[ra_dec]["IntDelt"] = Itime.to('h')
        else:
            self.star_dict[ra_dec]["IntTimes"] = None
            self.star_dict[ra_dec]["IntDelt"] = None




    def make_gaia_query(self, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100)):
        columns = ['N', 'RAJ2000','DEJ2000','Gmag','BPmag','RPmag','Teff','Rad','Lum','Plx']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        Vizier.query_constraints_async()
        result = v.query_constraints(catalog="I/345/gaia2",
                                     Gmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000="<%s || >%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        asdf=123

        good_vals = np.where(~np.isnan(result[0]['Rad']))

        self.gaia = result[0][good_vals]
        self.catalogs.append(self.gaia)
        self.cat_names.append("GAIA")


    def make_cadars_query(self, from_database=True, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100), load_vizier=True):
        columns = ['N', 'Type','Id1', 'Method', 'Lambda', 'UD', 'e_UD', 'LD', 'e_LD', 'RAJ2000', 'DEJ2000', 'Vmag', 'Kmag']
        v = Vizier()
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        result = v.query_constraints(catalog="II/224",
                                     Vmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000="<%s || >%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))


        good_val = np.where(~np.isnan(result[0]['Diam']))
        self.cadars = result[0][good_val]
        self.catalogs.append(self.cadars)
        self.cat_names.append("CEDARS")


    def make_charm2_query(self, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100)):
        columns = ['N', 'Type','Id1', 'Method', 'Lambda', 'UD', 'e_UD', 'LD', 'e_LD', 'RAJ2000', 'DEJ2000', 'Vmag', 'Kmag', 'Bmag']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        local_dat = [d for d in os.listdir() if '.dat' in d]

        result = v.query_constraints(catalog="J/A+A/431/773",
                                     Bmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000="<%s || >%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        good_val = np.where(~np.isnan(result[0]['UD']))
        self.charm2 = result[0][good_val]
        self.catalogs.append(self.charm2)
        self.cat_names.append("CHARM2")

    def make_jmmc_query(self, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100)):
        columns = ['RAJ2000','DEJ2000','2MASS','Tessmag','Teff','R*','M*','logg','Dis','Gmag','Vmag','Bmag']
        v = Vizier()
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        local_dat = [d for d in os.listdir() if '.dat' in d]

        result = v.query_constraints(catalog="II/346/jsdc_v2",
                                     Bmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000="<%s || >%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        good_val = np.where(~np.isnan(result[0]['Dis']))
        self.jmmc = result[0][good_val]
        self.catalogs.append(self.jmmc)
        self.cat_names.append("JMMC")



    def bright_star_cat(self, ra_range=(30, 100), dec_range=(30, 100)):
        from astroquery.vizier import Vizier
        columns = ['Name','RAJ2000','DEJ2000','Vmag','B-V','U-B', "SpType", "RotVel", "Multiple"]
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1
        result = v.query_constraints(catalog="V/50",
                                     RAJ2000="<%s || >%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        bs_cat = result[0]
        good_ind = np.where((bs_cat["RAJ2000"] != "") | (bs_cat["DEJ2000"] != ""))
        # RAJ2000 = Angle(bs_cat["RAJ2000"][good_ind], u.hourangle)
        # DEJ2000 = Angle(bs_cat["DEJ2000"][good_ind], u.deg)
        # viewable_stars = np.where((RAJ2000 > ra_range[0] * u.hourangle) & (RAJ2000 < ra_range[1] * u.hourangle) &
        #                           (DEJ2000 > dec_range[0] * u.deg) & (DEJ2000 < dec_range[1] * u.deg))
        self.BS_stars = bs_cat[good_ind]
        #(bs_cat[good_ind]["B-V"] + bs_cat[good_ind]["Vmag"])[np.where((bs_cat[good_ind]["B-V"] + bs_cat[good_ind]["Vmag"])<3)]
        # self.catalogs.append(self.BS_stars)
        # self.cat_names.append("BS")
        #
        # adf=12312

    def make_tess_query(self, mag_range=(1, 6), ra_range=(0, 360), dec_range=(-90, 90)):
        print("Retrieving Catalogue")

        columns = ['RAJ2000','DEJ2000','TIC','2MASS','Tessmag','Teff','R*','M*','logg','Dist','Gmag','Vmag']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1

        result = v.query_constraints(catalog="J/AJ/156/102",
                                     Vmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000="<%s || >%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        good_val = np.where(~np.isnan(result[0]['R_']) & ~np.isnan(result[0]['Dist']))

        self.tess = result[0][good_val]
        self.catalogs.append(self.tess)
        self.cat_names.append("TESS")

    def make_simbad_query(self, mag_range=(1, 6), ra_range=(0, 360), dec_range=(-90, 90)):
        from astroquery.simbad import Simbad
        Simbad.add_votable_fields('flux(B)', 'flux(G)')
        Simbad.query_criteria(catalog="J/AJ/156/102",
                                     Tessmag='Bmag >%s & Bmag <%s & '
                                             'RA >%s & RA <%s & '
                                             'DEC >%s & DEC <%s' %
                                             (mag_range[0], mag_range[1],ra_range[0], ra_range[1],dec_range[0], dec_range[1]))

    def simbad_matcher(self, ras, decs):
        from astroquery.simbad import Simbad
        Simbad.add_votable_fields('flux(B)', 'flux(G)', 'flux(V)', 'sptype', 'rot', "v*", "velocity", "distance",
                                  "diameter",
                                  "morphtype")
        sim_coords = SkyCoord(ras, decs, unit=(u.hourangle, u.deg))
        sim = Simbad.query_region(sim_coords)
        good_b = np.where(~np.isnan(sim["FLUX_B"]))
        simqcoord = SkyCoord(sim["RA"][good_b], sim["DEC"][good_b], unit=(u.hourangle, u.deg))
        simm, simd, sim3d = sim_coords.match_to_catalog_sky(simqcoord)
        return sim[good_b][simm], simd

    def download_vizier_cat(self, cat, name):
        from astroquery.vizier import Vizier
        Vizier.ROW_LIMIT = -1
        catalog = Vizier.find_catalogs(cat)
        cata = Vizier.get_catalogs(catalog.keys())
        ascii.write(cata, "%s.dat"%(name))

    # def simbad_matcher(self,ras,decs):
    #     from astroquery.simbad import Simbad
    #     coords = SkyCoord(ras,decs,unit=(u.deg, u.deg))
    #     Simbad.get_field_description()

    def ra_dec_diam_getter(self, tel, star):
        if tel.upper() == "CEDARS":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            ang_diam = star["Diam"].to('arcsec')
            mag_name = "Vmag"
            mag = star[mag_name]
        elif tel.upper() == "JMMC":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            ang_diam = star["UDDB"].to('arcsec')
            mag_name = "Bmag"
            mag = star[mag_name]
        elif tel.upper() == "CHARM2":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            ang_diam = star["UD"].to('arcsec')
            mag_name = "Bmag"
            mag = star[mag_name]
        elif tel.upper() == "TESS":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            dist= star['Dist']
            diam= (star['R_']*2)*u.solRad
            ang_diam = (((diam.to('m')/dist.to('m')))*u.rad).to('arcsec')
            mag_name = "Vmag"
            mag = star[mag_name]
        elif tel.upper() == "GAIA":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            dist = 1/(star['Plx']/1000)*u.parsec
            ang_diam = (((2*star["Rad"].to('m')/dist.to('m')))*u.rad).to('arcsec')
            mag_name = "Gmag"
            mag = star[mag_name]

        return ra,dec,ang_diam,mag,mag_name

    def track_error(self,sig1,m1,m2,t1,t2):
        return sig1*(2.512)**(m2-m1) * (t1/t2)**.5

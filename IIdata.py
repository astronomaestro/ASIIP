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
Vizier.ROW_LIMIT = -1
from astropy.coordinates import get_sun

class IItelescope():
    def __init__(self, telLat, telLon, telElv, time, steps):

        self.Bews = []
        self.Bnss = []
        self.Buds = []


        self.telLat = telLat * u.deg
        self.telLon = telLon * u.deg
        self.telElv = telElv * u.m
        self.tel_loc = EarthLocation(lat=telLat * u.deg, lon=telLon * u.deg, height=telElv * u.m)



        self.observable_times = None
        self.sidereal_times = None
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
        self.catalogs = []
        self.cat_names = []

    def add_baseline(self, Bew, Bns, Bud):
        self.Bews.append(Bew)
        self.Bnss.append(Bns)
        self.Buds.append(Bud)

    def star_track(self, ra=None, dec=None, star_name=None, sunangle=-15, veritas_ang=20):
        if star_name:
            starToTrack = SkyCoord.from_name(star_name)
            self.ra = starToTrack.ra
            self.dec = starToTrack.dec
        else:
            self.ra = ra
            self.dec = dec
            starToTrack = SkyCoord(ra=ra, dec=dec)


        starLoc = starToTrack.transform_to(self.telFrame)


        sky_ind = np.where((self.sunaltazs.alt < -15*u.deg) & (starLoc.alt > 20*u.deg))[0]
        observable_times = self.delta_time[sky_ind]


        self.observable_times = observable_times
        self.sidereal_times = self.telFrame.obstime.sidereal_time('apparent')[sky_ind] - starToTrack.ra
        self.star_degs = starLoc.alt.to("deg")[sky_ind]



    def make_gaia_query(self, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100)):
        columns = ['N', 'RAJ2000','DEJ2000','Gmag','BPmag','RPmag','Teff','Rad','Lum','Plx']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        result = v.query_constraints(catalog="I/345/gaia2",
                                     Gmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
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
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
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
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
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
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))
        good_val = np.where(~np.isnan(result[0]['Dis']))
        self.jmmc = result[0][good_val]
        self.catalogs.append(self.jmmc)
        self.cat_names.append("JMMC")



    # def bright_star_cat(self, ra_range=(30, 100), dec_range=(30, 100)):
    #     from astroquery.vizier import Vizier
    #     Vizier.ROW_LIMIT = -1
    #     bs_cat = Vizier.get_catalogs("V/50")[0]
    #     RAJ2000 = Angle(bs_cat["RAJ2000"], u.hourangle)
    #     DEJ2000 = Angle(bs_cat["DEJ2000"], u.deg)
    #     viewable_stars = np.where((RAJ2000 > ra_range[0] * u.hourangle) & (RAJ2000 < ra_range[1] * u.hourangle) &
    #                               (DEJ2000 > dec_range[0] * u.deg) & (DEJ2000 < dec_range[1] * u.deg))
    #     self.BS_stars = bs_cat[viewable_stars]
    #
    #     adf=12312

    def make_tess_query(self, mag_range=(1, 6), ra_range=(0, 360), dec_range=(-90, 90)):
        print("Retrieving Catalogue")

        columns = ['RAJ2000','DEJ2000','TIC','2MASS','Tessmag','Teff','R*','M*','logg','Dist','Gmag','Vmag']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1

        result = v.query_constraints(catalog="J/AJ/156/102",
                                     Tessmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))
        good_val = np.where(~np.isnan(result[0]['R_']))

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

    def download_vizier_cat(self, cat, name):
        from astroquery.vizier import Vizier
        Vizier.ROW_LIMIT = -1
        catalog = Vizier.find_catalogs(cat)
        cata = Vizier.get_catalogs(catalog.keys())
        ascii.write(cata, "%s.dat"%(name))

    def simbad_matcher(self,ras,decs):
        from astroquery.simbad import Simbad
        coords = SkyCoord(ras,decs,unit=(u.deg, u.deg))
        Simbad.get_field_description()

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
            mag_name = "Tessmag"
            mag = star[mag_name]
        elif tel.upper() == "GAIA":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            dist = 1/(star['Plx']/1000)*u.parsec
            ang_diam = (((2*star["Rad"].to('m')/dist.to('m')))*u.rad).to('arcsec')
            mag_name = "Gmag"
            mag = star[mag_name]

        return ra,dec,ang_diam,mag,mag_name

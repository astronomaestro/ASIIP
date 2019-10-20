import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle
from astropy.io import ascii
import os
from astroquery.vizier import Vizier
from II import IItools
from astropy.coordinates import get_sun
Vizier.ROW_LIMIT = -1



class IItelescope():
    def __init__(self, telLat, telLon, telElv, time, steps, sig1=.11, m1=1.7, t1=800, xlen=500, ylen=500,
                 mag_range=(-3, 3), dec_range=(-20, 90), ra_range=(0,0), max_sun_alt = -15, timestep=.5):

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

        self.time_info = None

        self.time_info = Time(time, location=self.tel_loc)
        self.delta_time = np.linspace(-12, 12, steps) * u.hour

        self.time_delt = self.delta_time[1]-self.delta_time[0]
        self.telFrame = AltAz(obstime=self.time_info + self.delta_time, location=self.tel_loc)


        #get indicies for when sky is dark
        self.sunaltazs = get_sun(self.delta_time+self.time_info).transform_to(self.telFrame)
        dark_times = np.where((self.sunaltazs.alt < max_sun_alt * u.deg))
        self.dark_times = self.telFrame.obstime.sidereal_time('apparent')[dark_times]
        self.max_sun_alt = max_sun_alt


        self.int_delta_time = np.append(np.arange(-12,12,timestep),12)*u.hour
        self.intTelFrame = AltAz(obstime=self.time_info + self.int_delta_time, location=self.tel_loc)
        self.intsunaltazs = get_sun(self.int_delta_time+self.time_info).transform_to(self.intTelFrame)

        int_dark_times = np.where((self.intsunaltazs.alt < max_sun_alt * u.deg))
        self.int_dark_times = self.intTelFrame.obstime.sidereal_time('apparent')[int_dark_times]


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


    def add_baseline(self, Bew, Bns, Bud):
        """
        This adds calculated baseline to an array for easy access for future calculations
        :param Bew: The East West part of the baseline
        :param Bns: The North South part of the baseline
        :param Bud: The Elevation baseline
        :return:
        """
        self.Bews.append(Bew)
        self.Bnss.append(Bns)
        self.Buds.append(Bud)

    def star_track(self, ra=None, dec=None, sunangle=-15, alt_cut=20, obs_start=None, obs_end=None, Itime =1800 * u.s):
        """
        This uses astropy to figure out the times available to measure a star throughout the night.
        :param ra: The right ascension of the star you wish to measure in J2000
        :param dec: The declenation of the star you wish to measure in J2000
        :param sunangle: The altitude the sun must be below. The times when the sun is above this are considered invalid
        :param alt_cut: The lowest possible altitude a observatory can measure at
        :param obs_start: when the observation will be starting, in astropy units of hours. 0h is midnight, None is as early as possible
        :param obs_end: when the observation will be ending, in astropy units of hours. 0h is midnight, None is as late as possible
        :param Itime: The Integration time of your observations
        :return: Appends the stars calculated informtion to the star dictionary using it's unique RA and DEC
        """
        ra_dec = str(ra) + str(dec)
        starToTrack = SkyCoord(ra=ra, dec=dec)


        starLoc = starToTrack.transform_to(self.telFrame)
        sky_ind = np.where((self.sunaltazs.alt < sunangle*u.deg) & (starLoc.alt > alt_cut * u.deg))[0]
        observable_times = self.delta_time[sky_ind]

        if np.alen(observable_times)==0:
            if ra_dec not in self.star_dict:self.star_dict[ra_dec] = {}
            self.star_dict[ra_dec]["ObsTimes"] = np.nan
            self.star_dict[ra_dec]["totTime"] = 0*u.s

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
            self.star_dict[ra_dec]["Alt"] = starLoc.alt[sky_ind]
            self.star_dict[ra_dec]["Airmass"] = starLoc.secz[sky_ind]
            self.observable_times = observable_times
            self.star_dict[ra_dec]["totTime"] = (self.time_delt * np.alen(observable_times)).to('s')


        # time_overlap = IItools.getIntersection([obs_start.to('h').value, obs_end.to('h').value],
        #                                        [mintime.to('h').value,maxtime.to('h').value])
            int_starLoc = starToTrack.transform_to(self.intTelFrame)
            int_sky_ind = np.where((self.intsunaltazs.alt <= sunangle * u.deg) & (int_starLoc.alt >= alt_cut * u.deg))[0]
            int_observable_times = self.int_delta_time[int_sky_ind]
            time_range = np.where((self.int_delta_time[int_sky_ind] >= obs_start-Itime) & (self.int_delta_time[int_sky_ind] <= obs_end+Itime))

            if np.alen(time_range[0]) > 1:
                self.star_dict[ra_dec]["IntTimes"] = self.int_delta_time[int_sky_ind][time_range]
                self.star_dict[ra_dec]["IntDelt"] = Itime.to('h')
                self.star_dict[ra_dec]["IntSideTimes"] = self.intTelFrame.obstime.sidereal_time('apparent')[int_sky_ind][time_range] - starToTrack.ra

            else:
                self.star_dict[ra_dec]["IntTimes"] = None
                self.star_dict[ra_dec]["IntDelt"] = None




    def make_gaia_query(self, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100)):
        '''
        Query gaia's data release 2 (DR2) using VizieR
        :param mag_range: the magnitude range you wish to constrain the query by
        :param ra_range: the RA range you wish to constrain the query by
        :param dec_range: The DEC range you wish to constrain the query by
        :return: appends results to the catalogs results array
        '''
        columns = ['N', 'RAJ2000','DEJ2000','Gmag','BPmag','RPmag','Teff','Rad','Lum','Plx']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        Vizier.query_constraints_async()
        result = v.query_constraints(catalog="I/345/gaia2",
                                     BPmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        asdf=123
        if result:
            good_vals = np.where(~np.isnan(result[0]['Rad']))
            self.gaia = result[0][good_vals]
            self.catalogs.append(self.gaia)
            self.cat_names.append("GAIA")


    def make_cadars_query(self, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100)):
        """
        Query the CADARS catalog using VizieR
        :param mag_range: the magnitude range you wish to constrain the query by
        :param ra_range: the RA range you wish to constrain the query by
        :param dec_range: The DEC range you wish to constrain the query by
        :return: appends results to the catalogs results array
        """
        columns = ['N', 'Type','Id1', 'Method', 'Lambda', 'UD', 'e_UD', 'LD', 'e_LD', 'RAJ2000', 'DEJ2000', 'Vmag', 'Kmag']
        v = Vizier()
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        result = v.query_constraints(catalog="II/224",
                                     Vmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        if result:
            good_val = np.where(~np.isnan(result[0]['Diam']))
            self.cadars = result[0][good_val]
            self.catalogs.append(self.cadars)
            self.cat_names.append("CEDARS")


    def make_charm2_query(self, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100)):
        """
        Query the CHARM2 catalog using VizieR
        :param mag_range: the magnitude range you wish to constrain the query by
        :param ra_range: the RA range you wish to constrain the query by
        :param dec_range: The DEC range you wish to constrain the query by
        :return: appends results to the catalogs results array
        """
        columns = ['N', 'Type','Id1', 'Method', 'Lambda', 'UD', 'e_UD', 'LD', 'e_LD', 'RAJ2000', 'DEJ2000', 'Vmag', 'Kmag', 'Bmag']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        local_dat = [d for d in os.listdir() if '.dat' in d]

        result = v.query_constraints(catalog="J/A+A/431/773",
                                     Bmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        if result:
            good_val = np.where(~np.isnan(result[0]['UD']))
            self.charm2 = result[0][good_val]
            self.catalogs.append(self.charm2)
            self.cat_names.append("CHARM2")

    def make_jmmc_query(self, mag_range=(1, 6), ra_range=(30, 100), dec_range=(30, 100)):
        """
        Query the JMMC catalog using VizieR
        :param mag_range: the magnitude range you wish to constrain the query by
        :param ra_range: the RA range you wish to constrain the query by
        :param dec_range: The DEC range you wish to constrain the query by
        :return: appends results to the catalogs results array
        """
        columns = ['RAJ2000','DEJ2000','2MASS','Tessmag','Teff','R*','M*','logg','Dis','Gmag','Vmag','Bmag']
        v = Vizier()
        v.ROW_LIMIT = -1
        print("Retrieving Catalogue")
        local_dat = [d for d in os.listdir() if '.dat' in d]

        result = v.query_constraints(catalog="II/346/jsdc_v2",
                                     Bmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        if result:
            good_val = np.where(~np.isnan(result[0]['Dis']))
            self.jmmc = result[0][good_val]
            self.catalogs.append(self.jmmc)
            self.cat_names.append("JMMC")



    def bright_star_cat(self, ra_range=(30, 100), dec_range=(30, 100)):
        """
        Query the Bright Star catalog using VizieR
        :param ra_range: the RA range you wish to constrain the query by
        :param dec_range: The DEC range you wish to constrain the query by
        :return: appends results to the object
        """
        from astroquery.vizier import Vizier
        columns = ['Name','RAJ2000','DEJ2000','Vmag','B-V','U-B', "SpType", "RotVel", "Multiple"]
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1
        result = v.query_constraints(catalog="V/50",
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        if result:
            bs_cat = result[0]
            good_ind = np.where((bs_cat["RAJ2000"] != "") | (bs_cat["DEJ2000"] != ""))
            self.BS_stars = bs_cat[good_ind]


    def make_tess_query(self, mag_range=(1, 6), ra_range=(0, 360), dec_range=(-90, 90)):
        """
        Query the TESS input catalog using VizieR
        :param mag_range: the magnitude range you wish to constrain the query by
        :param ra_range: the RA range you wish to constrain the query by
        :param dec_range: The DEC range you wish to constrain the query by
        :return: appends results to the catalogs results array
        """
        print("Retrieving Catalogue")

        columns = ['RAJ2000','DEJ2000','TIC','2MASS','Tessmag','Teff','R*','M*','logg','Dist','Gmag','Vmag']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1

        result = v.query_constraints(catalog="J/AJ/156/102",
                                     Vmag='>%s & <%s' %(mag_range[0], mag_range[1]),
                                     RAJ2000=">%s & <%s"%(ra_range[0], ra_range[1]),
                                     DEJ2000='>%s & <%s'%(dec_range[0], dec_range[1]))

        if result:
            good_val = np.where(~np.isnan(result[0]['R_']) & ~np.isnan(result[0]['Dist']))

            self.tess = result[0][good_val]
            self.catalogs.append(self.tess)
            self.cat_names.append("TESS")


    def simbad_matcher(self, ras, decs):
        """
        Match master SII catalog results to stars within SIMBAD.
        :param ras: The list of Right Ascensions you wish to query simbad with
        :param decs: The list of Declenations you wish to query simbad with
        :return: The closest matches that SIMBAD has found for your input RA and DEC
        """
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

    def cephied_finder(self, ras, decs, radius = 2*u.arcsec):
        """
        Match master SII catalog results to stars in GCVS.
        :param ras: The list of Right Ascensions you wish to query simbad with
        :param decs: The list of Declenations you wish to query simbad with
        :return: The closest matches that SIMBAD has found for your input RA and DEC
        """
        print("Retrieving Catalogue")

        columns = ['RAJ2000','DEJ2000','GCVS','VarType','magMax',"Period",'SpType','VarName']
        v = Vizier(columns=columns)
        v.ROW_LIMIT = -1
        gcvs_coords = SkyCoord(ras, decs, unit=(u.hourangle, u.deg))

        result = v.query_region(catalog="B/gcvs/gcvs_cat",
                                     coordinates=gcvs_coords,
                                     radius=radius)

        gcvscoord = SkyCoord(result[0]["RAJ2000"], result[0]["DEJ2000"], unit=(u.hourangle, u.deg))
        gcvsm, gcvsd, gcvs3d = gcvscoord.match_to_catalog_sky(gcvs_coords)

        return result[0], gcvsm, gcvsd




    def ra_dec_diam_getter(self, cat_name, star):
        """
        This is used to retrieve the right ascension, declenation, angular diameter, magnitude, and magnitude type.
        A function like this was used to improve readability of the code as each catalog contains keys which are unique
        to the catalog itself, the units were often not the same for differing catalogs, with some catalogs including
        angular diameter, while some only included information you could use to calculate an angular diameter.
        :param cat_name: The name of the catalog you wish to retrieve the information from
        :param star: The catalog entry containing all of the information from the query
        :return: The RA in J2000 hourangle, The DEC in J2000 degrees, the angular diameter in arcseconds, the magnitude,
        the name of the filter used to retrive the magnitude
        """
        if cat_name.upper() == "CEDARS":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            ang_diam = star["Diam"].to('arcsec')
            mag_name = "Vmag"
            mag = star[mag_name]
        elif cat_name.upper() == "JMMC":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            ang_diam = star["UDDB"].to('arcsec')
            mag_name = "Bmag"
            mag = star[mag_name]
        elif cat_name.upper() == "CHARM2":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            ang_diam = star["UD"].to('arcsec')
            mag_name = "Bmag"
            mag = star[mag_name]
        elif cat_name.upper() == "TESS":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            dist= star['Dist']
            diam= (star['R_']*2)*u.solRad
            ang_diam = (((diam.to('m')/dist.to('m')))*u.rad).to('arcsec')
            mag_name = "Vmag"
            mag = star[mag_name]
        elif cat_name.upper() == "GAIA":
            ra = Angle(star["RAJ2000"], 'hourangle')
            dec = Angle(star["DEJ2000"], 'deg')
            dist = 1/(star['Plx']/1000)*u.parsec
            ang_diam = (((2*star["Rad"].to('m')/dist.to('m')))*u.rad).to('arcsec')
            mag_name = "BPmag"
            mag = star[mag_name]

        return ra,dec,ang_diam,mag,mag_name

